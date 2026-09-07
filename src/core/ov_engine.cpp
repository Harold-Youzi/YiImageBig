// ============================================================================
//  ov_engine.cpp - OpenVINO C API 封装实现
// ============================================================================
#include "ov_engine.h"
#include "common.h"

#include <windows.h>
#include <cstring>

namespace yb {

// 绑定辅助: 按名字取函数指针
template <typename T>
static bool bind_fn(HMODULE m, const char* name, T& fn, std::string& err) {
    fn = reinterpret_cast<T>(::GetProcAddress(m, name));
    if (!fn) {
        err = str_format("openvino_c.dll 缺少导出函数: %s", name);
        return false;
    }
    return true;
}

bool OVEngine::bind_all(std::string& err) {
    HMODULE m = (HMODULE)mod_;
    bool good = true;
    good &= bind_fn(m, "ov_get_openvino_version",           p_version_, err);
    good &= bind_fn(m, "ov_version_free",                   p_version_free_, err);
    good &= bind_fn(m, "ov_core_create",                    p_core_create_, err);
    good &= bind_fn(m, "ov_core_free",                      p_core_free_, err);
    good &= bind_fn(m, "ov_core_get_available_devices",     p_avail_, err);
    good &= bind_fn(m, "ov_core_get_property",              p_get_prop_, err);
    good &= bind_fn(m, "ov_compiled_model_create_infer_request", p_create_req_, err);
    good &= bind_fn(m, "ov_compiled_model_input_by_index",  p_cm_input_, err);
    good &= bind_fn(m, "ov_compiled_model_output_by_index", p_cm_output_, err);
    good &= bind_fn(m, "ov_infer_request_set_input_tensor", p_set_input_, err);
    good &= bind_fn(m, "ov_infer_request_get_output_tensor", p_get_output_, err);
    good &= bind_fn(m, "ov_infer_request_infer",            p_infer_, err);
    good &= bind_fn(m, "ov_infer_request_start_async",      p_start_async_, err);
    good &= bind_fn(m, "ov_infer_request_wait",             p_wait_, err);
    good &= bind_fn(m, "ov_infer_request_cancel",           p_cancel_, err);
    good &= bind_fn(m, "ov_tensor_create",                  p_tensor_create_, err);
    good &= bind_fn(m, "ov_tensor_data",                    p_tensor_data_, err);
    good &= bind_fn(m, "ov_tensor_set_shape",               p_tensor_set_shape_, err);
    good &= bind_fn(m, "ov_tensor_free",                    p_tensor_free_, err);
    good &= bind_fn(m, "ov_const_port_get_shape",           p_cport_shape_, err);
    good &= bind_fn(m, "ov_output_const_port_free",         p_port_free_, err);
    good &= bind_fn(m, "ov_shape_free",                     p_shape_free_, err);
    good &= bind_fn(m, "ov_free",                           p_free_, err);

    // void 返回的释放函数
    p_model_free_    = (void (*)(ov_model_t*))::GetProcAddress(m, "ov_model_free");
    p_cm_free_       = (void (*)(ov_compiled_model_t*))::GetProcAddress(m, "ov_compiled_model_free");
    p_req_free_      = (void (*)(ov_infer_request_t*))::GetProcAddress(m, "ov_infer_request_free");
    p_avail_free_    = (void (*)(ov_available_devices_t*))::GetProcAddress(m, "ov_available_devices_free");

    // 变参编译函数
    p_compile_file_w_ = (FnCompileFileW)::GetProcAddress(m, "ov_core_compile_model_from_file_unicode");

    if (!p_compile_file_w_ || !p_model_free_ || !p_cm_free_ || !p_req_free_ || !p_avail_free_) {
        err = "openvino_c.dll 缺少导出函数 (compile/free)";
        good = false;
    }
    return good;
}

bool OVEngine::init(const std::wstring& runtime_dir, std::string& err) {
    if (core_) return true;

    // 将运行库目录加入 DLL 搜索路径 (openvino_c -> openvino -> tbb/plugins)
    SetDllDirectoryW(runtime_dir.c_str());

    mod2_ = (void*)::LoadLibraryW((runtime_dir + L"\\openvino.dll").c_str());
    mod_  = (void*)::LoadLibraryW((runtime_dir + L"\\openvino_c.dll").c_str());
    if (!mod_) {
        err = "无法加载 openvino_c.dll (缺少 OpenVINO 运行库)";
        return false;
    }

    if (!bind_all(err)) return false;
    if (!p_core_create_) return false;

    if (p_core_create_(&core_) != OK) {
        err = "ov_core_create 失败";
        core_ = nullptr;
        return false;
    }
    return true;
}

std::string OVEngine::version_str() {
    if (!p_version_) return "";
    ov_version_t ver = {};
    if (p_version_(&ver) != OK) return "";
    std::string s = ver.buildNumber ? ver.buildNumber : "";
    p_version_free_(&ver);
    return s;
}

std::vector<std::string> OVEngine::available_devices() {
    std::vector<std::string> out;
    if (!core_ || !p_avail_) return out;
    // 调用方分配结构体, 内部字符串数组由 API 分配, ov_available_devices_free 释放
    ov_available_devices_t devs;
    memset(&devs, 0, sizeof(devs));
    if (p_avail_(core_, &devs) != OK) return out;
    for (size_t i = 0; i < devs.size; ++i)
        if (devs.devices && devs.devices[i])
            out.push_back(devs.devices[i]);
    if (p_avail_free_) p_avail_free_(&devs);
    return out;
}

std::string OVEngine::device_full_name(const std::string& device) {
    if (!core_ || !p_get_prop_) return "";
    char* val = nullptr;
    if (p_get_prop_(core_, device.c_str(), "FULL_DEVICE_NAME", &val) != OK || !val)
        return "";
    std::string s = val;
    p_free_(val);
    return s;
}

// 尝试读取静态输入 shape 的空间尺寸; 动态模型返回 0
int OVEngine::query_static_tile(Model& m) {
    ov_output_const_port_t* port = nullptr;
    if (p_cm_input_(m.cm, (size_t)0, &port) != OK || !port) return 0;
    ov_shape_t shape = {};
    int tile = 0;
    if (p_cport_shape_(port, &shape) == OK && shape.rank >= 4) {
        int64_t a = shape.dims[2], b = shape.dims[3];
        if (a > 0 && b > 0 && a == b) tile = (int)a;
    }
    if (p_shape_free_) p_shape_free_(&shape);
    p_port_free_(port);
    return tile;
}

bool OVEngine::load_model(const std::wstring& xml_path, const std::string& device,
                          const std::vector<std::pair<std::string, std::string>>& props,
                          int requested_tile, Model& out, std::string& err) {
    if (!core_ || !p_compile_file_w_) { err = "OpenVINO 引擎未初始化"; return false; }

    // 变参: (core, path, device, prop_args_size, &cm, key1, val1, key2, val2, ...)
    size_t nargs = props.size() * 2;
    ov_status_e st;
    switch (props.size()) {
    case 0:
        st = p_compile_file_w_(core_, xml_path.c_str(), device.c_str(), nargs, &out.cm);
        break;
    case 1:
        st = p_compile_file_w_(core_, xml_path.c_str(), device.c_str(), nargs, &out.cm,
                               props[0].first.c_str(), props[0].second.c_str());
        break;
    case 2:
        st = p_compile_file_w_(core_, xml_path.c_str(), device.c_str(), nargs, &out.cm,
                               props[0].first.c_str(), props[0].second.c_str(),
                               props[1].first.c_str(), props[1].second.c_str());
        break;
    case 3:
        st = p_compile_file_w_(core_, xml_path.c_str(), device.c_str(), nargs, &out.cm,
                               props[0].first.c_str(), props[0].second.c_str(),
                               props[1].first.c_str(), props[1].second.c_str(),
                               props[2].first.c_str(), props[2].second.c_str());
        break;
    default:
        err = "属性数量过多";
        return false;
    }
    if (st != OK || !out.cm) {
        err = str_format("模型编译失败 (设备 %s, hr=%d)", device.c_str(), (int)st);
        return false;
    }

    out.device = device;

    // 输入 tile: 优先模型静态 shape, 否则用请求值
    int tile = query_static_tile(out);
    if (tile <= 0) tile = requested_tile > 0 ? requested_tile : 256;
    out.tile_in  = tile;
    out.tile_out = tile * 4;

    // 创建推理请求与张量
    int nreqs = (device == "NPU") ? 1 : 2;
    ov_shape_t ishape = {}, oshape = {};
    int64_t idims[4]  = {1, 3, out.tile_in,  out.tile_in};
    int64_t odims[4]  = {1, 3, out.tile_out, out.tile_out};
    ishape.rank = 4; ishape.dims = idims;
    oshape.rank = 4; oshape.dims = odims;

    out.reqs.resize(nreqs);
    for (int i = 0; i < nreqs; ++i) {
        Request& r = out.reqs[i];
        r.tile_in = out.tile_in; r.tile_out = out.tile_out;
        if (p_create_req_(out.cm, &r.req) != OK || !r.req) {
            err = "创建推理请求失败";
            free_model(out);
            return false;
        }
        if (p_tensor_create_(F32, ishape, &r.input) != OK || !r.input) {
            err = "创建输入张量失败";
            free_model(out);
            return false;
        }
        if (p_tensor_data_(r.input, &r.in_data) != OK || !r.in_data) {
            err = "获取输入张量指针失败";
            free_model(out);
            return false;
        }
        // 输出张量由请求持有; 动态模型需先设置形状以分配内存
        if (p_get_output_(r.req, &r.output) != OK || !r.output) {
            err = "获取输出张量失败";
            free_model(out);
            return false;
        }
        if (p_tensor_set_shape_(r.output, oshape) != OK) {
            err = "设置输出张量形状失败";
            free_model(out);
            return false;
        }
        if (p_tensor_data_(r.output, &r.out_data) != OK || !r.out_data) {
            err = "获取输出张量指针失败";
            free_model(out);
            return false;
        }
        // 绑定自建输入张量
        if (p_set_input_(r.req, r.input) != OK) {
            err = "绑定输入张量失败";
            free_model(out);
            return false;
        }
    }
    return true;
}

bool OVEngine::infer_sync(Request& r, const float* in, float* out, std::string& err) {
    if (!r.req) { err = "推理请求为空"; return false; }
    memcpy(r.in_data, in, sizeof(float) * 3 * r.tile_in * r.tile_in);
    if (p_infer_(r.req) != OK) { err = "ov_infer_request_infer 失败"; return false; }
    // 动态模型推理后输出张量可能重新分配, 每次重新获取指针
    ov_tensor_t* t = nullptr;
    if (p_get_output_(r.req, &t) != OK || !t) { err = "读取输出张量失败"; return false; }
    void* p = nullptr;
    bool ok = p_tensor_data_(t, &p) == OK && p != nullptr;
    if (ok) memcpy(out, p, sizeof(float) * 3 * r.tile_out * r.tile_out);
    else err = "读取输出数据失败";
    return ok;
}

bool OVEngine::async_start(Request& r, const float* in, std::string& err) {
    if (!r.req) { err = "推理请求为空"; return false; }
    memcpy(r.in_data, in, sizeof(float) * 3 * r.tile_in * r.tile_in);
    if (p_start_async_(r.req) != OK) { err = "start_async 失败"; return false; }
    return true;
}

bool OVEngine::async_wait(Request& r, std::string& err) {
    if (!r.req) { err = "推理请求为空"; return false; }
    if (p_wait_(r.req) != OK) { err = "异步等待失败"; return false; }
    return true;
}

// 辅助: 等待完成后读取指定请求的输出数据
bool OVEngine::read_output(Request& r, float* out, std::string& err) {
    ov_tensor_t* t = nullptr;
    if (p_get_output_(r.req, &t) != OK || !t) { err = "读取输出张量失败"; return false; }
    void* p = nullptr;
    bool ok = p_tensor_data_(t, &p) == OK && p != nullptr;
    if (ok) memcpy(out, p, sizeof(float) * 3 * r.tile_out * r.tile_out);
    else err = "读取输出数据失败";
    return ok;
}

void OVEngine::cancel(Model& m) {
    for (auto& r : m.reqs)
        if (r.req) p_cancel_(r.req);
}

void OVEngine::free_model(Model& m) {
    for (auto& r : m.reqs) {
        if (r.input && p_tensor_free_) p_tensor_free_(r.input);
        if (r.output && p_tensor_free_) p_tensor_free_(r.output);
        if (r.req && p_req_free_) p_req_free_(r.req);
        r = {};
    }
    if (m.cm && p_cm_free_) { p_cm_free_(m.cm); }
    m.cm = nullptr;
}

void OVEngine::shutdown() {
    if (core_ && p_core_free_) { p_core_free_(core_); }
    core_ = nullptr;
    if (mod_)  { FreeLibrary((HMODULE)mod_);  mod_  = nullptr; }
    if (mod2_) { FreeLibrary((HMODULE)mod2_); mod2_ = nullptr; }
}

} // namespace yb
