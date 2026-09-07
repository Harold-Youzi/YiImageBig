// ============================================================================
//  ov_engine.h - OpenVINO C API 封装 (运行时动态加载, Intel CPU/GPU/NPU)
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <utility>

// 启用 unicode 路径变体 API (Windows)
#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT 1
#endif

#include <openvino/c/openvino.h>

// OpenVINO 头文件用 "#define BOOLEAN OV_BOOLEAN" 兼容其 C++ API,
// 本项目不用其 C++ API, 立即解除宏污染, 避免破坏后续 windows.h
#undef BOOLEAN

namespace yb {

class OVEngine {
public:
    OVEngine() = default;
    ~OVEngine() { shutdown(); }

    OVEngine(const OVEngine&) = delete;
    OVEngine& operator=(const OVEngine&) = delete;

    // 加载 <runtime_dir>/openvino_c.dll 及其依赖
    bool init(const std::wstring& runtime_dir, std::string& err);

    // 载入是否成功 (DLL 缺失时为 false)
    bool ok() const { return core_ != nullptr; }

    std::string version_str();
    std::vector<std::string> available_devices();
    std::string device_full_name(const std::string& device);

    // ------------------------------------------------------------------
    struct Request {
        ov_infer_request_t* req    = nullptr;
        ov_tensor_t*        input  = nullptr;   // 自建输入张量 [1,3,ti,ti]
        ov_tensor_t*        output = nullptr;   // 请求自带的输出张量 [1,3,to,to]
        void*  in_data  = nullptr;
        void*  out_data = nullptr;
        int    tile_in = 0, tile_out = 0;
    };

    struct Model {
        ov_compiled_model_t* cm = nullptr;
        std::string device;          // "GPU" / "NPU" / "CPU"
        int  tile_in  = 0;           // 实际输入 tile 尺寸
        int  tile_out = 0;           // 输出尺寸 (tile*4)
        std::vector<Request> reqs;   // NPU=1 个, 其他=2 个(双缓冲)
    };

    // 编译模型并创建推理请求; requested_tile 仅在模型为动态输入时使用
    bool load_model(const std::wstring& xml_path, const std::string& device,
                    const std::vector<std::pair<std::string, std::string>>& props,
                    int requested_tile, Model& out, std::string& err);

    // 同步推理: in(CHW f32) -> out(CHW f32)
    bool infer_sync(Request& r, const float* in, float* out, std::string& err);

    // 异步双缓冲: async_start 后可再 start 另一请求, 再对先启动的 wait
    bool async_start(Request& r, const float* in, std::string& err);
    bool async_wait(Request& r, std::string& err);   // 完成后用 read_output 取数
    bool read_output(Request& r, float* out, std::string& err);

    void cancel(Model& m);
    void free_model(Model& m);
    void shutdown();

private:
    bool bind_all(std::string& err);   // 在 cpp 中绑定全部函数指针

    void*   mod_  = nullptr;   // HMODULE openvino_c.dll
    void*   mod2_ = nullptr;   // HMODULE openvino.dll (保引用, 防止意外卸载)
    ov_core_t* core_ = nullptr;

    // --- 绑定的函数指针 (签名直接复用官方头文件声明, 保证类型一致) ---
    decltype(&ov_get_openvino_version)  p_version_  = nullptr;
    decltype(&ov_version_free)          p_version_free_ = nullptr;
    decltype(&ov_core_create)           p_core_create_ = nullptr;
    decltype(&ov_core_free)             p_core_free_ = nullptr;
    decltype(&ov_core_get_available_devices) p_avail_ = nullptr;
    decltype(&ov_core_get_property)     p_get_prop_ = nullptr;
    decltype(&ov_compiled_model_create_infer_request) p_create_req_ = nullptr;
    decltype(&ov_compiled_model_input_by_index)  p_cm_input_ = nullptr;
    decltype(&ov_compiled_model_output_by_index) p_cm_output_ = nullptr;
    decltype(&ov_infer_request_set_input_tensor) p_set_input_ = nullptr;
    decltype(&ov_infer_request_get_output_tensor) p_get_output_ = nullptr;
    decltype(&ov_infer_request_infer)   p_infer_ = nullptr;
    decltype(&ov_infer_request_start_async) p_start_async_ = nullptr;
    decltype(&ov_infer_request_wait)    p_wait_ = nullptr;
    decltype(&ov_infer_request_cancel)  p_cancel_ = nullptr;
    decltype(&ov_tensor_create)         p_tensor_create_ = nullptr;
    decltype(&ov_tensor_data)           p_tensor_data_ = nullptr;
    decltype(&ov_tensor_set_shape)      p_tensor_set_shape_ = nullptr;
    decltype(&ov_tensor_free)           p_tensor_free_ = nullptr;
    decltype(&ov_const_port_get_shape)  p_cport_shape_ = nullptr;
    decltype(&ov_output_const_port_free) p_port_free_ = nullptr;
    decltype(&ov_shape_free)            p_shape_free_ = nullptr;
    decltype(&ov_free)                  p_free_ = nullptr;

    // 变参编译函数 (属性对: key,value,...)
    using FnCompileFileW = ov_status_e (__cdecl*)(
        const ov_core_t*, const wchar_t*, const char*, size_t,
        ov_compiled_model_t**, ...);
    FnCompileFileW p_compile_file_w_ = nullptr;

    // 对象释放函数 (void 返回)
    void (*p_model_free_)(ov_model_t*) = nullptr;
    void (*p_cm_free_)(ov_compiled_model_t*) = nullptr;
    void (*p_req_free_)(ov_infer_request_t*) = nullptr;
    void (*p_avail_free_)(ov_available_devices_t*) = nullptr;

    // 内部: 通过输入端口判断静态 shape
    int query_static_tile(Model& m);
};

} // namespace yb
