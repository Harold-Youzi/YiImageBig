// ============================================================================
//  ort_engine.cpp - ONNX Runtime DirectML 封装实现
// ============================================================================
#include "ort_engine.h"
#include "common.h"

#include <windows.h>
#include <cstring>

namespace yb {

std::string ORTEngine::last_err(OrtStatus* st) {
    std::string msg = st && api_ ? api_->GetErrorMessage(st) : "unknown";
    if (st && api_) api_->ReleaseStatus(st);
    return msg;
}

bool ORTEngine::init(const std::wstring& runtime_dir, std::string& err) {
    if (api_) return true;

    SetDllDirectoryW(runtime_dir.c_str());
    // DirectML.dll 与 onnxruntime.dll 同目录
    mod_ = (void*)::LoadLibraryW((runtime_dir + L"\\onnxruntime.dll").c_str());
    if (!mod_) {
        err = "无法加载 onnxruntime.dll";
        return false;
    }
    auto get_base = (const OrtApiBase* (*)())::GetProcAddress((HMODULE)mod_, "OrtGetApiBase");
    if (!get_base) { err = "onnxruntime.dll 缺少 OrtGetApiBase"; return false; }

    const OrtApiBase* base = get_base();
    api_ = base->GetApi(ORT_API_VERSION);
    if (!api_) {
        err = str_format("ORT GetApi(%d) 失败 (运行库版本不匹配)", ORT_API_VERSION);
        return false;
    }

    // DML EP
    const void* out = nullptr;
    if (api_->GetExecutionProviderApi("DML", ORT_API_VERSION, &out) == nullptr && out) {
        dml_api_ = (const OrtDmlApi*)out;
    }

    if (api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "YiImageBig", &env_) != nullptr) {
        err = "ORT CreateEnv 失败";
        return false;
    }
    if (api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_cpu_) != nullptr) {
        err = "ORT CreateCpuMemoryInfo 失败";
        return false;
    }
    return true;
}

bool ORTEngine::create_session(const std::wstring& onnx_path, bool use_dml, int device_id,
                               int requested_tile, Session& out, std::string& err) {
    if (!api_) { err = "ORT 引擎未初始化"; return false; }
    if (use_dml && !dml_api_) { err = "DirectML EP 不可用"; return false; }

    OrtSessionOptions* so = nullptr;
    if (api_->CreateSessionOptions(&so) != nullptr) { err = "CreateSessionOptions 失败"; return false; }

    // 与 Python 版一致: 2 个 intra-op 线程 + 全量图优化
    api_->SetIntraOpNumThreads(so, 2);
    api_->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);

    if (use_dml) {
        OrtStatus* st = dml_api_->SessionOptionsAppendExecutionProvider_DML(so, device_id);
        if (st) { err = "追加 DirectML EP 失败: " + last_err(st); api_->ReleaseSessionOptions(so); return false; }
    }

    OrtStatus* st = api_->CreateSession(env_, onnx_path.c_str(), so, &out.s);
    api_->ReleaseSessionOptions(so);
    if (st) { err = "创建会话失败: " + last_err(st); return false; }

    // 输入/输出名
    OrtAllocator* alloc = nullptr;
    if (api_->GetAllocatorWithDefaultOptions(&alloc) != nullptr) {
        err = "GetAllocator 失败"; free_session(out); return false;
    }
    char* n1 = nullptr, *n2 = nullptr;
    if (api_->SessionGetInputName(out.s, 0, alloc, &n1) != nullptr ||
        api_->SessionGetOutputName(out.s, 0, alloc, &n2) != nullptr) {
        err = "获取输入/输出名失败"; free_session(out); return false;
    }
    out.in_name = n1 ? n1 : "";
    out.out_name = n2 ? n2 : "";
    if (n1) api_->AllocatorFree(alloc, n1);
    if (n2) api_->AllocatorFree(alloc, n2);

    // 读取输入静态尺寸 (动态则为 0)
    out.dml = use_dml;
    out.device_id = device_id;
    out.tile_in = 0;
    OrtTypeInfo* tinfo = nullptr;
    if (api_->SessionGetInputTypeInfo(out.s, 0, &tinfo) == nullptr && tinfo) {
        const OrtTensorTypeAndShapeInfo* info = nullptr;
        if (api_->CastTypeInfoToTensorInfo(tinfo, &info) == nullptr && info) {
            size_t rank = 0;
            if (api_->GetDimensionsCount(info, &rank) == nullptr && rank >= 4) {
                int64_t dims[8] = {};
                api_->GetDimensions(info, dims, rank);
                if (dims[2] > 0 && dims[2] == dims[3]) out.tile_in = (int)dims[2];
            }
        }
        api_->ReleaseTypeInfo(tinfo);
    }
    out.tile_in  = out.tile_in  > 0 ? out.tile_in  : (requested_tile > 0 ? requested_tile : 256);
    out.tile_out = out.tile_in * 4;
    return true;
}

bool ORTEngine::run(Session& s, const float* in, float* out, std::string& err) {
    if (!s.s || !api_) { err = "ORT 会话为空"; return false; }

    const int ti = s.tile_in, to = s.tile_out;
    int64_t ishape[4] = {1, 3, ti, ti};

    OrtValue* in_v = nullptr;
    OrtValue* out_v = nullptr;
    const char* in_names[1]  = {s.in_name.c_str()};
    const char* out_names[1] = {s.out_name.c_str()};

    if (api_->CreateTensorWithDataAsOrtValue(
            mem_cpu_, (void*)in, sizeof(float) * 3 * ti * ti,
            ishape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_v) != nullptr) {
        err = "创建输入张量失败";
        return false;
    }

    OrtStatus* st = api_->Run(s.s, nullptr, in_names, (const OrtValue* const*)&in_v, 1,
                              out_names, 1, &out_v);
    bool ok = false;
    if (st == nullptr && out_v) {
        float* data = nullptr;
        if (api_->GetTensorMutableData(out_v, (void**)&data) == nullptr && data) {
            memcpy(out, data, sizeof(float) * 3 * to * to);
            ok = true;
        } else {
            err = "读取输出数据失败";
        }
    } else if (st) {
        err = "ORT Run 失败: " + last_err(st);
    }
    if (in_v)  api_->ReleaseValue(in_v);
    if (out_v) api_->ReleaseValue(out_v);
    return ok;
}

void ORTEngine::free_session(Session& s) {
    if (s.s && api_) api_->ReleaseSession(s.s);
    s.s = nullptr;
}

void ORTEngine::shutdown() {
    if (mem_cpu_ && api_) api_->ReleaseMemoryInfo(mem_cpu_);
    if (env_ && api_)     api_->ReleaseEnv(env_);
    mem_cpu_ = nullptr;
    env_ = nullptr;
    api_ = nullptr;
    if (mod_) { FreeLibrary((HMODULE)mod_); mod_ = nullptr; }
}

} // namespace yb
