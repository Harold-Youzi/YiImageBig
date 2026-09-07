// ============================================================================
//  ort_engine.h - ONNX Runtime DirectML 封装 (运行时动态加载, NVIDIA GPU)
// ============================================================================
#pragma once

#include <string>
#include <vector>

#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>   // OrtDmlApi

namespace yb {

class ORTEngine {
public:
    ORTEngine() = default;
    ~ORTEngine() { shutdown(); }

    ORTEngine(const ORTEngine&) = delete;
    ORTEngine& operator=(const ORTEngine&) = delete;

    // 加载 <runtime_dir>/onnxruntime.dll (DirectML 构建版)
    bool init(const std::wstring& runtime_dir, std::string& err);
    bool ok() const { return api_ != nullptr; }

    // DML EP 是否可用
    bool dml_available() const { return dml_api_ != nullptr; }

    struct Session {
        OrtSession* s = nullptr;
        std::string in_name, out_name;
        int  tile_in = 0;        // 0 = 动态, 使用请求值
        int  tile_out = 0;
        bool dml = false;
        int  device_id = 0;
    };

    // 创建会话; use_dml 时走 DirectML EP (device_id 为 DXGI 适配器序号)
    bool create_session(const std::wstring& onnx_path, bool use_dml, int device_id,
                        int requested_tile, Session& out, std::string& err);

    // 同步推理: in(CHW f32, ti*ti*3) -> out(CHW f32, to*to*3)
    bool run(Session& s, const float* in, float* out, std::string& err);

    void free_session(Session& s);
    void shutdown();

private:
    void* mod_ = nullptr;                  // HMODULE
    const OrtApi* api_ = nullptr;
    const OrtDmlApi* dml_api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* mem_cpu_ = nullptr;

    std::string last_err(OrtStatus* st);
};

} // namespace yb
