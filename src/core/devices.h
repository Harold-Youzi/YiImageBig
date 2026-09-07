// ============================================================================
//  devices.h - 硬件检测与智能选择 (移植自 Python 版 hardware_detector.py)
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace yb {

class OVEngine;
class ORTEngine;

enum class DevKind {
    Cuda,          // NVIDIA GPU (CUDA EP, 预留)
    DirectML,      // NVIDIA GPU (DirectML)
    OpenVinoGpu,   // Intel GPU / Arc
    OpenVinoNpu,   // Intel NPU
    OpenVinoCpu,   // OpenVINO CPU
    Cpu,           // 纯 CPU (兜底, 未用 OpenVINO 时)
};

struct DeviceInfo {
    DevKind    kind;
    int        score = 0;        // 自动选择评分
    std::string device_id;       // OpenVINO 设备名 ("GPU"/"NPU"/"CPU"), DML 为适配器序号
    int        dml_adapter = 0;  // DXGI 适配器序号
    std::string name;            // 简称: "Intel Arc GPU"
    std::string full_name;       // 全名
    std::string backend;         // "OpenVINO 2025.4" / "DirectML"
};

struct SystemSpecs {
    uint64_t total_ram = 0;
    uint64_t avail_ram = 0;
    int      cpu_cores = 0;
    int      cpu_threads = 0;
    std::string cpu_name;
    double   gpu_vram_gb = 0.0;   // Intel 集显按内存一半估算
    int      max_tile_size = 256;
};

class DeviceDetector {
public:
    // ov/ort 引擎需已 init (DLL 已加载)
    void detect(OVEngine& ov, ORTEngine& ort);

    const std::vector<DeviceInfo>& devices() const { return devices_; }
    bool empty() const { return devices_.empty(); }

    // 智能选择 (评分最高)
    const DeviceInfo* optimal() const;

    // 按序号取设备
    const DeviceInfo* at(size_t i) const;

    const SystemSpecs& specs() const { return specs_; }

private:
    void detect_nvidia(ORTEngine& ort);
    void detect_openvino(OVEngine& ov);
    static bool is_nvidia_name(const std::string& full_name);

    std::vector<DeviceInfo> devices_;
    SystemSpecs specs_;
};

} // namespace yb
