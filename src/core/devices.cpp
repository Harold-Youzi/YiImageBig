// ============================================================================
//  devices.cpp - 硬件检测实现 (NVIDIA 走 DXGI 枚举, OpenVINO 走设备枚举)
// ============================================================================
#include "devices.h"
#include "common.h"
#include "ov_engine.h"
#include "ort_engine.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#include <cstring>
#include <algorithm>

namespace yb {

// 评分 (与 Python 版 SCORES 一致)
static constexpr int kScoreCuda = 100;
static constexpr int kScoreNpu = 90;
static constexpr int kScoreOvGpu = 80;
static constexpr int kScoreDml = 70;
static constexpr int kScoreOvCpu = 40;
static constexpr int kScoreCpu = 30;

bool DeviceDetector::is_nvidia_name(const std::string& full_name) {
    static const char* kws[] = {"nvidia", "geforce", "rtx", "gtx", "quadro",
                                "tesla", "titan", "nvda", "nv_", "cuve"};
    std::string lower = full_name;
    for (auto& c : lower) c = (char)tolower((unsigned char)c);
    for (const char* kw : kws)
        if (strstr(lower.c_str(), kw)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// NVIDIA GPU: DXGI 枚举硬件适配器 (VendorId 0x10DE)
// 调试: 设 YIIMAGEBIG_FORCE_DML=1 时, 无 NVIDIA 也会把第一个 DX12 硬件适配器
//       (如 Intel Arc) 提供给 DirectML, 用于端到端验证 ORT/DML 代码路径
// ---------------------------------------------------------------------------
void DeviceDetector::detect_nvidia(ORTEngine& ort) {
    const bool force_dml = [] {
        char buf[8] = {};
        DWORD n = GetEnvironmentVariableA("YIIMAGEBIG_FORCE_DML", buf, sizeof(buf));
        return n > 0 && n < sizeof(buf) && buf[0] != '0';
    }();

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)))
        return;

    std::string first_name;
    int first_adapter = -1;
    std::string force_name;
    int force_adapter = -1;
    int count = 0;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* ad = nullptr;
        if (factory->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc = {};
        if (SUCCEEDED(ad->GetDesc1(&desc))) {
            // 排除软件适配器
            const bool software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
            if (!software) {
                if (desc.VendorId == 0x10DE) {
                    ++count;
                    if (first_adapter < 0) {
                        first_adapter = (int)i;
                        std::wstring ws(desc.Description);
                        first_name = wide_to_utf8(ws);
                        specs_.gpu_vram_gb = (double)(desc.DedicatedVideoMemory) / (1024.0 * 1024.0 * 1024.0);
                    }
                } else if (force_dml && force_adapter < 0) {
                    force_adapter = (int)i;
                    force_name = wide_to_utf8(desc.Description);
                }
            }
        }
        ad->Release();
    }
    factory->Release();

    if (first_adapter < 0 && force_adapter >= 0) {
        first_adapter = force_adapter;
        first_name = force_name;
    }
    if (first_adapter < 0) {
        Logger::instance().info("未检测到 NVIDIA GPU");
        return;
    }
    if (!ort.dml_available()) {
        Logger::instance().warn("检测到 NVIDIA GPU 但 DirectML EP 不可用");
        return;
    }

    DeviceInfo d;
    d.kind = DevKind::DirectML;
    d.score = kScoreDml;
    d.device_id = str_format("%d", first_adapter);
    d.dml_adapter = first_adapter;
    d.name = "NVIDIA GPU (DirectML)";
    d.full_name = first_name;
    d.backend = "DirectML (ONNX Runtime)";
    devices_.push_back(d);
    Logger::instance().info(str_format("添加 NVIDIA GPU (DirectML): %s", d.full_name.c_str()));
}

// ---------------------------------------------------------------------------
// OpenVINO 设备 (Intel CPU/GPU/NPU)
// ---------------------------------------------------------------------------
void DeviceDetector::detect_openvino(OVEngine& ov) {
    if (!ov.ok()) {
        Logger::instance().warn("OpenVINO 运行库不可用");
        return;
    }
    Logger::instance().info(str_format("OpenVINO %s", ov.version_str().c_str()));

    bool intel_gpu = false, npu = false;
    for (const auto& dev : ov.available_devices()) {
        std::string full = ov.device_full_name(dev);
        if (full.empty()) full = dev;

        if (dev == "CPU") {
            DeviceInfo d;
            d.kind = DevKind::OpenVinoCpu;
            d.score = kScoreOvCpu;
            d.device_id = "CPU";
            d.name = "OpenVINO CPU";
            d.full_name = full;
            d.backend = "OpenVINO";
            specs_.cpu_name = full;
            devices_.push_back(d);
        } else if (dev.rfind("GPU", 0) == 0) {
            if (is_nvidia_name(full)) {
                Logger::instance().info("跳过 OpenVINO GPU (非 Intel): " + full);
                continue;
            }
            std::string lower = full;
            for (auto& c : lower) c = (char)tolower((unsigned char)c);
            if (lower.find("amd") != std::string::npos ||
                lower.find("radeon") != std::string::npos ||
                lower.find("rx ") != std::string::npos) {
                Logger::instance().info("跳过 OpenVINO GPU (AMD): " + full);
                continue;
            }
            bool is_arc = full.find("Arc") != std::string::npos ||
                          full.find("DG") != std::string::npos;
            DeviceInfo d;
            d.kind = DevKind::OpenVinoGpu;
            d.score = kScoreOvGpu;
            d.device_id = dev;
            d.name = is_arc ? "Intel Arc GPU" : "Intel GPU";
            d.full_name = full;
            d.backend = "OpenVINO";
            devices_.push_back(d);
            intel_gpu = true;
        } else if (dev == "NPU") {
            DeviceInfo d;
            d.kind = DevKind::OpenVinoNpu;
            d.score = kScoreNpu;
            d.device_id = "NPU";
            d.name = "Intel NPU";
            d.full_name = full;
            d.backend = "OpenVINO";
            devices_.push_back(d);
            npu = true;
        }
    }

    // Intel 集显显存估算 = 系统内存一半, 上限 8GB
    if (intel_gpu && specs_.gpu_vram_gb <= 0) {
        specs_.gpu_vram_gb = std::min((double)specs_.total_ram * 0.5 / (1024.0 * 1024.0 * 1024.0), 8.0);
    }
    Logger::instance().info(str_format("NPU 可用: %s", npu ? "是" : "否"));
}

// ---------------------------------------------------------------------------
// 系统信息 + max_tile_size 规则
// ---------------------------------------------------------------------------
void DeviceDetector::detect(OVEngine& ov, ORTEngine& ort) {
    devices_.clear();

    SYSTEM_INFO si = {};
    GetSystemInfo(&si);
    specs_.cpu_cores = si.dwNumberOfProcessors;      // 物理核近似 (含超线程时为逻辑数)
    specs_.cpu_threads = si.dwNumberOfProcessors;

    MEMORYSTATUSEX ms = { sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    specs_.total_ram = ms.ullTotalPhys;
    specs_.avail_ram = ms.ullAvailPhys;

    // CPU 名称
    {
        char buf[128] = {};
        DWORD size = sizeof(buf);
        HKEY hk;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                          "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                          0, KEY_READ, &hk) == ERROR_SUCCESS) {
            RegQueryValueExA(hk, "ProcessorNameString", nullptr, nullptr, (LPBYTE)buf, &size);
            RegCloseKey(hk);
        }
        specs_.cpu_name = buf;
    }

    detect_nvidia(ort);
    detect_openvino(ov);

    // max_tile_size 规则 (与 Python 版一致):
    //   显存/可用内存 >= 8GB -> 512; >= 4GB -> 256; 否则 128
    double vram = specs_.gpu_vram_gb > 0
                      ? specs_.gpu_vram_gb
                      : (double)specs_.avail_ram / (1024.0 * 1024.0 * 1024.0);
    if (vram >= 8)      specs_.max_tile_size = 512;
    else if (vram >= 4) specs_.max_tile_size = 256;
    else                specs_.max_tile_size = 128;

    Logger::instance().info(str_format(
        "检测到 %d 个可用设备, 推荐 tile=%d", (int)devices_.size(), specs_.max_tile_size));
    for (auto& d : devices_)
        Logger::instance().info(str_format("  [%s] %s (%s) score=%d",
                                           d.name.c_str(), d.full_name.c_str(),
                                           d.backend.c_str(), d.score));
}

const DeviceInfo* DeviceDetector::optimal() const {
    const DeviceInfo* best = nullptr;
    for (auto& d : devices_)
        if (!best || d.score > best->score) best = &d;
    return best;
}

const DeviceInfo* DeviceDetector::at(size_t i) const {
    return i < devices_.size() ? &devices_[i] : nullptr;
}

} // namespace yb
