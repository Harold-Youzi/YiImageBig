// ============================================================================
//  upscaler.h - RealESRGAN x4 分块推理与融合 (移植自 Python 版 model_manager.py)
//
//  算法要点 (与 Python 版逐一对应):
//    SCALE = 4
//    DEVICE_CONFIGS:  ov_gpu(512,32) / ov_npu(128,64) / ov_cpu(256,32)
//                     dml(256,32) / ort_cpu(256,32)
//    权值图: 余弦窗口可分离乘积  0.5*(1-cos(2*pi*i/size))
//    tile 定位: step = tile - overlap, 尾部对齐右/下边缘
//    补边: 仅右/下方向 reflect 到 tile 整数倍
//    NPU: 同步推理 + 输出边界中值均衡 (RealESRGAN_x4plus_npu_128)
//    GPU/CPU(OpenVINO): 双请求异步流水线
//    DirectML / ORT-CPU: 同步逐块推理
// ============================================================================
#pragma once

#include <functional>
#include <string>
#include <atomic>
#include "image.h"

namespace yb {

// 分块推理引擎描述 (由 OVEngine / ORTEngine 适配后提供)
struct TileEngine {
    int  tile = 256;        // 模型输入 tile
    int  overlap = 32;      // 重叠像素
    bool npu = false;       // 是否 NPU (128 静态 + 边界均衡, 同步)
    int  slots = 1;         // 1 = 同步; 2 = 双请求异步 (OpenVINO GPU/CPU)

    // 同步推理: in[tile*tile*3] -> out[(tile*4)*(tile*4)*3]
    std::function<bool(const float* in, float* out, std::string& err)> infer;

    // 异步双缓冲 (slots == 2 时必须提供)
    std::function<bool(const float* in, int slot, std::string& err)>       astart;
    std::function<bool(float* out, int slot, std::string& err)>            await;   // wait 并拷出
    std::function<void()> cancel;

    bool valid() const { return (bool)infer || (bool)astart; }
};

// 进度回调: (已完成 tile 数, tile 总数)
using TileProgressFn = std::function<void(int done, int total)>;

class Upscaler {
public:
    static constexpr int SCALE = 4;

    // 权值图 / tile 定位 (暴露为静态方法便于单元测试)
    static std::vector<float> make_weight_map(int size);            // size*size
    static std::vector<std::pair<int, int>> tile_positions(int ph, int pw, int tile, int overlap);

    // 完整放大流程: BGR8 -> BGR8 (4x 或通过 target_scale 降采样到 2x/3x)
    static bool upscale(const Image& input, const TileEngine& engine,
                        int target_scale, Image& out,
                        TileProgressFn progress,
                        std::atomic<bool>& cancel_flag,
                        std::string& err);

private:
    // NPU 输出边界中值均衡 (逐字移植 _equalize_npu_tile_borders)
    static void equalize_npu_borders(float* new_tile, float* dst,
                                     int x0, int y0, int out_h, int out_w,
                                     int tile, int scale);

    // reflect 模式补边 (等效 np.pad(mode="reflect"), 支持多倍反射)
    static void pad_reflect(const Image& src, int pad_h, int pad_w, Image& dst);
};

} // namespace yb
