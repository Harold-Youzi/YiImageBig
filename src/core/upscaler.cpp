// ============================================================================
//  upscaler.cpp - 分块推理与融合实现
// ============================================================================
#include "upscaler.h"
#include "common.h"

#include <windows.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace yb {

// ---------------------------------------------------------------------------
// 权值图: ramp[i] = 0.5*(1-cos(2*pi*i/size)), 二维可分离乘积
// ---------------------------------------------------------------------------
std::vector<float> Upscaler::make_weight_map(int size) {
    std::vector<float> axis(size);
    for (int i = 0; i < size; ++i) {
        float t = (float)i / (float)size;
        axis[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * t));
    }
    std::vector<float> map((size_t)size * size);
    parallel_for(size, [&](int64_t b, int64_t e) {
        for (int64_t y = b; y < e; ++y)
            for (int x = 0; x < size; ++x)
                map[(size_t)y * size + x] = axis[y] * axis[x];
    });
    return map;
}

// ---------------------------------------------------------------------------
// tile 起点: step = tile - overlap, 不足时补右/下边缘对齐点
// 等效 Python:
//   ys = list(range(0, max(ph - tile + 1, 1), step))
//   if ys[-1] + tile < ph: ys.append(max(ph - tile, 0))
// ---------------------------------------------------------------------------
std::vector<std::pair<int, int>> Upscaler::tile_positions(int ph, int pw, int tile, int overlap) {
    const int step = std::max(1, tile - overlap);
    auto axis = [&](int len) {
        std::vector<int> ys;
        int limit = std::max(len - tile + 1, 1);
        for (int v = 0; v < limit; v += step)
            ys.push_back(v);
        if (ys.empty()) ys.push_back(0);
        if ((size_t)ys.back() + tile < (size_t)len)
            ys.push_back(std::max(len - tile, 0));
        return ys;
    };
    std::vector<int> ys = axis(ph), xs = axis(pw);
    std::vector<std::pair<int, int>> pos;
    pos.reserve(ys.size() * xs.size());
    for (int y : ys)
        for (int x : xs)
            pos.emplace_back(y, x);
    return pos;
}

// ---------------------------------------------------------------------------
// reflect 补边 (np.pad mode="reflect", 即 reflect_101, 支持多倍反射)
// ---------------------------------------------------------------------------
static int reflect_index(int i, int n) {
    if (n == 1) return 0;
    const int period = 2 * n - 2;
    int m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
}

void Upscaler::pad_reflect(const Image& src, int pad_h, int pad_w, Image& dst) {
    const int sh = src.h, sw = src.w;
    const int dh = sh + pad_h, dw = sw + pad_w;
    dst.alloc(dw, dh);
    if (pad_h == 0 && pad_w == 0) {
        dst.px = src.px;
        return;
    }
    parallel_for(dh, [&](int64_t b, int64_t e) {
        for (int64_t y64 = b; y64 < e; ++y64) {
            const int y = (int)y64;
            const int sy = y < sh ? y : reflect_index(y, sh);
            const uint8_t* srow = src.px.data() + (size_t)sy * src.stride();
            uint8_t* drow = dst.px.data() + (size_t)y * dst.stride();
            // 原图区域整体拷贝
            if (pad_w == 0) {
                memcpy(drow, srow, (size_t)sw * 3);
            } else {
                for (int x = 0; x < sw; ++x) {
                    const uint8_t* p = srow + (size_t)x * 3;
                    uint8_t* d = drow + (size_t)x * 3;
                    d[0] = p[0]; d[1] = p[1]; d[2] = p[2];
                }
                for (int x = sw; x < dw; ++x) {
                    const int sx = reflect_index(x, sw);
                    const uint8_t* p = srow + (size_t)sx * 3;
                    uint8_t* d = drow + (size_t)x * 3;
                    d[0] = p[0]; d[1] = p[1]; d[2] = p[2];
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NPU 边界中值均衡 (逐字移植 Python 版)
//   new_tile: (3, t4, t4) CHW;  dst: (3, out_h, out_w)
// ---------------------------------------------------------------------------
void Upscaler::equalize_npu_borders(float* new_tile, float* dst,
                                    int x0, int y0, int out_h, int out_w,
                                    int tile, int scale) {
    const int s4 = tile * scale;
    const int px = x0 * scale, py = y0 * scale;
    const int px1 = std::min(px + s4, out_w);
    const int py1 = std::min(py + s4, out_h);
    if (px1 - px != s4 || py1 - py != s4) return;   // 非整块不做均衡

    const size_t out_plane = (size_t)out_h * out_w;
    const size_t tile_plane = (size_t)s4 * s4;

    // 中值: 对 3 通道分别取中值 (复用 nth_element)
    auto median_of = [&](std::vector<float>& v) {
        if (v.empty()) return 0.0f;
        size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    };

    // 左边界: new_tile[:, :, :3] vs dst[:, py:py1, px-3:px]
    if (px >= 3) {
        std::vector<float> m(3, 0.0f);
        bool apply = true;
        std::vector<float> sa, sb;
        for (int c = 0; c < 3 && apply; ++c) {
            sa.clear();
            for (int r = 0; r < s4; ++r)
                for (int k = 0; k < 3; ++k)
                    sa.push_back(new_tile[(size_t)c * tile_plane + (size_t)r * s4 + k]);
            sb.clear();
            for (int r = 0; r < s4; ++r)
                for (int k = 0; k < 3; ++k)
                    sb.push_back(dst[(size_t)c * out_plane + (size_t)(py + r) * out_w + (px - 3 + k)]);
            float d = median_of(sa) - median_of(sb);
            if (!(std::fabs(d) < 24.0f)) apply = false;
            m[c] = d;
        }
        if (apply) {
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < s4; ++r)
                    for (int k = 0; k < 3; ++k)
                        new_tile[(size_t)c * tile_plane + (size_t)r * s4 + k] -= m[c];
        }
    }
    // 上边界: new_tile[:, :3, :] vs dst[:, py-3:py, px:px1]
    if (py >= 3) {
        std::vector<float> m(3, 0.0f);
        bool apply = true;
        std::vector<float> sa, sb;
        for (int c = 0; c < 3 && apply; ++c) {
            sa.clear();
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < s4; ++k)
                    sa.push_back(new_tile[(size_t)c * tile_plane + (size_t)r * s4 + k]);
            sb.clear();
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < s4; ++k)
                    sb.push_back(dst[(size_t)c * out_plane + (size_t)(py - 3 + r) * out_w + px + k]);
            float d = median_of(sa) - median_of(sb);
            if (!(std::fabs(d) < 24.0f)) apply = false;
            m[c] = d;
        }
        if (apply) {
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r)
                    for (int k = 0; k < s4; ++k)
                        new_tile[(size_t)c * tile_plane + (size_t)r * s4 + k] -= m[c];
        }
    }
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
bool Upscaler::upscale(const Image& input, const TileEngine& engine,
                       int target_scale, Image& out,
                       TileProgressFn progress,
                       std::atomic<bool>& cancel_flag,
                       std::string& err) {
    if (input.empty()) { err = "空图像"; return false; }
    if (!engine.valid()) { err = "推理引擎未就绪"; return false; }

    const int tile = engine.tile;
    const int overlap = engine.overlap;
    const int t4 = tile * SCALE;

    // ---- 1. 右/下补边到 tile 整数倍 (reflect) ----
    const int pad_h = (tile - (input.h % tile)) % tile;
    const int pad_w = (tile - (input.w % tile)) % tile;
    Image padded;
    pad_reflect(input, pad_h, pad_w, padded);
    const int ph = padded.h, pw = padded.w;
    const int out_h = ph * SCALE, out_w = pw * SCALE;

    // ---- 2. BGR8(HWC) -> float32 RGB(CHW) /255 ----
    std::vector<float> input_f;
    bgr8_to_chw_f32(padded, input_f);
    padded.clear();

    // ---- 3. 输出/权值累加缓冲 ----
    // 内存保护: 输出 + 权值 ≈ out_px*16 字节
    MEMORYSTATUSEX ms = { sizeof(ms) };
    GlobalMemoryStatusEx(&ms);
    const uint64_t need = (uint64_t)out_h * out_w * 16;
    if (need > ms.ullAvailPhys * 4 / 5 && need > (uint64_t)2 << 30) {
        err = str_format("图片放大后需要约 %s 内存, 可用 %s, 内存不足",
                         human_size(need).c_str(), human_size(ms.ullAvailPhys).c_str());
        return false;
    }

    std::vector<float> output((size_t)out_h * out_w * 3, 0.0f);
    std::vector<float> weight((size_t)out_h * out_w, 0.0f);
    std::vector<float> tile_w = make_weight_map(t4);

    auto positions = tile_positions(ph, pw, tile, overlap);
    const size_t total = positions.size();
    const size_t tile_elems = (size_t)t4 * t4 * 3;

    // 单块融合: out[c, y4+r, x4+k] += t[c,r,k] * w[r,k]
    auto fuse = [&](const float* t, int x, int y) {
        const float* w = tile_w.data();
        parallel_for(t4, [&](int64_t b, int64_t e) {
            for (int64_t r = b; r < e; ++r) {
                // tile 起点 (x,y) 是 padded 图坐标, 输出图上为 (x*SCALE, y*SCALE);
                // 不能乘 t4(tile*SCALE), 否则行号越界导致堆损坏/崩溃
                const size_t orow = (size_t)(y * SCALE + r) * out_w + (size_t)x * SCALE;
                const size_t trow = (size_t)r * t4;
                float* wrow = weight.data() + orow;
                for (int k = 0; k < t4; ++k) {
                    const float wv = w[trow + k];
                    wrow[k] += wv;
                }
                for (int c = 0; c < 3; ++c) {
                    float* orowp = output.data() + (size_t)c * out_h * out_w + orow;
                    const float* trowp = t + (size_t)c * t4 * t4 + trow;
                    for (int k = 0; k < t4; ++k)
                        orowp[k] += trowp[k] * w[trow + k];
                }
            }
        });
    };

    // ---- 4. 分块推理 ----
    std::vector<float> tile_in_buf((size_t)tile * tile * 3);
    std::vector<float> tile_out_bufs[2];
    tile_out_bufs[0].assign(tile_elems, 0.0f);
    tile_out_bufs[1].assign(tile_elems, 0.0f);

    auto extract = [&](int x, int y, float* buf) {
        parallel_for(tile, [&](int64_t b, int64_t e) {
            for (int64_t r = b; r < e; ++r) {
                const size_t srow = (size_t)(y + r) * pw + (size_t)x;
                const size_t drow = (size_t)r * tile;
                for (int c = 0; c < 3; ++c) {
                    const float* sp = input_f.data() + (size_t)c * ph * pw + srow;
                    float* dp = buf + (size_t)c * tile * tile + drow;
                    for (int k = 0; k < tile; ++k)
                        dp[k] = sp[k];
                }
            }
        });
    };

    int done = 0;
    auto finish_tile = [&](int idx, int slot) {
        const int x = positions[idx].second, y = positions[idx].first;
        float* t = tile_out_bufs[slot & 1].data();
        if (engine.npu)
            equalize_npu_borders(t, output.data(), x, y, out_h, out_w, tile, SCALE);
        fuse(t, x, y);
        ++done;
        if (progress) progress(done, (int)total);
    };

    // NPU / ORT: 同步逐块
    if (engine.slots <= 1) {
        for (size_t i = 0; i < total; ++i) {
            if (cancel_flag) break;
            const int x = positions[i].second, y = positions[i].first;
            extract(x, y, tile_in_buf.data());
            if (!engine.infer(tile_in_buf.data(), tile_out_bufs[0].data(), err)) {
                err = str_format("第 %d/%d 块推理失败: %s", (int)i + 1, (int)total, err.c_str());
                return false;
            }
            finish_tile((int)i, 0);
        }
    } else {
        // OpenVINO GPU/CPU: 双请求异步流水线
        auto start_one = [&](int idx, int slot, std::string& e2) -> bool {
            const int x = positions[idx].second, y = positions[idx].first;
            extract(x, y, tile_in_buf.data());
            return engine.astart(tile_in_buf.data(), slot, e2);
        };

        if (!start_one(0, 0, err)) return false;
        for (size_t i = 1; i < total; ++i) {
            if (cancel_flag) break;
            int slot = (int)(i & 1);
            if (!start_one((int)i, slot, err)) {
                if (engine.cancel) engine.cancel();
                return false;
            }
            int prev = (int)((i - 1) & 1);
            if (!engine.await(tile_out_bufs[prev].data(), prev, err)) {
                if (engine.cancel) engine.cancel();
                return false;
            }
            finish_tile((int)(i - 1), prev);
        }
        if (!cancel_flag) {
            int last = (int)((total - 1) & 1);
            if (!engine.await(tile_out_bufs[last].data(), last, err)) {
                if (engine.cancel) engine.cancel();
                return false;
            }
            finish_tile((int)(total - 1), last);
        } else {
            // 取消: 等待未完成的异步任务安全结束
            if (engine.cancel) engine.cancel();
            for (int s = 0; s < engine.slots; ++s) {
                std::string e2;
                engine.await(tile_out_bufs[s & 1].data(), s, e2); // 尽力等待, 忽略错误
            }
        }
    }

    if (cancel_flag) { err = "__CANCELLED__"; return false; }

    // ---- 5. 权值归一化 ----
    parallel_for((int64_t)out_h * out_w, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float wv = std::max(weight[i], 1e-6f);
            output[i] /= wv;
            output[(size_t)out_h * out_w + i] /= wv;
            output[(size_t)out_h * out_w * 2 + i] /= wv;
        }
    });

    // ---- 6. float RGB(CHW) -> BGR8(HWC), 裁剪到原尺寸*4 ----
    const int final_h = input.h * SCALE, final_w = input.w * SCALE;
    Image cropped4;
    chw_f32_to_bgr8(output.data(), out_w, out_h, cropped4);
    output.clear(); weight.clear();

    if (final_h == out_h && final_w == out_w) {
        out = std::move(cropped4);
    } else {
        out.alloc(final_w, final_h);
        parallel_for(final_h, [&](int64_t b, int64_t e) {
            for (int64_t y = b; y < e; ++y)
                memcpy(out.px.data() + (size_t)y * out.stride(),
                       cropped4.px.data() + (size_t)y * cropped4.stride(),
                       (size_t)final_w * 3);
        });
    }

    // ---- 7. 非 4x 目标: Lanczos 降采样 (2x / 3x) ----
    if (target_scale > 0 && target_scale != SCALE) {
        Image scaled = resize_lanczos(out, final_w * target_scale / SCALE,
                                      final_h * target_scale / SCALE);
        out = std::move(scaled);
    }
    return true;
}

} // namespace yb
