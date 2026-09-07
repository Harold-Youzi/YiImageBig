// ============================================================================
//  image.h - 图像缓冲区 / WIC 读写 / Lanczos 重采样 / 颜色转换
//  移植自 Python 版 image_upscaler.py (unicode 安全读写、LANCZOS 缩放)
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace yb {

// ---------------------------------------------------------------------------
// 基础图像: 8bit BGR 三通道, 紧密排列 (stride = w*3)
// ---------------------------------------------------------------------------
struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;   // size = w*h*3, 顺序 BGR

    bool empty() const { return w <= 0 || h <= 0 || px.empty(); }
    size_t stride() const { return (size_t)w * 3; }
    void alloc(int width, int height) {
        w = width; h = height;
        px.assign((size_t)w * h * 3, 0);
    }
    void clear() { w = h = 0; px.clear(); }
};

// ---------------------------------------------------------------------------
// 读写 (unicode 路径安全): WIC 主路径 + stb 内置编解码后备
// 读取支持 png/jpg/jpeg/bmp/tif/gif/webp 等, 保存 png/jpg
// ---------------------------------------------------------------------------
bool load_image(const std::wstring& path, Image& out, std::string& err);
bool save_image(const std::wstring& path, const Image& img,
                const std::wstring& format /* L"png" | L"jpg" */,
                int jpeg_quality /* 1-95 */, std::string& err);

// ---------------------------------------------------------------------------
// Lanczos 重采样 (等效 cv2.INTER_LANCZOS4, 自适应抗锯齿)
// ---------------------------------------------------------------------------
Image resize_lanczos(const Image& src, int dw, int dh);

// ---------------------------------------------------------------------------
// BGR8(HWC) <-> float32 RGB(CHW) 转换 (推理数据布局, /255)
// ---------------------------------------------------------------------------
void bgr8_to_chw_f32(const Image& img, std::vector<float>& out);      // [0,1]
void chw_f32_to_bgr8(const float* data, int w, int h, Image& out);    // 逐像素截断

} // namespace yb
