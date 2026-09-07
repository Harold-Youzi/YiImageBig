// ============================================================================
//  image.cpp - WIC 编解码 / Lanczos 重采样 / 颜色转换
// ============================================================================
#include "image.h"
#include "common.h"

#include <initguid.h>   // 必须在 wincodec.h 之前, 使 DEFINE_GUID 在本编译单元实例化
#include <wincodecsdk.h>
#include <objidl.h>
#include <ocidl.h>
#include <propidl.h>
#include <oleauto.h>

// stb: 纯头文件编解码 (不依赖 COM, 在 WIC 编码器注册异常的机器上仍可保存)
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_ONLY_GIF
#endif
#include <stb_image.h>
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <stb_image_write.h>

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace yb {

// MinGW 头文件的 CLSID_WICPngEncoder 值有误 (高位字段错误),
// 此处使用 Windows 官方值 {27949969-0864-434F-A40A-431C9BA47EFD}
static const GUID kClsidWicPngEncoder =
    {0x27949969, 0x0864, 0x434F, {0xA4, 0x0A, 0x43, 0x1C, 0x9B, 0xA4, 0x7E, 0xFD}};

// ---------------------------------------------------------------------------
// WIC 工厂
// ---------------------------------------------------------------------------
static IWICImagingFactory* wic_factory() {
    static IWICImagingFactory* factory = nullptr;
    if (!factory) {
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                      CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
        if (FAILED(hr)) return nullptr;
    }
    return factory;
}

// ---------------------------------------------------------------------------
// 读取任意 WIC 支持格式, 转为 24bpp BGR
// ---------------------------------------------------------------------------
// WIC 解码实现 (前置声明)
static bool load_image_wic(const std::wstring& path, Image& out, std::string& err);

bool load_image(const std::wstring& path, Image& out, std::string& err) {
    // ---- 主路径: WIC (支持 png/jpg/bmp/tiff/gif/webp/heif 等) ----
    if (load_image_wic(path, out, err)) return true;

    // ---- 后备: stb (png/jpg/bmp/tga/gif/psd 等) ----
    FILE* fp = _wfopen(path.c_str(), L"rb");
    if (!fp) { err = "无法打开文件"; return false; }
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load_from_file(fp, &w, &h, &comp, 3);
    fclose(fp);
    if (!data) {
        err = "不支持的图像格式或文件损坏 (WIC 与内置解码均失败)";
        return false;
    }
    out.alloc(w, h);
    // RGB -> BGR
    parallel_for(h, [&](int64_t b, int64_t e) {
        for (int64_t y = b; y < e; ++y) {
            const uint8_t* s = data + (size_t)y * w * 3;
            uint8_t* d = out.px.data() + (size_t)y * out.stride();
            for (int x = 0; x < w; ++x) {
                d[x * 3 + 0] = s[x * 3 + 2];
                d[x * 3 + 1] = s[x * 3 + 1];
                d[x * 3 + 2] = s[x * 3 + 0];
            }
        }
    });
    stbi_image_free(data);
    return true;
}

// WIC 解码实现
static bool load_image_wic(const std::wstring& path, Image& out, std::string& err) {
    IWICImagingFactory* f = wic_factory();
    if (!f) { err = "无法创建 WIC 工厂 (COM 未初始化?)"; return false; }

    IWICBitmapDecoder* dec = nullptr;
    HRESULT hr = f->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                              WICDecodeMetadataCacheOnDemand, &dec);
    if (FAILED(hr)) {
        err = str_format("WIC CreateDecoderFromFilename 失败 hr=0x%08lX (文件不存在或格式不支持)", (unsigned long)hr);
        return false;
    }

    IWICBitmapFrameDecode* frame = nullptr;
    hr = dec->GetFrame(0, &frame);
    if (SUCCEEDED(hr)) {
        IWICFormatConverter* conv = nullptr;
        hr = f->CreateFormatConverter(&conv);
        if (SUCCEEDED(hr)) {
            hr = conv->Initialize(frame, GUID_WICPixelFormat24bppBGR,
                                  WICBitmapDitherTypeNone, nullptr, 0.0,
                                  WICBitmapPaletteTypeCustom);
            if (SUCCEEDED(hr)) {
                UINT w = 0, h = 0;
                conv->GetSize(&w, &h);
                if (w == 0 || h == 0) {
                    err = "图像尺寸为 0";
                } else if ((uint64_t)w * h > 400000000ull) {
                    err = str_format("图像过大 (%u x %u), 超出处理上限", w, h);
                } else {
                    out.alloc((int)w, (int)h);
                    const size_t stride = out.stride();
                    const size_t row_pad = (stride % 4) ? (4 - stride % 4) : 0;
                    const size_t buf_stride = stride + row_pad;
                    std::vector<uint8_t> buf(buf_stride * h);
                    hr = conv->CopyPixels(nullptr, (UINT)buf_stride,
                                          (UINT)buf.size(), buf.data());
                    if (SUCCEEDED(hr)) {
                        // 去除行填充
                        for (int y = 0; y < out.h; ++y)
                            memcpy(out.px.data() + (size_t)y * stride,
                                   buf.data() + (size_t)y * buf_stride, stride);
                    } else {
                        err = str_format("WIC CopyPixels 失败 hr=0x%08lX", (unsigned long)hr);
                        out.clear();
                    }
                }
            } else {
                err = str_format("WIC 格式转换失败 hr=0x%08lX", (unsigned long)hr);
            }
            conv->Release();
        } else {
            err = str_format("WIC CreateFormatConverter 失败 hr=0x%08lX", (unsigned long)hr);
        }
        frame->Release();
    } else {
        err = str_format("WIC GetFrame 失败 hr=0x%08lX", (unsigned long)hr);
    }
    dec->Release();
    return out.empty() == false && err.empty();
}

// ---------------------------------------------------------------------------
// 保存 PNG / JPEG
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// stb 保存辅助: 编码到内存, 再以 unicode 路径写入
// ---------------------------------------------------------------------------
static void stb_append(void* ctx, void* data, int size) {
    auto* buf = (std::vector<uint8_t>*)ctx;
    const uint8_t* p = (const uint8_t*)data;
    buf->insert(buf->end(), p, p + size);
}

static bool write_file_w(const std::wstring& path, const std::vector<uint8_t>& data,
                         std::string& err) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = str_format("无法创建文件 hr=0x%08lX (目录不存在/权限/只读?)", (unsigned long)GetLastError());
        return false;
    }
    DWORD written = 0;
    BOOL ok = TRUE;
    const DWORD chunk = 32 * 1024 * 1024;
    for (size_t off = 0; off < data.size() && ok; ) {
        const DWORD n = (DWORD)std::min<size_t>(chunk, data.size() - off);
        ok = WriteFile(h, data.data() + off, n, &written, nullptr);
        off += written;
    }
    CloseHandle(h);
    if (!ok) { err = "写入文件内容失败"; return false; }
    return true;
}

// BGR(Image) -> RGB 连续缓冲
static std::vector<uint8_t> bgr_to_rgb(const Image& img) {
    std::vector<uint8_t> rgb((size_t)img.w * img.h * 3);
    parallel_for(img.h, [&](int64_t b, int64_t e) {
        for (int64_t y = b; y < e; ++y) {
            const uint8_t* s = img.px.data() + (size_t)y * img.stride();
            uint8_t* d = rgb.data() + (size_t)y * img.w * 3;
            for (int x = 0; x < img.w; ++x) {
                d[x * 3 + 0] = s[x * 3 + 2];
                d[x * 3 + 1] = s[x * 3 + 1];
                d[x * 3 + 2] = s[x * 3 + 0];
            }
        }
    });
    return rgb;
}

bool save_image(const std::wstring& path, const Image& img,
                const std::wstring& format, int jpeg_quality, std::string& err) {
    if (img.empty()) { err = "空图像无法保存"; return false; }
    const bool is_jpg = (format == L"jpg" || format == L"jpeg");

    // ---- 主路径: stb (无 COM 依赖, 全平台可靠) ----
    {
        std::vector<uint8_t> rgb = bgr_to_rgb(img);
        std::vector<uint8_t> out;
        int ok = 0;
        if (is_jpg) {
            ok = stbi_write_jpg_to_func(stb_append, &out, img.w, img.h, 3,
                                        rgb.data(), std::min(std::max(jpeg_quality, 1), 100));
        } else {
            ok = stbi_write_png_to_func(stb_append, &out, img.w, img.h, 3, rgb.data(), img.w * 3);
        }
        if (ok && !out.empty()) {
            // 编码成功: 文件写入失败即最终失败 (文件系统问题, WIC 也无法解决)
            std::string werr;
            if (write_file_w(path, out, werr)) return true;
            err = werr;
            return false;
        }
        // stb 编码失败 -> 回落 WIC
    }

    // ---- 后备: WIC 编码器 ----
    IWICImagingFactory* f = wic_factory();
    if (!f) { err = "无法创建 WIC 工厂"; return false; }

    const GUID& clsid = (format == L"jpg" || format == L"jpeg")
                            ? CLSID_WICJpegEncoder : kClsidWicPngEncoder;

    IWICBitmapEncoder* enc = nullptr;
    HRESULT hr = f->CreateEncoder(clsid, nullptr, &enc);
    if (FAILED(hr)) { err = str_format("WIC CreateEncoder 失败 hr=0x%08lX", (unsigned long)hr); return false; }

    IWICStream* stream = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;

    hr = f->CreateStream(&stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = enc->Initialize(stream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = enc->CreateNewFrame(&frame, &props);
    if (SUCCEEDED(hr)) {
        if (clsid == CLSID_WICJpegEncoder && props) {
            // JPEG 质量 0.0-1.0
            PROPBAG2 opt = {};
            opt.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
            opt.vt = VT_R4;
            VARIANT v;
            VariantInit(&v);
            v.vt = VT_R4;
            v.fltVal = std::min(std::max(jpeg_quality, 1), 100) / 100.0f;
            props->Write(1, &opt, &v);
        }
        hr = frame->Initialize(props);
        if (SUCCEEDED(hr)) {
            hr = frame->SetSize((UINT)img.w, (UINT)img.h);
            if (SUCCEEDED(hr)) {
                WICPixelFormatGUID pf = GUID_WICPixelFormat24bppBGR;
                hr = frame->SetPixelFormat(&pf);
                if (SUCCEEDED(hr)) {
                    const size_t stride = img.stride();
                    // WritePixels 只读该缓冲, const_cast 安全
                    BYTE* bits = const_cast<BYTE*>(img.px.data());
                    hr = frame->WritePixels((UINT)img.h, (UINT)stride,
                                            (UINT)(stride * img.h), bits);
                }
            }
        }
        if (SUCCEEDED(hr)) hr = frame->Commit();
        if (SUCCEEDED(hr)) hr = enc->Commit();
    }

    if (props)   props->Release();
    if (frame)   frame->Release();
    if (stream)  stream->Release();
    enc->Release();

    if (FAILED(hr)) {
        err = str_format("保存失败 hr=0x%08lX (磁盘空间/权限?)", (unsigned long)hr);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Lanczos3 重采样 (带下采样抗锯齿)
// ---------------------------------------------------------------------------
namespace {

inline double sinc_lanczos(double x) {
    if (x < 1e-8 && x > -1e-8) return 1.0;
    const double PI = 3.14159265358979323846;
    x *= PI;
    return std::sin(x) / x;
}

inline double lanczos3(double x) {
    const double a = 3.0;
    if (x < -a || x > a) return 0.0;
    return sinc_lanczos(x) * sinc_lanczos(x / a);
}

struct WeightTable {
    int   src_begin = 0;      // 该输出像素对应的源起点
    std::vector<double> w;    // 权重
};

// 生成一维重采样权重表 (支持放大/缩小)
static void build_weights(int src_len, int dst_len, std::vector<WeightTable>& table) {
    const double scale = (double)src_len / dst_len;
    const double filter_scale = scale > 1.0 ? scale : 1.0;  // 缩小时加宽窗口抗锯齿
    const double support = 3.0 * filter_scale;

    table.resize(dst_len);
    for (int dx = 0; dx < dst_len; ++dx) {
        double center = ((double)dx + 0.5) * scale - 0.5;
        int left  = (int)std::floor(center - support + 0.5);
        int right = (int)std::floor(center + support + 0.5);
        if (left < 0) left = 0;
        if (right > src_len - 1) right = src_len - 1;
        if (right < left) right = left;

        WeightTable& t = table[dx];
        t.src_begin = left;
        t.w.assign(right - left + 1, 0.0);
        double sum = 0.0;
        for (int sx = left; sx <= right; ++sx) {
            double d = ((double)sx) - center;
            double v = lanczos3(d / filter_scale);
            t.w[sx - left] = v;
            sum += v;
        }
        if (sum != 0.0) {
            for (auto& v : t.w) v /= sum;
        }
    }
}

} // anonymous namespace

Image resize_lanczos(const Image& src, int dw, int dh) {
    Image dst;
    if (src.empty() || dw <= 0 || dh <= 0) return dst;
    if (dw == src.w && dh == src.h) return src;

    dst.alloc(dw, dh);

    // ---- 水平/垂直权重 ----
    std::vector<WeightTable> wx, wy;
    build_weights(src.w, dw, wx);
    build_weights(src.h, dh, wy);

    // 分带处理以控制内存: 每批处理 64 个目标行
    const int band = 64;
    std::vector<float> hbuf;      // 水平重采样后的若干源行: band_src_rows * dw * 3
    std::vector<int>   band_src;  // 本带涉及的源行号

    for (int dy0 = 0; dy0 < dh; dy0 += band) {
        const int dy1 = std::min(dy0 + band, dh);

        // 收集本带需要的源行 (wy[dy].src_begin 区间并集)
        int smin = INT_MAX, smax = -1;
        for (int dy = dy0; dy < dy1; ++dy) {
            const WeightTable& t = wy[dy];
            smin = std::min(smin, t.src_begin);
            smax = std::max(smax, t.src_begin + (int)t.w.size() - 1);
        }
        band_src.clear();
        for (int sy = smin; sy <= smax; ++sy) band_src.push_back(sy);
        const int band_h = (int)band_src.size();
        hbuf.assign((size_t)band_h * dw * 3, 0.0f);

        // 水平重采样这些源行
        parallel_for(band_h, [&](int64_t b, int64_t e) {
            for (int64_t bi = b; bi < e; ++bi) {
                const uint8_t* srow = src.px.data() + (size_t)band_src[bi] * src.stride();
                float* drow = hbuf.data() + (size_t)bi * dw * 3;
                for (int dx = 0; dx < dw; ++dx) {
                    const WeightTable& t = wx[dx];
                    double r = 0, g = 0, bl = 0;
                    const int   n = (int)t.w.size();
                    const uint8_t* p = srow + (size_t)t.src_begin * 3;
                    for (int k = 0; k < n; ++k) {
                        const double wv = t.w[k];
                        const uint8_t* q = p + (size_t)k * 3;
                        bl += wv * q[0];
                        g  += wv * q[1];
                        r  += wv * q[2];
                    }
                    float* o = drow + (size_t)dx * 3;
                    o[0] = (float)bl; o[1] = (float)g; o[2] = (float)r;
                }
            }
        });

        // 垂直重采样得到目标行
        for (int dy = dy0; dy < dy1; ++dy) {
            const WeightTable& t = wy[dy];
            const int   n = (int)t.w.size();
            const int   base = t.src_begin - smin;   // band 内索引
            uint8_t* drow = dst.px.data() + (size_t)dy * dst.stride();
            for (int dx = 0; dx < dw; ++dx) {
                double r = 0, g = 0, bl = 0;
                const float* fp = hbuf.data() + ((size_t)base) * dw * 3 + (size_t)dx * 3;
                for (int k = 0; k < n; ++k) {
                    const double wv = t.w[k];
                    const float* q = fp + (size_t)k * dw * 3;
                    bl += wv * q[0];
                    g  += wv * q[1];
                    r  += wv * q[2];
                }
                uint8_t* o = drow + (size_t)dx * 3;
                o[0] = (uint8_t)std::min(255.0, std::max(0.0, bl + 0.5));
                o[1] = (uint8_t)std::min(255.0, std::max(0.0, g  + 0.5));
                o[2] = (uint8_t)std::min(255.0, std::max(0.0, r  + 0.5));
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// 颜色转换
// ---------------------------------------------------------------------------
void bgr8_to_chw_f32(const Image& img, std::vector<float>& out) {
    out.assign((size_t)img.w * img.h * 3, 0.0f);
    const size_t plane = (size_t)img.w * img.h;
    parallel_for(img.h, [&](int64_t b, int64_t e) {
        for (int64_t y = b; y < e; ++y) {
            const uint8_t* srow = img.px.data() + (size_t)y * img.stride();
            float* dch0 = out.data() + (size_t)y * img.w;             // B
            float* dch1 = dch0 + plane;                               // G
            float* dch2 = dch1 + plane;                               // R
            for (int x = 0; x < img.w; ++x) {
                dch0[x] = srow[(size_t)x * 3 + 0] / 255.0f;
                dch1[x] = srow[(size_t)x * 3 + 1] / 255.0f;
                dch2[x] = srow[(size_t)x * 3 + 2] / 255.0f;
            }
        }
    });
}

void chw_f32_to_bgr8(const float* data, int w, int h, Image& out) {
    out.alloc(w, h);
    const size_t plane = (size_t)w * h;
    parallel_for(h, [&](int64_t b, int64_t e) {
        for (int64_t y = b; y < e; ++y) {
            const float* ch0 = data + (size_t)y * w;              // B
            const float* ch1 = ch0 + plane;                       // G
            const float* ch2 = ch1 + plane;                       // R
            uint8_t* drow = out.px.data() + (size_t)y * out.stride();
            for (int x = 0; x < w; ++x) {
                drow[(size_t)x * 3 + 0] = (uint8_t)std::min(255.0f, std::max(0.0f, ch0[x] * 255.0f + 0.5f));
                drow[(size_t)x * 3 + 1] = (uint8_t)std::min(255.0f, std::max(0.0f, ch1[x] * 255.0f + 0.5f));
                drow[(size_t)x * 3 + 2] = (uint8_t)std::min(255.0f, std::max(0.0f, ch2[x] * 255.0f + 0.5f));
            }
        }
    });
}

} // namespace yb
