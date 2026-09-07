// ============================================================================
//  main.cpp - YiImageBig 入口
//    GUI 模式:   YiImageBig.exe
//    CLI 模式:   YiImageBig.exe --upscale <图片> [--device auto|gpu|npu|cpu|dml]
//                              [--scale 2|3|4] [--out <目录>] [--format png|jpg]
// ============================================================================
#include "gui/app.h"
#include "core/common.h"
#include "core/image.h"
#include "core/upscaler.h"
#include "core/devices.h"
#include "core/ov_engine.h"
#include "core/ort_engine.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// CLI 模式实现 (用于自动化测试与批处理脚本)
// ---------------------------------------------------------------------------
namespace yb {

static void cli_print(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // 控制台输出
    DWORD written = 0;
    WriteConsoleA(GetStdHandle(STD_OUTPUT_HANDLE), buf, (DWORD)strlen(buf), &written, nullptr);
    printf("%s", buf);
    fflush(stdout);
    // 同步写日志文件 (重定向场景下控制台输出可能丢失)
    Logger::instance().info(std::string(buf));
}

int run_cli(const std::vector<std::wstring>& args) {
    // 解析参数
    std::wstring input, out_dir = exe_dir() + L"\\output";
    std::string device = "auto";
    int scale = 4;
    std::wstring format = L"png";

    for (size_t i = 0; i < args.size(); ++i) {
        const std::wstring& a = args[i];
        auto next = [&](std::wstring& dst) -> bool {
            if (i + 1 >= args.size()) return false;
            dst = args[++i];
            return true;
        };
        if (a == L"--upscale") { if (!next(input)) { cli_print("缺少 --upscale 参数值\n"); return 2; } }
        else if (a == L"--device") { std::wstring v; if (!next(v)) return 2; device = wide_to_utf8(v); }
        else if (a == L"--scale") { std::wstring v; if (!next(v)) return 2; scale = _wtoi(v.c_str()); }
        else if (a == L"--out") { if (!next(out_dir)) return 2; }
        else if (a == L"--format") { if (!next(format)) return 2; }
    }

    if (input.empty()) {
        cli_print("用法: YiImageBig.exe --upscale <图片> [--device auto|gpu|npu|cpu|dml]\n"
                  "                  [--scale 2|3|4] [--out 目录] [--format png|jpg]\n");
        return 2;
    }
    if (scale != 2 && scale != 3 && scale != 4) scale = 4;
    if (format != L"png" && format != L"jpg") format = L"png";

    Logger::instance().set_file(app_data_dir() + L"\\YiImageBig.log");
    Logger::instance().info("========== CLI 模式 ==========");

    std::wstring runtime = exe_dir() + L"\\runtime";
    std::string err;

    OVEngine ov;
    ORTEngine ort;
    ov.init(runtime, err);
    ort.init(runtime, err);

    DeviceDetector det;
    det.detect(ov, ort);

    // 选择设备
    const DeviceInfo* dev = nullptr;
    if (device == "auto") {
        dev = det.optimal();
    } else {
        for (auto& d : det.devices()) {
            std::string k;
            switch (d.kind) {
            case DevKind::OpenVinoGpu:  k = "gpu";  break;
            case DevKind::OpenVinoNpu:  k = "npu";  break;
            case DevKind::OpenVinoCpu:  k = "cpu";  break;
            case DevKind::DirectML:     k = "dml";  break;
            default: k = "cpu"; break;
            }
            if (k == device) { dev = &d; break; }
        }
    }
    if (!dev) {
        cli_print("[错误] 设备 \"%s\" 不可用. 可用设备:\n", device.c_str());
        for (auto& d : det.devices())
            cli_print("  - %s (%s)\n", d.name.c_str(), d.full_name.c_str());
        return 3;
    }
    cli_print("[设备] %s (%s, %s)\n", dev->name.c_str(), dev->full_name.c_str(),
              dev->backend.c_str());

    // 加载模型
    std::wstring models_dir = exe_dir() + L"\\models";
    TileEngine te;
    if (dev->kind == DevKind::DirectML) {
        std::wstring onnx = models_dir + L"\\RealESRGAN_x4plus.onnx";
        ORTEngine::Session s;
        if (!ort.create_session(onnx, true, dev->dml_adapter, 256, s, err)) {
            cli_print("[错误] DirectML 会话失败: %s\n", err.c_str());
            return 4;
        }
        te.tile = s.tile_in; te.overlap = 32; te.slots = 1;
        ORTEngine* p = &ort;
        te.infer = [p, s](const float* in, float* out, std::string& e2) mutable {
            return p->run(s, in, out, e2);
        };
    } else {
        std::string devname = dev->device_id;
        std::wstring xml = devname == "NPU"
                               ? models_dir + L"\\RealESRGAN_x4plus_npu_128.xml"
                               : models_dir + L"\\RealESRGAN_x4plus.xml";
        std::vector<std::pair<std::string, std::string>> props;
        if (devname == "NPU")      props = {{"PERFORMANCE_HINT", "LATENCY"}};
        else if (devname == "GPU") props = {{"PERFORMANCE_HINT", "THROUGHPUT"},
                                            {"INFERENCE_PRECISION_HINT", "f16"}};
        else                       props = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
        OVEngine::Model m;
        if (!ov.load_model(xml, devname, props, devname == "NPU" ? 128 : 256, m, err)) {
            cli_print("[错误] 模型编译失败: %s\n", err.c_str());
            return 4;
        }
        te.tile = m.tile_in;
        te.overlap = devname == "NPU" ? 64 : 32;
        te.npu = devname == "NPU";
        if (te.npu) {
            te.slots = 1;
            OVEngine* p = &ov;
            te.infer = [p, m](const float* in, float* out, std::string& e2) mutable {
                return p->infer_sync(m.reqs[0], in, out, e2);
            };
        } else {
            te.slots = 2;
            OVEngine* p = &ov;
            te.astart = [p, m](const float* in, int slot, std::string& e2) mutable {
                return p->async_start(m.reqs[(size_t)slot], in, e2);
            };
            te.await = [p, m](float* out, int slot, std::string& e2) mutable {
                OVEngine::Request& rq = m.reqs[(size_t)slot];
                if (!p->async_wait(rq, e2)) return false;
                return p->read_output(rq, out, e2);
            };
            te.cancel = [p, m]() mutable { p->cancel(m); };
        }
    }
    cli_print("[模型] tile=%d overlap=%d\n", te.tile, te.overlap);

    // 读取并放大
    Image img;
    if (!load_image(input, img, err)) {
        cli_print("[错误] 读取失败: %s\n", err.c_str());
        return 5;
    }
    cli_print("[输入] %dx%d\n", img.w, img.h);

    CreateDirectoryW(out_dir.c_str(), nullptr);
    const wchar_t* ext = format == L"jpg" ? L".jpg" : L".png";
    wchar_t tag[16];
    swprintf(tag, 16, L"_x%d", scale);
    std::wstring outp = unique_path(out_dir + L"\\" + path_stem(input) + tag + ext);

    std::atomic<bool> nocancel{false};
    auto t0 = std::chrono::steady_clock::now();
    auto progress = [](int d, int t) {
        if (t > 0 && d % 8 == 0) cli_print("  进度 %d/%d\n", d, t);
    };

    Image result;
    if (!Upscaler::upscale(img, te, scale, result, progress, nocancel, err)) {
        cli_print("[错误] 超分失败: %s\n", err.c_str());
        return 6;
    }
    double sec = seconds_since(t0);

    if (!save_image(outp, result, format, 95, err)) {
        cli_print("[错误] 保存失败: %s\n", err.c_str());
        return 7;
    }
    cli_print("[输出] %s (%dx%d, %.2fs)\n",
              wide_to_utf8(path_filename(outp)).c_str(), result.w, result.h, sec);
    cli_print("[完成] OK\n");
    return 0;
}

} // namespace yb

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR cmdLine, int nShow) {
    (void)cmdLine;

    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    // CLI 模式: 带参数启动时 (第一个参数为 --upscale)
    if (argc > 1 && argv && wcsstr(argv[1], L"--upscale") != nullptr) {
        std::vector<std::wstring> args;
        for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
        bool no_pause = false;
        for (auto& a : args)
            if (a == L"--no-pause") { no_pause = true; break; }
        if (argv) LocalFree(argv);

        // CLI 模式需要 COM (WIC 图像读写)
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

        // 附着控制台输出 (优先复用父进程控制台)
        bool interactive = AttachConsole(ATTACH_PARENT_PROCESS) != 0;
        if (interactive) {
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
            freopen("CONIN$", "r", stdin);
        } else {
            AllocConsole();
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
        }
        int rc = yb::run_cli(args);
        // 仅交互式控制台暂停 (--no-pause 跳过, 便于自动化测试)
        DWORD mode = 0;
        if (!no_pause && interactive &&
            GetFileType(GetStdHandle(STD_INPUT_HANDLE)) == FILE_TYPE_CHAR &&
            GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode)) {
            printf("\nPress Enter to exit...");
            fflush(stdout);
            wchar_t tmp[8];
            DWORD read = 0;
            ReadConsoleW(GetStdHandle(STD_INPUT_HANDLE), tmp, 1, &read, nullptr);
        }
        CoUninitialize();
        return rc;
    }
    if (argv) LocalFree(argv);

    // GUI 模式
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    int rc = yb::App::instance().run(hInst, nShow);
    CoUninitialize();
    return rc;
}
