// ============================================================================
//  app.h - YiImageBig 主窗口 (Win32 原生 GUI)
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

#include "../core/image.h"
#include "../core/ov_engine.h"
#include "../core/ort_engine.h"
#include "../core/upscaler.h"
#include "../core/devices.h"

namespace yb {

// 控件创建助手 (widgets.cpp)
HWND make_label(HWND parent, HFONT font, const wchar_t* text, int x, int y, int w, int h);
HWND make_button(HWND parent, HFONT font, const wchar_t* text, int id, int x, int y, int w, int h);
HWND make_combo(HWND parent, HFONT font, int id, int x, int y, int w, int h);
HWND make_edit(HWND parent, HFONT font, int id, int x, int y, int w, int h, DWORD extra_style = 0);
HWND make_groupbox(HWND parent, HFONT font, const wchar_t* text, int x, int y, int w, int h);
HWND make_progress(HWND parent, int id, int x, int y, int w, int h);

// 控件 ID
enum {
    IDC_COMBO_DEVICE = 101,
    IDC_STATIC_DEVINFO,
    IDC_COMBO_SCALE,
    IDC_COMBO_FORMAT,
    IDC_EDIT_OUTDIR,
    IDC_BTN_BROWSE,
    IDC_BTN_OPENDIR,
    IDC_LIST_FILES,
    IDC_BTN_ADD,
    IDC_BTN_REMOVE,
    IDC_BTN_CLEAR,
    IDC_BTN_START,
    IDC_BTN_CANCEL,
    IDC_PROGRESS,
    IDC_STATIC_STATUS,
    IDC_EDIT_LOG,
    IDC_STATIC_HINT,
};

// 文件条目状态
enum class FileState { Pending, Working, Done, Failed, Cancelled };

struct FileItem {
    std::wstring path;
    std::wstring size_txt;  // 分辨率
    std::wstring status;    // 状态文本
    std::wstring elapsed;   // 耗时
    FileState    state = FileState::Pending;
};

class App {
public:
    static App& instance();

    int run(HINSTANCE hInst, int nCmdShow);   // 消息循环

private:
    App() = default;

    // --- 窗口 ---
    static LRESULT CALLBACK wnd_proc_static(HWND, UINT, WPARAM, LPARAM);
    LRESULT wnd_proc(HWND, UINT, WPARAM, LPARAM);
    void create_controls(HWND hwnd);
    void layout(int width, int height);
    int  scale_dpi(int v) const { return MulDiv(v, dpi_, 96); }

    // --- 事件 ---
    void on_add_files();
    void on_remove_selected();
    void on_clear_files();
    void on_browse_outdir();
    void on_open_outdir();
    void on_start();
    void on_cancel();
    void on_drop(void* hdrop);   // HDROP (shellapi 完整定义在 cpp)
    void on_detect_done();

    // --- 列表 ---
    void list_refresh_item(int idx);
    void list_update_column_widths();
    void log_append(const std::string& utf8);       // UTF-8
    void set_status(const std::wstring& s);
    void set_controls_running(bool running);
    void post_status(const std::wstring& s);        // 线程安全

    // --- 配置 ---
    void load_config();
    void save_config();
    std::wstring config_get(const wchar_t* key, const wchar_t* def);
    void config_set(const wchar_t* key, const wchar_t* val);

    // --- 后台线程 ---
    void detect_thread();
    void worker_thread(std::vector<std::wstring> files);
    bool ensure_model(const std::string& device_key, std::string& err);  // 按需加载/切换模型
    bool build_tile_engine(TileEngine& te, std::string& err);

    // --- 工具 ---
    std::wstring out_path_for(const std::wstring& src) const;
    void mark_item(int idx, FileState st, const std::wstring& status,
                   const std::wstring& size_txt, const std::wstring& elapsed);

    // --- 数据 ---
    HWND hwnd_ = nullptr;
    HINSTANCE inst_ = nullptr;
    HFONT font_ = nullptr, font_log_ = nullptr, font_bold_ = nullptr;
    int dpi_ = 96;

    std::vector<HWND> ctrl_;
    HWND lbl_dev_ = nullptr, lbl_scale_ = nullptr, lbl_fmt_ = nullptr;
    std::vector<FileItem> items_;
    std::mutex items_mtx_;

    // 引擎与设备
    OVEngine   ov_;
    ORTEngine  ort_;
    DeviceDetector detector_;
    std::thread detect_th_;
    std::atomic<bool> detect_done_{false};
    std::atomic<bool> detect_quit_{false};

    // 处理状态
    std::thread work_th_;
    std::atomic<bool> cancel_flag_{false};
    std::atomic<bool> running_{false};
    std::wstring processing_outdir_;

    // 模型缓存 (worker 线程专用)
    struct ModelCache {
        std::string device_key;      // "ov:GPU" / "dml:0" / "ov:NPU" / "ov:CPU"
        OVEngine::Model  ov_model;
        ORTEngine::Session ort_session;
        bool use_ort = false;
        bool valid = false;
    } cache_;

    // 当前选择
    int  sel_device_ = 0;        // 0 = 自动
    int  sel_scale_ = 4;
    int  sel_format_ = 0;        // 0=PNG 1=JPEG
    std::wstring outdir_;
};

// CLI 模式 (命令行超分, 便于自动化测试)
int run_cli(const std::vector<std::wstring>& args);

} // namespace yb
