// ============================================================================
//  app.cpp - YiImageBig 主窗口实现
// ============================================================================
#include "app.h"
#include "../core/common.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlwapi.h>
#include <shlobj.h>

#include <thread>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cwchar>

namespace yb {

// 自定义消息
static constexpr UINT WM_APP_LOG      = WM_APP + 1;   // lParam: heap std::string*
static constexpr UINT WM_APP_PROGRESS = WM_APP + 2;   // wParam 进度码, 见 wnd_proc
static constexpr UINT WM_APP_ITEM     = WM_APP + 3;   // lParam: heap ItemUpd*
static constexpr UINT WM_APP_DETECT   = WM_APP + 4;   // 设备检测完成
static constexpr UINT WM_APP_STATUS   = WM_APP + 5;   // lParam: heap std::wstring*

struct ItemUpd {
    int idx;
    FileState state;
    std::wstring status;
    std::wstring size_txt;
    std::wstring elapsed;
};

App& App::instance() {
    static App g;
    return g;
}

// ---------------------------------------------------------------------------
// 配置存取 (%APPDATA%\YiImageBig\config.ini)
// ---------------------------------------------------------------------------
static std::wstring config_path() { return app_data_dir() + L"\\config.ini"; }

std::wstring App::config_get(const wchar_t* key, const wchar_t* def) {
    wchar_t buf[MAX_PATH] = {};
    GetPrivateProfileStringW(L"settings", key, def, buf, MAX_PATH, config_path().c_str());
    return buf;
}

void App::config_set(const wchar_t* key, const wchar_t* val) {
    WritePrivateProfileStringW(L"settings", key, val, config_path().c_str());
}

void App::load_config() {
    outdir_       = config_get(L"outdir", (exe_dir() + L"\\output").c_str());
    sel_scale_    = _wtoi(config_get(L"scale", L"4").c_str());
    if (sel_scale_ != 2 && sel_scale_ != 3 && sel_scale_ != 4) sel_scale_ = 4;
    sel_format_   = _wtoi(config_get(L"format", L"0").c_str());
    sel_device_   = _wtoi(config_get(L"device", L"0").c_str());
}

void App::save_config() {
    config_set(L"outdir", outdir_.c_str());
    wchar_t b[16];
    _itow_s(sel_scale_, b, 10); config_set(L"scale", b);
    _itow_s(sel_format_, b, 10); config_set(L"format", b);
    _itow_s(sel_device_, b, 10); config_set(L"device", b);
}

// ---------------------------------------------------------------------------
// 日志 / 状态
// ---------------------------------------------------------------------------
void App::log_append(const std::string& utf8) {
    if (!ctrl_[IDC_EDIT_LOG - 101]) return;
    std::wstring line = utf8_to_wide(now_time_str()) + L"  " + utf8_to_wide(utf8) + L"\r\n";
    HWND e = ctrl_[IDC_EDIT_LOG - 101];
    int len = GetWindowTextLengthW(e);
    SendMessageW(e, EM_SETSEL, len, len);
    if (len > 128 * 1024) {   // 限制日志长度
        SendMessageW(e, EM_SETSEL, 0, 64 * 1024);
        SendMessageW(e, EM_REPLACESEL, FALSE, (LPARAM)L"");
        len = GetWindowTextLengthW(e);
        SendMessageW(e, EM_SETSEL, len, len);
    }
    SendMessageW(e, EM_REPLACESEL, FALSE, (LPARAM)line.c_str());
}

void App::set_status(const std::wstring& s) {
    if (ctrl_[IDC_STATIC_STATUS - 101])
        SetWindowTextW(ctrl_[IDC_STATIC_STATUS - 101], s.c_str());
}

void App::post_status(const std::wstring& s) {
    PostMessageW(hwnd_, WM_APP_STATUS, 0, (LPARAM)new std::wstring(s));
}

// ---------------------------------------------------------------------------
// 窗口创建
// ---------------------------------------------------------------------------
int App::run(HINSTANCE hInst, int nCmdShow) {
    inst_ = hInst;

    INITCOMMONCONTROLSEX icc = { sizeof(icc),
        ICC_LISTVIEW_CLASSES | ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    // Per-Monitor DPI 感知 (动态加载, 兼容旧系统)
    using FnSetCtx = DPI_AWARENESS_CONTEXT (WINAPI*)(DPI_AWARENESS_CONTEXT);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    auto set_ctx = (FnSetCtx)GetProcAddress(user32, "SetThreadDpiAwarenessContext");
    if (set_ctx) set_ctx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    Logger::instance().set_file(app_data_dir() + L"\\YiImageBig.log");
    Logger::instance().info("========== YiImageBig 启动 ==========");

    // 公共对话框 (GetOpenFileNameW) 内部依赖 COM; 主线程必须以 STA 模式初始化,
    // 否则对话框行为不稳定 (可能直接失败), 尤其存在外部 COM 客户端交互时
    HRESULT hrCo = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(hrCo))
        Logger::instance().warn(str_format("CoInitializeEx 失败: 0x%08lX", (unsigned long)hrCo));

    WNDCLASSEXW wc = { sizeof(wc) };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = &App::wnd_proc_static;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(hInst, MAKEINTRESOURCEW(1));
    wc.hIconSm = wc.hIcon;
    wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    wc.lpszClassName = L"YiImageBigWnd";
    RegisterClassExW(&wc);

    load_config();

    int ww = scale_dpi(980), wh = scale_dpi(700);
    RECT work = {};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    int wx = work.left + ((work.right - work.left) - ww) / 2;
    int wy = work.top  + ((work.bottom - work.top) - wh) / 2;

    hwnd_ = CreateWindowExW(0, wc.lpszClassName,
                            L"YiImageBig — 图片超分工具 (RealESRGAN x4)",
                            WS_OVERLAPPEDWINDOW,
                            wx, wy, ww, wh, nullptr, nullptr, hInst, this);
    if (!hwnd_) return -1;

    ShowWindow(hwnd_, nCmdShow);
    UpdateWindow(hwnd_);

    // 后台检测硬件
    detect_th_ = std::thread([this] { detect_thread(); });

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        if (IsDialogMessageW(hwnd_, &msg)) continue;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    detect_quit_ = true;
    if (detect_th_.joinable()) detect_th_.join();
    if (work_th_.joinable()) work_th_.join();
    CoUninitialize();
    return (int)msg.wParam;
}

LRESULT CALLBACK App::wnd_proc_static(HWND h, UINT m, WPARAM w, LPARAM l) {
    App* self = nullptr;
    if (m == WM_NCCREATE) {
        auto cs = (CREATESTRUCTW*)l;
        self = (App*)cs->lpCreateParams;
        self->hwnd_ = h;
        SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)self);
    } else {
        self = (App*)GetWindowLongPtrW(h, GWLP_USERDATA);
    }
    return self ? self->wnd_proc(h, m, w, l) : DefWindowProcW(h, m, w, l);
}

void App::create_controls(HWND hwnd) {
    dpi_ = (int)GetDpiForWindow(hwnd);

    auto mkfont = [&](int pt, bool bold) {
        return CreateFontW(-MulDiv(pt, dpi_, 72), 0, 0, 0,
                           bold ? FW_SEMIBOLD : FW_NORMAL, FALSE, FALSE, FALSE,
                           DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                           CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                           L"Microsoft YaHei UI");
    };
    font_      = mkfont(9, false);
    font_bold_ = mkfont(9, true);
    font_log_  = CreateFontW(-MulDiv(9, dpi_, 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                             DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                             CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");

    ctrl_.resize(32, nullptr);   // 索引 = 控件ID - 101

    // 设备行
    ctrl_[IDC_COMBO_DEVICE - 101] = make_combo(hwnd, font_, IDC_COMBO_DEVICE, 0, 0, 0, 0);
    ctrl_[IDC_STATIC_DEVINFO - 101] = make_label(hwnd, font_, L"正在检测硬件…", 0, 0, 0, 0);
    lbl_dev_  = make_label(hwnd, font_, L"处理设备:", 0, 0, 0, 0);
    lbl_scale_ = make_label(hwnd, font_, L"放大倍数:", 0, 0, 0, 0);
    lbl_fmt_   = make_label(hwnd, font_, L"输出格式:", 0, 0, 0, 0);

    // 输出设置
    ctrl_[IDC_COMBO_SCALE - 101] = make_combo(hwnd, font_, IDC_COMBO_SCALE, 0, 0, 0, 0);
    SendMessageW(ctrl_[IDC_COMBO_SCALE - 101], CB_ADDSTRING, 0, (LPARAM)L"2x (高清)");
    SendMessageW(ctrl_[IDC_COMBO_SCALE - 101], CB_ADDSTRING, 0, (LPARAM)L"3x (高清)");
    SendMessageW(ctrl_[IDC_COMBO_SCALE - 101], CB_ADDSTRING, 0, (LPARAM)L"4x (原生)");
    SendMessageW(ctrl_[IDC_COMBO_SCALE - 101], CB_SETCURSEL,
                 sel_scale_ == 2 ? 0 : sel_scale_ == 3 ? 1 : 2, 0);

    ctrl_[IDC_COMBO_FORMAT - 101] = make_combo(hwnd, font_, IDC_COMBO_FORMAT, 0, 0, 0, 0);
    SendMessageW(ctrl_[IDC_COMBO_FORMAT - 101], CB_ADDSTRING, 0, (LPARAM)L"PNG (无损)");
    SendMessageW(ctrl_[IDC_COMBO_FORMAT - 101], CB_ADDSTRING, 0, (LPARAM)L"JPEG (高质量 95)");
    SendMessageW(ctrl_[IDC_COMBO_FORMAT - 101], CB_SETCURSEL,
                 sel_format_ == 1 ? 1 : 0, 0);

    ctrl_[IDC_EDIT_OUTDIR - 101] = make_edit(hwnd, font_, IDC_EDIT_OUTDIR, 0, 0, 0, 0);
    SetWindowTextW(ctrl_[IDC_EDIT_OUTDIR - 101], outdir_.c_str());
    ctrl_[IDC_BTN_BROWSE - 101]  = make_button(hwnd, font_, L"浏览…", IDC_BTN_BROWSE, 0, 0, 0, 0);
    ctrl_[IDC_BTN_OPENDIR - 101] = make_button(hwnd, font_, L"打开输出目录", IDC_BTN_OPENDIR, 0, 0, 0, 0);

    // 文件列表
    ctrl_[IDC_LIST_FILES - 101] = CreateWindowExW(
        WS_EX_CLIENTEDGE, WC_LISTVIEWW, L"",
        WS_CHILD | WS_VISIBLE | WS_TABSTOP | LVS_REPORT | LVS_SHOWSELALWAYS,
        0, 0, 0, 0, hwnd, (HMENU)(INT_PTR)IDC_LIST_FILES, nullptr, nullptr);
    SendMessageW(ctrl_[IDC_LIST_FILES - 101], WM_SETFONT, (WPARAM)font_, TRUE);
    ListView_SetExtendedListViewStyle(ctrl_[IDC_LIST_FILES - 101],
                                      LVS_EX_FULLROWSELECT | LVS_EX_DOUBLEBUFFER | LVS_EX_GRIDLINES);

    struct Col { const wchar_t* name; int width; };
    Col cols[] = { {L"文件", 430}, {L"分辨率", 120}, {L"状态", 160}, {L"耗时", 90} };
    for (int i = 0; i < 4; ++i) {
        LVCOLUMNW c = {};
        c.mask = LVCF_TEXT | LVCF_WIDTH;
        c.pszText = (LPWSTR)cols[i].name;
        c.cx = scale_dpi(cols[i].width);
        ListView_InsertColumn(ctrl_[IDC_LIST_FILES - 101], i, &c);
    }

    ctrl_[IDC_BTN_ADD - 101]    = make_button(hwnd, font_, L"添加图片", IDC_BTN_ADD, 0, 0, 0, 0);
    ctrl_[IDC_BTN_REMOVE - 101] = make_button(hwnd, font_, L"移除选中", IDC_BTN_REMOVE, 0, 0, 0, 0);
    ctrl_[IDC_BTN_CLEAR - 101]  = make_button(hwnd, font_, L"清空列表", IDC_BTN_CLEAR, 0, 0, 0, 0);

    // 处理行
    ctrl_[IDC_BTN_START - 101]  = make_button(hwnd, font_bold_, L"开始超分  ▶", IDC_BTN_START, 0, 0, 0, 0);
    ctrl_[IDC_BTN_CANCEL - 101] = make_button(hwnd, font_, L"取消", IDC_BTN_CANCEL, 0, 0, 0, 0);
    EnableWindow(ctrl_[IDC_BTN_CANCEL - 101], FALSE);
    ctrl_[IDC_PROGRESS - 101] = make_progress(hwnd, IDC_PROGRESS, 0, 0, 0, 0);
    ctrl_[IDC_STATIC_STATUS - 101] = make_label(hwnd, font_, L"空闲", 0, 0, 0, 0);

    // 日志
    ctrl_[IDC_EDIT_LOG - 101] = make_edit(hwnd, font_log_, IDC_EDIT_LOG, 0, 0, 0, 0,
                                          ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL);

    // 拖放
    DragAcceptFiles(hwnd, TRUE);

    // 日志回调 (任意线程 → UI 线程)
    Logger::instance().set_sink([this](LogLevel, const std::string& msg) {
        PostMessageW(hwnd_, WM_APP_LOG, 0, (LPARAM)(new std::string(msg)));
    });
}

void App::layout(int width, int height) {
    if (ctrl_.empty()) return;
    const int M = scale_dpi(14);            // 外边距
    const int W = width - 2 * M;

    int y = M;
    const int row_h = scale_dpi(24);
    const int lbl_w = scale_dpi(64);
    const int cmb_w = scale_dpi(240);

    // --- 设备行 ---
    MoveWindow(lbl_dev_, M, y + scale_dpi(3), lbl_w, row_h, TRUE);
    MoveWindow(ctrl_[IDC_COMBO_DEVICE - 101], M + lbl_w + scale_dpi(6), y, cmb_w, row_h * 8, TRUE);
    MoveWindow(ctrl_[IDC_STATIC_DEVINFO - 101], M + lbl_w + cmb_w + scale_dpi(18),
               y + scale_dpi(5), W - lbl_w - cmb_w - scale_dpi(24), row_h, TRUE);
    y += row_h + scale_dpi(14);

    // --- 放大倍数 / 输出格式行 ---
    MoveWindow(lbl_scale_, M, y + scale_dpi(3), lbl_w, row_h, TRUE);
    MoveWindow(ctrl_[IDC_COMBO_SCALE - 101], M + lbl_w + scale_dpi(6), y, scale_dpi(130), row_h * 8, TRUE);
    MoveWindow(lbl_fmt_, M + lbl_w + scale_dpi(150), y + scale_dpi(3), lbl_w, row_h, TRUE);
    MoveWindow(ctrl_[IDC_COMBO_FORMAT - 101], M + lbl_w * 2 + scale_dpi(156), y,
               scale_dpi(150), row_h * 8, TRUE);
    y += row_h + scale_dpi(10);

    // --- 输出目录行 ---
    const int btn_w = scale_dpi(100);
    MoveWindow(ctrl_[IDC_EDIT_OUTDIR - 101], M, y,
               W - 2 * btn_w - 2 * scale_dpi(8), row_h, TRUE);
    MoveWindow(ctrl_[IDC_BTN_BROWSE - 101], width - M - 2 * btn_w - scale_dpi(8),
               y - scale_dpi(1), btn_w, row_h + scale_dpi(2), TRUE);
    MoveWindow(ctrl_[IDC_BTN_OPENDIR - 101], width - M - btn_w,
               y - scale_dpi(1), btn_w, row_h + scale_dpi(2), TRUE);
    y += row_h + scale_dpi(12);

    // --- 文件列表区 ---
    const int btn_h = scale_dpi(26);
    const int list_btn_w = scale_dpi(88);
    int list_btn_x = width - M - list_btn_w;
    MoveWindow(ctrl_[IDC_BTN_ADD - 101], list_btn_x, y, list_btn_w, btn_h, TRUE);
    list_btn_x -= list_btn_w + scale_dpi(6);
    MoveWindow(ctrl_[IDC_BTN_REMOVE - 101], list_btn_x, y, list_btn_w, btn_h, TRUE);
    list_btn_x -= list_btn_w + scale_dpi(6);
    MoveWindow(ctrl_[IDC_BTN_CLEAR - 101], list_btn_x, y, list_btn_w, btn_h, TRUE);
    y += btn_h + scale_dpi(6);

    int list_h = height - y - (btn_h + scale_dpi(16)) - scale_dpi(130) - M;
    list_h = std::max(list_h, scale_dpi(120));
    MoveWindow(ctrl_[IDC_LIST_FILES - 101], M, y, W, list_h, TRUE);
    list_update_column_widths();
    y += list_h + scale_dpi(10);

    // --- 处理行 ---
    const int start_w = scale_dpi(150), cancel_w = scale_dpi(80);
    MoveWindow(ctrl_[IDC_BTN_START - 101], M, y, start_w, btn_h + scale_dpi(4), TRUE);
    MoveWindow(ctrl_[IDC_BTN_CANCEL - 101], M + start_w + scale_dpi(8), y,
               cancel_w, btn_h + scale_dpi(4), TRUE);
    int pb_x = M + start_w + cancel_w + 2 * scale_dpi(8);
    int pb_w = width - M - pb_x - scale_dpi(240);
    MoveWindow(ctrl_[IDC_PROGRESS - 101], pb_x, y + scale_dpi(6),
               std::max(pb_w, scale_dpi(120)), scale_dpi(15), TRUE);
    MoveWindow(ctrl_[IDC_STATIC_STATUS - 101], width - M - scale_dpi(230), y + scale_dpi(5),
               scale_dpi(230), row_h, TRUE);
    y += btn_h + scale_dpi(4) + scale_dpi(10);

    // --- 日志 ---
    int log_h = height - y - M;
    log_h = std::max(log_h, scale_dpi(60));
    MoveWindow(ctrl_[IDC_EDIT_LOG - 101], M, y, W, log_h, TRUE);
}

void App::list_update_column_widths() {
    HWND lv = ctrl_[IDC_LIST_FILES - 101];
    RECT rc;
    GetClientRect(lv, &rc);
    int total = rc.right - rc.left;
    int w1 = total - scale_dpi(120 + 160 + 90);
    if (w1 < scale_dpi(160)) w1 = scale_dpi(160);
    ListView_SetColumnWidth(lv, 0, w1);
    ListView_SetColumnWidth(lv, 1, scale_dpi(120));
    ListView_SetColumnWidth(lv, 2, scale_dpi(160));
    ListView_SetColumnWidth(lv, 3, scale_dpi(90));
}

void App::list_refresh_item(int idx) {
    if (idx < 0 || idx >= (int)items_.size()) return;
    HWND lv = ctrl_[IDC_LIST_FILES - 101];
    FileItem& it = items_[idx];

    LVITEMW li = {};
    li.mask = LVIF_TEXT;
    li.iItem = idx;
    std::wstring name = path_filename(it.path);
    li.pszText = (LPWSTR)name.c_str();
    if (idx >= ListView_GetItemCount(lv)) ListView_InsertItem(lv, &li);
    else ListView_SetItem(lv, &li);

    ListView_SetItemText(lv, idx, 1, (LPWSTR)it.size_txt.c_str());
    ListView_SetItemText(lv, idx, 2, (LPWSTR)it.status.c_str());
    ListView_SetItemText(lv, idx, 3, (LPWSTR)it.elapsed.c_str());
}

void App::mark_item(int idx, FileState st, const std::wstring& status,
                    const std::wstring& size_txt, const std::wstring& elapsed) {
    if (idx < 0 || idx >= (int)items_.size()) return;
    FileItem& it = items_[idx];
    it.state = st;
    if (!status.empty()) it.status = status;
    if (!size_txt.empty()) it.size_txt = size_txt;
    if (!elapsed.empty()) it.elapsed = elapsed;
    if (st == FileState::Pending) {
        it.status = L"等待中";
        it.elapsed.clear();
    }
    list_refresh_item(idx);
}

// ---------------------------------------------------------------------------
// 事件: 文件管理
// ---------------------------------------------------------------------------
void App::on_add_files() {
    // 采用 Vista+ 的 IFileOpenDialog (纯 COM 对话框)。
    // 旧式 GetOpenFileNameW 在加载了 OpenVINO/ORT 等大量 DLL 的进程中,
    // 其内部 shell 兼容层可能非确定性卡死 (loader-lock 类问题), 故弃用。
    IFileOpenDialog* dlg = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_IFileOpenDialog, (void**)&dlg);
    if (FAILED(hr) || !dlg) {
        Logger::instance().error(str_format("创建文件对话框失败: 0x%08lX", (unsigned long)hr));
        return;
    }

    DWORD opts = 0;
    dlg->GetOptions(&opts);
    dlg->SetOptions(opts | FOS_ALLOWMULTISELECT | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST | FOS_FORCEFILESYSTEM);
    dlg->SetTitle(L"添加要放大的图片");
    static const COMDLG_FILTERSPEC kFilters[] = {
        { L"图片文件", L"*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp" },
        { L"所有文件", L"*.*" },
        { nullptr, nullptr },
    };
    dlg->SetFileTypes(2, kFilters);
    dlg->SetFileTypeIndex(1);

    hr = dlg->Show(hwnd_);
    if (FAILED(hr)) {
        if (hr != HRESULT_FROM_WIN32(ERROR_CANCELLED))
            Logger::instance().error(str_format("文件对话框返回: 0x%08lX", (unsigned long)hr));
        dlg->Release();
        return;
    }

    std::vector<std::wstring> added;
    IShellItemArray* arr = nullptr;
    if (SUCCEEDED(dlg->GetResults(&arr)) && arr) {
        DWORD n = 0;
        arr->GetCount(&n);
        for (DWORD i = 0; i < n; ++i) {
            IShellItem* it = nullptr;
            if (SUCCEEDED(arr->GetItemAt(i, &it)) && it) {
                PWSTR p = nullptr;
                if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p)) && p) {
                    added.push_back(p);
                    CoTaskMemFree(p);
                }
                it->Release();
            }
        }
        arr->Release();
    }
    dlg->Release();
    for (auto& f : added) {
        FileItem it;
        it.path = f;
        items_.push_back(it);
        list_refresh_item((int)items_.size() - 1);
    }
    wchar_t st[64];
    swprintf(st, 64, L"共 %d 张图片", (int)items_.size());
    set_status(st);
}

void App::on_remove_selected() {
    int sel = ListView_GetNextItem(ctrl_[IDC_LIST_FILES - 101], -1, LVNI_SELECTED);
    if (sel < 0) return;
    ListView_DeleteItem(ctrl_[IDC_LIST_FILES - 101], sel);
    items_.erase(items_.begin() + sel);
}

void App::on_clear_files() {
    if (running_) return;
    ListView_DeleteAllItems(ctrl_[IDC_LIST_FILES - 101]);
    items_.clear();
    set_status(L"空闲");
}

void App::on_drop(void* hdrop) {
    HDROP hd = (HDROP)hdrop;
    if (running_) { DragFinish(hd); return; }
    UINT n = DragQueryFileW(hd, 0xFFFFFFFF, nullptr, 0);
    for (UINT i = 0; i < n; ++i) {
        wchar_t path[MAX_PATH * 2] = {};
        if (DragQueryFileW(hd, i, path, MAX_PATH * 2)) {
            DWORD attr = GetFileAttributesW(path);
            if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) continue;
            FileItem it;
            it.path = path;
            items_.push_back(it);
            list_refresh_item((int)items_.size() - 1);
        }
    }
    DragFinish(hd);
    wchar_t st[64];
    swprintf(st, 64, L"共 %d 张图片", (int)items_.size());
    set_status(st);
}

// ---------------------------------------------------------------------------
// 事件: 输出目录
// ---------------------------------------------------------------------------
void App::on_browse_outdir() {
    wchar_t dir[MAX_PATH] = {};
    wcsncpy_s(dir, outdir_.c_str(), MAX_PATH - 1);
    BROWSEINFOW bi = {};
    bi.hwndOwner = hwnd_;
    bi.lpszTitle = L"选择输出目录";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    bi.lpfn = [](HWND h, UINT msg, LPARAM, LPARAM data) -> int {
        if (msg == BFFM_INITIALIZED)
            SendMessageW(h, BFFM_SETSELECTIONW, TRUE, data);
        return 0;
    };
    bi.lParam = (LPARAM)dir;
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (!pidl) return;
    wchar_t path[MAX_PATH] = {};
    if (SHGetPathFromIDListW(pidl, path)) {
        outdir_ = path;
        SetWindowTextW(ctrl_[IDC_EDIT_OUTDIR - 101], path);
    }
    CoTaskMemFree(pidl);
}

void App::on_open_outdir() {
    wchar_t buf[MAX_PATH] = {};
    GetWindowTextW(ctrl_[IDC_EDIT_OUTDIR - 101], buf, MAX_PATH);
    std::wstring dir = buf;
    if (dir.empty()) return;
    if (!file_exists(dir)) CreateDirectoryW(dir.c_str(), nullptr);
    ShellExecuteW(hwnd_, L"open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// ---------------------------------------------------------------------------
// 事件: 开始 / 取消
// ---------------------------------------------------------------------------
void App::on_start() {
    if (running_) return;
    if (items_.empty()) {
        MessageBoxW(hwnd_, L"请先添加图片 (支持直接拖放图片到列表)", L"提示", MB_ICONINFORMATION);
        return;
    }

    int idx = (int)SendMessageW(ctrl_[IDC_COMBO_DEVICE - 101], CB_GETCURSEL, 0, 0);
    sel_device_ = idx < 0 ? 0 : idx;
    int sidx = (int)SendMessageW(ctrl_[IDC_COMBO_SCALE - 101], CB_GETCURSEL, 0, 0);
    sel_scale_ = sidx == 0 ? 2 : sidx == 1 ? 3 : 4;
    sel_format_ = (int)SendMessageW(ctrl_[IDC_COMBO_FORMAT - 101], CB_GETCURSEL, 0, 0);
    if (sel_format_ != 0 && sel_format_ != 1) sel_format_ = 0;

    wchar_t buf[MAX_PATH] = {};
    GetWindowTextW(ctrl_[IDC_EDIT_OUTDIR - 101], buf, MAX_PATH);
    outdir_ = buf;
    if (outdir_.empty()) {
        MessageBoxW(hwnd_, L"请设置输出目录", L"提示", MB_ICONINFORMATION);
        return;
    }
    CreateDirectoryW(outdir_.c_str(), nullptr);
    save_config();

    std::vector<std::wstring> files;
    for (size_t i = 0; i < items_.size(); ++i) {
        if (items_[i].state != FileState::Done) {
            files.push_back(items_[i].path);
            mark_item((int)i, FileState::Pending, L"", L"", L"");
        }
    }
    if (files.empty()) {
        MessageBoxW(hwnd_, L"所有图片都已处理完成", L"提示", MB_ICONINFORMATION);
        return;
    }

    running_ = true;
    cancel_flag_ = false;
    set_controls_running(true);
    // worker 线程结束时仅置 running_=false, thread 对象仍处于 joinable 状态;
    // 若不 join 直接对 work_th_ 赋新线程, std::thread 移动赋值会调用
    // std::terminate() -> abort (0xc0000409), 表现为第二次点"开始超分"瞬间闪退
    if (work_th_.joinable()) work_th_.join();
    work_th_ = std::thread([this, files] {
        try {
            worker_thread(std::move(files));
        } catch (const std::exception& e) {
            // 兜底: 任何异常不能击穿线程导致整个进程闪退
            Logger::instance().error(std::string("内部错误: ") + e.what());
            running_ = false;
            PostMessageW(hwnd_, WM_APP_PROGRESS, (WPARAM)-3, 0);
            PostMessageW(hwnd_, WM_APP_STATUS, 0,
                         (LPARAM)new std::wstring(utf8_to_wide(std::string("发生内部错误: ") + e.what())));
        } catch (...) {
            Logger::instance().error("未知内部错误");
            running_ = false;
            PostMessageW(hwnd_, WM_APP_PROGRESS, (WPARAM)-3, 0);
            PostMessageW(hwnd_, WM_APP_STATUS, 0,
                         (LPARAM)new std::wstring(L"发生未知内部错误"));
        }
    });
}

void App::on_cancel() {
    if (running_) {
        cancel_flag_ = true;
        set_status(L"正在取消…");
        Logger::instance().info("用户请求取消处理");
    }
}

void App::set_controls_running(bool running) {
    EnableWindow(ctrl_[IDC_BTN_START - 101], !running);
    EnableWindow(ctrl_[IDC_BTN_CANCEL - 101], running);
    EnableWindow(ctrl_[IDC_BTN_ADD - 101], !running);
    EnableWindow(ctrl_[IDC_BTN_REMOVE - 101], !running);
    EnableWindow(ctrl_[IDC_BTN_CLEAR - 101], !running);
    EnableWindow(ctrl_[IDC_COMBO_DEVICE - 101], !running);
    EnableWindow(ctrl_[IDC_COMBO_SCALE - 101], !running);
    EnableWindow(ctrl_[IDC_COMBO_FORMAT - 101], !running);
    EnableWindow(ctrl_[IDC_EDIT_OUTDIR - 101], !running);
    EnableWindow(ctrl_[IDC_BTN_BROWSE - 101], !running);
    SendMessageW(ctrl_[IDC_BTN_START - 101], WM_SETTEXT, 0,
                 (LPARAM)(running ? L"处理中…" : L"开始超分  ▶"));
}

// ---------------------------------------------------------------------------
// 设备检测线程
// ---------------------------------------------------------------------------
void App::detect_thread() {
    std::string err;
    std::wstring runtime = exe_dir() + L"\\runtime";

    if (!ov_.init(runtime, err)) {
        Logger::instance().error("OpenVINO 初始化失败: " + err);
    }
    if (!ort_.init(runtime, err)) {
        Logger::instance().error("ONNX Runtime 初始化失败: " + err);
    }
    detector_.detect(ov_, ort_);
    PostMessageW(hwnd_, WM_APP_DETECT, 0, 0);
}

void App::on_detect_done() {
    HWND cb = ctrl_[IDC_COMBO_DEVICE - 101];
    SendMessageW(cb, CB_RESETCONTENT, 0, 0);
    SendMessageW(cb, CB_ADDSTRING, 0, (LPARAM)L"自动选择 (推荐)");
    for (auto& d : detector_.devices()) {
        std::wstring item = utf8_to_wide(d.name);
        SendMessageW(cb, CB_ADDSTRING, 0, (LPARAM)item.c_str());
    }
    int n = (int)detector_.devices().size() + 1;
    if (sel_device_ < n) SendMessageW(cb, CB_SETCURSEL, sel_device_, 0);
    else SendMessageW(cb, CB_SETCURSEL, 0, 0);

    const SystemSpecs& sp = detector_.specs();
    std::string info = str_format("%s | 内存 %s",
                                  sp.cpu_name.c_str(), human_size(sp.total_ram).c_str());
    SetWindowTextW(ctrl_[IDC_STATIC_DEVINFO - 101], utf8_to_wide(info).c_str());

    if (detector_.empty()) {
        log_append("警告: 未检测到任何推理设备, 处理将会失败");
    } else {
        const DeviceInfo* best = detector_.optimal();
        log_append(str_format("推荐设备: %s (%s)", best->name.c_str(), best->full_name.c_str()));
    }
    log_append(str_format("建议 tile 尺寸: %d", sp.max_tile_size));
}

// ---------------------------------------------------------------------------
// 模型管理 (worker 线程内调用)
// ---------------------------------------------------------------------------
static const DeviceInfo* resolve_device(const DeviceDetector& det, int sel, std::string& err) {
    if (det.empty()) { err = "未检测到可用推理设备"; return nullptr; }
    if (sel == 0) return det.optimal();
    return det.at((size_t)(sel - 1));
}

bool App::ensure_model(const std::string& device_key, std::string& err) {
    if (cache_.valid && cache_.device_key == device_key) return true;

    if (cache_.use_ort) ort_.free_session(cache_.ort_session);
    else ov_.free_model(cache_.ov_model);
    cache_ = ModelCache{};

    std::wstring models_dir = exe_dir() + L"\\models";
    post_status(L"正在加载模型…");

    if (device_key.rfind("dml:", 0) == 0) {
        int adapter = atoi(device_key.c_str() + 4);
        std::wstring onnx = models_dir + L"\\RealESRGAN_x4plus.onnx";
        if (!file_exists(onnx)) {
            err = "缺少模型文件 RealESRGAN_x4plus.onnx (models 目录)";
            return false;
        }
        Logger::instance().info(str_format("加载 DirectML 模型 (NVIDIA GPU, 适配器 %d)…", adapter));
        if (!ort_.create_session(onnx, true, adapter, 256, cache_.ort_session, err)) {
            err = "创建 DirectML 会话失败: " + err;
            return false;
        }
        cache_.use_ort = true;
        Logger::instance().info(str_format("模型加载完成 (tile=%d)", cache_.ort_session.tile_in));
    } else {
        std::string dev = device_key.substr(3);
        std::wstring xml;
        std::vector<std::pair<std::string, std::string>> props;
        if (dev == "NPU") {
            xml = models_dir + L"\\RealESRGAN_x4plus_npu_128.xml";
            props = {{"PERFORMANCE_HINT", "LATENCY"}};
        } else if (dev == "GPU") {
            xml = models_dir + L"\\RealESRGAN_x4plus.xml";
            props = {{"PERFORMANCE_HINT", "THROUGHPUT"},
                     {"INFERENCE_PRECISION_HINT", "f16"}};
        } else {
            xml = models_dir + L"\\RealESRGAN_x4plus.xml";
            props = {{"PERFORMANCE_HINT", "THROUGHPUT"}};
        }
        if (!file_exists(xml)) {
            err = "缺少模型文件 " + wide_to_utf8(path_filename(xml)) + " (models 目录)";
            return false;
        }
        Logger::instance().info(str_format("编译 OpenVINO 模型 (%s)…", dev.c_str()));
        post_status(utf8_to_wide(str_format("正在编译模型 (%s)…", dev.c_str())));
        int req_tile = dev == "NPU" ? 128 : 256;
        if (!ov_.load_model(xml, dev, props, req_tile, cache_.ov_model, err)) {
            err = "模型编译失败: " + err;
            return false;
        }
        cache_.use_ort = false;
        Logger::instance().info(str_format("模型编译完成 (tile=%d)",
                                           cache_.ov_model.tile_in));
    }
    cache_.device_key = device_key;
    cache_.valid = true;
    return true;
}

bool App::build_tile_engine(TileEngine& te, std::string& err) {
    te = {};
    if (cache_.use_ort) {
        ORTEngine::Session& s = cache_.ort_session;
        te.tile = s.tile_in;
        te.overlap = 32;
        te.npu = false;
        te.slots = 1;
        ORTEngine* ort = &ort_;
        ORTEngine::Session* sp = &s;
        te.infer = [ort, sp](const float* in, float* out, std::string& e2) {
            return ort->run(*sp, in, out, e2);
        };
    } else {
        OVEngine::Model& m = cache_.ov_model;
        te.tile = m.tile_in;
        te.overlap = m.device == "NPU" ? 64 : 32;
        te.npu = m.device == "NPU";
        te.slots = te.npu ? 1 : 2;
        if (m.device == "CPU") {
            // CPU 设备使用同步推理 (动态输出张量场景更稳妥)
            te.slots = 1;
        }
        OVEngine* ov = &ov_;
        OVEngine::Model* mp = &m;
        if (te.slots == 1) {
            te.infer = [ov, mp](const float* in, float* out, std::string& e2) {
                return ov->infer_sync(mp->reqs[0], in, out, e2);
            };
        } else {
            te.astart = [ov, mp](const float* in, int slot, std::string& e2) {
                return ov->async_start(mp->reqs[(size_t)slot], in, e2);
            };
            te.await = [ov, mp](float* out, int slot, std::string& e2) {
                OVEngine::Request& rq = mp->reqs[(size_t)slot];
                if (!ov->async_wait(rq, e2)) return false;
                return ov->read_output(rq, out, e2);
            };
            te.cancel = [ov, mp] { ov->cancel(*mp); };
        }
    }
    if (!te.infer && !te.astart) { err = "推理引擎未就绪"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// 输出路径
// ---------------------------------------------------------------------------
std::wstring App::out_path_for(const std::wstring& src) const {
    const wchar_t* ext = sel_format_ == 1 ? L".jpg" : L".png";
    wchar_t tag[16];
    swprintf(tag, 16, L"_x%d", sel_scale_);
    std::wstring name = path_stem(src) + tag + ext;
    return unique_path(processing_outdir_ + L"\\" + name);
}

// ---------------------------------------------------------------------------
// worker 线程
// ---------------------------------------------------------------------------
void App::worker_thread(std::vector<std::wstring> files) {
    const int total_files = (int)files.size();
    int ok_count = 0, fail_count = 0;

    Logger::instance().info(str_format("========== 开始处理 %d 张图片 ==========", total_files));
    post_status(utf8_to_wide(str_format("正在处理 %d 张图片…", total_files)));

    std::string err;
    const DeviceInfo* dev = resolve_device(detector_, sel_device_, err);
    if (!dev) {
        Logger::instance().error(err);
        MessageBoxW(hwnd_, utf8_to_wide(err).c_str(), L"无法开始处理", MB_ICONERROR);
        post_status(L"设备不可用");
        running_ = false;
        PostMessageW(hwnd_, WM_APP_PROGRESS, (WPARAM)-3, 0);
        return;
    }
    Logger::instance().info(str_format("使用设备: %s (%s, %s)",
                                       dev->name.c_str(), dev->full_name.c_str(),
                                       dev->backend.c_str()));
    std::string device_key = dev->kind == DevKind::DirectML
                                 ? "dml:" + dev->device_id
                                 : "ov:" + dev->device_id;

    processing_outdir_ = outdir_;

    for (int fi = 0; fi < total_files; ++fi) {
        if (cancel_flag_) break;
        const std::wstring& src = files[fi];

        int item_idx = -1;
        for (size_t i = 0; i < items_.size(); ++i)
            if (items_[i].path == src && items_[i].state != FileState::Done) {
                item_idx = (int)i;
                break;
            }

        {
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Working, L"处理中…", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
        }

        auto t0 = std::chrono::steady_clock::now();
        Image img;
        std::string ferr;

        if (!load_image(src, img, ferr)) {
            Logger::instance().error("读取失败 [" + wide_to_utf8(path_filename(src)) + "]: " + ferr);
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Failed, L"读取失败", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
            ++fail_count;
            continue;
        }

        if (cancel_flag_) break;

        Logger::instance().info(str_format("[%d/%d] %s (%dx%d)", fi + 1, total_files,
                                           wide_to_utf8(path_filename(src)).c_str(),
                                           img.w, img.h));
        {
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Working, L"处理中",
                                      utf8_to_wide(str_format("%dx%d", img.w, img.h)), L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
        }

        if (!ensure_model(device_key, err)) {
            Logger::instance().error(err);
            MessageBoxW(hwnd_, utf8_to_wide(err).c_str(), L"模型加载失败", MB_ICONERROR);
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Failed, L"模型加载失败", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
            ++fail_count;
            break;
        }

        TileEngine te;
        if (!build_tile_engine(te, err)) {
            Logger::instance().error(err);
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Failed, L"引擎错误", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
            ++fail_count;
            break;
        }

        int tile_total = 1;
        TileProgressFn progress = [this, fi, total_files, &tile_total](int d, int t) {
            tile_total = std::max(t, 1);
            int overall = (int)(((double)fi + (double)d / tile_total) * 1000.0 / total_files);
            PostMessageW(hwnd_, WM_APP_PROGRESS, (WPARAM)overall, (LPARAM)1000);
        };

        Image result;
        bool ok = Upscaler::upscale(img, te, sel_scale_, result, progress,
                                    cancel_flag_, ferr);
        img.clear();

        if (!ok) {
            if (ferr == "__CANCELLED__") {
                ItemUpd* u = new ItemUpd{ item_idx, FileState::Cancelled, L"已取消", L"", L"" };
                PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
                break;
            }
            Logger::instance().error("超分失败 [" + wide_to_utf8(path_filename(src)) + "]: " + ferr);
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Failed, L"超分失败", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
            ++fail_count;
            continue;
        }

        if (cancel_flag_) break;

        std::wstring outp = out_path_for(src);
        std::wstring fmt = sel_format_ == 1 ? L"jpg" : L"png";
        if (!save_image(outp, result, fmt, 95, ferr)) {
            Logger::instance().error("保存失败: " + ferr);
            ItemUpd* u = new ItemUpd{ item_idx, FileState::Failed, L"保存失败", L"", L"" };
            PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
            ++fail_count;
            continue;
        }

        double sec = seconds_since(t0);
        Logger::instance().info(str_format("完成: %s -> %s (%dx%d, %.1fs)",
                                           wide_to_utf8(path_filename(src)).c_str(),
                                           wide_to_utf8(path_filename(outp)).c_str(),
                                           result.w, result.h, sec));
        wchar_t el[32];
        swprintf(el, 32, L"%.1fs", sec);
        ItemUpd* u = new ItemUpd{ item_idx, FileState::Done,
                                  utf8_to_wide(str_format("完成")),
                                  utf8_to_wide(str_format("%dx%d", result.w, result.h)),
                                  el };
        PostMessageW(hwnd_, WM_APP_ITEM, 0, (LPARAM)u);
        ++ok_count;
    }

    if (cancel_flag_) {
        Logger::instance().info(str_format("处理已取消 (完成 %d, 失败 %d)", ok_count, fail_count));
        post_status(utf8_to_wide(str_format("已取消 (完成 %d, 失败 %d)", ok_count, fail_count)));
    } else if (fail_count > 0) {
        Logger::instance().info(str_format("========== 结束: 成功 %d, 失败 %d ==========",
                                           ok_count, fail_count));
        post_status(utf8_to_wide(str_format("完成: 成功 %d, 失败 %d", ok_count, fail_count)));
    } else {
        Logger::instance().info(str_format("========== 全部完成: 成功 %d 张 ==========", ok_count));
        post_status(utf8_to_wide(str_format("完成: 成功 %d 张", ok_count)));
    }
    running_ = false;
    PostMessageW(hwnd_, WM_APP_PROGRESS, (WPARAM)-3, 0);
}

// ---------------------------------------------------------------------------
// 主窗口过程
// ---------------------------------------------------------------------------
LRESULT App::wnd_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_CREATE:
        create_controls(h);
        return 0;

    case WM_SIZE: {
        if (w == SIZE_MINIMIZED) break;
        layout(LOWORD(l), HIWORD(l));
        return 0;
    }

    case WM_DPICHANGED: {
        dpi_ = HIWORD(w);
        RECT* r = (RECT*)l;
        SetWindowPos(h, nullptr, r->left, r->top,
                     r->right - r->left, r->bottom - r->top, SWP_NOZORDER);
        return 0;
    }

    case WM_DROPFILES:
        on_drop((HDROP)w);
        return 0;

    case WM_GETMINMAXINFO: {
        auto mm = (MINMAXINFO*)l;
        mm->ptMinTrackSize.x = scale_dpi(860);
        mm->ptMinTrackSize.y = scale_dpi(600);
        return 0;
    }

    case WM_COMMAND: {
        int id = LOWORD(w);
        int code = HIWORD(w);
        if (code == BN_CLICKED)
            Logger::instance().info(str_format("按钮点击 id=%d", id));
        switch (id) {
        case IDC_BTN_ADD:     if (code == BN_CLICKED) on_add_files(); return 0;
        case IDC_BTN_REMOVE:  if (code == BN_CLICKED) on_remove_selected(); return 0;
        case IDC_BTN_CLEAR:   if (code == BN_CLICKED) on_clear_files(); return 0;
        case IDC_BTN_BROWSE:  if (code == BN_CLICKED) on_browse_outdir(); return 0;
        case IDC_BTN_OPENDIR: if (code == BN_CLICKED) on_open_outdir(); return 0;
        case IDC_BTN_START:   if (code == BN_CLICKED) on_start(); return 0;
        case IDC_BTN_CANCEL:  if (code == BN_CLICKED) on_cancel(); return 0;
        }
        break;
    }

    case WM_APP_LOG: {
        std::string* p = (std::string*)l;
        log_append(*p);
        delete p;
        return 0;
    }

    case WM_APP_ITEM: {
        ItemUpd* u = (ItemUpd*)l;
        mark_item(u->idx, u->state, u->status, u->size_txt, u->elapsed);
        delete u;
        return 0;
    }

    case WM_APP_STATUS: {
        std::wstring* p = (std::wstring*)l;
        set_status(*p);
        delete p;
        return 0;
    }

    case WM_APP_PROGRESS: {
        int v = (int)w;
        if (v >= 0) {
            SendMessageW(ctrl_[IDC_PROGRESS - 101], PBM_SETPOS, v, 0);
        } else if (v == -3) {
            set_controls_running(false);
            SendMessageW(ctrl_[IDC_PROGRESS - 101], PBM_SETPOS,
                         cancel_flag_ ? 0 : 1000, 0);
        }
        return 0;
    }

    case WM_APP_DETECT:
        on_detect_done();
        return 0;

    case WM_CLOSE:
        if (running_) {
            if (MessageBoxW(h, L"正在处理图片, 确定退出吗?", L"确认退出",
                            MB_YESNO | MB_ICONQUESTION) != IDYES)
                return 0;
            cancel_flag_ = true;
        }
        save_config();
        DestroyWindow(h);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

} // namespace yb
