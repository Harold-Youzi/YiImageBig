// GUI 端到端自动化回归: 两次任务, 验证第二次"开始超分"不再闪退
// 全程纯 Win32 (无 UIAutomation), 控件按 GetDlgCtrlID 匹配
// 返回: 0=PASS  2=CRASH(第二次启动闪退)  1=其它失败
#include <windows.h>
#include <cstdio>
#include <string>

static HWND g_main = nullptr;
static FILE* g_out = nullptr;

#define OUT(...) do { fprintf(g_out, __VA_ARGS__); fprintf(g_out, "\n"); fflush(g_out); } while (0)

static bool wait_main_window(DWORD pid, HWND* out) {
    for (int i = 0; i < 100; ++i) {
        HWND h = FindWindowW(L"YiImageBigWnd", nullptr);
        if (h) {
            DWORD wpid = 0;
            GetWindowThreadProcessId(h, &wpid);
            if (wpid == pid) { *out = h; return true; }
        }
        Sleep(200);
    }
    return false;
}

// 按 ID 找直接子控件 (不能用全局变量暂存, 否则会覆盖主窗口句柄)
struct FindCtx { int id; HWND found; };
static BOOL CALLBACK find_child_cb(HWND h, LPARAM lp) {
    FindCtx* c = (FindCtx*)lp;
    if (GetDlgCtrlID(h) == c->id) { c->found = h; return FALSE; }
    return TRUE;
}
static HWND find_child(HWND parent, int id) {
    FindCtx c = { id, nullptr };
    EnumChildWindows(parent, find_child_cb, (LPARAM)&c);
    return c.found;
}

static bool wait_dialog(DWORD pid, HWND* out, int timeout_ms) {
    DWORD t0 = GetTickCount();
    for (;;) {
        if (GetTickCount() - t0 > (DWORD)timeout_ms) return false;
        HWND d = FindWindowW(L"#32770", L"添加要放大的图片");
        if (d) {
            DWORD wpid = 0;
            GetWindowThreadProcessId(d, &wpid);
            if (wpid == pid) { *out = d; return true; }
        }
        Sleep(200);
    }
}

static bool wait_dialog_gone(DWORD pid, int timeout_ms) {
    DWORD t0 = GetTickCount();
    for (;;) {
        if (GetTickCount() - t0 > (DWORD)timeout_ms) return false;
        HWND d = FindWindowW(L"#32770", L"添加要放大的图片");
        if (!d) return true;
        DWORD wpid = 0;
        GetWindowThreadProcessId(d, &wpid);
        if (wpid != pid) return true;
        Sleep(200);
    }
}

static bool wait_start_enabled(HWND btnStart, DWORD pid, int timeout_ms) {
    DWORD t0 = GetTickCount();
    for (;;) {
        DWORD code = 0;
        if (GetExitCodeProcess(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid), &code) && code != STILL_ACTIVE)
            return false;
        if (IsWindow(btnStart) && IsWindowEnabled(btnStart)) return true;
        if (GetTickCount() - t0 > (DWORD)timeout_ms) return false;
        Sleep(400);
    }
}

static BOOL CALLBACK edit_cb(HWND h, LPARAM lp) {
    wchar_t cls[64] = {};
    GetClassNameW(h, cls, 64);
    if (wcscmp(cls, L"Edit") == 0 && IsWindowVisible(h)) {
        *(HWND*)lp = h;
        return FALSE;
    }
    return TRUE;
}

static bool add_file(HWND mainH, DWORD pid, const wchar_t* path) {
    PostMessageW(mainH, WM_COMMAND, MAKEWPARAM(109, 0), 0);   // IDC_BTN_ADD
    HWND dlg = nullptr;
    if (!wait_dialog(pid, &dlg, 12000)) { OUT("FAIL: 对话框未出现"); return false; }
    Sleep(700);
    HWND edit = nullptr;
    EnumChildWindows(dlg, edit_cb, (LPARAM)&edit);
    if (!edit) { OUT("FAIL: 未找到文件名编辑框"); return false; }
    SendMessageW(edit, WM_SETTEXT, 0, (LPARAM)path);
    Sleep(400);
    PostMessageW(dlg, WM_COMMAND, MAKEWPARAM(1, 0), 0);        // IDOK
    if (!wait_dialog_gone(pid, 5000)) {
        // 回退: 真实回车键 (对话框默认按钮)
        SetForegroundWindow(dlg);
        Sleep(300);
        keybd_event(VK_RETURN, 0, 0, 0);
        keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0);
        if (!wait_dialog_gone(pid, 12000)) { OUT("FAIL: 对话框未关闭"); return false; }
    }
    Sleep(800);
    return true;
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 4) { printf("usage: autotest <exe> <imgA> <imgB>\n"); return 1; }
    const wchar_t* exe = argv[1];
    const wchar_t* imgA = argv[2];
    const wchar_t* imgB = argv[3];

    wchar_t logp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, logp);
    wcscat(logp, L"autotest_out.txt");
    g_out = _wfopen(logp, L"w");
    if (!g_out) g_out = stdout;

    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = {};
    std::wstring cmd = std::wstring(L"\"") + exe + L"\"";
    if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        OUT("FAIL: CreateProcess"); fclose(g_out); return 1;
    }
    CloseHandle(pi.hThread);
    HANDLE hProc = pi.hProcess;
    DWORD pid = pi.dwProcessId;

    auto crashed = [&]() -> bool {
        DWORD code = 0;
        GetExitCodeProcess(hProc, &code);
        return code != STILL_ACTIVE;
    };

    int result = 1;
    HWND btnStart = nullptr;
    do {
        if (!wait_main_window(pid, &g_main)) { OUT("FAIL: 主窗口未出现"); break; }
        OUT("[1] 主窗口 OK");

        if (!add_file(g_main, pid, imgA)) break;
        OUT("[2] 已添加图片 A");
        btnStart = find_child(g_main, 112);                       // IDC_BTN_START
        if (!btnStart) { OUT("FAIL: 找不到开始按钮"); break; }

        PostMessageW(g_main, WM_COMMAND, MAKEWPARAM(112, 0), 0);  // 开始
        OUT("[3] 任务1 已开始");
        Sleep(1500);
        if (crashed()) { OUT("FAIL: 任务1 启动即退出"); result = 2; break; }
        if (!wait_start_enabled(btnStart, pid, 240000)) { OUT("FAIL: 任务1 未完成/进程退出"); result = crashed() ? 2 : 1; break; }
        OUT("[4] 任务1 完成, 程序存活");

        PostMessageW(g_main, WM_COMMAND, MAKEWPARAM(111, 0), 0);  // 清空列表
        Sleep(600);
        if (!add_file(g_main, pid, imgB)) break;
        OUT("[5] 已清空列表并添加图片 B");

        PostMessageW(g_main, WM_COMMAND, MAKEWPARAM(112, 0), 0);  // 第二次开始
        OUT("[6] 任务2 已启动 (旧版本此处闪退)");
        Sleep(5000);
        if (crashed()) { OUT("RESULT: CRASH (第二次启动闪退复现)"); result = 2; break; }
        OUT("[7] 任务2 进行中且程序存活");
        if (!wait_start_enabled(btnStart, pid, 240000)) { OUT("FAIL: 任务2 未完成"); result = crashed() ? 2 : 1; break; }
        OUT("[8] 任务2 完成, 程序存活");
        OUT("RESULT: PASS");
        result = 0;
    } while (false);

    if (!crashed()) {
        PostMessageW(g_main, WM_CLOSE, 0, 0);
        Sleep(2000);
        if (!crashed()) TerminateProcess(hProc, 0);
    } else {
        DWORD code = 0;
        GetExitCodeProcess(hProc, &code);
        char b[32];
        sprintf(b, "%08X", (unsigned)code);
        OUT(("进程退出码: 0x" + std::string(b)).c_str());
    }
    WaitForSingleObject(hProc, 5000);
    CloseHandle(hProc);
    if (g_out != stdout) fclose(g_out);
    return result;
}
