// ============================================================================
//  widgets.cpp - Win32 控件创建小工具
// ============================================================================
#include "app.h"
#include "../core/common.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>

namespace yb {

HWND make_label(HWND parent, HFONT font, const wchar_t* text, int x, int y, int w, int h) {
    HWND c = CreateWindowExW(0, L"STATIC", text,
                             WS_CHILD | WS_VISIBLE | SS_LEFT,
                             x, y, w, h, parent, nullptr, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

HWND make_button(HWND parent, HFONT font, const wchar_t* text, int id,
                 int x, int y, int w, int h) {
    HWND c = CreateWindowExW(0, L"BUTTON", text,
                             WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON,
                             x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

HWND make_combo(HWND parent, HFONT font, int id, int x, int y, int w, int h) {
    HWND c = CreateWindowExW(0, L"COMBOBOX", L"",
                             WS_CHILD | WS_VISIBLE | WS_TABSTOP |
                             CBS_DROPDOWNLIST | WS_VSCROLL,
                             x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

HWND make_edit(HWND parent, HFONT font, int id, int x, int y, int w, int h,
               DWORD extra_style) {
    HWND c = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                             WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL |
                             extra_style,
                             x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

HWND make_groupbox(HWND parent, HFONT font, const wchar_t* text, int x, int y, int w, int h) {
    HWND c = CreateWindowExW(0, L"BUTTON", text,
                             WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                             x, y, w, h, parent, nullptr, nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, (WPARAM)font, TRUE);
    return c;
}

HWND make_progress(HWND parent, int id, int x, int y, int w, int h) {
    HWND c = CreateWindowExW(0, PROGRESS_CLASSW, L"",
                             WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
                             x, y, w, h, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    SendMessageW(c, PBM_SETRANGE, 0, MAKELPARAM(0, 1000));
    return c;
}

} // namespace yb
