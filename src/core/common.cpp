// ============================================================================
//  common.cpp - 通用工具实现
// ============================================================================
#include "common.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <cstdarg>
#include <cstdio>
#include <fstream>

namespace yb {

// ---------------------------------------------------------------------------
// 字符串转换
// ---------------------------------------------------------------------------
std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), out.data(), n);
    return out;
}

std::string wide_to_utf8(const std::wstring& ws) {
    if (ws.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(),
                                nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(), out.data(), n,
                        nullptr, nullptr);
    return out;
}

// ---------------------------------------------------------------------------
// 日志
// ---------------------------------------------------------------------------
Logger& Logger::instance() {
    static Logger g;
    return g;
}

void Logger::set_file(const std::wstring& path) {
    std::lock_guard<std::mutex> lk(mtx_);
    file_path_ = path;
}

void Logger::set_sink(Sink sink) {
    std::lock_guard<std::mutex> lk(mtx_);
    sink_ = std::move(sink);
}

void Logger::log(LogLevel lv, const std::string& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    const char* tag = lv == LogLevel::Info  ? "[INFO ]"
                    : lv == LogLevel::Warn  ? "[WARN ]"
                    :                         "[ERROR]";
    std::string line = str_format("%s %s %s", now_time_str().c_str(), tag, msg.c_str());

    // 写文件 (UTF-8 追加)
    if (!file_path_.empty()) {
        std::ofstream f(file_path_.c_str(), std::ios::binary | std::ios::app);
        if (f) f << line << "\r\n";
    }
    if (sink_) {
        // 粘贴给 GUI 时去掉时间戳前缀 (GUI 自己加)
        sink_(lv, msg);
    }
}

std::string str_format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string out(n > 0 ? n : 0, '\0');
    if (n > 0) vsnprintf(out.data(), (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return out;
}

std::string now_time_str() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buf[32];
    snprintf(buf, sizeof(buf), "%02u:%02u:%02u.%03u",
             st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return buf;
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// 路径工具
// ---------------------------------------------------------------------------
std::wstring exe_dir() {
    wchar_t buf[MAX_PATH + 2] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf);
    size_t pos = p.find_last_of(L"\\/");
    if (pos == std::wstring::npos) return L".";
    return p.substr(0, pos);
}

std::wstring app_data_dir() {
    wchar_t* p = nullptr;
    std::wstring base;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &p))) {
        base = p;
        CoTaskMemFree(p);
    }
    std::wstring dir = base + L"\\YiImageBig";
    CreateDirectoryW(dir.c_str(), nullptr);
    return dir;
}

bool file_exists(const std::wstring& path) {
    DWORD attr = GetFileAttributesW(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

std::wstring path_filename(const std::wstring& path) {
    size_t pos = path.find_last_of(L"\\/:");
    return pos == std::wstring::npos ? path : path.substr(pos + 1);
}

std::wstring path_stem(const std::wstring& path) {
    std::wstring name = path_filename(path);
    size_t pos = name.find_last_of(L'.');
    return (pos == std::wstring::npos || pos == 0) ? name : name.substr(0, pos);
}

std::wstring path_extension(const std::wstring& path) {
    std::wstring name = path_filename(path);
    size_t pos = name.find_last_of(L'.');
    if (pos == std::wstring::npos || pos == 0) return L"";
    std::wstring ext = name.substr(pos);
    for (auto& c : ext) c = (wchar_t)towlower(c);
    return ext;
}

std::wstring unique_path(const std::wstring& path) {
    if (!file_exists(path)) return path;
    std::wstring ext  = path_extension(path);
    std::wstring stem = path.substr(0, path.size() - ext.size());
    for (int i = 2; i < 10000; ++i) {
        wchar_t tag[16];
        swprintf(tag, 16, L"_%d", i);
        std::wstring cand = stem + tag + ext;
        if (!file_exists(cand)) return cand;
    }
    return path;
}

std::string human_size(uint64_t bytes) {
    char buf[64];
    if (bytes >= (1ull << 30))
        snprintf(buf, sizeof(buf), "%.1f GB", bytes / 1073741824.0);
    else if (bytes >= (1ull << 20))
        snprintf(buf, sizeof(buf), "%.1f MB", bytes / 1048576.0);
    else if (bytes >= 1024)
        snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    return buf;
}

} // namespace yb
