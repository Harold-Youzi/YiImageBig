// ============================================================================
//  common.h - 通用工具: 日志 / UTF-8 字符串转换 / 并行循环 / 计时
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdint>

namespace yb {

// ---------------------------------------------------------------------------
// UTF-8 <-> UTF-16(windows 宽字符) 转换
// ---------------------------------------------------------------------------
std::wstring utf8_to_wide(const std::string& s);
std::string  wide_to_utf8(const std::wstring& ws);

// ---------------------------------------------------------------------------
// 日志: 同时写入 <exe目录>/logs/YiImageBig.log, 并可挂接 GUI 回调
// ---------------------------------------------------------------------------
enum class LogLevel { Info, Warn, Error };

class Logger {
public:
    using Sink = std::function<void(LogLevel, const std::string&)>; // UTF-8

    static Logger& instance();

    void set_file(const std::wstring& path);   // UTF-16 路径
    void set_sink(Sink sink);

    void log(LogLevel lv, const std::string& msg);   // UTF-8 文本

    void info(const std::string& msg)  { log(LogLevel::Info,  msg); }
    void warn(const std::string& msg)  { log(LogLevel::Warn,  msg); }
    void error(const std::string& msg) { log(LogLevel::Error, msg); }

private:
    Logger() = default;
    std::mutex   mtx_;
    std::wstring file_path_;
    Sink         sink_;
};

// 便捷格式化 (snprintf 风格)
std::string str_format(const char* fmt, ...);

// 当前时间字符串 "HH:MM:SS"
std::string now_time_str();

// 毫秒级高精度计时
double seconds_since(const std::chrono::steady_clock::time_point& t0);

// ---------------------------------------------------------------------------
// 简易并行 for: 将 [0,n) 分块派发给多个 std::thread
// ---------------------------------------------------------------------------
template <typename Fn>
void parallel_for(int64_t n, Fn&& fn, int max_threads = 0) {
    if (n <= 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    int threads = max_threads > 0 ? max_threads : (int)(hw ? hw : 4);
    threads = (int)std::min<int64_t>(threads, n);
    if (threads <= 1) { fn(0, n); return; }

    int64_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int i = 0; i < threads; ++i) {
        int64_t b = (int64_t)i * chunk;
        int64_t e = std::min<int64_t>(b + chunk, n);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e] { fn(b, e); });
    }
    for (auto& t : pool) t.join();
}

// ---------------------------------------------------------------------------
// 路径工具
// ---------------------------------------------------------------------------
std::wstring exe_dir();                     // UTF-16, 不含末尾反斜杠
std::wstring app_data_dir();                // %APPDATA%\YiImageBig, 自动创建
bool        file_exists(const std::wstring& path);
std::wstring path_filename(const std::wstring& path);   // 去目录
std::wstring path_stem(const std::wstring& path);       // 去目录+扩展名
std::wstring path_extension(const std::wstring& path);  // 含点, 小写

// 输出重名时自动追加 "_2" "_3" ...
std::wstring unique_path(const std::wstring& path);

// 格式化字节数
std::string human_size(uint64_t bytes);

} // namespace yb
