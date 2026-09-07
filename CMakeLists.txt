# ============================================================================
#  YiImageBig - 图片超分工具 (RealESRGAN x4)
#  后端: OpenVINO (Intel CPU/GPU/NPU) + ONNX Runtime DirectML (NVIDIA GPU)
#  工具链: MinGW-w64 (UCRT64 GCC) / 兼容 MSVC
# ============================================================================
cmake_minimum_required(VERSION 3.24)
project(YiImageBig LANGUAGES CXX RC VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
    message(FATAL_ERROR "YiImageBig 仅支持 Windows 平台")
endif()

# ---------------------------------------------------------------------------
# 第三方依赖 (headers only, 运行时 DLL 动态加载, 无需链接导入库)
#   third_party/ov/include   -> OpenVINO 2025.4.0 C API 头文件
#   third_party/ort/...      -> ONNX Runtime DirectML 1.24.4 C API 头文件
# ---------------------------------------------------------------------------
set(OV_INCLUDE_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ov/include")
set(ORT_INCLUDE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ort/build/native/include")
set(DML_INCLUDE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dml/include")
set(STB_INCLUDE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")

if(NOT EXISTS "${OV_INCLUDE_DIR}/openvino/c/openvino.h")
    message(FATAL_ERROR "缺少 OpenVINO C API 头文件, 请先运行 scripts/fetch_sdks.ps1")
endif()
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "缺少 ONNX Runtime C API 头文件, 请先运行 scripts/fetch_sdks.ps1")
endif()
if(NOT EXISTS "${STB_INCLUDE_DIR}/stb_image_write.h")
    message(FATAL_ERROR "缺少 stb 头文件, 请先运行 scripts/fetch_sdks.ps1")
endif()

# ---------------------------------------------------------------------------
# 资源 (图标 / 版本信息 / 清单)
# ---------------------------------------------------------------------------
set(APP_RC "${CMAKE_CURRENT_SOURCE_DIR}/src/res/app.rc")
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/res/app.ico")
    set(APP_RC "") # 无图标时跳过资源编译
endif()

# ---------------------------------------------------------------------------
# 可执行程序
# ---------------------------------------------------------------------------
add_executable(YiImageBig WIN32
    src/main.cpp
    src/gui/app.cpp
    src/gui/widgets.cpp
    src/core/common.cpp
    src/core/image.cpp
    src/core/ov_engine.cpp
    src/core/ort_engine.cpp
    src/core/upscaler.cpp
    src/core/devices.cpp
    ${APP_RC}
)

target_include_directories(YiImageBig PRIVATE
    src
    ${OV_INCLUDE_DIR}
    ${ORT_INCLUDE_DIR}
    ${DML_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
)

target_compile_definitions(YiImageBig PRIVATE
    UNICODE _UNICODE
    NOMINMAX WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    APP_VERSION="1.0.0"
)

# MinGW / GCC 专用选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # -municode: wWinMain 入口; UTF-8 源码与执行字符集
    target_compile_options(YiImageBig PRIVATE
        -municode -mwindows
        -finput-charset=UTF-8 -fexec-charset=UTF-8
        -Wall -Wextra -Wno-unused-parameter
    )
    target_link_options(YiImageBig PRIVATE
        -municode -mwindows
        -static-libgcc -static-libstdc++
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(YiImageBig PRIVATE /utf-8 /W3 /MP)
    target_link_options(YiImageBig PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:wWinMainCRTStartup)
endif()

target_link_libraries(YiImageBig PRIVATE
    user32 gdi32 gdiplus comctl32 comdlg32 shell32 shlwapi
    ole32 oleaut32 uuid advapi32 uxtheme dwmapi version dxgi
)

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
set_target_properties(YiImageBig PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
