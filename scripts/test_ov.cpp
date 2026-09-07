// 最小化 OpenVINO C API 变参编译调用测试 (MinGW 兼容性验证)
#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT 1
#endif
#include <openvino/c/openvino.h>
#undef BOOLEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    // 可选参数 1: 运行库目录 (默认: ../YiImageBig(C++)/runtime)
    std::wstring runtime = L"YiImageBig(C++)\\runtime";
    {
        int wargc = 0;
        LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargs && wargc > 1) runtime = wargs[1];
        if (wargs) LocalFree(wargs);
    }
    const wchar_t* xml = L"RealESRGAN\\RealESRGAN_x4plus.xml";
    printf("runtime: ");
    fflush(stdout);
    printf("%ls\n", runtime.c_str());
    fflush(stdout);

    SetDllDirectoryW(runtime.c_str());
    printf("step1: loading dll\n"); fflush(stdout);
    HMODULE m = LoadLibraryW((runtime + L"\\openvino_c.dll").c_str());
    if (!m) { printf("FAIL LoadLibrary\n"); return 1; }
    printf("step2: dll loaded\n"); fflush(stdout);

    auto core_create = (ov_status_e(__cdecl*)(ov_core_t**))GetProcAddress(m, "ov_core_create");
    auto core_free   = (void(__cdecl*)(ov_core_t*))GetProcAddress(m, "ov_core_free");
    auto compile_w   = (ov_status_e(__cdecl*)(const ov_core_t*, const wchar_t*, const char*,
                                              size_t, ov_compiled_model_t**, ...))
                       GetProcAddress(m, "ov_core_compile_model_from_file_unicode");
    auto avail       = (ov_status_e(__cdecl*)(const ov_core_t*, ov_available_devices_t*))
                       GetProcAddress(m, "ov_core_get_available_devices");
    auto cm_input    = (ov_status_e(__cdecl*)(const ov_compiled_model_t*, size_t,
                                              ov_output_const_port_t**))
                       GetProcAddress(m, "ov_compiled_model_input_by_index");
    auto port_shape  = (ov_status_e(__cdecl*)(const ov_output_const_port_t*, ov_shape_t*))
                       GetProcAddress(m, "ov_const_port_get_shape");
    auto shape_free  = (ov_status_e(__cdecl*)(ov_shape_t*))GetProcAddress(m, "ov_shape_free");
    auto port_free   = (ov_status_e(__cdecl*)(ov_output_const_port_t*))
                       GetProcAddress(m, "ov_output_const_port_free");
    auto version     = (ov_status_e(__cdecl*)(ov_version_t*))GetProcAddress(m, "ov_get_openvino_version");
    auto vfree       = (ov_status_e(__cdecl*)(ov_version_t*))GetProcAddress(m, "ov_version_free");

    if (!core_create || !compile_w || !avail) { printf("FAIL GetProcAddress\n"); return 1; }

    ov_version_t ver = {};
    if (version(&ver) == OK) { printf("OpenVINO: %s\n", ver.buildNumber); vfree(&ver); }

    ov_core_t* core = nullptr;
    printf("step3: core_create\n"); fflush(stdout);
    if (core_create(&core) != OK) { printf("FAIL core_create\n"); return 1; }
    printf("step4: core ok\n"); fflush(stdout);

    ov_available_devices_t devs; memset(&devs, 0, sizeof(devs));
    printf("step5: calling avail\n"); fflush(stdout);
    if (avail(core, &devs) == OK) {
        printf("step6: avail ok\n"); fflush(stdout);
        printf("raw: size=%llu devices=%p\n",
               (unsigned long long)devs.size, (void*)devs.devices); fflush(stdout);
        for (size_t i = 0; i < devs.size; ++i) {
            printf("  dev[%llu] = %s\n", (unsigned long long)i,
                   devs.devices[i] ? devs.devices[i] : "(null)"); fflush(stdout);
        }
    } else {
        printf("step6b: avail FAILED\n"); fflush(stdout);
    }

    // --- 无属性编译 ---
    ov_compiled_model_t* cm = nullptr;
    ov_status_e st = compile_w(core, xml, "CPU", 0, &cm);
    printf("compile(no props): %d %s\n", (int)st, st == OK ? "OK" : "FAIL");
    if (st != OK) return 2;

    // 输入 shape
    ov_output_const_port_t* port = nullptr;
    if (cm_input(cm, 0, &port) == OK && port) {
        ov_shape_t shape;
        if (port_shape(port, &shape) == OK && shape.rank >= 4) {
            printf("input shape: [%lld,%lld,%lld,%lld]\n",
                   (long long)shape.dims[0], (long long)shape.dims[1],
                   (long long)shape.dims[2], (long long)shape.dims[3]);
        }
        shape_free(&shape);
        port_free(port);
    }

    // --- 带属性编译 (变参) ---
    ov_compiled_model_t* cm2 = nullptr;
    st = compile_w(core, xml, "CPU", 2, &cm2,
                   "PERFORMANCE_HINT", "THROUGHPUT");
    printf("compile(1 prop): %d %s\n", (int)st, st == OK ? "OK" : "FAIL");

    // --- 2 属性 (相同 key 测试变参计数) ---
    printf("step7: compile 2 props same key\n"); fflush(stdout);
    ov_compiled_model_t* cm3 = nullptr;
    st = compile_w(core, xml, "CPU", 4, &cm3,
                   "PERFORMANCE_HINT", "THROUGHPUT",
                   "LOG_LEVEL", "3");
    printf("compile(2 props A): %d %s\n", (int)st, st == OK ? "OK" : "FAIL");
    fflush(stdout);

    // --- 2 属性 (f16 on CPU, GUI 中仅 GPU 使用该配置) ---
    printf("step8: compile f16 on CPU\n"); fflush(stdout);
    ov_compiled_model_t* cm4 = nullptr;
    st = compile_w(core, xml, "CPU", 4, &cm4,
                   "PERFORMANCE_HINT", "THROUGHPUT",
                   "INFERENCE_PRECISION_HINT", "f16");
    printf("compile(f16 cpu): %d %s\n", (int)st, st == OK ? "OK" : "FAIL");
    fflush(stdout);

    // ================= NPU 完整请求周期 =================
    {
        printf("npu: compile\n"); fflush(stdout);
        const wchar_t* npuXml = L"RealESRGAN\\RealESRGAN_x4plus_npu_128.xml";
        auto create_req = (ov_status_e(__cdecl*)(ov_compiled_model_t*, ov_infer_request_t**))
                          GetProcAddress(m, "ov_compiled_model_create_infer_request");
        auto set_input  = (ov_status_e(__cdecl*)(ov_infer_request_t*, const ov_tensor_t*))
                          GetProcAddress(m, "ov_infer_request_set_input_tensor");
        auto get_output = (ov_status_e(__cdecl*)(ov_infer_request_t*, ov_tensor_t**))
                          GetProcAddress(m, "ov_infer_request_get_output_tensor");
        auto tensor_create = (ov_status_e(__cdecl*)(ov_element_type_e, const ov_shape_t, ov_tensor_t**))
                             GetProcAddress(m, "ov_tensor_create");
        auto tensor_data   = (ov_status_e(__cdecl*)(const ov_tensor_t*, void**))
                             GetProcAddress(m, "ov_tensor_data");
        auto tensor_setsh  = (ov_status_e(__cdecl*)(ov_tensor_t*, const ov_shape_t))
                             GetProcAddress(m, "ov_tensor_set_shape");
        auto tensor_free   = (ov_status_e(__cdecl*)(ov_tensor_t*))
                             GetProcAddress(m, "ov_tensor_free");
        auto infer_fn      = (ov_status_e(__cdecl*)(ov_infer_request_t*))
                             GetProcAddress(m, "ov_infer_request_infer");

        ov_compiled_model_t* nc = nullptr;
        st = compile_w(core, npuXml, "NPU", 2, &nc, "PERFORMANCE_HINT", "LATENCY");
        printf("npu compile: %d %s\n", (int)st, st == OK ? "OK" : "FAIL");
        if (st != OK) return 3;

        ov_infer_request_t* req = nullptr;
        create_req(nc, &req);
        printf("npu req: %p\n", (void*)req); fflush(stdout);

        ov_shape_t ishape = {4, nullptr};
        int64_t id[4] = {1,3,128,128};
        ov_shape_t oshape = {4, nullptr};
        int64_t od[4] = {1,3,512,512};
        ishape.dims = id; oshape.dims = od;

        ov_tensor_t* tin = nullptr;
        tensor_create(F32, ishape, &tin);
        void* ip = nullptr;
        tensor_data(tin, &ip);
        printf("npu input tensor: %p data=%p\n", (void*)tin, ip); fflush(stdout);
        memset(ip, 0x3c, 128*128*3*4);  // ~0.01 浮点

        set_input(req, tin);
        printf("npu set_input OK\n"); fflush(stdout);

        // 读一次输出张量指针 (加载时)
        ov_tensor_t* tout = nullptr;
        get_output(req, &tout);
        printf("npu output tensor: %p\n", (void*)tout); fflush(stdout);
        void* op = nullptr;
        ov_status_e sd = tensor_setsh(tout, oshape);
        printf("npu set_shape(out): %d\n", (int)sd); fflush(stdout);
        sd = tensor_data(tout, &op);
        printf("npu tensor_data(out): %d ptr=%p\n", (int)sd, op); fflush(stdout);

        printf("npu infer...\n"); fflush(stdout);
        st = infer_fn(req);
        printf("npu infer: %d %s\n", (int)st, st == OK ? "OK" : "FAIL"); fflush(stdout);

        ov_tensor_t* tout2 = nullptr;
        get_output(req, &tout2);
        void* op2 = nullptr;
        st = tensor_data(tout2, &op2);
        printf("npu read after infer: %d ptr=%p\n", (int)st, op2); fflush(stdout);
        float sum = 0.f;
        const float* fp2 = (const float*)op2;
        for (int i = 0; i < 100; ++i) sum += fp2[i];
        printf("npu out[0..100] sum=%f\n", sum);
        printf("NPU CYCLE DONE\n"); fflush(stdout);
    }
    printf("ALL DONE\n");
    return 0;
}
