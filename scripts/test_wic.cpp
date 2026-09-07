// WIC encoder CLSID test
#include <initguid.h>
#include <wincodecsdk.h>
#include <windows.h>
#include <objidl.h>
#include <ocidl.h>
#include <stdio.h>

// official PNG encoder GUID
static const GUID kPng = {0x27949969,0x0864,0x434F,{0xA4,0x0A,0x43,0x1C,0x9B,0xA4,0x7E,0xFD}};

int main() {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    IWICImagingFactory* f = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                  CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f));
    printf("factory: hr=0x%08lX\n", (unsigned long)hr);
    if (FAILED(hr)) return 1;

    GUID try1 = kPng;
    GUID try2 = CLSID_WICPngEncoder;
    GUID try3 = CLSID_WICJpegEncoder;
    printf("official png: {%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}\n",
        try1.Data1, try1.Data2, try1.Data3, try1.Data4[0], try1.Data4[1],
        try1.Data4[2], try1.Data4[3], try1.Data4[4], try1.Data4[5], try1.Data4[6], try1.Data4[7]);
    printf("mingw png   : {%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}\n",
        try2.Data1, try2.Data2, try2.Data3, try2.Data4[0], try2.Data4[1],
        try2.Data4[2], try2.Data4[3], try2.Data4[4], try2.Data4[5], try2.Data4[6], try2.Data4[7]);

    IWICBitmapEncoder* e = nullptr;
    hr = f->CreateEncoder(try1, nullptr, &e);
    printf("CreateEncoder(official png): hr=0x%08lX %s\n", (unsigned long)hr, SUCCEEDED(hr)?"OK":"FAIL");
    if (SUCCEEDED(hr)) e->Release();
    hr = f->CreateEncoder(try2, nullptr, &e);
    printf("CreateEncoder(mingw png)   : hr=0x%08lX %s\n", (unsigned long)hr, SUCCEEDED(hr)?"OK":"FAIL");
    if (SUCCEEDED(hr)) e->Release();
    hr = f->CreateEncoder(try3, nullptr, &e);
    printf("CreateEncoder(jpeg)        : hr=0x%08lX %s\n", (unsigned long)hr, SUCCEEDED(hr)?"OK":"FAIL");
    if (SUCCEEDED(hr)) e->Release();

    // bypass factory: direct CoCreateInstance
    hr = CoCreateInstance(try1, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&e));
    printf("CoCreate(png encoder)      : hr=0x%08lX %s\n", (unsigned long)hr, SUCCEEDED(hr)?"OK":"FAIL");
    if (SUCCEEDED(hr)) e->Release();
    hr = CoCreateInstance(CLSID_WICImagingFactory1, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f));
    printf("factory1: hr=0x%08lX\n", (unsigned long)hr);
    if (SUCCEEDED(hr)) {
        hr = f->CreateEncoder(try1, nullptr, &e);
        printf("CreateEncoder via factory1 : hr=0x%08lX %s\n", (unsigned long)hr, SUCCEEDED(hr)?"OK":"FAIL");
        if (SUCCEEDED(hr)) e->Release();
    }
    return 0;
}
