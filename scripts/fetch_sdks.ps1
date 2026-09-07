# ============================================================================
#  fetch_sdks.ps1 - Download third-party runtime DLLs / headers (NuGet official)
#  OpenVINO.runtime.win 2025.4.0  +  ONNX Runtime DirectML 1.24.4
#  + Microsoft.AI.DirectML 1.15.4  +  OpenVINO C API headers (GitHub)
# ============================================================================
$ErrorActionPreference = "Stop"
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path $MyInvocation.MyCommand.Path -Parent }
$tp = Join-Path $scriptDir "..\third_party"
New-Item -ItemType Directory -Force $tp | Out-Null
Set-Location $tp

function Fetch-NuGet($id, $version) {
    $pkgId = $id.ToLower()
    $pkg = "$pkgId.$version.nupkg"
    $dir = $pkgId
    if (-not (Test-Path $pkg)) {
        Write-Host "Downloading $pkg ..."
        curl.exe -s -L -o $pkg "https://api.nuget.org/v3-flatcontainer/$pkgId/$version/$pkgId.$version.nupkg"
    } else {
        Write-Host "Already exists: $pkg"
    }
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force $dir | Out-Null
        tar.exe -xf $pkg -C $dir
        Write-Host "Extracted $pkg -> $dir"
    }
}

Fetch-NuGet "OpenVINO.runtime.win"              "2025.4.0"
Fetch-NuGet "Microsoft.ML.OnnxRuntime.DirectML" "1.24.4"
Fetch-NuGet "Microsoft.AI.DirectML"             "1.15.4"

# rename extraction dirs to match code references
if (Test-Path "openvino.runtime.win") { Move-Item "openvino.runtime.win" "ov" }
if (Test-Path "microsoft.ml.onnxruntime.directml") { Move-Item "microsoft.ml.onnxruntime.directml" "ort" }
if (Test-Path "microsoft.ai.directml") { Move-Item "microsoft.ai.directml" "dml" }

# OpenVINO C API headers (2025.4.0, GitHub raw)
$ovInc = "ov\include\openvino\c"
if (-not (Test-Path "$ovInc\openvino.h")) {
    Write-Host "Downloading OpenVINO C API headers..."
    $base = "https://raw.githubusercontent.com/openvinotoolkit/openvino/2025.4.0/src/bindings/c/include/openvino/c"
    New-Item -ItemType Directory -Force "$ovInc\auto", "$ovInc\gpu" | Out-Null
    $files = @("deprecated.h","openvino.h","ov_common.h","ov_compiled_model.h","ov_core.h",
               "ov_dimension.h","ov_infer_request.h","ov_layout.h","ov_model.h","ov_node.h",
               "ov_partial_shape.h","ov_prepostprocess.h","ov_property.h","ov_rank.h",
               "ov_remote_context.h","ov_shape.h","ov_tensor.h","ov_util.h")
    foreach ($f in $files) { curl.exe -s -o "$ovInc\$f" "$base/$f" }
    curl.exe -s -o "$ovInc\auto\properties.h" "$base/auto/properties.h"
    curl.exe -s -o "$ovInc\gpu\gpu_plugin_properties.h" "$base/gpu/gpu_plugin_properties.h"
}
Write-Host "SDK ready"
