# ============================================================================
#  deploy.ps1 - Build and package YiImageBig into the "YiImageBig(C++)" folder
#  Usage: powershell -ExecutionPolicy Bypass -File deploy.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path $MyInvocation.MyCommand.Path -Parent }
$root     = Split-Path $scriptDir -Parent             # SourceCode(C++)
$parent   = Split-Path $root -Parent                  # project root
$buildDir = Join-Path $root "build"
$outDir   = Join-Path $parent "YiImageBig(C++)"

Write-Host "==> 1/4 Build Release  (buildDir=$buildDir)"
$env:Path = "C:\msys64\ucrt64\bin;$env:Path"
if (-not (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
    & cmake -G Ninja -S $root -B $buildDir -DCMAKE_BUILD_TYPE=Release
}
& cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host "==> 2/4 Create directories"
New-Item -ItemType Directory -Force (Join-Path $outDir "models"),
                                   (Join-Path $outDir "runtime") | Out-Null

Write-Host "==> 3/4 Copy app + models"
Copy-Item (Join-Path $buildDir "bin\YiImageBig.exe") $outDir -Force

$models = @(
    "RealESRGAN_x4plus.xml",
    "RealESRGAN_x4plus.bin",
    "RealESRGAN_x4plus.onnx",
    "RealESRGAN_x4plus_npu_128.xml",
    "RealESRGAN_x4plus_npu_128.bin"
)
$modelsSrc = Join-Path $parent "RealESRGAN"
foreach ($m in $models) {
    $src = Join-Path $modelsSrc $m
    if (Test-Path $src) { Copy-Item $src (Join-Path $outDir "models") -Force }
    else { Write-Warning "Missing model: $m" }
}

# copy the user guide (Chinese filename built from code points for safety)
$readme = Join-Path $root ([string][char]0x4F7F + [char]0x7528 + [char]0x8BF4 + [char]0x660E + ".md")
if (Test-Path $readme) { Copy-Item $readme $outDir -Force }

Write-Host "==> 4/4 Copy runtime DLLs"
$ovNative = Join-Path $root "third_party\ov\runtimes\win-x64\native"
Get-ChildItem $ovNative -File | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $outDir "runtime") -Force
}
Copy-Item "$root\third_party\ort\runtimes\win-x64\native\onnxruntime.dll" (Join-Path $outDir "runtime") -Force
$dmlDll = Get-ChildItem "$root\third_party\dml" -Recurse -Filter "DirectML.dll" |
          Where-Object { $_.FullName -match "x64-win" } | Select-Object -First 1
if ($dmlDll) { Copy-Item $dmlDll.FullName (Join-Path $outDir "runtime") -Force }
else { Write-Warning "Missing DirectML.dll" }

Write-Host ""
Write-Host "Deploy OK: $outDir"
Get-ChildItem $outDir | Format-Table Name
$rt = Get-ChildItem (Join-Path $outDir "runtime") -File
Write-Host ("runtime files: {0}, total {1:N1} MB" -f $rt.Count,
    (($rt | Measure-Object Length -Sum).Sum / 1MB))
