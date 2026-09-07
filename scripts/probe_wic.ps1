# Extract IWICImagingFactory method order from mingw header
$msys = Get-ChildItem "C:\msys64\*\include\wincodec.h" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $msys) { Write-Host "未找到 msys64 wincodec.h"; exit 1 }
$h = [System.IO.File]::ReadAllText($msys.FullName)
$start = $h.IndexOf("interface IWICImagingFactory :")
if ($start -lt 0) { $start = $h.IndexOf("IWICImagingFactory :") }
Write-Host "start=$start"
$end = $h.IndexOf("IWICImagingFactory2", $start)
$seg = $h.Substring($start, $end - $start)
$lines = $seg -split "`n"
foreach ($l in $lines) {
    if ($l -match "STDMETHOD\(\w+\)|STDMETHODCALLTYPE\s+\w+\(") {
        Write-Host ($l.Trim().Substring(0, [Math]::Min(80, $l.Trim().Length)))
    }
}
