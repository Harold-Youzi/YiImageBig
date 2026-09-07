# Create a test image (256x256 with gradients + shapes) for upscaling tests
Add-Type -AssemblyName System.Drawing

$size = 256
$bmp = New-Object System.Drawing.Bitmap($size, $size)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

# background gradient
$rect = New-Object System.Drawing.Rectangle(0, 0, $size, $size)
$brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect,
    [System.Drawing.Color]::FromArgb(255, 30, 60, 120),
    [System.Drawing.Color]::FromArgb(255, 200, 120, 40), 45.0)
$g.FillRectangle($brush, $rect)
$brush.Dispose()

# circles
$colors = @([System.Drawing.Color]::Red, [System.Drawing.Color]::Lime,
            [System.Drawing.Color]::Cyan, [System.Drawing.Color]::Yellow,
            [System.Drawing.Color]::Magenta, [System.Drawing.Color]::White)
for ($i = 0; $i -lt 6; $i++) {
    $cx = 40 + ($i % 3) * 80
    $cy = 70 + [Math]::Floor($i / 3) * 110
    $b = New-Object System.Drawing.SolidBrush($colors[$i])
    $g.FillEllipse($b, $cx - 28, $cy - 28, 56, 56)
    $b.Dispose()
}

# text
$font = New-Object System.Drawing.Font("Arial", 26, [System.Drawing.FontStyle]::Bold)
$tb = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
$g.DrawString("Yi 4X", $font, $tb, 60, 96)
$font.Dispose(); $tb.Dispose()

# fine grid lines (test detail preservation)
$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(120, 255, 255, 255), 1)
for ($i = 0; $i -lt $size; $i += 16) {
    $g.DrawLine($pen, $i, 0, $i, $size)
    $g.DrawLine($pen, 0, $i, $size, $i)
}
$pen.Dispose()

$g.Dispose()

$dir = Join-Path $PSScriptRoot "test_images"
New-Item -ItemType Directory -Force $dir | Out-Null
$bmp.Save((Join-Path $dir "test_256.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Save((Join-Path $dir "test_small_100.jpg"),
          [System.Drawing.Imaging.ImageFormat]::Jpeg)
$bmp.Dispose()
Write-Host "Test images created in $dir"
