# 生成 YiImageBig 应用图标 (渐变圆角方块 + "4X")
Add-Type -AssemblyName System.Drawing

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             else { Split-Path $MyInvocation.MyCommand.Path -Parent }
$outFile = Join-Path $scriptDir "..\src\res\app.ico"
$srcDir  = Split-Path $outFile
New-Item -ItemType Directory -Force $srcDir | Out-Null

function New-IconBitmap([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias

    $g.Clear([System.Drawing.Color]::Transparent)

    # 圆角矩形区域
    $r = [Math]::Max(2, [int]($size * 0.22))
    $rect = New-Object System.Drawing.Rectangle(1, 1, ($size - 2), ($size - 2))
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($rect.X, $rect.Y, 2*$r, 2*$r, 180, 90)
    $path.AddArc($rect.Right - 2*$r, $rect.Y, 2*$r, 2*$r, 270, 90)
    $path.AddArc($rect.Right - 2*$r, $rect.Bottom - 2*$r, 2*$r, 2*$r, 0, 90)
    $path.AddArc($rect.X, $rect.Bottom - 2*$r, 2*$r, 2*$r, 90, 90)
    $path.CloseFigure()

    # 对角渐变 (蓝 -> 紫)
    $c1 = [System.Drawing.Color]::FromArgb(255, 41, 118, 255)
    $c2 = [System.Drawing.Color]::FromArgb(255, 148, 66, 255)
    $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $c1, $c2, 35.0)
    $g.FillPath($brush, $path)
    $brush.Dispose()

    # 上部高光
    $hl = New-Object System.Drawing.Drawing2D.GraphicsPath
    $hr = [Math]::Max(2, [int]($size * 0.18))
    $hlRect = New-Object System.Drawing.Rectangle(2, 2, ($size - 4), [int]($size * 0.45))
    $hl.AddArc($hlRect.X, $hlRect.Y, 2*$hr, 2*$hr, 180, 90)
    $hl.AddArc($hlRect.Right - 2*$hr, $hlRect.Y, 2*$hr, 2*$hr, 270, 90)
    $hl.AddLine($hlRect.Right, $hlRect.Bottom, $hlRect.X, $hlRect.Bottom)
    $hl.CloseFigure()
    $hlBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(45, 255, 255, 255))
    $g.FillPath($hlBrush, $hl)
    $hlBrush.Dispose(); $hl.Dispose()

    # "4X" 文字
    $font = New-Object System.Drawing.Font("Segoe UI", ($size * 0.42),
        [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
    $textRect = New-Object System.Drawing.RectangleF(0, ($size * 0.03), $size, ($size * 0.94))
    $textBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $g.DrawString("4X", $font, $textBrush, $textRect, $fmt)
    $font.Dispose(); $fmt.Dispose(); $textBrush.Dispose()
    $path.Dispose()
    $g.Dispose()
    return $bmp
}

# 写入多尺寸 ICO
$sizes = @(16, 24, 32, 48, 64, 128, 256)
$memStream = New-Object System.IO.MemoryStream
$bw = New-Object System.IO.BinaryWriter($memStream)

# ICONDIR
$bw.Write([UInt16]0); $bw.Write([UInt16]1); $bw.Write([UInt16]$sizes.Count)

$bmps = @()
$offset = 6 + 16 * $sizes.Count
foreach ($s in $sizes) {
    $bmp = New-IconBitmap $s
    $bmps += $bmp
    $ms2 = New-Object System.IO.MemoryStream
    $bmp.Save($ms2, [System.Drawing.Imaging.ImageFormat]::Png)
    $png = $ms2.ToArray()
    $ms2.Dispose()
    # ICONDIRENTRY
    $bw.Write([byte]$(if ($s -ge 256) {0} else {$s}))   # width
    $bw.Write([byte]$(if ($s -ge 256) {0} else {$s}))   # height
    $bw.Write([byte]0)                                   # colors
    $bw.Write([byte]0)                                   # reserved
    $bw.Write([UInt16]1)                                 # planes
    $bw.Write([UInt16]32)                                # bpp
    $bw.Write([UInt32]$png.Length)
    $bw.Write([UInt32]$offset)
    $offset += $png.Length
}
foreach ($bmp in $bmps) {
    $ms2 = New-Object System.IO.MemoryStream
    $bmp.Save($ms2, [System.Drawing.Imaging.ImageFormat]::Png)
    $bw.Write($ms2.ToArray())
    $ms2.Dispose()
    $bmp.Dispose()
}
$bw.Flush()
[System.IO.File]::WriteAllBytes($outFile, $memStream.ToArray())
$bw.Dispose(); $memStream.Dispose()

Write-Host "图标已生成: $outFile ($((Get-Item $outFile).Length) bytes)"
