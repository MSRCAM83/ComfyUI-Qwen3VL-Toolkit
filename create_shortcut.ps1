$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut('C:\Users\matth\Desktop\Fleet Pipeline.lnk')
$sc.TargetPath = 'C:\Users\matth\Desktop\Fleet Pipeline.bat'
$sc.WorkingDirectory = 'C:\Users\matth\ComfyUI-Qwen3VL-Toolkit'
$sc.IconLocation = 'C:\Users\matth\ComfyUI-Qwen3VL-Toolkit\fleet.ico'
$sc.Description = 'LoRA Dataset Fleet Pipeline'
$sc.Save()
Write-Host "Shortcut created successfully on desktop with custom icon!"
