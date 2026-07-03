@echo off
REM 隔夜实验调度器 - 双击即启动
REM 用 cmd 窗口绕开 PowerShell 5.x 的 native exe stdout 重编码 UTF-16 坑 (附录 H)
REM 脚本内部自带 Tee 写 _run_pending.log, 不在这里做 shell 重定向

cd /d "%~dp0"
echo === 隔夜实验调度器 ===
echo cwd = %CD%
echo.

".venv\Scripts\python.exe" -u run_pending.py %*

echo.
echo === 完成. 按任意键关窗口 ===
pause
