@echo off
REM ============================================================================
REM Download ImGui for GPU-Benchmark GUI
REM ============================================================================

echo ========================================
echo   Downloading ImGui v1.90.1
echo ========================================
echo.

cd external\imgui

REM Download ImGui from GitHub
echo Downloading ImGui files...
curl -L -o imgui.zip https://github.com/ocornut/imgui/archive/refs/tags/v1.90.1.zip

echo Extracting...
tar -xf imgui.zip --strip-components=1

echo Cleaning up...
del imgui.zip

echo.
echo ========================================
echo   ImGui Downloaded Successfully!
echo ========================================
echo.
echo Files downloaded to: external\imgui\
echo.

cd ..\..

pause
