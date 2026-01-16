@echo off
cls
echo.
echo     ╔═══════════════════════════════════════════════════════════╗
echo     ║                                                           ║
echo     ║        GPU BENCHMARK SUITE v2.0                          ║
echo     ║        Professional GPU Benchmarking Tool                 ║
echo     ║                                                           ║
echo     ║        All 3 Backends Working:                            ║
echo     ║        ✓ CUDA    ✓ OpenCL    ✓ DirectCompute             ║
echo     ║                                                           ║
echo     ╚═══════════════════════════════════════════════════════════╝
echo.
echo                    Created by: Soham Dave
echo              GitHub: github.com/davesohamm
echo.
echo     ───────────────────────────────────────────────────────────
echo.
echo     Launching GUI application...
echo.

cd /d "%~dp0"
start "" "build\Release\GPU-Benchmark-GUI.exe"

timeout /t 2 /nobreak >nul

echo     ✓ GPU Benchmark GUI launched!
echo.
echo     Window should appear in 2-3 seconds.
echo.
echo     ───────────────────────────────────────────────────────────
echo     QUICK START:
echo     ───────────────────────────────────────────────────────────
echo.
echo     1. Wait for window to appear
echo     2. Check "System Information" - Should show 3 green OK's
echo     3. Select any backend (CUDA recommended)
echo     4. Select suite (Standard for best balance)
echo     5. Click "Start Benchmark"
echo     6. Watch results appear!
echo     7. View performance graph
echo     8. Try other backends!
echo.
echo     ───────────────────────────────────────────────────────────
echo.
