# 🔧 GUI Application Troubleshooting

## **Issue**: GUI Window Not Appearing

### **Symptom:**
When you run `GPU-Benchmark-GUI.exe` or `RUN_GUI.cmd`, the window doesn't appear.

---

## ✅ **Solutions**

### **Solution 1: Kill Existing Instances**

The GUI might already be running in the background!

```cmd
REM Check if running
tasklist | findstr "GPU-Benchmark-GUI"

REM Kill it
taskkill /F /IM "GPU-Benchmark-GUI.exe"

REM Now run again
RUN_GUI.cmd
```

### **Solution 2: Run Directly**

Instead of the script, run the exe directly:

```cmd
cd build\Release
GPU-Benchmark-GUI.exe
```

### **Solution 3: Check Task Manager**

1. Open Task Manager (Ctrl+Shift+Esc)
2. Look for "GPU-Benchmark-GUI.exe"
3. If it exists:
   - End the task
   - Run again

### **Solution 4: Wait for Initialization**

The GUI needs 2-3 seconds to:
- Detect your GPU
- Initialize CUDA/OpenCL/DirectCompute
- Create the window

**Be patient!** The window WILL appear.

---

## 🐛 **Common Issues**

### **1. Window Flashes and Closes**

**Cause**: Initialization error

**Fix**:
```cmd
REM Run from command line to see errors
cd build\Release
GPU-Benchmark-GUI.exe
```

### **2. "Application Error" Dialog**

**Cause**: Missing DLLs or GPU drivers

**Fix**:
- Install/update NVIDIA drivers
- Install Visual C++ Redistributable 2022
- Install DirectX Runtime

### **3. Black Screen**

**Cause**: DirectX initialization issue

**Fix**:
- Update graphics drivers
- Run Windows Update
- Check DirectX diagnostic (dxdiag.exe)

### **4. Process Running But No Window**

**Cause**: Window created but not visible

**Fix**:
```cmd
REM Kill and restart
taskkill /F /IM "GPU-Benchmark-GUI.exe"
timeout /t 2
RUN_GUI.cmd
```

---

## 📊 **What the GUI Does on Startup**

When you run `GPU-Benchmark-GUI.exe`, it:

1. **Creates Window** (~100ms)
   - Win32 window initialization
   - DirectX 11 device creation

2. **Initializes ImGui** (~50ms)
   - GUI framework setup
   - Font loading

3. **Detects GPU Hardware** (~1-2 seconds) ⏱️
   - Enumerates all GPUs
   - Checks CUDA availability
   - Checks OpenCL availability
   - Checks DirectCompute availability
   - Queries system info (CPU, RAM, OS)

4. **Shows Window** 
   - Window appears with system info
   - Ready for user interaction

**Total Time**: 2-3 seconds

**Be patient during this time!**

---

## 🎯 **Expected Behavior**

### **Correct Startup:**

1. Run `RUN_GUI.cmd` or `GPU-Benchmark-GUI.exe`
2. Wait 2-3 seconds (initialization)
3. Window appears showing:
   - "GPU BENCHMARK SUITE" title
   - Your GPU name
   - Available backends (CUDA, OpenCL, DirectCompute)
   - Configuration controls

### **If This Doesn't Happen:**

Follow the solutions above!

---

## 🔍 **Debug Mode**

Want to see what's happening?

Create a file `DEBUG_GUI.cmd`:

```cmd
@echo off
cd /d "%~dp0"
cd build\Release

echo Starting GPU Benchmark GUI in debug mode...
echo.
echo Press Ctrl+C to abort
echo.

REM Run with pause so you can see any errors
GPU-Benchmark-GUI.exe
pause
```

Then run `DEBUG_GUI.cmd` to see any error messages.

---

## ✅ **Verification Steps**

### **1. Check Build:**
```cmd
dir build\Release\GPU-Benchmark-GUI.exe
```
Should show the file exists (~5-10 MB)

### **2. Check Dependencies:**
```cmd
cd build\Release
dumpbin /dependents GPU-Benchmark-GUI.exe
```
Should show: d3d11.dll, dxgi.dll, OpenCL.dll, etc.

### **3. Check GPU:**
```cmd
dxdiag
```
Should show DirectX 11 support

---

## 💡 **Quick Fixes**

### **Method 1: Fresh Start**
```cmd
REM Clean build
rmdir /s /q build
BUILD.cmd
RUN_GUI.cmd
```

### **Method 2: Direct Execution**
```cmd
cd build\Release
start GPU-Benchmark-GUI.exe
```

### **Method 3: Administrator Mode**
```cmd
REM Right-click RUN_GUI.cmd
REM Select "Run as administrator"
```

---

## 🎉 **Success Indicators**

You'll know it's working when you see:

✅ Window titled "GPU Benchmark Suite"  
✅ System Information panel  
✅ Your GPU name displayed  
✅ Green checkmarks for available backends  
✅ "Start Benchmark" button visible  

---

## 📞 **Still Having Issues?**

1. **Check System Requirements:**
   - Windows 10/11
   - DirectX 11 compatible GPU
   - Updated graphics drivers

2. **Try CLI Version:**
   ```cmd
   build\Release\GPU-Benchmark.exe --quick
   ```
   If this works, the GPU is fine.

3. **Check Event Viewer:**
   - Open Event Viewer
   - Look for application crashes
   - Check error messages

---

## 🚀 **Alternative: Use CLI Version**

If GUI doesn't work, you can still benchmark:

```cmd
REM Quick benchmark
build\Release\GPU-Benchmark.exe --quick

REM Standard benchmark  
build\Release\GPU-Benchmark.exe --standard

REM Full benchmark
build\Release\GPU-Benchmark.exe --full
```

The CLI version is fully functional!

---

**Remember: The window takes 2-3 seconds to appear. Be patient!** ⏱️
