# 🎨 App Icon Integration Guide

## Current Status

**Icon File:** ✅ Located at `assets/icon.png`
**Format:** PNG (needs conversion to ICO)
**Status:** ⏳ Ready for integration

---

## Why Icon Isn't Showing Yet

Windows executables require:
1. **ICO format** (not PNG)
2. **Resource file** (.rc)
3. **CMake integration**

---

## How to Add Icon (3 Steps)

### Step 1: Convert PNG to ICO

**Option A: Online Converter (Easiest)**
1. Go to: https://convertio.co/png-ico/
2. Upload: `assets/icon.png`
3. Download: `icon.ico`
4. Save to: `assets/icon.ico`

**Option B: Using GIMP (Free Software)**
1. Install GIMP (https://www.gimp.org/)
2. Open `assets/icon.png`
3. Export As → `icon.ico`
4. Save to `assets/icon.ico`

**Option C: Using ImageMagick (Command Line)**
```cmd
magick convert assets\icon.png -define icon:auto-resize=256,128,64,48,32,16 assets\icon.ico
```

---

### Step 2: Create Resource File

Create file: `src/gui/app.rc`

```rc
// App Icon Resource File
IDI_ICON1 ICON "../../assets/icon.ico"
```

---

### Step 3: Update CMakeLists.txt

Find the GUI target section and add:

```cmake
# GUI Application
add_executable(GPU-Benchmark-GUI 
    src/gui/main_gui_fixed.cpp
    ${CUDA_KERNELS}
    ${OPENCL_SOURCES}
    ${DIRECTCOMPUTE_SOURCES}
)

# Add icon resource file for Windows
if(WIN32)
    target_sources(GPU-Benchmark-GUI PRIVATE src/gui/app.rc)
endif()

# ... rest of configuration ...
```

---

### Step 4: Rebuild

```cmd
cmake --build build --config Release --target GPU-Benchmark-GUI
```

---

## Verification

After rebuild:
1. Right-click `GPU-Benchmark-GUI.exe`
2. Check Properties
3. Icon should appear in file explorer
4. Icon shows in taskbar when running

---

## Alternative: Set Icon at Runtime (C++ Code)

If resource file doesn't work, you can set icon programmatically:

```cpp
// In WinMain, after CreateWindowW
HICON hIcon = (HICON)LoadImageA(NULL, "assets/icon.ico", IMAGE_ICON, 
                                 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
if (hIcon) {
    SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
    SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
}
```

---

## Why Not Done Automatically?

1. **PNG → ICO conversion** requires external tool
2. **Can't programmatically convert** in build process
3. **Manual step needed** (one-time)

---

## Quick Summary

**Current:**
- ✅ Icon PNG exists
- ⏳ Needs conversion to ICO
- ⏳ Needs resource file
- ⏳ Needs CMake update

**To Complete:**
1. Convert `icon.png` → `icon.ico` (use online converter)
2. Create `src/gui/app.rc` (copy text above)
3. Update `CMakeLists.txt` (add if(WIN32) section)
4. Rebuild

**Time Required:** 5 minutes

---

**Once done, your exe will have a custom icon!** 🎨
