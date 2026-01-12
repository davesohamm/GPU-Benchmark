# 📦 ImGui Setup Instructions

## **Quick Setup** (Recommended)

### **Option 1: Run Download Script**
```cmd
DOWNLOAD_IMGUI.cmd
```

This will automatically download ImGui v1.90.1 to `external/imgui/`.

---

### **Option 2: Manual Download**

If the script doesn't work, follow these steps:

1. **Download ImGui**:
   - Go to: https://github.com/ocornut/imgui/releases/tag/v1.90.1
   - Download `Source code (zip)`

2. **Extract to Project**:
   - Extract the ZIP file
   - Copy ALL files from `imgui-1.90.1/` to `y:\GPU-Benchmark\external\imgui\`

3. **Verify Files**:
   Make sure these files exist in `external/imgui/`:
   ```
   imgui.h
   imgui.cpp
   imgui_demo.cpp
   imgui_draw.cpp
   imgui_tables.cpp
   imgui_widgets.cpp
   imconfig.h
   imgui_internal.h
   imstb_rectpack.h
   imstb_textedit.h
   imstb_truetype.h
   backends/imgui_impl_win32.h
   backends/imgui_impl_win32.cpp
   backends/imgui_impl_dx11.h
   backends/imgui_impl_dx11.cpp
   ```

---

## **What is ImGui?**

ImGui (Immediate Mode GUI) is a lightweight, powerful C++ GUI library used in:
- Game development tools
- Professional applications
- Real-time visualization software

**Perfect for our GPU Benchmark Suite!** 🎯

---

## **After Download:**

Run the CMake configuration again:
```cmd
BUILD.cmd
```

This will detect ImGui and build the GUI application!

---

## **Need Help?**

ImGui is ~50 files, all in one directory. Super simple!

If you have any issues, let me know and I'll help troubleshoot.
