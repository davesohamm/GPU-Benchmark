/********************************************************************************
 * @file    KernelLoader.h
 * @brief   Utility for loading OpenCL kernel source from files
 * 
 * @details Since OpenCL kernels are compiled at runtime, we need to load
 *          source code from .cl files. This helper provides convenient
 *          functions for loading and caching kernel source.
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace GPUBenchmark {
namespace OpenCLUtils {

/**
 * @brief Load OpenCL kernel source from file
 * @param filepath Path to .cl file
 * @return Kernel source code as string, empty on failure
 */
inline std::string LoadKernelSource(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

/**
 * @brief Kernel source cache for embedded kernels
 * @details For deployment, kernels can be embedded as strings rather than files
 */
class KernelSourceCache {
public:
    static KernelSourceCache& Instance() {
        static KernelSourceCache instance;
        return instance;
    }
    
    void RegisterKernel(const std::string& name, const std::string& source) {
        m_sources[name] = source;
    }
    
    std::string GetKernel(const std::string& name) const {
        auto it = m_sources.find(name);
        return (it != m_sources.end()) ? it->second : "";
    }
    
    bool HasKernel(const std::string& name) const {
        return m_sources.find(name) != m_sources.end();
    }
    
private:
    KernelSourceCache() = default;
    std::unordered_map<std::string, std::string> m_sources;
};

} // namespace OpenCLUtils
} // namespace GPUBenchmark
