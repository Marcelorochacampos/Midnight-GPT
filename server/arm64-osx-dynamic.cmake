# Targeting Apple Silicon (M4 = arm64 architecture)
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic) # Not applicable on macOS, but kept for consistency
set(VCPKG_LIBRARY_LINKAGE dynamic) # Use dynamic to avoid mixed linking issues with libtorch

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)

# Optional: CUDA is not typically available on Mac (especially Apple Silicon), so omit CUDA settings
# If you use ROCm or other accelerators on external devices, adapt accordingly

# Special handling for gflags and glog
if (${PORT} MATCHES "gflags|glog")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
