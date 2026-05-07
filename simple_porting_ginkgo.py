#!/usr/bin/env python3
"""
Porting script for Ginkgo CUDA-to-MUSA migration.
Creates a musa/ directory from cuda/ with proper file renaming and API replacements.
"""
import os
import json
import shutil

PROJECT_DIR = "/workspace/repos/ginkgo"
CUDA_DIR = os.path.join(PROJECT_DIR, "cuda")
MUSA_DIR = os.path.join(PROJECT_DIR, "musa")

def load_json(path):
    with open(path) as f:
        return json.load(f)

GENERAL_MAP = load_json("/usr/local/musa/tools/general.json")
INCLUDE_MAP = load_json("/usr/local/musa/tools/include.json")

# File renaming rules (basename level)
FILE_RENAME = {
    "cublas_bindings.hpp": "mublas_bindings.hpp",
    "cublas_handle.hpp": "mublas_handle.hpp",
    "curand_bindings.hpp": "murand_bindings.hpp",
    "cusparse_bindings.hpp": "musparse_bindings.hpp",
    "cusparse_block_bindings.hpp": "musparse_block_bindings.hpp",
    "cusparse_handle.hpp": "musparse_handle.hpp",
    "nvtx.cpp": "mtprofile.cpp",
    "cuda_executor.cu": "musa_executor.mu",
    "cuda_executor_topology.cu": "musa_executor_topology.mu",
}

EXT_MAP = {".cu": ".mu", ".cuh": ".muh"}

# Content replacement rules (applied in order, before general mapping)
CONTENT_RULES = [
    # Internal include path fixes
    ("cuda/base/cublas_bindings.hpp", "musa/base/mublas_bindings.hpp"),
    ("cuda/base/cublas_handle.hpp", "musa/base/mublas_handle.hpp"),
    ("cuda/base/curand_bindings.hpp", "musa/base/murand_bindings.hpp"),
    ("cuda/base/cusparse_bindings.hpp", "musa/base/musparse_bindings.hpp"),
    ("cuda/base/cusparse_block_bindings.hpp", "musa/base/musparse_block_bindings.hpp"),
    ("cuda/base/cusparse_handle.hpp", "musa/base/musparse_handle.hpp"),
    ("cuda/base/config.hpp", "musa/base/config.hpp"),
    ("cuda/base/device.hpp", "musa/base/device.hpp"),
    ("cuda/base/pointer_mode_guard.hpp", "musa/base/pointer_mode_guard.hpp"),
    ("cuda/base/scoped_device_id.hpp", "musa/base/scoped_device_id.hpp"),
    ("cuda/base/types.hpp", "musa/base/types.hpp"),
    ("cuda/components/cooperative_groups.cuh", "musa/components/cooperative_groups.muh"),
    ("cuda/components/format_conversion.cuh", "musa/components/format_conversion.muh"),
    ("cuda/components/memory.cuh", "musa/components/memory.muh"),
    ("cuda/solver/common_trs_kernels.cuh", "musa/solver/common_trs_kernels.muh"),
    ("cuda/solver/batch_bicgstab_launch.cuh", "musa/solver/batch_bicgstab_launch.muh"),
    ("cuda/solver/batch_cg_launch.cuh", "musa/solver/batch_cg_launch.muh"),
    # Ginkgo specific macros/names
    ("GKO_COMPILING_CUDA", "GKO_COMPILING_MUSA"),
    ("GKO_DEVICE_NAMESPACE=cuda", "GKO_DEVICE_NAMESPACE=musa"),
    ("ginkgo_cuda_device", "ginkgo_musa_device"),
    ("ginkgo_cuda", "ginkgo_musa"),
    # Header guards
    ("GKO_CUDA_BASE_", "GKO_MUSA_BASE_"),
    ("GKO_CUDA_COMPONENTS_", "GKO_MUSA_COMPONENTS_"),
    ("GKO_CUDA_SOLVER_", "GKO_MUSA_SOLVER_"),
    # Namespace
    ("namespace cuda {", "namespace musa {"),
    ("}  // namespace cuda", "}  // namespace musa"),
    ("kernels::cuda::", "kernels::musa::"),
    ("kernels::cuda,", "kernels::musa,"),
    # CMake targets
    ("CUDA::cudart", "musart"),
    ("CUDA::cublas", "mublas"),
    ("CUDA::cusparse", "musparse"),
    ("CUDA::curand", "murand"),
    ("CUDA::cufft", "mufft"),
    ("nvtx::nvtx", ""),
    # NVTX (no direct equivalent, stub)
    ('#include <nvtx3/nvToolsExt.h>', '// MUSA: no NVTX equivalent'),
]

# Strings that should NOT be replaced by general mapping
# We'll protect them by replacing with placeholders before general mapping,
# then restoring after.
PROTECT_STRINGS = [
    "common/cuda_hip/",
    "cuda_hip/",
    "GKO_CUDA_HIP_COMMON_SOURCES",
    "CudaExecutor",
    "CudaAllocator",
    "CudaAllocatorBase",
    "CudaUnifiedAllocator",
    "cuda_allocator_from_mode",
    "GINKGO_HAS_CUDA",
    "GINKGO_BUILD_CUDA",
    "CUDA_VERSION",
    "CUstream_st",
    "default_cuda_alloc_mode",
    "GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE",
    "THRUST_CUB_WRAPPED_NAMESPACE",
]

def map_filename(filename):
    if filename in FILE_RENAME:
        return FILE_RENAME[filename]
    base, ext = os.path.splitext(filename)
    if ext in EXT_MAP:
        return base + EXT_MAP[ext]
    return filename

def apply_replacements(content):
    # 1. Include mapping
    for cuda_inc, musa_inc in INCLUDE_MAP.items():
        content = content.replace(cuda_inc, musa_inc)
    # 2. Project-specific content rules
    for old, new in CONTENT_RULES:
        content = content.replace(old, new)
    # 3. Protect certain strings from general mapping
    placeholders = {}
    for i, s in enumerate(PROTECT_STRINGS):
        ph = f"__PROTECT_{i:04d}__"
        placeholders[ph] = s
        content = content.replace(s, ph)
    # 4. General API mapping (longest first)
    sorted_keys = sorted(GENERAL_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in content:
            content = content.replace(key, GENERAL_MAP[key])
    # 5. Restore protected strings
    for ph, s in placeholders.items():
        content = content.replace(ph, s)
    return content

def port_directory():
    if os.path.exists(MUSA_DIR):
        shutil.rmtree(MUSA_DIR)
    count = 0
    for root, dirs, files in os.walk(CUDA_DIR):
        rel_path = os.path.relpath(root, CUDA_DIR)
        dest_dir = os.path.join(MUSA_DIR, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        for filename in files:
            src = os.path.join(root, filename)
            dst = os.path.join(dest_dir, map_filename(filename))
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = apply_replacements(content)
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(content)
            except UnicodeDecodeError:
                shutil.copy2(src, dst)
            count += 1
    print(f"Ported {count} files from cuda/ to musa/")

if __name__ == "__main__":
    port_directory()
