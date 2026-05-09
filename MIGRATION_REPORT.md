# Ginkgo MUSA 迁移报告（更新版）

## 项目概述

- **项目名称**: Ginkgo - 高性能数值线性代数库
- **仓库版本**: v2.0.0-develop (commit 8e9859e58)
- **项目类型**: C++ CMake 库（Lib 类型）
- **原有 GPU 后端**: CUDA、HIP、DPC++/SYCL

## 迁移分支

| 分支 | 说明 |
|------|------|
| `official-ginkgo` | 官方源码（基于 develop 分支） |
| `musa-ginkgo` | MUSA 迁移分支 |

## 迁移进度

### 第一轮迁移（已完成）

1. 创建 `musa/` 后端目录（60个文件），基于CUDA自动转换
2. 修改 `common/cuda_hip/` 共享代码层（dispatcher头文件）
3. CMake 构建系统（musa.cmake, autodetect, CMakeLists.txt）
4. 核心库适配（exception_helpers, bfloat16, musparse compat）

### 第二轮迁移（本次完成）

1. **CMake 测试框架**
   - `cmake/create_test.cmake`: 添加 `musagpu` 资源类型
   - 添加 `ginkgo_create_musa_test` 和 `ginkgo_create_musa_test_internal` 函数
   - 添加 MUSA 到 `ginkgo_create_common_test` 和 `ginkgo_create_common_device_test`

2. **Thrust 适配**
   - `common/cuda_hip/base/thrust.hpp`: MUSA 使用 `thrust::musa::par` 而非 `thrust::cuda::par`
   - 引入 `thrust/system/musa/detail/execution_policy.h`

3. **CUstream/CUevent 类型兼容**
   - `include/ginkgo/core/base/fwd_decls.hpp`: MUSA 下 typedef `CUstream_st` = `MUstream_st`，`CUevent_st` = `MUevent_st`
   - 同时 typedef `cublasContext` = `_mublasHandle_t`，`cusparseContext` = `_musparse_handle`

4. **内核编译修复**
   - `include/ginkgo/core/base/types.hpp`: 添加 `__MUSACC__` 到 `GKO_ATTRIBUTES` 宏
   - `include/ginkgo/core/base/abstract_factory.hpp`: 添加 `__MUSACC__` 检查
   - `cmake/musa.cmake`: 修复 `get_filename_component` 使用 `LAST_EXT` 替代 `EXT`

5. **内核声明宏**
   - `core/base/kernel_declaration.hpp`: 在 `GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES` 宏中添加 `musa` 命名空间
   - `include/ginkgo/core/base/executor.hpp`: 添加 `musa::DefaultExecutor` 别名

6. **统一内核启动**
   - `common/unified/base/kernel_launch.hpp`: 添加 `GKO_COMPILING_MUSA` 分支
   - `common/unified/base/kernel_launch_reduction.hpp`: 同上
   - `common/unified/base/kernel_launch_solver.hpp`: 同上

7. **批量求解器调度**
   - `core/solver/batch_dispatch.hpp`: 添加 `GKO_COMPILING_MUSA` 分支，使用 `musa` 命名空间

8. **musparse API 修复**
   - ILU0/IC0 buffer_size: `int*` → `size_t*`
   - ILU0/IC0 analysis: 添加 `musparseAnalysisPolicy_t` 参数
   - ILU0/IC0 analysis: `const_cast` 去除 vals 的 const 限定
   - 移除冲突的 compat 层（MUSA SDK 已原生支持 SpGEMM/Csr2CscEx2）
   - 添加 CUDA→MUSA 类型别名和常量映射

9. **内存操作**
   - `musa/components/memory.muh`: 完全重写，使用 volatile 读写替代 NVIDIA PTX assembly
   - 添加 `atomic_cas_relaxed`, `atomic_min_relaxed` 等原子操作

10. **其他修复**
    - `musa/base/stream.cpp`: 修复类名 `musa_stream` → `cuda_stream`
    - `musa/base/timer.cpp`: 修复 `time_point::type::musa` → `cuda`
    - `musa/base/version.cpp`: 修复函数名 `get_musa_version` → `get_cuda_version`
    - `musa/base/mtprofile.cpp`: NVTX profiling 改为 no-op（MUSA 无等效 API）
    - `musa/matrix/fft_kernels.mu`: `mufftMakePlanMany64` → `mufftMakePlanMany`
    - `musa/base/musparse_bindings.hpp`: 修复 `create_dmtec` → `create_dnvec`
    - `include/ginkgo/core/base/bfloat16.hpp`: MUSA 下不声明 `__nv_bfloat16`
    - `common/cuda_hip/components/atomic.hpp`: MUSA 下排除 `__nv_bfloat16` 特化
    - 修复 dispatcher 中 `.hpp` → `.muh` 引用

## 当前编译状态

大部分 MUSA 后端文件成功编译。以下文件仍有编译问题：

| 文件 | 错误原因 | 严重程度 |
|------|----------|----------|
| csr_kernels.instantiate.33-36 | mcc 编译器 SIGSEGV（模板过大） | 高 |
| fbcsr_kernels.instantiate.0-2 | mcc 编译器 SIGSEGV（模板过大） | 高 |
| sparsity_csr_kernels | musparse_block_bindings.hpp API 差异 | 中 |
| rcm_kernels | musparse_block_bindings.hpp API 差异 | 中 |
| lower/upper_trs_kernels | common_trs_kernels.muh SpSM API 差异 | 中 |
| executor.cpp | intrinsics.hpp 中 __builtin 不匹配 | 低 |
| csr_kernels (unified) | 某些 sparselib 函数签名差异 | 中 |

## 编译命令

```bash
mkdir build && cd build
cmake .. \
  -DGINKGO_BUILD_CUDA=OFF \
  -DGINKGO_BUILD_HIP=OFF \
  -DGINKGO_BUILD_SYCL=OFF \
  -DGINKGO_BUILD_MUSA=ON \
  -DGINKGO_BUILD_REFERENCE=ON \
  -DGINKGO_BUILD_OMP=ON \
  -DGINKGO_ENABLE_HALF=OFF \
  -DGINKGO_ENABLE_BFLOAT16=OFF \
  -DGINKGO_BUILD_TESTS=ON \
  -DGINKGO_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUSA_ARCH=22

cmake --build . -j$(nproc)
```

## 待解决问题

1. **mcc 编译器崩溃**: CSR/FBCSR 大型模板实例化文件导致 mcc SIGSEGV，需要拆分实例化或升级 mcc
2. **musparse block bindings**: `musparseXbsrsv2_bufferSize` 等 BSR 稀疏 API 签名差异
3. **SpSM (Sparse Triangular Solve)**: `musparseSpSM_analysis` API 参数差异
4. **__builtin 内置函数**: MUSA 的 clang 14 基础与 GCC 12 的一些内置函数声明不兼容
