# Ginkgo MUSA 迁移报告

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

## 迁移范围

### 已完成的工作

1. **创建 `musa/` 后端目录**（60 个文件）
   - 基于 `cuda/` 目录使用 Python 脚本（`simple_porting_ginkgo.py`）自动转换
   - CUDA API → MUSA API 替换（cudaMalloc→musaMalloc, cublas→mublas 等）
   - 文件重命名（`.cu`→`.mu`, `cublas_bindings.hpp`→`mublas_bindings.hpp` 等）
   - 头文件路径更新

2. **修改 `common/cuda_hip/` 共享代码层**
   - 9 个 dispatcher 头文件添加 `#elif defined(GKO_COMPILING_MUSA)` 分支
   - `thrust.hpp` 添加 MUSA thrust 策略
   - `math.hpp` 添加 `__mt_bfloat16` 设备数值极限
   - `bf16_alias.hpp` 添加 MUSA bfloat16 类型别名
   - `sorting.hpp` MUSA 与 CUDA 共享 `__shfl_xor_sync` 路径

3. **CMake 构建系统**
   - `cmake/musa.cmake`: MUSA SDK 发现、mcc 编译函数
   - `cmake/autodetect_executors.cmake`: 添加 MUSA 自动检测
   - `CMakeLists.txt`: 添加 `GINKGO_BUILD_MUSA` 选项
   - `musa/CMakeLists.txt`: MUSA 后端构建脚本（使用自定义 mcc 编译）
   - `devices/musa/`: 设备 executor 对象库

4. **核心库适配**
   - `include/ginkgo/core/base/exception_helpers.hpp`: 添加 MUSA 错误检查宏
   - `include/ginkgo/core/base/bfloat16.hpp`: 添加 `__mt_bfloat16` 前向声明和 traits
   - `musa/base/musparse_compat.hpp`: musparse API 兼容层

### 待完成的工作

| 问题 | 原因 | 建议修复方案 |
|------|------|------------|
| musparse API 不完全兼容 | MUSA 缺少 `musparseCsr2cscEx2`、`musparseSpGEMMDescr_t` 等 | 需要为缺失 API 实现真正的转换函数（用 musparseXcsr2csc 替代 Csr2cscEx2） |
| Stream 类型不兼容 | `CUstream_st*` vs `musaStream_t`(MUstream_st*) | 在 musa 后端中用 `reinterpret_cast` 转换，或修改 core 添加 MUSA stream 类型支持 |
| thrust complex 模板特化 | MUSA thrust target attribute 不匹配 | 为 MUSA 提供独立的半精度 thrust 特化或禁用半精度 |
| bfloat16 类型差异 | CUDA 用 `__nv_bfloat16`，MUSA 用 `__mt_bfloat16` | 已部分修复，需要更完整的转换支持 |
| general.json 误替换 | `environment` 被替换为 `emtironment` 等 | 已手动修复，需要在 porting 脚本中添加更多保护字符串 |

## 编译命令

```bash
# 环境要求
# - MUSA SDK (mcc >= 4.3.6) 安装在 /usr/local/musa
# - CMake >= 3.18
# - GCC >= 11

# 配置（仅 MUSA + Reference）
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
  -DGINKGO_BUILD_TESTS=OFF \
  -DGINKGO_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUSA_ARCH=22

# 编译
cmake --build . -j$(nproc)

# 注意: 当前 MUSA 后端编译尚需进一步修复 musparse 绑定
# Core + Reference + OMP 可以正常编译和测试
```

## 技术难点分析

1. **Ginkgo 架构与 MUSA 适配**
   - Ginkgo 的 `CudaExecutor` 深度集成在 core 库中，使用 CUDA 特有类型（`CUstream_st*`, `cublasContext*` 等）
   - 理想方案是新增 `MUSAExecutor` 类（类似 `HipExecutor`），但需要修改大量 core 代码
   - 当前方案复用 `CudaExecutor` 并在实现中调用 MUSA API

2. **musparse API 差异显著**
   - cusparse SpGEMM 使用 descriptor pattern（create/work_estimation/compute/copy/destroy）
   - musparse SpGEMM 使用 stage 参数的单函数 pattern
   - Csr2Csc: cusparse 有通用 `Csr2cscEx2`，musparse 只有类型特定的 `musparseXcsr2csc`

3. **mcc 编译器限制**
   - CMake 3.22 不支持 mcc 作为 CUDA 编译器（需要 nvcc 检测）
   - 使用 `add_custom_command` 绕过，但增加了构建复杂度
   - mcc 在编译大模板文件时内存消耗大（容易 OOM/segfault）

## 文件统计

| 类别 | 文件数 |
|------|--------|
| 新增 musa/ 后端文件 | 60 |
| 修改 common/cuda_hip/ 文件 | 13 |
| 修改 cmake 构建文件 | 4 |
| 修改 include/ 核心头文件 | 2 |
| 修改 devices/ 文件 | 2 |
| SimplePorting 脚本 | 1 |
| **总计修改/新增** | **82** |
