cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)

# Template instantiation (shared with CUDA/HIP)
add_instantiation_files(
    ${PROJECT_SOURCE_DIR}/common/cuda_hip
    matrix/csr_kernels.instantiate.cpp
    CSR_INSTANTIATE
)
add_instantiation_files(
    ${PROJECT_SOURCE_DIR}/common/cuda_hip
    matrix/fbcsr_kernels.instantiate.cpp
    FBCSR_INSTANTIATE
)
add_instantiation_files(
    ${PROJECT_SOURCE_DIR}/common/cuda_hip
    solver/batch_bicgstab_launch.instantiate.cpp
    BATCH_BICGSTAB_INSTANTIATE
)
add_instantiation_files(
    ${PROJECT_SOURCE_DIR}/common/cuda_hip
    solver/batch_cg_launch.instantiate.cpp
    BATCH_CG_INSTANTIATE
)
list(
    APPEND GKO_UNIFIED_COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp
)

# Jacobi block sizes
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_MUSA_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 32)
        list(APPEND GKO_MUSA_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_MUSA_JACOBI_BLOCK_SIZES 1 2 4 8 13 16 32)
endif()
jacobi_generated_files(GKO_MUSA_JACOBI_SOURCES "${GKO_MUSA_JACOBI_BLOCK_SIZES}")
string(REPLACE ";" "," GKO_JACOBI_BLOCK_SIZES_CODE "${GKO_MUSA_JACOBI_BLOCK_SIZES}")
configure_file(
    ${Ginkgo_SOURCE_DIR}/common/cuda_hip/preconditioner/jacobi_common.hpp.in
    common/cuda_hip/preconditioner/jacobi_common.hpp
)

# MUSA backend local sources
set(MUSA_LOCAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/base/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/event_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/mtprofile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/scoped_device_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/base/version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix/fft_kernels.mu
    ${CMAKE_CURRENT_SOURCE_DIR}/preconditioner/batch_jacobi_kernels.mu
    ${CMAKE_CURRENT_SOURCE_DIR}/solver/batch_bicgstab_kernels.mu
    ${CMAKE_CURRENT_SOURCE_DIR}/solver/batch_cg_kernels.mu
    ${CMAKE_CURRENT_SOURCE_DIR}/solver/lower_trs_kernels.mu
    ${CMAKE_CURRENT_SOURCE_DIR}/solver/upper_trs_kernels.mu
)

# All device sources
set(ALL_MUSA_SOURCES
    ${MUSA_LOCAL_SOURCES}
    ${GKO_CUDA_HIP_COMMON_SOURCES}
    ${GKO_UNIFIED_COMMON_SOURCES}
    ${CSR_INSTANTIATE}
    ${FBCSR_INSTANTIATE}
    ${BATCH_BICGSTAB_INSTANTIATE}
    ${BATCH_CG_INSTANTIATE}
    ${GKO_MUSA_JACOBI_SOURCES}
)

# Compile all sources with mcc
musa_compile(MUSA_OBJECTS ${ALL_MUSA_SOURCES})

# Create library from compiled objects + device executor object
add_library(ginkgo_musa $<TARGET_OBJECTS:ginkgo_musa_device> ${MUSA_OBJECTS})
set_target_properties(ginkgo_musa PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
    ginkgo_musa
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${MUSA_INCLUDE_DIRS}
)
target_compile_definitions(
    ginkgo_musa
    PRIVATE GKO_COMPILING_MUSA GKO_DEVICE_NAMESPACE=musa
)

target_link_libraries(ginkgo_musa PRIVATE ${MUSA_LIBRARIES})
target_link_libraries(ginkgo_musa PUBLIC ginkgo_device ${CMAKE_DL_LIBS})

ginkgo_compile_features(ginkgo_musa)
ginkgo_default_includes(ginkgo_musa)
ginkgo_install_library(ginkgo_musa)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_musa "GKO_COMPILING_MUSA;GKO_DEVICE_NAMESPACE=musa")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
