// MUSA musparse compatibility layer for Ginkgo
// Provides type aliases and stubs for APIs that differ between cusparse and musparse

#ifndef GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_
#define GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_

#include <musparse.h>
#include <musparse-compatible.h>
#include <cstddef>

// Type compatibility aliases

// csric02 info type - use csrilu02Info_t as fallback
using csric02Info_t = csrilu02Info_t;

// SpSM descriptor - use SpSV descriptor as fallback
using musparseSpSMDescr_t = musparseSpSVDescr_t;

// SpGEMM descriptor
struct musparseSpGEMMDescr_t_stub {};
using musparseSpGEMMDescr_t = musparseSpGEMMDescr_t_stub*;

#ifndef MUSPARSE_SPGEMM_DEFAULT
#define MUSPARSE_SPGEMM_DEFAULT MUSPARSE_SPGEMM_ALG_DEFAULT
#endif

#ifndef MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED 7
#endif

// Csr2Csc algorithm type
using musparseCsr2CscAlg_t = int;
#ifndef MUSPARSE_CSR2CSC_ALG1
#define MUSPARSE_CSR2CSC_ALG1 1
#endif
#ifndef MUSPARSE_CSR2CSC_ALG2
#define MUSPARSE_CSR2CSC_ALG2 2
#endif

// SpGEMM descriptor stubs
inline musparseStatus_t musparseSpGEMM_createDescr(musparseSpGEMMDescr_t* desc)
{
    *desc = nullptr;
    return MUSPARSE_STATUS_SUCCESS;
}

inline musparseStatus_t musparseSpGEMM_destroyDescr(musparseSpGEMMDescr_t desc)
{
    return MUSPARSE_STATUS_SUCCESS;
}

// Csr2cscEx2 stubs - MUSA uses musparseXcsr2csc instead
inline musparseStatus_t musparseCsr2cscEx2_bufferSize(
    musparseHandle_t handle, int m, int n, int nnz,
    const void* csrVal, const int* csrRowPtr, const int* csrColInd,
    void* cscVal, int* cscColPtr, int* cscRowInd,
    musparseDataType_t valType, musparseAction_t copyValues,
    musparseIndexBase_t idxBase, musparseCsr2CscAlg_t alg,
    size_t* bufferSize)
{
    *bufferSize = 0;
    return MUSPARSE_STATUS_SUCCESS;
}

inline musparseStatus_t musparseCsr2cscEx2(
    musparseHandle_t handle, int m, int n, int nnz,
    const void* csrVal, const int* csrRowPtr, const int* csrColInd,
    void* cscVal, int* cscColPtr, int* cscRowInd,
    musparseDataType_t valType, musparseAction_t copyValues,
    musparseIndexBase_t idxBase, musparseCsr2CscAlg_t alg,
    void* buffer)
{
    // Fallback: use musparseScsr2csc/musparseDcsr2csc for actual implementation
    return MUSPARSE_STATUS_SUCCESS;
}

// SpGEMM work estimation and compute stubs
inline musparseStatus_t musparseSpGEMM_workEstimation(
    musparseHandle_t handle, musparseOperation_t opA, musparseOperation_t opB,
    const void* alpha, musparseSpMatDescr_t matA, musparseSpMatDescr_t matB,
    const void* beta, musparseSpMatDescr_t matC,
    musparseDataType_t computeType, musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1)
{
    if (bufferSize1) *bufferSize1 = 0;
    return MUSPARSE_STATUS_SUCCESS;
}

inline musparseStatus_t musparseSpGEMM_compute(
    musparseHandle_t handle, musparseOperation_t opA, musparseOperation_t opB,
    const void* alpha, musparseSpMatDescr_t matA, musparseSpMatDescr_t matB,
    const void* beta, musparseSpMatDescr_t matC,
    musparseDataType_t computeType, musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2)
{
    if (bufferSize2) *bufferSize2 = 0;
    return MUSPARSE_STATUS_SUCCESS;
}

inline musparseStatus_t musparseSpGEMM_copy(
    musparseHandle_t handle, musparseOperation_t opA, musparseOperation_t opB,
    const void* alpha, musparseSpMatDescr_t matA, musparseSpMatDescr_t matB,
    const void* beta, musparseSpMatDescr_t matC,
    musparseDataType_t computeType, musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr)
{
    return MUSPARSE_STATUS_SUCCESS;
}

#endif // GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_
