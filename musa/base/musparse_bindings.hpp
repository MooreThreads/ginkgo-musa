// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_CUSPARSE_BINDINGS_HPP_
#define GKO_MUSA_BASE_CUSPARSE_BINDINGS_HPP_


#include <musa.h>
#include <musparse.h>
#include "musa/base/musparse_compat.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace musa {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType, typename IndexType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float, int32> : std::true_type {};

template <>
struct is_supported<double, int32> : std::true_type {};

template <>
struct is_supported<std::complex<float>, int32> : std::true_type {};

template <>
struct is_supported<std::complex<double>, int32> : std::true_type {};


template <typename ValueType>
inline void spmv_buffersize(musparseHandle_t handle, musparseOperation_t opA,
                            const ValueType* alpha,
                            const musparseSpMatDescr_t matA,
                            const musparseDnVecDescr_t vecX,
                            const ValueType* beta,
                            const musparseDnVecDescr_t vecY,
                            musparseSpMVAlg_t alg, size_type* bufferSize)
{
    constexpr auto value_type = musa_data_type<ValueType>();
    musparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY,
                            value_type, alg, bufferSize);
}

template <typename ValueType>
inline void spmv(musparseHandle_t handle, musparseOperation_t opA,
                 const ValueType* alpha, const musparseSpMatDescr_t matA,
                 const musparseDnVecDescr_t vecX, const ValueType* beta,
                 const musparseDnVecDescr_t vecY, musparseSpMVAlg_t alg,
                 void* externalBuffer)
{
    constexpr auto value_type = musa_data_type<ValueType>();
    musparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, value_type, alg,
                 externalBuffer);
}


template <typename ValueType>
inline void spmm_buffersize(musparseHandle_t handle, musparseOperation_t opB,
                            musparseOperation_t opA, const ValueType* alpha,
                            const musparseSpMatDescr_t matA,
                            const musparseDnMatDescr_t vecX,
                            const ValueType* beta,
                            const musparseDnMatDescr_t vecY,
                            musparseSpMMAlg_t alg, size_type* bufferSize)
{
    constexpr auto value_type = musa_data_type<ValueType>();
    musparseSpMM_bufferSize(handle, opA, opB, alpha, matA, vecX, beta, vecY,
                            value_type, alg, bufferSize);
}

template <typename ValueType>
inline void spmm(musparseHandle_t handle, musparseOperation_t opA,
                 musparseOperation_t opB, const ValueType* alpha,
                 const musparseSpMatDescr_t matA,
                 const musparseDnMatDescr_t vecX, const ValueType* beta,
                 const musparseDnMatDescr_t vecY, musparseSpMMAlg_t alg,
                 void* externalBuffer)
{
    constexpr auto value_type = musa_data_type<ValueType>();
    musparseSpMM(handle, opA, opB, alpha, matA, vecX, beta, vecY, value_type,
                 alg, externalBuffer);
}


#if defined(CUDA_VERSION) && (CUDA_VERSION < 11021)


template <typename ValueType, typename IndexType>
inline void spmv(musparseHandle_t handle, musparseAlgMode_t alg,
                 musparseOperation_t transA, IndexType m, IndexType n,
                 IndexType nnz, const ValueType* alpha,
                 const musparseMatDescr_t descrA, const ValueType* csrValA,
                 const IndexType* csrRowPtrA, const IndexType* csrColIndA,
                 const ValueType* x, const ValueType* beta, ValueType* y,
                 void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV(ValueType)                                      \
    template <>                                                                \
    inline void spmv<ValueType, int32>(                                        \
        musparseHandle_t handle, musparseAlgMode_t alg,                        \
        musparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType* alpha, const musparseMatDescr_t descrA,               \
        const ValueType* csrValA, const int32* csrRowPtrA,                     \
        const int32* csrColIndA, const ValueType* x, const ValueType* beta,    \
        ValueType* y, void* buffer)                                            \
    {                                                                          \
        auto data_type = gko::kernels::musa::musa_data_type<ValueType>();      \
        if (data_type == MUSA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCsrmvEx(                         \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, buffer));                                 \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV(float);
GKO_BIND_CUSPARSE_SPMV(double);
GKO_BIND_CUSPARSE_SPMV(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV


template <typename ValueType, typename IndexType>
inline void spmv_buffersize(musparseHandle_t handle, musparseAlgMode_t alg,
                            musparseOperation_t transA, IndexType m,
                            IndexType n, IndexType nnz, const ValueType* alpha,
                            const musparseMatDescr_t descrA,
                            const ValueType* csrValA,
                            const IndexType* csrRowPtrA,
                            const IndexType* csrColIndA, const ValueType* x,
                            const ValueType* beta, ValueType* y,
                            size_type* bufferSizeInBytes) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(ValueType)                           \
    template <>                                                                \
    inline void spmv_buffersize<ValueType, int32>(                             \
        musparseHandle_t handle, musparseAlgMode_t alg,                        \
        musparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType* alpha, const musparseMatDescr_t descrA,               \
        const ValueType* csrValA, const int32* csrRowPtrA,                     \
        const int32* csrColIndA, const ValueType* x, const ValueType* beta,    \
        ValueType* y, size_type* bufferSizeInBytes)                            \
    {                                                                          \
        auto data_type = gko::kernels::musa::musa_data_type<ValueType>();      \
        if (data_type == MUSA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCsrmvEx_bufferSize(              \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, bufferSizeInBytes));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(float);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(double);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION < 11021)


template <typename ValueType>
void spgemm_work_estimation(musparseHandle_t handle, const ValueType* alpha,
                            musparseSpMatDescr_t a_descr,
                            musparseSpMatDescr_t b_descr, const ValueType* beta,
                            musparseSpMatDescr_t c_descr,
                            musparseSpGEMMDescr_t spgemm_descr,
                            size_type& buffer1_size, void* buffer1)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpGEMM_workEstimation(
        handle, MUSPARSE_OPERATION_NON_TRANSPOSE,
        MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr, b_descr, beta,
        c_descr, musa_data_type<ValueType>(), MUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr, &buffer1_size, buffer1));
}


template <typename ValueType>
void spgemm_compute(musparseHandle_t handle, const ValueType* alpha,
                    musparseSpMatDescr_t a_descr, musparseSpMatDescr_t b_descr,
                    const ValueType* beta, musparseSpMatDescr_t c_descr,
                    musparseSpGEMMDescr_t spgemm_descr, void* buffer1,
                    size_type& buffer2_size, void* buffer2)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpGEMM_compute(
        handle, MUSPARSE_OPERATION_NON_TRANSPOSE,
        MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr, b_descr, beta,
        c_descr, musa_data_type<ValueType>(), MUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr, &buffer2_size, buffer2));
}


template <typename ValueType>
void spgemm_copy(musparseHandle_t handle, const ValueType* alpha,
                 musparseSpMatDescr_t a_descr, musparseSpMatDescr_t b_descr,
                 const ValueType* beta, musparseSpMatDescr_t c_descr,
                 musparseSpGEMMDescr_t spgemm_descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSpGEMM_copy(handle, MUSPARSE_OPERATION_NON_TRANSPOSE,
                            MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr,
                            b_descr, beta, c_descr, musa_data_type<ValueType>(),
                            MUSPARSE_SPGEMM_DEFAULT, spgemm_descr));
}


inline size_type sparse_matrix_nnz(musparseSpMatDescr_t descr)
{
    int64 dummy1{};
    int64 dummy2{};
    int64 nnz{};
    musparseSpMatGetSize(descr, &dummy1, &dummy2, &nnz);
    return static_cast<size_type>(nnz);
}


template <typename ValueType, typename IndexType>
void csr_set_pointers(musparseSpMatDescr_t descr, IndexType* row_ptrs,
                      IndexType* col_idxs, ValueType* vals)
{
    musparseCsrSetPointers(descr, row_ptrs, col_idxs, vals);
}


template <typename ValueType, typename IndexType>
inline void transpose_buffersize(
    musparseHandle_t handle, size_type m, size_type n, size_type nnz,
    const ValueType* OrigValA, const IndexType* OrigRowPtrA,
    const IndexType* OrigColIndA, ValueType* TransValA, IndexType* TransRowPtrA,
    IndexType* TransColIndA, musaDataType_t valType,
    musparseAction_t copyValues, musparseIndexBase_t idxBase,
    musparseCsr2CscAlg_t alg, size_type* buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(ValueType)                   \
    template <>                                                               \
    inline void transpose_buffersize<ValueType, int32>(                       \
        musparseHandle_t handle, size_type m, size_type n, size_type nnz,     \
        const ValueType* OrigValA, const int32* OrigRowPtrA,                  \
        const int32* OrigColIndA, ValueType* TransValA, int32* TransRowPtrA,  \
        int32* TransColIndA, musaDataType_t valType,                          \
        musparseAction_t copyValues, musparseIndexBase_t idxBase,             \
        musparseCsr2CscAlg_t alg, size_type* buffer_size)                     \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCsr2cscEx2_bufferSize(          \
            handle, m, n, nnz, OrigValA, OrigRowPtrA, OrigColIndA, TransValA, \
            TransRowPtrA, TransColIndA, valType, copyValues, idxBase, alg,    \
            buffer_size));                                                    \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(float);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(double);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(std::complex<float>);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(std::complex<double>);

template <typename ValueType, typename IndexType>
inline void transpose(musparseHandle_t handle, size_type m, size_type n,
                      size_type nnz, const ValueType* OrigValA,
                      const IndexType* OrigRowPtrA,
                      const IndexType* OrigColIndA, ValueType* TransValA,
                      IndexType* TransRowPtrA, IndexType* TransColIndA,
                      musaDataType_t valType, musparseAction_t copyValues,
                      musparseIndexBase_t idxBase, musparseCsr2CscAlg_t alg,
                      void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_TRANSPOSE32(ValueType)                              \
    template <>                                                               \
    inline void transpose<ValueType, int32>(                                  \
        musparseHandle_t handle, size_type m, size_type n, size_type nnz,     \
        const ValueType* OrigValA, const int32* OrigRowPtrA,                  \
        const int32* OrigColIndA, ValueType* TransValA, int32* TransRowPtrA,  \
        int32* TransColIndA, musaDataType_t valType,                          \
        musparseAction_t copyValues, musparseIndexBase_t idxBase,             \
        musparseCsr2CscAlg_t alg, void* buffer)                               \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCsr2cscEx2(                     \
            handle, m, n, nnz, OrigValA, OrigRowPtrA, OrigColIndA, TransValA, \
            TransRowPtrA, TransColIndA, valType, copyValues, idxBase, alg,    \
            buffer));                                                         \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_TRANSPOSE32(float);
GKO_BIND_CUSPARSE_TRANSPOSE32(double);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<float>);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<double>);


inline musparseMatDescr_t create_mat_descr()
{
    musparseMatDescr_t descr{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatDescr(&descr));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSetMatIndexBase(descr, MUSPARSE_INDEX_BASE_ZERO));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSetMatType(descr, MUSPARSE_MATRIX_TYPE_GENERAL));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSetMatDiagType(descr, MUSPARSE_DIAG_TYPE_NON_UNIT));
    return descr;
}


inline void set_mat_fill_mode(musparseMatDescr_t descr,
                              musparseFillMode_t fill_mode)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSetMatFillMode(descr, fill_mode));
}


inline void set_mat_diag_type(musparseMatDescr_t descr,
                              musparseDiagType_t diag_type)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSetMatDiagType(descr, diag_type));
}


inline void destroy(musparseMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyMatDescr(descr));
}


inline musparseSpGEMMDescr_t create_spgemm_descr()
{
    musparseSpGEMMDescr_t descr{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpGEMM_createDescr(&descr));
    return descr;
}


inline void destroy(musparseSpGEMMDescr_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpGEMM_destroyDescr(info));
}


template <typename ValueType>
inline musparseDnVecDescr_t create_dnvec(int64 size, ValueType* values)
{
    musparseDnVecDescr_t descr{};
    constexpr auto value_type = musa_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseCreateDnVec(&descr, size, values, value_type));
    return descr;
}


template <typename ValueType>
inline musparseDnMatDescr_t create_dnmat(gko::dim<2> size, size_type stride,
                                         ValueType* values)
{
    musparseDnMatDescr_t descr{};
    constexpr auto value_type = musa_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseCreateDnMat(&descr, size[0], size[1], stride, values,
                            value_type, MUSPARSE_ORDER_ROW));
    return descr;
}


inline void destroy(musparseDnVecDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyDnVec(descr));
}


inline void destroy(musparseDnMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyDnMat(descr));
}


template <typename ValueType, typename IndexType>
inline musparseSpVecDescr_t create_spvec(int64 size, int64 nnz,
                                         IndexType* indices, ValueType* values)
{
    musparseSpVecDescr_t descr{};
    constexpr auto index_type = cusparse_index_type<IndexType>();
    constexpr auto value_type = musa_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseCreateSpVec(&descr, size, nnz, indices, values, index_type,
                            MUSPARSE_INDEX_BASE_ZERO, value_type));
    return descr;
}


inline void destroy(musparseSpVecDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroySpVec(descr));
}


template <typename IndexType, typename ValueType>
inline musparseSpMatDescr_t create_csr(int64 rows, int64 cols, int64 nnz,
                                       IndexType* csrRowOffsets,
                                       IndexType* csrColInd,
                                       ValueType* csrValues)
{
    musparseSpMatDescr_t descr{};
    constexpr auto index_type = cusparse_index_type<IndexType>();
    constexpr auto value_type = musa_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateCsr(
        &descr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
        index_type, index_type, MUSPARSE_INDEX_BASE_ZERO, value_type));
    return descr;
}


inline void destroy(musparseSpMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroySpMat(descr));
}


#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11031)


template <typename AttribType>
inline void set_attribute(musparseSpMatDescr_t desc,
                          musparseSpMatAttribute_t attr, AttribType val)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        musparseSpMatSetAttribute(desc, attr, &val, sizeof(val)));
}


inline musparseSpSMDescr_t create_spsm_descr()
{
    musparseSpSMDescr_t desc{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpSM_createDescr(&desc));
    return desc;
}


inline void destroy(musparseSpSMDescr_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpSM_destroyDescr(info));
}


#endif  // CUDA_VERSION >= 11031


#if defined(CUDA_VERSION) && (CUDA_VERSION < 11031)


inline csrsm2Info_t create_solve_info()
{
    csrsm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatInfo(&info));
    return info;
}


inline void destroy(csrsm2Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyMatInfo(info));
}


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION < 11031)


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
inline musparseMatInfo_t create_ilu0_info()
{
    musparseMatInfo_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatInfo(&info));
    return info;
}


inline void destroy_ilu0_info(musparseMatInfo_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyMatInfo(info));
}


inline csric02Info_t create_ic0_info()
{
    csric02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatInfo(&info));
    return info;
}


inline void destroy_ic0_info(csric02Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseDestroyMatInfo(info));
}
GKO_END_DISABLE_DEPRECATION_WARNINGS


#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11031))


#define GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, CusparseName)            \
    inline void buffer_size_ext(                                              \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,        \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const musparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, const ValueType* rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,               \
        size_type* factor_work_size)                                          \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_size));             \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, CusparseName)           \
    inline void buffer_size_ext(                                             \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,       \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const musparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, const ValueType* rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,              \
        size_type* factor_work_size) GKO_NOT_IMPLEMENTED;                    \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(float, musparseScsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(double, musparseDcsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<float>,
                                  musparseCcsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<double>,
                                  musparseZcsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(float, musparseScsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(double, musparseDcsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<float>,
                                  musparseCcsrsm_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<double>,
                                  musparseZcsrsm_bufferSizeExt);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BUFFERSIZEEXT
#undef GKO_BIND_CUSPARSE64_BUFFERSIZEEXT


#define GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, CusparseName)          \
    inline void csrsm2_analysis(                                              \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,        \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const musparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, const ValueType* rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,               \
        void* factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_vec));              \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, CusparseName)         \
    inline void csrsm2_analysis(                                             \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,       \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const musparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, const ValueType* rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,              \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(float, musparseScsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(double, musparseDcsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<float>,
                                    musparseCcsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<double>,
                                    musparseZcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(float, musparseScsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(double, musparseDcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<float>,
                                    musparseCcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<double>,
                                    musparseZcsrsm_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS
#undef GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS


#define GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, CusparseName)             \
    inline void csrsm2_solve(                                                 \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,        \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const musparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, ValueType* rhs, int32 sol_stride,             \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,               \
        void* factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs),           \
                         sol_stride, factor_info, policy, factor_work_vec));  \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void csrsm2_solve(                                                \
        musparseHandle_t handle, int algo, musparseOperation_t trans1,       \
        musparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const musparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, ValueType* rhs, int64 sol_stride,            \
        csrsm2Info_t factor_info, musparseSolvePolicy_t policy,              \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(float, musparseScsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(double, musparseDcsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<float>, musparseCcsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<double>, musparseZcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(float, musparseScsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(double, musparseDcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<float>, musparseCcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<double>, musparseZcsrsm_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_SOLVE
#undef GKO_BIND_CUSPARSE64_CSRSM2_SOLVE


#else  // if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename ValueType>
size_type spsm_buffer_size(musparseHandle_t handle, musparseOperation_t op_a,
                           musparseOperation_t op_b, ValueType alpha,
                           musparseSpMatDescr_t descr_a,
                           musparseDnMatDescr_t descr_b,
                           musparseDnMatDescr_t descr_c, musparseSpSMAlg_t algo,
                           musparseSpSMDescr_t spsm_descr)
{
    size_type work_size;
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpSM_bufferSize(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        musa_data_type<ValueType>(), algo, spsm_descr, &work_size));
    return work_size;
}


template <typename ValueType>
void spsm_analysis(musparseHandle_t handle, musparseOperation_t op_a,
                   musparseOperation_t op_b, ValueType alpha,
                   musparseSpMatDescr_t descr_a, musparseDnMatDescr_t descr_b,
                   musparseDnMatDescr_t descr_c, musparseSpSMAlg_t algo,
                   musparseSpSMDescr_t spsm_descr, void* work)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpSM_analysis(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        musa_data_type<ValueType>(), algo, spsm_descr, work));
}


template <typename ValueType>
void spsm_solve(musparseHandle_t handle, musparseOperation_t op_a,
                musparseOperation_t op_b, ValueType alpha,
                musparseSpMatDescr_t descr_a, musparseDnMatDescr_t descr_b,
                musparseDnMatDescr_t descr_c, musparseSpSMAlg_t algo,
                musparseSpSMDescr_t spsm_descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseSpSM_solve(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        musa_data_type<ValueType>(), algo, spsm_descr));
}


#endif  // (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename IndexType>
void csrsort_buffer_size(musparseHandle_t handle, IndexType m, IndexType n,
                         IndexType nnz, const IndexType* row_ptrs,
                         const IndexType* col_idxs,
                         size_type& buffer_size) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort_buffer_size<int32>(musparseHandle_t handle, int32 m,
                                       int32 n, int32 nnz,
                                       const int32* row_ptrs,
                                       const int32* col_idxs,
                                       size_type& buffer_size)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, row_ptrs, col_idxs, &buffer_size));
}


template <typename IndexType>
void csrsort(musparseHandle_t handle, IndexType m, IndexType n, IndexType nnz,
             const musparseMatDescr_t descr, const IndexType* row_ptrs,
             IndexType* col_idxs, IndexType* permutation,
             void* buffer) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort<int32>(musparseHandle_t handle, int32 m, int32 n, int32 nnz,
                           const musparseMatDescr_t descr,
                           const int32* row_ptrs, int32* col_idxs,
                           int32* permutation, void* buffer)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseXcsrsort(
        handle, m, n, nnz, descr, row_ptrs, col_idxs, permutation, buffer));
}


inline void gather(musparseHandle_t handle, musparseDnVecDescr_t in,
                   musparseSpVecDescr_t out)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseGather(handle, in, out));
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
template <typename ValueType, typename IndexType>
void ilu0_buffer_size(musparseHandle_t handle, IndexType m, IndexType nnz,
                      const musparseMatDescr_t descr, const ValueType* vals,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      musparseMatInfo_t info,
                      size_type& buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(ValueType, CusparseName)          \
    template <>                                                              \
    inline void ilu0_buffer_size<ValueType, int32>(                          \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, musparseMatInfo_t info,   \
        size_type& buffer_size)                                              \
    {                                                                        \
        size_t tmp_buffer_size{};                                            \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType*>(vals)),       \
                         row_ptrs, col_idxs, info, &tmp_buffer_size));       \
        buffer_size = tmp_buffer_size;                                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(float, musparseScsrilu0_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(double, musparseDcsrilu0_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<float>,
                                   musparseCcsrilu0_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<double>,
                                   musparseZcsrilu0_bufferSize);

#undef GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
void ilu0_analysis(musparseHandle_t handle, IndexType m, IndexType nnz,
                   const musparseMatDescr_t descr, const ValueType* vals,
                   const IndexType* row_ptrs, const IndexType* col_idxs,
                   musparseMatInfo_t info, musparseSolvePolicy_t policy,
                   void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_ANALYSIS(ValueType, CusparseName)             \
    template <>                                                              \
    inline void ilu0_analysis<ValueType, int32>(                             \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, musparseMatInfo_t info,   \
        musparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType*>(vals)),       \
                         row_ptrs, col_idxs, info,                           \
                         MUSPARSE_ANALYSIS_POLICY_FORCE, policy, buffer));   \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_ANALYSIS(float, musparseScsrilu0_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(double, musparseDcsrilu0_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<float>,
                                musparseCcsrilu0_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<double>,
                                musparseZcsrilu0_analysis);

#undef GKO_BIND_CUSPARSE_ILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void ilu0(musparseHandle_t handle, IndexType m, IndexType nnz,
          const musparseMatDescr_t descr, ValueType* vals,
          const IndexType* row_ptrs, const IndexType* col_idxs,
          musparseMatInfo_t info, musparseSolvePolicy_t policy,
          void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0(ValueType, CusparseName)                      \
    template <>                                                              \
    inline void ilu0<ValueType, int32>(                                      \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, ValueType* vals,                     \
        const int32* row_ptrs, const int32* col_idxs, musparseMatInfo_t info,   \
        musparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0(float, musparseScsrilu0);
GKO_BIND_CUSPARSE_ILU0(double, musparseDcsrilu0);
GKO_BIND_CUSPARSE_ILU0(std::complex<float>, musparseCcsrilu0);
GKO_BIND_CUSPARSE_ILU0(std::complex<double>, musparseZcsrilu0);

#undef GKO_BIND_CUSPARSE_ILU0


template <typename ValueType, typename IndexType>
void ic0_buffer_size(musparseHandle_t handle, IndexType m, IndexType nnz,
                     const musparseMatDescr_t descr, const ValueType* vals,
                     const IndexType* row_ptrs, const IndexType* col_idxs,
                     csric02Info_t info,
                     size_type& buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(ValueType, CusparseName)           \
    template <>                                                              \
    inline void ic0_buffer_size<ValueType, int32>(                           \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        size_type& buffer_size)                                              \
    {                                                                        \
        size_t tmp_buffer_size{};                                            \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType*>(vals)),       \
                         row_ptrs, col_idxs, info, &tmp_buffer_size));       \
        buffer_size = tmp_buffer_size;                                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(float, musparseScsric0_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(double, musparseDcsric0_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(std::complex<float>,
                                  musparseCcsric0_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(std::complex<double>,
                                  musparseZcsric0_bufferSize);

#undef GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
void ic0_analysis(musparseHandle_t handle, IndexType m, IndexType nnz,
                  const musparseMatDescr_t descr, const ValueType* vals,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  csric02Info_t info, musparseSolvePolicy_t policy,
                  void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0_ANALYSIS(ValueType, CusparseName)              \
    template <>                                                              \
    inline void ic0_analysis<ValueType, int32>(                              \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        musparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0_ANALYSIS(float, musparseScsric0_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(double, musparseDcsric0_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(std::complex<float>, musparseCcsric0_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(std::complex<double>, musparseZcsric0_analysis);

#undef GKO_BIND_CUSPARSE_ILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void ic0(musparseHandle_t handle, IndexType m, IndexType nnz,
         const musparseMatDescr_t descr, ValueType* vals,
         const IndexType* row_ptrs, const IndexType* col_idxs,
         csric02Info_t info, musparseSolvePolicy_t policy,
         void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0(ValueType, CusparseName)                       \
    template <>                                                              \
    inline void ic0<ValueType, int32>(                                       \
        musparseHandle_t handle, int32 m, int32 nnz,                         \
        const musparseMatDescr_t descr, ValueType* vals,                     \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        musparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0(float, musparseScsric0);
GKO_BIND_CUSPARSE_IC0(double, musparseDcsric0);
GKO_BIND_CUSPARSE_IC0(std::complex<float>, musparseCcsric0);
GKO_BIND_CUSPARSE_IC0(std::complex<double>, musparseZcsric0);
GKO_END_DISABLE_DEPRECATION_WARNINGS

#undef GKO_BIND_CUSPARSE_IC0


}  // namespace cusparse


namespace sparselib {


using namespace cusparse;


#define SPARSELIB_OPERATION_TRANSPOSE MUSPARSE_OPERATION_TRANSPOSE
#define SPARSELIB_OPERATION_NON_TRANSPOSE MUSPARSE_OPERATION_NON_TRANSPOSE
#define SPARSELIB_SOLVE_POLICY_USE_LEVEL MUSPARSE_SOLVE_POLICY_USE_LEVEL


}  // namespace sparselib
}  // namespace musa
}  // namespace kernels
}  // namespace gko

#ifdef GKO_COMPILING_MUSA
using cusparseOperation_t = musparseOperation_t;
using cusparseSpMMAlg_t = musparseSpMMAlg_t;
using cusparseAction_t = musparseAction_t;
using cusparseIndexBase_t = musparseIndexBase_t;
using cusparseCsr2CscAlg_t = musparseCsr2CscAlg_t;
using cudaDataType_t = musaDataType_t;
#define CUSPARSE_CSRMV_ALG1 MUSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMM_CSR_ALG2 MUSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_ACTION_NUMERIC MUSPARSE_ACTION_NUMERIC
#define CUSPARSE_INDEX_BASE_ZERO MUSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_CSR2CSC_ALG1 MUSPARSE_CSR2CSC_ALG1
namespace gko {
namespace kernels {
namespace musa {
template <typename T>
constexpr musaDataType_t cuda_data_type()
{
    return musa_data_type<T>();
}
}  // namespace musa
}  // namespace kernels
}  // namespace gko
#endif


#endif  // GKO_MUSA_BASE_CUSPARSE_BINDINGS_HPP_
