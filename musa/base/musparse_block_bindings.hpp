// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_CUSPARSE_BLOCK_BINDINGS_HPP_
#define GKO_MUSA_BASE_CUSPARSE_BLOCK_BINDINGS_HPP_


#include <musa.h>
#include <musparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/types.hpp"
#include "musa/base/musparse_bindings.hpp"


namespace gko {
namespace kernels {
namespace musa {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {


/// Default storage layout within each small dense block
constexpr musparseDirection_t blockDir = MUSPARSE_DIRECTION_COLUMN;


#define GKO_BIND_CUSPARSE32_BSRMV(ValueType, CusparseName)                    \
    inline void bsrmv(musparseHandle_t handle, musparseOperation_t transA,    \
                      int32 mb, int32 nb, int32 nnzb, const ValueType* alpha, \
                      const musparseMatDescr_t descrA, const ValueType* valA, \
                      const int32* rowPtrA, const int32* colIndA,             \
                      int block_size, const ValueType* x,                     \
                      const ValueType* beta, ValueType* y)                    \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                           \
            handle, blockDir, transA, mb, nb, nnzb, as_culibs_type(alpha),    \
            descrA, as_culibs_type(valA), rowPtrA, colIndA, block_size,       \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRMV(ValueType, CusparseName)                    \
    inline void bsrmv(musparseHandle_t handle, musparseOperation_t transA,    \
                      int64 mb, int64 nb, int64 nnzb, const ValueType* alpha, \
                      const musparseMatDescr_t descrA, const ValueType* valA, \
                      const int64* rowPtrA, const int64* colIndA,             \
                      int block_size, const ValueType* x,                     \
                      const ValueType* beta, ValueType* y)                    \
        GKO_NOT_IMPLEMENTED;                                                  \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRMV(float, musparseSbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(double, musparseDbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(std::complex<float>, musparseCbsrmv);
GKO_BIND_CUSPARSE32_BSRMV(std::complex<double>, musparseZbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(float, musparseSbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(double, musparseDbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(std::complex<float>, musparseCbsrmv);
GKO_BIND_CUSPARSE64_BSRMV(std::complex<double>, musparseZbsrmv);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRMV(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_BSRMV
#undef GKO_BIND_CUSPARSE64_BSRMV


#define GKO_BIND_CUSPARSE32_BSRMM(ValueType, CusparseName)                     \
    inline void bsrmm(musparseHandle_t handle, musparseOperation_t transA,     \
                      musparseOperation_t transB, int32 mb, int32 n, int32 kb, \
                      int32 nnzb, const ValueType* alpha,                      \
                      const musparseMatDescr_t descrA, const ValueType* valA,  \
                      const int32* rowPtrA, const int32* colIndA,              \
                      int block_size, const ValueType* B, int32 ldb,           \
                      const ValueType* beta, ValueType* C, int32 ldc)          \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, transA, transB, mb, n, kb, nnzb,    \
                         as_culibs_type(alpha), descrA, as_culibs_type(valA),  \
                         rowPtrA, colIndA, block_size, as_culibs_type(B), ldb, \
                         as_culibs_type(beta), as_culibs_type(C), ldc));       \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRMM(ValueType, CusparseName)                    \
    inline void bsrmm(                                                        \
        musparseHandle_t handle, musparseOperation_t transA,                  \
        musparseOperation_t transB, int64 mb, int64 n, int64 kb, int64 nnzb,  \
        const ValueType* alpha, const musparseMatDescr_t descrA,              \
        const ValueType* valA, const int64* rowPtrA, const int64* colIndA,    \
        int block_size, const ValueType* B, int64 ldb, const ValueType* beta, \
        ValueType* C, int64 ldc) GKO_NOT_IMPLEMENTED;                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRMM(float, musparseSbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(double, musparseDbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(std::complex<float>, musparseCbsrmm);
GKO_BIND_CUSPARSE32_BSRMM(std::complex<double>, musparseZbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(float, musparseSbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(double, musparseDbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(std::complex<float>, musparseCbsrmm);
GKO_BIND_CUSPARSE64_BSRMM(std::complex<double>, musparseZbsrmm);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRMM(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRMM(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_BSRMM
#undef GKO_BIND_CUSPARSE64_BSRMM


template <typename ValueType, typename IndexType>
inline int bsr_transpose_buffersize(musparseHandle_t handle, IndexType mb,
                                    IndexType nb, IndexType nnzb,
                                    const ValueType* origValA,
                                    const IndexType* origRowPtrA,
                                    const IndexType* origColIndA,
                                    int rowblocksize,
                                    int colblocksize) GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
inline void bsr_transpose(musparseHandle_t handle, IndexType mb, IndexType nb,
                          IndexType nnzb, const ValueType* origValA,
                          const IndexType* origRowPtrA,
                          const IndexType* origColIndA, int rowblocksize,
                          int colblocksize, ValueType* TransValA,
                          IndexType* transRowIndA, IndexType* transColPtrA,
                          musparseAction_t copyValues,
                          musparseIndexBase_t idxBase,
                          void* pBuffer) GKO_NOT_IMPLEMENTED;

// cuSparse does not transpose the blocks themselves,
//  only the sparsity pattern
#define GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(ValueType, CusparseName)           \
    template <>                                                                \
    inline int bsr_transpose_buffersize<ValueType, int32>(                     \
        musparseHandle_t handle, int32 mb, int32 nb, int32 nnzb,               \
        const ValueType* origValA, const int32* origRowPtrA,                   \
        const int32* origColIndA, int rowblocksize, int colblocksize)          \
    {                                                                          \
        int pBufferSize = -1;                                                  \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName##_bufferSize(               \
            handle, mb, nb, nnzb, as_culibs_type(origValA), origRowPtrA,       \
            origColIndA, rowblocksize, colblocksize, &pBufferSize));           \
        return pBufferSize;                                                    \
    }                                                                          \
    template <>                                                                \
    inline void bsr_transpose<ValueType, int32>(                               \
        musparseHandle_t handle, int32 mb, int32 nb, int32 nnzb,               \
        const ValueType* origValA, const int32* origRowPtrA,                   \
        const int32* origColIndA, int rowblocksize, int colblocksize,          \
        ValueType* transValA, int32* transRowIdxA, int32* transColPtrA,        \
        musparseAction_t copyValues, musparseIndexBase_t idxBase,              \
        void* pBuffer)                                                         \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, mb, nb, nnzb, as_culibs_type(origValA),       \
                         origRowPtrA, origColIndA, rowblocksize, colblocksize, \
                         as_culibs_type(transValA), transRowIdxA,              \
                         transColPtrA, copyValues, idxBase, pBuffer));         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(float, musparseSgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(double, musparseDgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(std::complex<float>, musparseCgebsr2gebsc);
GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32(std::complex<double>, musparseZgebsr2gebsc);

#undef GKO_BIND_CUSPARSE_BLOCK_TRANSPOSE32


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
inline std::unique_ptr<std::remove_pointer_t<bsrsm2Info_t>,
                       std::function<void(bsrsm2Info_t)>>
create_bsr_trsm_info()
{
    bsrsm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatInfo(&info));
    return {info, [](bsrsm2Info_t info) { musparseDestroyMatInfo(info); }};
}


inline std::unique_ptr<std::remove_pointer_t<bsrilu02Info_t>,
                       std::function<void(bsrilu02Info_t)>>
create_bilu0_info()
{
    bsrilu02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(musparseCreateMatInfo(&info));
    return {info,
            [](bsrilu02Info_t info) { musparseDestroyMatInfo(info); }};
}


#define GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(ValueType, CusparseName)          \
    inline int bsrsm2_buffer_size(                                             \
        musparseHandle_t handle, musparseOperation_t transA,                   \
        musparseOperation_t transX, int32 mb, int32 n, int32 nnzb,             \
        const musparseMatDescr_t descr, ValueType* val, const int32* rowPtr,   \
        const int32* colInd, int block_sz, bsrsm2Info_t factor_info)           \
    {                                                                          \
        int factor_work_size = -1;                                             \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, transA, transX, mb, n, nnzb, descr, \
                         as_culibs_type(val), rowPtr, colInd, block_sz,        \
                         factor_info, &factor_work_size));                     \
        return factor_work_size;                                               \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(ValueType, CusparseName)        \
    inline int64 bsrsm2_buffer_size(                                         \
        musparseHandle_t handle, musparseOperation_t transA,                 \
        musparseOperation_t transX, int64 mb, int64 n, int64 nnzb,           \
        const musparseMatDescr_t descr, ValueType* val, const int64* rowPtr, \
        const int64* colInd, int block_size, bsrsm2Info_t factor_info)       \
        GKO_NOT_IMPLEMENTED;                                                 \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(float, musparseSbsrsm_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(double, musparseDbsrsm_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(std::complex<float>,
                                     musparseCbsrsm_bufferSize);
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(std::complex<double>,
                                     musparseZbsrsm_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(float, musparseSbsrsm_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(double, musparseDbsrsm_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(std::complex<float>,
                                     musparseCbsrsm_bufferSize);
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(std::complex<double>,
                                     musparseZbsrsm_bufferSize);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM_BUFFERSIZE
#undef GKO_BIND_CUSPARSE64_BSRSM_BUFFERSIZE


#define GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(ValueType, CusparseName)           \
    inline void bsrsm2_analysis(                                               \
        musparseHandle_t handle, musparseOperation_t trans1,                   \
        musparseOperation_t trans2, int32 mb, int32 n, int32 nnzb,             \
        const musparseMatDescr_t descr, const ValueType* val,                  \
        const int32* rowPtr, const int32* colInd, int block_size,              \
        bsrsm2Info_t factor_info, musparseSolvePolicy_t policy,                \
        void* factor_work_vec)                                                 \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, blockDir, trans1, trans2, mb, n, nnzb, descr, \
                         as_culibs_type(val), rowPtr, colInd, block_size,      \
                         factor_info, policy, factor_work_vec));               \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(ValueType, CusparseName)           \
    inline void bsrsm2_analysis(                                               \
        musparseHandle_t handle, musparseOperation_t trans1,                   \
        musparseOperation_t trans2, size_type mb, size_type n, size_type nnzb, \
        const musparseMatDescr_t descr, const ValueType* val,                  \
        const int64* rowPtr, const int64* colInd, int block_size,              \
        bsrsm2Info_t factor_info, musparseSolvePolicy_t policy,                \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                            \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(float, musparseSbsrsm_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(double, musparseDbsrsm_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(std::complex<float>,
                                    musparseCbsrsm_analysis);
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(std::complex<double>,
                                    musparseZbsrsm_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(float, musparseSbsrsm_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(double, musparseDbsrsm_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(std::complex<float>,
                                    musparseCbsrsm_analysis);
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(std::complex<double>,
                                    musparseZbsrsm_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM2_ANALYSIS
#undef GKO_BIND_CUSPARSE64_BSRSM2_ANALYSIS


#define GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void bsrsm2_solve(                                                \
        musparseHandle_t handle, musparseOperation_t transA,                 \
        musparseOperation_t transX, int32 mb, int32 n, int32 nnzb,           \
        const ValueType* alpha, const musparseMatDescr_t descrA,             \
        const ValueType* valA, const int32* rowPtrA, const int32* colIndA,   \
        int blockSizeA, bsrsm2Info_t factor_info, const ValueType* b,        \
        int32 ldb, ValueType* x, int32 ldx, musparseSolvePolicy_t policy,    \
        void* factor_work_vec)                                               \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                          \
            handle, blockDir, transA, transX, mb, n, nnzb,                   \
            as_culibs_type(alpha), descrA, as_culibs_type(valA), rowPtrA,    \
            colIndA, blockSizeA, factor_info, as_culibs_type(b), ldb,        \
            as_culibs_type(x), ldx, policy, factor_work_vec));               \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(ValueType, CusparseName)              \
    inline void bsrsm2_solve(                                                  \
        musparseHandle_t handle, musparseOperation_t trans1,                   \
        musparseOperation_t trans2, size_type mb, size_type n, size_type nnzb, \
        const ValueType* alpha, const musparseMatDescr_t descr,                \
        const ValueType* val, const int64* rowPtr, const int64* colInd,        \
        int block_size, bsrsm2Info_t factor_info, const ValueType* b,          \
        int64 ldb, ValueType* x, int64 ldx, musparseSolvePolicy_t policy,      \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                            \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(float, musparseSbsrsm_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(double, musparseDbsrsm_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(std::complex<float>, musparseCbsrsm_solve);
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(std::complex<double>, musparseZbsrsm_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(float, musparseSbsrsm_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(double, musparseDbsrsm_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(std::complex<float>, musparseCbsrsm_solve);
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(std::complex<double>, musparseZbsrsm_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BSRSM2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BSRSM2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BSRSM2_SOLVE
#undef GKO_BIND_CUSPARSE64_BSRSM2_SOLVE


template <typename ValueType, typename IndexType>
int bilu0_buffer_size(musparseHandle_t handle, IndexType mb, IndexType nnzb,
                      const musparseMatDescr_t descr, const ValueType* vals,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      int block_sz, bsrilu02Info_t info) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(ValueType, CusparseName)          \
    template <>                                                               \
    inline int bilu0_buffer_size<ValueType, int32>(                           \
        musparseHandle_t handle, int32 mb, int32 nnzb,                        \
        const musparseMatDescr_t descr, const ValueType* vals,                \
        const int32* row_ptrs, const int32* col_idxs, int block_size,         \
        bsrilu02Info_t info)                                                  \
    {                                                                         \
        int tmp_buffer_sz{};                                                  \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                           \
            handle, blockDir, mb, nnzb, descr,                                \
            as_culibs_type(const_cast<ValueType*>(vals)), row_ptrs, col_idxs, \
            block_size, info, &tmp_buffer_sz));                               \
        return tmp_buffer_sz;                                                 \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(float, musparseSbsrilu0_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(double, musparseDbsrilu0_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(std::complex<float>,
                                    musparseCbsrilu0_bufferSize);
GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE(std::complex<double>,
                                    musparseZbsrilu0_bufferSize);

#undef GKO_BIND_CUSPARSE_BILU0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
inline void bilu0_analysis(musparseHandle_t handle, IndexType mb,
                           IndexType nnzb, const musparseMatDescr_t descr,
                           ValueType* vals, const IndexType* row_ptrs,
                           const IndexType* col_idxs, int block_size,
                           bsrilu02Info_t info, musparseSolvePolicy_t policy,
                           void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0_ANALYSIS(ValueType, CusparseName)              \
    template <>                                                                \
    inline void bilu0_analysis<ValueType, int32>(                              \
        musparseHandle_t handle, int32 mb, int32 nnzb,                         \
        const musparseMatDescr_t descr, ValueType* vals,                       \
        const int32* row_ptrs, const int32* col_idxs, int block_size,          \
        bsrilu02Info_t info, musparseSolvePolicy_t policy, void* buffer)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                            \
            handle, blockDir, mb, nnzb, descr, as_culibs_type(vals), row_ptrs, \
            col_idxs, block_size, info, policy, buffer));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0_ANALYSIS(float, musparseSbsrilu0_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(double, musparseDbsrilu0_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(std::complex<float>,
                                 musparseCbsrilu0_analysis);
GKO_BIND_CUSPARSE_BILU0_ANALYSIS(std::complex<double>,
                                 musparseZbsrilu0_analysis);

#undef GKO_BIND_CUSPARSE_BILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void bilu0(musparseHandle_t handle, IndexType mb, IndexType nnzb,
           const musparseMatDescr_t descr, ValueType* vals,
           const IndexType* row_ptrs, const IndexType* col_idxs, int block_size,
           bsrilu02Info_t info, musparseSolvePolicy_t policy,
           void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_BILU0(ValueType, CusparseName)                       \
    template <>                                                                \
    inline void bilu0<ValueType, int32>(                                       \
        musparseHandle_t handle, int32 mb, int32 nnzb,                         \
        const musparseMatDescr_t descr, ValueType* vals,                       \
        const int32* row_ptrs, const int32* col_idxs, int block_size,          \
        bsrilu02Info_t info, musparseSolvePolicy_t policy, void* buffer)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                            \
            handle, blockDir, mb, nnzb, descr, as_culibs_type(vals), row_ptrs, \
            col_idxs, block_size, info, policy, buffer));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_BILU0(float, musparseSbsrilu0);
GKO_BIND_CUSPARSE_BILU0(double, musparseDbsrilu0);
GKO_BIND_CUSPARSE_BILU0(std::complex<float>, musparseCbsrilu0);
GKO_BIND_CUSPARSE_BILU0(std::complex<double>, musparseZbsrilu0);

#undef GKO_BIND_CUSPARSE_BILU0
GKO_END_DISABLE_DEPRECATION_WARNINGS


}  // namespace cusparse
}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_CUSPARSE_BLOCK_BINDINGS_HPP_
