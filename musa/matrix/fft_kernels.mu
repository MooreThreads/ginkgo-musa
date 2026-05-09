// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fft_kernels.hpp"

#include <array>

#include <mufft.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace musa {
/**
 * @brief The FFT matrix format namespace.
 * @ref Fft
 * @ingroup fft
 */
namespace fft {


template <typename InValueType, typename OutValueType>
struct cufft_type_impl {};

template <>
struct cufft_type_impl<std::complex<float>, std::complex<float>> {
    constexpr static auto value = MUFFT_C2C;
};

template <>
struct cufft_type_impl<std::complex<double>, std::complex<double>> {
    constexpr static auto value = MUFFT_Z2Z;
};


class cufft_handle {
    struct cufft_deleter {
        void operator()(mufftHandle* ptr)
        {
            auto data = *ptr;
            delete ptr;
            mufftDestroy(data);
        }
    };

public:
    operator mufftHandle() const { return *handle_; }

    cufft_handle(musaStream_t stream) : handle_{new mufftHandle{}}
    {
        GKO_ASSERT_NO_CUFFT_ERRORS(mufftCreate(handle_.get()));
        GKO_ASSERT_NO_CUFFT_ERRORS(mufftSetStream(*handle_, stream));
    }

    template <int d, typename InValueType, typename OutValueType>
    void setup(std::array<size_type, d> fft_size, size_type in_batch_stride,
               size_type out_batch_stride, size_type batch_count,
               array<char>& work_area)
    {
        static_assert(d == 1 || d == 2 || d == 3,
                      "Only 1D, 2D or 3D FFT supported");
        std::array<int, d> cast_fft_size;
        for (int i = 0; i < d; i++) {
            cast_fft_size[i] = static_cast<int>(fft_size[i]);
        }
        size_type work_size{};
        GKO_ASSERT_NO_CUFFT_ERRORS(mufftSetAutoAllocation(*handle_, false));
        GKO_ASSERT_NO_CUFFT_ERRORS(mufftMakePlanMany(
            *handle_, d, cast_fft_size.data(), cast_fft_size.data(),
            static_cast<int>(in_batch_stride), 1, cast_fft_size.data(),
            static_cast<int>(out_batch_stride), 1,
            cufft_type_impl<InValueType, OutValueType>::value,
            static_cast<int>(batch_count), &work_size));
        work_area.resize_and_reset(work_size);
        GKO_ASSERT_NO_CUFFT_ERRORS(
            mufftSetWorkArea(*handle_, work_area.get_data()));
    }


    void execute(const std::complex<float>* in, std::complex<float>* out,
                 bool imterse)
    {
        mufftExecC2C(*handle_,
                     const_cast<mufftComplex*>(
                         reinterpret_cast<const mufftComplex*>(in)),
                     reinterpret_cast<mufftComplex*>(out),
                     imterse ? MUFFT_INVERSE : MUFFT_FORWARD);
    }

    void execute(const std::complex<double>* in, std::complex<double>* out,
                 bool imterse)
    {
        mufftExecZ2Z(*handle_,
                     const_cast<mufftDoubleComplex*>(
                         reinterpret_cast<const mufftDoubleComplex*>(in)),
                     reinterpret_cast<mufftDoubleComplex*>(out),
                     imterse ? MUFFT_INVERSE : MUFFT_FORWARD);
    }

private:
    std::unique_ptr<mufftHandle, cufft_deleter> handle_;
};


template <typename ValueType>
void fft(std::shared_ptr<const DefaultExecutor> exec,
         matrix::view::dense<const std::complex<ValueType>> b,
         matrix::view::dense<std::complex<ValueType>> x, bool imterse,
         array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<1, std::complex<ValueType>, std::complex<ValueType>>(
        {b.size[0]}, b.stride, x.stride, b.size[1], buffer);
    handle.execute(b.values, x.values, imterse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(GKO_DECLARE_FFT_KERNEL);


template <typename ValueType>
void fft2(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<const std::complex<ValueType>> b,
          matrix::view::dense<std::complex<ValueType>> x, size_type size1,
          size_type size2, bool imterse, array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<2, std::complex<ValueType>, std::complex<ValueType>>(
        {size1, size2}, b.stride, x.stride, b.size[1], buffer);
    handle.execute(b.values, x.values, imterse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(GKO_DECLARE_FFT2_KERNEL);


template <typename ValueType>
void fft3(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<const std::complex<ValueType>> b,
          matrix::view::dense<std::complex<ValueType>> x, size_type size1,
          size_type size2, size_type size3, bool imterse, array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<3, std::complex<ValueType>, std::complex<ValueType>>(
        {size1, size2, size3}, b.stride, x.stride, b.size[1], buffer);
    handle.execute(b.values, x.values, imterse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft
}  // namespace musa
}  // namespace kernels
}  // namespace gko
