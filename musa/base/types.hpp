// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_TYPES_HPP_
#define GKO_MUSA_BASE_TYPES_HPP_


#include <type_traits>

#include <musa.h>
#include <mublas.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musparse.h>
#include <thrust/complex.h>

#include <ginkgo/core/base/bfloat16.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace musa {
namespace detail {

/**
 * @internal
 *
 * replacement for thrust::complex without alignment restrictions.
 */
template <typename T>
struct alignas(std::complex<T>) fake_complex {
    T real;
    T imag;

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex() : real{}, imag{} {}

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex(thrust::complex<T> val)
        : real{val.real()}, imag{val.imag()}
    {}

    friend GKO_INLINE GKO_ATTRIBUTES fake_complex operator+(fake_complex a,
                                                            fake_complex b)
    {
        fake_complex result{};
        result.real = a.real + b.real;
        result.imag = a.imag + b.imag;
        return result;
    }

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator==(fake_complex a,
                                                               fake_complex b)
    {
        return a.real == b.real && a.imag == b.imag;
    }

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator!=(fake_complex a,
                                                               fake_complex b)
    {
        return !(a == b);
    }
};


template <typename ValueType>
struct fake_complex_unpack_impl {
    using type = ValueType;

    GKO_INLINE GKO_ATTRIBUTES static constexpr ValueType unpack(ValueType v)
    {
        return v;
    }
};

template <typename ValueType>
struct fake_complex_unpack_impl<fake_complex<ValueType>> {
    using type = thrust::complex<ValueType>;

    GKO_INLINE GKO_ATTRIBUTES static constexpr thrust::complex<ValueType>
    unpack(fake_complex<ValueType> v)
    {
        return {v.real, v.imag};
    }
};


template <typename T>
struct culibs_type_impl {
    using type = T;
};

template <typename T>
struct culibs_type_impl<T*> {
    using type = typename culibs_type_impl<T>::type*;
};

template <typename T>
struct culibs_type_impl<T&> {
    using type = typename culibs_type_impl<T>::type&;
};

template <typename T>
struct culibs_type_impl<const T> {
    using type = const typename culibs_type_impl<T>::type;
};

template <typename T>
struct culibs_type_impl<volatile T> {
    using type = volatile typename culibs_type_impl<T>::type;
};

template <>
struct culibs_type_impl<std::complex<float>> {
    using type = muComplex;
};

template <>
struct culibs_type_impl<std::complex<double>> {
    using type = muDoubleComplex;
};

template <>
struct culibs_type_impl<half> {
    using type = __half;
};

template <>
struct culibs_type_impl<std::complex<half>> {
    using type = __half2;
};

template <>
struct culibs_type_impl<bfloat16> {
    using type = __mt_bfloat16;
};

template <>
struct culibs_type_impl<std::complex<bfloat16>> {
    using type = __mt_bfloat162;
};


template <typename T>
struct culibs_type_impl<thrust::complex<T>> {
    using type = typename culibs_type_impl<std::complex<T>>::type;
};

template <typename T>
struct musa_type_impl {
    using type = T;
};

template <typename T>
struct musa_type_impl<T*> {
    using type = typename musa_type_impl<T>::type*;
};

template <typename T>
struct musa_type_impl<T&> {
    using type = typename musa_type_impl<T>::type&;
};

template <typename T>
struct musa_type_impl<const T> {
    using type = const typename musa_type_impl<T>::type;
};

template <typename T>
struct musa_type_impl<volatile T> {
    using type = volatile typename musa_type_impl<T>::type;
};

template <>
struct musa_type_impl<half> {
    using type = __half;
};

template <>
struct musa_type_impl<bfloat16> {
    using type = __mt_bfloat16;
};

template <typename T>
struct musa_type_impl<std::complex<T>> {
    using type = thrust::complex<typename musa_type_impl<T>::type>;
};

template <>
struct musa_type_impl<muDoubleComplex> {
    using type = thrust::complex<double>;
};

template <>
struct musa_type_impl<muComplex> {
    using type = thrust::complex<float>;
};

template <>
struct musa_type_impl<__half2> {
    using type = thrust::complex<__half>;
};

template <>
struct musa_type_impl<__mt_bfloat162> {
    using type = thrust::complex<__mt_bfloat16>;
};

template <typename T>
struct musa_struct_member_type_impl {
    using type = T;
};

template <typename T>
struct musa_struct_member_type_impl<std::complex<T>> {
    using type = fake_complex<typename musa_struct_member_type_impl<T>::type>;
};

template <>
struct musa_struct_member_type_impl<gko::half> {
    using type = __half;
};

template <>
struct musa_struct_member_type_impl<gko::bfloat16> {
    using type = __mt_bfloat16;
};

template <typename ValueType, typename IndexType>
struct musa_type_impl<matrix_data_entry<ValueType, IndexType>> {
    using type = matrix_data_entry<
        typename musa_struct_member_type_impl<ValueType>::type, IndexType>;
};


template <typename T>
struct musa_data_type_impl {};

#define GKO_CUDA_DATA_TYPE(_type, _value)               \
    template <>                                         \
    struct musa_data_type_impl<_type> {                 \
        constexpr static musaDataType_t value = _value; \
    }

GKO_CUDA_DATA_TYPE(gko::half, MUSA_R_16F);
GKO_CUDA_DATA_TYPE(gko::bfloat16, MUSA_R_16BF);
GKO_CUDA_DATA_TYPE(float, MUSA_R_32F);
GKO_CUDA_DATA_TYPE(double, MUSA_R_64F);
GKO_CUDA_DATA_TYPE(std::complex<gko::half>, MUSA_C_16F);
GKO_CUDA_DATA_TYPE(std::complex<gko::bfloat16>, MUSA_C_16BF);
GKO_CUDA_DATA_TYPE(std::complex<float>, MUSA_C_32F);
GKO_CUDA_DATA_TYPE(std::complex<double>, MUSA_C_64F);
GKO_CUDA_DATA_TYPE(int32, MUSA_R_32I);
GKO_CUDA_DATA_TYPE(int64, MUSA_R_64I);
GKO_CUDA_DATA_TYPE(int8, MUSA_R_8I);

#undef GKO_CUDA_DATA_TYPE


template <typename T>
struct cusparse_index_type_impl {};

#define GKO_CUDA_INDEX_TYPE(_type, _value)                   \
    template <>                                              \
    struct cusparse_index_type_impl<_type> {                 \
        constexpr static musparseIndexType_t value = _value; \
    }

GKO_CUDA_INDEX_TYPE(std::uint16_t, MUSPARSE_INDEX_16U);
GKO_CUDA_INDEX_TYPE(int32, MUSPARSE_INDEX_32I);
GKO_CUDA_INDEX_TYPE(int64, MUSPARSE_INDEX_64I);

#undef GKO_CUDA_INDEX_TYPE


}  // namespace detail


/**
 * This is an alias for the `musaDataType_t` equivalent of `T`. By default,
 * MUSA_C_8U (which is unsupported by C++) is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `musaDataType_t`
 */
template <typename T>
constexpr musaDataType_t musa_data_type()
{
    return detail::musa_data_type_impl<T>::value;
}


/**
 * This is an alias for the `musaIndexType_t` equivalent of `T`. By default,
 * MUSPARSE_INDEX_16U is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `musparseIndexType_t`
 */
template <typename T>
constexpr musparseIndexType_t cusparse_index_type()
{
    return detail::cusparse_index_type_impl<T>::value;
}


/**
 * This is an alias for CUDA's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using musa_type = typename detail::musa_type_impl<T>::type;

/**
 * This is an alias for CUDA/HIP's equivalent of `T` depending on the namespace.
 *
 * @tparam T  a type
 */
template <typename T>
using device_type = musa_type<T>;

/**
 * This works equivalently to device_type, except for replacing std::complex by
 * detail::fake_complex to avoid issues with thrust::complex (alignment etc.)
 *
 * @tparam T  a type
 */
template <typename T>
using device_member_type =
    typename detail::musa_struct_member_type_impl<T>::type;


/**
 * Reinterprets the passed in value as a CUDA type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA type
 */
template <typename T>
inline std::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, musa_type<T>>
as_musa_type(T val)
{
    return reinterpret_cast<musa_type<T>>(val);
}


/**
 * @copydoc as_musa_type()
 */
template <typename T>
inline std::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, musa_type<T>>
as_musa_type(T val)
{
    return *reinterpret_cast<musa_type<T>*>(&val);
}


/**
 * Reinterprets the passed in value as a CUDA/HIP type depending on the
 * namespace.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA/HIP type
 */
template <typename T>
inline device_type<T> as_device_type(T val)
{
    return as_musa_type(val);
}


/**
 * This is an alias for equivalent of type T used in CUDA libraries (cuBLAS,
 * cuSPARSE, etc.).
 *
 * @tparam T  a type
 */
template <typename T>
using culibs_type = typename detail::culibs_type_impl<T>::type;


/**
 * Reinterprets the passed in value as an equivalent type used by the CUDA
 * libraries.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to type used by CUDA libraries
 */
template <typename T>
inline culibs_type<T> as_culibs_type(T val)
{
    return reinterpret_cast<culibs_type<T>>(val);
}


/**
 * Casts fake_complex<T> to thrust::complex<T> and leaves any other types
 * unchanged.
 *
 * This is necessary to work around an issue with Thrust shipped in CUDA 9.2,
 * and the fact that thrust::complex has stronger alignment restrictions than
 * std::complex, i.e. structs containing them among other smaller members have
 * different sizes on device and host.
 *
 * @param val  The input value.
 *
 * @return val cast to the correct type.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr
    typename detail::fake_complex_unpack_impl<T>::type
    fake_complex_unpack(T v)
{
    return detail::fake_complex_unpack_impl<T>::unpack(v);
}


using deviceComplex = muComplex;
using deviceDoubleComplex = muDoubleComplex;


}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_TYPES_HPP_
