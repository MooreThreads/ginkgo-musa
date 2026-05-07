// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_MUSA_BASE_CURAND_BINDINGS_HPP_
#define GKO_MUSA_BASE_CURAND_BINDINGS_HPP_


#include <murand.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace musa {
/**
 * @brief The CURAND namespace.
 *
 * @ingroup murand
 */
namespace murand {
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(MURAND_STATUS_TYPE_ERROR);
}


}  // namespace detail


template <typename ValueType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float> : std::true_type {};

template <>
struct is_supported<double> : std::true_type {};

template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};


inline murandGenerator_t rand_generator(int64 seed,
                                        murandRngType generator_type,
                                        musaStream_t stream)
{
    murandGenerator_t gen;
    GKO_ASSERT_NO_CURAND_ERRORS(murandCreateGenerator(&gen, generator_type));
    GKO_ASSERT_NO_CURAND_ERRORS(murandSetStream(gen, stream));
    GKO_ASSERT_NO_CURAND_ERRORS(murandSetPseudoRandomGeneratorSeed(gen, seed));
    return gen;
}


inline void destroy(murandGenerator_t gen)
{
    GKO_ASSERT_NO_CURAND_ERRORS(murandDestroyGenerator(gen));
}


#define GKO_BIND_CURAND_RANDOM_VECTOR(ValueType, CurandName)                 \
    inline void rand_vector(                                                 \
        murandGenerator_t& gen, int n, remove_complex<ValueType> mean,       \
        remove_complex<ValueType> stddev, ValueType* values)                 \
    {                                                                        \
        n = is_complex<ValueType>() ? 2 * n : n;                             \
        GKO_ASSERT_NO_CURAND_ERRORS(CurandName(                              \
            gen, reinterpret_cast<remove_complex<ValueType>*>(values), n,    \
            mean, stddev));                                                  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CURAND_RANDOM_VECTOR(float, murandGenerateNormal);
GKO_BIND_CURAND_RANDOM_VECTOR(double, murandGenerateNormalDouble);
GKO_BIND_CURAND_RANDOM_VECTOR(std::complex<float>, murandGenerateNormal);
GKO_BIND_CURAND_RANDOM_VECTOR(std::complex<double>, murandGenerateNormalDouble);
template <typename ValueType>
GKO_BIND_CURAND_RANDOM_VECTOR(ValueType, detail::not_implemented);


#undef GKO_BIND_CURAND_RANDOM_VECTOR


}  // namespace murand


namespace randlib {


using namespace murand;


#define RANDLIB_RNG_PSEUDO_DEFAULT MURAND_RNG_PSEUDO_DEFAULT


}  // namespace randlib
}  // namespace musa
}  // namespace kernels
}  // namespace gko


#endif  // GKO_MUSA_BASE_CURAND_BINDINGS_HPP_
