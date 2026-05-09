// MUSA musparse compatibility layer for Ginkgo
// Provides type aliases and stubs for APIs that differ between cusparse and musparse

#ifndef GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_
#define GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_

#include <musparse.h>
#include <musparse-compatible.h>
#include <cstddef>

#ifndef MUSPARSE_CSR2CSC_ALG2
#define MUSPARSE_CSR2CSC_ALG2 2
#endif

#ifndef MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED 7
#endif

#endif // GKO_MUSA_BASE_MUSPARSE_COMPAT_HPP_
