// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ONNX_NAMESPACE {

// Returns true on overflow. Uses __builtin on GCC/Clang, manual check on MSVC.
// Callers must ensure a >= 0 && b >= 0 (enforced by safe_dim_product).
inline bool checked_mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, result);
#else
  if (a > 0 && b > INT64_MAX / a) {
    return true;
  }
  *result = a * b;
  return false;
#endif
}

// Safe product of dims. Calls on_error(const char*) on negative dim or overflow.
template <typename DimsContainer, typename ErrorHandler>
inline int64_t safe_dim_product(const DimsContainer& dims, ErrorHandler on_error) {
  int64_t result = 1;
  for (auto d : dims) {
    int64_t dim = static_cast<int64_t>(d);
    if (dim < 0) {
      on_error("Negative dimension value");
    }
    if (checked_mul_overflow(result, dim, &result)) {
      on_error("Tensor dimension product overflow");
    }
  }
  return result;
}

} // namespace ONNX_NAMESPACE
