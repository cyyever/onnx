// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_NAMESPACE {

/// Multiply two int64_t values, returning true on overflow.
/// Uses compiler builtins on GCC/Clang for optimal codegen (single imul + jo).
/// Falls back to manual division check on MSVC.
inline bool checked_mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, result);
#else
  // For non-negative values (the common case for tensor dimensions):
  if (a >= 0 && b >= 0) {
    if (a != 0 && b > INT64_MAX / a) {
      return true;
    }
    *result = a * b;
    return false;
  }
  // General case for mixed signs:
  if (a > 0 && b < 0) {
    if (b < INT64_MIN / a) return true;
  } else if (a < 0 && b > 0) {
    if (a < INT64_MIN / b) return true;
  } else if (a < 0 && b < 0) {
    if (a < INT64_MAX / b) return true;
  }
  *result = a * b;
  return false;
#endif
}

/// Compute the product of tensor dimensions with overflow checking.
/// Returns the product of all dimension values.
/// Throws std::overflow_error if the product overflows int64_t.
/// Throws std::invalid_argument if any dimension is negative.
template <typename DimsContainer>
inline int64_t safe_dim_product(const DimsContainer& dims) {
  int64_t result = 1;
  for (auto d : dims) {
    int64_t dim = static_cast<int64_t>(d);
    if (dim < 0) {
      throw std::invalid_argument("Negative dimension value: " + std::to_string(dim));
    }
    if (checked_mul_overflow(result, dim, &result)) {
      throw std::overflow_error("Tensor dimension product overflow");
    }
  }
  return result;
}

/// Compute the product of dimensions starting from a given index.
/// Throws on overflow or negative dimensions.
template <typename DimsContainer>
inline int64_t safe_dim_product_from(const DimsContainer& dims, int start) {
  int64_t result = 1;
  for (int i = start; i < static_cast<int>(dims.size()); ++i) {
    int64_t dim = static_cast<int64_t>(dims[i]);
    if (dim < 0) {
      throw std::invalid_argument("Negative dimension value: " + std::to_string(dim));
    }
    if (checked_mul_overflow(result, dim, &result)) {
      throw std::overflow_error("Tensor dimension product overflow");
    }
  }
  return result;
}

} // namespace ONNX_NAMESPACE
