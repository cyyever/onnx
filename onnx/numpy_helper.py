# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import typing_extensions

import onnx._custom_element_types as custom_np_types
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data

if TYPE_CHECKING:
    from collections.abc import Sequence


def _combine_pairs_to_complex(fa: Sequence[int]) -> list[complex]:
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def bfloat16_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Converts ndarray of bf16 (as uint32) to f32 (as uint32).

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: If specified, the function reshapes the results.

    Returns:
        A numpy array of float32 with the same dimension if dims is
        None, or reshaped to dims if specified
    """
    shift = lambda x: x << 16  # noqa: E731
    if dims is None:
        if len(data.shape) == 0:
            return shift(np.array([data]).astype(np.int32)).view(np.float32)[0]  # type: ignore[no-any-return]
        return shift(data.astype(np.int32)).view(np.float32)  # type: ignore[no-any-return]
    return shift(data.astype(np.int32)).reshape(dims).view(np.float32)  # type: ignore[no-any-return]


def _float8e4m3_to_float32_scalar(ival: int, fn: bool, uz: bool) -> np.float32:
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")
    if ival < 0 or ival > 255:  # noqa: PLR2004
        raise ValueError(f"{ival} is not a float8.")
    if uz:
        exponent_bias = 8
        if ival == 0x80:  # noqa: PLR2004
            return np.nan  # type: ignore[return-value]
    else:
        exponent_bias = 7
        if ival == 255:  # noqa: PLR2004
            return np.float32(-np.nan)
        if ival == 127:  # noqa: PLR2004
            return np.float32(np.nan)

    ival = np.uint32(ival)  # type: ignore[assignment]
    expo = (ival & 0x78) >> 3
    mant = ival & 0x07
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            res |= (mant & 0x3) << 21
            res |= expo << 23
    else:
        res |= mant << 20
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = np.uint32(res).view(np.float32)
    return f


_float8e4m3_to_float32 = np.vectorize(
    _float8e4m3_to_float32_scalar, excluded=["fn", "uz"]
)


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float8e4m3_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
    fn: bool = True,
    uz: bool = False,
) -> np.ndarray:
    """Converts ndarray of float8, e4m3 (as uint32) to f32 (as uint32).

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        dims: If specified, the function reshapes the results.
        fn: No infinite values.
        uz: No negative zero.

    Returns:
        A numpy array of float32 with the same dimension if dims is None,
        or reshaped to dims if specified.
    """
    if not fn:
        raise NotImplementedError(
            "float32_to_float8e4m3 not implemented with fn=False."
        )
    res = _float8e4m3_to_float32(data, fn=fn, uz=uz)
    if dims is None:
        return res  # type: ignore[no-any-return]
    return res.reshape(dims)  # type: ignore[no-any-return]


def _float8e5m2_to_float32_scalar(ival: int, fn: bool, uz: bool) -> np.float32:
    if fn and uz:
        if ival == 0x80:  # noqa: PLR2004
            return np.float32(np.nan)
        exponent_bias = 16
    elif not fn and not uz:
        if ival in {253, 254, 255}:
            return np.float32(-np.nan)
        if ival in {125, 126, 127}:
            return np.float32(np.nan)
        if ival == 252:  # noqa: PLR2004
            return np.float32(-np.inf)
        if ival == 124:  # noqa: PLR2004
            return np.float32(np.inf)
        exponent_bias = 15
    else:
        raise NotImplementedError("fn and uz must be both False or True.")

    ival = np.uint32(ival)  # type: ignore[assignment]
    expo = (ival & 0x7C) >> 2
    mant = ival & 0x03
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x2 == 0:
                mant &= 0x1
                mant <<= 1
                expo -= 1
            res |= (mant & 0x1) << 22
            res |= expo << 23
    else:
        res |= mant << 21
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = np.uint32(res).view(np.float32)
    return f


_float8e5m2_to_float32 = np.vectorize(
    _float8e5m2_to_float32_scalar, excluded=["fn", "uz"]
)


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float8e5m2_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
    fn: bool = False,
    uz: bool = False,
) -> np.ndarray:
    """Converts ndarray of float8, e5m2 (as uint32) to f32 (as uint32).

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        dims: If specified, the function reshapes the results.
        fn: No infinite values.
        uz: No negative zero.

    Returns:
        A numpy array of float32 with the same dimension if dims is None,
        or reshaped to dims if specified
    """
    res = _float8e5m2_to_float32(data, fn=fn, uz=uz)
    if dims is None:
        return res  # type: ignore[no-any-return]
    return res.reshape(dims)  # type: ignore[no-any-return]


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider implementing your own unpack logic",
    category=DeprecationWarning,
)
def unpack_int4(
    data: np.int32 | np.ndarray,
    dims: int | Sequence[int],
    signed: bool,
) -> np.ndarray:
    """Converts ndarray of int4 (as packed uint8) to f32
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: The dimensions are used to reshape the unpacked buffer
        signed: Whether the 4 bit integer is signed or unsigned

    Returns:
        A numpy array of float32 reshaped to dims.
    """
    single_func = lambda x: subbyte._unpack_single_4bitx2(x, signed)  # noqa: E731
    func = np.frompyfunc(single_func, 1, 2)

    res_high, res_low = func(data.ravel())
    res = np.empty((res_high.size + res_low.size,), dtype=np.float32)
    res[0::2] = res_high
    res[1::2] = res_low

    if (
        res.size == np.prod(dims) + 1
    ):  # handle single-element padding due to odd number of elements
        res = res.ravel()[:-1]
    res = res.reshape(dims)
    return res


def unpacked_float4e2m1_to_float32(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Evaluate the numerical value of an array of unpacked float4e2m1 values (as uint8)
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        x: an array of uint8 elements representing a float4e2m1 (using the 4 LSB)

    Returns:
        An array of float32 elements representing the values of the float4e2m1 input.
    """
    # x is stored in 4 LSB of int
    sign = np.where(np.bitwise_and(x, 0x08), -1, 1)
    mantissa = (x & 0x01).astype(np.float32)
    exponent = ((x & 0x06) >> 1).astype(np.float32)

    val = np.where(
        exponent == 0,
        sign * (mantissa / 2.0),
        sign * (1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1),
    )  # denormalized, normalized
    return val


def unpack_float4e2m1(
    data: npt.NDArray[np.uint8],
    dims: Sequence[int],
) -> np.ndarray:
    """Converts ndarray of float4e2m1 (as packed uint8) to f32
    See :ref:`onnx-detail-float4` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: The dimensions are used to reshape the unpacked buffer

    Returns:
        A numpy array of float32 reshaped to dims.
    """
    res = _unpack_uint4(data, dims)
    return unpacked_float4e2m1_to_float32(res)


def _unpack_uint4(
    data: npt.NDArray[np.uint8], dims: Sequence[int]
) -> npt.NDArray[np.uint8]:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def _extend_int4_sign_bits(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.int8]:
    """Extend 4-bit signed integer to 8-bit signed integer."""
    return np.where((x >> 3) == 0, x, x | 0xF0).astype(np.int8)


def _unpack_int4(
    data: npt.NDArray[np.uint8], dims: Sequence[int]
) -> npt.NDArray[np.int8]:
    """Convert a packed (signed) int4 array to unpacked int4 array represented as int8.

    The sign bit is extended to the most significant bit of the int8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8 reshaped to dims.
    """
    unpacked = _unpack_uint4(data, dims)
    return _extend_int4_sign_bits(unpacked)


def to_array(tensor: TensorProto, base_dir: str = "") -> np.ndarray:  # noqa: PLR0911
    """Converts a tensor def object to a numpy array.
    Supports types defined in :mod:`onnx._custom_element_types`.

    Args:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it

    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError("Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    storage_np_dtype = helper.tensor_dtype_to_np_dtype(
        helper.tensor_dtype_to_storage_tensor_dtype(tensor_dtype)
    )
    storage_field = helper.tensor_dtype_to_field(tensor_dtype)
    dims = tensor.dims

    if tensor.data_type == TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = [s.decode("utf-8") for s in utf8_strings]
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    # Load raw data from external tensor if it exists
    if uses_external_data(tensor):
        load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        raw_data = tensor.raw_data
        if sys.byteorder == "big":
            # Convert endian from little to big
            raw_data = np.frombuffer(raw_data, dtype=np_dtype).byteswap().tobytes()

        data: np.ndarray[Any, Any]
        if tensor_dtype == TensorProto.BFLOAT16:
            data = np.frombuffer(raw_data, dtype=np.int16).reshape(dims)
            return data.view(custom_np_types.bfloat16)

        if tensor_dtype == TensorProto.FLOAT8E4M3FN:
            data = np.frombuffer(raw_data, dtype=np.int8).reshape(dims)
            return data.view(custom_np_types.float8e4m3fn)

        if tensor_dtype == TensorProto.FLOAT8E4M3FNUZ:
            data = np.frombuffer(raw_data, dtype=np.int8).reshape(dims)
            return data.view(custom_np_types.float8e4m3fnuz)

        if tensor_dtype == TensorProto.FLOAT8E5M2:
            data = np.frombuffer(raw_data, dtype=np.int8).reshape(dims)
            return data.view(custom_np_types.float8e5m2)

        if tensor_dtype == TensorProto.FLOAT8E5M2FNUZ:
            data = np.frombuffer(raw_data, dtype=np.int8).reshape(dims)
            return data.view(custom_np_types.float8e5m2fnuz)

        if tensor_dtype == TensorProto.UINT4:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_uint4(
                data,
                dims,
            ).view(custom_np_types.uint4)

        if tensor_dtype == TensorProto.INT4:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_int4(data, dims).view(custom_np_types.int4)

        if tensor_dtype == TensorProto.FLOAT4E2M1:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_uint4(data, dims).view(custom_np_types.float4e2m1)

        return np.frombuffer(raw_data, dtype=np_dtype).reshape(dims)

    # float16 is stored as int32 (uint16 type); Need view to get the original value
    if tensor_dtype == TensorProto.FLOAT16:
        return (
            np.asarray(tensor.int32_data, dtype=np.uint16)
            .reshape(dims)
            .view(np.float16)
        )

    if tensor_dtype == TensorProto.BFLOAT16:
        data = (
            np.asarray(tensor.int32_data, dtype=np.int32)
            .astype(np.uint16)
            .reshape(dims)
        )
        return data.view(custom_np_types.bfloat16)

    if tensor_dtype == TensorProto.FLOAT8E4M3FN:
        data = (
            np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8).reshape(dims)
        )
        return data.view(custom_np_types.float8e4m3fn)

    if tensor_dtype == TensorProto.FLOAT8E4M3FNUZ:
        data = (
            np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8).reshape(dims)
        )
        return data.view(custom_np_types.float8e4m3fnuz)

    if tensor_dtype == TensorProto.FLOAT8E5M2:
        data = (
            np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8).reshape(dims)
        )
        return data.view(custom_np_types.float8e5m2)

    if tensor_dtype == TensorProto.FLOAT8E5M2FNUZ:
        data = (
            np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8).reshape(dims)
        )
        return data.view(custom_np_types.float8e5m2fnuz)

    if tensor_dtype == TensorProto.UINT4:
        data = np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8)
        return _unpack_uint4(data, dims).view(custom_np_types.uint4)

    if tensor_dtype == TensorProto.INT4:
        data = np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8)
        return _unpack_int4(data, dims).view(custom_np_types.int4)

    if tensor_dtype == TensorProto.FLOAT4E2M1:
        data = np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8)
        return _unpack_uint4(data, dims).view(custom_np_types.float4e2m1)

    data = getattr(tensor, storage_field)
    if tensor_dtype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        data = _combine_pairs_to_complex(data)  # type: ignore[assignment,arg-type]

    return np.asarray(data, dtype=storage_np_dtype).astype(np_dtype).reshape(dims)


def _from_array(arr: np.ndarray, name: str | None = None) -> TensorProto:
    """Converts a numpy array to a tensor def.

    Args:
        arr: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    if not isinstance(arr, (np.ndarray, np.generic)):
        raise TypeError(
            f"arr must be of type np.generic or np.ndarray, got {type(arr)}"
        )

    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
        # Special care for strings.
        tensor.data_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
        # TODO: Introduce full string support.
        # We flatten the array in case there are 2-D arrays are specified
        # We throw the error below if we have a 3-D array or some kind of other
        # object. If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = arr.flatten()
        for e in flat_array:
            if isinstance(e, str):
                tensor.string_data.append(e.encode("utf-8"))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, str):
                        tensor.string_data.append(s.encode("utf-8"))
                    elif isinstance(s, bytes):
                        tensor.string_data.append(s)
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ",
                    str(type(e)),
                )
        return tensor

    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = helper.np_dtype_to_tensor_dtype(arr.dtype)
    except KeyError as e:
        raise RuntimeError(f"Numpy data type not understood yet: {arr.dtype!r}") from e
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == "big":
        # Convert endian from big to little
        _convert_endian(tensor)

    return tensor


def from_array(tensor: np.ndarray, name: str | None = None) -> TensorProto:
    """Converts an array into a TensorProto including
    supported type defined in :mod:`onnx._custom_element_types`.

    Args:
        tensor: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    if not isinstance(tensor, np.ndarray):
        return _from_array(tensor, name)

    dt = tensor.dtype
    if dt == custom_np_types.float8e4m3fn and dt.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
        dt_to = np.uint8
    elif dt == custom_np_types.float8e4m3fnuz and dt.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
        dt_to = np.uint8
    elif dt == custom_np_types.float8e5m2 and dt.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
        dt_to = np.uint8
    elif dt == custom_np_types.float8e5m2fnuz and dt.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
        dt_to = np.uint8
    elif dt == custom_np_types.bfloat16 and dt.descr[0][0] == "bfloat16":
        to = TensorProto.BFLOAT16
        dt_to = np.uint16  # type: ignore[assignment]
    elif dt == custom_np_types.int4 and dt.descr[0][0] == "int4":
        to = TensorProto.INT4
        dt_to = np.int8  # type: ignore[assignment]
    elif dt == custom_np_types.uint4 and dt.descr[0][0] == "uint4":
        to = TensorProto.UINT4
        dt_to = np.uint8
    elif dt == custom_np_types.float4e2m1 and dt.descr[0][0] == "float4e2m1":
        to = TensorProto.FLOAT4E2M1
        dt_to = np.uint8
    else:
        return _from_array(tensor, name)

    if to in (TensorProto.UINT4, TensorProto.INT4, TensorProto.FLOAT4E2M1):
        value = tensor.astype(dt_to).ravel()
        if value.size % 2 == 1:
            raise ValueError(
                f"The conversion of a tensor of INT4 or UINT4 requires an even size "
                f"(shape={tensor.shape}). Every byte stores two INT4 or two UINT4."
            )
        buffer = (value[::2] & 0x0F) + (value[1::2] << 4)

        pb = TensorProto()
        pb.dims.extend(tensor.shape)
        if name:
            pb.name = name
        pb.raw_data = buffer.tobytes()
        pb.data_type = to
        return pb

    t = _from_array(tensor.astype(dt_to), name)
    t.data_type = to
    return t


def to_list(sequence: SequenceProto) -> list[Any]:
    """Converts a sequence def to a Python list.

    Args:
        sequence: a SequenceProto object.

    Returns:
        list: the converted list.
    """
    elem_type = sequence.elem_type
    if elem_type == SequenceProto.TENSOR:
        return [to_array(v) for v in sequence.tensor_values]
    if elem_type == SequenceProto.SPARSE_TENSOR:
        return [to_array(v) for v in sequence.sparse_tensor_values]  # type: ignore[arg-type]
    if elem_type == SequenceProto.SEQUENCE:
        return [to_list(v) for v in sequence.sequence_values]
    if elem_type == SequenceProto.MAP:
        return [to_dict(v) for v in sequence.map_values]
    raise TypeError("The element type in the input sequence is not supported.")


def from_list(
    lst: list[Any], name: str | None = None, dtype: int | None = None
) -> SequenceProto:
    """Converts a list into a sequence def.

    Args:
        lst: a Python list
        name: (optional) the name of the sequence.
        dtype: (optional) type of element in the input list, used for specifying
                          sequence values when converting an empty list.

    Returns:
        SequenceProto: the converted sequence def.
    """
    sequence = SequenceProto()
    if name:
        sequence.name = name

    if dtype:
        elem_type = dtype
    elif len(lst) > 0:
        first_elem = lst[0]
        if isinstance(first_elem, dict):
            elem_type = SequenceProto.MAP
        elif isinstance(first_elem, list):
            elem_type = SequenceProto.SEQUENCE
        else:
            elem_type = SequenceProto.TENSOR
    else:
        # if empty input list and no dtype specified
        # choose sequence of tensors on default
        elem_type = SequenceProto.TENSOR
    sequence.elem_type = elem_type

    if (len(lst) > 0) and not all(isinstance(elem, type(lst[0])) for elem in lst):
        raise TypeError(
            "The element type in the input list is not the same "
            "for all elements and therefore is not supported as a sequence."
        )

    if elem_type == SequenceProto.TENSOR:
        for tensor in lst:
            sequence.tensor_values.extend([from_array(tensor)])
    elif elem_type == SequenceProto.SEQUENCE:
        for seq in lst:
            sequence.sequence_values.extend([from_list(seq)])
    elif elem_type == SequenceProto.MAP:
        for mapping in lst:
            sequence.map_values.extend([from_dict(mapping)])
    else:
        raise TypeError(
            "The element type in the input list is not a tensor, "
            "sequence, or map and is not supported."
        )
    return sequence


def to_dict(map_proto: MapProto) -> dict[Any, Any]:
    """Converts a map def to a Python dictionary.

    Args:
        map_proto: a MapProto object.

    Returns:
        The converted dictionary.
    """
    key_list: list[Any] = []
    if map_proto.key_type == TensorProto.STRING:
        key_list = list(map_proto.string_keys)
    else:
        key_list = list(map_proto.keys)

    value_list = to_list(map_proto.values)
    if len(key_list) != len(value_list):
        raise IndexError(
            "Length of keys and values for MapProto (map name: ",
            map_proto.name,
            ") are not the same.",
        )
    dictionary = dict(zip(key_list, value_list))
    return dictionary


def from_dict(dict_: dict[Any, Any], name: str | None = None) -> MapProto:
    """Converts a Python dictionary into a map def.

    Args:
        dict_: Python dictionary
        name: (optional) the name of the map.

    Returns:
        MapProto: the converted map def.
    """
    map_proto = MapProto()
    if name:
        map_proto.name = name
    keys = list(dict_)
    raw_key_type = np.result_type(keys[0])
    key_type = helper.np_dtype_to_tensor_dtype(raw_key_type)

    valid_key_int_types = [
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    ]

    if not (all(np.result_type(key) == raw_key_type for key in keys)):
        raise TypeError(
            "The key type in the input dictionary is not the same "
            "for all keys and therefore is not valid as a map."
        )

    values = list(dict_.values())
    raw_value_type = np.result_type(values[0])
    if not all(np.result_type(val) == raw_value_type for val in values):
        raise TypeError(
            "The value type in the input dictionary is not the same "
            "for all values and therefore is not valid as a map."
        )

    value_seq = from_list(values)

    map_proto.key_type = key_type
    if key_type == TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    map_proto.values.CopyFrom(value_seq)
    return map_proto


def to_optional(optional: OptionalProto) -> Any | None:
    """Converts an optional def to a Python optional.

    Args:
        optional: an OptionalProto object.

    Returns:
        opt: the converted optional.
    """
    elem_type = optional.elem_type
    if elem_type == OptionalProto.UNDEFINED:
        return None
    if elem_type == OptionalProto.TENSOR:
        return to_array(optional.tensor_value)
    if elem_type == OptionalProto.SPARSE_TENSOR:
        return to_array(optional.sparse_tensor_value)  # type: ignore[arg-type]
    if elem_type == OptionalProto.SEQUENCE:
        return to_list(optional.sequence_value)
    if elem_type == OptionalProto.MAP:
        return to_dict(optional.map_value)
    if elem_type == OptionalProto.OPTIONAL:
        return to_optional(optional.optional_value)
    raise TypeError("The element type in the input optional is not supported.")


def from_optional(
    opt: Any | None, name: str | None = None, dtype: int | None = None
) -> OptionalProto:
    """Converts an optional value into a Optional def.

    Args:
        opt: a Python optional
        name: (optional) the name of the optional.
        dtype: (optional) type of element in the input, used for specifying
                          optional values when converting empty none. dtype must
                          be a valid OptionalProto.DataType value

    Returns:
        optional: the converted optional def.
    """
    # TODO: create a map and replace conditional branches
    optional = OptionalProto()
    if name:
        optional.name = name

    if dtype is not None:
        # dtype must be a valid OptionalProto.DataType
        if dtype not in OptionalProto.DataType.values():
            raise TypeError(f"{dtype} must be a valid OptionalProto.DataType.")
        elem_type = dtype
    elif isinstance(opt, dict):
        elem_type = OptionalProto.MAP
    elif isinstance(opt, list):
        elem_type = OptionalProto.SEQUENCE
    elif opt is None:
        elem_type = OptionalProto.UNDEFINED
    else:
        elem_type = OptionalProto.TENSOR

    optional.elem_type = elem_type

    if opt is not None:
        if elem_type == OptionalProto.TENSOR:
            optional.tensor_value.CopyFrom(from_array(opt))
        elif elem_type == OptionalProto.SEQUENCE:
            optional.sequence_value.CopyFrom(from_list(opt))
        elif elem_type == OptionalProto.MAP:
            optional.map_value.CopyFrom(from_dict(opt))
        else:
            raise TypeError(
                "The element type in the input is not a tensor, "
                "sequence, or map and is not supported."
            )
    return optional


def _convert_endian(tensor: TensorProto) -> None:
    """Call to convert endianness of raw data in tensor.

    Args:
        tensor: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    tensor.raw_data = (
        np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()
    )


def create_random_int(
    input_shape: tuple[int], dtype: np.dtype, seed: int = 1
) -> np.ndarray:
    """Create random integer array for backend/test/case/node.

    Args:
        input_shape: The shape for the returned integer array.
        dtype: The NumPy data type for the returned integer array.
        seed: The seed for np.random.

    Returns:
        np.ndarray: Random integer array.
    """
    np.random.seed(seed)
    if dtype in (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        # the range of np.random.randint is int32; set a fixed boundary if overflow
        end = min(np.iinfo(dtype).max, np.iinfo(np.int32).max)
        start = max(np.iinfo(dtype).min, np.iinfo(np.int32).min)
        return np.random.randint(start, end, size=input_shape).astype(dtype)
    else:
        raise TypeError(f"{dtype} is not supported by create_random_int.")
