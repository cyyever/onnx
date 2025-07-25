# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Graph utilities for checking whether an ONNX proto message is legal."""

from __future__ import annotations

__all__ = [
    "check_attribute",
    "check_function",
    "check_graph",
    "check_model",
    "check_node",
    "check_sparse_tensor",
    "check_tensor",
    "check_value_info",
    "DEFAULT_CONTEXT",
    "LEXICAL_SCOPE_CONTEXT",
    "ValidationError",
    "C",
    "MAXIMUM_PROTOBUF",
]

import os
import sys
from typing import TYPE_CHECKING

import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
from onnx.onnx_pb import IR_VERSION

if TYPE_CHECKING:
    from google.protobuf.message import Message

# Limitation of single protobuf file is 2GiB
MAXIMUM_PROTOBUF = 2147483648


# NB: Please don't edit this context!
DEFAULT_CONTEXT = C.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {"": onnx.defs.onnx_opset_version()}

LEXICAL_SCOPE_CONTEXT = C.LexicalScopeContext()


def _ensure_proto_type(proto: Message, proto_type: type[Message]) -> None:
    if not isinstance(proto, proto_type):
        raise TypeError(
            f"The proto message needs to be of type '{proto_type.__name__}'"
        )


def check_value_info(
    value_info: onnx.ValueInfoProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(value_info, onnx.ValueInfoProto)
    return C.check_value_info(value_info.SerializeToString(), ctx)


def check_tensor(
    tensor: onnx.TensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(tensor, onnx.TensorProto)
    return C.check_tensor(tensor.SerializeToString(), ctx)


def check_attribute(
    attr: onnx.AttributeProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(attr, onnx.AttributeProto)
    return C.check_attribute(attr.SerializeToString(), ctx, lexical_scope_ctx)


def check_node(
    node: onnx.NodeProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(node, onnx.NodeProto)
    return C.check_node(node.SerializeToString(), ctx, lexical_scope_ctx)


def check_function(
    function: onnx.FunctionProto,
    ctx: C.CheckerContext | None = None,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(function, onnx.FunctionProto)
    if ctx is None:
        ctx = C.CheckerContext()
        ctx.ir_version = onnx.helper.find_min_ir_version_for(
            function.opset_import, ignore_unknown=True
        )
        ctx.opset_imports = {
            domain_version.domain: domain_version.version
            for domain_version in function.opset_import
        }
    C.check_function(function.SerializeToString(), ctx, lexical_scope_ctx)


def check_graph(
    graph: onnx.GraphProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(graph, onnx.GraphProto)
    return C.check_graph(graph.SerializeToString(), ctx, lexical_scope_ctx)


def check_sparse_tensor(
    sparse: onnx.SparseTensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(sparse, onnx.SparseTensorProto)
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)


def check_model(
    model: onnx.ModelProto | str | bytes | os.PathLike,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
) -> None:
    """Check the consistency of a model.

    An exception will be raised if the model's ir_version is not set
    properly or is higher than checker's ir_version, or if the model
    has duplicate keys in metadata_props.

    If IR version >= 3, the model must specify opset_import.
    If IR version < 3, the model cannot have any opset_import specified.

    Args:
        model: Model to check. If model is a path, the function checks model
            path first. If the model bytes size is larger than 2GB, function
            should be called using model path.
        full_check: If True, the function also runs shape inference check.
        skip_opset_compatibility_check: If True, the function skips the check for
            opset compatibility.
        check_custom_domain: If True, the function will check all domains. Otherwise
            only check built-in domains.
    """
    # If model is a path instead of ModelProto
    if isinstance(model, (str, os.PathLike)):
        C.check_model_path(
            os.fspath(model),
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )
    else:
        protobuf_string = (
            model if isinstance(model, bytes) else model.SerializeToString()
        )
        # If the protobuf is larger than 2GiB,
        # remind users should use the model path to check
        if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError(
                "This protobuf of onnx model is too large (>2GiB). Call check_model with model path instead."
            )
        C.check_model(
            protobuf_string,
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )


ValidationError = C.ValidationError
