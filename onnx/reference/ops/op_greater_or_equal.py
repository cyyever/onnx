# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryComparison


class GreaterOrEqual(OpRunBinaryComparison):
    def _run(self, a, b):
        return (np.greater_equal(a, b),)
