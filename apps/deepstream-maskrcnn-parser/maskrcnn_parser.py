################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""
    Simple python MASKRCNN output parser.
    The function `extract_maskrcnn_mask` should be used.

    The core function, `resize_mask_vec`, is based on its cpp counterpart
    `resizeMask`, from deepstream cpp sources. It was ported to python
    and vectorized in numpy to improve speed.

    Author: <Pablo Woolvett pablowoolvett@gmail.com>

"""

from typing import Tuple

import numpy as np
import pyds


def _gen_ranges(
    original_height: int,
    original_width: int,
    target_height: int,
    target_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ratio_h = float(original_height / target_height)
    ratio_w = float(original_width / target_width)

    h = np.arange(0, original_height, ratio_h)
    w = np.arange(0, original_width, ratio_w)
    return h, w


def _gen_clips(
    w: np.ndarray,
    original_width: int,
    h: np.ndarray,
    original_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    left = np.clip(np.floor(w), 0.0, original_width - 1)
    right = np.clip(np.ceil(w), 0.0, original_width - 1)
    top = np.clip(np.floor(h), 0.0, original_height - 1)
    bottom = np.clip(np.ceil(h), 0.0, original_height - 1)
    return left, right, top, bottom


def _gen_idxs(
    original_width: int,
    left,
    right,
    top,
    bottom,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    left_top_idx = np.add.outer(top * original_width, left).astype(int)
    right_top_idx = np.add.outer(top * original_width, right).astype(int)
    left_bottom_idx = np.add.outer(bottom * original_width, left).astype(int)
    right_bottom_idx = np.add.outer(bottom * original_width, right).astype(int)

    return left_top_idx, right_top_idx, left_bottom_idx, right_bottom_idx


def _take_vals(
    src,
    *idxmats,
):
    return tuple(src.take(idxmat) for idxmat in idxmats)


def _interpolate(
    w: np.ndarray,
    left: np.ndarray,
    h: np.ndarray,
    top: np.ndarray,
    left_top_val: np.ndarray,
    right_top_val: np.ndarray,
    left_bottom_val: np.ndarray,
    right_bottom_val: np.ndarray,
) -> np.ndarray:
    delta_w = w - left
    top_lerp = left_top_val + (right_top_val - left_top_val) * delta_w
    bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * delta_w
    return top_lerp + ((bottom_lerp - top_lerp).T * (h - top)).T


def resize_mask_vec(
    src: np.ndarray,
    src_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Resize mask from original deepstream object into numpy array.

    Args:
        src: Mask array from deepstream object.
        src_shape: Shape of the original mask in (height,width) format.
        target_shape: Shape of the target mask in (height,width) format.
        threshold: Threshold for the mask.

    Returns:
        A 2d binary mask of np.uint8 valued 0 and 255.

    See Also:
        * `extract_maskrcnn_mask` in this module for sample usage from deepstream.
        * `resizeMask` function at `deepstream/sources/apps/sample_apps/deepstream-mrcnn-app/deepstream_mrcnn_test.cpp`
    """

    original_height, original_width = src_shape
    target_height, target_width = target_shape

    h, w = _gen_ranges(
        original_height, original_width, target_height, target_width
    )

    left, right, top, bottom = _gen_clips(
        w, original_width, h, original_height
    )

    left_top_idx, right_top_idx, left_bottom_idx, right_bottom_idx = _gen_idxs(
        original_width, left, right, top, bottom
    )

    left_top_val, right_top_val, left_bottom_val, right_bottom_val = _take_vals(
        src, left_top_idx, right_top_idx, left_bottom_idx, right_bottom_idx
    )

    lerp = _interpolate(
        w, left, h, top, left_top_val, right_top_val, left_bottom_val, right_bottom_val
    )

    ret = np.zeros_like(lerp, dtype=np.uint8)
    ret[lerp >= threshold] = 255
    return ret


def extract_maskrcnn_mask(obj_meta: pyds.NvDsObjectMeta) -> np.ndarray:
    """Extract maskrcnn mask from deepstream object.
    
    Args:
        obj_meta: Deepstream object meta from detection.

    Returns:
        A 2d binary mask of np.uint8 valued 0 and 255.

    Example:
        >>> obj_meta = pyds.NvDsObjectMeta.cast(...)
        >>> mask = extract_maskrcnn_mask(obj_meta)
        >>> mask.shape, mask.dtype
        ((300,100), dtype('uint8'))
    See Also:
        `resize_mask_vec` for the internal implementation.
    """

    rect_height = int(np.ceil(obj_meta.rect_params.height))
    rect_width = int(np.ceil(obj_meta.rect_params.width))
    return resize_mask_vec(
        obj_meta.mask_params.data,
        (obj_meta.mask_params.height, obj_meta.mask_params.width),
        (rect_height, rect_width),
        obj_meta.mask_params.threshold,
    )
