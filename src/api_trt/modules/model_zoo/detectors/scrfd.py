# -*- coding: utf-8 -*-
# Based on Jia Guo reference implementation at
# https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py


from __future__ import division
import time
from typing import Union
from functools import wraps
import logging

import cv2
import numpy as np
from numba import njit

from api_trt.modules.model_zoo.detectors.common.nms import nms
from api_trt.modules.model_zoo.exec_backends.onnxrt_backend import DetectorInfer as DIO
from api_trt.logger import logger
# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    import cupy as cp
    from api_trt.modules.model_zoo.exec_backends.trt_backend import DetectorInfer as DIT
except BaseException:
    DIT = None

import asyncio

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t0 = time.time()
        result = f(*args, **kw)
        took_ms = (time.time() - t0) * 1000
        logger.debug(f'func: "{f.__name__}" took: {took_ms:.4f} ms')
        return result

    return wrap


@njit(fastmath=True, cache=True)
def single_distance2bbox(point, distance, stride):
    """
    Fast conversion of single bbox distances to coordinates using SCRFD distance-based encoding.
    SCRFD uses distance-based encoding where outputs represent distances from anchor center 
    to box edges (left, top, right, bottom), not delta coordinates.

    :param point: Anchor point [x, y]
    :param distance: Bbox distances from anchor point [left, top, right, bottom]
    :param stride: Current stride scale
    :return: bbox [x1, y1, x2, y2]
    """
    # Ensure we're working with copies to avoid modifying input arrays
    left, top, right, bottom = distance[0], distance[1], distance[2], distance[3]
    center_x, center_y = point[0], point[1]
    
    # SCRFD distance-based encoding: distances from center to edges
    x1 = center_x - left * stride    # Distance from center to left edge
    y1 = center_y - top * stride     # Distance from center to top edge  
    x2 = center_x + right * stride   # Distance from center to right edge
    y2 = center_y + bottom * stride  # Distance from center to bottom edge
    
    # Update the distance array in-place for compatibility
    distance[0] = x1
    distance[1] = y1
    distance[2] = x2
    distance[3] = y2
    
    return distance


@njit(fastmath=True, cache=True)
def single_distance2kps(point, distance, stride):
    """
    Fast conversion of single keypoint distances to coordinates

    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :param stride: Current stride scale
    :return: keypoint
    """
    center_x, center_y = float(point[0]), float(point[1])
    stride_f = float(stride)
    
    for ix in range(0, distance.shape[0], 2):
        # Convert keypoint distances to absolute coordinates
        distance[ix] = distance[ix] * stride_f + center_x
        distance[ix + 1] = distance[ix + 1] * stride_f + center_y
    return distance


@njit(fastmath=True, cache=True)
def generate_proposals(score_blob, bbox_blob, kpss_blob, stride, anchors, threshold, score_out, bbox_out, kpss_out,
                       offset):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated np.ndarrays for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores np.ndarray
    :param bbox_out: Output bbox np.ndarray
    :param kpss_out: Output key points np.ndarray
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset
    threshold_f = float(threshold)

    for ix in range(0, anchors.shape[0]):
        # Ensure score comparison uses scalar values to avoid numpy array boolean issues
        score_val = float(score_blob[ix, 0])
        if score_val > threshold_f:
            score_out[total] = score_blob[ix]
            bbox_out[total] = single_distance2bbox(anchors[ix], bbox_blob[ix].copy(), stride)
            kpss_out[total] = single_distance2kps(anchors[ix], kpss_blob[ix].copy(), stride)
            total += 1

    return score_out, bbox_out, kpss_out, total


# @timing
@njit(fastmath=True, cache=True)
def filter(bboxes_list: np.ndarray, kpss_list: np.ndarray,
           scores_list: np.ndarray, nms_threshold: float = 0.4):
    """
    Filter postprocessed network outputs with NMS

    :param bboxes_list: List of bboxes (np.ndarray)
    :param kpss_list: List of keypoints (np.ndarray)
    :param scores_list: List of scores (np.ndarray)
    :param nms_threshold: NMS threshold
    :return: Face bboxes with scores [x1,y1,x2,y2,score], and key points
    """

    if bboxes_list.shape[0] == 0:
        # Return empty arrays with correct shapes
        empty_det = np.zeros((0, 5), dtype=np.float32)
        empty_kpss = np.zeros((0, 5, 2), dtype=np.float32)
        return empty_det, empty_kpss

    pre_det = np.hstack((bboxes_list, scores_list))
    keep = nms(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    det = pre_det[keep, :]
    kpss = kpss_list[keep, :]
    kpss = kpss.reshape((kpss.shape[0], -1, 2))

    return det, kpss


def _normalize_on_device(input, stream, out):
    """
    Normalize image on GPU using inference backend preallocated buffers

    :param input: Raw image as nd.ndarray with HWC shape
    :param stream: Inference backend CUDA stream
    :param out: Inference backend pre-allocated input buffer
    :return: Image shape after preprocessing
    """

    allocate_place = np.prod(input.shape)
    with stream:
        g_img = cp.asarray(input)
        g_img = g_img[..., ::-1]
        g_img = cp.transpose(g_img, (0, 3, 1, 2))
        g_img = cp.subtract(g_img, 127.5, dtype=cp.float32)
        out.device[:allocate_place] = cp.multiply(g_img, 1 / 128).flatten()
    return g_img.shape


class SCRFD:

    def __init__(self, inference_backend: Union[DIT, DIO], ver=1):
        self.session = inference_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.ver = ver
        self.out_shapes = None
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.stream = None
        self.input_ptr = None

    def prepare(self, nms_treshold: float = 0.4, **kwargs):
        """
        Read network params and populate class parameters

        :param nms_treshold: Threshold for NMS IoU

        """
        self.nms_threshold = nms_treshold
        self.session.prepare()
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

        # Preallocate reusable arrays for proposals
        max_prop_len = self._get_max_prop_len(self.input_shape,
                                              self._feat_stride_fpn,
                                              self._num_anchors)
        self.score_list = np.zeros((max_prop_len, 1), dtype='float32')
        self.bbox_list = np.zeros((max_prop_len, 4), dtype='float32')
        self.kpss_list = np.zeros((max_prop_len, 10), dtype='float32')

        # Check if exec backend provides CUDA stream
        try:
            self.stream = self.session.stream
            self.input_ptr = self.session.input_ptr
        except BaseException:
            pass

    # @timing
    def detect(self, imgs, threshold=0.5):
        """
        Run detection pipeline for provided image

        :param imgs: Raw image(s) as nd.ndarray with HWC shape or list/tuple of images
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [x1,y1,x2,y2,score], and key points
        """

        # Robust input handling
        if isinstance(imgs, (list, tuple)):
            if len(imgs) == 0:
                return [], []
            if len(imgs) == 1:
                imgs = np.expand_dims(imgs[0], 0)
            else:
                imgs = np.stack(imgs)
        elif isinstance(imgs, np.ndarray):
            if len(imgs.shape) == 3:
                imgs = np.expand_dims(imgs, 0)
            elif len(imgs.shape) != 4:
                raise ValueError(f"Invalid input shape: {imgs.shape}. Expected 3D or 4D array.")
        else:
            raise ValueError(f"Invalid input type: {type(imgs)}. Expected numpy array, list, or tuple.")

        if imgs.shape[0] == 0:
            return [], []

        input_height = int(imgs[0].shape[0])
        input_width = int(imgs[0].shape[1])
        
        blob = self._preprocess(imgs)
        net_outs = self._forward(blob)

        dets_list = []
        kpss_list = []

        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(net_outs, input_height, input_width, threshold)

        for e in range(self.infer_shape[0]):
            det, kpss = filter(
                bboxes_by_img[e], kpss_by_img[e], scores_by_img[e], self.nms_threshold)

            dets_list.append(det)
            kpss_list.append(kpss)

        return dets_list, kpss_list

    @staticmethod
    def _get_max_prop_len(input_shape, feat_strides, num_anchors):
        """
        Estimate maximum possible number of proposals returned by network

        :param input_shape: maximum input shape of model (i.e (1, 3, 640, 640))
        :param feat_strides: model feature strides (i.e. [8, 16, 32])
        :param num_anchors: model number of anchors (i.e 2)
        :return:
        """

        ln = 0
        pixels = input_shape[2] * input_shape[3]
        for e in feat_strides:
            ln += pixels / (e * e) * num_anchors
        return int(ln)

    # @timing
    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            # Generate anchor centers using meshgrid
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            # Convert grid coordinates to actual pixel coordinates
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            
            if num_anchors > 1:
                # Replicate anchor centers for multiple anchors per location
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            
            centers.append(anchor_centers)
        return centers

    # @timing
    def _preprocess(self, img):
        """
        Normalize image on CPU if backend can't provide CUDA stream,
        otherwise preprocess image on GPU using CuPy

        :param img: Raw image as np.ndarray with NCHW or HWC shape
        :return: Preprocessed image or None if image was processed on device
        """

        blob = None
        if self.stream:
            self.infer_shape = _normalize_on_device(
                img, self.stream, self.input_ptr)
        else:
            # Handle different input formats robustly
            if len(img.shape) == 4:  # Batch of images
                input_size = tuple(img[0].shape[0:2][::-1])
            else:  # Single image
                input_size = tuple(img.shape[0:2][::-1])
            
            blob = cv2.dnn.blobFromImages(
                img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        return blob

    def _forward(self, blob):
        """
        Send input data to inference backend.

        :param blob: Preprocessed image of shape NCHW or None
        :return: network outputs
        """

        t0 = time.time()
        if self.stream:
            net_outs = self.session.run(
                from_device=True, infer_shape=self.infer_shape)
        else:
            net_outs = self.session.run(blob)
        t1 = time.time()
        logger.debug(f'Inference cost: {(t1 - t0) * 1000:.3f} ms.')
        return net_outs

    # @timing
    def _postprocess(self, net_outs, input_height, input_width, threshold):
        """
        Precompute anchor points for provided image size and process network outputs

        :param net_outs: Network outputs
        :param input_height: Input image height
        :param input_width: Input image width
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        key = (input_height, input_width)

        if not self.center_cache.get(key):
            self.center_cache[key] = self._build_anchors(input_height, input_width, self._feat_stride_fpn,
                                                         self._num_anchors)
        anchor_centers = self.center_cache[key]
        bboxes, kpss, scores = self._process_strides(net_outs, threshold, anchor_centers)
        return bboxes, kpss, scores

    def _process_strides(self, net_outs, threshold, anchor_centers):
        """
        Process network outputs by strides and return results proposals filtered by threshold

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """

        batch_size = self.infer_shape[0]
        bboxes_by_img = []
        kpss_by_img = []
        scores_by_img = []

        for n_img in range(batch_size):
            offset = 0
            for idx, stride in enumerate(self._feat_stride_fpn):
                # Handle different tensor formats robustly
                score_blob = net_outs[idx][n_img]
                bbox_blob = net_outs[idx + self.fmc][n_img]
                kpss_blob = net_outs[idx + self.fmc * 2][n_img]
                
                # Ensure proper shape handling for different tensor formats
                if len(score_blob.shape) == 1:
                    score_blob = score_blob.reshape(-1, 1)
                if len(bbox_blob.shape) == 1:
                    bbox_blob = bbox_blob.reshape(-1, 4)
                if len(kpss_blob.shape) == 1:
                    kpss_blob = kpss_blob.reshape(-1, 10)
                
                stride_anchors = anchor_centers[idx]
                
                # Ensure anchors and blobs have matching lengths
                min_len = min(len(stride_anchors), len(score_blob), len(bbox_blob), len(kpss_blob))
                if min_len > 0:
                    self.score_list, self.bbox_list, self.kpss_list, total = generate_proposals(
                        score_blob[:min_len], 
                        bbox_blob[:min_len],
                        kpss_blob[:min_len], 
                        stride,
                        stride_anchors[:min_len], 
                        threshold,
                        self.score_list,
                        self.bbox_list,
                        self.kpss_list, 
                        offset
                    )
                    offset = total

            bboxes_by_img.append(np.copy(self.bbox_list[:offset]))
            kpss_by_img.append(np.copy(self.kpss_list[:offset]))
            scores_by_img.append(np.copy(self.score_list[:offset]))

        return bboxes_by_img, kpss_by_img, scores_by_img
