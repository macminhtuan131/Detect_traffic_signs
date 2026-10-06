import torch

import torch.nn as nn
import numpy as np
import math
class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        if std is None:
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        # Prevent malformed regression outputs from overflowing torch.exp.
        bbox_xform_clip = math.log(1000.0 / 16.0)
        dw = torch.clamp(deltas[:, :, 2] * self.std[2] + self.mean[2], max=bbox_xform_clip)
        dh = torch.clamp(deltas[:, :, 3] * self.std[3] + self.mean[3], max=bbox_xform_clip)

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0::2] = torch.clamp(boxes[:, :, 0::2], min=0, max=width)
        boxes[:, :, 1::2] = torch.clamp(boxes[:, :, 1::2], min=0, max=height)

        return boxes


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3, 4, 5, 6, 7]
        self.strides = strides if strides is not None else [2 ** x for x in self.pyramid_levels]
        self.sizes = sizes if sizes is not None else [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = np.asarray(ratios if ratios is not None else [0.5, 1, 2])
        self.scales = np.asarray(scales if scales is not None else [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.register_buffer('_cached_anchors', torch.empty(0), persistent=False)
        self._cached_shape = None

    def forward(self, image):

        image_shape = image.shape[2:]
        cache_key = (int(image_shape[0]), int(image_shape[1]), image.device.type, image.device.index)
        # Older pickled checkpoints predate the cache attributes.
        if not hasattr(self, '_cached_anchors'):
            self.register_buffer('_cached_anchors', torch.empty(0), persistent=False)
            self._cached_shape = None
        if self._cached_shape == cache_key and self._cached_anchors.device == image.device:
            return self._cached_anchors

        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        anchor_groups = []

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            anchor_groups.append(shifted_anchors)

        all_anchors = np.concatenate(anchor_groups, axis=0).astype(np.float32, copy=False)
        all_anchors = np.expand_dims(all_anchors, axis=0)

        self._cached_anchors = torch.from_numpy(all_anchors).to(image.device)
        self._cached_shape = cache_key
        return self._cached_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def compute_shape(image_shape, pyramid_levels):
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors
