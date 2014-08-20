"""
Functions for finding the edges of a subject in an
image-as-a-numpy-array
"""
from __future__ import division
from collections import namedtuple
import numpy as np


DEFAULT_THRESH = 15
DEFAULT_NEG_SAMPLE = 5

_row_data = namedtuple("row_data", "left_idx right_idx neg_space_l neg_space_r")
_subject_data = namedtuple("subject_data", "left right")
_edge_data = namedtuple("row_data", "idx neg_space")


def config(*overrides, **kw_overrides):
    return _config(*overrides, **kw_overrides)


_config = namedtuple("config",
                     "neg_sample_size threshold bg_change_tolerance")
_defaults = DEFAULT_NEG_SAMPLE, DEFAULT_THRESHOLD, BG_CHANGE_TOLERANCE


def iter_subject_edges(img, config=config()):
    """Finds the edges of the subject for each row of pixels. This assumes that
    the subject is on a background with significantly different RGB values.
    Yields `_row_data` instances."""
    _get_corrected_neg_space(img, neg_sample)
    edge_neg_space = _get_edge_neg_space(img, config.neg_sample_size)
    prev_neg_space = edge_neg_space
    for row in img:
        yield (_find_edge_indices(row, config.neg_sample, config.threshold),)


def _find_edge_indices(row, neg_sample_size, threshold, prev_edges=None):
    """Find edges of a single subject. Naively assume only a left
    and a right edge are present and that there's nothing in between."""
    neg_space_l = _gather_neg_sample(row[:neg_sample_size],
                                     threshold, prev_edges)
    neg_space_r = _gather_neg_sample(row[-neg_sample_size:],
                                     threshold, prev_edges)
    pos_space = np.all(np.abs(row - neg_space_l) > threshold, axis=1)
    left_edge = np.argmax(pos_space)
    if np.any(np.abs(neg_space_r - neg_space_l) > max(threshold // 2, 1)):
        pos_space_r = np.all(
            np.abs(row[left_edge:] - neg_space_r) > threshold, axis=1)
        right_edge = np.argmax(pos_space_r[::-1])
    else:
        right_edge = np.argmax(pos_space[::-1])
    if right_edge:
        width = row.shape[0]
        right_edge = width - right_edge
    return _row_data(left_edge, right_edge, neg_space_l, neg_space_r)


def _get_corrected_neg_space(img, sample_size):
    edge_space = _get_edge_neg_space(img, sample_size)
    left = np.median(img[::, :sample_size:], axis=1)
    right = np.median(img[::, -sample_size::], axis=1)


def _get_edge_neg_space(img, sample_size):
    """Median value of negative space on the edge of the image"""
    medians = [
        np.median(img[:sample_size:], axis=0),
        np.median(img[-sample_size::], axis=0),
        np.median(img[::, :sample_size:], axis=1),
        np.median(img[::, -sample_size::], axis=1)
    ]
    return np.median(medians[1:3], axis=0)


def _gather_neg_sample(sample, threshold, prev_edges):
    return np.mean(sample, axis=0)
    if np.any(sample.max(axis=0) - sample.min(axis=0) > threshold):
        return
    else:
        return np.mean(sample, axis=0)


def _find_edge_indices_simple(row, neg_sample):
    pos_space = np.all(row == neg_sample, axis=1)
    left_edge = np.argmax(pos_space)
    right_edge = np.argmax(pos_space[::-1])
    return _row_data(left_Edge, right_edge, neg_sample)

