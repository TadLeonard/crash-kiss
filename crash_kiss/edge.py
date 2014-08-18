"""
Functions for finding the edges of a subject in an
image-as-a-numpy-array
"""
from __future__ import division
from collections import namedtuple
import numpy as np



DEFAULT_THRESH = 15
DEFAULT_NEG_SAMPLE = 5

_row_data = namedtuple("row_data", "left_idx right_idx neg_space")


def iter_subject_edges(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Finds the edges of the subject for each row of pixels. This assumes that
    the subject is on a background with significantly different RGB values.
    Yields `_row_data` instances."""
    for row in img:
        yield (_find_edge_indices(row, neg_sample, threshold),)


def iter_all_subject_edges(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Like `iter_subject_edges`, but for any number of
    subjects. Detects ALL edges based on initial (leftmost) whitespace."""
    for row in img:
        neg_space = np.mean(row[:neg_sample], axis=0)
        pos_space = np.all(np.abs(row - neg_space) > threshold, axis=1)


def _find_edge_indices(row, neg_sample_size, threshold):
    """Find edges of a single subject. Naively assume only a left
    and a right edge are present and that there's nothing in between."""
    neg_space_l = np.mean(row[:neg_sample_size], axis=0)
    neg_space_r = np.mean(row[-neg_sample_size:], axis=0)
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
    return _row_data(left_edge, right_edge, neg_space_l)


def _find_edge_indices_simple(row, neg_sample):
    pos_space = np.all(row == neg_sample, axis=1)
    left_edge = np.argmax(pos_space)
    right_edge = np.argmax(pos_space[::-1])
    return _row_data(left_Edge, right_edge, neg_sample)

