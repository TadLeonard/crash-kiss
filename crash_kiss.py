from __future__ import print_function, division
import argparse
from collections import namedtuple
from functools import partial
import mahotas
import numpy as np


##################
# Mutating images

def center_smash_image(edges, img):
    """The original "crash kiss" method used to smash two people's
    profiles together in a grotesque "kiss". The rule is: move the
    subjects of each row towards each other until they touch.
    Write over the vacated space with whatever the row's negative space
    is (probably white or transparent pixels)."""
    #for row_data_group, row in _iter_subject_rows(edges, img):
    #TODO: not yet working


def wall_smash_image(edges, img):
    """Mutates a numpy array of an image so that the subject
    is smashed to an edge of the image boarders."""
    for row_data_group, row in _iter_subject_rows(edges, img):
        _shift_row_right(row_data_group, row)


def _shift_row_right(edges, row):
    target_idx = row.shape[0]  # shift to end of img initially
    for edge in edges:
        sub_data_l, sub_data_r = _get_shifted_indices(edge, target_idx)
        row[sub_data_l: sub_data_r] = row[edge.left: edge.right]
        target_idx = sub_data_l

    # We've shifted the subject(s) over, now we need to fill
    # the rest of the row with negative space
    row[:sub_data_l] = edge.neg_space


def _get_shifted_indices(edge, target_idx):
    rowlen = edge.right - edge.left
    sub_data_l = target_idx - rowlen
    sub_data_r = target_idx
    return sub_data_l, sub_data_r


def _iter_subject_rows(edges, img):
    """Iterate over edges, pixel rows that contain foreground info
    (i.e. not all whitespace)."""
    #TODO: This doesn't make sense for the multiple subject case
    for row_data_group, row in zip(edges, img):
        if any(r - l for l, r, _ in row_data_group):
            yield row_data_group, row


def gen_char_edges(edges, char=u"@", scale=1.0):
    for edge_group in edges:
        prev_right = 0
        for left, right, _ in edge_group:
            if not right:
                continue
            yield u" " * int((left - prev_right) * scale)
            yield char * int((right - left) * scale)
            prev_right = right
        yield u"\n"


_L_EDGE_REVEAL = [0, 255, 0]
_R_EDGE_REVEAL = [255, 0, 0]

def reveal_edges(edges, img, inplace=True):
    """Highlights the edges of an image with green (left edge)
    and red (right edge)"""
    new_img = img.copy() if not inplace else img
    for row, edge_group in zip(new_img, edges):
        for l, r, neg in edge_group:
            row[l-1:l+1] = _L_EDGE_REVEAL
            row[r-1:r+1] = _R_EDGE_REVEAL
    return new_img


##########################
# Finding subject's edges

row_data = namedtuple("row_data", "left_idx right_idx neg_space")
DEFAULT_THRESH = 15
DEFAULT_NEG_SAMPLE = 5


def iter_subject_edges(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Finds the edges of the subject for each row of pixels. This assumes that
    the subject is on a background with significantly different RGB values.
    Yields `row_data` instances."""
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
    return row_data(left_edge, right_edge, neg_space_l)


def _find_edge_indices_simple(row, neg_sample):
    pos_space = np.all(row == neg_sample, axis=1)
    left_edge = np.argmax(pos_space)
    right_edge = np.argmax(pos_space[::-1])
    return row_data(left_Edge, right_edge, neg_sample)


def combine_images(imgs, axis="horizontal"):
    axis = 1 if axis == "horizontal" else 0
    combined = imgs[0]
    for img in imgs[1:]:
        combined = np.append(combined, img, axis=axis)
    return combined


##############
# Arg parsing

parser = argparse.ArgumentParser(description="Crash images into things")
parser.add_argument("targets", nargs="+")
parser.add_argument("--threshold", default=DEFAULT_THRESH, type=int)
parser.add_argument("--smash", type=str,
                    choices=("left", "right"), default=None)
parser.add_argument("--output-type", type=str,
                    choices=("char", "gif", "img"), default="char")
parser.add_argument("--char-scale", type=float, default=1.0)
parser.add_argument("-l", "--smash_left", action="store_true",
                    default=False)
parser.add_argument("-v", "--smash_up", action="store_true",
                    default=False)
parser.add_argument("-a", "--combine-axis",
                    choices=("vertical", "horizontal"), default="horizontal")
parser.add_argument("-o", "--outfile", default=None)
parser.add_argument("-c", "--charout", action="store_true")
parser.add_argument("-e", "--reveal-edges", action="store_true")


def run():
    args = parser.parse_args()
    imgs = map(mahotas.imread, args.targets)
    if not imgs:
        parser.error("Must pass in one or more paths to images")
    if len(imgs) == 1:
        img = imgs[0]
    elif len(imgs) > 1:
        img = combine_images(imgs, axis=args.combine_axis)

    # After this point we're always working with one big, combined image
    if args.smash:
        if args.smash == "left":
            raise NotImplementedError("Can't smash left yet")
        elif args.smash == "right":
            edges = iter_subject_edges(img)
            wall_smash_image(img)
    else:
        edges = iter_subject_edges(img)  # TODO: hack for testing

    # Various things to do with our manipulated image
    if args.reveal_edges:
        reveal_edges(edges, img)
    if args.outfile:
        mahotas.imsave(args.outfile, reveal_edges(edges, img))
    if args.charout:
        char_img = gen_char_edges(edges, char=u"@", scale=args.char_scale)
        print("".join(char_img))



if __name__ == "__main__":
    run()
