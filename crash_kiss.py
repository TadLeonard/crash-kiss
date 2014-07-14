from __future__ import print_function, division
from functools import partial
from collections import namedtuple
import argparse
import mahotas
import numpy as np


##################
# Mutating images

def wall_smash_image(edges, shift_right=True):
    """Mutates a numpy array of an image so that the subject
    is smashed to the right or left edge."""
    #TODO: broken temporarily
    for left, right, negspace in edges:
        rowlen = right - left
        if not rowlen:
            continue
        row[-rowlen:] = row[left: right]
        row[:-rowlen] = negspace


def gen_ascii_edges(edges, char=u"@"):
    for edge_group in edges:
        prev_right = 0
        for left, right, _ in edge_group:
            if not right:
                continue
            yield u" " * (left - prev_right)
            yield char * (right - left)
            prev_right = right
        yield u"\n"


##########################
# Finding subject's edges

row_data = namedtuple("row_data", "left_idx right_idx neg_space")
DEFAULT_THRESH = 15
DEFAULT_NEG_SAMPLE = 15


def iter_subject_edges(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Finds the edges of the subject for each row of pixels. This assumes that
    the subject is on a background with significantly different RGB values.
    Yields `row_data` instances."""
    for row in img:
        yield (_find_edge_indices(row, neg_sample, threshold),)


def iter_two_subject_edges(
        img, neg_sample=DEFAULT_NEG_SAMPLE, threshold=DEFAULT_THRESH):
    """Like `iter_subject_edges`, but for two subjects. Naively halves the
    image and searches both halves for edges."""
    width = img.shape[1]
    halfway = width // 2
    find_edges = partial(_find_edge_indices,
                         neg_sample=neg_sample, threshold=threshold)
    for row in img:
        yield map(find_edges, (row[:halfway], row[halfway:]))


def _find_edge_indices(row, neg_sample, threshold):
    neg_space = np.mean(row[:neg_sample], axis=0)
    pos_space = np.all(np.abs(row - neg_space) > threshold, axis=1)
    left_edge = np.argmax(pos_space)
    right_edge = np.argmax(pos_space[::-1])
    if right_edge:
        width = row.shape[0]
        right_edge = width - right_edge
    return row_data(left_edge, right_edge, neg_space)



##############
# Arg parsing

parser = argparse.ArgumentParser(description="Crash images into things")
parser.add_argument("target")
parser.add_argument("--output", default=None)
parser.add_argument("--threshold", default=DEFAULT_THRESH, type=int)
parser.add_argument("--smash", type=str,
                    choices=("left", "right", "center"), default=None)
parser.add_argument("--output-type", type=str,
                    choices=("ascii", "gif", "img"), default="ascii")


def run():
    args = parser.parse_args()
    img = mahotas.imread(args.target)
    #im2 = img[::, ::-1]

    if args.smash:
        if args.smash == "left":
            raise NotImplementedError("Can't smash left yet")
        elif args.smash == "right":
            edges = iter_subject_edges(img)
            wall_smash_image(img)
        elif args.smash == "center":
            edges = iter_two_subject_edges(img)
    else:
        edges = iter_subject_edges(img)  # TODO: hack for testing
    mahotas.imsave("bleeeeup", img)
    print("".join(gen_ascii_edges(edges, char=u"@")))


if __name__ == "__main__":
    run()
