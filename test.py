from __future__ import print_function
import os
import itertools
import imageio
import numpy as np
from crash_kiss import util, foreground, crash
import kiss
import pytest


def _get_test_img():
    img = imageio.imread(os.path.join(os.path.dirname(__file__), "face.jpg"))
    # cut off bottom non-white edge
    # make sure it's not square to catch "wrong axis" bugs
    img = img[:-10:, :-7:]
    # make the top rows of the image pure white
    img[:5] = [255, 255, 255]
    return img


def test_test_image():
    """Make sure our test image, face.jpg, is white on the edges"""
    img = _get_test_img()
    all = np.all(img[::, :10:] > 240, axis=2)
    assert np.all(all)


### Test kiss.py util functions ###

def test_chunks():
    stuff = range(10)
    chunks = list(kiss._chunks(stuff, 2))
    assert chunks == [range(5), range(5, 10)]


def test_odd_chunks():
    """Make sure that leftover pieces of the iterable
    get tacked on to the last element."""
    stuff = range(13)  # odd, so one list will be longer than the other
    chunks = list(kiss._chunks(stuff, 2))
    assert chunks == [range(6), range(6, 13)]
    stuff = range(15)
    chunks = list(kiss._chunks(stuff, 4))
    assert chunks == [range(3), range(3, 6), range(6, 9), range(9, 15)]


### Test crashing two subjects towards the center

def test_conservation_of_foreground():
    """Ensure that `center_crash` doesn't overwrite pixels or somehow
    delete them when it crashes a simple image"""
    img = util.combine_images([_get_test_img(), _get_test_img()])
    params = crash.CrashParams(50, 10, 255, [0, 1, 2])
    total_fg_area, bounds = foreground.find_foreground(img, params)
    total_fg_pixels = np.sum(total_fg_area)
    crash.center_crash(img, total_fg_area, bounds, 0)#(0, 0, 0))
    total_fg_area_after = foreground.find_foreground(img, params)
    total_fg_pixels_after = np.sum(total_fg_area)
    assert total_fg_pixels_after == total_fg_pixels


@pytest.mark.skip
def test_center_crash_mov_crash_1():
    """Test a crash where the foreground intersects the middle.
    In this case, the foreground in the middle should be fixed
    in place. The data on either side will collapse based on how much
    negative space is present. So the negative space will "collapse"
    by a certain amount and the outer foreground will shift inward
    by that amount."""
    data_in =  _ints("0000 1010 2110 0010 1011")
    data_out = _ints("0000 0011 2110 1010 1111")  # expected result
    crash_data, row_data = _row(data_in, )
    params = crash.CrashParams(50, 10, 255, [0, 1, 2])
    total_fg_area, bounds = foreground.find_foreground(row_data, params)
    assert np.all(row_data.irow == data_in)  # just a sanity check
    #crash.mov_crash(crash_data, row_data)  # crash the row
    crash.center_crash(data_in, total_fg_area, bounds)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def test_center_crash_mov_crash_2():
    """Crash a row of even length with interleaved background space"""
    # NOTE: middle is here          |
    data_in =  _ints("00000 03010 00000 00010 03011")
    data_out = _ints("00000 00000 03110 03011 00000")
    crash_data, row_data = _row(data_in, )
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_crash(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def test_center_crash_mov_empty_fg():
    data_in =  _ints("11112 00000 00000 00000 31111")
    data_out = _ints("00011 11200 00000 00311 11100")
    crash_data, row_data = _row(data_in, 3)  # restricted depth
    assert not np.any(row_data.frow)
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_empty_fg(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def test_center_crash_mov_left_overshoot():
    """Test the case where the only foreground present is
    on the left side and the foreground will overshoot the center line"""
    data_in =  _ints("00000 00222 20000 00000 31111")
    data_out = _ints("00000 00000 22220 00311 11100")
    crash_data, row_data = _row(data_in, 3)  # restricted depth
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_left_overshoot(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def _test_center_crash_mov_right_overshoot():
    """Test the case where the only foreground present is
    on the left side and the foreground will overshoot the center line"""
    data_in =  _ints("11113 00000 00002 22200 00000")
    data_out = _ints("00011 11300 02222 00000 00000")
    crash_data, row_data = _row(data_in, 3)  # restricted depth
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_right_overshoot(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def test_center_crash_mov_near_collision():
    data_in =  _ints("00000 00110 00000 02200 00000")
    data_out = _ints("00000 00000 11022 00000 00000")
    crash_data, row_data = _row(data_in, 3)  # restricted depth
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_near_collision(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    assert np.all(row_data.irow == data_out)


@pytest.mark.skip
def test_double_overshoot():
    """Make sure things don't get compeltely messed up when we
    have both subjects overshooting the center"""
    data_in =  _ints("00000 00010 21330 10000 00000")
    data_out = _ints("00000 00001 21331 00000 00000")
    crash_data, row_data = _row(data_in, 3)  # restricted depth
    assert np.all(row_data.irow == data_in)  # just a sanity check
    crash.mov_near_collision(crash_data, row_data)  # crash the row
    _clear(crash_data, row_data)
    print("".join(map(str, row_data.irow)))
    print("".join(map(str, data_out)))
    assert np.all(row_data.irow == data_out)


def test_contiguous_chunks():
    img = np.arange(10)
    mask = np.zeros((10,))
    mask[3: 5] = 1
    mask[6] = 1
    mask[9] = 1
    chunks = crash._contiguous_chunks(mask, img)
    chunks = [list(chunk[0]) for chunk in chunks]
    assert chunks == [[3, 4], [6], [9]]


@pytest.mark.skip
def test_center_obstructed():
    O, L = [255, 255, 255], [0, 0, 0]
    img = np.ndarray(
        (3, 9, 3), int,
        np.array([[O, O, O, L, L, L, O, O, O],
                  [O, O, L, L, L, O, O, O, O],
                  [O, O, O, L, L, L, O, O, O],
        ]))
    params = crash.CrashParams(5, 5, 0xFF, [0, 1, 2])
    fg, bounds = foreground.find_foreground(img, params)


def _row(data, max_depth=None):
    """Make a `crash._row_data` namedtuple instance based on a list of
    ones (or other numbers) and zeros to represent a background mask
    (where 0==background and foreground>=1)"""
    max_depth = max_depth or len(data)
    img_row = np.ndarray(shape=(len(data),), dtype=np.uint8)
    img_row[:] = data

    # assemble _crash_data namedtuple
    bounds = foreground.get_fg_bounds(len(img_row), max_depth)
    start, stop, fg_mid, max_depth = bounds
    fg_row = img_row[start: stop] != 0
    fg_l = fg_mid - max_depth
    fg_r = fg_mid + max_depth
    mid_left = start + max_depth
    center = start + 2 * max_depth
    mid_right = center + max_depth
    side_len = max_depth * 2
    crash_data = crash._crash_data(
        start, stop, fg_mid, max_depth, fg_l, fg_r, mid_left,
        center, mid_right, side_len)

    # assemble _row_data namedtuple
    ls = fg_row[:fg_mid:-1].argmax()  # distance from center to left subject
    rs = fg_row[fg_mid:].argmax()  # distance from center to right subject
    row_data = crash._row_data(img_row, ls, rs, fg_row)
    return crash_data, row_data


def _ints(data):
    """Remove spaces, convert to list of ints"""
    return list(map(int, data.replace(" ", "")))


def _clear(crash_data, row_data):
    depth = crash_data.max_depth
    row_data.irow[:depth] = 0
    row_data.irow[-depth + 1:] = 0

