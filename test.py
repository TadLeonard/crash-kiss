import os
import itertools
import mahotas
import numpy as np
import pytest
from crash_kiss import edge


def _get_test_img():
    img = mahotas.imread(os.path.join(os.path.dirname(__file__), "face.jpg"))
    # cut off bottom non-white edge
    # make sure it's not square to catch "wrong axis" bugs
    img = img[:-10:, :-7:]
    # make the top rows of the image pure white
    img[:5] = [255, 255, 255]
    return img


def test_subject_default_sides():
    """Make sure `Subject` defaults to having four `Side` instances"""
    img = _get_test_img()
    subj = edge.Subject(img=img)
    assert len(list(subj)) == 2


def test_test_image():
    """Make sure our test image, face.jpg, is white on the edges"""
    img = _get_test_img()
    all = np.all(img[::, :10:] > 240, axis=2)
    assert np.all(all)


def test_subject_config():
    """Make sure config values are consumed by `Subject` correctly"""
    s = edge.Subject(config=edge.config(threshold=70, bg_change_tolerance=55))
    assert s._config["threshold"] == 70
    assert s._config["bg_change_tolerance"] == 55


def test_side_config():
    """Ensure that `Side` objs contained by `Subject` are given
    their container's config"""
    s = edge.Subject(config=edge.config(threshold=70, bg_change_tolerance=55))
    assert s.left._config["threshold"] == 70
    assert s.left._config["bg_change_tolerance"] == 55
    c = s._config
    assert all(side._config == c for side in s)
    

def test_no_edge():
    """Make sure parts of the image with no edge get an edge index of 0"""
    img = _get_test_img()
    assert np.all(img[:5] == [255, 255, 255])  # top rows aughtta be white
    sub = edge.Subject(img=img)
    # Edges are masked arrays. A True value in a masked array indicates that
    # the value is masked, so we want white rows's edges to be masked
    # or == True
    assert np.all(sub.left.edge[:5] == 0)
    

def test_edge_below_threshold():
    """Ensure that parts of the image that are similar to the background
    are not detected as edges of the foreground"""
    img = _get_test_img()
    img[::, 4:6] = [230, 230, 230]
    config = edge.config(threshold=60, bg_sample_size=1)
    sub = edge.Subject(img=img, config=config)
    nz_edge = sub.left.edge != 0
    nz_edge = sub.left.edge[nz_edge]
    assert np.all(nz_edge >= 5)
    

def test_edge_above_threshold():
    """Set a part of the image to be very unlike the background
    so that it's detected as an edge of the foreground"""
    img = _get_test_img()
    img[::, 4:6:] = [10, 10, 10]  # a very dark vert. line on the left side
    config = edge.config(threshold=60, bg_sample_size=1)  # large threshold
    subj = edge.Subject(img=img, config=config)
    l_edge = subj.left.edge
    assert np.all(l_edge >= 4)


def test_edge_below_threshold_2():
    img = _get_test_img()
    img[::, 4:6:] = [30, 30, 30]  # a very dark vert. line on the left side
    img[80, 80] = [0, 0, 0]  # black dot near the middle
    silly_config = edge.config(threshold=227, bg_sample_size=1)
    huge_threshold = edge.Subject(img=img, config=silly_config)
    # the dark line should not be picked up as an edge due to the huge thresh
    assert np.all(huge_threshold.left.edge[:5] == 0)
    # still, at least one part of the image is completely black...
    assert not np.all(huge_threshold.left.edge == 0)
    assert huge_threshold.left.edge[80] == 80  # we've located the black dot
    img[:8:, 4:6:] = [0, 0, 0]  # black line!
    huge_threshold = edge.Subject(img=img, config=silly_config)
    # the black line is dark enough (difference is 255, which is > 247)
    assert np.all(huge_threshold.left.edge[:5] != 0)


def test_column_blocks():
    img = _get_test_img()
    chunksize = 10
    chunks = list(edge._column_blocks(img, chunksize))
    for n, (chunk, _) in enumerate(chunks):
        color = n * 10
        chunk[::] = color
    for n, (chunk, _) in enumerate(chunks):
        color = n * 10
        rows = np.arange(chunk.shape[0])
        assert np.all(chunk[rows, :-1] == color)
    

def test_overlapping_column_blocks():
    """Make sure that columns sliced from the image overlap by
    one pixel so that we don't see issue #1 again"""
    img = _get_test_img()
    chunks = list(edge._column_blocks(img, chunksize=10))
    for n, (chunk, _) in enumerate(chunks):
        color = n * 10
        chunk[::] = color
    assert chunks, "empty loop"
    for c, _ in chunks[:-1]:
        assert np.median(c[::, :-1]) - np.median(c[::, -1]) == -10


def test_rbg_select_shape():
    """Make sure the RGB select feature creates a view of the image
    that is the correct shape. Each side's background sampling should
    also have a restricted third axis."""
    img = _get_test_img()
    no_red = edge.config(rgb_select=[1,2])
    only_red = edge.config(rgb_select=[0])
    cool = edge.Subject(img=img, config=no_red)
    hot = edge.Subject(img=img, config=only_red)
    assert cool.left.rgb_view.shape[2] == 2
    assert hot.left.rgb_view.shape[2] == 1
    assert cool.left.background.shape[2] == 2
    assert hot.left.background.shape[2] == 1


def _get_test_rgb_views():
    """Make sure the RGB select feature selects the correct colors"""
    img = _get_test_img()
    colors = range(3)
    ones = list(itertools.combinations(colors, 1))
    twos = list(itertools.combinations(colors, 2))
    threes = list(itertools.combinations(colors, 3))
    all_selects = list(itertools.chain(ones, twos, threes))
    configs = [edge.config(rgb_select=s) for s in all_selects]
    img[:, :, 0] = 0  # R -> 0
    img[:, :, 1] = 1  # B -> 1
    img[:, :, 2] = 2  # G -> 2
    subjs = [edge.Subject(img=img, config=c) for c in configs] 
    return zip(subjs, configs)
   

def test_test_rgb_views():
    """Make sure we're not testing nothing"""
    assert len(_get_test_rgb_views()) == 7


def test_rgb_view_bg_value():
    """Make sure each `Side` background is made up of only colors
    specified in the 'rgb_select' config value"""
    for sub, conf in _get_test_rgb_views():
        bgs = [side.background for side in sub]
        select = conf["rgb_select"]
        for bg in bgs:
            assert np.all(bg == select)
    

def test_rgb_view_view_value():
    """Make sure each `Side` view is restricted based on the
    'rgb_select' config value"""
    for sub, conf in _get_test_rgb_views():
        views = [side.rgb_view for side in sub]
        select = conf["rgb_select"]
        for view in views:
            assert np.all(view == select)
    

#def test_rgb_view_edge():
#    """Ensure that edge finding behavior changes based on
#    a restricted RGB view"""


def test_bad_edge_config():
    edge.config(threshold=10)  # okay
    with pytest.raises(Exception):
        edge.config(fleshold=10)  # not okay
   

