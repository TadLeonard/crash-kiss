import os
import itertools
import imread
import numpy as np
import pytest
from crash_kiss import outer_edge, edge, util
from crash_kiss.config import config


def _get_test_img():
    img = imread.imread(os.path.join(os.path.dirname(__file__), "face.jpg"))
    # cut off bottom non-white edge
    # make sure it's not square to catch "wrong axis" bugs
    img = img[:-10:, :-7:]
    # make the top rows of the image pure white
    img[:5] = [255, 255, 255]
    return img


def test_subject_default_sides():
    """Make sure `Subject` defaults to having four `Side` instances"""
    img = _get_test_img()
    subj = outer_edge.Subject(img=img)
    assert len(list(subj)) == 2


def test_test_image():
    """Make sure our test image, face.jpg, is white on the edges"""
    img = _get_test_img()
    all = np.all(img[::, :10:] > 240, axis=2)
    assert np.all(all)


def test_subject_config():
    """Make sure config values are consumed by `Subject` correctly"""
    s = outer_edge.Subject(config=config(threshold=70, bg_change_tolerance=55))
    assert s._config["threshold"] == 70
    assert s._config["bg_change_tolerance"] == 55


def test_side_config():
    """Ensure that `Side` objs contained by `Subject` are given
    their container's config"""
    s = outer_edge.Subject(config=config(threshold=70, bg_change_tolerance=55))
    assert s.left._config["threshold"] == 70
    assert s.left._config["bg_change_tolerance"] == 55
    c = s._config
    assert all(side._config == c for side in s)
    

def test_no_edge():
    """Make sure parts of the image with no edge get an edge index of 0"""
    img = _get_test_img()
    assert np.all(img[:5] == [255, 255, 255])  # top rows aughtta be white
    sub = outer_edge.Subject(img=img)
    # Edges are masked arrays. A True value in a masked array indicates that
    # the value is masked, so we want white rows's edges to be masked
    # or == True
    assert np.all(sub.left.edge[:5].mask)
    

def test_edge_below_threshold():
    """Ensure that parts of the image that are similar to the background
    are not detected as edges of the foreground"""
    img = _get_test_img()
    img[::, 4:6] = [230, 230, 230]
    conf = config(threshold=60, bg_sample_size=1)
    sub = outer_edge.Subject(img=img, config=conf)
    nz_edge = sub.left.edge != 0
    nz_edge = sub.left.edge[nz_edge]
    assert np.all(nz_edge >= 5)
    

def test_edge_above_threshold():
    """Set a part of the image to be very unlike the background
    so that it's detected as an edge of the foreground"""
    img = _get_test_img()
    img[::, 4:6:] = [10, 10, 10]  # a very dark vert. line on the left side
    conf = config(threshold=60, bg_sample_size=1)  # large threshold
    subj = outer_edge.Subject(img=img, config=conf)
    l_edge = subj.left.edge
    assert np.all(l_edge >= 4)


def test_edge_below_threshold_2():
    img = _get_test_img()
    img[::, 4:6:] = [30, 30, 30]  # a very dark vert. line on the left side
    img[80, 80] = [0, 0, 0]  # black dot near the middle
    silly_config = config(threshold=227, bg_sample_size=1)
    huge_threshold = outer_edge.Subject(img=img, config=silly_config)
    # the dark line should not be picked up as an edge due to the huge thresh
    assert np.all(huge_threshold.left.edge[:5].mask)
    # still, at least one part of the image is completely black...
    assert not np.all(huge_threshold.left.edge.mask)
    assert huge_threshold.left.edge[80] == 80  # we've located the black dot
    img[:8:, 4:6:] = [0, 0, 0]  # black line!
    huge_threshold = outer_edge.Subject(img=img, config=silly_config)
    # the black line is dark enough (difference is 255, which is > 247)
    assert np.all(huge_threshold.left.edge[:5] != 0)


def test_edge_at_0():
    """Ensure that we don't repeat the mistakes of issue #2.
    Valid edges can be found at index == 0."""
    img = _get_test_img()
    img[::, 0] = [0, 0, 0]  # black line at very left edge
    img[5, 0] = [255, 255, 255]  # ...except one white dot
    conf = config(bg_value=255)
    sub = outer_edge.Subject(img=img, config=conf)
    e = sub.left.edge
    assert np.all(e[:5] == 0)
    assert np.all(e[6:] == 0)
    assert e[5] != 0


def test_column_blocks():
    img = _get_test_img()
    chunksize = 10
    chunks = list(outer_edge._column_blocks(img, chunksize))
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
    chunks = list(outer_edge._column_blocks(img, chunksize=10))
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
    #TODO: At one point, 1D backgrounds seemed to work for 
    # 2D views of the image. Later on, they didn't! Why!? 
    img = _get_test_img()
    no_red = config(rgb_select=[1,2], bg_value="auto")
    only_red = config(rgb_select=[0])
    cool = outer_edge.Subject(img=img, config=no_red)
    hot = outer_edge.Subject(img=img, config=only_red)
    assert cool.left.rgb_view.shape[2] == 2
    assert len(hot.left.rgb_view.shape) == 2
    assert cool.left.background.shape[2] == 2
    assert len(hot.left.background.shape) == 2


def _get_test_rgb_views():
    """Make sure the RGB select feature selects the correct colors"""
    img = _get_test_img()
    configs = _get_all_rgb_configs()
    img[:, :, 0] = 0  # R -> 0
    img[:, :, 1] = 1  # B -> 1
    img[:, :, 2] = 2  # G -> 2
    subjs = [outer_edge.Subject(img=img, config=c) for c in configs] 
    return zip(subjs, configs)


def _get_all_rgb_configs():
    colors = range(3)
    ones = list(itertools.combinations(colors, 1))
    twos = list(itertools.combinations(colors, 2))
    threes = list(itertools.combinations(colors, 3))
    all_selects = list(itertools.chain(ones, twos, threes))
    return [config(rgb_select=s) for s in all_selects]
   

def test_test_rgb_views():
    """Make sure we're not testing nothing"""
    assert len(_get_test_rgb_views()) == 7


def test_rgb_view_bg_value():
    """Make sure each `Side` background is made up of only colors
    specified in the 'rgb_select' config value"""
    for sub, conf in _get_test_rgb_views():
        conf["bg_value"] = "auto"
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
    

def test_rgb_view_edge():
    """Ensure that edge finding behavior changes based on
    a restricted RGB view"""
    img = _get_test_img()
    only_red = config(rgb_select=[0], bg_sample_size=1)
    no_red = config(rgb_select=[1, 2], bg_sample_size=1)
    only_green = config(rgb_select=[1], bg_sample_size=1)
    only_blue = config(rgb_select=[2], bg_sample_size=1)
    red = outer_edge.Subject(img=img, config=only_red)
    cold = outer_edge.Subject(img=img, config=no_red)
    green = outer_edge.Subject(img=img, config=only_green)
    blue = outer_edge.Subject(img=img, config=only_blue)
    img[:] = 255
    img[:, 5] = [0, 255, 255]
    img[:, 10] = [255, 0, 255]
    assert np.all(red.left.edge == 5)
    assert not np.all(cold.left.edge.mask)
    assert np.all(green.left.edge == 10)
    assert np.all(blue.left.edge.mask)  # no blue in the image
        

def test_rgb_view_nocopy():
    """Ensure that fancing indexing is avoided given a slicable
    combination of RG and B. We're making sure that 
    `Subject.<side>.rgb_view` is a VIEW of `Subject.img` instead of a copy."""
    img = _get_test_img()
    configs = _get_all_rgb_configs()
    for i, config in enumerate(configs):
        s = outer_edge.Subject(img, config)
        view = s.left.rgb_view
        img[:] = i * 10
        assert np.all(view == (i * 10))
    

def test_bad_edge_config():
    config(threshold=10)  # okay
    with pytest.raises(Exception):
        config(fleshold=10)  # not okay
   

### Test smashing two subjects towards the center


def test_conservation_of_foreground():
    """Ensure that `center_smash` doesn't overwrite pixels or somehow 
    delete them when it smashes a simple image"""
    img = util.combine_images([_get_test_img(), _get_test_img()])
    view, bounds = edge.get_foreground_area(img, 50)
    total_fg_area = edge.find_foreground(img, 255, 10)
    total_fg_pixels = np.sum(total_fg_area)  
    fg = edge.find_foreground(view, 255, 10)
    edge.center_smash(img, fg, bounds)
    total_fg_area_after = edge.find_foreground(img, 255, 10)
    total_fg_pixels_after = np.sum(total_fg_area)  
    assert total_fg_pixels_after == total_fg_pixels

