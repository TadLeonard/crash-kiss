"""Configuration of edge finding behavior"""


def config(**kw_overrides):
    invalid = list(set(kw_overrides) - set(_config_defaults))
    if invalid:
        raise Exception("Invalid config keys: {0}".format(invalid))
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf


WHITE = 255  # the default white background value
BLACK = 0
PURPLE = [128, 0, 128]
TEAL = [40, 100, 140]
AUTO = "auto"  # key used to auto-gather the background
FULL_DEPTH = 0xFFFF


_config_defaults = dict(
    bg_sample_size=5,
    threshold=10,
    bg_change_tolerance=7,
    relative_sides=("left", "right"),
    chunksize=300,
    bg_value=WHITE,
    rgb_select=None,
    max_depth=FULL_DEPTH,
)


