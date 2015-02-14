"""Parameters for subject vs. background 
thresholds, background values, and more"""


def config(**kw_overrides):
    """Construct a config `dict` with optional overrides"""
    invalid = list(set(kw_overrides) - set(_config_defaults))
    if invalid:
        raise Exception("Invalid config keys: {0}".format(invalid))
    conf = dict(_config_defaults)
    conf.update(kw_overrides)
    return conf


WHITE = 255  # the default white background value
BLACK = 0
AUTO = "auto"  # key used to auto-gather the background
FULL_DEPTH = 0xFFFF


_config_defaults = dict(
    bg_sample_size=5,  # for outer edge finding approach
    threshold=10,  # default required difference from background value
    bg_change_tolerance=7,  # for outer edge finding approach
    relative_sides=("left", "right"),  # for outer edge finding approach
    chunksize=300,  # for outer edge finding approach
    bg_value=WHITE,  # default background value is white (255,255,255)
    rgb_select=None,  # selection of RGB(A) axis (all of them by default)
    max_depth=FULL_DEPTH,  # crash as far as possible by default
)


# input, output filename configuration
DEFAULT_OUTPUT_SUFFIX = "crashed"  # by default, add "_crashed" to outfiles
DEFAULT_INPUT_SUFFIX = ".jpg"  # in auto run mode, look for "*.jpg" inputs
DEFAULT_LATEST = "LAST_CRASH.{0}"  # to be formatted like `.format("jpg")`

