"""Parameters for subject vs. background
thresholds, background values, and more"""


WHITE = 255  # the default white background value
BLACK = 0
AUTO = "auto"  # key used to auto-gather the background
FULL_DEPTH = 0xFFFF


# default crash parameters
THRESHOLD = 10
BG_VALUE = 0xFF
RGB_SELECT = None
MAX_DEPTH = FULL_DEPTH


# input, output filename configuration
OUTPUT_SUFFIX = "crashed"  # by default, add "_crashed" to outfiles
INPUT_SUFFIX = ".jpg"  # in auto run mode, look for "*.jpg" inputs
LAST_CRASH = "LAST_CRASH.{0}"  # to be formatted like `.format("jpg")`

