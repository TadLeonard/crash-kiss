import dataclasses
import shutil
import numpy as np

from typing import Union, Tuple
from pathlib import Path

from moviepy.editor import VideoClip
import moviepy.video.fx.all as vfx
import yaspin.spinners

from . import foreground
from . import crash
from . import util


@dataclasses.dataclass
class AnimationConfig:
    progress_spinner: "yaspin.Spinner"
    max_depth: int = 350
    stepsize: int = 1
    threshold: int = 15
    bg_value: Union[int, Tuple[int, int, int]] = (0, 0, 0)
    fps: int = 60
    reveal_foreground: bool = False
    reveal_background: bool = False
    first_frame_pause: float = 2.0
    last_frame_pause: float = 5.0
    output_path: Path = Path(".")
    latest_output_path: Path = Path("crash.mp4")


def make_crash_video(image_path: Path, config: AnimationConfig) -> Path:
    """Creates an animated crash based on a still image input"""
    config.progress_spinner.spinner = yaspin.spinners.Spinners.hearts
    config.progress_spinner.text = "Preparing to crash"
    img = util.read_img(str(image_path))
    bounds = foreground.get_fg_bounds(img.shape[1], config.max_depth)
    max_depth = bounds.max_depth
    crash_params = crash.CrashParams(max_depth, config.threshold, config.bg_value)
    depths = range(max_depth, -config.stepsize, -config.stepsize)
    depths = [d for d in depths if d > 0]
    depths.append(0)
    n_frames = len(depths)

    fps = config.fps
    duration = len(depths) / fps
    fg, bounds = foreground.find_foreground(img, crash_params)
    start, stop = bounds[:2]
    temp_fg = np.zeros(img.shape[:2], dtype=np.uint8)
    temp_fg[:, start:stop] = fg
    first_image = img.copy()
    last_image = _process_img(img.copy(), fg.copy(), bounds, config)
    last_frame_path = (
        (config.output_path / image_path.name)
        .absolute()
        .with_stem(image_path.stem + "_last_frame")
        .with_suffix(".tiff")
    )
    util.save_img(last_frame_path, last_image)
    foreground_image_path = (
        (config.output_path / image_path.name)
        .absolute()
        .with_stem(image_path.stem + "_foreground")
        .with_suffix(".tiff")
    )
    util.save_img(foreground_image_path, fg.astype(np.uint8) * 255)

    def make_frame(time):
        frame_no = int(round(time * fps))
        if frame_no >= n_frames - 1:
            return last_image
        elif frame_no <= 1:
            return first_image

        depth = depths[-frame_no]
        this_img = img.copy()

        if depth == 0:
            # util.save_img("first.png", this_img)
            return this_img

        if depth:
            params = crash.CrashParams(depth, config.threshold, config.bg_value)
            new_fg, new_bounds = foreground.trim_foreground(this_img, fg, params)
            new_img = _process_img(this_img, new_fg, new_bounds, config)
        else:
            new_img = this_img
        config.progress_spinner.text = f"Crashed {frame_no}/{len(depths)} frames"
        return new_img

    # Write out the video file to the output directory
    animation = (
        VideoClip(make_frame, duration=duration)
        .fx(
            vfx.freeze,
            t=1.0 / 60.0,
            freeze_duration=config.first_frame_pause,
            padding_end=1.0 / 60.0,
        )
        .fx(
            vfx.freeze,
            t="end",
            freeze_duration=config.last_frame_pause,
            padding_end=1.0 / 60.0,
        )
    )
    video_path = (config.output_path / image_path.name).absolute().with_suffix(".mp4")
    animation.write_videofile(str(video_path), fps=fps, logger=None)

    # Copy over the latest video file
    shutil.copy(video_path, config.latest_output_path)
    return video_path


def _process_img(
    img: "np.ndarray", foreground: "np.ndarray", bounds, config: AnimationConfig
):
    """Does none or more of several things to `img` based on the given
    `AnimationConfig` options."""
    if config.reveal_foreground:
        util.reveal_foreground(img, foreground, bounds)
    if config.reveal_background:
        util.reveal_background(img, foreground, bounds)
    crash.center_crash(img, foreground, bounds, config.bg_value)
    return img
