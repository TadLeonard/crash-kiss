# Crash Kiss: automated intimacy
This is an art collaboration with [Rollin Leonard](http://rollinleonard.com).
Have a look at a [sample](http://thecreatorsproject.vice.com/blog/we-crash-kissed-julianna-huxtable-at-the-nowseethis-art-party) of the work
as shown at the Carnegie Museum of Art.

# The crash kiss process
Given an image of two people facing each other:

1. Find the foreground of the image (i.e. pixels that are not like the background)
2. For each row of the foreground, move the left and right hand side towards each other until they touch
3. Continue "crushing" background space out of the row until some maximum depth has been achieved

# Goals, challenges
1. Easy installation on Windows. If Rollin can't install it on his own, it's not easy enough.
2. No strange dependencies. Only common libraries like `numpy` should be used. In general, if there isn't a Windows binary on [Christoph Gohlke's page](http://www.lfd.uci.edu/~gohlke/pythonlibs/), it's not appropriate for this project.
3. Easily tuned from the command line. Parameters like the `threshold` (difference between the foreground and background), `max depth` (number of pixels by which faces are smashed towards each other), and so on must be variable.
4. Fast enough. Must be fast enough to generate images in the field in a photo booth configuration. Must also be fast enough to generate 1000-image .gif files (where each frame is a 4K image!) in a reasonable amount of time.
5. Must be accurate and it absolutely must produce images that are in the spirit of the original art.

# Getting started
## On Windows
1. Install [Anaconda](https://store.continuum.io/cshop/anaconda/)
2. Download the appropriate wheel file of [imread](https://github.com/luispedro/imread) from [Christoph's page](http://www.lfd.uci.edu/~gohlke/pythonlibs/#imread)
3. Install the .whl file with `pip install <path to file>`
4. Clone `https://github.com/TadLeonard/crash_kiss.git`
5. Run `python kiss.py [options]`

## On Linux
1. Run `python3 -m venv env; source env/bin/activate`
2. Clone `https://github.com/TadLeonard/crash_kiss.git`
3. In the project root, install requirements with `pip install -r requirements.txt`
3. Run `python kiss.py [options]`

