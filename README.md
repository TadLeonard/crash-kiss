Crash Kiss: an art project automated
====================================
Crash kiss is an image processing art project. The goal of the project is to replicate artist Rollin Leonard's 2013 work entitled Crash Kiss. Seeing a [sample](http://rollinleonard.com/projects/2013/crashKiss-guthrieEllis/) of the original work is the quickest way to understand what it's all about.

Rollin's original work was done very carefully (and slowly) in Photoshop. Automating his process will allow him to generate more content and make changes to the process more easily, and that's what this project is all about.

The process
-----------
Given an image of two people facing each other:

1. Find the foreground of the image (i.e. pixels that are not like the background)
2. For each row of the foreground, move the left and right hand side towards each other until they touch
3. Continue "crushing" background space out of the row until some maximum depth has been achieved
4. Save the processed image to the disk

Goals, challenges
-----------------
The project is finished and each of these goals have been met.

1. Easy installation on Ubuntu and Windows.
2. No strange dependencies. Only common libraries like `numpy` should be used. In general, if there isn't a Windows binary on [Christoph Gohlke's page](http://www.lfd.uci.edu/~gohlke/pythonlibs/), it's not appropriate for this project.
3. Easily tuned from the command line. Parameters like the threshold (difference between the foreground and background), max depth (number of pixels by which faces are smashed into each other), and so on must be variable.
4. Fast enough. Must be fast enough to generate images in the field in a photo booth configuration. Must also be fast enough to generate 1000-image .gif files (where each frame is a 4K image!) in a reasonable amount of time (i.e. minutes, not hours).
5. Accurate and behaves in the spirit of the original art.

Getting started
---------------

On Windows
##########
1. Install [Anaconda](https://store.continuum.io/cshop/anaconda/)
2. Download the appropriate wheel file of [imread](https://github.com/luispedro/imread) from [Christoph's page](http://www.lfd.uci.edu/~gohlke/pythonlibs/#imread)
3. Install the .whl file with `pip install <path to file>`
4. Clone https://github.com/TadLeonard/crash_kiss.git
5. Run `python ckiss.py [options]`

On Linux (Debian, Ubuntu)
#########################
1. Run `pip install numpy imread`
2. Clone https://github.com/TadLeonard/crash_kiss.git
3. Run `python ckiss.py [options]`

