from setuptools import setup, Extension
try:
    import numpy
    from Cython.Build import cythonize
except ImportError:
    import sys
    sys.exit("Cython and numpy required; pip install Cython numpy")

from Cython.Distutils import build_ext


extra_compile_args = [
#    "-fopenmp",
    "-O3",
    "-ffast-math"
]


# _omp_smoosh.c is where the main smooshing logic lives
omp_ext = Extension(
    "crash_kiss.omp_smoosh",
    sources=[
        "crash_kiss/_omp_smoosh.c",
        "crash_kiss/omp_smoosh.pyx"
    ],
    extra_compile_args=extra_compile_args,
    include_dirs=[
        "crash_kiss/",
        numpy.get_include()
    ],
    extra_link_args=[
    #    "-fopenmp"
    ]
)


# This is how we'd compile the Cython smoosh function
"""
cython_ext = Extension(
    "crash_kiss.smoosh",
    sources=["crash_kiss/smoosh.pyx"],
    extra_compile_args=extra_compile_args,
    include_dirs=[
        "crash_kiss/",
        numpy.get_include()
    ],
    extra_link_args=[
    #    "-fopenmp"
    ]
)
"""


extensions = omp_ext, #cython_ext


setup(
    name="crash_kiss",
    scripts=["kiss.py"],
    packages=["crash_kiss"],
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},  # unclear if necessary per Cy docs
    install_requires=["numpy", "imageio"],
    include_dirs=[
        numpy.get_include()
    ],
)

