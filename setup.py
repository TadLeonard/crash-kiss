from setuptools import setup
try:
    import numpy
    from Cython.Build import cythonize
except ImportError:
    import sys
    sys.exit("Cython and numpy required; pip install Cython numpy")



setup(
    name="crash_kiss",
    scripts=["kiss.py"], 
    packages=["crash_kiss"],
    ext_modules=cythonize("crash_kiss/smoosh.pyx"),
    install_requires=["six", "numpy", "imread"],
    include_dirs=[numpy.get_include()],
)

