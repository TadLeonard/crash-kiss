import numpy
from setuptools import setup
try:
    from Cython.Build import cythonize
except ImportError:
    import sys
    sys.exit("Cython required; pip install Cython")



setup(
    name="crash_kiss",
    scripts=["kiss.py"], 
    packages=["crash_kiss"],
    ext_modules=cythonize("crash_kiss/smoosh.pyx"),
    install_requires=["six", "numpy", "imread"],
    include_dirs=[numpy.get_include()],
)

