from setuptools import setup, Extension
try:
    import numpy
    from Cython.Build import cythonize
except ImportError:
    import sys
    sys.exit("Cython and numpy required; pip install Cython numpy")


extra_compile_args = ["-fopenmp", "-O3", "-ffast-math"]


omp_ext = Extension("crash_kiss.omp_smoosh",
                    sources=[
                        "crash_kiss/_omp_smoosh.c",
                        "crash_kiss/omp_smoosh.pyx",],
                    extra_compile_args=extra_compile_args,
                    include_dirs=["crash_kiss/"],
                    extra_link_args=["-fopenmp"])


cython_ext = Extension("crash_kiss.smoosh",
                       sources=["crash_kiss/smoosh.pyx"],
                       extra_compile_args=extra_compile_args,
                       extra_link_args=["-fopenmp"])


extensions = omp_ext, cython_ext


setup(
    name="crash_kiss",
    scripts=["kiss.py"],
    packages=["crash_kiss"],
    ext_modules=cythonize(extensions),
    install_requires=["six", "numpy", "imageio"],
    include_dirs=[numpy.get_include()],
)

