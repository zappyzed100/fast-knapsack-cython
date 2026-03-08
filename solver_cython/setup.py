import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# setup.pyがあるディレクトリの絶対パスを取得
base_dir = os.path.dirname(__file__)
pyx_path = os.path.join(base_dir, "core.pyx")

setup(
    ext_modules=cythonize(Extension("solver_cython.core", [pyx_path]), annotate=True),
    include_dirs=[numpy.get_include()],
)
