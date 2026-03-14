from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "solver_cython.core",  # インポート時の名前
        sources=["src/solver_cython/core.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fast-knapsack-cython-core",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    packages=[],  # パッケージ自動検出を無効化
)
