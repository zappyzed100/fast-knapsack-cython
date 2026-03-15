from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

ext_modules = [
    Extension(
        "solver_cython.core",  # インポート時の名前
        sources=["src/solver_cython/core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/openmp"] if os.name == "nt" else ["-fopenmp"],
        extra_link_args=[] if os.name == "nt" else ["-fopenmp"],
    )
]

setup(
    name="fast-knapsack-cython-core",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    packages=["solver_cython"],
    package_dir={"": "src"},
)
