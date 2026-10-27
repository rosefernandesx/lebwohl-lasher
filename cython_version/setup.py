from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "CythonLL",
        ["CythonLL.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="CythonLL",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)

