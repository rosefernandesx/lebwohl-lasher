from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import subprocess, os

try:
    prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
except Exception:
    prefix = "/opt/homebrew/opt/libomp"   # default on Apple Silicon
compile_args = ["-O3", "-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"]
link_args = [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"]


compile_args = ["-O3", "-Xpreprocessor", "-fopenmp"]
link_args = ["-lomp"]

extensions = [
    Extension(
        "CythonLL",
        ["CythonLL.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="CythonLL",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)