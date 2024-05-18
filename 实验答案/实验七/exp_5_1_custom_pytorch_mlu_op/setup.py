import os
import sys
from setuptools import setup, find_packages

from torch.utils import cpp_extension
from torch_mlu.utils.cpp_extension import MLUExtension, BuildExtension
import glob
import shutil
from setuptools.dist import Distribution

mlu_custom_src = "mlu_custom_ext"
cpath = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    os.path.join(mlu_custom_src, "mlu")
)


def source(src):
    cpp_src = glob.glob("{}/*.cpp".format(src))
    mlu_src = glob.glob("{}/*.mlu".format(src))
    cpp_src.extend(mlu_src)
    return cpp_src


def main():
    mlu_extension = MLUExtension(
        name="libmlu_custom_ext",
        sources=source(os.path.join(cpath, 'src')),
        include_dirs=[os.path.join(cpath, "include")],
        verbose=True,
        extra_cflags=['-w'],
        extra_link_args=['-w'],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++14",
            ],
            "cncc": ["-O3", "-I{}".format(os.path.join(cpath, "include"))]
        })
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    if dist.script_args == ["clean"]:
        if os.path.exists(os.path.abspath('build')):
            shutil.rmtree('build')
    setup(name="mlu_custom_ext",
          version="0.1",
          packages=find_packages(),
          ext_modules=[mlu_extension],
          cmdclass={
              "build_ext":
                  BuildExtension.with_options(no_python_abi_suffix=True)
          })


if __name__ == "__main__":
    main()
