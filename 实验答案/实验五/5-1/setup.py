## file: setup.py
from pathlib import Path
from shutil import copy2

from setuptools import setup
from torch.utils import cpp_extension


def copy_extension_to_parent_dir():
    """把编译生成的 .so 复制到 stu_upload，便于评测脚本直接 import。"""
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    # TODO：查找 build_ext --inplace 在 op_mysigmoid 目录下生成的动态链接库。
    built_extensions = sorted(current_dir.glob("mysigmoid_extension*.so"))
    if not built_extensions:
        raise FileNotFoundError("未找到 mysigmoid_extension*.so，请检查 C++ 扩展是否编译成功。")

    # TODO：将 .so 复制到上一级 stu_upload 目录，使 test_mysigmoid.py 可直接导入。
    for extension in built_extensions:
        copy2(extension, parent_dir / extension.name)


setup(
    # TODO: 给出编译后的链接库名称
    name="mysigmoid_extension",
    ext_modules=[
        cpp_extension.CppExtension(
            # TODO：以正确的格式给出编译文件即编译函数
            "mysigmoid_extension",
            ["mysigmoid.cpp"],
        )
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
    },
)

copy_extension_to_parent_dir()
print("generate .so PASS!\n")
