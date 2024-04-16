from setuptools import setup
from torch.utils import cpp_extension

setup(
    #TODO: 给出编译后的链接库名称
    ______________________________________
    ext_modules=[
        cpp_extension.CppExtension(
    #TODO：以正确的格式给出编译文件即编译函数
    ______________________________________
        )
    ],
    # 执行编译命令设置
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension
    }
)
print("generate .so PASS!\n")