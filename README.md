# 2024年国科大智能计算系统实验代码

目前已更新至第四章实验，如有问题欢迎大家提出issue和pr

如果有帮到大家的话希望大家能留个star~![](README/044D36C4.png)

|           | score |
| :-------: | :---: |
|  exp_2_1  |  100  |
|  exp_2_2  |  100  |
|  exp_3_1  |  100  |
|  exp_3_2  |  100  |
|  exp_3_3  |  100  |
|  exp_4_1  |  100  |
|  exp_4_2  |  100  |
|  exp_4_3  |  100  |
|  exp_4_4  |  100  |
| 待更新... |       |

## 项目问题

### exp_4_4

### 问题

在本地可以运行实验4.4，但是传到评测平台以后，编译完动态链接库要运行`test_hsigmoid.py`时就报错了，提示找不到`hsigmoid_extension`模块

### 分析

把`hsigmoid.cpp`编译成`.so`动态链接库时，生成的动态链接库名不是`hsigmoid_extension.so`，而是`hsigmoid_extension.xxxx.so`。中间的`xxxx`包含了cpython版本和系统信息等。一般python会自动忽略中间的信息，因此在本机或者算力平台上代码都可以正确运行，不知道为什么评测平台上不会忽略中间的信息，导致模块无法被正确导入

### 解决方法

投机取巧版的解决办法就是在本机上编译出`hsigmoid_extension.xxxx.so`，然后重命名为`hsigmoid_extension.so`。接着将`setup.py`中的代码进行注释，仅保留最后一行的`print("generate .so PASS!\n")`。然后在上传评测平台时把`hsigmoid_extension.so`也一起上传

正常的解决方法就是修改`setup.py`中的代码,让其编译`hsigmoid.cpp`时生成的动态链接库名字为`hsigmoid_extension.so`（暂未实现该方法，有实现该方法的同学可以向仓库提交pr）
