# 2024年国科大智能计算系统实验代码

## 项目介绍

本项目是国科大《智能计算系统》2024年新版实验的完整资料存档。2024年的智能计算系统实验内容对知识体系和实验题目进行了大范围的调整，调整内容包括但不限于以下几点：

- 不再使用TensorFlow，全面使用PyTorch
- 大量修改旧版的题目代码
- 新增大模型实验等
- 新增《智能计算系统开发与实践》相关文件

## 项目状态

- 本项目跟随课程进度实时更新，截止2025.06.03已更新完毕
- 本项目适配2025年春季学期，后续课程实验若发生更改，欢迎各位同学提交pr
- 如果有帮到大家的话希望大家能给仓库留个Star~

### 更新日志

- 2025/06/05：更新《智能计算系统开发与实践》相关课件和实验手册、题目及答案

- 2025/06/03：更新实验八-大模型相关实验手册、题目及答案

- 2024/05/31：更新第九章和期末复习PPT以及第一版教材

- 2024/05/25：补充实验五-三选一实验中的tacotron2实验

- 2024/05/24：补充实验五-三选一实验中的yolov5实验

- 2024/05/21：更新理论教学的最新课件

- 2024/05/20：由于实验六老师提供的代码具有众多bug，且无法比较四个result的值，只能比较其中一个，因此直接在答案中上传了修改后的tb_top(0-2).v的代码，可供参考。同时，实验六对应的实验题目中的data_gen.py及tb_top(0-2).v的代码已经替换为修改后的代码，需要重新做实验的同学可以直接使用

- 2024/05/20：更新实验六的parallel_pe部分代码

- 2024/05/18：更新实验七：智能编程语言算子实验的实验手册、题目及答案（由于希冀平台故障，因此未在平台进行评测，但是本地测试都通过了）

- 2024/05/17：更新实验六：modelsim仿真实验的实验手册、题目及答案（安装modelsim的最后一步千万别让他安装那个硬件锁还是啥的，直接让我电脑无限蓝屏了T.T，如果不小心和我一样就进入安全模式然后删除电脑上的hardlock.sys）

- 2024/04/20：由于实验题目配套数据集大小超过Github限制的100MB，因此实验题目改为百度云托管

- 2024/04/20：更新实验五三选一实验手册、题目及答案

### 实验得分

|                                                     | score |
| :-------------------------------------------------: | :---: |
|        exp_2_1（手写数字分类实验：满分100）         |  100  |
| exp_2_2（基于DLP平台实现手写数字分类实验：满分100） |  100  |
|   exp_3_1（python实现VGG19图像分类实验：满分100）   |  100  |
|   exp_3_2（基于DLP平台实现图像分类实验：满分100）   |  100  |
|     exp_3_3（非实时图像风格迁移实验：满分100）      |  100  |
|  exp_4_1（pytorch实现VGG19图像分类实验：满分100）   |  100  |
|      exp_4_2（实时风格迁移推断实验：满分100）       |  100  |
|      exp_4_3（实时风格迁移训练实验：满分100）       |  100  |
|    exp_4_4（自定义pytorch cpu算子实验：满分100）    |  100  |
|          exp_5_1(三选一yolov5实验：满分80)          |  80   |
|        exp_5_1(三选一tacotron2实验：满分90)         |  90   |
|          exp_5_3(三选一bert实验：满分100)           |  100  |
|           exp_6（modelsim实验：满分120）            |  120  |
|    exp_7_1（智能编程语言算子开发实验：满分100）     |  100  |
|    exp_7_2（智能编程语言性能优化实验：满分100）     |  100  |
|              exp_8(CodeLLAMA：满分100)              |  100  |
|              exp_8(LLAMA3.2: 满分100)              |  100  |
=======
|    以下三个实验为《智能计算系统开发与实践》课程实验     |    |
|    （模型推理: YOLOv5 目标检测：满分100）           |  100  |
|    （模型推理:BERT的SQuAD任务：满分100）            |  100  |
|     (模型推理: Swin-Transformer 图像分类：满分100)    |  100  |

## 致谢

感谢以下同学对仓库代码提出的issue和pr：
- [WwwwwyDev](https://github.com/WwwwwyDev)
- [yangyu-wang](https://github.com/yangyu-wang)
- [Herry](https://github.com/Herry0w0)
- [PandaChen-pro](https://github.com/PandaChen-pro)
- [dragonzhoulong](https://github.com/Dragonzhoulong)
- [JohnLocke](https://github.com/JohnLocke)

## Star History

<a href="https://star-history.com/#Yuichi1001/2024-AICS-EXP&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Yuichi1001/2024-AICS-EXP&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Yuichi1001/2024-AICS-EXP&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Yuichi1001/2024-AICS-EXP&type=Timeline" />
 </picture>
</a>
