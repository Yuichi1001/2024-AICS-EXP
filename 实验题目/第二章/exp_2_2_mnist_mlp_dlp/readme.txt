补全 stu_upload 中的 mnist_mlp_demo.py 文件, 并复制实验2-1中实现的layer_1.py、mnist_mlp_cpu.py 以及训练得到的参数复制到 stu_upload 目录下，执行 main_exp_2_2.py 运行实验。

注意：
上传的实验2-1中训练生成的模型参数，如 mlp-32-16-10epoch.npy，需要修改名称为 weight.npy，否则无法识别。
上传的 mnist mlp 网络的 cpu 实现，即实验2-1中完成的 mnist_mlp_cpu.py 文件，需要做出以下修改：

修改 build_mnist_mlp() 函数中的内容：
1.  修改 batch_size.
    将 mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e) 
    修改为 mlp = MNIST_MLP(batch_size=10000, hidden1=h1, hidden2=h2, max_epoch=e)

2.  注释掉训练的函数
    mlp.train()
    和
    mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    两句，并将
    mlp.load_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    取消注释，同时修改函数参数为 param_dir
    mlp.load_model(param_dir)
