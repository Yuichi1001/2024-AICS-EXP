# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import codecs
import os
import random
import time

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.utils.data import DataLoader

from AttModel import AttModel
from data_load import TestDataSet, load_de_vocab, load_en_vocab
from hyperparams import Hyperparams as hp


def eval(args):
    # Load data
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # TODO：调用函数加载德语词表。
    de2idx, idx2de = load_de_vocab()
    # TODO：调用函数加载英语词表。
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)

    # TODO：初始化 Transformer 翻译模型。
    model = AttModel(hp, enc_voc, dec_voc)
    print("AttModel PASS!")

    source_test = args.dataset_path + hp.source_test
    target_test = args.dataset_path + hp.target_test
    # TODO: 构建测试数据集对象。
    test_dataset = TestDataSet(source_test, target_test)
    # TODO: 使用 PyTorch 的 DataLoader 对测试集进行批量加载。
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    if args.device == "MLU":
        model.mlu()
    elif args.device == "GPU":
        model.cuda()

    # TODO: 从指定路径加载预训练模型参数。
    state = torch.load(args.pretrained, map_location="cpu")
    # TODO: 兼容课程服务器中的训练断点格式；真正的模型权重保存在 "model" 字段中。
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # TODO: 将加载的模型参数赋值到当前模型中，以完成模型权重的恢复。
    model.load_state_dict(state)

    print("Model Loaded.")
    # TODO: 设置模型为评估模式。
    model.eval()

    # TODO: 以 UTF-8 编码、追加模式打开指定日志文件 args.log_path，用于记录模型评估的输出结果。
    with codecs.open(args.log_path, "a", "utf-8") as fout:
        list_of_refs, hypotheses = [], []
        t1 = time.time()
        sample_count = 0
        with torch.no_grad():
            # TODO：遍历测试集，每次获取一个批次的输入数据、原始句子和目标句子。
            for i, (x, sources, targets) in enumerate(test_loader):
                if i == args.iterations:
                    break

                if args.device == "GPU":
                    x_ = x.long().cuda()
                    preds_t = torch.LongTensor(
                        np.zeros((x.size(0), hp.maxlen), np.int32)
                    ).cuda()
                    preds = Variable(preds_t).cuda()
                elif args.device == "MLU":
                    x_ = x.long().to("mlu")
                    preds_t = torch.LongTensor(
                        np.zeros((x.size(0), hp.maxlen), np.int32)
                    ).to("mlu")
                    preds = Variable(preds_t.to("mlu"))
                else:
                    x_ = x.long()
                    preds_t = torch.LongTensor(np.zeros((x.size(0), hp.maxlen), np.int32))
                    preds = Variable(preds_t)

                for j in range(hp.maxlen):
                    _, _preds, _ = model(x_, preds)
                    preds_t[:, j] = _preds.data[:, j]
                    preds = Variable(preds_t.long())
                    if args.device == "GPU":
                        preds = preds.cuda()
                    elif args.device == "MLU":
                        preds = preds.to("mlu")

                preds = preds.data.cpu().numpy()
                sample_count += x.size(0)

                for source, target, pred in zip(sources, targets, preds):
                    got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                    fout.write("- source: " + source + "\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()

                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

        # TODO：计算整个推理过程所消耗的总时间。
        temp_time = time.time() - t1
        print("time:", temp_time)
        # TODO：计算每秒处理的样本数（吞吐率）。
        print("qps:", sample_count / temp_time if temp_time > 0 else 0)
        # TODO：计算模型翻译结果与参考答案之间的 BLEU 评分，用于评价翻译质量。
        score = corpus_bleu(list_of_refs, hypotheses) if list_of_refs and hypotheses else 0.0
        fout.write("Bleu Score = " + str(100 * score))
        print("Bleu Score = {}".format(100 * score))

    if os.getenv("AVG_LOG"):
        with open(os.getenv("AVG_LOG"), "a") as train_avg:
            train_avg.write("Bleu Score: {}\n".format(100 * score))
    print("Eval PASS!")


if __name__ == "__main__":
    # TODO: 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="Transformer evaluation.")
    parser.add_argument(
        "--device",
        default="MLU",
        type=str,
        help="set the type of hardware used for evaluation.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument(
        "--pretrained",
        default="model_epoch_20.pth",
        type=str,
        help="training ckps path",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="evaluation batch size.",
    )
    parser.add_argument("--workers", default=4, type=int, help="number of workers.")
    parser.add_argument(
        "--log-path",
        default="output.txt",
        type=str,
        help="evaluation file path.",
    )
    parser.add_argument(
        "--dataset-path",
        default="corpora/",
        type=str,
        help="The path of dataset.",
    )
    parser.add_argument(
        "--iterations",
        default=-1,
        type=int,
        help="Number of evaluation iterations.",
    )
    parser.add_argument(
        "--bitwidth",
        default=8,
        type=int,
        help="Set the initial quantization width of network training.",
    )
    parser.add_argument(
        "--opt_level",
        type=str,
        default="O0",
        help="choose level of mixing precision",
    )
    # TODO: 解析命令行输入的参数。
    args = parser.parse_args()

    if args.device == "MLU":
        import torch_mlu  # noqa: F401

    # TODO: 调用 eval 函数开始模型评估流程。
    eval(args)

    if args.device == "MLU":
        print("Transformer MLU PASS!")
    else:
        print("Transformer CPU PASS!")
