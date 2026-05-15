# -*- coding: utf-8 -*-
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        """Embeds a given variable.

        Args:
            vocab_size: Vocabulary size.
            num_units: Embedding hidden units.
            zeros_pad: If True, the first row is fixed to zeros.
            scale: If True, outputs are multiplied by sqrt(num_units).
        """
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        # TODO：使用 Xavier 正态初始化方法对嵌入矩阵进行初始化。
        nn.init.xavier_normal_(self.lookup_table.data)

        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        padding_idx = 0 if self.zeros_pad else -1
        # TODO：调用内置函数将输入的词索引映射为对应的词向量表示，用于实现嵌入层功能。
        outputs = F.embedding(
            inputs,
            self.lookup_table,
            padding_idx,
            None,
            2,
            False,
            False,
        )

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        """Applies layer normalization over the last dimension."""
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        # TODO：定义可训练的缩放参数 gamma，初始值全为 1，形状与 features 维度一致。
        self.gamma = Parameter(torch.ones(features))
        # TODO：定义可训练的偏移参数 beta，初始值全为 0，形状与 features 维度一致。
        self.beta = Parameter(torch.zeros(features))

    def forward(self, x):
        # TODO：对输入计算最后一个维度的均值。
        mean = torch.mean(x, dim=-1, keepdim=True)
        # TODO：对输入计算最后一个维度的标准差。
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        # TODO：对输入进行层归一化。
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class positional_encoding(nn.Module):
    def __init__(self, num_units, zeros_pad=True, scale=True):
        """Sinusoidal positional encoding."""
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs, y=None):
        # inputs: 2D tensor with shape (N, T).
        N, T = inputs.size()[0:2]
        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1)
        position_ind = position_ind.to(device=inputs.device, dtype=torch.long)

        # TODO: 根据论文公式，计算位置编码矩阵，形状为 (T, num_units)。
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2 * (i // 2) / self.num_units)
                    for i in range(self.num_units)
                ]
                for pos in range(T)
            ],
            dtype=np.float32,
        )
        # TODO：对偶数列（从 0 开始）使用 sin。
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # TODO：对奇数列使用 cos。
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        lookup_table = torch.tensor(position_enc, dtype=torch.float32, device=inputs.device)
        if y is not None and inputs.device.type == "mlu":
            lookup_table = lookup_table.to(y.dtype)

        if self.zeros_pad:
            zero_pad = torch.zeros(1, self.num_units, dtype=lookup_table.dtype, device=inputs.device)
            lookup_table = torch.cat((zero_pad, lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        # TODO: 根据位置索引 position_ind，从位置编码查找表 lookup_table 中取出对应的位置编码向量，生成最终的编码输出。
        outputs = F.embedding(
            position_ind,
            lookup_table,
            padding_idx,
            None,
            2,
            False,
            False,
        )

        if self.scale:
            # TODO：将输出进行缩放。
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class multihead_attention(nn.Module):
    def __init__(self, hp_, num_units, num_heads=8, dropout_rate=0, causality=False):
        """Applies multihead attention."""
        super(multihead_attention, self).__init__()
        if num_units % num_heads != 0:
            raise ValueError("num_units must be divisible by num_heads")

        self.hp = hp_
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        # TODO：构建 Q、K、V 的线性映射层，包含全连接和 ReLU 激活，用于将输入特征投影到注意力空间。
        self.Q_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        # TODO：输出 dropout 层。
        self.output_dropout = nn.Dropout(self.dropout_rate)
        # TODO：调用自定义函数实现层归一化，标准化输出。
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        # keys, values: [N, T_k, C], queries: [N, T_q, C].
        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)

        # TODO：将 Q、K、V 按最后一维均分为 num_heads 份，并在 batch 维拼接。
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)

        # TODO：计算 Q 与 K 的转置在每个注意力头内的批量矩阵乘法，得到注意力得分。
        outputs = torch.bmm(Q_, K_.transpose(1, 2))
        # TODO：按键的最后一维度平方根缩放注意力得分。
        outputs = outputs / math.sqrt(K_.size(-1))

        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size(1), 1)
        padding = torch.ones_like(outputs) * (-(2**32) + 1)
        condition = key_masks.eq(0.0)
        outputs = padding * condition + outputs * (~condition)

        if self.causality:
            diag_vals = torch.ones(
                outputs.size(1),
                outputs.size(2),
                dtype=queries.dtype,
                device=queries.device,
            )
            # TODO：生成一个下三角矩阵，主对角线及其以下由 diag_vals 填充，其余位置为零。
            tril = torch.tril(diag_vals)
            masks = torch.unsqueeze(tril, 0).repeat(outputs.size(0), 1, 1)
            padding = torch.ones_like(masks) * (-(2**32) + 1)
            condition = masks.eq(0.0)
            outputs = padding * condition + outputs * (~condition)

        # TODO：对最后一个维度做 softmax，计算注意力权重。
        outputs = F.softmax(outputs, dim=-1)

        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size(1))
        # TODO：屏蔽无效的 query 位置，防止其影响注意力结果。
        outputs = outputs * query_masks

        # TODO：对注意力权重做 dropout。
        outputs = self.output_dropout(outputs)
        # TODO：执行批量矩阵乘法，将注意力权重与值向量相乘。
        outputs = torch.bmm(outputs, V_)
        # TODO：将多头的输出按特征维度拼接，还原为 (N, T_q, C)。
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        # TODO：加入残差连接。
        outputs += queries
        # TODO：对输出进行层归一化。
        outputs = self.normalization(outputs)

        return outputs


class feedforward(nn.Module):
    def __init__(self, in_channels, num_units=None):
        """Point-wise feed-forward network."""
        super(feedforward, self).__init__()
        if num_units is None:
            num_units = [2048, 512]
        self.in_channels = in_channels
        self.num_units = num_units

        self.conv = False
        if self.conv:
            params = {
                "in_channels": self.in_channels,
                "out_channels": self.num_units[0],
                "kernel_size": 1,
                "stride": 1,
                "bias": True,
            }
            # TODO：采用卷积构建第一层线性映射，包含一维卷积和 ReLU 激活。
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {
                "in_channels": self.num_units[0],
                "out_channels": self.num_units[1],
                "kernel_size": 1,
                "stride": 1,
                "bias": True,
            }
            # TODO：采用卷积构建第二层线性映射。
            self.conv2 = nn.Conv1d(**params)
        else:
            # TODO：采用全连接方式实现第一层线性映射。
            self.conv1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
            # TODO：采用全连接方式实现第二层线性映射。
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
        # TODO：调用自定义函数实现层归一化，标准化输出。
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        residual = inputs
        if self.conv:
            # TODO：调整输入形状 (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)。
            inputs = inputs.transpose(1, 2)

        # TODO：构建第一层线性映射。
        outputs = self.conv1(inputs)
        # TODO：构建第二层线性映射。
        outputs = self.conv2(outputs)

        if self.conv:
            # TODO：如果是卷积实现，先进行形状转换再进行层归一化。
            outputs = outputs.transpose(1, 2)

        # TODO：残差连接。
        outputs += residual
        # TODO：如果是线性映射，直接归一化。
        outputs = self.normalization(outputs)
        return outputs


class label_smoothing(nn.Module):
    def __init__(self, epsilon=0.1):
        """Applies label smoothing."""
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        # TODO：获取类别数量。
        K = inputs.size(-1)
        # TODO：应用公式进行标签平滑。
        return (1 - self.epsilon) * inputs + self.epsilon / K
