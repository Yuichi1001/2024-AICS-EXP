# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import embedding, feedforward, label_smoothing, multihead_attention, positional_encoding


class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        """Transformer seq2seq model."""
        super(AttModel, self).__init__()
        self.hp = hp_
        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        # Encoder
        # TODO：调用基本单元模块完成编码器的词嵌入，将词索引映射为 hidden_units 维稠密向量，并进行缩放。
        self.enc_emb = embedding(enc_voc, self.hp.hidden_units, zeros_pad=True, scale=True)
        print("Embedding PASS!")

        # TODO：如果超参数中设置使用正弦位置编码，调用位置编码模块，维度为 hidden_units，不使用零填充，也不进行缩放。
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(
                self.hp.hidden_units,
                zeros_pad=False,
                scale=False,
            )
        else:
            # TODO：否则使用可学习的嵌入方式生成位置编码，词表大小为 maxlen，嵌入维度为 hidden_units，不使用零填充，也不进行缩放。
            self.enc_positional_encoding = embedding(
                self.hp.maxlen,
                self.hp.hidden_units,
                zeros_pad=False,
                scale=False,
        )
        print("PositionEncoding PASS!")

        # TODO：定义 dropout 层。
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            # TODO：调用多头注意力机制模块，隐藏维度为 hidden_units，包含 num_heads 个注意力头，使用指定 dropout 率，且不使用因果掩码。
            self.__setattr__(
                "enc_self_attention_%d" % i,
                multihead_attention(
                    self.hp,
                    self.hp.hidden_units,
                    num_heads=self.hp.num_heads,
                    dropout_rate=self.hp.dropout_rate,
                    causality=False,
                ),
            )
            # TODO：调用前馈神经网络模块进行构建，隐藏层维度为输入 hidden_units 的 4 倍，输出层维度恢复到输入。
            self.__setattr__(
                "enc_feed_forward_%d" % i,
                feedforward(
                    self.hp.hidden_units,
                    num_units=[4 * self.hp.hidden_units, self.hp.hidden_units],
                ),
            )

        print("LayerNormalization PASS!")
        print("MutiheadAtt PASS!")
        print("FeedForward PASS!")

        # Decoder
        # TODO: 调用基本单元模块完成解码器的词嵌入，将词索引映射为 hidden_units 维稠密向量，并进行缩放。
        self.dec_emb = embedding(dec_voc, self.hp.hidden_units, zeros_pad=True, scale=True)
        # TODO：如果超参数中设置使用正弦位置编码，调用位置编码模块，维度为 hidden_units，不使用零填充，也不进行缩放。
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(
                self.hp.hidden_units,
                zeros_pad=False,
                scale=False,
            )
        else:
            # TODO：否则使用可学习的嵌入方式生成位置编码，词表大小为 maxlen，嵌入维度为 hidden_units，不使用零填充，也不进行缩放。
            self.dec_positional_encoding = embedding(
                self.hp.maxlen,
                self.hp.hidden_units,
                zeros_pad=False,
                scale=False,
            )
        # TODO：定义 dropout 层。
        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            # TODO：调用多头注意力机制模块，隐藏维度为 hidden_units，包含 num_heads 个注意力头，使用指定 dropout 率，使用掩码。
            self.__setattr__(
                "dec_self_attention_%d" % i,
                multihead_attention(
                    self.hp,
                    self.hp.hidden_units,
                    num_heads=self.hp.num_heads,
                    dropout_rate=self.hp.dropout_rate,
                    causality=True,
                ),
            )
            # TODO：调用多头注意力机制模块，隐藏维度为 hidden_units，包含 num_heads 个注意力头，使用指定 dropout 率，不使用掩码。
            self.__setattr__(
                "dec_vanilla_attention_%d" % i,
                multihead_attention(
                    self.hp,
                    self.hp.hidden_units,
                    num_heads=self.hp.num_heads,
                    dropout_rate=self.hp.dropout_rate,
                    causality=False,
                ),
            )
            # TODO：调用前馈神经网络模块进行构建，隐藏层维度为输入 hidden_units 的 4 倍，输出层维度恢复到输入。
            self.__setattr__(
                "dec_feed_forward_%d" % i,
                feedforward(
                    self.hp.hidden_units,
                    num_units=[4 * self.hp.hidden_units, self.hp.hidden_units],
                ),
            )

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        # TODO：调用标签平滑模块。
        self.label_smoothing = label_smoothing()
        print("LabelSmoothing PASS!")

    def forward(self, x, y):
        input_tensor = torch.ones(y[:, :1].size(), device=y.device)
        if x.device.type == "mlu":
            # TODO: 将 input_tensor 放在寒武纪卡上。
            input_tensor = input_tensor.to("mlu")
        self.decoder_inputs = torch.cat(
            [Variable(input_tensor * 2).long(), y[:, :-1]],
            dim=-1,
        )

        # Encoder
        # TODO：通过编码器端的词嵌入。
        self.enc = self.enc_emb(x)
        # TODO：使用正弦位置编码。
        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x, self.enc)
        else:
            enc_positional = torch.unsqueeze(torch.arange(0, x.size(1), device=x.device), 0)
            enc_positional = enc_positional.repeat(x.size(0), 1).long()
            self.enc += self.enc_positional_encoding(Variable(enc_positional))

        # TODO：Dropout 正则化。
        self.enc = self.enc_dropout(self.enc)

        for i in range(self.hp.num_blocks):
            # 实现编码器的 self-attention 机制。
            self.enc = self.__getattr__("enc_self_attention_%d" % i)(self.enc, self.enc, self.enc)
            # 实现编码器的 Feed Forward 机制。
            self.enc = self.__getattr__("enc_feed_forward_%d" % i)(self.enc)

        # Decoder
        # TODO：解码器端的词嵌入。
        self.dec = self.dec_emb(self.decoder_inputs)
        # TODO：使用正弦位置编码。
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs, self.dec)
        else:
            dec_positional = torch.unsqueeze(
                torch.arange(0, self.decoder_inputs.size(1), device=x.device),
                0,
            )
            dec_positional = dec_positional.repeat(self.decoder_inputs.size(0), 1).long()
            # TODO：加上位置编码。
            self.dec += self.dec_positional_encoding(Variable(dec_positional))

        # TODO：进行 Dropout。
        self.dec = self.dec_dropout(self.dec)

        for i in range(self.hp.num_blocks):
            # 实现解码器内部 self-attention 机制。
            self.dec = self.__getattr__("dec_self_attention_%d" % i)(
                self.dec,
                self.dec,
                self.dec,
            )
            # 实现解码器内部 vanilla attention 机制。
            self.dec = self.__getattr__("dec_vanilla_attention_%d" % i)(
                self.dec,
                self.enc,
                self.enc,
            )
            # 实现解码器的 feed forward。
            self.dec = self.__getattr__("dec_feed_forward_%d" % i)(self.dec)

        # TODO：线性映射到词表维度。
        self.logits = self.logits_layer(self.dec)
        # TODO：计算概率分布，并展平成二维 (batch_size * seq_len, vocab_size)。
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc)
        # TODO：获取最大概率对应的预测词索引。
        _, self.preds = torch.max(self.logits, dim=-1)

        self.istarget = (1.0 - y.eq(0.0).float()).view(-1)
        target_count = torch.sum(self.istarget).clamp_min(1.0)
        self.acc = torch.sum(self.preds.eq(y).float().view(-1) * self.istarget) / target_count

        self.y_onehot = torch.zeros(
            self.logits.size(0) * self.logits.size(1),
            self.dec_voc,
            device=x.device,
        )
        self.y_onehot = Variable(self.y_onehot.scatter_(1, y.view(-1, 1).data, 1))
        # TODO：调用标签平滑模块，得到平滑后的标签分布。
        self.y_smoothed = self.label_smoothing(self.y_onehot)

        self.loss = -torch.sum(self.y_smoothed * torch.log(self.probs.clamp_min(1e-9)), dim=-1)
        self.mean_loss = torch.sum(self.loss * self.istarget) / target_count

        return self.mean_loss, self.preds, self.acc
