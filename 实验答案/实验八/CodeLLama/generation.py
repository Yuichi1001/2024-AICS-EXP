# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
#from typing import List, Literal, Optional, Tuple, TypedDict
from typing_extensions import List, Literal, Optional, Tuple, TypedDict

import torch
import torch_mlu
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama_mlu.model import ModelArgs, Transformer
from llama_mlu.tokenizer import Tokenizer


#TODO: 如果 MLU 设备可用
if torch.mlu.is_available():
#TODO: 将设备设置为 MLU
    device = "mlu"
else:
#TODO: 将设备设置为 CPU
    device = "cpu"

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    destination: str  # required for model responses


class InfillingPrediction(TypedDict, total=False):
    generation: str
    full_text: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>", "<step>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            if device == "mlu":
                #TODO: 使用 MLU 设备初始化分布式进程组
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("gloo")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device == "mlu":
            #TODO： 如果设备为 MLU，则设置当前进程的 MLU 设备
            torch.mlu.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        #TODO：加载模型的检查点文件，并将模型加载到 CPU 上。
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        #TODO: 调用Tokenizer函数
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        # support for mac
        #print(device)
        if device == "mlu":
            torch.set_default_tensor_type(torch.HalfTensor)
            #if torch.mlu.is_bf16_supported():
            #    torch.set_default_tensor_type(torch.mlu.BFloat16Tensor)
            #else:
            #    torch.set_default_tensor_type(torch.mlu.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.HalfTensor)
        #TODO: 调用Transformer 模型
        model = Transformer(model_args)
        #TODO：加载模型的参数字典
        model.load_state_dict(checkpoint, strict=False)
        print("TRANSFORMER MODEL PASS!")
        #add start
        #print(device)
        if device == "cpu":
            model = model.float()

        #add end
        model.to(device)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        
        print("LLAMA BUILD PASS!")
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        stop_token: Optional[int] = None,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        if stop_token is None:
            stop_token = self.tokenizer.eos_id
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        #TODO:  获取提示文本序列中最短的长度
        min_prompt_len = min(len(t) for t in prompt_tokens)
        #TODO: 获取提示文本序列中最长的长度
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        #TODO: 计算生成的总长度，需考虑提示文本和最大生成长度
        total_len = min(params.max_seq_len, max_prompt_len + max_gen_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            #TODO：将提示文本编码添加到张量中
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if logprobs:
            #TODO：创建一个与tokens张量具有相同形状的全零张量
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        prev_pos = 0
        stop_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                #TODO: 对模型的输出进行 softmax 归一化，以得到每个可能的下一个token的概率分布，其中 temperature 用于控制模型输出的多样性
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                #TODO: 根据概率分布采样出下一个 token
                next_token = sample_top_p(probs, top_p)
            else:
                #TODO：直接选择logits最大的位置作为下一个token，不进行随机采样
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            stop_reached |= (~input_text_mask[:, cur_pos]) & (next_token == stop_token)
            prev_pos = cur_pos
            if all(stop_reached):
                break

        if logprobs:
            #TODO: 将张量转换为列表格式
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            #TODO: 截取生成的标记序列，直到达到最大生成长度
            toks = toks[start:len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to stop token if present
            if stop_token in toks:
                stop_idx = toks.index(stop_token)
                toks = toks[:stop_idx]
                probs = probs[:stop_idx] if logprobs else None
            #TODO: 将截取后的标记序列添加到输出列表中
            out_tokens.append(toks)
            #TODO: 将截取后的log概率列表添加到输出列表中
            if logprobs:
                out_logprobs.append(probs)
        print("LLAMA GENERATE PASS!")
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        #TODO: 调用 generate 方法生成文本
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            assert generation_logprobs is not None
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.token_piece(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        print("LLAMA TEXTCOMPLETION PASS!")
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def text_infilling(
        self,
        prefixes: List[str],
        suffixes: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        suffix_first: bool = False,
    ) -> List[InfillingPrediction]:
        assert self.tokenizer.eot_id is not None
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [
            #TODO: 调用函数对每个前缀和后缀进行处理，生成填充问题的编码
            infilling_prompt_tokens(self.tokenizer, prefix, suffix, suffix_first)
            for prefix, suffix in zip(prefixes, suffixes)
        ]
        #TODO：调用 generate 方法生成文本
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        generations = [self.tokenizer.decode_infilling(t) for t in generation_tokens]
        print("LLAMA TEXTINFILLING PASS!")
        if logprobs:
            assert generation_logprobs is not None
            return [
                {
                    "generation": generation,
                    "logprobs": logprobs_i,
                    "tokens": [self.tokenizer.token_piece(x) for x in t],
                    "full_text": prefix + generation + suffix,
                }
                for prefix, suffix, generation, t, logprobs_i in zip(
                    prefixes,
                    suffixes,
                    generations,
                    generation_tokens,
                    generation_logprobs,
                )
            ]
        else:
            return [
                {
                    "generation": generation,
                    "full_text": prefix + generation + suffix,
                }
                for prefix, suffix, generation in zip(prefixes, suffixes, generations)
            ]
        

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if self.tokenizer.step_id is not None:
            ## 如果模型支持 step_id，则使用另一种chat_completion的方法
            return self._chat_completion_turns(dialogs, temperature, top_p, max_gen_len, logprobs)
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [  # type: ignore
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {dialog[-1]['content'].strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        #TODO：调用 generate 方法生成文本
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        print("LLAMA CHATCOMPLETION PASS!") 
        if logprobs:
            assert generation_logprobs is not None
            return [
                {
                    "generation": {  # type: ignore
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.token_piece(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {  # type: ignore
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]

    def _chat_completion_turns(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if self.tokenizer.step_id is None:
            raise RuntimeError("Model not suitable for chat_completion_step()")
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )

            # Insert system message if not provided
            if dialog[0]["role"] != "system":
                dialog = [{"role": "system", "content": ""}] + dialog  # type: ignore
            #TODO:调用函数将对话格式化为模型可处理的对话提示编码
            dialog_tokens = dialog_prompt_tokens(self.tokenizer, dialog)
            prompt_tokens.append(dialog_tokens)
        #TODO：调用 generate 方法生成文本
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            assert generation_logprobs is not None
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "destination": "user",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.token_piece(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "destination": "user",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]



def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def infilling_prompt_tokens(
    tokenizer: Tokenizer,
    pre: str,
    suf: str,
    suffix_first: bool = False,
) -> List[int]:
    """
    Format and encode an infilling problem.
    If `suffix_first` is set, format in suffix-prefix-middle format.
    """
    assert tokenizer.prefix_id is not None
    assert tokenizer.middle_id is not None
    assert tokenizer.suffix_id is not None
    if suffix_first:
        # format as "<PRE> <SUF>{suf} <MID> {pre}"
        return (
            [tokenizer.bos_id, tokenizer.prefix_id, tokenizer.suffix_id]
            + tokenizer.encode_infilling(suf)
            + [tokenizer.middle_id]
            + tokenizer.encode(pre, bos=False, eos=False)
        )
    else:
        # format as "<PRE> {pre} <SUF>{suf} <MID>"
        return (
            [tokenizer.bos_id, tokenizer.prefix_id]
            + tokenizer.encode(pre, bos=False, eos=False)
            + [tokenizer.suffix_id]
            + tokenizer.encode_infilling(suf)
            + [tokenizer.middle_id]
        )


def dialog_prompt_tokens(tokenizer: Tokenizer, dialog: Dialog) -> List[int]:
    """
    Prompt formatting for multi-turn dialogs.
    The dialog is expected to start with a system message and then alternate
    between user and assistant messages.
    """
    assert tokenizer.step_id is not None
    assert all([msg["role"] == "user" for msg in dialog[1::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[2::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    # Format context
    dialog_tokens: List[int] = [tokenizer.bos_id]
    headers: List[str] = []
    for message in dialog:
        headers.clear()
        headers.append(f"Source: {message['role'].strip()}")
        if message.get("destination") is not None:
            headers.append(f"Destination: {message['destination'].strip()}")
        header = " " + "\n".join(headers)
        dialog_tokens += tokenizer.encode(header, bos=False, eos=False)

        if message["content"]:
            body = "\n\n " + message["content"].strip()
            dialog_tokens += tokenizer.encode(body, bos=False, eos=False)

        dialog_tokens += [tokenizer.step_id]

    # Start of reply
    headers.clear()
    headers.append("Source: assistant")
    headers.append("Destination: user")
    header = " " + "\n".join(headers)
    dialog_tokens += tokenizer.encode(header, bos=False, eos=False)
    dialog_tokens += tokenizer.encode("\n\n ", bos=False, eos=False)

    return dialog_tokens
