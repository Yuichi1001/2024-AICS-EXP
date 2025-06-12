import sys
import cv2
import torch
import torch_mlu
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        #TODO: 将图片从RGB格式转换为BGR格式（OpenCV默认格式）
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        #TODO: 将编码后的图片数组转换为PIL格式，并转换为RGB格式
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    #TODO: 加载模型参数
    pl_sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(pl_sd["state_dict"], strict=False) # TODO: is this correct?
    #TODO: 根据系统环境来选择运行设备,如果支持 MLU 设备，则选择 MLU 设备，否则选择 CPU。
    device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu")
    #TODO: 将模型加载到指定设备
    model = model.to(device)
    #TODO: 使用DDIMSampler对模型进行采样
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    #TODO: 将输入图片转换为RGB格式的NumPy数组
    image =np.array(image.convert("RGB"))
    #TODO: 添加一个维度，并将维度顺序从HWC转换为BCHW
    image = image[None].transpose(0, 3, 1, 2)
    #TODO: 将NumPy数组转换为PyTorch张量，进行归一化处理，使像素值范围在[-1, 1]之间
    image = torch.from_numpy(image).float() / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    #TODO: 将掩码数组转换为浮点数并归一化处理，使像素值范围在[0, 1]之间
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    #TODO: 将掩码数组中小于0.5的像素值设为0，表示不需要修复的部分
    mask[mask < 0.5] = 0.0
    #TODO: 将掩码数组中大于等于0.5的像素值设为1，表示需要修复的部分
    mask[mask >= 0.5] = 1.0
    #TODO: 将NumPy数组转换为PyTorch张量
    mask = torch.from_numpy(mask)
    #TODO: 使用掩码对输入图片进行掩码处理，将不需要修复的部分设置为0
    masked_image = image * (1 - mask)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "mlu") if torch.mlu.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    #TODO：将 NumPy 数组 start_code 转换为 PyTorch 张量，并将其发送到指定的设备上，并指定数据类型为 torch.float32。
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)


    #TODO:关闭梯度计算，启用MLU自动混合精度
    with torch.no_grad(), torch.mlu.amp.autocast():
        #TODO:调用函数构造批次数据
        batch = make_batch_sd(image, mask, prompt, device, num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            #TODO：将处理后的数据添加到列表中    
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)
        #TODO:将生成的图片结果从 PyTorch 张量转换为 NumPy 数组，并将维度顺序调整为常用的图像格式，最后将像素值恢复到原始范围
        result = (result.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, prompt, ddim_steps, num_samples, scale, seed):
    #TODO: 从输入图像中获取原始图像，并转换为RGB模式
    init_image = input_image["image"].convert("RGB")
    #TODO: 从输入图像中获取掩码图像，并转换为RGB模式
    init_mask = input_image["mask"].convert("RGB")
    #TODO: 调用函数对原始图像和掩码图像进行填充
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)
    #TODO： 调用图像修复函数进行修复
    result = inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples, width, height)

    return result

#TODO：调用函数初始化模型
sampler = initialize_model(sys.argv[1], sys.argv[2])

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=4, value=4, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")
    #TODO: 定义点击按钮时执行的函数，即调用 predict 函数进行图像修复
    run_button.click(fn=predict, inputs=[input_image, prompt, ddim_steps, num_samples, scale, seed], outputs=[gallery])


# block.launch(share=True)
print("inpainting PASS")
