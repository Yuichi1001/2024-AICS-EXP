# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import sys
import os
cur_path = os.getcwd()
models_path = cur_path + "/src/"
sys.path.append(models_path)
import models
import torch
import argparse
import numpy as np
import glob
from scipy.io.wavfile import write


from inference import checkpoint_from_distributed, unwrap_distributed, MeasureTime, prepare_input_sequence, load_and_setup_model

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

# cambricon-note: no apex in cambricon-pytorch
#from apex import amp

from waveglow.denoiser import Denoiser
# cambricon-note: use cambricon-pytorch
import torch_mlu
from plot_data import *

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true',
                        help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('-il', '--input-length', type=int, default=64,
                        help='Input length')
    parser.add_argument('-bs', '--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--device-param', type=str, default='mlu',
                        help='inference device, optional val is gpu/mlu/cpu.')
    return parser

"""
def load_and_setup_model(model_name, parser, checkpoint, amp_run, device_param, forward_is_infer=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, device_param, forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if device_param != "gpu":
            # cambricon-note:for mlu or cpu
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']

        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

        if device_param == "mlu":
            cur_device = torch.device('mlu')
            model = model.to(cur_device)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if amp_run:
        model, _ = amp.initialize(model, [], opt_level="O3")

    return model
"""


def print_stats(measurements_all):

    print(np.mean(measurements_all['latency'][1:]),
          np.mean(measurements_all['throughput'][1:]),
          np.mean(measurements_all['pre_processing'][1:]),
          np.mean(measurements_all['type_conversion'][1:])+
          np.mean(measurements_all['storage'][1:])+
          np.mean(measurements_all['data_transfer'][1:]),
          np.mean(measurements_all['num_mels_per_audio'][1:]))

    throughput = measurements_all['throughput']
    preprocessing = measurements_all['pre_processing']
    type_conversion = measurements_all['type_conversion']
    storage = measurements_all['storage']
    data_transfer = measurements_all['data_transfer']
    postprocessing = [sum(p) for p in zip(type_conversion,storage,data_transfer)]
    latency = measurements_all['latency']
    num_mels_per_audio = measurements_all['num_mels_per_audio']

    latency.sort()

    cf_50 = max(latency[:int(len(latency)*0.50)])
    cf_90 = max(latency[:int(len(latency)*0.90)])
    cf_95 = max(latency[:int(len(latency)*0.95)])
    cf_99 = max(latency[:int(len(latency)*0.99)])
    cf_100 = max(latency[:int(len(latency)*1.0)])

    print("Throughput average (samples/sec) = {:.4f}".format(np.mean(throughput)))
    print("Preprocessing average (seconds) = {:.4f}".format(np.mean(preprocessing)))
    print("Postprocessing average (seconds) = {:.4f}".format(np.mean(postprocessing)))
    print("Number of mels per audio average = {}".format(np.mean(num_mels_per_audio)))
    print("Latency average (seconds) = {:.4f}".format(np.mean(latency)))
    print("Latency std (seconds) = {:.4f}".format(np.std(latency)))
    print("Latency cl 50 (seconds) = {:.4f}".format(cf_50))
    print("Latency cl 90 (seconds) = {:.4f}".format(cf_90))
    print("Latency cl 95 (seconds) = {:.4f}".format(cf_95))
    print("Latency cl 99 (seconds) = {:.4f}".format(cf_99))
    print("Latency cl 100 (seconds) = {:.4f}".format(cf_100))


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    cur_path = os.getcwd()
    pretrained_model_dir = cur_path + "/model/pretrained/"
    # 设定预训练模型路径，推理设备，waveglow_channel, 输入文本的input_length
    args.tacotron2 = pretrained_model_dir + "nvidia_tacotron2pyt_fp32_20190427"
    args.waveglow = pretrained_model_dir + "nvidia_waveglowpyt_fp32_20190427"
    #TODO：设定模型推理使用的设备参数为寒武纪DLP
    args.device_param = "mlu"
    args.input_length=167
    #args.input_length=128
    args.batch_size=1
    args.num_iters=14
    # wn_channels 参数的解析在models.py完成
    args.wn_channels=512
    
    print("args:",args,unknown_args)

    # 初始化 DLLogger，设置日志输出的后端（backends）
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    measurements_all = {"pre_processing": [],
                        "tacotron2_latency": [],
                        "waveglow_latency": [],
                        "latency": [],
                        "type_conversion": [],
                        "data_transfer": [],
                        "storage": [],
                        "tacotron2_items_per_sec": [],
                        "waveglow_items_per_sec": [],
                        "num_mels_per_audio": [],
                        "throughput": []}

    print("args:", args, unknown_args)

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2, args.amp_run, args.device_param, forward_is_infer=True)
    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow, args.amp_run, args.device_param)

    if args.device_param == "cpu":
        denoiser = Denoiser(waveglow, args.device_param)
    elif args.device_param == "mlu":
        denoiser = Denoiser(waveglow, args.device_param).mlu()
    else:
        denoiser = Denoiser(waveglow, args.device_param).cuda()

    # cambricon-note: use native tacotron2 model to infer
    #jitted_tacotron2 = torch.jit.script(tacotron2)
    texts = ["The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."]
    print(len(texts))
    texts = [texts[0][:args.input_length]]
    texts = texts*args.batch_size

    warmup_iters = 3
    cambricon_cnt = 0

    for iter in range(args.num_iters):

        measurements = {}

        with MeasureTime(measurements, "pre_processing", args.device_param):
            sequences_padded, input_lengths = prepare_input_sequence(texts, args.device_param)
     #TODO: 禁用梯度计算，进行推理过程。
        with torch.no_grad():
            with MeasureTime(measurements, "latency", args.device_param):
                with MeasureTime(measurements, "tacotron2_latency", args.device_param):
                    #mel, mel_lengths, _ = tacotron2(sequences_padded, input_lengths)
                    mel, mel_lengths, alignment  = tacotron2(sequences_padded, input_lengths)

                with MeasureTime(measurements, "waveglow_latency", args.device_param):
                    audios = waveglow.infer(mel, sigma=args.sigma_infer)
                    audios = audios.float()
                    audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

        num_mels = mel.size(0)*mel.size(2)
        num_samples = audios.size(0)*audios.size(1)

        with MeasureTime(measurements, "type_conversion", args.device_param):
            audios = audios.float()

        with MeasureTime(measurements, "data_transfer", args.device_param):
            audios = audios.cpu()

        with MeasureTime(measurements, "storage", args.device_param):
            audios = audios.numpy()
            for i, audio in enumerate(audios):
                audio_path = "audio_"+str(i)+".wav"
                write(audio_path, args.sampling_rate,
                      audio[:mel_lengths[i]*args.stft_hop_length])

        #print("Save audio file Pass!")

        measurements['tacotron2_items_per_sec'] = num_mels/measurements['tacotron2_latency']
        measurements['waveglow_items_per_sec'] = num_samples/measurements['waveglow_latency']
        measurements['num_mels_per_audio'] = mel.size(2)
        measurements['throughput'] = num_samples/measurements['latency']

        cambricon_cnt += 1
        if iter >= warmup_iters:
            for k,v in measurements.items():
                measurements_all[k].append(v)
                DLLogger.log(step=(iter-warmup_iters), data={k: v})
    file_list = glob.glob("*.wav")
    if file_list:
         print("Save audio file Pass!")
    else:
        print("Save audio file Fail!")

    DLLogger.flush()

    print_stats(measurements_all)

if __name__ == '__main__':
    main()
