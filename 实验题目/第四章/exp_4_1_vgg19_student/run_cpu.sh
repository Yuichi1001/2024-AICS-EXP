#!/bin/bash
rm models/vgg19.pth
python stu_upload/generate_pth.py
python stu_upload/evaluate_cpu.py
