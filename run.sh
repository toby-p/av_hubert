#!/usr/bin/env bash

lrs3=/home/ubuntu/w251-final-project/data
ffmpeg=/usr/bin/ffmpeg
python3 /home/ubuntu/av_hubert/avhubert/preparation/w251_lrs3_script.py --lrs3 ${lrs3} --ffmpeg ${ffmpeg} --fraction 0.06928 --seed 42
