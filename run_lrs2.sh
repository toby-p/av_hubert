#!/usr/bin/env bash

lrs2=/home/ubuntu/w251-final-project/lrs2/untarred/mvlrs_v1
ffmpeg=/usr/bin/ffmpeg
python3 /home/ubuntu/av_hubert/avhubert/preparation/w251_lrs3_script.py --lrs3 ${lrs2} --ffmpeg ${ffmpeg}
