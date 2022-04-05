#!/usr/bin/env bash

lrs2=/home/ubuntu/w251-final-project/lrs2/untarred/mvlrs_v1
ffmpeg=/usr/bin/ffmpeg

python3 /home/ubuntu/av_hubert/avhubert/preparation/detect_landmark.py \
 --root ${lrs2} \
 --landmark ${lrs2}/landmark \
 --manifest ${lrs2}/file.list \
 --cnn_detector /home/ubuntu/av_hubert/avhubert/preparation/cnn_face_detector.py \
 --face_detector /home/ubuntu/av_hubert/avhubert/preparation/face_landmark_detection.py \
 --ffmpeg ${ffmpeg} \
 --rank 0 --nshard 1
