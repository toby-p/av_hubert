#!/usr/bin/env bash

lrs2=/home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample
ffmpeg=/usr/bin/ffmpeg


# Step 1: Data preparation
echo "PIPELINE STEP 1"
#fraction=0.01
#seed=42
python3 /home/ubuntu/av_hubert/avhubert/preparation/w251_lrs3_script.py \
 --lrs3 ${lrs2} \
 --ffmpeg ${ffmpeg}
# --fraction ${fraction} \
# --seed ${seed}


# Step 2: Landmark + ROI
echo "PIPELINE STEP 2 - detect_landmark"
python3 /home/ubuntu/av_hubert/avhubert/preparation/detect_landmark.py \
 --root ${lrs2} \
 --landmark ${lrs2}/landmark \
 --manifest ${lrs2}/file.list \
 --cnn_detector /home/ubuntu/w251-final-project/lrs2/dlib_models/mmod_human_face_detector.dat \
 --face_predictor /home/ubuntu/w251-final-project/lrs2/dlib_models/shape_predictor_68_face_landmarks.dat \
 --ffmpeg ${ffmpeg} \
 --nshard 1 --rank 0

echo "PIPELINE STEP 2 - align_mouth"
python3 /home/ubuntu/av_hubert/avhubert/preparation/align_mouth.py \
 --video-direc ${lrs2} \
 --landmark_direc ${lrs2}/landmark \
 --filename-path ${lrs2}/file.list \
 --save-direc ${lrs2}/video \
 --mean-face /home/ubuntu/w251-final-project/lrs2/20words_mean_face.npy \
 --ffmpeg ${ffmpeg} \
 --nshard 1 --rank 0


# Step 3: Count Frames
echo "PIPELINE STEP 3"
python3 /home/ubuntu/av_hubert/avhubert/preparation/count_frames.py \
 --root ${lrs2} \
 --manifest ${lrs2}/file.list \
 --nshard 1 --rank 0


# Step 4: Set up data directory:
echo "PIPELINE STEP 4"
vocab_size=41427
python3 /home/ubuntu/av_hubert/avhubert/preparation/lrs3_manifest.py \
 --lrs3 ${lrs2} \
 --manifest ${lrs2}/file.list \
 --valid-ids /home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample/lrs2-valid.id \
 --vocab-size ${vocab_size}
