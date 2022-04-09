#!/usr/bin/env bash

lrs2=/home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample
ffmpeg=/usr/bin/ffmpeg


# Step 1: Data preparation
#echo "PIPELINE STEP 1"
#python3 /home/ubuntu/av_hubert/avhubert/preparation/w251_lrs3_script.py \
# --lrs3 ${lrs2} \
# --ffmpeg ${ffmpeg}


# Step 2: Landmark + ROI
#echo "PIPELINE STEP 2 - detect_landmark"
#python3 /home/ubuntu/av_hubert/avhubert/preparation/detect_landmark.py \
# --root ${lrs2} \
# --landmark ${lrs2}/landmark \
# --manifest ${lrs2}/file.list \
# --cnn_detector /home/ubuntu/w251-final-project/lrs2/dlib_models/mmod_human_face_detector.dat \
# --face_predictor /home/ubuntu/w251-final-project/lrs2/dlib_models/shape_predictor_68_face_landmarks.dat \
# --ffmpeg ${ffmpeg} \
# --nshard 1 --rank 0

#echo "PIPELINE STEP 2 - align_mouth"
#python3 /home/ubuntu/av_hubert/avhubert/preparation/align_mouth.py \
# --video-direc ${lrs2} \
# --landmark-direc ${lrs2}/landmark \
# --filename-path ${lrs2}/file.list \
# --save-direc ${lrs2}/video \
# --mean-face /home/ubuntu/w251-final-project/lrs2/20words_mean_face.npy \
# --ffmpeg ${ffmpeg} \
# --nshard 1 --rank 0


# Step 3: Count Frames
#echo "PIPELINE STEP 3"
#python3 /home/ubuntu/av_hubert/avhubert/preparation/count_frames.py \
# --root ${lrs2} \
# --manifest ${lrs2}/file.list \
# --nshard 1 --rank 0


# Step 4: Set up data directory:
#echo "PIPELINE STEP 4"
#vocab_size=41427
#vocab_size=6850
#python3 /home/ubuntu/av_hubert/avhubert/preparation/lrs3_manifest.py \
# --lrs3 ${lrs2} \
# --valid-ids /home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample/lrs2-valid.id \
# --vocab-size ${vocab_size}
# --manifest ${lrs2}/file.list \


# Step 5: Finetune model:
echo "PIPELINE STEP 5 - FINETUNING MODEL"
data=/home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample/433h_data
fairseq-hydra-train \
 --config-dir /home/ubuntu/av_hubert/avhubert/conf/finetune \
 --config-name self_large_vox_433h.yaml \
 task.data=${data} \
 task.label_dir=${data} \
 task.tokenizer_bpe_model=sentencepiece \
 model.w2v_path=/home/ubuntu/w251-final-project/models/large_vox_iter5.pt \
 hydra.run.dir=/home/ubuntu/w251-final-project/finetune \
 common.user_dir=`pwd`
