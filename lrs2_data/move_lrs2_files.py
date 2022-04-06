"""Script to move the files from the downloaded lrs2 pretrain dir into the
correct places."""

import os
import shutil

from tqdm import tqdm

SRC_PRETRAIN = "/home/ubuntu/w251-final-project/lrs2/untarred/mvlrs_v1/pretrain"
SRC_MAIN = "/home/ubuntu/w251-final-project/lrs2/untarred/mvlrs_v1/main"
TARGET_DIR = "/home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample"

MANIFEST_SRC = {
    "train.txt": SRC_MAIN,
    "pretrain.txt": SRC_PRETRAIN,
    "test.txt": SRC_MAIN,
    "val.txt": SRC_MAIN,
}

for manifest, src_dir in MANIFEST_SRC.items():

    manifest_target_dir = manifest.split(".")[0]
    if manifest_target_dir in ("train", "val"):
        manifest_target_dir = "trainval"

    target_dir = os.path.join(TARGET_DIR, manifest_target_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    with open(os.path.join(os.getcwd(), manifest), "r") as f:
        videos = f.readlines()
        print(f"{manifest} - {len(videos):,} lines")
        for v in tqdm(videos):
            v = v.split(" ")[0].rstrip()
            video_dir, filename = v.split("/")
            target_video_dir = os.path.join(target_dir, video_dir)
            if not os.path.isdir(target_video_dir):
                os.mkdir(target_video_dir)
            for extension in ("mp4", "txt"):
                src = os.path.join(src_dir, video_dir, f"{filename}.{extension}")
                dest = os.path.join(target_video_dir, f"{filename}.{extension}")
                shutil.copy(src, dest)

