
import os
import random
import shutil

from tqdm import tqdm


FULL_LRS2 = "/home/ubuntu/w251-final-project/lrs2/untarred/lrs2_full_dataset"
SAMPLE_DIR = "/home/ubuntu/w251-final-project/lrs2/untarred/lrs2-sample"

FRACTION = 0.05
SEED = 42

for dataset in ("pretrain", "trainval", "test"):

    target_dir = os.path.join(SAMPLE_DIR, dataset)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    src_dir = os.path.join(FULL_LRS2, dataset)
    all_subdirs = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
    n = int(len(all_subdirs) * FRACTION)
    assert n > 0, "FRACTION too small"
    random.seed(SEED)
    random.shuffle(all_subdirs)
    subdirs = all_subdirs[:n]

    print(f"{dataset} - {n:,} samples")
    for sd in tqdm(subdirs):
        src = os.path.join(FULL_LRS2, dataset, sd)
        dest = os.path.join(target_dir, sd)
        shutil.copytree(src, dest)
