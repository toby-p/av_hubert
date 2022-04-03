
import argparse

from lrs3_prepare import get_file_label, prep_wav, trim_pretrain

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--ffmpeg', type=str, help='path to ffmpeg')
    parser.add_argument('--rank', type=int, help='rank id', default=0)
    parser.add_argument('--nshard', type=int, help='number of shards', default=1)
    parser.add_argument('--step', type=int, help='Steps (1: split labels, 2: trim video/audio, 3: prep audio for '
                                                 'trainval/test, 4: get labels and file list)')
    parser.add_argument('--fraction', type=float, default=None,
                        help="Fraction of the total pretrain dataset folders to use.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for sampling of dataset.")

    args = parser.parse_args()

    # Steps 1-4 from original script:
    trim_pretrain(args.lrs3, args.ffmpeg, args.rank, args.nshard, step=1, fraction=args.fraction, seed=args.seed)
    trim_pretrain(args.lrs3, args.ffmpeg, args.rank, args.nshard, step=2)
    print(f"Extracting audio for trainval/test")
    prep_wav(args.lrs3, args.ffmpeg, args.rank, args.nshard)
    get_file_label(args.lrs3)
