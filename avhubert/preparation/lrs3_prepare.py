# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict

from pydub import AudioSegment
from tqdm import tqdm


def read_csv(csv_file, delimit=','):
    """Just converts a CSV to a dict where column headers become dict keys, and
    the dict values are just lists of the values in each column."""
    lns = open(csv_file, 'r').readlines()
    keys = lns[0].strip().split(delimit)
    df = {key: [] for key in keys}
    for ln in lns[1:]:
        ln = ln.strip().split(delimit)
        for j, key in enumerate(keys):
            df[key].append(ln[j])
    return df


def make_short_manifest(pretrain_dir, output_fn, min_interval: float = 0.4,
                        max_duration: int = 15):
    """Creates the CSV file which identifies the individual "sentences" in each
    video clip. A new sentence is created when the gap between 2 words is
    greater than `min_interval`.

    Args:
        pretrain_dir: filepath to pretrain directory storing subdirectories of
            video files.
        output_fn: filepath to the output CSV file which this function creates.
        min_interval: the minumum period of time between words that defines
            the start of a new "sentence".
        max_duration: if the entire video is less than this value, then it
            is saved whole and not divided into "sentences".
    """
    # Create dict to store processed data:
    df = {'fid': [], 'sent': [], 'start': [], 'end': []}

    # Iterate through each subdir of video files:
    subdirs = os.listdir(pretrain_dir)
    for subdir in tqdm(subdirs):

        # Get list of the text files which contain video transcripts
        # and start/end times for each word in the video:
        txt_fns = glob.glob(os.path.join(pretrain_dir, subdir+'/*txt'))

        for txt_fn in txt_fns:

            # Create relative path of {subdir/video_number} e.g. 00j9bKdiOjk/00003
            fid = os.path.relpath(txt_fn, pretrain_dir)[:-4]
            lines = open(txt_fn).readlines()

            # Get the full video transcript:
            raw_text = lines[0].strip().split(':')[-1].strip()

            # Get the conf integer value (??? - not sure what this is)
            conf = lines[1].strip().split(':')[-1].strip()

            # Work out on which line the word start/end data begins in the file:
            for i_line, ln in enumerate(lines):
                if ln[:4] == 'WORD':
                    start_index = i_line
                    break

            # Load the word start/end data into a list of 1 list per line:
            word_intervals = []
            for ln in lines[start_index+1:]:  # NOQA.
                word, start, end, score = ln.strip().split()
                word_intervals.append([word, float(start), float(end)])

            # Check if the entire video is less than the max duration. If so
            # just add the data to df immediately without further processing:
            if word_intervals[-1][-1] < max_duration:
                df['fid'].append(fid)
                df['sent'].append(raw_text)
                df['start'].append(0)
                df['end'].append(-1)
                continue

            # This part basically divides the full list of words into chunks
            # separated by pauses between words longer than `min_interval`:
            sents, cur_sent = [], []
            for i_word, (word, start, end) in enumerate(word_intervals):
                if i_word == 0:
                    cur_sent.append([word, start, end])
                else:
                    # Check that all the words are in ascending time order:
                    assert start >= cur_sent[-1][-1], f"{fid} , {word}, start-{start}, prev-{cur_sent[-1][-1]}"

                    # If the word start time minus previous word end time is greater than min_interval
                    # then add the current list of words to the main list, and restart current list:
                    if start - cur_sent[-1][-1] > min_interval:
                        sents.append(cur_sent)
                        cur_sent = [[word, start, end]]
                    else:
                        cur_sent.append([word, start, end])
            if len(cur_sent) > 0:
                sents.append(cur_sent)

            # This part aggregates the data for each "sentence" identified, recording
            # unique sentence ID, the words in the sentence, and the start/end times.
            for i_sent, sent in enumerate(sents):
                df['fid'].append(f"{fid}_{i_sent}")
                sent_words = ' '.join([x[0] for x in sent])
                if i_sent == 0:  # First sentence.
                    sent_start = 0
                else:
                    # Sentence starts half way between time of last word of previous
                    # sentence ending and time of first word starting.
                    sent_start = (sent[0][1] + sents[i_sent-1][-1][2])/2
                if i_sent == len(sents)-1:  # Last sentence.
                    sent_end = -1
                else:
                    # Sentence ends half way between time of first word of next
                    # sentence ending and time of last word ending.
                    sent_end = (sent[-1][2] + sents[i_sent+1][0][1])/2
                df['sent'].append(sent_words)
                df['start'].append(sent_start)
                df['end'].append(sent_end)

    # Print the proportion of sentences which are greater than 15s and 20s:
    durations = [y-x for x, y in zip(df['start'], df['end'])]
    num_long = len(list(filter(lambda x: x > 15, durations)))
    print(f"Percentage >15 second: {100*num_long/len(durations)}%")
    num_long = len(list(filter(lambda x: x > 20, durations)))
    print(f"Percentage >20 second: {100*num_long/len(durations)}%")

    # Write the data to CSV:
    with open(output_fn, 'w') as fo:
        fo.write('id,text,start,end\n')
        for i in range(len(df['fid'])):
            fo.write(','.join([df['fid'][i], df['sent'][i], '%.3f' % (df['start'][i]), '%.3f' % (df['end'][i])])+'\n')
    return output_fn


def trim_video_frame(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    """Cut up an input video into the "sentence" chunks identified by the
    `make_short_manifest` function.

    Args:
        csv_fn: filepath where output from `make_short_manifest` is saved.
        raw_dir: directory where the raw (pretrain) videos are located.
        output_dir: directory to create videos.
        ffmpeg: path to ffmpeg executable.
        rank: ???
        nshard: ???
    """
    decimal, fps = 9, 25

    # Create dict of video ID to list of videos for that ID:
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        raw2fid[raw_fid] = raw2fid.get(raw_fid, []) + [[fid, start, end]]

    # Calculate the number of videos per "shard" (minimum = 1):
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)

    # Select a chunk of the videos as the "shard":
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total videos in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")

    # Iterate through the videos:
    for raw_fid, fid_info in tqdm(fid_info_shard):

        # Save the frames from the video into a temp directory as png files:
        raw_path = os.path.join(raw_dir, f"{raw_fid}.mp4")
        tmp_dir = tempfile.mkdtemp()
        cmd = f"{ffmpeg} -i {raw_path} {tmp_dir}/%0{decimal}d.png -loglevel quiet"
        subprocess.call(cmd, shell=True)

        # Total number of png files saved for the video:
        num_frames = len(glob.glob(tmp_dir+'/*png'))

        # Iterate through the start/end times for each "sentence" video:
        for fid, start_sec, end_sec in fid_info:
            sub_dir = os.path.join(tmp_dir, fid)
            os.makedirs(sub_dir, exist_ok=True)
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600  # 24 hours.

            # Calculate first frame as start time * number of frames per second:
            start_frame_id = int(start_sec*fps)

            # Calculate last frame as end time * number of frames per second, or max number of frames:
            end_frame_id = min(int(end_sec*fps), num_frames)

            # Get all the image names for the "sentence" video:
            imnames = [tmp_dir+'/'+str(x+1).zfill(decimal)+'.png' for x in range(start_frame_id, end_frame_id)]

            # Copy relevant images to temp sub-directory:
            for ix, imname in enumerate(imnames):
                shutil.copyfile(imname, sub_dir+'/'+str(ix).zfill(decimal)+'.png')

            # Use ffmpeg to recombine images to video:
            output_path = os.path.join(output_dir, f"{fid}.mp4")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [ffmpeg, "-i", f"{sub_dir}/%0{decimal}d.png", "-y", "-crf", "20", output_path, "-loglevel", "quiet"]
            _ = subprocess.call(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        shutil.rmtree(tmp_dir)
    return


def trim_audio(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    """Cut up an input video's audio into the "sentence" chunks identified by
    the `make_short_manifest` function.

    Args:
        csv_fn: filepath where output from `make_short_manifest` is saved.
        raw_dir: directory where the raw (pretrain) videos are located.
        output_dir: directory to create videos.
        ffmpeg: path to ffmpeg executable.
        rank: ???
        nshard: ???
    """
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total audios in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        tmp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(tmp_dir, 'tmp.wav')
        cmd = ffmpeg + " -i " + os.path.join(raw_dir, raw_fid+'.mp4') + " -f wav -vn -y " + wav_path + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
        raw_audio = AudioSegment.from_wav(wav_path)
        for fid, start_sec, end_sec in fid_info:
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            t1, t2 = int(start_sec*1000), int(end_sec*1000)
            new_audio = raw_audio[t1: t2]
            output_path = os.path.join(output_dir, fid+'.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_audio.export(output_path, format="wav")
        shutil.rmtree(tmp_dir)
    return


def trim_pretrain(root_dir, ffmpeg, rank=0, nshard=1, step=1):
    pretrain_dir = os.path.join(root_dir, 'pretrain')
    print(f"Trim original videos in pretrain")
    csv_fn = os.path.join(root_dir, 'short-pretrain.csv')
    if step == 1:
        print(f"Step 1. Make csv file {csv_fn}")
        make_short_manifest(pretrain_dir, csv_fn)
    else:
        print(f"Step 2. Trim video and audio")
        output_video_dir = os.path.join(root_dir, 'short-pretrain')
        output_audio_dir = os.path.join(root_dir, 'audio/short-pretrain/')
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_audio_dir, exist_ok=True)
        trim_video_frame(csv_fn, pretrain_dir, output_video_dir, ffmpeg, rank, nshard)
        trim_audio(csv_fn, pretrain_dir, output_audio_dir, ffmpeg, rank, nshard)
    return


def prep_wav(lrs3_root, ffmpeg, rank, nshard):
    output_dir = f"{lrs3_root}/audio/"
    video_fns = glob.glob(lrs3_root + '/trainval/*/*mp4') + glob.glob(lrs3_root + '/test/*/*mp4')
    video_fns = sorted(video_fns)
    num_per_shard = math.ceil(len(video_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    video_fns = video_fns[start_id: end_id]
    print(f"{len(video_fns)} videos")
    # subdirs = os.listdir(input_dir)
    for video_fn in tqdm(video_fns):
        base_name = '/'.join(video_fn.split('/')[-3:])
        audio_fn = os.path.join(output_dir, base_name.replace('mp4', 'wav'))
        os.makedirs(os.path.dirname(audio_fn), exist_ok=True)
        cmd = ffmpeg + " -i " + video_fn + " -f wav -vn -y " + audio_fn + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
    return


def get_file_label(lrs3_root):
    video_ids_total, labels_total = [], []
    for split in ['trainval', 'test']:
        subdirs = os.listdir(os.path.join(lrs3_root, split))
        for subdir in tqdm(subdirs):
            video_fns = glob.glob(os.path.join(lrs3_root, split, subdir, '*mp4'))
            video_ids = ['/'.join(x.split('/')[-3:])[:-4] for x in video_fns]
            for video_id in video_ids:
                txt_fn = os.path.join(lrs3_root, video_id+'.txt')
                label = open(txt_fn).readlines()[0].split(':')[1].strip()
                labels_total.append(label)
                video_ids_total.append(video_id)
    pretrain_csv = os.path.join(lrs3_root, 'short-pretrain.csv')
    df = read_csv(pretrain_csv)
    for video_id, label in zip(df['id'], df['text']):
        video_ids_total.append(os.path.join('short-pretrain', video_id))
        labels_total.append(label)
    video_id_fn, label_fn = os.path.join(lrs3_root, 'file.list'), os.path.join(lrs3_root, 'label.list')
    print(video_id_fn, label_fn)
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--ffmpeg', type=str, help='path to ffmpeg')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--step', type=int, help='Steps (1: split labels, 2: trim video/audio, 3: prep audio for '
                                                 'trainval/test, 4: get labels and file list)')
    args = parser.parse_args()
    if args.step <= 2:
        trim_pretrain(args.lrs3, args.ffmpeg, args.rank, args.nshard, step=args.step)
    elif args.step == 3:
        print(f"Extracting audio for trainval/test")
        prep_wav(args.lrs3, args.ffmpeg, args.rank, args.nshard)
    elif args.step == 4:
        get_file_label(args.lrs3)
