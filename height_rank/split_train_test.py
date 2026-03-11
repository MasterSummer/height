import os
import random

video_dir = 'tmp_test_videos'
train_list_path = 'height_rank/train_list.txt'
test_list_path = 'height_rank/test_list.txt'

videos = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
random.seed(42)
random.shuffle(videos)

split_idx = int(len(videos) * 0.75)
train_videos = videos[:split_idx]
test_videos = videos[split_idx:]

with open(train_list_path, 'w') as f:
    for vid in train_videos:
        f.write(f'{vid}\n')

with open(test_list_path, 'w') as f:
    for vid in test_videos:
        f.write(f'{vid}\n')

print(f'Train: {len(train_videos)}, Test: {len(test_videos)}')
