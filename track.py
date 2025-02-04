import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sort.tracker import SortTracker


def load_detection(fname):
    # load the detections
    with open(fname, 'r') as f:
        detections = json.load(f)

    # put the detections in the xyxys + labels format
    boxes  = np.array(detections['boxes' ]).reshape(-1, 4)
    scores = np.array(detections['scores']).reshape(-1, 1)
    labels = np.array(detections['labels']).reshape(-1, 1)

    xyxysl = np.hstack((boxes, scores, labels))
    return xyxysl


def track(detections):
    tracker = SortTracker(max_age=3, min_hits=2)
    tracked = []
    for dets in tqdm(detections):
        tracks = tracker.update(dets, None)
        tracked.append(tracks)

    return tracked


def save_track(tracks, fname, vocabulary):
    with open(fname, 'w') as f:
        json.dump({
            'boxes': tracks[:, :4].tolist(),
            'track_ids': tracks[:, 4].astype(int).tolist(),
            'labels': tracks[:, 5].astype(int).tolist(),
            'scores': tracks[:, 6].tolist(),
            'vocabulary': vocabulary,
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track objects in detections.')
    parser.add_argument('detections_folder', type=Path, help='The folder containing the JSON files with the detections.')
    parser.add_argument('-o', '--output', type=Path, default=None, help='The file to save the tracking results.')
    args = parser.parse_args()

    det_files = sorted(args.detections_folder.glob('*.json'))
    detections = [load_detection(d) for d in det_files]
    tracks = track(detections)

    with open(det_files[0], 'r') as f:
        vocabulary = json.load(f)['vocabulary']

    args.output = args.output or args.detections_folder
    args.output.mkdir(parents=True, exist_ok=True)
    for t, d in zip(tracks, det_files):
        save_track(t, args.output / f'{d.stem}_tracked.json', vocabulary)

