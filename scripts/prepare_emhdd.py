"""Audit public EMHDD recognizer sample and create reproducible grouped pilot splits."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn
import yaml

COMMIT = '137bb39cf5580c14e54e08e9ff2c7a8a420595f0'

def phash(im):
    pixels = np.asarray(im.convert('L').resize((32, 32)), dtype=float)
    values = dctn(pixels, norm='ortho')[:8, :8].flatten()[1:]
    return values > np.median(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if (source / 'label/classes.txt').read_text().splitlines() != ['no_helmet', 'helmet']:
        raise ValueError('Unexpected class mapping')
    images = sorted((source / 'image').glob('*.jpg'))
    records, hashes, parent, seen_pixels = [], [], list(range(len(images))), {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    for i, p in enumerate(images):
        label = source / 'label' / (p.stem + '.txt')
        if not label.exists():
            raise ValueError(f'Missing label {p.name}')
        with Image.open(p) as im:
            im.load()
            size = list(im.size)
            pixels = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            hashes.append(phash(im))
        if pixels in seen_pixels:
            union(i, seen_pixels[pixels])
        else:
            seen_pixels[pixels] = i
        counts = Counter()
        for line in label.read_text().splitlines():
            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError(f'Invalid YOLO label: {label.name}')
            cls = int(tokens[0]); x, y, w, h = map(float, tokens[1:])
            if cls not in (0, 1) or not all(np.isfinite([x, y, w, h])) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f'Invalid box {label.name}: {line}')
            if x-w/2 < -1e-5 or y-h/2 < -1e-5 or x+w/2 > 1+1e-5 or y+h/2 > 1+1e-5:
                raise ValueError(f'Out of bounds {label.name}: {line}')
            counts[cls] += 1
        records.append({'id': p.stem, 'size': size, 'pixel_sha256': pixels, 'objects': dict(counts)})
    pairs = []
    for i in range(len(images)):
        for j in range(i):
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            if distance <= 6:
                union(i, j)
                pairs.append([images[i].stem, images[j].stem, distance])
    groups = {}
    for i in range(len(images)):
        groups.setdefault(find(i), []).append(i)
    group_list = list(groups.values())
    random.Random(args.seed).shuffle(group_list)
    # Fixed grouped split for this pilot; no test-based model or threshold selection.
    n = len(group_list)
    ntrain, nval = int(n * .7), max(1, int(n * .15))
    partitions = {'train': group_list[:ntrain], 'val': group_list[ntrain:ntrain+nval], 'test': group_list[ntrain+nval:]}
    # Refuse stale files when the seed/source changes, rather than silently leaking splits.
    for split, selected in partitions.items():
        desired = {images[i].stem for group in selected for i in group}
        for directory, pattern in [('images', '*.jpg'), ('labels', '*.txt')]:
            folder = output / directory / split
            stale = {p.stem for p in folder.glob(pattern)} - desired
            if stale:
                raise ValueError(f'Stale {split} files; use a new --output directory: {sorted(stale)}')
    summary = {}
    for split, selected in partitions.items():
        counts = Counter(); ids = []
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for group in selected:
            for i in group:
                p = images[i]; ids.append(p.stem)
                counts.update(records[i]['objects'])
                records[i]['split'] = split
                records[i]['group'] = find(i)
                dest = output / 'images' / split / p.name
                if dest.exists() and dest.resolve() != p.resolve():
                    raise ValueError(f'Image destination points to a different source: {dest}')
                if not dest.exists():
                    dest.symlink_to(p)
                (output / 'labels' / split / (p.stem + '.txt')).write_text((source / 'label' / (p.stem + '.txt')).read_text())
        if not ids or not all(counts[c] for c in (0,1)):
            raise ValueError(f'{split} lacks samples of both classes')
        summary[split] = {'images': len(ids), 'objects': {str(k):v for k,v in counts.items()}, 'ids': ids}
    dataset = {'path': str(output), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': {0:'no_helmet',1:'helmet'}}
    (output / 'data.yaml').write_text(yaml.safe_dump(dataset, sort_keys=False))
    audit = {'source': 'https://github.com/AlanLowe007/EMHDD', 'commit': COMMIT, 'seed': args.seed,
             'images': len(images), 'unique_pixel_hashes':len(seen_pixels), 'near_duplicate_pairs_phash_hamming_le6': pairs,
             'groups': len(groups), 'splits': summary, 'records': records,
             'limitations':['Only public daytime sample; not the claimed full 52,800-image corpus.',
              'No video/camera IDs: perceptual grouping does not guarantee source-disjoint splits.',
              'No explicit dataset license found; research pilot only; obtain permission before redistribution/deployment.',
              'Some labels cover bicycle/e-bike riders or nearby pedestrians; rider association is not provided.']}
    (output / 'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='records'},indent=2))

if __name__ == '__main__':
    main()
