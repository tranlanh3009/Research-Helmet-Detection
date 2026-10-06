"""Measure fixed-threshold head detection on a held-out split; no threshold tuning."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

runtime = Path(os.environ.get('HELMET_RUNTIME','/workspace/helmet-runtime'))
(runtime/'config').mkdir(parents=True,exist_ok=True)
(runtime/'mpl').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(runtime/'config'))
os.environ.setdefault('MPLCONFIGDIR',str(runtime/'mpl'))
import torch
from ultralytics import YOLO


def iou(box, others):
    lo = np.maximum(box[:2],others[:,:2]); hi = np.minimum(box[2:],others[:,2:])
    overlap = np.maximum(hi-lo,0).prod(axis=1)
    area = np.maximum(box[2:]-box[:2],0).prod()
    other_area = np.maximum(others[:,2:]-others[:,:2],0).prod(axis=1)
    return overlap / np.maximum(area+other_area-overlap,1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('artifacts/best.pt'))
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--output',type=Path,default=Path('artifacts/operating-point.json'))
    args = p.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(args.data.read_text()); root = Path(config['path'])
    images = sorted((root/config[args.split]).glob('*.jpg'))
    if not images:
        raise ValueError('Empty evaluation split')
    model = YOLO(str(args.weights)); model.predict(str(images[0]),imgsz=args.imgsz,conf=args.conf,device='cpu',verbose=False)
    counts = {0:{'tp':0,'fp':0,'fn':0},1:{'tp':0,'fp':0,'fn':0}}
    latencies = []; detail = []; tiny = 0
    for image in images:
        with Image.open(image) as im: width,height = im.size
        labels = root/'labels'/args.split/(image.stem+'.txt')
        gt = []
        for line in labels.read_text().splitlines():
            c,x,y,w,h = map(float,line.split())
            gt.append([int(c),(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
            tiny += min(w,h)*args.imgsz < 8
        gt = np.array(gt,dtype=float).reshape(-1,5)
        start = time.monotonic()
        result = model.predict(str(image),imgsz=args.imgsz,conf=args.conf,device='cpu',verbose=False)[0]
        latencies.append((time.monotonic()-start)*1000)
        used = set(); per_image = {0:{'tp':0,'fp':0,'fn':0},1:{'tp':0,'fp':0,'fn':0}}
        boxes = sorted(result.boxes,key=lambda b:float(b.conf),reverse=True)
        for b in boxes:
            c = int(b.cls.item()); box = b.xyxy[0].cpu().numpy()
            candidates = [i for i,g in enumerate(gt) if int(g[0])==c and i not in used]
            match = None
            if candidates:
                overlaps = iou(box,gt[candidates,1:]); index = int(np.argmax(overlaps))
                if overlaps[index] >= .5:
                    match = candidates[index]
            if match is not None:
                used.add(match); per_image[c]['tp'] += 1
            else:
                per_image[c]['fp'] += 1
        for i,g in enumerate(gt):
            if i not in used:
                per_image[int(g[0])]['fn'] += 1
        for c in counts:
            for k in counts[c]: counts[c][k] += per_image[c][k]
        detail.append({'image':image.stem,'counts':per_image})
    for c,v in counts.items():
        v['precision'] = v['tp']/max(v['tp']+v['fp'],1)
        v['recall'] = v['tp']/max(v['tp']+v['fn'],1)
        v['f1'] = 2*v['precision']*v['recall']/max(v['precision']+v['recall'],1e-9)
    report = {'split':args.split,'images':len(images),'confidence':args.conf,'match_iou':.5,
              'class_metrics':{model.names[c]:v for c,v in counts.items()},
              'latency_ms_wall':{'median':float(np.median(latencies)),'p95':float(np.percentile(latencies,95)),
                                 'mean':float(np.mean(latencies)), 'fps_inverse_mean':1000/float(np.mean(latencies))},
              'timing_scope':'CPU batch=1, warmed model, image decoding + preprocessing + model + NMS; no network/video/rendering',
              'objects_with_min_normalized_side_times_imgsz_under_8':int(tiny),
              'per_image':detail}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2))

if __name__ == '__main__':
    main()
