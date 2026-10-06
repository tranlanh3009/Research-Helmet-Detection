"""Local head-region detection. Class 0 does not itself establish a traffic violation."""
import argparse
import json
import os
from pathlib import Path

runtime = Path(os.environ.get('HELMET_RUNTIME', '/workspace/helmet-runtime'))
(runtime/'config').mkdir(parents=True,exist_ok=True)
(runtime/'mpl').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(runtime/'config'))
os.environ.setdefault('MPLCONFIGDIR',str(runtime/'mpl'))
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, default=Path('artifacts/best.pt'))
    p.add_argument('--source', required=True, help='Local image, directory, or video path')
    p.add_argument('--output', type=Path, default=Path('artifacts/predictions'))
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if not Path(args.source).exists():
        p.error('source must be a local path')
    if not 0 < args.conf <= 1:
        p.error('confidence must be in (0,1]')
    torch.set_num_threads(4)
    model = YOLO(str(args.weights))
    if model.names != {0:'no_helmet',1:'helmet'}:
        raise ValueError('Unexpected model class mapping')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'detections.jsonl').open('w') as f:
        for frame, result in enumerate(model.predict(source=args.source,conf=args.conf,
                imgsz=args.imgsz,device=args.device,stream=True,verbose=False)):
            detections = [{'class':model.names[int(b.cls.item())],
                           'confidence':float(b.conf.item()),
                           'xyxy':b.xyxy[0].tolist()} for b in result.boxes]
            f.write(json.dumps({'frame_index':frame,'source':str(result.path),
                                'detections':detections,'status':'pilot_head_detection_only'})+'\n')
            result.save(filename=str(args.output/f'frame_{frame:06d}.jpg'))
    print(args.output/'detections.jsonl')

if __name__ == '__main__':
    main()
