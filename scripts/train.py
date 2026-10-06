"""Fine-tune a pretrained detector; select on validation and evaluate held-out test once."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

runtime = Path(os.environ.get('HELMET_RUNTIME', '/workspace/helmet-runtime'))
for directory in ['config', 'mpl']:
    (runtime / directory).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(runtime / 'config'))
os.environ.setdefault('MPLCONFIGDIR', str(runtime / 'mpl'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import torch
import ultralytics
from ultralytics import YOLO


def serialize_metrics(metrics):
    box = metrics.box
    return {'aggregate': {k:float(v) for k,v in metrics.results_dict.items()},
            'per_class': [{'id':int(c), 'name':metrics.names[int(c)],
                           'precision':float(box.p[i]), 'recall':float(box.r[i]),
                           'mAP50':float(box.ap50[i]), 'mAP50_95':float(box.ap[i].mean())}
                          for i,c in enumerate(box.ap_class_index)],
            'speed_ms_per_image': {k:float(v) for k,v in metrics.speed.items()},
            'note':'Precision/recall use the evaluator F1 operating point, not the deployed confidence threshold.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--weights', default=str(runtime / 'yolo11n.pt'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--name', default='emhdd-pilot')
    p.add_argument('--project', type=Path, default=runtime / 'runs')
    p.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    args = p.parse_args()
    torch.set_num_threads(4)
    model = YOLO(args.weights)
    started = time.monotonic()
    model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, seed=42, deterministic=True,
                optimizer='AdamW', lr0=.001, lrf=.1, weight_decay=.0005,
                freeze=10, patience=10, mosaic=.5, close_mosaic=5, fliplr=.5,
                flipud=0., degrees=5., translate=.05, scale=.2,
                project=str(args.project), name=args.name, exist_ok=False,
                plots=True, cache=False, amp=False)
    elapsed = time.monotonic() - started
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / 'weights/best.pt'
    trained = YOLO(str(best))
    test = trained.val(data=str(Path(args.data).resolve()), split='test', imgsz=args.imgsz,
                       batch=args.batch, device=args.device, workers=0, plots=True,
                       project=str(args.project), name=args.name+'-test', exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.artifacts / 'best.pt')
    shutil.copy2(save_dir / 'results.csv', args.artifacts / 'results.csv')
    shutil.copy2(save_dir / 'args.yaml', args.artifacts / 'train-args.yaml')
    manifest = {'model':'YOLO11n pretrained COCO, frozen first 10 modules',
                'torch':torch.__version__, 'ultralytics':ultralytics.__version__,
                'device':args.device, 'epochs_requested':args.epochs,
                'epochs_completed':sum(1 for _ in (save_dir/'results.csv').open())-1,
                'imgsz':args.imgsz, 'batch':args.batch, 'seed':42,
                'train_seconds':elapsed, 'run_directory':str(save_dir),
                'best_checkpoint':str((args.artifacts/'best.pt').resolve()),
                'test':serialize_metrics(test),
                'status':'research pilot; insufficient data for deployment'}
    (args.artifacts / 'metrics.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
