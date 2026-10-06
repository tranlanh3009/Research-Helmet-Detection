"""Export a fixed 640x640 ONNX detector and verify CPU inference parity."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

runtime=Path(os.environ.get('HELMET_RUNTIME','/workspace/helmet-runtime'))
(runtime/'config').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(runtime/'config'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,default=Path('artifacts/best.pt'));args=p.parse_args()
    torch.set_num_threads(4)
    model=YOLO(str(args.weights))
    output=Path(model.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,nms=False,device='cpu'))
    onnx.checker.check_model(onnx.load(str(output)))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider'])
    # Test the actual exported torch graph before NMS, including a nonconstant input.
    rng=np.random.default_rng(42);tensor=rng.random((1,3,640,640),dtype=np.float32)
    with torch.inference_mode(): expected=model.model(torch.from_numpy(tensor))
    if isinstance(expected,(tuple,list)):expected=expected[0]
    actual=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    np.testing.assert_allclose(actual,expected.cpu().numpy(),rtol=1e-3,atol=1e-3)
    report={'onnx':str(output.resolve()),'opset':17,'input_shape':list(tensor.shape),
            'output_shape':list(actual.shape),'max_absolute_error':float(np.max(np.abs(actual-expected.cpu().numpy()))),
            'onnx_checker':'passed','onnxruntime_cpu_parity':'passed',
            'preprocessing':'RGB, letterbox to 640x640, CHW float32 /255',
            'postprocessing':'xywh boxes + scores for no_helmet/helmet; apply confidence filter, NMS and inverse letterbox',
            'limitation':'Synthetic tensor verifies export parity, not real-world model accuracy.'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
