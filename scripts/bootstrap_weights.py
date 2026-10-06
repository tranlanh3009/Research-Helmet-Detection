"""Download the pinned official YOLO11n checkpoint with TLS and checksum verification."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
SHA256='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and hashlib.sha256(args.output.read_bytes()).hexdigest()==SHA256:
        print(args.output);return
    temp=args.output.with_suffix('.pt.part')
    with urllib.request.urlopen(URL,timeout=60) as r: temp.write_bytes(r.read())
    if hashlib.sha256(temp.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Unexpected pretrained weights checksum')
    temp.replace(args.output);print(args.output)

if __name__=='__main__':main()
