"""Retrieve the pinned public EMHDD sample; no access to its unreleased full corpus."""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

COMMIT = '137bb39cf5580c14e54e08e9ff2c7a8a420595f0'
SHA256 = '8b12d5af28ee62e94b0a46df284ec616ede2363077c7e55535f7a73a592d1fa3'
SIZE = 372531792

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    archive=args.output/'emhdd.zip'
    if not archive.exists() or archive.stat().st_size!=SIZE or digest(archive)!=SHA256:
        url=f'https://codeload.github.com/AlanLowe007/EMHDD/zip/{COMMIT}'
        temporary=archive.with_suffix('.zip.part')
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:
            shutil.copyfileobj(response,f)
        if temporary.stat().st_size!=SIZE or digest(temporary)!=SHA256:
            raise ValueError('Unexpected dataset artifact: review upstream provenance before updating hash')
        temporary.replace(archive)
    destination=args.output;destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination/name).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe ZIP path')
        z.extractall(destination)
    print(destination/f'EMHDD-{COMMIT}'/'Recognizer/1.Daytime')

if __name__=='__main__': main()
