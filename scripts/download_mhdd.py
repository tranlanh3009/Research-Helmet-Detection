"""Download pinned MHDD color archive through public GitHub LFS; verify upstream SHA256."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = 'kunalagrawal2611/Motorcycle-Helmet-Detection-Dataset--MHDD-'
COMMIT = '0d483027b6807e06272138a34e8823a5ed78575f'
OID = '884c925925d3ea573d6bf7533486c7c8c7230b05bfe1fa5ab91014fdfa62a9e4'
SIZE = 655088895

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'color.zip'
    if not archive.exists() or archive.stat().st_size != SIZE or digest(archive) != OID:
        body = {'operation': 'download', 'transfers': ['basic'], 'objects': [{'oid': OID, 'size': SIZE}]}
        request = urllib.request.Request(
            f'https://github.com/{REPO}.git/info/lfs/objects/batch',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/vnd.git-lfs+json', 'Accept': 'application/vnd.git-lfs+json'},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            obj = json.load(response)['objects'][0]
        if 'error' in obj:
            raise RuntimeError(obj['error'])
        action = obj['actions']['download']
        request = urllib.request.Request(action['href'], headers=action.get('header', {}))
        temporary = archive.with_suffix('.zip.part')
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
            shutil.copyfileobj(response, f)
        if temporary.stat().st_size != SIZE or digest(temporary) != OID:
            raise RuntimeError('MHDD size/SHA256 mismatch; refusing extraction')
        temporary.replace(archive)
    destination = args.output / 'extracted'
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            resolved = (destination / member.filename).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe ZIP path')
        z.extractall(destination)
    for name in ['License', 'README.md']:
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        with urllib.request.urlopen(url, timeout=30) as r:
            (args.output / name).write_bytes(r.read())
    (args.output / 'provenance.json').write_text(json.dumps({
        'repository': f'https://github.com/{REPO}', 'commit': COMMIT,
        'archive_sha256': OID, 'archive_bytes': SIZE, 'license': 'CC BY 4.0',
    }, indent=2) + '\n')
    print(destination)

if __name__ == '__main__':
    main()
