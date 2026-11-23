"""Verify RAW-FABRID v1 and prepare exactly the four published metadata files."""
import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from audit_zip import PUBLISHER_SHA256, PUBLISHER_SIZE, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARCHIVE = DATA / 'RAW_FABRID.zip'
URL = ('https://data.mendeley.com/public-files/datasets/db6g85xsyg/files/'
       '93e17d48-b5e6-45f7-9064-be84cc5fb91c/file_downloaded')
METADATA = [f'RAW_FABRID_{kind}_{part}' for kind in ('HighRes', 'Patches')
            for part in ('COCO.json', 'Metadata.csv')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download the official ZIP if absent')
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        if not args.download:
            parser.error(f'Place the publisher ZIP at {ARCHIVE} or use --download')
        with tempfile.TemporaryDirectory(dir=DATA) as folder:
            temp = Path(folder) / 'download.zip'
            with urlopen(URL, timeout=120) as response, temp.open('wb') as file:
                shutil.copyfileobj(response, file, length=1024*1024)
            if temp.stat().st_size != PUBLISHER_SIZE or sha256(temp) != PUBLISHER_SHA256:
                raise ValueError('Downloaded bytes do not match RAW-FABRID v1')
            temp.replace(ARCHIVE)
    if ARCHIVE.stat().st_size != PUBLISHER_SIZE or sha256(ARCHIVE) != PUBLISHER_SHA256:
        raise ValueError('Local archive hash or size differs from the publisher record')
    out = DATA / 'remote_metadata'
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in METADATA:
            contents = archive.read('RAW_FABRID/' + name)
            with tempfile.NamedTemporaryFile(dir=out, delete=False) as f:
                f.write(contents)
            Path(f.name).replace(out / name)
            print(f'{hashlib.sha256(contents).hexdigest()}  {name}')


if __name__ == '__main__':
    main()
