#!/usr/bin/env python3
"""Extract only an archive cleared by audit_zip.py."""
import argparse, json, shutil, zipfile
from pathlib import Path, PurePosixPath
from audit_zip import PUBLISHER_SHA256, sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('archive',type=Path); ap.add_argument('audit_report',type=Path); ap.add_argument('destination',type=Path); a=ap.parse_args()
    archive=a.archive.resolve(); dest=a.destination.resolve(); report=json.loads(a.audit_report.read_text())
    recorded = Path(report['archive'])
    if report.get('archive_path_base') == 'report_directory':
        recorded = a.audit_report.resolve().parent / recorded
    if (recorded.resolve()!=archive or report.get('errors') or archive.stat().st_size!=report['archive_bytes']
            or sha256(archive) != PUBLISHER_SHA256):
        raise SystemExit('Archive/report mismatch or audit errors; extraction refused')
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            rel=PurePosixPath(info.filename); target=(dest/Path(*rel.parts)).resolve()
            if not target.is_relative_to(dest): raise SystemExit(f'Unsafe target path: {info.filename}')
            if info.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as out: shutil.copyfileobj(src,out,8*1024*1024)
    print(f'Extracted {len(report["members"])} members to {dest}')

if __name__=='__main__': main()
