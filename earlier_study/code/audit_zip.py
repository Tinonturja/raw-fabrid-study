#!/usr/bin/env python3
"""Verify publisher hash and inspect ZIP members before any extraction."""
import argparse
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

PUBLISHER_SHA256 = "f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9"
PUBLISHER_SIZE = 4_133_046_386
MAX_MEMBERS = 100_000
MAX_UNCOMPRESSED = 25 * 1024**3
MAX_RATIO = 500

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('archive',type=Path); ap.add_argument('report',type=Path); a=ap.parse_args()
    archive=a.archive.resolve(); errors=[]; members=[]; ncomp=nraw=0
    if archive.stat().st_size != PUBLISHER_SIZE: errors.append(f"size mismatch: {archive.stat().st_size} != {PUBLISHER_SIZE}")
    digest=sha256(archive)
    if digest != PUBLISHER_SHA256: errors.append(f"SHA-256 mismatch: {digest}")
    try:
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist()
            if len(infos)>MAX_MEMBERS: errors.append(f"too many members: {len(infos)}")
            seen=set()
            for i in infos:
                name=i.filename; p=PurePosixPath(name); mode=i.external_attr>>16
                ratio=i.file_size/max(i.compress_size,1); ncomp+=i.compress_size; nraw+=i.file_size
                if p.is_absolute() or '..' in p.parts or '\\' in name: errors.append(f"unsafe member path: {name}")
                if stat.S_ISLNK(mode): errors.append(f"symlink member: {name}")
                if name in seen: errors.append(f"duplicate member path: {name}")
                seen.add(name)
                if ratio>MAX_RATIO and i.file_size>1024*1024: errors.append(f"suspicious compression ratio {ratio:.1f}: {name}")
                if i.flag_bits&1: errors.append(f"encrypted member: {name}")
                members.append({"name":name,"compressed":i.compress_size,"uncompressed":i.file_size,"crc32":f"{i.CRC:08x}","compression":i.compress_type,"ratio":round(ratio,3)})
            if nraw>MAX_UNCOMPRESSED: errors.append(f"uncompressed size exceeds limit: {nraw}")
            bad=z.testzip()
            if bad: errors.append(f"ZIP CRC failure: {bad}")
    except Exception as e:
        errors.append(f"ZIP read/CRC failure: {type(e).__name__}: {e}")
        infos=[]
    report={"archive":os.path.relpath(archive, a.report.resolve().parent),"archive_path_base":"report_directory",
            "publisher_expected_bytes":PUBLISHER_SIZE,"archive_bytes":archive.stat().st_size,
            "publisher_sha256":PUBLISHER_SHA256,"archive_sha256":digest,"member_count":len(members),
            "compressed_member_bytes":ncomp,"uncompressed_member_bytes":nraw,
            "safety_limits":{"max_members":MAX_MEMBERS,"max_uncompressed_bytes":MAX_UNCOMPRESSED,"max_ratio":MAX_RATIO},
            "errors":errors,"members":members}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='members'},indent=2))
    if errors: raise SystemExit('Archive not cleared for extraction')

if __name__=='__main__': main()
