#!/usr/bin/env python3
"""Fetch an archive range independently in smaller, verified HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import os

URL='https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4bce7d9c-2438-4b20-92e7-7065a26f73bf'
SIZE=4_133_046_386
PART_SIZE=128*1024*1024
SUB_SIZE=16*1024*1024
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('part',type=int); args=ap.parse_args()
    start=args.part*PART_SIZE; end=min(SIZE,start+PART_SIZE)
    outdir=ROOT/'data'/'range_repair'; outdir.mkdir(parents=True,exist_ok=True)
    jobs=[(i,s,min(s+SUB_SIZE,end)) for i,s in enumerate(range(start,end,SUB_SIZE))]
    def fetch(job):
        i,s,e=job; p=outdir/f'part_{args.part:03d}_{i:02d}'
        if p.exists() and p.stat().st_size==e-s: return p
        req=Request(URL,headers={'Range':f'bytes={s}-{e-1}'})
        with urlopen(req,timeout=180) as response,p.open('wb') as out:
            if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {s}-{e-1}/'):
                raise RuntimeError(f'Unexpected HTTP range for {s}-{e-1}')
            while b:=response.read(1024*1024): out.write(b)
        if p.stat().st_size!=e-s: raise RuntimeError(f'Unexpected length for {p}')
        return p
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch,j):j for j in jobs}
        for f in as_completed(futures): print('Downloaded',f.result(),flush=True)
    archive=ROOT/'data'/'RAW_FABRID.zip'
    with archive.open('r+b') as out:
        out.seek(start)
        for i,_,_ in jobs:
            with (outdir/f'part_{args.part:03d}_{i:02d}').open('rb') as source:
                while b:=source.read(1024*1024): out.write(b)
        out.flush(); os.fsync(out.fileno())
    print('Replaced archive range',start,end-1,flush=True)

if __name__=='__main__': main()
