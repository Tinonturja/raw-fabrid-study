#!/usr/bin/env python3
"""Download the public RAW-FABRID archive in verified HTTP byte ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import os
import time

URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4bce7d9c-2438-4b20-92e7-7065a26f73bf"
SIZE = 4_133_046_386
CHUNK = 128 * 1024 * 1024
WORKERS = 32
ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "range_parts"
DEST = ROOT / "data" / "RAW_FABRID.zip"


def fetch(i: int, start: int, end: int) -> tuple[int, int]:
    path = PARTS / f"part_{i:03d}"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return i, expected
    for attempt in range(5):
        tmp = path.with_suffix(".partial")
        try:
            req = Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(req, timeout=180) as response, tmp.open("wb") as out:
                if response.status != 206:
                    raise RuntimeError(f"range {i}: expected HTTP 206, got {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"range {i}: unexpected Content-Range {content_range!r}")
                while block := response.read(1024 * 1024):
                    out.write(block)
            if tmp.stat().st_size != expected:
                raise RuntimeError(f"range {i}: expected {expected} bytes, got {tmp.stat().st_size}")
            tmp.replace(path)
            return i, expected
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(1 + attempt * 2)


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    ranges = [(i, start, min(start + CHUNK, SIZE) - 1)
              for i, start in enumerate(range(0, SIZE, CHUNK))]
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, i, start, end) for i, start, end in ranges]
        for future in as_completed(futures):
            i, amount = future.result()
            completed += amount
            print(f"Downloaded range {i + 1}/{len(ranges)}; {completed / SIZE:.1%} complete", flush=True)
    tmpdest = DEST.with_suffix(".zip.partial")
    with tmpdest.open("wb") as out:
        for i, _, _ in ranges:
            with (PARTS / f"part_{i:03d}").open("rb") as part:
                while block := part.read(8 * 1024 * 1024):
                    out.write(block)
    if tmpdest.stat().st_size != SIZE:
        raise RuntimeError(f"assembled archive size mismatch: {tmpdest.stat().st_size}")
    os.replace(tmpdest, DEST)
    print(f"Assembled {DEST} ({SIZE} bytes)", flush=True)


if __name__ == "__main__":
    main()
