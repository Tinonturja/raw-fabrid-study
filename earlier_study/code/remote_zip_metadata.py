#!/usr/bin/env python3
"""Read ZIP catalog and CSV metadata via HTTP ranges without the full download."""
import json
import struct
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4bce7d9c-2438-4b20-92e7-7065a26f73bf"
SIZE = 4_133_046_386
ROOT = Path(__file__).resolve().parents[1]

def get_range(start, end):
    req = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req, timeout=120) as resp:
        if resp.status != 206 or not resp.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Unexpected range response for {start}-{end}: {resp.status} {resp.headers}")
        return resp.read()

def main():
    tail_start = SIZE - 65_557
    tail = get_range(tail_start, SIZE - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP end record not found")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk or cd_disk or n_disk != n_total or n_total == 0xffff or cd_size == 0xffffffff or cd_offset == 0xffffffff:
        raise RuntimeError("Unsupported multi-disk or ZIP64 archive")
    cd = get_range(cd_offset, cd_offset + cd_size - 1)
    entries, pos = [], 0
    while pos < len(cd):
        if cd[pos:pos+4] != b"PK\x01\x02":
            raise RuntimeError(f"Central directory signature mismatch at byte {pos}")
        v = struct.unpack_from("<4s6H3L5H2L", cd, pos)
        _, _, _, flags, method, _, _, crc, csize, usize, fnlen, extralen, comlen, _, _, _, offset = v
        name = cd[pos+46:pos+46+fnlen].decode("utf-8" if flags & 0x800 else "cp437")
        entries.append({"name": name, "flags": flags, "method": method, "crc32": f"{crc:08x}",
                        "compressed_size": csize, "uncompressed_size": usize, "local_offset": offset})
        pos += 46 + fnlen + extralen + comlen
    out_dir = ROOT / "data" / "remote_metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "remote_catalog.json").write_text(json.dumps({"size": SIZE, "entries": entries}, indent=2) + "\n")
    print(json.dumps({"entries": len(entries), "metadata_csv": [e for e in entries if "metadata.csv" in e["name"].lower()]}, indent=2))
    for e in entries:
        if "metadata.csv" not in e["name"].lower():
            continue
        header = get_range(e["local_offset"], e["local_offset"]+29)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Missing local header for {e['name']}")
        local = struct.unpack_from("<4s5H3L2H", header, 0)
        start = e["local_offset"] + 30 + local[-2] + local[-1]
        packed = get_range(start, start + e["compressed_size"] - 1)
        raw = packed if e["method"] == 0 else zlib.decompress(packed, -15) if e["method"] == 8 else None
        if raw is None or len(raw) != e["uncompressed_size"]:
            raise RuntimeError(f"Could not decode metadata member {e['name']}")
        target = out_dir / Path(e["name"]).name
        target.write_bytes(raw)
        print(f"Saved {target} ({len(raw)} bytes)", flush=True)

if __name__ == "__main__":
    main()
