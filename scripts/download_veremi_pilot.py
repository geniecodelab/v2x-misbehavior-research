from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


RECORD_ID = "19665762"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"
FILES = {
    "InTAS_highway_2.zip": "d5fefb873c5e766b702bd036f18a326d",
    "InTAS_highway_2_constantPositionOffset.zip": "4b8aa6b684a093de218256bae74c1a6e",
    "InTAS_highway_2_randomPositionOffset.zip": "0b72ffd770351817deb18fa3ec630a61",
    "InTAS_highway_2_constantSpeedOffset.zip": "27a7d170fd937b65736d06054ba97bc4",
    "InTAS_highway_2_randomSpeedOffset.zip": "092c7f5d150f613414de0b0797430968",
    "LICENSE.txt": "2ab724713fdaf49e4523c4503bfd068d",
    "README.md": "b2a121a1ab2f362283fb60fffaaf31e2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the approved VeReMi NextGen pilot")
    parser.add_argument(
        "--destination",
        default="data/raw/veremi_nextgen/archives",
        help="raw archive destination",
    )
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)

    for filename, expected_md5 in FILES.items():
        target = destination / filename
        if target.is_file():
            actual = md5(target)
            if actual != expected_md5:
                raise SystemExit(
                    f"existing file has an unexpected MD5 and was preserved: {target} ({actual})"
                )
            print(f"verified existing: {filename}")
            continue

        partial = target.with_suffix(target.suffix + ".part")
        request = Request(f"{BASE_URL}/{filename}?download=1", headers={"User-Agent": "v2x-ids/0.1"})
        print(f"downloading: {filename}")
        with urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        actual = md5(partial)
        if actual != expected_md5:
            raise SystemExit(
                f"downloaded file MD5 mismatch; partial preserved for inspection: {partial} ({actual})"
            )
        partial.replace(target)
        print(f"verified: {filename} {expected_md5}")


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
