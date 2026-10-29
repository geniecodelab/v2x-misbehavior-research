#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py
data/raw/veremi_nextgen/archives 의 VeReMi NextGen 배포본은 이중 zip 구조이다.

    InTAS_highway_2.zip
      └ InTAS_highway_2/{Train,Test,Validation}/InTAS_highway_2.zip
          └ highway_2/veh_*.json

v2x_evidence_timing.py 는 veh_*.json 이 바로 들어있는 폴더를 --test / --val 로 받으므로,
바깥 zip -> 안쪽 zip 순서로 풀어 다음 위치에 평평하게 놓는다.

    data/raw/veremi_nextgen/extracted/<시나리오>/<분할>/*.json

사용법
    python prepare_data.py                       # InTAS_highway_2, Test+Validation
    python prepare_data.py --splits Test Validation Train
    python prepare_data.py --scenario InTAS_highway_2_constantPositionOffset
"""
import argparse
import io
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ...\V2X_IDS
ARCHIVES = ROOT / "data" / "raw" / "veremi_nextgen" / "archives"
EXTRACTED = ROOT / "data" / "raw" / "veremi_nextgen" / "extracted"


def extract(scenario: str, splits, force: bool = False):
    outer_path = ARCHIVES / f"{scenario}.zip"
    if not outer_path.exists():
        raise FileNotFoundError(f"배포본이 없습니다: {outer_path}")
    made = {}
    with zipfile.ZipFile(outer_path) as outer:
        names = set(outer.namelist())
        for split in splits:
            inner_name = f"{scenario}/{split}/{scenario}.zip"
            if inner_name not in names:
                raise KeyError(f"{outer_path.name} 안에 {inner_name} 이 없습니다")
            target = EXTRACTED / scenario / split
            if target.exists():
                if not force:
                    n = len(list(target.glob("*.json")))
                    print(f"{split}: 이미 있음 ({n} json) -> {target}")
                    made[split] = target
                    continue
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            with outer.open(inner_name) as fh:
                blob = io.BytesIO(fh.read())
            with zipfile.ZipFile(blob) as inner:
                for m in inner.infolist():
                    if m.is_dir() or not m.filename.endswith(".json"):
                        continue
                    # highway_2/veh_1234.json -> veh_1234.json (평평하게)
                    with inner.open(m) as src, open(target / Path(m.filename).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            n = len(list(target.glob("*.json")))
            print(f"{split}: {n} json -> {target}")
            made[split] = target
    return made


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="InTAS_highway_2")
    p.add_argument("--splits", nargs="+", default=["Test", "Validation"])
    p.add_argument("--force", action="store_true", help="이미 푼 폴더를 지우고 다시 푼다")
    a = p.parse_args()
    extract(a.scenario, a.splits, a.force)


if __name__ == "__main__":
    main()
