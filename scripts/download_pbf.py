#!/usr/bin/env python3
"""
Скачать PBF-дамп OSM для региона (по умолчанию — Санкт-Петербург, BBBike).
После скачивания: uv run python scripts/fetch_osm_addresses.py data/northwestern-fed-district-latest.osm.pbf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

from rich.console import Console
from tqdm import tqdm

# По умолчанию — Северо-Западный ФО с Geofabrik (СПб входит; ~601 MB, стабильная раздача)
# BBBike только СПб (~29 MB) часто недоступен: https://download.bbbike.org/osm/bbbike/SanktPetersburg/
DEFAULT_PBF_URL = "https://download.geofabrik.de/russia/northwestern-fed-district-latest.osm.pbf"
DEFAULT_OUTPUT = "data/northwestern-fed-district-latest.osm.pbf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать OSM PBF-дамп")
    parser.add_argument(
        "--url",
        default=os.environ.get("OSM_PBF_URL", DEFAULT_PBF_URL),
        help="URL PBF-файла",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.environ.get("OSM_PBF_OUTPUT", DEFAULT_OUTPUT),
        help="Путь для сохранения (по умолчанию data/northwestern-fed-district-latest.osm.pbf)",
    )
    args = parser.parse_args()

    console = Console()
    url = args.url.strip()
    out_path = Path(args.output)

    console.print(f"[bold green]Скачиваю PBF[/bold green]")
    console.print(f"[dim]{url}[/dim]")
    console.print(f"[dim]→ {out_path}[/dim]")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "osm-address-dataset/1.0"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        chunk_size = 32 * 1024  # 32 KB — чаще обновления, чтобы скорость не висела на 0
        with open(out_path, "wb") as f:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="PBF",
                file=sys.stderr,
                dynamic_ncols=False,
                mininterval=0.25,
                smoothing=0.1,
            ) as pbar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    console.print(f"[bold green]Готово: [link=file://{out_path.absolute()}]{out_path}[/link][/bold green]")
    console.print(f"[dim]Запуск извлечения адресов:[/dim]")
    console.print(f"[dim]  uv run python scripts/fetch_osm_addresses.py {out_path}[/dim]")


if __name__ == "__main__":
    main()
