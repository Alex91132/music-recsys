"""Скачать Yambda-50M (события) и привязку треков к исполнителям и альбомам → data/raw/ (~477 МБ).

    python scripts/download.py

Источник: https://huggingface.co/datasets/yandex/yambda — Apache 2.0; датасет опубликован для научных
и исследовательских целей.
"""
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/"
FILES = ["flat/50m/multi_event.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = out / Path(f).name
        if not dst.exists():
            print("→", f)
            urllib.request.urlretrieve(URL + f, dst)


if __name__ == "__main__":
    main()
