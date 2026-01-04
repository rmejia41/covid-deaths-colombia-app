from __future__ import annotations

from pathlib import Path
import requests

GITHUB_RAW_DATA_URL = (
    "https://github.com/rmejia41/open_datasets/raw/main/"
    "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"
)

TARGET = Path("data") / "Fallecidos_COVID_en_Colombia_20251229_divipola_normalized.xlsx"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"[fetch_data] Already exists: {TARGET} ({TARGET.stat().st_size:,} bytes)")
        return

    print(f"[fetch_data] Downloading dataset to: {TARGET}")
    with requests.get(GITHUB_RAW_DATA_URL, stream=True, timeout=(20, 300)) as r:
        r.raise_for_status()
        tmp = TARGET.with_suffix(TARGET.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(TARGET)

    print(f"[fetch_data] Done: {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

