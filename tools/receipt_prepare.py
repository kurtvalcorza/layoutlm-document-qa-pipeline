"""Build-only bounded network retries for public asset preparation."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import freeze_receipt_assets as assets


def retry_json(url):
    # Use the smaller version-specific metadata endpoint for the same version.
    release = url == "https://api.anaconda.org/package/conda-forge/micromamba"
    endpoint = "https://api.anaconda.org/release/conda-forge/micromamba/2.3.2" if release else url
    for attempt in range(4):
        try:
            request = urllib.request.Request(endpoint, headers={"User-Agent": "DIMER-receipt-freezer/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.load(response)
            if release:
                value = {"files": [dict(item, version="2.3.2") for item in value["distributions"]]}
            return value
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    assets.js = retry_json
    assets.CACHE.mkdir(parents=True, exist_ok=True)
    assets.freeze_data()
    assets.freeze_ocr()
    assets.freeze_model()
