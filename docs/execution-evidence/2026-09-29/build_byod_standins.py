"""Build BYOD stand-in archives (org.dimer.receipt-byod.v1) from pinned CORD v2 receipts (CC BY 4.0).

These are stand-ins for a learner's own receipts, used only to exercise the BYOD code paths.
They are not independent data. Output: byod_ds/{inference,adapt,refused_zipslip}.bin (ZIP bytes).
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "repo" / "tools"))
import receipt_byod as rb  # noqa: E402
import receipt_data as rd  # noqa: E402
from receipt_common import FIELDS  # noqa: E402
from receipt_fields import amount  # noqa: E402

POLICY = "dot_decimal_comma_grouping"  # keeps 133 of CORD val+test totals (vs 63 for the other policy)
OUT = HERE / "byod_ds"
OUT.mkdir(exist_ok=True)
PROV = "CORD v2 annotation (naver-clova-ix/cord-v2@7f0115a4b758, CC BY 4.0), repackaged as a BYOD stand-in"


def rows(split):
    return pq.read_table(HERE / "data" / f"{split}.parquet").to_pylist()


def ext(data):
    return "png" if data[:4] == b"\x89PNG" else "jpg"


def annotated(row):
    """Return BYOD field annotations, or None if the receipt cannot be expressed under POLICY."""
    image, info = rd.inspect_image(row["image"]["bytes"])
    refs = rd.references(row["ground_truth"], info, "cord_mixed_v1")["fields"]
    fields, any_present = {}, False
    for f in FIELDS:
        r = refs[f]
        if r["state"] == "not_annotated":
            continue
        if r["state"] != "present_usable":
            if f == "total_amount":
                return None
            continue  # left not_annotated in the stand-in
        parsed = amount(r["raw_text"], POLICY, "IDR")
        if parsed["parse_status"] != "ok" or parsed["normalized_amount"] != r["normalized_amount"]:
            if f == "total_amount":
                return None
            continue  # not expressible under this archive's single number policy; left not_annotated
        assert info["orientation"]["exif"] == 1  # boxes are already in original coordinates
        fields[f] = {"status": "present", "raw_value": r["raw_text"],
                     "value_boxes": [[round(v, 1) for v in b] for b in r["value_boxes"]], "provenance": PROV}
        any_present = any_present or f == "total_amount"
    return fields if any_present else None


def write_zip(path, manifest, images, annotations=None, extra=()):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for name, data in images.items():
            z.writestr(name, data)
        if annotations is not None:
            z.writestr("annotations.jsonl", "".join(json.dumps(a) + "\n" for a in annotations))
        for name, data in extra:
            z.writestr(name, data)


base = {"schema": rb.BYOD_SCHEMA, "authorized": True, "number_format_policy": POLICY, "currency": "IDR",
        "processing_scope": "Kaggle BYOD journey check with CORD v2 stand-ins (CC BY 4.0)"}

# --- inference: 12 CORD test receipts, images only
test = rows("test")
inf_imgs, inf_entries = {}, []
for i, row in enumerate(test[:12]):
    data = row["image"]["bytes"]
    name = f"images/inf{i:03d}.{ext(data)}"
    inf_imgs[name] = data
    inf_entries.append({"receipt_id": f"inf{i:03d}", "image": name, "role": "inference"})
write_zip(OUT / "inference.bin", {**base, "receipts": inf_entries}, inf_imgs)

# --- adapt: validation + test receipts that are expressible under POLICY, split by receipt
pool = []
for split in ("validation", "test"):
    for i, row in enumerate(rows(split)):
        f = annotated(row)
        if f:
            pool.append((f"{split[:3]}{i:03d}", row["image"]["bytes"], f))
pool.sort(key=lambda t: hashlib.sha256(t[0].encode()).hexdigest())
plan = [("validation_model", 20), ("validation_policy", 20), ("test", 25)]
roles, k = {}, 0
for role, n in plan:
    for rid, _, _ in pool[k:k + n]:
        roles[rid] = role
    k += n
for rid, _, _ in pool[k:]:
    roles[rid] = "train"
ad_imgs, ad_entries, ad_ann = {}, [], []
for rid, data, fields in pool:
    name = f"images/{rid}.{ext(data)}"
    ad_imgs[name] = data
    ad_entries.append({"receipt_id": rid, "image": name, "role": roles[rid], "group_id": "g-" + rid})
    ad_ann.append({"receipt_id": rid, "fields": fields})
write_zip(OUT / "adapt.bin", {**base, "receipts": ad_entries}, ad_imgs, ad_ann)

# --- refused: an inference archive carrying a path-traversal member
first = next(iter(inf_imgs.items()))
write_zip(OUT / "refused_zipslip.bin",
          {**base, "receipts": [{"receipt_id": "inf000", "image": first[0], "role": "inference"}]},
          dict([first]), extra=[("../escape.png", first[1])])

from collections import Counter  # noqa: E402

print("adapt pool", len(pool), Counter(roles.values()))
for p in sorted(OUT.glob("*.bin")):
    print(p.name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
