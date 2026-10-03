"""Stages of DIMER_Document_QA_LayoutLM_vs_Pix2Struct_Workshop.ipynb, run in its isolated environment.

The notebook carries this file byte-for-byte, writes it next to the hash-locked requirements, and runs
every model stage as ``python document_qa_workshop.py --config <json> --stage <name>`` with the
interpreter of a uv-built Python 3.12 virtual environment. Nothing is installed into the notebook
kernel, which only displays the text, JSON and PNG files written here.

Function bodies are the notebook's former cell code (review revision of 2026-10-02), moved unchanged
except where noted: each model is loaded inside the stage that uses it (so "release" is the end of the
stage process), results that used to be kernel objects are written under ``WORK_DIR/state``, and the
test-page pixels are re-read from the verified CORD-v2 shards and checked against their recorded
SHA-256 instead of being kept in memory between cells.
"""
# ruff: noqa: E501,E701,E702,E731,E741,SIM108,UP017
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import platform
import random
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STAGE_FORMAT = "dimer.document-qa-workshop.stages.v1"

# ---- Configuration (overwritten by configure() from the notebook controls, section 3) ---------------
USE_BYOD = False
BYOD_PATH = ""
MAX_NEW_TOKENS = 32
RUN_PARAPHRASE_EXPERIMENT = True
PARAPHRASE_MAX_RECORDS = 20
RUN_UNANSWERABLE_PROBE = True
UNANSWERABLE_PAGES = 5
RUN_MODALITY_ROBUSTNESS = True
ROBUSTNESS_MAX_RECORDS = 20
OUTPUT_DIR = "outputs/document_qa_comparison"
WORK_DIR = "work/document_qa_workshop"

# Heavy libraries load only in the stages that use them (see load_torch).
torch = None
DEVICE = "cpu"
layout_model = layout_tokenizer = layout_sep_id = None
pix_model = pix_processor = None
FONT_BYTES = None

# ---- Immutable model provenance (section 5) -----------------------------------------------------------
LAYOUTLM_MANIFEST = {
    "format": "dimer_hf_snapshot",
    "formatVersion": 1,
    "modelKey": "layoutlm-document-qa",
    "modelId": "impira/layoutlm-document-qa",
    "revision": "beed3c4d02d86017ebca5bd0fdf210046b907aa6",
    "files": [
        {"path": "README.md", "bytes": 2326, "sha256": "40fd65fc734cc7eb73cc815465b8073bc4e5b7484406e537c7b8d763f88493f7"},
        {"path": "config.json", "bytes": 789, "sha256": "6d0fc068193109d0d053fa4de00963778beffbde05067c3e9f3454235044380f"},
        {"path": "merges.txt", "bytes": 456356, "sha256": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862"},
        {"path": "model.safetensors", "bytes": 511200628, "sha256": "e4bbad3e4a1b5ae50c787b7afd6049a0bfa99fd823b50436e444e092ae2347b9"},
        {"path": "special_tokens_map.json", "bytes": 239, "sha256": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"},
        {"path": "tokenizer.json", "bytes": 1355881, "sha256": "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd"},
        {"path": "tokenizer_config.json", "bytes": 315, "sha256": "ea11996be5d083d63c72700810b18a6cfdf78c55131b754a75ae94e66b0ad6ab"},
        {"path": "vocab.json", "bytes": 798293, "sha256": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e"},
    ],
    "totalBytes": 513814827,
}
PIX2STRUCT_MANIFEST = {
    "format": "dimer_hf_snapshot",
    "formatVersion": 1,
    "modelKey": "pix2struct-docvqa-base",
    "modelId": "google/pix2struct-docvqa-base",
    "revision": "63f6b3de436e39f75c7a486881a9c2c14a7f4e89",
    "files": [
        {"path": "README.md", "bytes": 4476, "sha256": "794175546e80948e4efef30e95ebe859fdd26d07bcda26f738a1c97feb1e912a"},
        {"path": "config.json", "bytes": 4892, "sha256": "8d39973772a4218b555e30daecabdd5ea11aa1345dd711ff7f88fa90750b464f"},
        {"path": "model.safetensors", "bytes": 1129177976, "sha256": "067f7f314d87fa56daa5bcfaf36fa0b33ceebf7b7d4fae6a1e51ab7af64ee0b5"},
        {"path": "preprocessor_config.json", "bytes": 249, "sha256": "c84e4eebc84171d6069533d9f0147ec7b4afd02ab78697cb5c30f9419ef7dc45"},
        {"path": "special_tokens_map.json", "bytes": 2201, "sha256": "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"},
        {"path": "spiece.model", "bytes": 851388, "sha256": "7fd650335add59bed55a432186ca0437a09e185c2d241faab468a538fe6bcf94"},
        {"path": "tokenizer.json", "bytes": 3265159, "sha256": "0af109b23840545ef2c286073f4373959badba1faa73c8557881d5126f6287c9"},
        {"path": "tokenizer_config.json", "bytes": 2583, "sha256": "5fdb6767a49aca48fdfa43d0279321918185fc4997bdb3ea72bf3a6301a1b43d"},
    ],
    "totalBytes": 1133308924,
}
LAYOUTLM_DIR = Path("weights/layoutlm-document-qa")
PIX2STRUCT_DIR = Path("weights/pix2struct-docvqa-base")
MANIFEST_NAME = "dimer-base-manifest.json"
LAYOUTLM_ID = LAYOUTLM_MANIFEST["modelId"]; LAYOUTLM_REVISION = LAYOUTLM_MANIFEST["revision"]
PIX_ID = PIX2STRUCT_MANIFEST["modelId"]; PIX_REVISION = PIX2STRUCT_MANIFEST["revision"]

# ---- CORD-v2 provenance (section 6) -------------------------------------------------------------------
CORD_REPO = "naver-clova-ix/cord-v2"
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
CORD_LICENSE = "CC BY 4.0"
CORD_FILES = {
    "test": {
        "path": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
        "bytes": 234202795,
        "sha256": "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7",
        "rows": 100,
    },
    "validation": {
        "path": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
        "bytes": 242080800,
        "sha256": "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15",
        "rows": 100,
    },
}
CORD_DIR = Path("weights/cord-v2-full")

# ---- Question templates and split (sections 7-8) ---------------------------------------------------------
QUESTION_TEMPLATES={
    "total.total_price":"What is the total amount?",
    "sub_total.subtotal_price":"What is the subtotal?",
    "sub_total.tax_price":"What is the tax amount?",
    "sub_total.service_price":"What is the service charge?",
    "sub_total.discount_price":"What is the discount amount?",
    "total.cashprice":"How much cash was paid?",
    "total.changeprice":"How much change was given?",
    "total.creditcardprice":"How much was paid by card?",
    "total.menuqty_cnt":"How many items were bought?",
    "menu.nm":"What is the name of the first item?",
}
FIRST_ITEM_CATEGORY="menu.nm"
SAMPLE_SEED=42
SAMPLE_PAGE_SPLIT={"train":119,"validation":30,"test":50}

PARAPHRASES={
    "total.total_price":"How much is the total?","sub_total.subtotal_price":"How much is the subtotal?",
    "sub_total.tax_price":"What tax amount is shown?","sub_total.service_price":"How much is the service charge?",
    "sub_total.discount_price":"What discount was applied?","total.cashprice":"How much cash did the customer pay?",
    "total.changeprice":"How much change was returned?","total.creditcardprice":"How much was charged to the card?",
    "total.menuqty_cnt":"How many items are on the receipt?","menu.nm":"What is the first item listed?",
}
UNANSWERABLE_QUESTION="What is the loyalty membership number?"

MAX_SEQ_LEN=512; DOC_STRIDE=128; MAX_ANSWER_TOKENS=15; BOX_GRID=1000
PUNCT_RE=re.compile(r"[^\w\s]")
ANLS_THRESHOLD=0.5
DIGIT_RE=re.compile(r"\d")
QUESTION_STOP_WORDS=frozenset([
    "what","is","the","how","much","many","was","were","paid","given","by","of","name",
    "first","item","items","bought","amount","charge"
])
BYOD_LIMITS={"records":(1,500),"pages":(1,100),"words":(1,2000),"question_chars":256,"image_side":(16,4096),
             "zip_files":1000,"zip_bytes":2_000_000_000}


def validate_controls():
    """The section 3 control checks, repeated here so a stage never runs on an out-of-range setting."""
    if isinstance(MAX_NEW_TOKENS, bool) or not isinstance(MAX_NEW_TOKENS, int) or not 1 <= MAX_NEW_TOKENS <= 128:
        raise ValueError("MAX_NEW_TOKENS must be in 1..128")
    if not 1 <= PARAPHRASE_MAX_RECORDS <= 20:
        raise ValueError("PARAPHRASE_MAX_RECORDS must be in 1..20")
    if not 1 <= UNANSWERABLE_PAGES <= 10:
        raise ValueError("UNANSWERABLE_PAGES must be in 1..10")
    if isinstance(ROBUSTNESS_MAX_RECORDS, bool) or not isinstance(ROBUSTNESS_MAX_RECORDS, int) or not 1 <= ROBUSTNESS_MAX_RECORDS <= 50:
        raise ValueError("ROBUSTNESS_MAX_RECORDS must be an integer in 1..50; correct it in this cell and run again from here")


def configure(config):
    """Apply the notebook controls written by run_stage() (stage_config.json)."""
    global USE_BYOD, BYOD_PATH, MAX_NEW_TOKENS, RUN_PARAPHRASE_EXPERIMENT, PARAPHRASE_MAX_RECORDS
    global RUN_UNANSWERABLE_PROBE, UNANSWERABLE_PAGES, RUN_MODALITY_ROBUSTNESS, ROBUSTNESS_MAX_RECORDS
    global OUTPUT_DIR, WORK_DIR
    USE_BYOD = bool(config["use_byod"]); BYOD_PATH = str(config["byod_path"])
    MAX_NEW_TOKENS = config["max_new_tokens"]
    RUN_PARAPHRASE_EXPERIMENT = bool(config["run_paraphrase_experiment"])
    PARAPHRASE_MAX_RECORDS = config["paraphrase_max_records"]
    RUN_UNANSWERABLE_PROBE = bool(config["run_unanswerable_probe"])
    UNANSWERABLE_PAGES = config["unanswerable_pages"]
    RUN_MODALITY_ROBUSTNESS = bool(config["run_modality_robustness"])
    ROBUSTNESS_MAX_RECORDS = config["robustness_max_records"]
    OUTPUT_DIR = str(config["output_dir"]); WORK_DIR = str(config["work_dir"])
    validate_controls()


# ---- State shared between stages -------------------------------------------------------------------------
def state_path(name):
    return Path(WORK_DIR)/"state"/name


def save_state(name,value):
    path=state_path(name); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def load_state(name):
    path=state_path(name)
    if not path.is_file():
        raise RuntimeError(f"{path} is missing: an earlier stage has not run in this session. Choose Run all.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_torch():
    """Import torch once and choose the device (the former runtime cell)."""
    global torch, DEVICE
    if torch is None:
        import torch as _torch
        torch = _torch
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch


# ---- Section 5: snapshot staging ---------------------------------------------------------------------
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def stage_and_verify_model(root,manifest):
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True,exist_ok=True)
    (root/MANIFEST_NAME).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for entry in manifest["files"]:
        path=root/entry["path"]
        if not path.is_file():
            hf_hub_download(
                repo_id=manifest["modelId"],filename=entry["path"],revision=manifest["revision"],
                local_dir=str(root),
            )
    for entry in manifest["files"]:
        path=root/entry["path"]
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"{manifest['modelId']} {entry['path']} size mismatch")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"{manifest['modelId']} {entry['path']} SHA-256 mismatch")
    return {"model_id":manifest["modelId"],"revision":manifest["revision"],"files":len(manifest["files"]),"total_bytes":manifest["totalBytes"]}


# ---- Section 7: page reconstruction ----------------------------------------------------------------------
def quad_to_box(quad,width,height):
    xs=[float(quad[k]) for k in ("x1","x2","x3","x4")]
    ys=[float(quad[k]) for k in ("y1","y2","y3","y4")]
    x0,y0=max(0.0,min(xs)),max(0.0,min(ys))
    x1,y1=min(float(width),max(xs)),min(float(height),max(ys))
    return None if x1<=x0 or y1<=y0 else [x0,y0,x1,y1]


def decode_image_cell(cell):
    if isinstance(cell,Image.Image): return cell.convert("RGB")
    if isinstance(cell,dict):
        data=cell.get("bytes")
        if data is not None:
            im=Image.open(io.BytesIO(data)); im.load(); return im.convert("RGB")
        path=cell.get("path")
        if path and Path(path).is_file():
            im=Image.open(path); im.load(); return im.convert("RGB")
    if isinstance(cell,(bytes,bytearray)):
        im=Image.open(io.BytesIO(cell)); im.load(); return im.convert("RGB")
    raise TypeError(f"Unsupported CORD image representation: {type(cell).__name__}")


def pixel_sha256(image):
    return hashlib.sha256(f"{image.width}x{image.height}:".encode()+image.tobytes()).hexdigest()


def page_from_ground_truth(ground_truth,page_id,image,source_split,source_row):
    data=json.loads(ground_truth) if isinstance(ground_truth,str) else ground_truth
    size=data["meta"]["image_size"]
    width,height=int(size["width"]),int(size["height"])
    if image.size != (width,height):
        raise RuntimeError(f"{page_id}: image {image.size} != annotation {(width,height)}")
    words,boxes,lines=[],[],[]
    for line in data.get("valid_line",[]):
        start=len(words); values=[]
        for word in line.get("words",[]):
            text=" ".join(str(word.get("text","")).split())
            box=quad_to_box(word["quad"],width,height) if text else None
            if box is None: continue
            if not int(word.get("is_key",0)): values.append(len(words))
            words.append(text); boxes.append(box)
        if len(words)>start:
            contiguous=bool(values) and values==list(range(values[0],values[-1]+1))
            lines.append({
                "category":str(line.get("category","")),"group_id":int(line.get("group_id",-1)),
                "start":start,"end":len(words)-1,
                "value_start":values[0] if contiguous else None,
                "value_end":values[-1] if contiguous else None,
            })
    pixel_digest=pixel_sha256(image)
    ocr_digest=hashlib.sha256(json.dumps([words,boxes],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    return {
        "id":page_id,"image":image,"words":words,"boxes":boxes,"image_size":[width,height],"lines":lines,
        "source_split":source_split,"source_row":source_row,"pixel_sha256":pixel_digest,"ocr_sha256":ocr_digest,
    }


# ---- Section 8: split ------------------------------------------------------------------------------------
def questions_for_page(page):
    by_category=defaultdict(list)
    for line in page["lines"]: by_category[line["category"]].append(line)
    out=[]
    for category,question in QUESTION_TEMPLATES.items():
        candidates=by_category.get(category,[])
        if not candidates: continue
        if category==FIRST_ITEM_CATEGORY:
            topmost=min(candidates,key=lambda line:(page["boxes"][line["start"]][1],line["start"]))
            same_group=[line for line in candidates if line["group_id"]==topmost["group_id"]]
            if len(same_group)!=1: continue
            line=topmost
        elif len(candidates)==1:
            line=candidates[0]
        else:
            continue
        if line["value_start"] is None: continue
        start,end=int(line["value_start"]),int(line["value_end"])
        answer=" ".join(page["words"][start:end+1])
        out.append({
            "page_id":page["id"],"field":category,"question":question,"words":page["words"],"boxes":page["boxes"],
            "image_size":page["image_size"],"answer_start":start,"answer_end":end,"answers":[answer],
        })
    return out


def split_pages(pages_by_source):
    """The former section 8 cell: deduplicate, shuffle with SAMPLE_SEED, split 119 / 30 / 50 pages."""
    pool=[dict(page) for split in sorted(pages_by_source) for page in pages_by_source[split]]
    seen=set(); supported=[]
    for page in pool:
        key=tuple(str(w).lower() for w in page["words"])
        if key in seen or not questions_for_page(page): continue
        seen.add(key); supported.append(page)

    random.Random(SAMPLE_SEED).shuffle(supported)
    page_splits={}; qa_splits={}; offset=0
    for name in ("train","validation","test"):
        chosen=supported[offset:offset+SAMPLE_PAGE_SPLIT[name]]; offset+=SAMPLE_PAGE_SPLIT[name]
        page_splits[name]=chosen
        rows=[]
        for page in chosen: rows.extend(questions_for_page(page))
        qa_splits[name]=[{"id":f"{name}-{i:04d}",**row} for i,row in enumerate(rows)]

    page_counts={k:len(v) for k,v in page_splits.items()}
    qa_counts={k:len(v) for k,v in qa_splits.items()}
    if len(supported)!=199: raise RuntimeError(f"Expected 199 supported unique pages, got {len(supported)}")
    if page_counts!={"train":119,"validation":30,"test":50}: raise RuntimeError(page_counts)
    if qa_counts!={"train":595,"validation":152,"test":229}: raise RuntimeError(qa_counts)

    owners={}
    for split,pages in page_splits.items():
        for page in pages:
            if page["id"] in owners: raise RuntimeError(f"Page leakage: {page['id']}")
            owners[page["id"]]=split

    test_records=qa_splits["test"]
    qa_digest=hashlib.sha256("\n".join(
        json.dumps([r["id"],r["page_id"],r["field"],r["question"],r["answers"],r["answer_start"],r["answer_end"]],separators=(",",":"),ensure_ascii=False)
        for r in test_records
    ).encode()).hexdigest()
    return {"supported_unique_pages":len(supported),"page_splits":page_counts,"qa_splits":qa_counts,
            "test_qa_digest":qa_digest,"test_records":test_records,
            "test_pages":[{k:v for k,v in p.items() if k!="image"} for p in page_splits["test"]]}


def load_dataset():
    """Test records and test pages (with pixels re-read from the verified shards and digest-checked)."""
    data=load_state("dataset.json")
    cord_paths=load_state("cord.json")
    import pyarrow.parquet as pq
    tables={}
    test_pages={}
    for page in data["test_pages"]:
        split=page["source_split"]
        if split not in tables: tables[split]=pq.read_table(cord_paths[split],columns=["image"])
        image=decode_image_cell(tables[split].slice(page["source_row"],1).to_pylist()[0]["image"])
        if pixel_sha256(image)!=page["pixel_sha256"]:
            raise RuntimeError(f"{page['id']}: page pixels differ from the split stage (SHA-256)")
        test_pages[page["id"]]={**page,"image":image}
    return data["test_records"],test_pages


# ---- Section 9: common answer metrics ----------------------------------------------------------------
def normalize_answer(text): return " ".join(PUNCT_RE.sub(" ",str(text).lower()).split())


def levenshtein(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]


def anls(prediction,golds,threshold=ANLS_THRESHOLD):
    pred=normalize_answer(prediction); best=0.0
    for gold in golds:
        ref=normalize_answer(gold); longest=max(len(pred),len(ref))
        sim=1.0 if longest==0 else 1.0-levenshtein(pred,ref)/longest
        best=max(best,sim)
    return best if best>=threshold else 0.0


def exact_match(prediction,golds):
    pred=normalize_answer(prediction)
    return any(pred==normalize_answer(g) for g in golds)


def answer_in_ocr(prediction,words,max_span_words=20):
    target=normalize_answer(prediction)
    if not target: return False
    cleaned=[normalize_answer(w) for w in words]
    for start in range(len(cleaned)):
        parts=[]
        for end in range(start,min(len(cleaned),start+max_span_words)):
            if cleaned[end]: parts.append(cleaned[end])
            joined=" ".join(parts)
            if joined==target: return True
            if len(joined)>len(target)+32: break
    return False


def corpus_metrics(rows):
    if not rows: raise ValueError("No prediction rows")
    return {
        "n":len(rows),"anls":float(np.mean([r["anls"] for r in rows])),
        "exact_match":float(np.mean([r["exact_match"] for r in rows])),
        "empty_rate":float(np.mean([r["empty"] for r in rows])),
        "answer_in_ocr_rate":float(np.mean([r["answer_in_ocr"] for r in rows])),
    }


# ---- Section 10: OCR baselines -------------------------------------------------------------------------
def last_number_answer(words):
    for word in reversed(words):
        if DIGIT_RE.search(word): return word
    return ""


def prefix_overlap(a,b):
    n=0
    for x,y in zip(a,b,strict=False):
        if x!=y: break
        n+=1
    return n


def keyword_lookup_answer(question,words,min_overlap=3):
    content=[w for w in normalize_answer(question).split() if w not in QUESTION_STOP_WORDS]
    page=[normalize_answer(w) for w in words]
    best_index,best_overlap=None,min_overlap-1
    for idx,word in enumerate(page):
        overlap=max((prefix_overlap(word,q) for q in content),default=0)
        if overlap>=best_overlap and overlap>=min_overlap:
            best_index,best_overlap=idx,overlap
    if best_index is not None:
        for word in words[best_index+1:]:
            if DIGIT_RE.search(word): return word
    return last_number_answer(words)


def baseline_rows(test_records,name,answer_fn):
    rows=[]
    for r in test_records:
        answer=answer_fn(r)
        rows.append({
            "record_id":r["id"],"page_id":r["page_id"],"field":r["field"],"question":r["question"],"gold_answer":r["answers"][0],
            "model":name,"answer":answer,"normalized_answer":normalize_answer(answer),"anls":anls(answer,r["answers"]),
            "exact_match":exact_match(answer,r["answers"]),"empty":not bool(answer.strip()),
            "answer_in_ocr":answer_in_ocr(answer,r["words"]),"latency_seconds":None,
        })
    return rows


# ---- Section 11: LayoutLM ------------------------------------------------------------------------------
def normalize_box(box,width,height):
    x0,y0,x1,y1=[float(v) for v in box]
    return [int(BOX_GRID*x0/width),int(BOX_GRID*y0/height),int(BOX_GRID*x1/width),int(BOX_GRID*y1/height)]


def layout_encode(question,words):
    return layout_tokenizer(
        text=question.split(),text_pair=words,is_split_into_words=True,max_length=MAX_SEQ_LEN,stride=DOC_STRIDE,
        truncation="only_second",return_overflowing_tokens=True,return_token_type_ids=True,
        padding="max_length",return_tensors="pt",
    )


def layout_window_boxes(encoding,window,grid_boxes):
    out=[]
    for input_id,sequence_id,word_id in zip(
        encoding["input_ids"][window].tolist(),encoding.sequence_ids(window),encoding.word_ids(window),strict=True
    ):
        if sequence_id==1: out.append(list(grid_boxes[word_id]))
        elif input_id==layout_sep_id: out.append([BOX_GRID]*4)
        else: out.append([0]*4)
    return out


def layout_answer(question,words,boxes,image_size):
    width,height=image_size
    if not words or len(words)!=len(boxes): raise ValueError("LayoutLM requires one box per OCR word")
    q=" ".join(str(question).split())
    if not q or len(q)>256: raise ValueError("Question must contain 1..256 characters")
    for box in boxes:
        x0,y0,x1,y1=[float(v) for v in box]
        if not (0<=x0<=x1<=width and 0<=y0<=y1<=height): raise ValueError("OCR box outside page")
    if globals().get("layout_model") is None or globals().get("layout_tokenizer") is None:
        raise RuntimeError("LayoutLM is not loaded: each stage loads it in its own process and frees it on exit. "
                           "Use the 'Try it yourself' activity cell, which reloads it, or choose Run all again.")
    grid_boxes=[normalize_box(b,width,height) for b in boxes]
    enc=layout_encode(q,words); n_windows=int(enc["input_ids"].shape[0])
    best={"start":None,"end":None,"score":0.0,"n_windows":n_windows}
    model_device=next(layout_model.parameters()).device
    for window in range(n_windows):
        sequence_ids=enc.sequence_ids(window); word_ids=enc.word_ids(window)
        inputs={
            "input_ids":enc["input_ids"][window].unsqueeze(0).to(model_device),
            "attention_mask":enc["attention_mask"][window].unsqueeze(0).to(model_device),
            "token_type_ids":enc["token_type_ids"][window].unsqueeze(0).to(model_device),
            "bbox":torch.tensor(layout_window_boxes(enc,window,grid_boxes),dtype=torch.long).unsqueeze(0).to(model_device),
        }
        with torch.inference_mode(): outputs=layout_model(**inputs)
        allowed=torch.tensor([sid==1 for sid in sequence_ids],device=model_device)
        start=outputs.start_logits[0].float().masked_fill(~allowed,float("-inf")).softmax(-1)
        end=outputs.end_logits[0].float().masked_fill(~allowed,float("-inf")).softmax(-1)
        candidates=start[:,None]*end[None,:]
        candidates=torch.triu(candidates)-torch.triu(candidates,diagonal=MAX_ANSWER_TOKENS)
        flat=int(candidates.argmax()); s_idx,e_idx=divmod(flat,candidates.shape[1]); score=float(candidates[s_idx,e_idx])
        if score>best["score"] and word_ids[s_idx] is not None and word_ids[e_idx] is not None:
            best={"start":int(word_ids[s_idx]),"end":int(word_ids[e_idx]),"score":score,"n_windows":n_windows}
    if best["start"] is None:
        answer=""; union_box=None
    else:
        answer=" ".join(words[best["start"]:best["end"]+1]); selected=boxes[best["start"]:best["end"]+1]
        union_box=[min(b[0] for b in selected),min(b[1] for b in selected),max(b[2] for b in selected),max(b[3] for b in selected)]
    return {**best,"answer":answer,"answer_union_box":union_box}


def load_layoutlm():
    """Load the digest-verified LayoutLM snapshot; returns the load time and parameter count."""
    global layout_tokenizer, layout_model, layout_sep_id
    load_torch()
    from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
    layout_tokenizer=AutoTokenizer.from_pretrained(str(LAYOUTLM_DIR),local_files_only=True,trust_remote_code=False)
    if not layout_tokenizer.is_fast: raise RuntimeError("LayoutLM requires a fast tokenizer")
    t0=time.perf_counter()
    layout_model=LayoutLMForQuestionAnswering.from_pretrained(
        str(LAYOUTLM_DIR),local_files_only=True,trust_remote_code=False,dtype=torch.float32
    ).to(DEVICE).eval()
    load_seconds=time.perf_counter()-t0
    for p in layout_model.parameters(): p.requires_grad_(False)
    parameter_count=sum(p.numel() for p in layout_model.parameters())
    if parameter_count!=127_792_898: raise RuntimeError(f"Unexpected LayoutLM parameter count: {parameter_count}")
    layout_sep_id=layout_tokenizer.sep_token_id
    return load_seconds,parameter_count


def release_layoutlm():
    global layout_model, layout_tokenizer
    layout_model=layout_tokenizer=None
    gc.collect()
    if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()


def rejection_probe(check,call):
    """Run one invalid input that must be refused with ValueError and record the observed message."""
    try: call()
    except ValueError as exc: return {"check":check,"refused":True,"message":str(exc)}
    raise RuntimeError(f"Input validation regression: {check} was not refused")


def layoutlm_rejection_probes():
    # Non-blocking refusal checks: each input below is invalid and is refused before the model runs.
    return [
        rejection_probe("LayoutLM: mismatched word/box counts",
                        lambda:layout_answer("What is the total?",["TOTAL","9.50"],[[0,0,10,10]],(100,100))),
        rejection_probe("LayoutLM: box outside the page",
                        lambda:layout_answer("What is the total?",["TOTAL"],[[0,0,500,10]],(100,100))),
        rejection_probe("LayoutLM: empty question",lambda:layout_answer("   ",["TOTAL"],[[0,0,10,10]],(100,100))),
    ]


def model_row(r,model,answer,dt,extra):
    return {
        "record_id":r["id"],"page_id":r["page_id"],"field":r["field"],"question":r["question"],"gold_answer":r["answers"][0],
        "model":model,"answer":answer,"normalized_answer":normalize_answer(answer),"anls":anls(answer,r["answers"]),
        "exact_match":exact_match(answer,r["answers"]),"empty":not bool(answer.strip()),"answer_in_ocr":answer_in_ocr(answer,r["words"]),
        "latency_seconds":dt,**extra,
    }


def evaluate_layoutlm(test_records):
    """The former section 12 cell: one warm-up call, then the 229 measured questions."""
    warm=test_records[0]
    _=layout_answer(warm["question"],warm["words"],warm["boxes"],warm["image_size"])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    layout_rows=[]; layout_times=[]
    for i,r in enumerate(test_records):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.perf_counter(); result=layout_answer(r["question"],r["words"],r["boxes"],r["image_size"])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt=time.perf_counter()-t0; layout_times.append(dt)
        layout_rows.append(model_row(r,"LayoutLM",result["answer"],dt,{
            "start_word":result["start"],"end_word":result["end"],"span_score":result["score"],
            "n_windows":result["n_windows"],"answer_union_box":result["answer_union_box"],"new_tokens":None,"truncated":None,
        }))
        if (i+1)%50==0: print(f"LayoutLM: {i+1}/{len(test_records)}")
    layout_peak_gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    return layout_rows,layout_times,layout_peak_gpu


# ---- Section 13: experiment helpers ------------------------------------------------------------------------
def subset_coverage(records):
    """Size and coverage of a predetermined experiment subset, printed beside its means."""
    return {"records":len(records),"pages":len({r["page_id"] for r in records}),"fields":len({r["field"] for r in records})}


def print_field_deltas(rows,label):
    """Mean ANLS per question field, canonical vs variant (few records per field: read as anecdotes)."""
    by_field=defaultdict(list)
    for row in rows: by_field[row["field"]].append(row)
    print(f"{'Field':<28} {'N':>3} {'canonical':>10} {label[:10]:>10} {'change':>8}")
    for field in QUESTION_TEMPLATES:
        group=by_field.get(field)
        if not group: continue
        c=float(np.mean([r["canonical_anls"] for r in group])); v=float(np.mean([r["variant_anls"] for r in group]))
        print(f"{field:<28} {len(group):>3} {c:>10.3f} {v:>10.3f} {v-c:>+8.3f}")


def experiment_records(test_records):
    return sorted(test_records,key=lambda r:r["id"])[:ROBUSTNESS_MAX_RECORDS]


def paraphrase_records(test_records):
    return sorted(test_records,key=lambda r:r["id"])[:PARAPHRASE_MAX_RECORDS]


def probe_pages(test_pages):
    return sorted(test_pages)[:UNANSWERABLE_PAGES]


def layout_no_layout(test_records,layout_rows):
    """The former section 13 cell: the same OCR text with every box set to zero."""
    records=experiment_records(test_records)
    layout_no_layout_rows=[]
    if RUN_MODALITY_ROBUSTNESS:
        for r in records:
            zero_boxes=[[0.0,0.0,0.0,0.0] for _ in r["boxes"]]
            result=layout_answer(r["question"],r["words"],zero_boxes,r["image_size"])
            canonical=next(x for x in layout_rows if x["record_id"]==r["id"])
            layout_no_layout_rows.append({
                "model":"LayoutLM","record_id":r["id"],"field":r["field"],"experiment":"no_layout",
                "canonical_answer":canonical["answer"],"variant_answer":result["answer"],
                "canonical_anls":canonical["anls"],"variant_anls":anls(result["answer"],r["answers"]),
                "canonical_exact":canonical["exact_match"],"variant_exact":exact_match(result["answer"],r["answers"]),
            })
        for row in layout_no_layout_rows: row["delta_anls"]=row["variant_anls"]-row["canonical_anls"]
        print({"subset":subset_coverage(records),
               "canonical_mean_anls":float(np.mean([r["canonical_anls"] for r in layout_no_layout_rows])),
               "no_layout_mean_anls":float(np.mean([r["variant_anls"] for r in layout_no_layout_rows]))})
        print_field_deltas(layout_no_layout_rows,"no layout")
    else:
        print("Layout robustness experiment disabled.")
    return layout_no_layout_rows


def layout_probes(test_records,test_pages,layout_rows):
    """The former section 14 cell: paraphrased questions and the unanswerable probe for LayoutLM."""
    layout_paraphrase_rows=[]
    if RUN_PARAPHRASE_EXPERIMENT:
        for r in paraphrase_records(test_records):
            variant_q=PARAPHRASES[r["field"]]; result=layout_answer(variant_q,r["words"],r["boxes"],r["image_size"])
            canonical=next(x for x in layout_rows if x["record_id"]==r["id"])
            layout_paraphrase_rows.append({
                "model":"LayoutLM","record_id":r["id"],"field":r["field"],"canonical_question":r["question"],
                "paraphrased_question":variant_q,"canonical_answer":canonical["answer"],"paraphrased_answer":result["answer"],
                "canonical_anls":canonical["anls"],"paraphrased_anls":anls(result["answer"],r["answers"]),
                "answer_stable":normalize_answer(canonical["answer"])==normalize_answer(result["answer"]),
            })
    layout_unanswerable_rows=[]
    if RUN_UNANSWERABLE_PROBE:
        for page_id in probe_pages(test_pages):
            page=test_pages[page_id]; result=layout_answer(UNANSWERABLE_QUESTION,page["words"],page["boxes"],page["image_size"])
            layout_unanswerable_rows.append({
                "model":"LayoutLM","page_id":page_id,"question":UNANSWERABLE_QUESTION,"answer":result["answer"],
                "empty":not bool(result["answer"].strip()),"answer_in_ocr":answer_in_ocr(result["answer"],page["words"]),
                "span_score":result["score"],"new_tokens":None,"truncated":None,
            })
    print("Paraphrase rows:",len(layout_paraphrase_rows))
    print("Unanswerable rows:",len(layout_unanswerable_rows))
    return layout_paraphrase_rows,layout_unanswerable_rows


# ---- Section 16: Pix2Struct ----------------------------------------------------------------------------
def pix_answer(image,question,max_new_tokens=None):
    if max_new_tokens is None: max_new_tokens=MAX_NEW_TOKENS  # read at call time, so a changed setting is never stale
    q=" ".join(str(question).split())
    if not q or len(q)>256: raise ValueError("Question must contain 1..256 characters")
    if min(image.size)<16 or max(image.size)>4096: raise ValueError("Pix2Struct image side outside 16..4096")
    if isinstance(max_new_tokens,bool) or not isinstance(max_new_tokens,int) or not 1<=max_new_tokens<=128:
        raise ValueError("max_new_tokens must be an int in 1..128")
    if globals().get("pix_model") is None or globals().get("pix_processor") is None:
        raise RuntimeError("Pix2Struct is not loaded: each stage loads it in its own process and frees it on exit. "
                           "Choose Run all again, or use BYOD, which reloads it.")
    inputs=pix_processor.image_processor(image.convert("RGB"),header_text=q,return_tensors="pt",font_bytes=FONT_BYTES).to(DEVICE)
    with torch.inference_mode():
        generated=pix_model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    answer=pix_processor.tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip()
    new_tokens=int(generated[0].shape[0])-1
    return {"answer":answer,"new_tokens":new_tokens,"truncated":new_tokens>=max_new_tokens}


def header_font_bytes():
    """Pillow's bundled Aileron font, supplied so the processor never downloads an unpinned Arial font."""
    font_obj=ImageFont.load_default(size=36); font_bytes=getattr(font_obj,"font_bytes",None)
    if not font_bytes: raise RuntimeError("Pillow bundled TrueType font unavailable")
    return bytes(font_bytes)


def load_pix2struct():
    """Load the digest-verified Pix2Struct snapshot; returns the load time and parameter count."""
    global pix_processor, pix_model, FONT_BYTES
    load_torch()
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    pix_processor=Pix2StructProcessor.from_pretrained(str(PIX2STRUCT_DIR),local_files_only=True,trust_remote_code=False)
    if not getattr(pix_processor.image_processor,"is_vqa",False): raise RuntimeError("Pix2Struct processor is not VQA mode")
    FONT_BYTES=header_font_bytes()
    t0=time.perf_counter()
    pix_model=Pix2StructForConditionalGeneration.from_pretrained(
        str(PIX2STRUCT_DIR),local_files_only=True,trust_remote_code=False,dtype=torch.float32
    ).to(DEVICE).eval()
    load_seconds=time.perf_counter()-t0
    for p in pix_model.parameters(): p.requires_grad_(False)
    return load_seconds,sum(p.numel() for p in pix_model.parameters())


def release_pix2struct():
    global pix_model, pix_processor
    pix_model=pix_processor=None
    gc.collect()
    if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()


def pix2struct_rejection_probes():
    _blank_page=Image.new("RGB",(64,64),"white")
    return [
        rejection_probe("Pix2Struct: empty question",lambda:pix_answer(_blank_page,"   ")),
        rejection_probe("Pix2Struct: image side above 4096",
                        lambda:pix_answer(Image.new("RGB",(4100,16),"white"),"What is the total?")),
        rejection_probe("Pix2Struct: generation budget outside 1..128",
                        lambda:pix_answer(_blank_page,"What is the total?",max_new_tokens=0)),
    ]


def evaluate_pix2struct(test_records,test_pages):
    """The former section 17 cell: one warm-up call, then the 229 measured questions."""
    warm=test_records[0]
    _=pix_answer(test_pages[warm["page_id"]]["image"],warm["question"])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    pix_rows=[]; pix_times=[]
    for i,r in enumerate(test_records):
        page=test_pages[r["page_id"]]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.perf_counter(); result=pix_answer(page["image"],r["question"])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt=time.perf_counter()-t0; pix_times.append(dt)
        pix_rows.append(model_row(r,"Pix2Struct",result["answer"],dt,{
            "start_word":None,"end_word":None,"span_score":None,"n_windows":None,"answer_union_box":None,
            "new_tokens":result["new_tokens"],"truncated":result["truncated"],
        }))
        if (i+1)%25==0: print(f"Pix2Struct: {i+1}/{len(test_records)}")
    pix_peak_gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    return pix_rows,pix_times,pix_peak_gpu


def pix_degraded(test_records,test_pages,pix_rows):
    """The former section 18 cell: the same subset with each page halved in resolution and resized back."""
    records=experiment_records(test_records)
    pix_degraded_rows=[]
    if RUN_MODALITY_ROBUSTNESS:
        for r in records:
            page=test_pages[r["page_id"]]; image=page["image"]
            low=image.resize((max(16,image.width//2),max(16,image.height//2)),Image.Resampling.BILINEAR)
            degraded=low.resize(image.size,Image.Resampling.BILINEAR)
            result=pix_answer(degraded,r["question"]); canonical=next(x for x in pix_rows if x["record_id"]==r["id"])
            pix_degraded_rows.append({
                "model":"Pix2Struct","record_id":r["id"],"field":r["field"],"experiment":"low_resolution",
                "canonical_answer":canonical["answer"],"variant_answer":result["answer"],
                "canonical_anls":canonical["anls"],"variant_anls":anls(result["answer"],r["answers"]),
                "canonical_exact":canonical["exact_match"],"variant_exact":exact_match(result["answer"],r["answers"]),
            })
        for row in pix_degraded_rows: row["delta_anls"]=row["variant_anls"]-row["canonical_anls"]
        print({"subset":subset_coverage(records),
               "canonical_mean_anls":float(np.mean([r["canonical_anls"] for r in pix_degraded_rows])),
               "degraded_mean_anls":float(np.mean([r["variant_anls"] for r in pix_degraded_rows]))})
        print_field_deltas(pix_degraded_rows,"degraded")
    else:
        print("Pix2Struct robustness experiment disabled.")
    return pix_degraded_rows


def pix_probes(test_records,test_pages,pix_rows):
    """The former section 19 cell: the same paraphrase and unanswerable subsets for Pix2Struct."""
    pix_paraphrase_rows=[]
    if RUN_PARAPHRASE_EXPERIMENT:
        for r in paraphrase_records(test_records):
            variant_q=PARAPHRASES[r["field"]]; result=pix_answer(test_pages[r["page_id"]]["image"],variant_q)
            canonical=next(x for x in pix_rows if x["record_id"]==r["id"])
            pix_paraphrase_rows.append({
                "model":"Pix2Struct","record_id":r["id"],"field":r["field"],"canonical_question":r["question"],
                "paraphrased_question":variant_q,"canonical_answer":canonical["answer"],"paraphrased_answer":result["answer"],
                "canonical_anls":canonical["anls"],"paraphrased_anls":anls(result["answer"],r["answers"]),
                "answer_stable":normalize_answer(canonical["answer"])==normalize_answer(result["answer"]),
            })
    pix_unanswerable_rows=[]
    if RUN_UNANSWERABLE_PROBE:
        for page_id in probe_pages(test_pages):
            page=test_pages[page_id]; result=pix_answer(page["image"],UNANSWERABLE_QUESTION)
            pix_unanswerable_rows.append({
                "model":"Pix2Struct","page_id":page_id,"question":UNANSWERABLE_QUESTION,"answer":result["answer"],
                "empty":not bool(result["answer"].strip()),"answer_in_ocr":answer_in_ocr(result["answer"],page["words"]),
                "span_score":None,"new_tokens":result["new_tokens"],"truncated":result["truncated"],
            })
    print("Paraphrase rows:",len(pix_paraphrase_rows))
    print("Unanswerable rows:",len(pix_unanswerable_rows))
    return pix_paraphrase_rows,pix_unanswerable_rows


# ---- Sections 20-27: comparison ------------------------------------------------------------------------
def load_results():
    """Every row the model stages wrote, under the names the former cells used."""
    base=load_state("baselines.json"); lev=load_state("layoutlm_eval.json"); pev=load_state("pix2struct_eval.json")
    res={
        "last_number_rows":base["last_number_rows"],"keyword_rows":base["keyword_rows"],
        "layout_rows":lev["rows"],"layout_times":lev["times"],"layout_peak_gpu":lev["peak_gpu_memory_bytes"],
        "pix_rows":pev["rows"],"pix_times":pev["times"],"pix_peak_gpu":pev["peak_gpu_memory_bytes"],
    }
    res["layout_no_layout_rows"]=load_state("layoutlm_no_layout.json")["rows"]
    res["pix_degraded_rows"]=load_state("pix2struct_degraded.json")["rows"]
    lp=load_state("layoutlm_probes.json"); pp=load_state("pix2struct_probes.json")
    res["layout_paraphrase_rows"]=lp["paraphrase_rows"]; res["layout_unanswerable_rows"]=lp["unanswerable_rows"]
    res["pix_paraphrase_rows"]=pp["paraphrase_rows"]; res["pix_unanswerable_rows"]=pp["unanswerable_rows"]
    return res


def system_table(res):
    return [("Last-number baseline",res["last_number_rows"]),("Keyword lookup",res["keyword_rows"]),
            ("LayoutLM",res["layout_rows"]),("Pix2Struct",res["pix_rows"])]


def field_metrics(rows):
    by_field=defaultdict(list)
    for row in rows: by_field[row["field"]].append(row)
    result=[]
    for field in QUESTION_TEMPLATES:
        group=by_field.get(field,[])
        if not group: continue
        result.append({
            "field":field,"support":len(group),"mean_anls":float(np.mean([r["anls"] for r in group])),
            "exact_match":float(np.mean([r["exact_match"] for r in group])),"empty_rate":float(np.mean([r["empty"] for r in group])),
            "answer_in_ocr_rate":float(np.mean([r["answer_in_ocr"] for r in group])),
        })
    return result


def agreement(test_records,layout_rows,pix_rows):
    """The former section 22 cell: exact-match agreement category per question."""
    layout_by_id={r["record_id"]:r for r in layout_rows}; pix_by_id={r["record_id"]:r for r in pix_rows}
    agreement_rows=[]; agreement_counts=defaultdict(int)
    for r in test_records:
        l,p=layout_by_id[r["id"]],pix_by_id[r["id"]]
        if l["exact_match"] and p["exact_match"]: cat="both_exact"
        elif l["exact_match"]: cat="layoutlm_only"
        elif p["exact_match"]: cat="pix2struct_only"
        else: cat="neither_exact"
        agreement_counts[cat]+=1
        agreement_rows.append({
            "record_id":r["id"],"page_id":r["page_id"],"field":r["field"],"question":r["question"],"gold_answer":r["answers"][0],
            "layoutlm_answer":l["answer"],"layoutlm_anls":l["anls"],"layoutlm_exact":l["exact_match"],
            "pix2struct_answer":p["answer"],"pix2struct_anls":p["anls"],"pix2struct_exact":p["exact_match"],
            "agreement_category":cat,
        })
    return agreement_rows,dict(agreement_counts)


def print_off_page(pix_rows):
    off_page=sorted((r for r in pix_rows if r["answer"].strip() and not r["answer_in_ocr"]),key=lambda r:r["record_id"])
    print(f"Pix2Struct answers not found as a contiguous OCR span: {len(off_page)} of {len(pix_rows)}" + (" (first 15 shown)" if len(off_page)>15 else ""))
    print(f"{'Record':<10} {'Field':<26} {'Gold':<20} {'Pix2Struct':<20} {'ANLS':>5}")
    for r in off_page[:15]:
        print(f"{r['record_id']:<10} {r['field']:<26} {r['gold_answer'][:20]:<20} {r['answer'][:20]:<20} {r['anls']:>5.2f}")
    return off_page


def window_summary(layout_rows):
    window_counts=np.array([r["n_windows"] for r in layout_rows],dtype=int)
    return {"one_window":int(np.sum(window_counts==1)),"multi_window":int(np.sum(window_counts>1)),
            "mean":float(window_counts.mean()),"max":int(window_counts.max())}


def generation_summary(pix_rows):
    new_tokens=np.array([r["new_tokens"] for r in pix_rows],dtype=int)
    return {"mean_new_tokens":float(new_tokens.mean()),"max_new_tokens":int(new_tokens.max()),
            "truncation_rate":float(np.mean([r["truncated"] for r in pix_rows]))}


def p95(values): return float(np.percentile(np.asarray(values,dtype=float),95))


def resource_rows(res):
    lc=load_state("layoutlm_check.json"); pc=load_state("pix2struct_check.json")
    layout_times,pix_times=res["layout_times"],res["pix_times"]
    return [
        {"model":"LayoutLM","model_id":LAYOUTLM_ID,"revision":LAYOUTLM_REVISION,"parameter_count":lc["parameter_count"],
         "weight_bytes":next(x["bytes"] for x in LAYOUTLM_MANIFEST["files"] if x["path"]=="model.safetensors"),
         "load_seconds":lc["load_seconds"],"mean_latency_s":float(np.mean(layout_times)),"median_latency_s":float(np.median(layout_times)),
         "p95_latency_s":p95(layout_times),"total_eval_seconds":float(sum(layout_times)),"peak_gpu_memory_bytes":res["layout_peak_gpu"]},
        {"model":"Pix2Struct","model_id":PIX_ID,"revision":PIX_REVISION,"parameter_count":pc["parameter_count"],
         "weight_bytes":next(x["bytes"] for x in PIX2STRUCT_MANIFEST["files"] if x["path"]=="model.safetensors"),
         "load_seconds":pc["load_seconds"],"mean_latency_s":float(np.mean(pix_times)),"median_latency_s":float(np.median(pix_times)),
         "p95_latency_s":p95(pix_times),"total_eval_seconds":float(sum(pix_times)),"peak_gpu_memory_bytes":res["pix_peak_gpu"]},
    ]


def select_examples(agreement_rows,layout_rows,pix_rows):
    """Deterministic panel examples: the four agreement categories plus the two diagnostic cases."""
    def select_example(category):
        rows=[r for r in agreement_rows if r["agreement_category"]==category]
        return sorted(rows,key=lambda x:x["record_id"])[0] if rows else None

    def relabel(record_id,category):
        row=next((r for r in agreement_rows if r["record_id"]==record_id),None)
        return None if row is None else {**row,"agreement_category":category}

    off_page_first=next((r for r in sorted(pix_rows,key=lambda r:r["record_id"]) if r["answer"].strip() and not r["answer_in_ocr"]),None)
    confident_wrong=sorted((r for r in layout_rows if not r["exact_match"]),key=lambda r:(-r["span_score"],r["record_id"]))
    return [x for x in [
        select_example("both_exact"),select_example("layoutlm_only"),select_example("pix2struct_only"),select_example("neither_exact"),
        relabel(off_page_first["record_id"],"pix2struct_not_in_ocr") if off_page_first else None,
        relabel(confident_wrong[0]["record_id"],"layoutlm_confident_wrong") if confident_wrong else None,
    ] if x]


def draw_qa_panel(example,test_pages,layout_by_id,pix_by_id):
    page=test_pages[example["page_id"]]; image=page["image"].convert("RGB").copy(); draw=ImageDraw.Draw(image)
    for box in page["boxes"]: draw.rectangle(box,outline=(180,180,180),width=1)
    l=layout_by_id[example["record_id"]]; p=pix_by_id[example["record_id"]]
    if l["answer_union_box"] is not None: draw.rectangle(l["answer_union_box"],outline=(0,0,0),width=4)
    max_w=900
    if image.width>max_w:
        scale=max_w/image.width; image=image.resize((max_w,max(1,int(image.height*scale))))
    canvas=Image.new("RGB",(image.width,image.height+218),"white"); canvas.paste(image,(0,0)); d=ImageDraw.Draw(canvas); y=image.height+6
    lines=[f"[{example['agreement_category']}] Q: {example['question']}",f"Gold: {example['gold_answer']}",
           f"LayoutLM: {example['layoutlm_answer']}  ANLS={example['layoutlm_anls']:.3f}  span score={l['span_score']:.3f} (uncalibrated)",
           f"Pix2Struct: {example['pix2struct_answer']}  ANLS={example['pix2struct_anls']:.3f}  truncated={p['truncated']}",
           "Black box = LayoutLM's answer words. Pix2Struct gives no location."]
    for line in lines: d.text((8,y),line,fill="black"); y+=38
    return canvas


def draw_preview(test_records,test_pages,path):
    """The section 8 preview: the first test receipt with its OCR word boxes outlined."""
    preview_record=sorted(test_records,key=lambda r:r["id"])[0]
    preview_page=test_pages[preview_record["page_id"]]
    preview=preview_page["image"].convert("RGB").copy(); preview_draw=ImageDraw.Draw(preview)
    for box in preview_page["boxes"]: preview_draw.rectangle(box,outline=(150,150,150),width=1)
    if preview.width>700: preview=preview.resize((700,max(1,int(preview.height*700/preview.width))))
    preview.save(path)
    print(f"Page {preview_page['id']}: {len(preview_page['words'])} OCR words with one box each")
    print("Question:",preview_record["question"])
    print("First 25 OCR words:",preview_page["words"][:25])
    return preview


# ---- Section 29: exports ---------------------------------------------------------------------------------
def write_csv(path,rows,fieldnames):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for row in rows:
            out={}
            for k in fieldnames:
                value=row.get(k)
                if isinstance(value,(list,dict)): value=json.dumps(value)
                out[k]=value
            w.writerow(out)


PREDICTION_COLUMNS=[
    "record_id","page_id","field","question","gold_answer","model","answer","normalized_answer","anls","exact_match",
    "empty","answer_in_ocr","latency_seconds","start_word","end_word","span_score","n_windows","answer_union_box","new_tokens","truncated"
]
REQUIRED_OUTPUTS=["predictions.csv","field_metrics.csv","agreement.csv","paraphrase_experiment.csv","unanswerable_probe.csv",
                  "modality_robustness.csv","resource_metrics.csv","input_manifest.json","metrics.json","provenance.json"]


def export_all(out_dir):
    data=load_state("dataset.json"); test_records=data["test_records"]; test_pages=data["test_pages"]
    res=load_results(); runtime=load_state("environment.json")
    lc=load_state("layoutlm_check.json"); pc=load_state("pix2struct_check.json")
    layout_rows,pix_rows=res["layout_rows"],res["pix_rows"]
    system_metrics={name:corpus_metrics(rows) for name,rows in system_table(res)}
    layout_field_metrics=field_metrics(layout_rows); pix_field_metrics=field_metrics(pix_rows)
    agreement_rows,agreement_counts=agreement(test_records,layout_rows,pix_rows)
    paraphrase_all=res["layout_paraphrase_rows"]+res["pix_paraphrase_rows"]
    unanswerable_rows=res["layout_unanswerable_rows"]+res["pix_unanswerable_rows"]
    robustness_rows=res["layout_no_layout_rows"]+res["pix_degraded_rows"]
    resources=resource_rows(res)
    out_dir.mkdir(parents=True,exist_ok=True)

    prediction_rows=res["last_number_rows"]+res["keyword_rows"]+layout_rows+pix_rows
    write_csv(out_dir/"predictions.csv",prediction_rows,PREDICTION_COLUMNS)
    field_rows=[]
    for model_name,rows in [("LayoutLM",layout_field_metrics),("Pix2Struct",pix_field_metrics)]:
        for row in rows: field_rows.append({"model":model_name,**row})
    write_csv(out_dir/"field_metrics.csv",field_rows,["model","field","support","mean_anls","exact_match","empty_rate","answer_in_ocr_rate"])
    write_csv(out_dir/"agreement.csv",agreement_rows,[
        "record_id","page_id","field","question","gold_answer","layoutlm_answer","layoutlm_anls","layoutlm_exact",
        "pix2struct_answer","pix2struct_anls","pix2struct_exact","agreement_category"
    ])
    write_csv(out_dir/"paraphrase_experiment.csv",paraphrase_all,[
        "model","record_id","field","canonical_question","paraphrased_question","canonical_answer","paraphrased_answer",
        "canonical_anls","paraphrased_anls","answer_stable"
    ])
    write_csv(out_dir/"unanswerable_probe.csv",unanswerable_rows,["model","page_id","question","answer","empty","answer_in_ocr","span_score","new_tokens","truncated"])
    write_csv(out_dir/"modality_robustness.csv",robustness_rows,[
        "model","record_id","field","experiment","canonical_answer","variant_answer","canonical_anls","variant_anls","delta_anls","canonical_exact","variant_exact"
    ])
    write_csv(out_dir/"resource_metrics.csv",resources,[
        "model","model_id","revision","parameter_count","weight_bytes","load_seconds","mean_latency_s","median_latency_s","p95_latency_s",
        "total_eval_seconds","peak_gpu_memory_bytes"
    ])

    input_manifest={
        "dataset":{"repo":CORD_REPO,"revision":CORD_REVISION,"test_pages":len(test_pages),"test_qa_records":len(test_records),"test_qa_digest":data["test_qa_digest"]},
        "observed":{
            "image_width_range":[min(p["image_size"][0] for p in test_pages),max(p["image_size"][0] for p in test_pages)],
            "image_height_range":[min(p["image_size"][1] for p in test_pages),max(p["image_size"][1] for p in test_pages)],
            "ocr_word_count_range":[min(len(r["words"]) for r in test_records),max(len(r["words"]) for r in test_records)],
            "question_chars_range":[min(len(r["question"]) for r in test_records),max(len(r["question"]) for r in test_records)],
        },
        "verdict":"accepted",
        "rejection_probes":lc["rejection_probes"]+pc["rejection_probes"],
    }
    (out_dir/"input_manifest.json").write_text(json.dumps(input_manifest,indent=2),encoding="utf-8")
    metrics_export={
        "systems":system_metrics,"layoutlm_per_field":layout_field_metrics,"pix2struct_per_field":pix_field_metrics,
        "agreement_counts":agreement_counts,"paraphrase":{"layoutlm":res["layout_paraphrase_rows"],"pix2struct":res["pix_paraphrase_rows"]},
        "unanswerable":unanswerable_rows,"modality_robustness":robustness_rows,
        "layoutlm_window_summary":window_summary(layout_rows),
        "pix2struct_generation_summary":generation_summary(pix_rows),
    }
    (out_dir/"metrics.json").write_text(json.dumps(metrics_export,indent=2),encoding="utf-8")
    provenance={
        "created_utc":datetime.now(timezone.utc).isoformat(),"notebook_spec":"2.1","profile":"TASK-INFERENCE","pedagogical_mode":"WORKSHOP","comparison_scope":"MULTI-MODEL",
        "dataset":{"repo":CORD_REPO,"revision":CORD_REVISION,"license":CORD_LICENSE,"shards":CORD_FILES,"split_seed":SAMPLE_SEED,"page_split":SAMPLE_PAGE_SPLIT,
                   "qa_counts":data["qa_splits"],"test_qa_digest":data["test_qa_digest"],"test_pages":[
                       {"page_id":p["id"],"source_split":p["source_split"],"source_row":p["source_row"],"image_sha256":p["pixel_sha256"],"ocr_sha256":p["ocr_sha256"]}
                       for p in test_pages]},
        "layoutlm":{"model_id":LAYOUTLM_ID,"revision":LAYOUTLM_REVISION,"weight_sha256":next(x["sha256"] for x in LAYOUTLM_MANIFEST["files"] if x["path"]=="model.safetensors"),
                    "parameter_count":lc["parameter_count"],"input_boundary":"question + OCR words + pixel boxes; page pixels not read"},
        "pix2struct":{"model_id":PIX_ID,"revision":PIX_REVISION,"weight_sha256":next(x["sha256"] for x in PIX2STRUCT_MANIFEST["files"] if x["path"]=="model.safetensors"),
                      "parameter_count":pc["parameter_count"],"header_font_sha256":pc["font_sha256"],"max_new_tokens":MAX_NEW_TOKENS,
                      "input_boundary":"page pixels + question rendered as header; no external OCR input"},
        "question_templates":QUESTION_TEMPLATES,"anls_threshold":ANLS_THRESHOLD,"runtime":runtime,"resource_metrics":resources,
    }
    (out_dir/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Exports:")
    for p in sorted(out_dir.iterdir()): print(" ",p)


# ---- Section 30: Bring Your Own Document QA data ---------------------------------------------------------
def byod_root(path_text):
    """The dataset folder: BYOD_PATH itself, or a .zip extracted with every member kept inside the target folder."""
    text=str(path_text).strip()
    if not text: raise ValueError("BYOD_PATH is empty: set it in section 3 to your dataset folder or .zip (see section 30)")
    path=Path(text)
    if path.is_file() and path.suffix.lower()==".zip":
        target=Path(OUTPUT_DIR).parent/"byod_input"
        if target.exists(): shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            members=[m for m in archive.infolist() if not m.is_dir()]
            if len(members)>BYOD_LIMITS["zip_files"]:
                raise ValueError(f"{path.name}: {len(members)} files; the ZIP may hold at most {BYOD_LIMITS['zip_files']}")
            if sum(m.file_size for m in members)>BYOD_LIMITS["zip_bytes"]:
                raise ValueError(f"{path.name}: more than 2 GB uncompressed; split the dataset")
            root=target.resolve()
            for m in members:
                if root not in (target/m.filename).resolve().parents:
                    raise ValueError(f"{path.name}: member {m.filename!r} would extract outside the dataset folder; "
                                     "rebuild the ZIP with relative paths")
            archive.extractall(target)
        inner=[p for p in target.iterdir() if p.is_dir() and (p/"records.jsonl").is_file()]
        path=target if (target/"records.jsonl").is_file() or len(inner)!=1 else inner[0]
    if not (path/"records.jsonl").is_file() or not (path/"pages").is_dir():
        raise FileNotFoundError(f"{path}: expected records.jsonl and a pages/ folder (section 30 shows the layout)")
    return path


def validate_byod(root):
    """Check every record against the section 30 contract before any model loads; refusals name the record id."""
    lines=[line for line in (root/"records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lo,hi=BYOD_LIMITS["records"]
    if not lo<=len(lines)<=hi: raise ValueError(f"records.jsonl has {len(lines)} records; BYOD accepts {lo}..{hi}")
    records=[]; seen=set(); pages={}
    for n,line in enumerate(lines,1):
        try: r=json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"records.jsonl line {n}: not valid JSON ({exc.msg})") from None
        if not isinstance(r,dict): raise ValueError(f"records.jsonl line {n}: each line must be one JSON object")
        rid=str(r.get("id","")).strip() or f"line {n}"
        where=f"BYOD record {rid!r} (records.jsonl line {n})"
        missing=sorted({"id","page_id","file","question","words","boxes"}-set(r))
        if missing: raise ValueError(f"{where}: missing required keys {missing}")
        if rid in seen: raise ValueError(f"{where}: duplicate id; every record needs a unique id")
        seen.add(rid)
        question=" ".join(str(r["question"]).split())
        if not question or len(question)>BYOD_LIMITS["question_chars"]:
            raise ValueError(f"{where}: question must be 1..{BYOD_LIMITS['question_chars']} characters")
        file=Path(str(r["file"])).name
        if file not in pages:
            image_path=root/"pages"/file
            if not image_path.is_file():
                raise ValueError(f"{where}: page file pages/{file} not found; check 'file' and the pages/ folder")
            try:
                with Image.open(image_path) as im:
                    im.load(); image=im.convert("RGB")
            except Exception as exc:
                raise ValueError(f"{where}: pages/{file} is not a readable image ({type(exc).__name__})") from None
            slo,shi=BYOD_LIMITS["image_side"]
            if min(image.size)<slo or max(image.size)>shi:
                raise ValueError(f"{where}: pages/{file} is {image.width}x{image.height} px; each side must be "
                                 f"{slo}..{shi} px (resize the page and scale its boxes)")
            pages[file]=image
        image=pages[file]
        words,boxes=r["words"],r["boxes"]
        wlo,whi=BYOD_LIMITS["words"]
        if not isinstance(words,list) or not wlo<=len(words)<=whi:
            raise ValueError(f"{where}: 'words' must be a list of {wlo}..{whi} OCR words")
        if any(not isinstance(w,str) or not w.strip() for w in words):
            raise ValueError(f"{where}: every OCR word must be a non-empty string; drop empty words and their boxes")
        if not isinstance(boxes,list) or len(boxes)!=len(words):
            raise ValueError(f"{where}: {len(words)} words need exactly {len(words)} boxes, one per word")
        for k,box in enumerate(boxes):
            if not (isinstance(box,(list,tuple)) and len(box)==4 and all(
                    isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)):
                raise ValueError(f"{where}: box {k} must be four numbers [x0, y0, x1, y1] in pixels")
            x0,y0,x1,y1=[float(v) for v in box]
            if not (0<=x0<=x1<=image.width and 0<=y0<=y1<=image.height):
                raise ValueError(f"{where}: box {k} {list(box)} lies outside the {image.width}x{image.height} page pages/{file}")
        answers=r.get("answers")
        if answers is not None and (not isinstance(answers,list) or not answers
                                    or not all(isinstance(a,str) and a.strip() for a in answers)):
            raise ValueError(f"{where}: 'answers' must be a non-empty list of strings, or omitted")
        records.append({**r,"id":rid,"question":question,"file":file,"image":image,
                        "image_size":[image.width,image.height],"answers":answers})
    plo,phi=BYOD_LIMITS["pages"]
    if not plo<=len(pages)<=phi: raise ValueError(f"the dataset uses {len(pages)} page images; BYOD accepts {plo}..{phi}")
    labelled=[r["answers"] is not None for r in records]
    if any(labelled) and not all(labelled):
        without=[r["id"] for r in records if r["answers"] is None][:5]
        raise ValueError(f"{sum(labelled)} of {len(records)} records have 'answers' and the rest do not "
                         f"(e.g. {without}); give answers on every record, or remove them from every record")
    return records,all(labelled)


def run_byod():
    """The former section 30 cell body: validate everything, then LayoutLM, then Pix2Struct, then exports."""
    byod_dir=byod_root(BYOD_PATH)
    byod_records,byod_labelled=validate_byod(byod_dir)
    byod_pages=len({r["file"] for r in byod_records})
    print(f"BYOD accepted: {len(byod_records)} records over {byod_pages} pages; "
          f"answers {'given' if byod_labelled else 'absent (evaluation not-measurable)'}")
    load_layoutlm()
    layout_byod=[layout_answer(r["question"],r["words"],r["boxes"],r["image_size"]) for r in byod_records]
    release_layoutlm()
    load_pix2struct()
    pix_byod=[pix_answer(r["image"],r["question"]) for r in byod_records]
    release_pix2struct()
    byod_rows=[]
    for r,l,p in zip(byod_records,layout_byod,pix_byod,strict=True):
        for model,res in (("LayoutLM",l),("Pix2Struct",p)):
            a=res["answer"]
            byod_rows.append({
                "record_id":r["id"],"page_id":r["page_id"],"file":r["file"],"question":r["question"],
                "gold_answer":r["answers"][0] if byod_labelled else None,"model":model,"answer":a,
                "normalized_answer":normalize_answer(a),
                "anls":anls(a,r["answers"]) if byod_labelled else None,
                "exact_match":exact_match(a,r["answers"]) if byod_labelled else None,
                "empty":not a.strip(),"answer_in_ocr":answer_in_ocr(a,r["words"]),
                "start_word":res.get("start"),"end_word":res.get("end"),"span_score":res.get("score"),
                "n_windows":res.get("n_windows"),"answer_union_box":res.get("answer_union_box"),
                "new_tokens":res.get("new_tokens"),"truncated":res.get("truncated"),
            })
    verdict="measured" if byod_labelled else "not-measurable"
    byod_metrics={"evaluation_verdict":verdict,"n_records":len(byod_records),"n_pages":byod_pages}
    for model in ("LayoutLM","Pix2Struct"):
        rows=[x for x in byod_rows if x["model"]==model]
        m={"empty_rate":float(np.mean([x["empty"] for x in rows])),
           "answer_in_ocr_rate":float(np.mean([x["answer_in_ocr"] for x in rows]))}
        if byod_labelled:
            m["anls"]=float(np.mean([x["anls"] for x in rows])); m["exact_match"]=float(np.mean([x["exact_match"] for x in rows]))
        byod_metrics[model]=m
    byod_out=Path(OUTPUT_DIR)/"byod"; byod_out.mkdir(parents=True,exist_ok=True)
    write_csv(byod_out/"predictions.csv",byod_rows,[
        "record_id","page_id","file","question","gold_answer","model","answer","normalized_answer","anls","exact_match",
        "empty","answer_in_ocr","start_word","end_word","span_score","n_windows","answer_union_box","new_tokens","truncated",
    ])
    (byod_out/"metrics.json").write_text(json.dumps(byod_metrics,indent=2),encoding="utf-8")
    header=f"{'Record':<12} {'Question':<32} {'LayoutLM':<20} {'Pix2Struct':<20}"
    print(header+(f" {'Gold':<16}" if byod_labelled else ""))
    for r,l,p in zip(byod_records,layout_byod,pix_byod,strict=True):
        line=f"{r['id'][:12]:<12} {r['question'][:32]:<32} {l['answer'][:20]:<20} {p['answer'][:20]:<20}"
        print(line+(f" {r['answers'][0][:16]:<16}" if byod_labelled else ""))
    print(json.dumps(byod_metrics,indent=1))
    print("Wrote",byod_out/"predictions.csv","and",byod_out/"metrics.json")
    return {"metrics":byod_metrics,"rows":byod_rows}


# ---- Try it yourself: layout activity ----------------------------------------------------------------------
ACTIVITY_CHANGES=("zero","shuffle","coarse")


def changed_boxes(record,change):
    width,height=record["image_size"]; boxes=record["boxes"]
    if change=="zero": return [[0.0,0.0,0.0,0.0] for _ in boxes]
    if change=="shuffle":
        order=list(range(len(boxes))); random.Random(SAMPLE_SEED).shuffle(order)
        return [boxes[i] for i in order]
    cw,ch=width/4,height/8  # coarse: snap each box to the 4 x 8 page cell holding its top-left corner
    return [[cw*int(x0//cw),ch*int(y0//ch),min(width,cw*(int(x0//cw)+1)),min(height,ch*(int(y0//ch)+1))]
            for x0,y0,_x1,_y1 in boxes]


def run_activity(test_records,layout_rows,change,max_records):
    """The former activity cell body: reload LayoutLM, change only the boxes, write activity/layout_<change>.csv."""
    if change not in ACTIVITY_CHANGES:
        raise ValueError("ACTIVITY_LAYOUT_CHANGE must be 'zero', 'shuffle' or 'coarse'")
    if isinstance(max_records,bool) or not isinstance(max_records,int) or not 1<=max_records<=len(test_records):
        raise ValueError(f"ACTIVITY_MAX_RECORDS must be an integer in 1..{len(test_records)}")
    activity_records=sorted(test_records,key=lambda r:r["id"])[:max_records]
    canonical_by_id={x["record_id"]:x for x in layout_rows}
    load_layoutlm()
    activity_rows=[]
    for r in activity_records:
        result=layout_answer(r["question"],r["words"],changed_boxes(r,change),r["image_size"])
        c=canonical_by_id[r["id"]]
        activity_rows.append({"record_id":r["id"],"field":r["field"],"change":change,
                              "canonical_answer":c["answer"],"variant_answer":result["answer"],
                              "canonical_anls":c["anls"],"variant_anls":anls(result["answer"],r["answers"])})
    release_layoutlm()
    print({"change":change,"subset":subset_coverage(activity_records),
           "canonical_mean_anls":float(np.mean([x["canonical_anls"] for x in activity_rows])),
           "variant_mean_anls":float(np.mean([x["variant_anls"] for x in activity_rows]))})
    print_field_deltas(activity_rows,change)
    activity_dir=Path(OUTPUT_DIR)/"activity"; activity_dir.mkdir(parents=True,exist_ok=True)
    activity_path=activity_dir/f"layout_{change}.csv"
    write_csv(activity_path,activity_rows,["record_id","field","change","canonical_answer","variant_answer",
                                          "canonical_anls","variant_anls"])
    print("Wrote",activity_path,"(the canonical exports are unchanged)")
    return activity_rows


# ---- Stages ---------------------------------------------------------------------------------------------
STAGES = {}


def stage(name):
    def register(function):
        STAGES[name] = function
        return function
    return register


@stage("environment")
def stage_environment(args):
    import importlib.metadata as importlib_metadata

    import huggingface_hub
    import pyarrow
    import torchvision
    import transformers
    load_torch()
    runtime = {
        "python":sys.version.split()[0],"torch":torch.__version__,"torchvision":torchvision.__version__,
        "transformers":transformers.__version__,"huggingface_hub":huggingface_hub.__version__,
        "pyarrow":pyarrow.__version__,"numpy":np.__version__,
        "numpy_source":"hash-locked isolated environment","pillow":importlib_metadata.version("pillow"),
        "device":DEVICE,"cuda_available":torch.cuda.is_available(),
        "environment":"uv isolated environment (requirements.lock.txt), Python "+platform.python_version(),
    }
    if torch.cuda.is_available():
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    save_state("environment.json",runtime)
    print(runtime)
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. The notebook will run on CPU, but the 229 Pix2Struct questions will be much slower. "
              "In Colab choose Runtime > Change runtime type > T4 GPU, then Run all again.")


@stage("models")
def stage_models(args):
    layout_snapshot=stage_and_verify_model(LAYOUTLM_DIR,LAYOUTLM_MANIFEST)
    pix_snapshot=stage_and_verify_model(PIX2STRUCT_DIR,PIX2STRUCT_MANIFEST)
    save_state("snapshots.json",{"layoutlm":layout_snapshot,"pix2struct":pix_snapshot})
    print("LayoutLM:",layout_snapshot)
    print("Pix2Struct:",pix_snapshot)


@stage("cord")
def stage_cord(args):
    from huggingface_hub import hf_hub_download
    CORD_DIR.mkdir(parents=True,exist_ok=True)
    cord_paths={}
    for split,spec in CORD_FILES.items():
        local=Path(hf_hub_download(
            repo_id=CORD_REPO,repo_type="dataset",filename=spec["path"],revision=CORD_REVISION,
            local_dir=str(CORD_DIR),
        ))
        if local.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"{split} shard size mismatch")
        digest=sha256_file(local)
        if digest != spec["sha256"]:
            raise RuntimeError(f"{split} shard SHA-256 mismatch")
        cord_paths[split]=str(local)
        print(split, local.stat().st_size, digest)
    save_state("cord.json",cord_paths)


@stage("parse")
def stage_parse(args):
    import pyarrow.parquet as pq
    cord_paths=load_state("cord.json")
    pages_by_source={}
    for split,path in cord_paths.items():
        table=pq.read_table(path,columns=["image","ground_truth"])
        rows=table.to_pylist()
        if len(rows)!=CORD_FILES[split]["rows"]:
            raise RuntimeError(f"{split}: expected {CORD_FILES[split]['rows']} rows, got {len(rows)}")
        pages=[]
        for idx,row in enumerate(rows):
            image=decode_image_cell(row["image"])
            page=page_from_ground_truth(row["ground_truth"],f"{split}-{idx:03d}",image,split,idx)
            pages.append({k:v for k,v in page.items() if k!="image"})
        pages_by_source[split]=pages
    save_state("pages.json",pages_by_source)
    print({k:len(v) for k,v in pages_by_source.items()})


@stage("split")
def stage_split(args):
    data=split_pages(load_state("pages.json"))
    save_state("dataset.json",data)
    print({k:data[k] for k in ("supported_unique_pages","page_splits","qa_splits","test_qa_digest")})


@stage("preview")
def stage_preview(args):
    test_records,test_pages=load_dataset()
    path=state_path("preview.png")
    draw_preview(test_records,test_pages,path)
    save_state("preview.json",{"path":str(path)})


@stage("baselines")
def stage_baselines(args):
    test_records=load_state("dataset.json")["test_records"]
    last_number_rows=baseline_rows(test_records,"Last-number baseline",lambda r:last_number_answer(r["words"]))
    keyword_rows=baseline_rows(test_records,"Keyword lookup",lambda r:keyword_lookup_answer(r["question"],r["words"]))
    save_state("baselines.json",{"last_number_rows":last_number_rows,"keyword_rows":keyword_rows})
    print("Last-number:",corpus_metrics(last_number_rows))
    print("Keyword lookup:",corpus_metrics(keyword_rows))


@stage("layoutlm-check")
def stage_layoutlm_check(args):
    load_seconds,parameter_count=load_layoutlm()
    probes=layoutlm_rejection_probes()
    for probe in probes: print("Refused as expected —",probe["check"],"->",probe["message"])
    save_state("layoutlm_check.json",{"parameter_count":parameter_count,"load_seconds":load_seconds,"rejection_probes":probes})
    print({"parameters":parameter_count,"load_seconds":load_seconds,"device":DEVICE})


@stage("layoutlm-eval")
def stage_layoutlm_eval(args):
    test_records=load_state("dataset.json")["test_records"]
    load_layoutlm()
    layout_rows,layout_times,layout_peak_gpu=evaluate_layoutlm(test_records)
    save_state("layoutlm_eval.json",{"rows":layout_rows,"times":layout_times,"peak_gpu_memory_bytes":layout_peak_gpu})
    print(corpus_metrics(layout_rows))


@stage("layoutlm-no-layout")
def stage_layoutlm_no_layout(args):
    test_records=load_state("dataset.json")["test_records"]; layout_rows=load_state("layoutlm_eval.json")["rows"]
    if RUN_MODALITY_ROBUSTNESS: load_layoutlm()
    rows=layout_no_layout(test_records,layout_rows)
    save_state("layoutlm_no_layout.json",{"rows":rows})


@stage("layoutlm-probes")
def stage_layoutlm_probes(args):
    data=load_state("dataset.json"); layout_rows=load_state("layoutlm_eval.json")["rows"]
    test_pages={p["id"]:p for p in data["test_pages"]}
    if RUN_PARAPHRASE_EXPERIMENT or RUN_UNANSWERABLE_PROBE: load_layoutlm()
    paraphrase_rows,unanswerable_rows=layout_probes(data["test_records"],test_pages,layout_rows)
    save_state("layoutlm_probes.json",{"paraphrase_rows":paraphrase_rows,"unanswerable_rows":unanswerable_rows})


@stage("pix2struct-check")
def stage_pix2struct_check(args):
    load_seconds,parameter_count=load_pix2struct()
    font_sha256=hashlib.sha256(FONT_BYTES).hexdigest()
    probes=pix2struct_rejection_probes()
    for probe in probes: print("Refused as expected —",probe["check"],"->",probe["message"])
    save_state("pix2struct_check.json",{"parameter_count":parameter_count,"load_seconds":load_seconds,
                                         "font_sha256":font_sha256,"rejection_probes":probes})
    print({"parameters":parameter_count,"load_seconds":load_seconds,"font_sha256":font_sha256})


@stage("pix2struct-eval")
def stage_pix2struct_eval(args):
    test_records,test_pages=load_dataset()
    load_pix2struct()
    pix_rows,pix_times,pix_peak_gpu=evaluate_pix2struct(test_records,test_pages)
    save_state("pix2struct_eval.json",{"rows":pix_rows,"times":pix_times,"peak_gpu_memory_bytes":pix_peak_gpu})
    print(corpus_metrics(pix_rows))


@stage("pix2struct-degraded")
def stage_pix2struct_degraded(args):
    test_records,test_pages=load_dataset(); pix_rows=load_state("pix2struct_eval.json")["rows"]
    if RUN_MODALITY_ROBUSTNESS: load_pix2struct()
    rows=pix_degraded(test_records,test_pages,pix_rows)
    save_state("pix2struct_degraded.json",{"rows":rows})


@stage("pix2struct-probes")
def stage_pix2struct_probes(args):
    test_records,test_pages=load_dataset(); pix_rows=load_state("pix2struct_eval.json")["rows"]
    if RUN_PARAPHRASE_EXPERIMENT or RUN_UNANSWERABLE_PROBE: load_pix2struct()
    paraphrase_rows,unanswerable_rows=pix_probes(test_records,test_pages,pix_rows)
    save_state("pix2struct_probes.json",{"paraphrase_rows":paraphrase_rows,"unanswerable_rows":unanswerable_rows})


@stage("compare")
def stage_compare(args):
    res=load_results()
    systems=system_table(res)
    system_metrics={name:corpus_metrics(rows) for name,rows in systems}
    print(f"{'System':<24} {'ANLS':>8} {'Exact':>8} {'Empty':>8} {'In OCR':>8}")
    print("-"*62)
    for name,_ in systems:
        m=system_metrics[name]
        print(f"{name:<24} {m['anls']:>8.3f} {m['exact_match']:>8.3f} {m['empty_rate']:>8.3f} {m['answer_in_ocr_rate']:>8.3f}")


@stage("fields")
def stage_fields(args):
    res=load_results()
    layout_by_field={r["field"]:r for r in field_metrics(res["layout_rows"])}
    pix_by_field={r["field"]:r for r in field_metrics(res["pix_rows"])}
    print(f"{'Field':<30} {'N':>4} {'L-ANLS':>8} {'P-ANLS':>8} {'L-EM':>7} {'P-EM':>7}")
    for field in QUESTION_TEMPLATES:
        if field in layout_by_field and field in pix_by_field:
            l,p=layout_by_field[field],pix_by_field[field]
            print(f"{field:<30} {l['support']:>4} {l['mean_anls']:>8.3f} {p['mean_anls']:>8.3f} {l['exact_match']:>7.3f} {p['exact_match']:>7.3f}")


@stage("agreement")
def stage_agreement(args):
    res=load_results()
    _,agreement_counts=agreement(load_state("dataset.json")["test_records"],res["layout_rows"],res["pix_rows"])
    print(agreement_counts)


@stage("off-page")
def stage_off_page(args):
    print_off_page(load_results()["pix_rows"])


@stage("paraphrase")
def stage_paraphrase(args):
    res=load_results()
    paraphrase_all=res["layout_paraphrase_rows"]+res["pix_paraphrase_rows"]
    if paraphrase_all:
        print({"subset":subset_coverage(paraphrase_records(load_state("dataset.json")["test_records"]))})
        for model_name in ("LayoutLM","Pix2Struct"):
            rows=[r for r in paraphrase_all if r["model"]==model_name]
            print({"model":model_name,"n":len(rows),
                   "canonical_anls":float(np.mean([r["canonical_anls"] for r in rows])),
                   "paraphrased_anls":float(np.mean([r["paraphrased_anls"] for r in rows])),
                   "answer_stability":float(np.mean([r["answer_stable"] for r in rows]))})
    else:
        print("Paraphrase experiment disabled.")


@stage("unanswerable")
def stage_unanswerable(args):
    res=load_results()
    for row in res["layout_unanswerable_rows"]+res["pix_unanswerable_rows"]: print(row)


@stage("windows")
def stage_windows(args):
    res=load_results()
    w=window_summary(res["layout_rows"]); g=generation_summary(res["pix_rows"])
    print({"layoutlm_one_window":w["one_window"],"layoutlm_multi_window":w["multi_window"],
           "layoutlm_mean_windows":w["mean"],"layoutlm_max_windows":w["max"]})
    print({"pix2struct_mean_new_tokens":g["mean_new_tokens"],"pix2struct_max_new_tokens_observed":g["max_new_tokens"],
           "pix2struct_truncation_rate":g["truncation_rate"]})


@stage("resources")
def stage_resources(args):
    for row in resource_rows(load_results()): print(row)


@stage("panels")
def stage_panels(args):
    test_records,test_pages=load_dataset(); res=load_results()
    layout_rows,pix_rows=res["layout_rows"],res["pix_rows"]
    agreement_rows,_=agreement(test_records,layout_rows,pix_rows)
    layout_by_id={r["record_id"]:r for r in layout_rows}; pix_by_id={r["record_id"]:r for r in pix_rows}
    examples_dir=Path(OUTPUT_DIR)/"examples"; examples_dir.mkdir(parents=True,exist_ok=True)
    panels=[]
    for ex in select_examples(agreement_rows,layout_rows,pix_rows):
        path=examples_dir/f"{ex['agreement_category']}_{ex['record_id']}.png"
        draw_qa_panel(ex,test_pages,layout_by_id,pix_by_id).save(path)
        panels.append({"category":ex["agreement_category"],"record_id":ex["record_id"],"path":str(path)})
    save_state("panels.json",panels)


@stage("export")
def stage_export(args):
    export_all(Path(OUTPUT_DIR))


@stage("byod")
def stage_byod(args):
    run_byod()


@stage("summary")
def stage_summary(args):
    out_dir=Path(OUTPUT_DIR)
    missing=[str(out_dir/name) for name in REQUIRED_OUTPUTS if not (out_dir/name).is_file()]
    if missing: raise RuntimeError(f"Required outputs missing: {missing}")
    data=load_state("dataset.json"); test_records=data["test_records"]; res=load_results()
    system_metrics={name:corpus_metrics(rows) for name,rows in system_table(res)}
    print("DIMER Document Question Answering notebook: summary of this run")
    print("-"*62)
    print(f"Test pages: {len(data['test_pages'])}")
    print(f"QA records: {len(test_records)}")
    print(f"Fields represented: {len({r['field'] for r in test_records})}")
    print()
    print(f"{'System':<24} {'ANLS':>8} {'Exact':>8} {'Empty':>8}")
    print("-"*52)
    for name in ("Last-number baseline","Keyword lookup","LayoutLM","Pix2Struct"):
        m=system_metrics[name]; print(f"{name:<24} {m['anls']:>8.3f} {m['exact_match']:>8.3f} {m['empty_rate']:>8.3f}")
    print()
    print("LayoutLM")
    print("  OCR required: yes")
    print("  answer localization: yes")
    print(f"  mean latency: {np.mean(res['layout_times']):.4f} s")
    print("Pix2Struct")
    print("  OCR required: no")
    print("  answer localization: no")
    print(f"  mean latency: {np.mean(res['pix_times']):.4f} s")
    print(f"  truncation rate: {np.mean([r['truncated'] for r in res['pix_rows']]):.4f}")
    print(f"Outputs: {out_dir}/")


@stage("activity")
def stage_activity(args):
    run_activity(load_state("dataset.json")["test_records"],load_state("layoutlm_eval.json")["rows"],
                 args.change,args.max_records)


REFUSAL_EXIT = 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one stage of the document QA workshop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--change", default="shuffle")
    parser.add_argument("--max-records", type=int, default=60)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    refusal = Path(config["work_dir"]) / "state" / "refusal.txt"
    refusal.unlink(missing_ok=True)
    try:
        configure(config)
        STAGES[args.stage](args)
    except (ValueError, FileNotFoundError) as exc:
        # An input refusal (BYOD contract, activity controls): the notebook re-raises it as a ValueError.
        refusal.parent.mkdir(parents=True, exist_ok=True)
        refusal.write_text(str(exc), encoding="utf-8")
        print("REFUSED:", exc, flush=True)
        return REFUSAL_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
