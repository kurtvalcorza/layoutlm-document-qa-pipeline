"""Patch a generated suite executor for a BYOD journey of the receipt capstone (78ef7c3).

The preamble still fetches the committed notebook bytes and asserts the git blob first. It then
(1) stages the BYOD stand-in archives from the attached Kaggle dataset (.bin -> .zip under
/kaggle/working/byod), (2) replaces the code-02 run controls in-run, guarded by the SHA-256 of the
committed cell source, as a Colab user would tick the form before Run all, and (3) for B1 appends
one harness cell that re-executes code-40's exact committed source against a path-traversal ZIP
to record the refusal. The applied diff and the committed bytes are saved as evidence.

usage: python patch_byod.py b1|b2 <committed tutorial .ipynb>
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

variant, committed = sys.argv[1], Path(sys.argv[2])
SLUG = "dimer-nb2-small-business-receipt-intelligence-caps"
KDIR = Path(__file__).parent / f"byod_{variant}" / SLUG
orig = json.loads(committed.read_text(encoding="utf-8"))
cells = {c["id"]: "".join(c["source"]) for c in orig["cells"]}

mode, archive = {"b1": ("inference", "inference"), "b2": ("adapt", "adapt")}[variant]
c02 = cells["code-02"]
new02 = c02
for old, new in [
    ("USE_BYOD = False # @param", "USE_BYOD = True # @param"),
    ("BYOD_MODE = 'inference' # @param", f"BYOD_MODE = '{mode}' # @param"),
    ("BYOD_PATH = '' # @param", f"BYOD_PATH = '/kaggle/working/byod/{archive}.zip' # @param"),
    ("BYOD_AUTHORIZED = False # @param", "BYOD_AUTHORIZED = True # @param"),
]:
    assert new02.count(old) == 1, old
    new02 = new02.replace(old, new)

replace = {"code-02": {"committed_sha256": hashlib.sha256(c02.encode()).hexdigest(), "source": new02}}
append = []
if variant == "b1":
    harness = (
        "# Kaggle harness (NOT part of the tutorial): refused-ZIP journey.\n"
        "# Re-executes code-40's exact committed source with an archive that carries '../escape.png'.\n"
        "import hashlib as _h\n"
        f"_SRC40 = {cells['code-40']!r}\n"
        f"assert _h.sha256(_SRC40.encode()).hexdigest() == {hashlib.sha256(cells['code-40'].encode()).hexdigest()!r}\n"
        "BYOD_MODE = 'inference'\n"
        "BYOD_PATH = '/kaggle/working/byod/refused_zipslip.zip'\n"
        "_before = sorted(p.name for p in ROOT.parent.iterdir())\n"
        "try:\n"
        "    exec(_SRC40)\n"
        "except ValueError as _e:\n"
        "    print('REFUSED as designed:', _e)\n"
        "    assert 'Unsafe archive path' in str(_e), _e\n"
        "else:\n"
        "    raise AssertionError('The path-traversal archive was accepted')\n"
        "assert sorted(p.name for p in ROOT.parent.iterdir()) == _before, 'a refused archive created a run directory'\n"
        "print('No BYOD run directory was created for the refused archive.')\n"
    )
    ast.parse(harness)
    append.append({"cell_type": "code", "execution_count": None, "id": "kaggleharness-refused-zip",
                   "metadata": {}, "outputs": [], "source": harness.splitlines(keepends=True)})
spec = {"replace": replace, "append": append}

INJECT = '''
# --- Kaggle harness for the capstone BYOD journey (NOT part of the tutorial) ---
import difflib, glob
os.makedirs("/kaggle/working/byod", exist_ok=True)
for _p in glob.glob("/kaggle/input/**/*.bin", recursive=True):
    _dst = "/kaggle/working/byod/" + os.path.basename(_p)[:-4] + ".zip"
    shutil.copy(_p, _dst)
    print("staged", _dst, os.path.getsize(_dst), hashlib.sha256(open(_dst, "rb").read()).hexdigest())
HARNESS = json.loads({spec!r})
_nb = json.loads(raw.decode("utf-8"))
_diff = []
for _c in _nb["cells"]:
    _r = HARNESS["replace"].get(_c.get("id"))
    if _r is None:
        continue
    _old = "".join(_c["source"])
    assert hashlib.sha256(_old.encode()).hexdigest() == _r["committed_sha256"], _c["id"]
    _diff += list(difflib.unified_diff(_old.splitlines(), _r["source"].splitlines(), "a/" + _c["id"], "b/" + _c["id"], lineterm="", n=0))
    _c["source"] = _r["source"].splitlines(keepends=True)
_nb["cells"] += HARNESS["append"]
if HARNESS["append"]:
    _diff.append("+ appended harness cells: " + ", ".join(c["id"] for c in HARNESS["append"]))
open(NB_PATH, "w", encoding="utf-8").write(json.dumps(_nb, ensure_ascii=False, indent=1))
open(os.path.join(SCRATCH, "tutorial-committed.ipynb"), "wb").write(raw)
open(os.path.join(SCRATCH, "harness-diff.txt"), "w", encoding="utf-8").write("\\n".join(_diff) + "\\n")
print("\\n".join(_diff))
'''.format(spec=json.dumps(spec, ensure_ascii=False))

nb_path = KDIR / (SLUG + ".ipynb")
k = json.loads(nb_path.read_text(encoding="utf-8"))
pre = "".join(k["cells"][0]["source"])
anchor = 'open(NB_PATH, "wb").write(raw)\n'
assert pre.count(anchor) == 1
k["cells"][0]["source"] = pre.replace(anchor, anchor + INJECT).splitlines(keepends=True)
ev = "".join(k["cells"][-1]["source"])
ev_anchor = 'os.makedirs(os.path.join(EV, "outputs"))\n'
assert ev.count(ev_anchor) == 1
k["cells"][-1]["source"] = ev.replace(
    ev_anchor,
    ev_anchor + 'for _n in ("tutorial-committed.ipynb", "harness-diff.txt"):\n'
    '    shutil.copy(os.path.join(SCRATCH, _n), os.path.join(EV, _n))\n',
).splitlines(keepends=True)
for c in k["cells"]:
    ast.parse("".join(c["source"]))
nb_path.write_text(json.dumps(k, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
meta_path = KDIR / "kernel-metadata.json"
meta = json.loads(meta_path.read_text())
meta["dataset_sources"] = ["kurtvalcorza/dimer-receipt-byod-standins"]
meta_path.write_text(json.dumps(meta, indent=2))
print(variant, "patched:", list(replace), [c["id"] for c in append], "| metadata:", meta)
