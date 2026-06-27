"""
Chạy script này TRƯỚC để hiểu cấu trúc thực của PeerRead trên máy bạn.
Output sẽ giúp sửa đúng select_peerread_10papers.py
"""
import os, json
from glob import glob

PEERREAD_ROOT = r"./PeerRead"
DATA_DIR = os.path.join(PEERREAD_ROOT, "data")

# ── 1. Cấu trúc thư mục ──────────────────────────────────────────────────────
print("=" * 60)
print("1. TOP-LEVEL FOLDERS IN data/")
print("=" * 60)
for item in sorted(os.listdir(DATA_DIR)):
    full = os.path.join(DATA_DIR, item)
    if os.path.isdir(full):
        subs = sorted(os.listdir(full))
        print(f"  {item}/  ->  {subs}")

# ── 2. Nội dung một paper JSON ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SAMPLE PAPER JSON  (parsed_pdfs)")
print("=" * 60)
paper_files = glob(os.path.join(DATA_DIR, "**", "parsed_pdfs", "*.json"), recursive=True)
print(f"Total parsed_pdfs files: {len(paper_files)}")
for fp in paper_files[:3]:
    print(f"\n  File: {fp}")
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    print(f"  Top-level keys: {list(d.keys())}")
    meta = d.get("metadata", {})
    if isinstance(meta, dict):
        print(f"  metadata keys: {list(meta.keys())}")
        print(f"  title   : {str(meta.get('title',''))[:80]}")
        print(f"  abstract: {str(meta.get('abstract',''))[:120]}")
        print(f"  accepted: {meta.get('accepted', 'N/A')}")
        print(f"  decision: {meta.get('decision', 'N/A')}")

# ── 3. Nội dung một review JSON ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SAMPLE REVIEW JSON  (reviews/)")
print("=" * 60)
review_files = glob(os.path.join(DATA_DIR, "**", "reviews", "*.json"), recursive=True)
print(f"Total review files: {len(review_files)}")
for fp in review_files[:3]:
    print(f"\n  File: {fp}")
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        print(f"  Type: LIST,  len={len(d)}")
        if d and isinstance(d[0], dict):
            print(f"  Item[0] keys: {list(d[0].keys())}")
            for k, v in list(d[0].items())[:8]:
                print(f"    {k}: {str(v)[:100]}")
    elif isinstance(d, dict):
        print(f"  Type: DICT,  keys={list(d.keys())}")
        reviews_inside = d.get("reviews", [])
        print(f"  reviews key has {len(reviews_inside)} items")
        if reviews_inside and isinstance(reviews_inside[0], dict):
            print(f"  reviews[0] keys: {list(reviews_inside[0].keys())}")
            for k, v in list(reviews_inside[0].items())[:8]:
                print(f"    {k}: {str(v)[:100]}")

# ── 4. Kiểm tra xem review file và paper file match nhau không ───────────────
print("\n" + "=" * 60)
print("4. MATCH CHECK: paper <-> review")
print("=" * 60)
if paper_files and review_files:
    p0 = paper_files[0]
    stem = os.path.basename(p0).replace(".pdf.json", "").replace(".json", "")
    split_dir = os.path.dirname(os.path.dirname(p0))
    reviews_dir = os.path.join(split_dir, "reviews")
    print(f"  Paper file : {p0}")
    print(f"  Stem       : {stem}")
    print(f"  Reviews dir: {reviews_dir}")
    print(f"  Exists?    : {os.path.isdir(reviews_dir)}")
    if os.path.isdir(reviews_dir):
        rfiles = os.listdir(reviews_dir)
        print(f"  Review files in dir: {rfiles[:10]}")
        # Try to find match
        for rf in rfiles:
            rstem = rf.replace(".pdf.json", "").replace(".json", "")
            if rstem == stem:
                print(f"  MATCH FOUND: {rf}")
                break
        else:
            print("  No exact match. First few stems:", [r.replace(".pdf.json","").replace(".json","") for r in rfiles[:5]])

print("\nDone. Copy output này và cho tôi xem để sửa script chính xác.")