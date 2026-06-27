"""
select_peerread_10papers.py
============================
- Chọn 10 paper từ PeerRead dataset
- Lưu MỖI paper thành 1 file JSON riêng  →  output_papers/<id>_<slug>.json
- Tải PDF của từng paper                  →  output_papers/<id>_<slug>.pdf
- Tạo index tổng hợp                      →  output_papers/index.json

PDF strategy:
  acl / conll  →  copy từ PeerRead/data/.../pdfs/<id>.pdf  (có sẵn trong repo)
  iclr         →  tải từ OpenReview  https://openreview.net/pdf?id=<openreview_id>
                  (openreview_id lấy từ field 'histories' trong review JSON)
  arxiv        →  tải từ https://arxiv.org/pdf/<arxiv_id>
                  (paper_id là arxiv ID dạng 1234.56789)
"""

import os
import re
import json
import shutil
import random
import time
import urllib.request
import urllib.error
from glob import glob
from collections import defaultdict

import pandas as pd

SEED = 42
random.seed(SEED)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PEERREAD_ROOT    = r"./PeerRead"
DATA_DIR         = os.path.join(PEERREAD_ROOT, "data")
OUTPUT_DIR       = "output_papers"          
INDEX_JSON       = os.path.join(OUTPUT_DIR, "index.json")

MIN_REVIEWS      = 2
MIN_REVIEW_WORDS = 50
MIN_ABSTRACT_LEN = 30

PDF_TIMEOUT      = 30   
PDF_RETRY        = 2    
PDF_DELAY        = 1.5  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def slugify(text: str, max_len: int = 40) -> str:
    """Chuyển title thành tên file an toàn."""
    s = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    s = "_".join(s.split())[:max_len]
    return s or "paper"


def paper_source(filepath: str) -> str:
    for part in filepath.replace("\\", "/").lower().split("/"):
        if part.startswith("acl"):   return "acl"
        if part.startswith("iclr"):  return "iclr"
        if part.startswith("nips"):  return "nips"
        if part.startswith("conll"): return "conll"
        if part.startswith("arxiv"): return "arxiv"
    return "other"


def review_text(r: dict) -> str:
    return str(r.get("comments", "")).strip()


def avg_review_words(reviews: list) -> float:
    lens = [len(review_text(r).split()) for r in reviews if review_text(r)]
    return sum(lens) / len(lens) if lens else 0.0


def infer_accepted(reviews: list):
    scores = []
    for r in reviews:
        rec = r.get("RECOMMENDATION", None)
        if rec is None:
            continue
        try:
            scores.append(float(rec))
        except (ValueError, TypeError):
            s = str(rec).lower()
            if "accept" in s:   scores.append(4.0)
            elif "reject" in s: scores.append(1.0)
    if not scores:
        return None
    avg = sum(scores) / len(scores)
    return avg >= 3.0 if max(scores) <= 5 else avg >= 6.0


def download_url(url: str, dest: str, retries: int = PDF_RETRY) -> bool:
    """Tải file từ URL về dest. Trả về True nếu thành công."""
    headers = {"User-Agent": "Mozilla/5.0 (research use)"}
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=PDF_TIMEOUT) as resp:
                data = resp.read()
            if len(data) < 1000:         
                print(f"    [skip] Response too small ({len(data)} bytes)")
                return False
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"    [attempt {attempt}/{retries}] {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(PDF_DELAY * attempt)
    return False


def find_local_pdf(review_path: str, paper_id: str) -> str | None:
    """
    Tìm PDF có sẵn trong repo PeerRead.
    Thư mục pdfs/ là anh em với reviews/ trong cùng split.
    """
    split_dir = os.path.dirname(os.path.dirname(review_path))
    pdfs_dir  = os.path.join(split_dir, "pdfs")
    if not os.path.isdir(pdfs_dir):
        return None
    for candidate in (f"{paper_id}.pdf", f"{paper_id}.PDF"):
        full = os.path.join(pdfs_dir, candidate)
        if os.path.exists(full):
            return full
    # Thử tìm bất kỳ PDF nào khớp stem
    for fname in os.listdir(pdfs_dir):
        stem = os.path.splitext(fname)[0]
        if stem == str(paper_id) and fname.lower().endswith(".pdf"):
            return os.path.join(pdfs_dir, fname)
    return None


def get_openreview_id(review_data: dict) -> str | None:
    """
    Với ICLR, review JSON có thể chứa openreview ID trong:
      - 'histories' list  (field 'id' hoặc 'forum')
      - top-level 'id'  (dạng số nguyên → không phải openreview ID)
    """
    histories = review_data.get("histories", [])
    if isinstance(histories, list):
        for h in histories:
            if isinstance(h, dict):
                for key in ("id", "forum", "openreview_id", "url"):
                    val = str(h.get(key, "")).strip()
                    if val and len(val) >= 6 and not val.isdigit():
                        return val
    return None


def fetch_pdf(paper: dict, dest_path: str) -> tuple[bool, str]:
    """
    Tải PDF cho một paper. Trả về (success, method).
    """
    source     = paper["source"]
    paper_id   = paper["paper_id"]
    review_path = paper["review_path"]

    # ── 1. Tìm PDF local trong repo (acl, conll — có sẵn) ────────────────
    local = find_local_pdf(review_path, paper_id)
    if local:
        shutil.copy2(local, dest_path)
        return True, "local_copy"

    # ── 2. Arxiv ──────────────────────────────────────────────────────────
    #   paper_id của arxiv là dạng "1234.56789" hoặc "cs/0612056"
    if source == "arxiv" or re.match(r"^\d{4}\.\d{4,5}$", str(paper_id)):
        url = f"https://arxiv.org/pdf/{paper_id}"
        print(f"    Downloading from arxiv: {url}")
        ok = download_url(url, dest_path)
        return ok, "arxiv"

    # ── 3. ICLR → OpenReview ──────────────────────────────────────────────
    if source == "iclr":
        # Thử lấy openreview ID từ histories
        rd = load_json(review_path)
        or_id = get_openreview_id(rd) if rd else None

        if or_id:
            url = f"https://openreview.net/pdf?id={or_id}"
            print(f"    Downloading from OpenReview: {url}")
            ok = download_url(url, dest_path)
            if ok:
                return True, "openreview"

        # Fallback: thử tìm qua Semantic Scholar bằng title
        return False, "iclr_no_id"

    # ── 4. ACL Anthology ──────────────────────────────────────────────────
    if source == "acl":
        return False, "acl_no_direct_url"

    return False, "unknown_source"


# ── 1. Scan review files ──────────────────────────────────────────────────────
print(f"Scanning: {DATA_DIR}\n")
review_files = glob(os.path.join(DATA_DIR, "**", "reviews", "*.json"), recursive=True)
review_files = [f for f in review_files
                if "nips_2013" not in f.replace("\\", "/").lower()]
print(f"Review files found: {len(review_files)}")

# ── 2. Parse + filter ─────────────────────────────────────────────────────────
papers = []
skip   = defaultdict(int)

for rp in review_files:
    rd = load_json(rp)
    if not rd or not isinstance(rd, dict):
        continue

    reviews = rd.get("reviews", [])
    if not isinstance(reviews, list):
        continue
    reviews = [r for r in reviews if not r.get("is_meta_review", False)]

    if len(reviews) < MIN_REVIEWS:
        skip["few_reviews"] += 1
        continue

    avg_len = avg_review_words(reviews)
    if avg_len < MIN_REVIEW_WORDS:
        skip["short_reviews"] += 1
        continue

    title = str(rd.get("title", "")).strip()
    if not title or title.lower() == "none":
        skip["no_title"] += 1
        continue

    abstract = str(rd.get("abstract", "")).strip()
    if len(abstract.split()) < MIN_ABSTRACT_LEN:
        stem       = os.path.basename(rp).replace(".json", "")
        split_dir  = os.path.dirname(os.path.dirname(rp))
        paper_path = os.path.join(split_dir, "parsed_pdfs", stem + ".pdf.json")
        pd_data    = load_json(paper_path)
        if pd_data:
            abstract = str(pd_data.get("metadata", {}).get("abstractText", "")).strip()

    if len(abstract.split()) < MIN_ABSTRACT_LEN:
        skip["no_abstract"] += 1
        continue

    accepted = infer_accepted(reviews)
    if accepted is None:
        skip["no_decision"] += 1
        continue

    source   = paper_source(rp)
    paper_id = str(rd.get("id", os.path.basename(rp).replace(".json", "")))

    papers.append({
        "paper_id":         paper_id,
        "title":            title,
        "abstract":         abstract,
        "source":           source,
        "accepted":         accepted,
        "num_reviews":      len(reviews),
        "avg_review_words": round(avg_len, 1),
        "review_path":      rp,
        "reviews":          reviews,
        "_raw_review_data": rd,   # tạm giữ để lấy openreview ID
    })

print(f"Filter summary:  {dict(skip)}")
print(f"Candidates:      {len(papers)}\n")

# ── 3. Group + sample ─────────────────────────────────────────────────────────
groups = defaultdict(list)
for p in papers:
    groups[p["source"]].append(p)

print("Group breakdown:")
for g in sorted(groups):
    acc = sum(1 for p in groups[g] if p["accepted"])
    rej = len(groups[g]) - acc
    print(f"  {g:8s}: {len(groups[g]):4d}  (accept={acc}, reject={rej})")

plan = [
    ("acl",   True,  2), ("acl",   False, 1),
    ("iclr",  True,  2), ("iclr",  False, 1),
    ("arxiv", True,  1), ("arxiv", False, 1),
    ("conll", True,  1), ("conll", False, 1),
]

selected = []
used_ids = set()

for src, acc_flag, k in plan:
    pool = [p for p in groups.get(src, [])
            if p["accepted"] == acc_flag and p["paper_id"] not in used_ids]
    pool = sorted(pool, key=lambda x: (x["num_reviews"], x["avg_review_words"]), reverse=True)
    top  = pool[:50] if len(pool) > 50 else pool
    take = random.sample(top, min(k, len(top)))
    if len(take) < k:
        print(f"[WARNING] {src} accept={acc_flag}: need {k}, got {len(take)}")
    for p in take:
        used_ids.add(p["paper_id"])
    selected.extend(take)

if len(selected) < 10:
    needed   = 10 - len(selected)
    fallback = [p for p in papers if p["paper_id"] not in used_ids]
    fallback = sorted(fallback, key=lambda x: (x["num_reviews"], x["avg_review_words"]), reverse=True)
    selected.extend(random.sample(fallback[:100], min(needed, len(fallback[:100]))))

selected = selected[:10]

if not selected:
    raise SystemExit("[ERROR] Không chọn được paper nào.")

# ── 4. In bảng tóm tắt ────────────────────────────────────────────────────────
rows = [{
    "title":     (p["title"][:50] + "…") if len(p["title"]) > 50 else p["title"],
    "source":    p["source"],
    "accepted":  p["accepted"],
    "#reviews":  p["num_reviews"],
    "avg_words": p["avg_review_words"],
    "paper_id":  p["paper_id"],
} for p in selected]
print("\n===== SELECTED 10 PAPERS =====")
print(pd.DataFrame(rows).to_string(index=False))

# ── 5. Lưu từng file JSON + tải PDF ──────────────────────────────────────────
print(f"\nSaving to: {OUTPUT_DIR}/\n")

index_entries = []
pdf_results   = []

for i, p in enumerate(selected, 1):
    tag  = "ACCEPT" if p["accepted"] else "REJECT"
    slug = slugify(p["title"])
    base = f"{i:02d}_{p['paper_id']}_{slug}"

    # ── 5a. Enriched reviews ──────────────────────────────────────────────
    enriched = []
    for r in p["reviews"]:
        enriched.append({
            "comments":              r.get("comments", ""),
            "RECOMMENDATION":        r.get("RECOMMENDATION"),
            "REVIEWER_CONFIDENCE":   r.get("REVIEWER_CONFIDENCE"),
            "ORIGINALITY":           r.get("ORIGINALITY"),
            "SUBSTANCE":             r.get("SUBSTANCE"),
            "SOUNDNESS_CORRECTNESS": r.get("SOUNDNESS_CORRECTNESS"),
            "MEANINGFUL_COMPARISON": r.get("MEANINGFUL_COMPARISON"),
            "IMPACT":                r.get("IMPACT"),
            "CLARITY":               r.get("CLARITY"),
            "APPROPRIATENESS":       r.get("APPROPRIATENESS"),
            "PRESENTATION_FORMAT":   r.get("PRESENTATION_FORMAT"),
            "is_meta_review":        r.get("is_meta_review", False),
        })

    # ── 5b. Ghi JSON riêng ───────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, base + ".json")
    paper_record = {
        "paper_id":         p["paper_id"],
        "title":            p["title"],
        "abstract":         p["abstract"],
        "source":           p["source"],
        "accepted":         p["accepted"],
        "num_reviews":      p["num_reviews"],
        "avg_review_words": p["avg_review_words"],
        "review_path":      p["review_path"],
        "reviews":          enriched,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(paper_record, f, ensure_ascii=False, indent=2)

    # ── 5c. Tải PDF ──────────────────────────────────────────────────────
    pdf_path = os.path.join(OUTPUT_DIR, base + ".pdf")
    print(f"[{i:2d}/10] [{tag}] {p['title'][:55]}")
    print(f"       source={p['source']}  id={p['paper_id']}")

    pdf_ok, method = fetch_pdf(p, pdf_path)

    if pdf_ok:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"       ✓ PDF saved  ({method}, {size_kb:.0f} KB)")
        pdf_status = "ok"
    else:
        print(f"       ✗ PDF not available ({method})")
        pdf_path   = None
        pdf_status = method

    time.sleep(PDF_DELAY)

    # ── 5d. Index entry ──────────────────────────────────────────────────
    index_entries.append({
        "index":       i,
        "paper_id":    p["paper_id"],
        "title":       p["title"],
        "source":      p["source"],
        "accepted":    p["accepted"],
        "num_reviews": p["num_reviews"],
        "json_file":   os.path.basename(json_path),
        "pdf_file":    os.path.basename(pdf_path) if pdf_path else None,
        "pdf_status":  pdf_status,
    })
    pdf_results.append((p["title"][:55], pdf_ok, method))
    print()

# ── 6. Ghi index.json ─────────────────────────────────────────────────────────
with open(INDEX_JSON, "w", encoding="utf-8") as f:
    json.dump(index_entries, f, ensure_ascii=False, indent=2)

# ── 7. Báo cáo kết quả ────────────────────────────────────────────────────────
print("=" * 65)
print(f"OUTPUT DIRECTORY: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 65)
print(f"{'#':>3}  {'Status':8}  {'Method':20}  Title")
print("-" * 65)
for idx, (title, ok, method) in enumerate(pdf_results, 1):
    status = "✓ OK" if ok else "✗ FAIL"
    print(f"{idx:3d}  {status:8}  {method:20}  {title}")

print("-" * 65)
ok_count   = sum(1 for _, ok, _ in pdf_results if ok)
fail_count = len(pdf_results) - ok_count
print(f"PDFs downloaded: {ok_count}/10   Failed: {fail_count}/10")
print(f"\nindex.json saved → {INDEX_JSON}")

if fail_count:
    print(f"""
NOTE: {fail_count} PDF(s) could not be downloaded automatically.
Possible reasons:
  - ACL 2017 / ICLR papers: no direct URL mapping in PeerRead repo
  - Network issue or rate-limit

Manual download tips:
  • ACL papers  → https://aclanthology.org  (search by title)
  • ICLR papers → https://openreview.net    (search by title)
  • arxiv       → https://arxiv.org/abs/<id>
""")