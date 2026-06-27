import os
import asyncio
import httpx
import io
import uuid
import json
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber  

load_dotenv()

app = FastAPI(title="Paper Review Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/paper_review_realtime.html")


# ── Agent URLs ────────────────────────────────────────────────────────────────
AGENT_URLS = {
    "reviewer_1": os.getenv("REVIEWER_A_URL", "http://localhost:8001"),
    "reviewer_2": os.getenv("REVIEWER_B_URL", "http://localhost:8002"),
    "area_chair": os.getenv("JUDGE_URL", "http://localhost:8003"),
}

MAX_PAPER_CHARS        = int(os.getenv("MAX_PAPER_CHARS", "20000"))
DEFAULT_ROUNDS         = int(os.getenv("DEFAULT_DEBATE_ROUNDS", "2"))
MAX_CROSS_REVIEW_CHARS = int(os.getenv("MAX_CROSS_REVIEW_CHARS", "2500"))

print("\n[ORCHESTRATOR] STARTED")
print("[ORCHESTRATOR] AGENT_URLS =", AGENT_URLS)
print("[ORCHESTRATOR] MAX_PAPER_CHARS =", MAX_PAPER_CHARS)
print("[ORCHESTRATOR] DEFAULT_ROUNDS =", DEFAULT_ROUNDS)
print("[ORCHESTRATOR] MAX_CROSS_REVIEW_CHARS =", MAX_CROSS_REVIEW_CHARS)
print("")


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_content(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Returns (paper_text, tables_section):
    - paper_text: full text extracted per page, layout-aware
    - tables_section: all tables rendered as markdown, NOT truncated
    
    Using pdfplumber instead of pypdf because:
    - pdfplumber preserves spatial layout, crucial for multi-column papers
    - pdfplumber.extract_tables() recovers table cell structure
    - pypdf.extract_text() collapses table rows into unreadable strings
    """
    pages_text = []
    all_tables_md = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract page text (layout-aware)
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            pages_text.append(f"--- Page {page_num} ---\n{text}")

            # Extract tables on this page → render as markdown
            tables = page.extract_tables()
            for tbl_idx, table in enumerate(tables):
                if not table:
                    continue
                md_rows = []
                for row_idx, row in enumerate(table):
                    cells = [str(c or "").replace("\n", " ").strip() for c in row]
                    md_rows.append("| " + " | ".join(cells) + " |")
                    # Add separator after header row
                    if row_idx == 0:
                        md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
                md_table = (
                    f"\n[Table on page {page_num}, index {tbl_idx + 1}]\n"
                    + "\n".join(md_rows)
                )
                all_tables_md.append(md_table)

    paper_text = "\n\n".join(pages_text)
    tables_section = (
        "=== EXTRACTED TABLES (complete, not truncated) ===\n"
        + "\n\n".join(all_tables_md)
        if all_tables_md
        else ""
    )

    print(f"[PDF] Extracted {len(paper_text)} chars text, {len(all_tables_md)} tables")
    return paper_text, tables_section


def build_paper_context(paper_text: str, tables_section: str, max_chars: int) -> str:
    """
    Build the full paper context string sent to agents.
    - Text is truncated to max_chars if needed
    - Tables are NEVER truncated (agents must see full table content)
    """
    if len(paper_text) > max_chars:
        truncated_text = (
            paper_text[:max_chars]
            + "\n\n[...text truncated — full tables are preserved below...]"
        )
    else:
        truncated_text = paper_text

    parts = ["=== PAPER TEXT ===\n" + truncated_text]
    if tables_section:
        parts.append(tables_section)

    return "\n\n".join(parts)


# ── A2A helper ────────────────────────────────────────────────────────────────

async def send_to_agent(client: httpx.AsyncClient, agent_url: str, message: str) -> str:
    resp = await client.post(
        f"{agent_url}/",
        json={
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"type": "text", "text": message}],
                },
            },
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", {})
    parts = result.get("parts")
    if parts is None and isinstance(result, dict):
        msg = result.get("message", {})
        if isinstance(msg, dict):
            parts = msg.get("parts")

    if not parts:
        return ""

    for part in parts:
        if isinstance(part, dict) and (
            part.get("kind") == "text" or part.get("type") == "text"
        ):
            return part.get("text", "")

    return ""


def truncate_for_cross_review(text: str, max_chars: int = MAX_CROSS_REVIEW_CHARS) -> str:
    """
    Truncate one reviewer's output before passing to the other.
    Prevents style/length bleeding between reviewers.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    return truncated + "\n\n[...truncated — focus on your own independent assessment...]"


# ── Prompt builders ───────────────────────────────────────────────────────────

# Prepended to every initial review prompt so agents know tables are present
TABLE_READING_INSTRUCTION = """
=== MANDATORY READING INSTRUCTION ===
The paper context below contains TWO sections:
1. "=== PAPER TEXT ===" — full extracted text (may be truncated for length)
2. "=== EXTRACTED TABLES ===" — ALL tables from the paper rendered as markdown (NEVER truncated)

BEFORE listing any experiment or ablation as "missing", you MUST:
- Search the "=== EXTRACTED TABLES ===" section for relevant table rows/columns
- Search the paper text for mentions of the experiment in section headings or captions
- Only claim an experiment is missing if it is GENUINELY ABSENT from BOTH sections

This is critical: pypdf-style text extraction often loses table content. The tables
section is the authoritative source for what experiments and ablations exist in the paper.
======================================

"""


def make_initial_prompt(role_label: str, paper_context: str) -> str:
    return (
        f"Write your INITIAL independent review as {role_label}.\n"
        "Do NOT be influenced by any other reviewer — this is your independent assessment.\n"
        + TABLE_READING_INSTRUCTION
        + f"Paper:\n{paper_context}"
    )


def make_rebuttal_prompt(paper_context: str, r1_truncated: str, r2_truncated: str) -> str:
    return (
        "IMPORTANT: You are now temporarily acting as the PAPER AUTHORS, NOT as a reviewer.\n\n"
        "Write a concise, factual author rebuttal (max 400 words) responding to the reviews below.\n\n"
        "Your rebuttal MUST:\n"
        "- Point to SPECIFIC table numbers, row labels, section names, or equation numbers "
        "in the paper that DIRECTLY address each reviewer concern.\n"
        "- Clarify misunderstandings — e.g., if a reviewer claims an ablation is missing "
        "but it appears in Table X row (Y), state this explicitly with the table/row reference.\n"
        "- Acknowledge legitimate concerns honestly — do not dismiss valid critique.\n"
        "- Be factual and calm, not defensive.\n"
        "- For limitations the paper already explicitly acknowledges (in abstract, footnotes, "
        "or limitations section), remind reviewers that these were disclosed transparently.\n\n"
        f"Paper (text + tables for reference):\n{paper_context}\n\n"
        f"Reviewer 1 said:\n{r1_truncated}\n\n"
        f"Reviewer 2 said:\n{r2_truncated}\n\n"
        "Write the author rebuttal now (max 400 words):"
    )


def make_revision_prompt_r1(r2_truncated: str, author_rebuttal: str) -> str:
    return (
        "You are Reviewer 1 (Domain Expert). Revise your assessment after reading:\n"
        "1. The author rebuttal — it may point to tables/sections you had missed.\n"
        "2. Reviewer 2's critique.\n\n"
        f"Reviewer 2 said:\n{r2_truncated}\n\n"
        f"Author Rebuttal:\n{author_rebuttal}\n\n"
        "Revision instructions:\n"
        "- If the rebuttal cites a specific table row or section that addresses your concern "
        "→ ACKNOWLEDGE it explicitly and update your assessment.\n"
        "- If Reviewer 2 raises a valid new technical concern → concede with reasoning.\n"
        "- If Reviewer 2 critiques something the authors already acknowledged "
        "→ note that this critique should be discounted.\n"
        "- If Reviewer 2's critique is factually wrong (e.g., claims missing ablation that "
        "exists in a table) → rebut it with the specific table/row reference.\n"
        "- Do NOT change your stance for politeness or social pressure.\n"
        "- Update Overall Rating ONLY if your view has genuinely changed.\n"
        "- Keep output under 600 words. Maintain your structured format."
    )


def make_revision_prompt_r2(r1_truncated: str, author_rebuttal: str) -> str:
    return (
        "You are Reviewer 2 (Independent Critic). Revise your assessment after reading:\n"
        "1. The author rebuttal — it may point to tables/sections you had missed.\n"
        "2. Reviewer 1's revised assessment.\n\n"
        f"Reviewer 1 (revised) said:\n{r1_truncated}\n\n"
        f"Author Rebuttal:\n{author_rebuttal}\n\n"
        "Revision instructions:\n"
        "- If the rebuttal cites a specific table row or section that addresses your concern "
        "→ WITHDRAW that critique explicitly and state why.\n"
        "- If Reviewer 1 makes a valid technical counter-argument → concede it.\n"
        "- If your critique targeted something the authors already explicitly acknowledged "
        "→ significantly reduce the weight of that critique.\n"
        "- Do NOT escalate harshness without new evidence from the paper itself.\n"
        "- Do NOT repeat a critique you already raised if the rebuttal has addressed it.\n"
        "- Do NOT converge toward Reviewer 1's conclusion unless you have independent reasoning.\n"
        "- Update Recommendation ONLY if your view has genuinely changed.\n"
        "- Keep output under 600 words. Maintain your structured format."
    )


def make_ac_prompt(transcript: str) -> str:
    return (
        "You are the Area Chair. Produce the final meta-review and decision in JSON.\n\n"
        "BIAS CORRECTION RULES (apply before scoring):\n"
        "1. If the Author Rebuttal pointed to a specific table row, section, or footnote "
        "that a reviewer had claimed was missing → DISCOUNT that reviewer critique heavily. "
        "A reviewer who misread the paper should not drive the final decision.\n"
        "2. If a reviewer escalated the same critique across multiple rounds WITHOUT citing "
        "new evidence from the paper → treat later-round escalations as LESS reliable.\n"
        "3. If both reviewers criticized something the authors explicitly acknowledged "
        "(in abstract, limitations, or footnotes) → discount as 'known limitation, transparent'.\n\n"
        f"Full discussion transcript:\n\n{transcript}"
    )


# ── Review endpoint ───────────────────────────────────────────────────────────

@app.post("/review")
async def review_paper(
    file: UploadFile = File(...),
    rounds: int = DEFAULT_ROUNDS,
):
    print("\n===================================")
    print("[ORCHESTRATOR] /review called")
    print("[ORCHESTRATOR] filename:", file.filename)
    print("[ORCHESTRATOR] rounds:", rounds)
    print("===================================")

    pdf_bytes = await file.read()
    print("[ORCHESTRATOR] PDF bytes size:", len(pdf_bytes))

    # ── Extract text + tables ─────────────────────────────────────────────────
    try:
        paper_text, tables_section = extract_pdf_content(pdf_bytes)
    except Exception as exc:
        print("[ORCHESTRATOR] ERROR reading PDF:", str(exc))
        raise HTTPException(status_code=400, detail=f"Cannot read PDF: {exc}")

    if not paper_text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF has no extractable text (scanned / empty).",
        )

    # Build full context for agents (text truncated, tables always complete)
    paper_context = build_paper_context(paper_text, tables_section, MAX_PAPER_CHARS)
    # Shorter context for rebuttal — still includes full tables
    paper_context_short = build_paper_context(paper_text, tables_section, 4000)

    print(f"[ORCHESTRATOR] paper_context length: {len(paper_context)}")
    print(f"[ORCHESTRATOR] tables_section length: {len(tables_section)}")

    debate_log: list[dict] = []

    async with httpx.AsyncClient() as client:
        try:
            # ── Round 0: Independent initial reviews (parallel) ───────────────
            print("\n[ORCHESTRATOR] Round 0: Parallel initial reviews...")
            reviewer1_initial, reviewer2_initial = await asyncio.gather(
                send_to_agent(
                    client,
                    AGENT_URLS["reviewer_1"],
                    make_initial_prompt("Reviewer 1 (Domain Expert)", paper_context),
                ),
                send_to_agent(
                    client,
                    AGENT_URLS["reviewer_2"],
                    make_initial_prompt(
                        "Reviewer 2 (Independent Scientific Critic)", paper_context
                    ),
                ),
            )

            print("[ORCHESTRATOR] Reviewer1 initial length:", len(reviewer1_initial))
            print("[ORCHESTRATOR] Reviewer2 initial length:", len(reviewer2_initial))

            debate_log.append(
                {"agent": "Reviewer 1 (Domain Expert)", "round": 0, "content": reviewer1_initial}
            )
            debate_log.append(
                {"agent": "Reviewer 2 (Independent Critic)", "round": 0, "content": reviewer2_initial}
            )

            # ── Author Rebuttal ───────────────────────────────────────────────
            print("\n[ORCHESTRATOR] Generating Author Rebuttal...")
            r1_for_rebuttal = truncate_for_cross_review(reviewer1_initial, max_chars=1500)
            r2_for_rebuttal = truncate_for_cross_review(reviewer2_initial, max_chars=1500)

            author_rebuttal = await send_to_agent(
                client,
                AGENT_URLS["reviewer_1"],
                make_rebuttal_prompt(paper_context_short, r1_for_rebuttal, r2_for_rebuttal),
            )

            print("[ORCHESTRATOR] Author rebuttal length:", len(author_rebuttal))
            debate_log.append(
                {"agent": "Author Rebuttal", "round": 0, "content": author_rebuttal}
            )

            reviewer1_current = reviewer1_initial
            reviewer2_current = reviewer2_initial

            # ── Rounds 1..N ───────────────────────────────────────────────────
            for r in range(1, rounds + 1):
                print(f"\n[ORCHESTRATOR] Debate round {r}/{rounds}")

                r2_for_r1 = truncate_for_cross_review(reviewer2_current)
                reviewer1_revised = await send_to_agent(
                    client,
                    AGENT_URLS["reviewer_1"],
                    make_revision_prompt_r1(r2_for_r1, author_rebuttal),
                )
                print("[ORCHESTRATOR] Reviewer1 revised length:", len(reviewer1_revised))
                debate_log.append(
                    {
                        "agent": "Reviewer 1 (Domain Expert)",
                        "round": r,
                        "content": reviewer1_revised,
                    }
                )
                reviewer1_current = reviewer1_revised

                r1_for_r2 = truncate_for_cross_review(reviewer1_current)
                reviewer2_revised = await send_to_agent(
                    client,
                    AGENT_URLS["reviewer_2"],
                    make_revision_prompt_r2(r1_for_r2, author_rebuttal),
                )
                print("[ORCHESTRATOR] Reviewer2 revised length:", len(reviewer2_revised))
                debate_log.append(
                    {
                        "agent": "Reviewer 2 (Independent Critic)",
                        "round": r,
                        "content": reviewer2_revised,
                    }
                )
                reviewer2_current = reviewer2_revised

            # ── Final: Area Chair ─────────────────────────────────────────────
            transcript = "\n\n".join(
                f"[{d['agent']} | Round {d['round']}]\n{d['content']}"
                for d in debate_log
            )
            print("\n[ORCHESTRATOR] Transcript length:", len(transcript))

            ac_verdict = await send_to_agent(
                client,
                AGENT_URLS["area_chair"],
                make_ac_prompt(transcript),
            )
            print("[ORCHESTRATOR] AreaChair verdict length:", len(ac_verdict))

        except Exception as exc:
            print("[ORCHESTRATOR] FATAL ERROR in /review:", str(exc))
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content={"debate_log": debate_log, "ac_verdict": ac_verdict})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "agents": AGENT_URLS}