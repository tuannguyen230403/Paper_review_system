import os
import json
import traceback
from openai import OpenAI
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


SYSTEM_PROMPT = """You are the Area Chair (AC) producing a fair, evidence-driven meta-review.

CONTEXT:
You receive a full debate transcript between:
- Reviewer 1 (R1): Domain expert
- Reviewer 2 (R2): Independent critic
- Author Rebuttal (if present)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — HALLUCINATION AUDIT (do this FIRST before scoring anything)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each critique raised by R1 or R2, verify:

A) TABLE EXISTENCE CHECK:
   Did a reviewer claim an experiment/ablation/baseline was "missing"?
   → Did the Author Rebuttal point to a specific table or section that CONTAINS it?
   → If YES: this is a HALLUCINATED critique. Discount it heavily.
     Record it in discounted_critiques as: "<critique> — Discounted: table exists (Table N)"

B) ESCALATION CHECK:
   Did a reviewer's harshness INCREASE across rounds WITHOUT citing new evidence?
   → Treat later-round escalations as LESS reliable. Note in resolution_notes.

C) ACKNOWLEDGMENT CHECK:
   Did reviewers penalize something the authors explicitly acknowledged?
   → Discount significantly. Transparency is a positive signal, not a flaw.

D) SCOPE INFLATION CHECK:
   Are reviewers demanding journal-level experiments for an 8-page conference paper?
   → Discount scope-inflated demands. Apply conference-appropriate bar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — DECISION LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After discounting hallucinated/inflated critiques, ask:
  1. What GENUINE weaknesses remain?
  2. Do they prevent verifying the core claims?
  3. Does the paper make a real contribution despite those weaknesses?

Accept    if: genuine contribution + results reproducible + claims honest
Major Rev if: core claims CANNOT be verified without new experiments
Reject    if: fundamental flaw that revisions cannot fix

Do NOT anchor on the harshest reviewer's recommendation.
Calibrate to: "Would a real program committee accept this?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — STRICT JSON ONLY
No markdown fences. No preamble. No text outside the JSON object.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "scores": {
    "ORIGINALITY": <1-5>,
    "SUBSTANCE": <1-5>,
    "SOUNDNESS_CORRECTNESS": <1-5>,
    "MEANINGFUL_COMPARISON": <1-5>,
    "IMPACT": <1-5>,
    "CLARITY": <1-5>,
    "APPROPRIATENESS": <1-5>,
    "REVIEWER_CONFIDENCE": <1-5>
  },
  "overall_recommendation": <1-5>,
  "decision_label": "<Accept | Minor Revision | Major Revision | Reject>",
  "presentation_format": "<Poster | Oral Presentation | Reject>",
  "confidence_level": <1-5>,
  "disagreement_level": <1-5>,
  "needs_additional_discussion": <true | false>,
  "comments": {
    "meta_review": "<2-4 paragraphs. Must explain: (1) which critiques were discounted and why, (2) genuine weaknesses that remain, (3) rationale for final decision>",
    "strengths": ["<specific strength with paper evidence>"],
    "weaknesses": ["<only genuine, non-hallucinated weaknesses>"],
    "required_revisions": ["<concrete, actionable revision>"],
    "suggested_experiments": ["<optional improvement, not blocking acceptance>"]
  },
  "meta": {
    "reviewer_1_key_points": ["<R1 point 1>", "<R1 point 2>"],
    "reviewer_2_key_points": ["<R2 point 1>", "<R2 point 2>"],
    "main_disagreements": ["<disagreement + your resolution>"],
    "discounted_critiques": ["<critique> — Discounted: <reason>"],
    "author_rebuttal_impact": "<How rebuttal affected decision, or 'No rebuttal provided'>",
    "resolution_notes": "<How you resolved disagreements, including escalation patterns detected>",
    "decision_rationale": "<2-3 sentence concrete rationale for final decision>"
  }
}

SCORING:
1=Very Poor/Strong Reject  2=Below Average/Reject  3=Borderline
4=Good/Accept  5=Excellent/Strong Accept

CONFIDENCE_LEVEL:   1=very uncertain  3=moderate  5=very confident
DISAGREEMENT_LEVEL: 1=mostly agree    3=moderate  5=severe disagreement

If disagreement_level >= 4 OR confidence_level <= 2 → needs_additional_discussion = true
"""


class AreaChairExecutor(AgentExecutor):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        print("\n[AreaChairExecutor] INIT")
        print("[AreaChairExecutor] OPENAI_API_KEY exists?:", "YES" if api_key else "NO")
        if not api_key:
            print("[AreaChairExecutor] WARNING: OPENAI_API_KEY is missing!")

        self.client = OpenAI(api_key=api_key)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("\n==============================")
        print("[AreaChairExecutor] execute() CALLED")

        try:
            user_input = context.get_user_input()
            model_name = os.getenv("JUDGE_MODEL", "gpt-4o").strip()

            print("[AreaChairExecutor] model_name:", model_name)
            print("[AreaChairExecutor] user_input length:", len(user_input))
            print("[AreaChairExecutor] user_input preview:", user_input[:400])

            response = self.client.chat.completions.create(
                model=model_name,
                max_tokens=3500,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_input},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip() if response.choices else ""
            print("[AreaChairExecutor] raw length:", len(raw))
            print("[AreaChairExecutor] raw preview:", raw[:500])

            # Clean and validate JSON
            try:
                clean = raw
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                    clean = clean.strip()
                if clean.endswith("```"):
                    clean = clean[:-3].strip()

                parsed = json.loads(clean)
                output = json.dumps(parsed, ensure_ascii=False, indent=2)
                print("[AreaChairExecutor] JSON parsed OK.")

            except json.JSONDecodeError as e:
                print("[AreaChairExecutor] JSON parse FAILED:", str(e))
                output = json.dumps({
                    "error": "invalid_json_from_model",
                    "json_error": str(e),
                    "raw_output": raw,
                }, ensure_ascii=False, indent=2)

            await event_queue.enqueue_event(new_agent_text_message(output))
            print("[AreaChairExecutor] Event enqueued OK.")

        except Exception as e:
            print("[AreaChairExecutor] FATAL ERROR:", str(e))
            print(traceback.format_exc())

            fail_payload = json.dumps({
                "error": "area_chair_executor_exception",
                "exception": str(e),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2)

            await event_queue.enqueue_event(new_agent_text_message(fail_payload))

        print("[AreaChairExecutor] execute() DONE")
        print("==============================\n")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("[AreaChairExecutor] cancel() called")
        pass