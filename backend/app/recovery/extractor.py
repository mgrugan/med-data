"""Structure doctor-provided discharge instructions WITHOUT inventing anything.

Two extractors produce the same schema:
* ``rule_based_extract`` - deterministic; every field is a literal span of the input.
* ``llm_extract`` - Claude structures the text (used only when ANTHROPIC_API_KEY is set).

Both pass through ``verify_grounding``: every item must quote the source verbatim, and
every dose / frequency / duration / date value must be traceable to the quoted text.
Anything that cannot be traced is replaced with "Not specified" and reported as a
warning. Missing information always stays "Not specified".
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

NS = "Not specified"
CATEGORIES = ["medications", "activity", "restrictions", "wound_care", "nutrition", "appointments", "monitoring"]
FIELDS = ["name", "dose", "frequency", "timing", "duration"]

_KEYWORDS = {
    "restrictions": [r"\bdo not\b", r"\bdon'?t\b", r"\bavoid\b", r"\bno (?:heavy )?lifting\b", r"\bnot (?:to )?lift\b",
                     r"\brefrain\b", r"\bshould not\b", r"\bmust not\b", r"\bno driving\b", r"\bno (?:baths?|swimming)\b",
                     r"\bnot drive\b", r"\bnever\b", r"\bstop taking\b"],
    "appointments": [r"\bappointment\b", r"\bfollow[- ]?up\b", r"\bsee (?:dr|doctor|your)\b", r"\bclinic\b",
                     r"\bvisit\b", r"\bcheck[- ]?up\b", r"\bschedule(?:d)?\b", r"\bsurgeon\b.*\b(?:in|on)\b"],
    "wound_care": [r"\bwound\b", r"\bdressing\b", r"\bincision\b", r"\bbandage\b", r"\bstitch", r"\bsuture",
                   r"\bstaples?\b", r"\bdrain\b", r"\bsteri-?strips?\b", r"\bgauze\b"],
    "medications": [r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|units?|iu)\b", r"\btablets?\b", r"\bcapsules?\b",
                    r"\bpills?\b", r"\binhaler\b", r"\bpuffs?\b", r"\bantibiotics?\b", r"\bmedications?\b",
                    r"\bmedicine\b", r"\binjections?\b", r"\btake\b"],
    "monitoring": [r"\bmonitor\b", r"\bcheck (?:your|blood|temperature|weight)\b", r"\bmeasure\b", r"\brecord\b",
                   r"\bcall (?:your|the|911|if)\b", r"\bwatch for\b", r"\btemperature\b", r"\bblood pressure\b",
                   r"\bglucose\b", r"\bblood sugar\b", r"\bweigh\b", r"\bsigns of\b", r"\bseek\b", r"\bfever\b"],
    "nutrition": [r"\bdiet\b", r"\beat\b", r"\bdrink\b", r"\bfluids?\b", r"\bfood\b", r"\bmeals?\b", r"\bsalt\b",
                  r"\bsodium\b", r"\bsugar\b", r"\bwater\b", r"\balcohol\b", r"\bcaffeine\b", r"\bfibre\b", r"\bfiber\b"],
    "activity": [r"\bwalk", r"\bexercis", r"\bactivit", r"\bphysical therapy\b", r"\bphysiotherapy\b", r"\bstairs\b",
                 r"\brest\b", r"\bstretch", r"\bdriv", r"\breturn to work\b", r"\bshower\b"],
}
# priority when a sentence matches several categories
_PRIORITY = ["restrictions", "appointments", "wound_care", "medications", "monitoring", "nutrition", "activity"]

_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|units?|iu|IU|tablets?|capsules?|puffs?|drops?)\b"
                   r"(?:\s?(?:/|per)\s?\w+)?", re.I)
_FREQ = re.compile(
    r"\bevery\s+\d+(?:\s*(?:-|to)\s*\d+)?\s*(?:hours?|hrs?|h)\b|\bevery\s+(?:other\s+)?(?:day|morning|evening|night)\b|"
    r"\b(?:once|twice|three times|four times|\d+\s*times)\s+(?:a|per|each)?\s*(?:day|daily|week|weekly)\b|"
    r"\b(?:once|twice)\s+daily\b|\b(?:three|four|\d+)\s+times\s+daily\b|\bdaily\b|\bweekly\b|\bnightly\b|"
    r"\b(?:bid|tid|qid|qd|qhs|prn|q\d+h)\b|\bas needed\b", re.I)
_TIMING = re.compile(r"\b(?:in the (?:morning|evening|afternoon)|at (?:bedtime|night|noon)|before bed|"
                     r"with (?:food|meals?|breakfast|lunch|dinner)|before (?:meals?|breakfast)|after (?:meals?|dinner)|"
                     r"on an empty stomach|at \d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.))", re.I)
_DUR = re.compile(r"\bfor\s+(?:the\s+)?(?:next\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                  r"eleven|twelve|fourteen|twenty|thirty)\s+(?:more\s+)?(?:days?|weeks?|months?)\b|"
                  r"\buntil\s+[^,.;]+|\bfor\s+(?:at least\s+)?\d+\s+(?:days?|weeks?)\b", re.I)
_WHEN = re.compile(r"\b(?:in|within)\s+(?:\d+|one|two|three|four|five|six|seven|ten|fourteen)\s+(?:days?|weeks?)\b|"
                   r"\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                   r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}|\d{4}-\d{2}-\d{2})", re.I)
_DRUG = re.compile(r"\b(?:take|start|continue|use)\s+([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+)?)|"
                   r"\b([A-Za-z][a-zA-Z\-]{2,})\s+\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?)\b", re.I)
_NOT_DRUGS = {"take", "the", "your", "one", "two", "of", "and", "a", "an", "to", "it", "this", "all", "with", "any"}


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    # split on line breaks / bullets and on sentence ends, but not after common abbreviations
    parts = re.split(r"(?:\n\s*(?:[-•*]|\d+[.)])?\s*)|(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bSt\.)"
                     r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?;])\s+(?=[A-Z0-9])", text)
    parts = [re.sub(r"^\d+[.)]\s+", "", p.strip(" -•*\t")) for p in parts if p]
    return [p for p in parts if len(p) > 2]


def _first(rx: re.Pattern, s: str) -> str:
    m = rx.search(s)
    return m.group(0).strip() if m else NS


def classify(sentence: str) -> str | None:
    s = sentence.lower()
    hits = [c for c in _PRIORITY if any(re.search(k, s) for k in _KEYWORDS[c])]
    if not hits:
        return None
    # "take ... mg" style sentences are medications even if they also mention food or monitoring
    if "medications" in hits and _DOSE.search(sentence) and "restrictions" not in hits:
        return "medications"
    return hits[0]


def _drug_name(sentence: str) -> str:
    for m in _DRUG.finditer(sentence):
        name = (m.group(1) or m.group(2) or "").strip()
        if name and name.split()[0].lower() not in _NOT_DRUGS:
            return name
    return NS


def empty_plan() -> dict[str, list]:
    return {c: [] for c in CATEGORIES}


def rule_based_extract(text: str) -> dict[str, Any]:
    plan = empty_plan()
    unclassified = []
    for s in split_sentences(text):
        cat = classify(s)
        if cat is None:
            unclassified.append(s)
            continue
        item = {"instruction": s, "source_quote": s, "name": NS, "dose": NS,
                "frequency": _first(_FREQ, s), "timing": _first(_TIMING, s), "duration": _first(_DUR, s)}
        if cat == "medications":
            item["name"], item["dose"] = _drug_name(s), _first(_DOSE, s)
        if cat == "appointments":
            item["timing"] = _first(_WHEN, s) if _first(_WHEN, s) != NS else item["timing"]
        plan[cat].append(item)
    return {"plan": plan, "unclassified": unclassified, "extractor": "rule-based"}


# ------------------------------------------------------------------ LLM extraction
_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["instruction", "source_quote"] + FIELDS,
    "properties": {
        "instruction": {"type": "string"}, "source_quote": {"type": "string"},
        **{f: {"type": "string"} for f in FIELDS},
    },
}
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": CATEGORIES + ["unclassified"],
    "properties": {**{c: {"type": "array", "items": _ITEM_SCHEMA} for c in CATEGORIES},
                   "unclassified": {"type": "array", "items": {"type": "string"}}},
}
SYSTEM = f"""You structure discharge instructions that a doctor wrote. You are a transcriber, not a clinician.

Rules:
- Use ONLY information stated in the provided text. Never add, infer, complete or "standardise" medications, doses,
  frequencies, durations, restrictions, wound-care steps, exercises, diet advice or appointments.
- Every item must include "source_quote": an exact, verbatim, contiguous copy of the sentence or phrase it came from.
- For any field the text does not state explicitly, write exactly "{NS}". Do not guess typical values.
- "instruction" restates the quoted text faithfully in plain language without adding content.
- Put text you cannot place in a category into "unclassified" verbatim.
- Categories: medications, activity, restrictions (things not to do), wound_care, nutrition, appointments, monitoring
  (what to watch for, measurements, when to call for help)."""


def llm_extract(text: str) -> dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=os.environ.get("RECOVERYAI_LLM_MODEL", "claude-opus-5-5"),
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<discharge_instructions>\n{text}\n</discharge_instructions>"}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("model output was truncated")
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    data = json.loads(raw)
    return {"plan": {c: data.get(c, []) for c in CATEGORIES}, "unclassified": data.get("unclassified", []),
            "extractor": f"llm:{resp.model}"}


# ------------------------------------------------------------------ grounding guard
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def verify_grounding(result: dict[str, Any], text: str) -> dict[str, Any]:
    src = _norm(text)
    warnings, plan = [], empty_plan()
    for cat in CATEGORIES:
        for item in result["plan"].get(cat, []):
            quote = _norm(item.get("source_quote", ""))
            if not quote or quote not in src:
                warnings.append(f"Dropped a {cat} item whose quote was not found in the text: "
                                f"{item.get('instruction', '')[:80]!r}")
                continue
            clean = {"instruction": item.get("instruction") or item["source_quote"], "source_quote": item["source_quote"]}
            for f in FIELDS:
                v = (item.get(f) or NS).strip()
                if v != NS:
                    nv = _norm(v)
                    grounded = nv in quote or (_numbers(v) and _numbers(v) <= _numbers(quote) and
                                               all(w in quote for w in re.findall(r"[a-z]{4,}", nv)[:3]))
                    if not grounded:
                        warnings.append(f"Removed unsupported {f} {v!r} for {cat} item (not in the quoted text).")
                        v = NS
                clean[f] = v
            # instruction text must not introduce numbers absent from the quote
            if _numbers(clean["instruction"]) - _numbers(quote):
                clean["instruction"] = item["source_quote"]
            plan[cat].append(clean)
    return {**result, "plan": plan, "grounding_warnings": warnings}


def extract(text: str, use_llm: bool | None = None) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"plan": empty_plan(), "unclassified": [], "extractor": "none", "grounding_warnings": []}
    want_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) if use_llm is None else use_llm
    note = None
    if want_llm:
        try:
            result = llm_extract(text)
        except Exception as e:  # network/auth/refusal -> deterministic fallback
            note = f"LLM structuring unavailable ({type(e).__name__}); used the rule-based extractor."
            result = rule_based_extract(text)
    else:
        result = rule_based_extract(text)
    out = verify_grounding(result, text)
    out["note"] = note
    out["counts"] = {c: len(out["plan"][c]) for c in CATEGORIES}
    out["not_specified_categories"] = [c for c in CATEGORIES if not out["plan"][c]]
    return out
