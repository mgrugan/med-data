"""RecoveryAI API (FastAPI).

Educational / research prototype trained on SYNTHETIC data. Nothing returned by this API
is a clinical discharge decision.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import date, time
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .explanations import explain_patient
from .ml.features import CATEGORY_LEVELS
from .ml.predictor import days_between, get_predictor, get_report
from .recovery.extractor import CATEGORIES, extract
from .recovery.schedule import build_schedule

DISCLAIMER = ("RecoveryAI is an educational research prototype trained on a synthetic dataset. Its outputs are not "
              "medical advice and must never be used as a clinical discharge decision.")

app = FastAPI(title="RecoveryAI API", version="1.0.0", description=DISCLAIMER)
origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _lit(col):
    return Literal[tuple(CATEGORY_LEVELS[col])]  # type: ignore[valid-type]


class PatientIn(BaseModel):
    age: int = Field(ge=0, le=110)
    gender: _lit("Gender")
    blood_type: _lit("Blood Type")
    medical_condition: _lit("Medical Condition")
    admission_type: _lit("Admission Type")
    medication: _lit("Medication")
    test_result: _lit("Test Results")
    admission_date: date | None = None
    today: date | None = None
    days_hospitalized: float | None = Field(default=None, ge=0, le=365)
    model: str | None = None
    level: float = Field(default=0.9, ge=0.5, le=0.95)


class ExtractIn(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    use_llm: bool | None = None


class Item(BaseModel):
    instruction: str
    source_quote: str
    name: str = "Not specified"
    dose: str = "Not specified"
    frequency: str = "Not specified"
    timing: str = "Not specified"
    duration: str = "Not specified"


class ScheduleIn(BaseModel):
    plan: dict[str, list[Item]]
    start_date: date | None = None
    view_days: int = Field(default=7, ge=1, le=90)
    wake_time: str = "08:00"
    bed_time: str = "22:00"


def _t(s: str) -> time:
    if not re.fullmatch(r"\d{1,2}:\d{2}", s):
        raise HTTPException(422, f"invalid time {s!r}; use HH:MM")
    h, m = map(int, s.split(":"))
    return time(h % 24, m % 60)


@app.get("/", include_in_schema=False)
def root():
    """The API has no UI; point visitors to the web app and the interactive docs."""
    return {"service": "RecoveryAI API", "status": "ok",
            "message": "This is the backend API. Open the recoveryai-web service URL for the app.",
            "docs": "/docs", "health": "/api/health", "disclaimer": DISCLAIMER}


@app.get("/api/health")
def health():
    p = get_predictor()
    return {"status": "ok", "models": list(p.entries), "llm_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "disclaimer": DISCLAIMER}


@app.get("/api/options")
def options():
    return {**get_predictor().options(), "disclaimer": DISCLAIMER}


@app.post("/api/predict")
def predict(body: PatientIn):
    p = get_predictor()
    if body.model and body.model not in p.entries:
        raise HTTPException(404, f"unknown model {body.model!r}")
    data = body.model_dump()
    if data["days_hospitalized"] is None:
        data["days_hospitalized"] = (days_between(body.admission_date, body.today or date.today())
                                     if body.admission_date else 0.0)
    out = p.predict(data, model=body.model, level=body.level)
    out["understanding"] = explain_patient(body.medical_condition, body.medication, body.test_result,
                                           body.admission_type)
    out["language"] = ("The model estimates that discharge may occur approximately within this range. This is a "
                       "statistical estimate from synthetic data, not a discharge date.")
    out["disclaimer"] = DISCLAIMER
    return out


@app.get("/api/validation")
def validation():
    r = get_report()
    if not r:
        raise HTTPException(503, "validation report not generated yet; run `python -m training.run_pipeline`")
    return r


@app.post("/api/recovery/extract")
def recovery_extract(body: ExtractIn):
    return extract(body.text, body.use_llm)


def _read_upload(name: str, raw: bytes) -> str:
    name = name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(raw)).pages)
    if name.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            xml = z.read("word/document.xml").decode("utf8", "ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    if name.endswith((".txt", ".md", ".text", ".rtf")) or "." not in name:
        return raw.decode("utf8", "ignore")
    raise HTTPException(415, "Upload a .txt, .md, .pdf or .docx file")


@app.post("/api/recovery/upload")
async def recovery_upload(file: UploadFile = File(...), use_llm: bool | None = None):
    raw = await file.read()
    if len(raw) > 5_000_000:
        raise HTTPException(413, "file too large (5 MB max)")
    text = _read_upload(file.filename or "", raw).strip()
    if not text:
        raise HTTPException(422, "no readable text found in the file (scanned images are not supported)")
    return {**extract(text, use_llm), "text": text[:50_000]}


@app.post("/api/recovery/schedule")
def recovery_schedule(body: ScheduleIn):
    plan = {c: [i.model_dump() for i in body.plan.get(c, [])] for c in CATEGORIES}
    return build_schedule(plan, body.start_date or date.today(), body.view_days, _t(body.wake_time), _t(body.bed_time))
