"""Plain-language explanations of the information present in the dataset.

The dataset records HbA1c and maximum serum glucose only as ranges (or "not tested").
Exact values are not available and none are invented here.
"""
from __future__ import annotations

DATASET_NOTE = ("The dataset records these tests only as ranges (for example \"above 8%\"), not exact values, "
                "and does not record when during the stay they were taken.")

A1C = {
    "Normal": {"title": "HbA1c: normal (below 7%)",
               "summary": "HbA1c reflects average blood sugar over roughly the past 2–3 months.",
               "detail": "A result below 7% is in the range commonly targeted for many adults with diabetes."},
    ">7%": {"title": "HbA1c: above 7% (7–8%)",
            "summary": "HbA1c reflects average blood sugar over roughly the past 2–3 months.",
            "detail": ("A result between 7% and 8% is above the level commonly targeted for many adults with "
                       "diabetes, suggesting blood sugar has been somewhat high recently.")},
    ">8%": {"title": "HbA1c: above 8%",
            "summary": "HbA1c reflects average blood sugar over roughly the past 2–3 months.",
            "detail": ("A result above 8% suggests blood sugar has been high over recent months. Care teams often "
                       "review diabetes treatment when HbA1c is in this range.")},
    "Not tested": {"title": "HbA1c: not tested",
                   "summary": "No HbA1c result was recorded for this stay.",
                   "detail": "This is common; most encounters in this dataset have no HbA1c recorded."},
}

GLUCOSE = {
    "Normal": {"title": "Blood glucose: normal",
               "detail": "The highest blood-glucose reading recorded was within the normal range."},
    ">200": {"title": "Blood glucose: above 200 mg/dL",
             "detail": "The highest blood-glucose reading recorded was above 200 mg/dL, which is high."},
    ">300": {"title": "Blood glucose: above 300 mg/dL",
             "detail": "The highest blood-glucose reading recorded was above 300 mg/dL, which is very high."},
    "Not tested": {"title": "Blood glucose: not tested",
                   "detail": "No serum-glucose maximum was recorded for this stay."},
}

QUESTIONS = ["What do my HbA1c and glucose results mean for me?",
             "Will my diabetes medicines change after I go home?",
             "How often should I check my blood sugar, and what numbers should prompt a call?"]

CONDITIONS = {
    "Circulatory": "Heart and blood-vessel conditions, such as heart failure, heart attack or stroke.",
    "Respiratory": "Lung and breathing conditions, such as pneumonia or COPD.",
    "Digestive": "Conditions of the stomach, bowel, liver or pancreas.",
    "Diabetes": "The main reason for admission was diabetes itself, or one of its complications.",
    "Injury": "Injuries and poisonings, such as fractures.",
    "Musculoskeletal": "Bone, joint and muscle conditions, such as arthritis or back problems.",
    "Genitourinary": "Kidney, bladder and reproductive-system conditions.",
    "Neoplasms": "Tumours, cancerous or non-cancerous.",
    "Other": "Other primary diagnoses not in the groups above.",
}

MEDICATIONS = {
    "Insulin only": "Insulin was part of diabetes treatment during the stay; no oral diabetes medicine was recorded.",
    "Oral agents only": "Oral diabetes medicines (such as metformin) were recorded, but no insulin.",
    "Insulin + oral": "Both insulin and at least one oral diabetes medicine were recorded.",
    "None": "No diabetes medicine was recorded for this stay.",
}

ADMISSION_TYPES = {
    "Elective": "A planned admission, scheduled in advance.",
    "Urgent": "An admission needed soon, but not an immediate emergency.",
    "Emergency": "An unplanned admission needing immediate care.",
    "Other/Unknown": "The admission type was not recorded.",
}


def explain_patient(condition: str, medication: str, a1c: str, glucose: str, admission_type: str) -> dict:
    return {
        "a1c": {**A1C.get(a1c, {}), "value": a1c},
        "glucose": {**GLUCOSE.get(glucose, {}), "value": glucose},
        "test_note": DATASET_NOTE,
        "ask": QUESTIONS,
        "condition": {"name": condition, "summary": CONDITIONS.get(condition, "")},
        "medication": {"name": medication, "summary": MEDICATIONS.get(medication, "")},
        "admission_type": {"name": admission_type, "summary": ADMISSION_TYPES.get(admission_type, "")},
        "disclaimer": "General educational information only. It is not a diagnosis or medical advice.",
    }
