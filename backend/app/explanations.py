"""Plain-language explanations of the information present in the dataset.

The dataset gives only a categorical test result (Normal / Abnormal / Inconclusive). It
does not say which test was performed and contains no laboratory values, so none are
invented here.
"""
from __future__ import annotations

DATASET_NOTE = ("This synthetic dataset records only a test-result category. It does not say which test was "
                "done, and it contains no measured values, reference ranges or dates for the test.")

TEST_RESULTS = {
    "Normal": {
        "title": "Normal",
        "summary": "The recorded result was within the range expected for the test.",
        "detail": ("In general, a normal result means the measurement fell inside the reference range the lab uses. "
                   "A normal result does not rule out every problem, and your care team reads results in the context "
                   "of your symptoms and history."),
        "ask": ["Which test was this, and what did it check for?", "Does this result change my care plan?"],
    },
    "Abnormal": {
        "title": "Abnormal",
        "summary": "The recorded result was outside the range expected for the test.",
        "detail": ("In general, an abnormal result means a value was higher or lower than the reference range, or a "
                   "finding was present. Abnormal does not automatically mean serious: results can be slightly out of "
                   "range for many harmless reasons, and the significance depends on the specific test."),
        "ask": ["Which test was abnormal and by how much?", "Is follow-up testing needed?",
                "Is this related to the reason I was admitted?"],
    },
    "Inconclusive": {
        "title": "Inconclusive",
        "summary": "The result could not be clearly interpreted as normal or abnormal.",
        "detail": ("In general, an inconclusive result means the test did not give a clear answer. Reasons include "
                   "sample-quality issues, borderline values, or interference. A repeat or different test is often "
                   "used to clarify."),
        "ask": ["Will the test be repeated?", "What could make the result unclear?"],
    },
}

CONDITIONS = {
    "Arthritis": "A group of conditions causing joint pain, swelling or stiffness.",
    "Asthma": "A long-term condition in which the airways can narrow and swell, making breathing harder.",
    "Cancer": "A broad group of diseases in which abnormal cells grow uncontrollably. Types differ greatly.",
    "Diabetes": "A condition affecting how the body regulates blood sugar (glucose).",
    "Hypertension": "Persistently high blood pressure.",
    "Obesity": "Excess body weight to a degree that may affect health.",
}

MEDICATIONS = {
    "Aspirin": "A common pain reliever and anti-inflammatory that also reduces blood clotting.",
    "Ibuprofen": "A non-steroidal anti-inflammatory drug (NSAID) used for pain, fever and inflammation.",
    "Lipitor": "Atorvastatin, a statin medicine used to lower cholesterol.",
    "Paracetamol": "Acetaminophen, a common pain and fever reliever.",
    "Penicillin": "An antibiotic used to treat certain bacterial infections.",
}

ADMISSION_TYPES = {
    "Elective": "A planned admission, scheduled in advance.",
    "Urgent": "An admission needed soon, but not an immediate emergency.",
    "Emergency": "An unplanned admission needing immediate care.",
}

SYNTHETIC_PAIRING_NOTE = ("Condition, medication and test result are assigned independently at random in this "
                          "synthetic dataset, so combinations may not make clinical sense (for example, an "
                          "antibiotic recorded for hypertension).")


def explain_patient(condition: str, medication: str, test_result: str, admission_type: str) -> dict:
    return {
        "test_result": {**TEST_RESULTS.get(test_result, {}), "dataset_note": DATASET_NOTE},
        "condition": {"name": condition, "summary": CONDITIONS.get(condition, "")},
        "medication": {"name": medication, "summary": MEDICATIONS.get(medication, "")},
        "admission_type": {"name": admission_type, "summary": ADMISSION_TYPES.get(admission_type, "")},
        "synthetic_note": SYNTHETIC_PAIRING_NOTE,
        "disclaimer": "General educational information only. It is not a diagnosis or medical advice.",
    }
