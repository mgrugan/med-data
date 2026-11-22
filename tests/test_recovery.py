from datetime import date

from backend.app.recovery.extractor import NS, empty_plan, extract, rule_based_extract, verify_grounding
from backend.app.recovery.schedule import build_schedule, parse_duration_days, parse_interval_hours, parse_times_per_day

TEXT = """Take Amoxicillin 500 mg every 8 hours for 14 days.
Take acetaminophen 650 mg twice daily as needed for pain.
Do not lift more than 10 pounds.
Change the dressing once daily.
Follow up with Dr. Patel in 2 weeks."""


def _all_items(plan):
    return [i for v in plan.values() for i in v]


def test_rule_based_never_invents_values():
    out = extract(TEXT, use_llm=False)
    src = TEXT.lower()
    for item in _all_items(out["plan"]):
        assert item["source_quote"].lower() in src
        for f in ["name", "dose", "frequency", "timing", "duration"]:
            assert item[f] == NS or item[f].lower() in item["source_quote"].lower()


def test_missing_information_stays_not_specified():
    out = extract("Do not lift more than 10 pounds.", use_llm=False)
    r = out["plan"]["restrictions"][0]
    assert r["duration"] == NS and r["frequency"] == NS
    assert set(out["not_specified_categories"]) >= {"medications", "nutrition", "appointments"}


def test_grounding_removes_hallucinated_fields_and_items():
    fake = {"plan": {**empty_plan(), "medications": [
        {"instruction": "Take Amoxicillin 875 mg", "source_quote": "Take Amoxicillin 500 mg every 8 hours for 14 days.",
         "name": "Amoxicillin", "dose": "875 mg", "frequency": "every 8 hours", "timing": NS, "duration": "for 21 days"},
        {"instruction": "Take metoprolol 25 mg", "source_quote": "Take metoprolol 25 mg daily", "name": "metoprolol",
         "dose": "25 mg", "frequency": "daily", "timing": NS, "duration": NS}]}, "unclassified": []}
    g = verify_grounding(fake, TEXT)
    meds = g["plan"]["medications"]
    assert len(meds) == 1
    assert meds[0]["dose"] == NS and meds[0]["duration"] == NS and meds[0]["frequency"] == "every 8 hours"
    assert "875" not in meds[0]["instruction"]
    assert len(g["grounding_warnings"]) == 3


def test_warning_signs_are_monitoring():
    out = extract("Call your doctor if your temperature is above 101 F or the wound becomes red.", use_llm=False)
    assert out["plan"]["monitoring"] and not out["plan"]["wound_care"]


def test_parsers():
    assert parse_interval_hours("every 8 hours") == 8
    assert parse_times_per_day("twice daily") == 2 and parse_times_per_day("TID") == 3
    assert parse_duration_days("for 14 days") == 14 and parse_duration_days("for 2 weeks") == 14


def test_schedule_every_8_hours_for_14_days_only():
    plan = rule_based_extract(TEXT)["plan"]
    s = build_schedule(plan, date(2026, 1, 1), view_days=20)
    amox = lambda d: [e for e in d["timed"] if "Amoxicillin" in e["instruction"]]
    assert [e["time"] for e in amox(s["days"][0])] == ["00:00", "08:00", "16:00"]
    assert len(amox(s["days"][13])) == 3 and len(amox(s["days"][14])) == 0


def test_as_needed_is_never_given_fixed_times():
    s = build_schedule(rule_based_extract(TEXT)["plan"], date(2026, 1, 1), view_days=3)
    day = s["days"][0]
    assert not any("acetaminophen" in e["instruction"] for e in day["timed"])
    assert any("acetaminophen" in e["instruction"] for e in day["as_directed"])


def test_restriction_all_day_and_appointment_placed():
    s = build_schedule(rule_based_extract(TEXT)["plan"], date(2026, 1, 1), view_days=20)
    assert any("lift" in e["instruction"] for e in s["days"][0]["all_day"])
    assert s["days"][14]["appointments"] and s["days"][14]["date"] == "2026-01-15"
