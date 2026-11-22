"""Turn a structured, doctor-derived plan into a day-by-day recovery schedule.

The scheduler only *arranges* what the instructions say:
* "every 8 hours" -> recurring clock times starting at the wake time
* "twice daily" / "BID" -> spread across waking hours
* "in the morning" / "at bedtime" / "with meals" -> the matching anchor time
* "for 14 days" -> applied on calendar days 1..14 from the start date
* Restrictions with no time -> "All day"
* Anything without a stated frequency -> "As directed" (never given an invented time)
* No stated duration -> shown in the chosen viewing window, labelled "duration not specified"
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any

from .extractor import CATEGORIES, NS

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "twenty": 20, "thirty": 30,
            "once": 1, "twice": 2}
_ABBR = {"qd": 1, "bid": 2, "tid": 3, "qid": 4, "qhs": 1}
LABELS = {"medications": "Medication", "activity": "Activity", "restrictions": "Restriction",
          "wound_care": "Wound care", "nutrition": "Nutrition", "appointments": "Appointment",
          "monitoring": "Monitoring"}


def _num(tok: str) -> int | None:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else _WORDNUM.get(tok)


def parse_interval_hours(freq: str) -> float | None:
    m = re.search(r"every\s+(\d+)(?:\s*(?:-|to)\s*(\d+))?\s*(?:hours?|hrs?|h)\b", freq, re.I)
    if m:
        return float(m.group(1))  # for "every 4-6 hours" use the stated lower bound; shown with the range text
    m = re.search(r"\bq(\d+)h\b", freq, re.I)
    return float(m.group(1)) if m else None


def parse_times_per_day(freq: str) -> int | None:
    f = freq.lower()
    for k, v in _ABBR.items():
        if re.search(rf"\b{k}\b", f):
            return v
    m = re.search(r"\b(once|twice|one|two|three|four|\d+)\s*(?:times)?\s*(?:a|per|each)?\s*(?:day|daily)\b", f)
    if m:
        return _num(m.group(1))
    if re.search(r"\b(daily|every day|nightly|every (?:morning|evening|night))\b", f):
        return 1
    return None


def parse_duration_days(dur: str) -> int | None:
    m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|twenty|thirty)\s+"
                  r"(?:more\s+)?(day|week|month)s?", dur, re.I)
    if not m:
        return None
    n = _num(m.group(1))
    unit = m.group(2).lower()
    return n * {"day": 1, "week": 7, "month": 30}[unit] if n else None


def parse_appointment_offset(text: str, start: date) -> date | None:
    m = re.search(r"\b(?:in|within)\s+(\d+|one|two|three|four|five|six|seven|ten|fourteen)\s+(day|week)s?", text, re.I)
    if m and _num(m.group(1)):
        return start + timedelta(days=_num(m.group(1)) * (7 if m.group(2).lower() == "week" else 1))
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return date.fromisoformat(m.group(1))
    return None


def _anchor_times(timing: str, wake: time, bed: time, meals: dict[str, time]) -> list[time]:
    t = timing.lower()
    if "bed" in t or "night" in t:
        return [bed]
    if "morning" in t or "breakfast" in t:
        return [meals["breakfast"] if "breakfast" in t else wake]
    if "noon" in t or "lunch" in t:
        return [meals["lunch"]]
    if "evening" in t or "dinner" in t:
        return [meals["dinner"]]
    if "meal" in t or "food" in t:
        return [meals["breakfast"], meals["lunch"], meals["dinner"]]
    m = re.search(r"at (\d{1,2})(?::(\d{2}))?\s?(am|pm|a\.m\.|p\.m\.)", t)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
        return [time(h, int(m.group(2) or 0))]
    return []


def _spread(n: int, wake: time, bed: time) -> list[time]:
    if n <= 1:
        return [wake]
    start = wake.hour * 60 + wake.minute
    end = bed.hour * 60 + bed.minute - 60  # last dose an hour before bed
    if end <= start:
        end = start + 12 * 60
    step = (end - start) / (n - 1)
    return [time(int((start + i * step) // 60) % 24, int((start + i * step) % 60)) for i in range(n)]


def _fmt(t: time) -> str:
    return datetime.combine(date.today(), t).strftime("%-I:%M %p")


def item_times(item: dict, wake: time, bed: time, meals: dict[str, time]) -> tuple[list[time], str]:
    """Return (clock times, how they were derived)."""
    freq, timing = item.get("frequency", NS), item.get("timing", NS)
    quote = item.get("source_quote", "")
    if re.search(r"\bas needed\b|\bprn\b|\bif needed\b|\bwhen needed\b", f"{freq} {quote}", re.I):
        return [], "as needed (no fixed times; the stated frequency is a maximum, not a schedule)"
    if freq != NS:
        h = parse_interval_hours(freq)
        if h:
            n = max(1, int(24 // h))
            base = wake.hour * 60 + wake.minute
            return ([time(int((base + i * h * 60) // 60) % 24, int((base + i * h * 60) % 60)) for i in range(n)],
                    f"'{freq}' -> every {h:g} h starting at wake-up")
        n = parse_times_per_day(freq)
        if n:
            anchors = _anchor_times(timing, wake, bed, meals) if timing != NS else []
            if anchors and len(anchors) == n:
                return anchors, f"'{freq}' + '{timing}'"
            return _spread(n, wake, bed), f"'{freq}' -> {n}x spread across waking hours"
    if timing != NS:
        anchors = _anchor_times(timing, wake, bed, meals)
        if anchors:
            return anchors, f"'{timing}'"
    return [], "no time stated"


def build_schedule(plan: dict[str, list], start: date, view_days: int = 7, wake: time = time(8, 0),
                   bed: time = time(22, 0), meals: dict[str, time] | None = None) -> dict[str, Any]:
    meals = meals or {"breakfast": time(8, 0), "lunch": time(12, 30), "dinner": time(18, 0)}
    view_days = max(1, min(int(view_days), 90))
    days = [{"date": (start + timedelta(days=i)).isoformat(), "day": i + 1, "timed": [], "all_day": [],
             "as_directed": [], "appointments": []} for i in range(view_days)]
    undated_appointments, notes = [], []
    for cat in CATEGORIES:
        for item in plan.get(cat, []):
            dur_days = parse_duration_days(item.get("duration", NS)) if item.get("duration", NS) != NS else None
            base = {"category": cat, "label": LABELS[cat], "instruction": item["instruction"],
                    "source_quote": item["source_quote"], "dose": item.get("dose", NS), "name": item.get("name", NS),
                    "duration": item.get("duration", NS),
                    "duration_note": f"Day 1-{dur_days} (as written)" if dur_days else
                    ("Until: " + item["duration"] if item.get("duration", NS) != NS else "Duration not specified")}
            if cat == "appointments":
                when = parse_appointment_offset(item["source_quote"], start)
                if when and 0 <= (when - start).days < view_days:
                    days[(when - start).days]["appointments"].append(base)
                elif when:
                    undated_appointments.append({**base, "date": when.isoformat()})
                else:
                    undated_appointments.append({**base, "date": NS})
                continue
            times, how = item_times(item, wake, bed, meals)
            for d in days:
                if dur_days and d["day"] > dur_days:
                    continue
                if times:
                    for t in times:
                        d["timed"].append({**base, "time": t.strftime("%H:%M"), "display_time": _fmt(t), "derived": how})
                elif cat == "restrictions":
                    d["all_day"].append(base)
                else:
                    d["as_directed"].append({**base, "derived": how})
            if not dur_days and item.get("duration", NS) == NS:
                notes.append(f"No duration was given for: {item['source_quote'][:90]!r}. It is shown for the "
                             f"{view_days}-day viewing window only. Follow your doctor's instructions on when to stop.")
    for d in days:
        d["timed"].sort(key=lambda e: e["time"])
    return {"start_date": start.isoformat(), "view_days": view_days, "wake": wake.strftime("%H:%M"),
            "bed": bed.strftime("%H:%M"), "days": days, "appointments_outside_window": undated_appointments,
            "notes": notes,
            "disclaimer": ("This schedule only rearranges the instructions you supplied. It does not add, change or "
                           "remove medical advice. If anything is unclear, ask your doctor or pharmacist.")}
