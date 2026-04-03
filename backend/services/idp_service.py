"""Intelligent Document Parsing — stdlib only."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple
from backend.models.schemas import (
    HospitalRecord, HospitalCapabilities, HospitalQualityMetrics, ParseResponse
)

_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

_CAPS = {
    "icu":                ["icu", "intensive care", "critical care"],
    "surgery":            ["surgery", "surgical", "operating room"],
    "maternity":          ["maternity", "obstetric", "labor", "delivery"],
    "pediatrics":         ["pediatric", "children", "neonatal"],
    "rehabilitation":     ["rehab", "rehabilitation"],
    "mental_health":      ["psychiatr", "mental health", "behavioral"],
    "cardiac_care":       ["cardiac", "cardio", "heart"],
    "oncology":           ["oncol", "cancer", "chemotherapy"],
    "trauma_center":      ["trauma", "level i", "level 1"],
    "emergency_services": ["emergency", "er ", "urgent care"],
}


def _name(text: str) -> Optional[str]:
    for p in [
        r"([A-Z][A-Za-z\s&'\-]{3,50})\s+(?:hospital|medical center|health system|clinic|center)",
        r"(?:hospital|medical center|clinic)[:\s]+([A-Z][A-Za-z\s&'\-]{3,50})",
        r"(?:name|facility)[:\s]+([A-Z][A-Za-z\s&'\-]{3,50})",
    ]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _location(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    city, state, zcode = None, None, None
    zm = re.search(r"\b(\d{5})\b", text)
    if zm: zcode = zm.group(1)
    sm = re.search(r"\b([A-Z]{2})\b", text)
    if sm and sm.group(1) in _US_STATES: state = sm.group(1)
    if state:
        cm = re.search(rf"([A-Z][a-z]+(?: [A-Z][a-z]+)*),?\s+{state}\b", text)
        if cm: city = cm.group(1)
    if not city:
        cm = re.search(r"(?:in|city[:\s]+)\s*([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text, re.IGNORECASE)
        if cm: city = cm.group(1)
    return city, state, zcode


def _caps(text: str) -> HospitalCapabilities:
    tl = text.lower()
    return HospitalCapabilities(**{k: any(kw in tl for kw in kws) for k, kws in _CAPS.items()})


def _doctors(text: str) -> Tuple[int, Optional[str]]:
    count, extras = 0, []
    m = re.search(r"(\d+)\s*(?:doctors?|physicians?|surgeons?)", text, re.IGNORECASE)
    if m: count = int(m.group(1))
    m = re.search(r"(\d+)\s*nurses?", text, re.IGNORECASE)
    if m: extras.append(f"{m.group(1)} nurses")
    m = re.search(r"(\d+)\s*beds?", text, re.IGNORECASE)
    if m: extras.append(f"{m.group(1)} beds")
    return count, ("; ".join(extras) if extras else None)


def _rating(text: str) -> Optional[int]:
    for p in [r"rating[:\s]+(\d)", r"(\d)\s*(?:star|\/5|out of 5)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 5:
                return v
    return None


def _confidence(name, city, state, caps, docs) -> float:
    s = sum([
        0.25 if name  else 0,
        0.15 if city  else 0,
        0.15 if state else 0,
        0.20 if any(vars(caps).values()) else 0,
        0.10 if docs > 0 else 0,
        0.15,
    ])
    return round(min(s, 1.0), 2)


def parse_hospital_text(text: str, strict_mode: bool = False) -> ParseResponse:
    notes: List[str] = []
    if not text or len(text.strip()) < 10:
        return ParseResponse(success=False, raw_input=text, processing_notes=["Input too short"])

    clean = re.sub(r"\s+", " ", text.strip())
    name             = _name(clean)
    city, state, zip_= _location(clean)
    caps             = _caps(clean)
    doc_count, extra = _doctors(clean)
    rating           = _rating(clean)

    if name is None:
        notes.append("Could not confidently extract hospital name")
        if strict_mode:
            return ParseResponse(success=False, raw_input=text, processing_notes=notes)
    if state is None:
        notes.append("State not detected")
    if not any(vars(caps).values()):
        notes.append("No capabilities detected")

    conf = _confidence(name, city, state, caps, doc_count)
    if conf < 0.4:
        notes.append(f"Low confidence ({conf:.0%}) — review recommended")

    prefix = (state or "XX")[:2].upper()
    slug   = re.sub(r"[^a-z0-9]", "", (name or "unknown").lower())[:8]

    return ParseResponse(
        success=True,
        raw_input=text,
        processing_notes=notes,
        hospital=HospitalRecord(
            facility_id   = f"{prefix}_PARSED_{slug}",
            facility_name = name or "Unknown Facility",
            city=city, state=state, zip_code=zip_,
            capabilities  = caps,
            quality       = HospitalQualityMetrics(overall_rating=rating),
            doctor_count  = doc_count,
            confidence_score = conf,
            notes         = extra,
        ),
    )
