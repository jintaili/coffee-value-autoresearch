"""Deterministic feature extraction for CoffeeReview rows.

This module is intentionally shared by preparation and model training code so
train and validation rows cannot drift through different processing paths.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


UNKNOWN = "unknown"


COUNTRY_ALIASES = {
    "bolivia": "Bolivia",
    "brazil": "Brazil",
    "burundi": "Burundi",
    "cameroon": "Cameroon",
    "china": "China",
    "colombia": "Colombia",
    "congo": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "costa rica": "Costa Rica",
    "cote d ivoire": "Cote d'Ivoire",
    "dominican republic": "Dominican Republic",
    "ecuador": "Ecuador",
    "el salvador": "El Salvador",
    "ethiopia": "Ethiopia",
    "guatemala": "Guatemala",
    "haiti": "Haiti",
    "honduras": "Honduras",
    "india": "India",
    "indonesia": "Indonesia",
    "jamaica": "Jamaica",
    "java": "Indonesia",
    "kenya": "Kenya",
    "laos": "Laos",
    "malawi": "Malawi",
    "mexico": "Mexico",
    "myanmar": "Myanmar",
    "nicaragua": "Nicaragua",
    "panama": "Panama",
    "papua new guinea": "Papua New Guinea",
    "peru": "Peru",
    "philippines": "Philippines",
    "puerto rico": "Puerto Rico",
    "rwanda": "Rwanda",
    "sumatra": "Indonesia",
    "sulawesi": "Indonesia",
    "taiwan": "Taiwan",
    "tanzania": "Tanzania",
    "thailand": "Thailand",
    "timor": "Timor-Leste",
    "timor leste": "Timor-Leste",
    "uganda": "Uganda",
    "vietnam": "Vietnam",
    "yemen": "Yemen",
    "zambia": "Zambia",
}

US_STATES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "hawai'i",
    "hawai’i",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}

TAIWAN_LOCATIONS = {
    "taipei",
    "new taipei city",
    "taichung",
    "tainan",
    "kaohsiung",
    "hsinchu",
    "pingtung",
    "chia-yi",
    "chiayi",
    "taoyuan",
}

PROCESS_PATTERNS = [
    ("carbonic_maceration", r"\bcarbonic\s+maceration\b"),
    ("wet_hulled", r"\bwet[-\s]?hulled\b|\bgiling\s+basah\b"),
    ("anaerobic", r"\banaerobic\b"),
    ("lactic", r"\blactic\b"),
    ("honey", r"\bhoney[-\s]?(?:processed|process)?\b|\bblack honey\b|\bred honey\b|\byellow honey\b|\bwhite honey\b"),
    ("natural", r"\bnatural[-\s]?(?:processed|process)?\b|\bdry[-\s]processed\b|\bdried in the fruit\b"),
    ("washed", r"\bwashed\b|\bwet[-\s]processed\b|\bfully washed\b|\btraditional washed\b"),
]

VARIETY_PATTERNS = [
    ("pink_bourbon", r"\bpink\s+bourbon\b"),
    ("gesha", r"\bgesha\b|\bgeisha\b"),
    ("bourbon", r"\bbourbon\b"),
    ("typica", r"\btypica\b"),
    ("caturra", r"\bcaturra\b"),
    ("catuai", r"\bcatuai\b|\bcatuai\b"),
    ("sl28", r"\bsl\s*[-]?\s*28\b"),
    ("sl34", r"\bsl\s*[-]?\s*34\b"),
    ("pacamara", r"\bpacamara\b"),
    ("maragogipe", r"\bmaragogipe\b|\bmaragogype\b"),
    ("mokka", r"\bmokka\b"),
    ("mokkita", r"\bmokkita\b"),
    ("ruiru", r"\bruiru\b"),
    ("castillo", r"\bcastillo\b"),
    ("java", r"\bjava\b"),
]

FARM_CUES = re.compile(
    r"\b(farm|estate|producer|co-?op(?:erative)?|cooperative|mill|washing station|station|finca|hacienda|"
    r"plantation|grower|smallholder|small-holder|lot|micro-?lot|nanolot|project|collective)\b",
    re.I,
)

ALTITUDE_CUES = re.compile(
    r"\b(altitude|elevation|masl|m\.a\.s\.l\.|meters?|metres?|feet|ft\.)\b|"
    r"\b\d{3,4}\s*(?:-\s*\d{3,4}\s*)?(?:m|masl|meters?|metres?|feet|ft)\b",
    re.I,
)

ESPRESSO_RE = re.compile(r"\bespresso\b|\bristretto\b|\bcappuccino\b|\bshot\b", re.I)
DECAF_RE = re.compile(r"\bdecaf(?:feinated)?\b|\bswiss water\b|\bmountain water process\b", re.I)
BLEND_RE = re.compile(r"\bblend\b|\bblended\b", re.I)


@dataclass(frozen=True)
class CoffeeFeatures:
    row_id: str
    coffee_name: str
    roaster: str
    review_date: str
    review_year: str
    rating: str
    origin_country: str
    origin_region: str
    process_method: str
    variety: str
    is_blend: str
    is_espresso: str
    is_decaf: str
    producer_or_farm_present: str
    altitude_present: str
    roaster_country: str
    sensory_text: str
    producer_text: str


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\ufeff", " ")).strip()


def normalize_for_match(value: str | None) -> str:
    value = normalize_space(value).lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return normalize_space(value)


def joined_labels(labels: list[str]) -> str:
    return "|".join(sorted(set(labels))) if labels else UNKNOWN


def parse_review_year(review_date: str) -> str:
    raw = normalize_space(review_date)
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%B %Y"):
        try:
            return str(datetime.strptime(raw, fmt).year)
        except ValueError:
            pass
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    return match.group(0) if match else ""


def extract_origin_country(origin: str) -> str:
    text = normalize_for_match(origin)
    if not text or text in {"na", "n a"}:
        return UNKNOWN
    labels = []
    padded = f" {text} "
    for alias, country in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", padded):
            labels.append(country)
    labels = sorted(set(labels))
    if len(labels) > 1:
        return "blend_multi_origin"
    return labels[0] if labels else UNKNOWN


def extract_origin_region(origin: str, origin_country: str) -> str:
    raw = normalize_space(origin)
    if not raw or raw.upper() == "NA":
        return UNKNOWN
    if origin_country == "blend_multi_origin":
        return "multi_origin"
    region = raw
    if origin_country not in {"", UNKNOWN}:
        region = re.sub(re.escape(origin_country), "", region, flags=re.I)
    for alias, country in COUNTRY_ALIASES.items():
        if country == origin_country:
            region = re.sub(rf"\b{re.escape(alias)}\b", "", region, flags=re.I)
    region = re.sub(r"\b(growing region|region|province|department|district|central|southern|south-central|north|south|east|west)\b", "", region, flags=re.I)
    region = normalize_space(region.strip(" ,.;:-"))
    if not region or len(region) < 3:
        return UNKNOWN
    return region.lower()


def extract_roaster_country(location: str) -> str:
    raw = normalize_space(location)
    if not raw:
        return UNKNOWN
    parts = [normalize_space(p) for p in raw.split(",") if normalize_space(p)]
    last = normalize_for_match(parts[-1] if parts else raw)
    if last in US_STATES:
        return "United States"
    if last in TAIWAN_LOCATIONS:
        return "Taiwan"
    for alias, country in COUNTRY_ALIASES.items():
        if last == alias:
            return country
    if last in {"united states", "usa", "u s a"}:
        return "United States"
    if last in {"england", "scotland", "wales"}:
        return "United Kingdom"
    return normalize_space(parts[-1] if parts else raw)


def extract_process_method(text: str) -> str:
    norm = normalize_for_match(text)
    labels = [label for label, pattern in PROCESS_PATTERNS if re.search(pattern, norm, re.I)]
    return joined_labels(labels)


def extract_variety(text: str) -> str:
    norm = normalize_for_match(text)
    labels = [label for label, pattern in VARIETY_PATTERNS if re.search(pattern, norm, re.I)]
    if "gesha" in labels and "geisha" in labels:
        labels = [x for x in labels if x != "geisha"]
    return joined_labels(labels)


def bool_str(value: bool) -> str:
    return "1" if value else "0"


def extract_row(row: dict[str, str], row_id: int) -> CoffeeFeatures:
    coffee_name = normalize_space(row.get("bean"))
    roaster = normalize_space(row.get("roaster"))
    origin = normalize_space(row.get("origin"))
    review_date = normalize_space(row.get("review_date"))
    blind = normalize_space(row.get("blind_assessment"))
    notes = normalize_space(row.get("notes"))
    bottom = normalize_space(row.get("bottom_line"))
    full_text = " ".join(x for x in [coffee_name, origin, blind, notes, bottom] if x)

    origin_country = extract_origin_country(origin)
    process_method = extract_process_method(full_text)
    variety = extract_variety(full_text)
    is_blend = BLEND_RE.search(full_text) is not None or origin_country == "blend_multi_origin"

    return CoffeeFeatures(
        row_id=str(row_id),
        coffee_name=coffee_name,
        roaster=roaster,
        review_date=review_date,
        review_year=parse_review_year(review_date),
        rating=normalize_space(row.get("rating")),
        origin_country=origin_country,
        origin_region=extract_origin_region(origin, origin_country),
        process_method=process_method,
        variety=variety,
        is_blend=bool_str(is_blend),
        is_espresso=bool_str(ESPRESSO_RE.search(full_text) is not None),
        is_decaf=bool_str(DECAF_RE.search(full_text) is not None),
        producer_or_farm_present=bool_str(FARM_CUES.search(full_text) is not None),
        altitude_present=bool_str(ALTITUDE_CUES.search(full_text) is not None),
        roaster_country=extract_roaster_country(row.get("location", "")),
        sensory_text=blind,
        producer_text=normalize_space(" ".join(x for x in [notes, bottom] if x)),
    )


def read_source(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def extract_rows(source_path: Path) -> list[CoffeeFeatures]:
    return [extract_row(row, i) for i, row in enumerate(read_source(source_path))]


def feature_fieldnames() -> list[str]:
    return list(CoffeeFeatures.__dataclass_fields__.keys())


def write_features(rows: list[CoffeeFeatures], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
