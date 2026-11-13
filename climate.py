"""Season-average weather for a place in India, from Open-Meteo.

Open-Meteo is free and needs no API key. This module finds a place by name,
downloads the last 5 full years of daily weather for it, and averages the
months of the chosen Indian cropping season.

Rainfall note: the dataset never states its rainfall period. Its values
(20–300 mm) match AVERAGE MONTHLY rainfall during a growing season, so that
is what we compute: total season rainfall / (months in season × years).
"""

import json
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
YEARS = 5

# Months each season's crops are in the ground
SEASON_MONTHS = {
    "kharif": [6, 7, 8, 9, 10],
    "rabi": [11, 12, 1, 2, 3],
    "zaid": [3, 4, 5, 6],
}

# Rough box around India: (min lat, max lat, min lon, max lon)
INDIA_BOUNDS = (6.0, 37.5, 68.0, 97.5)


# Old names people still type -> the name the geocoder knows
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bombay": "Mumbai", "calcutta": "Kolkata",
    "madras": "Chennai", "gurgaon": "Gurugram", "trivandrum": "Thiruvananthapuram",
    "mysore": "Mysuru", "gulbarga": "Kalaburagi", "poona": "Pune",
    "baroda": "Vadodara", "cochin": "Kochi", "mangalore": "Mangaluru",
    "belgaum": "Belagavi", "simla": "Shimla", "benares": "Varanasi",
    "banaras": "Varanasi", "allahabad": "Prayagraj", "pondicherry": "Puducherry",
    "cawnpore": "Kanpur", "hubli": "Hubballi", "bellary": "Ballari",
    "tumkur": "Tumakuru", "shimoga": "Shivamogga",
}

# States and UTs: too big for one weather average (Delhi, Chandigarh and Puducherry are also cities, so allowed)
STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal", "jammu and kashmir",
    "ladakh", "andaman and nicobar islands", "lakshadweep",
    "dadra and nagar haveli and daman and diu",
}


class ClimateError(Exception):
    """A problem whose message can be shown to the user as-is."""


def _get_json(url: str, params: dict) -> dict:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(full_url, timeout=20) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise ClimateError(
            "The weather service could not be reached. "
            "Check your internet connection, or enter the values manually."
        ) from err


def in_india(lat: float, lon: float) -> bool:
    """True if the point is inside a rough box around India (NaN gives False)."""
    min_lat, max_lat, min_lon, max_lon = INDIA_BOUNDS
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _normalize(text: str) -> str:
    """Lowercase without accents, so 'Aurangābād' matches 'aurangabad'."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(plain.lower().split())


def is_state(name: str) -> bool:
    return _normalize(name) in STATES


def find_place(query: str) -> dict | None:
    """Looks up a city, town or district in India. Returns None if not found.

    Accepts "Name" or "Name, State" (to pick between places with the same name).
    The geocoder also returns partial matches ("Punjab" -> "Punjabpura"), so only
    exact name matches are kept, and the most populated one wins.
    """
    name, _, region = query.partition(",")
    name = CITY_ALIASES.get(_normalize(name), name.strip())
    data = _get_json(GEOCODE_URL, {
        "name": name,
        "count": 10,
        "countryCode": "IN",
        "language": "en",
        "format": "json",
    })
    wanted, region = _normalize(name), _normalize(region)
    matches = [
        r for r in data.get("results") or []
        if _normalize(r.get("name", "")) == wanted
        and (not region or _normalize(r.get("admin1", "")).startswith(region))
    ]
    if not matches:
        return None
    best = max(matches, key=lambda r: r.get("population") or 0)
    label = ", ".join(part for part in (best.get("name"), best.get("admin1")) if part)
    return {"label": label, "lat": best["latitude"], "lon": best["longitude"]}


def season_averages(daily: dict, months: list[int], years: int) -> dict:
    """Averages daily weather over the season's months.

    Args:
        daily: Open-Meteo daily data: lists under "time", "temperature_2m_mean",
            "relative_humidity_2m_mean" and "precipitation_sum".
        months: The season's month numbers (1–12).
        years: How many years the data covers.

    Returns:
        {"temperature": mean °C, "humidity": mean %, "rainfall": average monthly mm}
    """
    temps, hums, rain_total = [], [], 0.0
    for i, day in enumerate(daily["time"]):
        if int(day[5:7]) not in months:
            continue
        if daily["temperature_2m_mean"][i] is not None:
            temps.append(daily["temperature_2m_mean"][i])
        if daily["relative_humidity_2m_mean"][i] is not None:
            hums.append(daily["relative_humidity_2m_mean"][i])
        rain_total += daily["precipitation_sum"][i] or 0.0

    if not temps or not hums:
        raise ClimateError("No weather data is available for this location.")

    return {
        "temperature": round(sum(temps) / len(temps), 1),
        "humidity": round(sum(hums) / len(hums), 1),
        "rainfall": round(rain_total / (len(months) * years), 1),
    }


def fetch_season_climate(lat: float, lon: float, season: str, today: date | None = None) -> dict:
    """Season averages for a point over the last YEARS full calendar years."""
    end_year = (today or date.today()).year - 1
    start_year = end_year - YEARS + 1
    data = _get_json(ARCHIVE_URL, {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "Asia/Kolkata",
    })
    if "daily" not in data:
        raise ClimateError("No weather data is available for this location.")
    averages = season_averages(data["daily"], SEASON_MONTHS[season], YEARS)
    return {**averages, "start_year": start_year, "end_year": end_year}
