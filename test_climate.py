"""Tests for the location weather feature (climate.py and the /climate endpoint).

The weather service is replaced with fakes, so these tests run offline.
"""

from datetime import date

import pytest

import app as app_module
from app import app
from climate import (
    ClimateError,
    fetch_season_climate,
    find_place,
    in_india,
    is_state,
    season_averages,
)


def fake_geocoder(results):
    """Returns a stand-in for climate._get_json that records the searched name."""
    calls = []

    def fake(url, params):
        calls.append(params["name"])
        return {"results": results}

    return fake, calls


def place(name, state, population=None):
    return {"name": name, "admin1": state, "population": population, "latitude": 20.0, "longitude": 78.0}


def test_find_place_ignores_partial_matches(monkeypatch):
    """The geocoder answers 'Punjab' with 'Punjābpura'; that must not count."""
    fake, _ = fake_geocoder([place("Punjābpura", "Uttar Pradesh")])
    monkeypatch.setattr("climate._get_json", fake)
    assert find_place("Punjab") is None


def test_find_place_prefers_biggest_exact_match_ignoring_accents(monkeypatch):
    fake, _ = fake_geocoder([
        place("Aurangābād", "Haryana"),
        place("Aurangabad", "Maharashtra", 1175116),
        place("Aurangābād", "Bihar", 102244),
    ])
    monkeypatch.setattr("climate._get_json", fake)
    assert find_place("aurangabad")["label"] == "Aurangabad, Maharashtra"
    assert find_place("Aurangabad, Bihar")["label"] == "Aurangābād, Bihar"


def test_find_place_maps_old_city_names(monkeypatch):
    fake, calls = fake_geocoder([place("Bengaluru", "Karnataka", 8495492)])
    monkeypatch.setattr("climate._get_json", fake)
    assert find_place("Bangalore")["label"] == "Bengaluru, Karnataka"
    assert calls == ["Bengaluru"]


def test_is_state():
    assert is_state("Punjab")
    assert is_state("  tamil   NADU ")
    assert not is_state("Delhi")      # city as well as a territory
    assert not is_state("Ludhiana")


def test_climate_rejects_state_names(client):
    response = client.get("/climate?season=kharif&place=Kerala")
    assert response.status_code == 400
    assert "state" in response.get_json()["error"]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_season_averages_uses_only_season_months():
    """July and August are averaged; January is ignored. Rainfall is per month."""
    daily = {
        "time": ["2024-01-15", "2024-07-01", "2024-07-02", "2024-08-01"],
        "temperature_2m_mean": [10.0, 30.0, 28.0, 32.0],
        "relative_humidity_2m_mean": [40, 80, 90, None],
        "precipitation_sum": [500.0, 100.0, None, 200.0],
    }
    result = season_averages(daily, months=[7, 8], years=1)
    assert result["temperature"] == 30.0         # (30 + 28 + 32) / 3
    assert result["humidity"] == 85.0            # missing value skipped
    assert result["rainfall"] == 150.0           # 300 mm over 2 months × 1 year


def test_season_averages_no_data_raises():
    daily = {"time": ["2024-01-01"], "temperature_2m_mean": [20.0],
             "relative_humidity_2m_mean": [50], "precipitation_sum": [0.0]}
    with pytest.raises(ClimateError):
        season_averages(daily, months=[7], years=1)


def test_fetch_season_climate_requests_last_five_full_years(monkeypatch):
    seen = {}

    def fake_get_json(url, params):
        seen.update(params)
        return {"daily": {"time": ["2021-06-01"], "temperature_2m_mean": [30.0],
                          "relative_humidity_2m_mean": [80], "precipitation_sum": [50.0]}}

    monkeypatch.setattr("climate._get_json", fake_get_json)
    result = fetch_season_climate(22.5, 88.3, "kharif", today=date(2026, 10, 27))
    assert seen["start_date"] == "2021-01-01"
    assert seen["end_date"] == "2025-12-31"
    assert (result["start_year"], result["end_year"]) == (2021, 2025)


def test_in_india():
    assert in_india(22.57, 88.36)          # Kolkata
    assert not in_india(51.5, -0.12)       # London
    assert not in_india(float("nan"), 80)  # NaN never passes


def test_climate_by_place(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_place",
                        lambda name: {"label": "Nagpur, Maharashtra", "lat": 21.15, "lon": 79.09})
    monkeypatch.setattr(app_module, "fetch_season_climate",
                        lambda lat, lon, season: {"temperature": 27.5, "humidity": 76.2, "rainfall": 287.0,
                                                  "start_year": 2021, "end_year": 2025})
    response = client.get("/climate?season=Kharif&place=nagpur")
    assert response.status_code == 200
    data = response.get_json()
    assert data["place"] == "Nagpur, Maharashtra"
    assert data["season"] == "kharif"
    assert data["rainfall"] == 287.0


def test_climate_by_coordinates(client, monkeypatch):
    monkeypatch.setattr(app_module, "fetch_season_climate",
                        lambda lat, lon, season: {"temperature": 20.0, "humidity": 50.0, "rainfall": 10.0,
                                                  "start_year": 2021, "end_year": 2025})
    response = client.get("/climate?season=rabi&lat=22.72&lon=75.86")
    assert response.status_code == 200
    assert response.get_json()["place"] == "your location"


@pytest.mark.parametrize("query", [
    "place=Nagpur",                          # no season
    "season=monsoon&place=Nagpur",           # unknown season
    "season=kharif",                         # no place and no coordinates
    "season=kharif&lat=abc&lon=79",          # coordinates not numbers
    "season=kharif&lat=51.5&lon=-0.12",      # outside India
    "season=kharif&lat=nan&lon=nan",
    "season=kharif&place=" + "x" * 101,      # too long
])
def test_climate_bad_requests(client, query):
    response = client.get(f"/climate?{query}")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_climate_place_not_found(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_place", lambda name: None)
    response = client.get("/climate?season=kharif&place=Atlantis")
    assert response.status_code == 404
    assert "Atlantis" in response.get_json()["error"]


def test_climate_service_down(client, monkeypatch):
    def broken(name):
        raise ClimateError("The weather service could not be reached.")

    monkeypatch.setattr(app_module, "find_place", broken)
    response = client.get("/climate?season=kharif&place=Nagpur")
    assert response.status_code == 502
    assert "could not be reached" in response.get_json()["error"]
