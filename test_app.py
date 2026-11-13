"""Tests for the AgriSmart Flask application (app.py).

Uses the Flask test client and the trained model bundle to verify endpoints,
server-side validation, soft limit warnings, and recommendation responses.
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Configures the Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_route(client):
    """GET / renders templates/index.html and returns 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AgriSmart" in response.data


def test_predict_valid_example_input(client):
    """Valid example input returns 200, 1 to 3 recommendations, sorted confidences,

    and rice as the top recommendation.
    """
    example_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=example_input)
    assert response.status_code == 200

    data = response.get_json()
    assert "recommendations" in data
    assert "input_labels" in data
    assert "warnings" in data
    assert data["warnings"] == []

    recs = data["recommendations"]
    assert 1 <= len(recs) <= 3

    # Top recommendation must be rice
    top_rec = recs[0]
    assert top_rec["crop"] == "rice"
    assert top_rec["name"] == "Rice"
    assert top_rec["local_name"] == "Dhaan"
    assert top_rec["season"] == "Kharif"
    assert 0.0 <= top_rec["confidence"] <= 1.0

    # Confidences must be sorted descending
    confidences = [r["confidence"] for r in recs]
    assert confidences == sorted(confidences, reverse=True)

    # Check structure of each recommendation
    for rec in recs:
        assert "crop" in rec
        assert "name" in rec
        assert "local_name" in rec
        assert "season" in rec
        assert "confidence" in rec
        assert "summary" in rec
        assert isinstance(rec["reasons"], list)
        assert isinstance(rec["cautions"], list)

    # Input labels should have all 7 features
    assert len(data["input_labels"]) == 7
    assert data["input_labels"]["ph"] == "neutral soil"
    assert data["input_labels"]["rainfall"] == "high rainfall"


def test_predict_missing_field(client):
    """Missing field rejected with 400 and error/fields details."""
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        # 'ph' is missing
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "fields" in data
    assert "ph" in data["fields"]


def test_predict_string_value(client):
    """String value rejected with 400 ('5' is not accepted as number)."""
    payload = {
        "N": "90",  # string instead of number
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "fields" in data
    assert "N" in data["fields"]


def test_predict_boolean_value(client):
    """Boolean value rejected with 400."""
    payload = {
        "N": True,  # boolean must be rejected
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "fields" in data
    assert "N" in data["fields"]


def test_predict_ph_15_hard_limit(client):
    """pH 15 exceeds hard limit (0-14) and is rejected with 400."""
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 15,  # hard limit 0-14
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "fields" in data
    assert data["fields"]["ph"] == "pH must be between 0 and 14."


def test_predict_non_json_body(client):
    """Non-JSON payload rejected with 400."""
    response = client.post(
        "/predict",
        data="not a json string",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_temperature_48_soft_warning(client):
    """Temperature 48 is inside hard limits (-10 to 55) but outside dataset range (8.8 to 43.7),

    returning 200 with an advisory warning.
    """
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 48,  # soft limit violation
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "warnings" in data
    assert len(data["warnings"]) >= 1
    assert (
        "Temperature 48 °C is outside the range the model was trained on (8.8–43.7 °C), "
        "so results may be less reliable."
    ) in data["warnings"]


def test_predict_huge_integer_rejected(client):
    """An integer too large for a float is a 400, not a server crash."""
    body = (
        '{"N": 1' + "0" * 400 + ', "P": 42, "K": 43, "temperature": 20.9, '
        '"humidity": 82.0, "ph": 6.5, "rainfall": 202.9}'
    )
    response = client.post("/predict", data=body, content_type="application/json")
    assert response.status_code == 400
    assert "N" in response.get_json()["fields"]


def test_predict_get_method_not_allowed(client):
    """GET /predict returns 405 Method Not Allowed (not 500)."""
    response = client.get("/predict")
    assert response.status_code == 405


def test_unknown_url_not_found(client):
    """Unknown URL /abc returns 404 Not Found (not 500)."""
    response = client.get("/abc")
    assert response.status_code == 404

