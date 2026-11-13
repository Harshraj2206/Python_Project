"""Tests for the rule-based explanation layer (explain.py).

Verifies input description bands, boundary conditions, crop fit explanation,
importance-based sorting, limits (at most 3 matches, at most 2 cautions),
and the no-matches fallback summary.
"""

import pytest
from explain import describe_inputs, explain_crop


def test_spec_required_input_descriptions():
    """Verify the exact spec checks:

    - pH 5.5 -> 'slightly acidic soil'
    - pH 5.49 -> 'strongly acidic soil'
    - rainfall 250 -> 'high rainfall'
    """
    labels = describe_inputs({"ph": 5.5, "rainfall": 250})
    assert labels["ph"] == "slightly acidic soil"
    assert labels["rainfall"] == "high rainfall"

    labels_acidic = describe_inputs({"ph": 5.49})
    assert labels_acidic["ph"] == "strongly acidic soil"


def test_describe_inputs_all_boundaries():
    """Verify boundary behavior across all features (boundary goes to higher band)."""
    inputs = {
        "ph": 6.5,          # 6.5 -> neutral soil (5.5-6.5 is slightly acidic, boundary 6.5 -> neutral)
        "rainfall": 75,     # 75 -> moderate rainfall (< 75 low, 75-200 moderate)
        "temperature": 18,  # 18 -> mild climate (< 18 cool, 18-28 mild)
        "humidity": 40,     # 40 -> moderate humidity (< 40 dry, 40-70 moderate)
        "N": 40,            # 40 -> medium nitrogen (< 40 low, 40-90 medium)
        "P": 30,            # 30 -> medium phosphorus (< 30 low, 30-70 medium)
        "K": 30,            # 30 -> medium potassium (< 30 low, 30-80 medium)
    }
    labels = describe_inputs(inputs)
    assert labels["ph"] == "neutral soil"
    assert labels["rainfall"] == "moderate rainfall"
    assert labels["temperature"] == "mild climate"
    assert labels["humidity"] == "moderate humidity"
    assert labels["N"] == "medium nitrogen"
    assert labels["P"] == "medium phosphorus"
    assert labels["K"] == "medium potassium"

    # Test upper boundaries
    inputs_upper = {
        "ph": 7.5,          # 7.5 -> alkaline soil
        "rainfall": 200,    # 200 -> high rainfall
        "temperature": 28,  # 28 -> hot climate
        "humidity": 70,     # 70 -> humid climate
        "N": 90,            # 90 -> high nitrogen
        "P": 70,            # 70 -> high phosphorus
        "K": 80,            # 80 -> high potassium
    }
    labels_upper = describe_inputs(inputs_upper)
    assert labels_upper["ph"] == "alkaline soil"
    assert labels_upper["rainfall"] == "high rainfall"
    assert labels_upper["temperature"] == "hot climate"
    assert labels_upper["humidity"] == "humid climate"
    assert labels_upper["N"] == "high nitrogen"
    assert labels_upper["P"] == "high phosphorus"
    assert labels_upper["K"] == "high potassium"

    # Test strongly alkaline (> 8.5) and at 8.5
    assert describe_inputs({"ph": 8.5})["ph"] == "strongly alkaline soil"
    assert describe_inputs({"ph": 8.49})["ph"] == "alkaline soil"


def test_explain_crop_matches_cautions_and_limits():
    """Verify matches and cautions are sorted by importance, capped at 3 and 2 respectively,

    and format numbers and units accurately.
    """
    hand_made_ranges = {
        "rice": {
            "rainfall": (190, 260),
            "humidity": (80, 85),
            "temperature": (20, 25),
            "N": (60, 90),
            "ph": (5.5, 7.0),
            "P": (35, 55),
            "K": (35, 45),
        }
    }

    # Importance order: rainfall > humidity > temperature > N > ph > P > K
    importances = {
        "rainfall": 0.35,
        "humidity": 0.25,
        "temperature": 0.15,
        "N": 0.10,
        "ph": 0.08,
        "P": 0.05,
        "K": 0.02,
    }

    # 4 matches: rainfall, humidity, temperature, N
    # 3 cautions: ph (higher), P (lower), K (lower)
    values = {
        "rainfall": 230,     # match
        "humidity": 82,      # match
        "temperature": 22,   # match
        "N": 70,             # match (4th match -> should be excluded by top 3 cap)
        "ph": 8.1,           # caution: higher
        "P": 20,             # caution: lower
        "K": 15,             # caution: lower (3rd caution -> should be excluded by top 2 cap)
    }

    result = explain_crop("rice", values, hand_made_ranges, importances)

    # Must return at most 3 matches (reasons) and at most 2 cautions
    assert len(result["reasons"]) == 3
    assert len(result["cautions"]) == 2

    # Check that matches are sorted by importance: rainfall, humidity, temperature
    assert "Rainfall 230 mm is within Rice's typical range (190–260 mm)" in result["reasons"][0]
    assert "Humidity 82% is within Rice's typical range (80–85%)" in result["reasons"][1]
    assert "Temperature 22 °C is within Rice's typical range (20–25 °C)" in result["reasons"][2]

    # Check cautions: ph (higher) and P (lower)
    assert "pH 8.1 is higher than Rice's typical range (5.5–7)" in result["cautions"][0]
    assert "Phosphorus 20 is lower than Rice's typical range (35–55)" in result["cautions"][1]

    # Check summary sentence: top 3 matches joined with ' + ' ending with '→ Rice is a strong fit'
    # rainfall 230 -> high rainfall; humidity 82 -> humid climate; temperature 22 -> mild climate
    expected_summary = "High rainfall + humid climate + mild climate → Rice is a strong fit"
    assert result["summary"] == expected_summary


def test_explain_crop_summary_wording_follows_confidence():
    """A low-confidence alternative must not be called a 'strong fit'."""
    ranges = {"rice": {"rainfall": (190, 260)}}
    importances = {"rainfall": 1.0}
    values = {"rainfall": 230}

    def summary(conf):
        return explain_crop("rice", values, ranges, importances, confidence=conf)["summary"]

    assert summary(0.94) == "High rainfall → Rice is a strong fit"
    assert summary(0.40) == "High rainfall → Rice is a possible fit"
    assert summary(0.07) == "High rainfall → Rice is a weaker alternative"


def test_explain_crop_boundary_inclusive():
    """Verify that inputs on the exact p25 or p75 boundaries count as matches (inclusive)."""
    ranges = {
        "rice": {
            "rainfall": (190.0, 260.0),
            "humidity": (80.0, 85.0),
        }
    }
    importances = {"rainfall": 0.6, "humidity": 0.4}
    values = {
        "rainfall": 190.0,  # exactly on p25 boundary
        "humidity": 85.0,   # exactly on p75 boundary
    }

    result = explain_crop("rice", values, ranges, importances)
    assert len(result["reasons"]) == 2
    assert len(result["cautions"]) == 0
    assert "within Rice's typical range" in result["reasons"][0]
    assert "within Rice's typical range" in result["reasons"][1]


def test_explain_crop_no_matches_fallback_summary():
    """Verify that when no features fall inside typical ranges, the fallback summary is used."""
    hand_made_ranges = {
        "rice": {
            "rainfall": (190, 260),
            "humidity": (80, 85),
            "temperature": (20, 25),
        }
    }
    importances = {"rainfall": 0.5, "humidity": 0.3, "temperature": 0.2}

    # All values out of range
    values = {
        "rainfall": 100,    # lower
        "humidity": 60,     # lower
        "temperature": 35,  # higher
    }

    result = explain_crop("rice", values, hand_made_ranges, importances)

    assert result["reasons"] == []
    assert len(result["cautions"]) == 2  # capped at 2
    assert result["summary"] == "→ Rice is the closest match, but conditions are not ideal"
