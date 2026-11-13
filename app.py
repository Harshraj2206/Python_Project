"""Flask web server for AgriSmart crop recommendation.

Loads the trained Random Forest model once at startup, serves the web interface,
and provides the /predict API endpoint with server-side validation, soft warnings,
and rule-based explanations.
"""

import math
import os
import sys
import logging
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from climate import (
    SEASON_MONTHS,
    ClimateError,
    fetch_season_climate,
    find_place,
    in_india,
    is_state,
)
from crops import CROPS
from explain import (
    FEATURES,
    _format_num,
    _format_value_with_unit,
    describe_inputs,
    explain_crop,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model bundle path
# Relative to this file, so the app works no matter which folder it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "agrismart.joblib")

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    print("Run `python train.py` first.")
    sys.exit(1)

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features_list = bundle["features"]
    crop_ranges = bundle["crop_ranges"]
    importances = bundle["importances"]
    logger.info("Model bundle loaded successfully.")
except Exception as err:
    print(f"Error loading model bundle: {err}")
    sys.exit(1)

app = Flask(__name__)

# Field display names for validation messages
FIELD_NAMES = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "pH",
    "rainfall": "Rainfall",
}

# Hard validation limits: (min, max, error_message)
HARD_LIMITS = {
    "N": (0, 300, "Nitrogen must be between 0 and 300."),
    "P": (0, 300, "Phosphorus must be between 0 and 300."),
    "K": (0, 300, "Potassium must be between 0 and 300."),
    "temperature": (-10, 55, "Temperature must be between -10 and 55 °C."),
    "humidity": (0, 100, "Humidity must be between 0 and 100%."),
    "ph": (0, 14, "pH must be between 0 and 14."),
    "rainfall": (0, 5000, "Rainfall must be between 0 and 5000 mm."),
}

# Soft limits based on Kaggle training dataset ranges: (min, max, name, unit)
SOFT_LIMITS = {
    "N": (0, 140, "Nitrogen", ""),
    "P": (5, 145, "Phosphorus", ""),
    "K": (5, 205, "Potassium", ""),
    "temperature": (8.8, 43.7, "Temperature", "°C"),
    "humidity": (14.3, 100, "Humidity", "%"),
    "ph": (3.5, 9.9, "pH", ""),
    "rainfall": (20.2, 298.6, "Rainfall", "mm"),
}


def is_real_number(val):
    """Returns True if val is a real int or float, rejecting booleans, strings, NaN, and Inf."""
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    try:
        val = float(val)
    except OverflowError:
        # JSON integers can be longer than any float, e.g. 1 followed by 400 zeros
        return False
    if math.isnan(val) or math.isinf(val):
        return False
    return True


@app.route("/", methods=["GET"])
def index():
    """Renders the main single-page interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predicts top crop recommendations for given soil and climate inputs."""
    # 1. Reject requests larger than 10 KB
    if request.content_length and request.content_length > 10 * 1024:
        return jsonify({"error": "Request body must be under 10 KB."}), 400

    # 2. Reject non-JSON payload
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    field_errors = {}
    validated_values = {}

    # 3. Check presence and real number types
    for feat in FEATURES:
        name = FIELD_NAMES.get(feat, feat)
        if feat not in data:
            field_errors[feat] = f"{name} is required."
            continue

        raw_val = data[feat]
        if not is_real_number(raw_val):
            field_errors[feat] = f"{name} must be a number."
            continue

        val = float(raw_val)

        # 4. Check hard limits
        min_limit, max_limit, limit_msg = HARD_LIMITS[feat]
        if val < min_limit or val > max_limit:
            field_errors[feat] = limit_msg
            continue

        validated_values[feat] = val

    if field_errors:
        return jsonify({
            "error": "Some values are invalid.",
            "fields": field_errors
        }), 400

    # 5. Check soft limits for warnings
    warnings = []
    for feat in FEATURES:
        val = validated_values[feat]
        min_soft, max_soft, name, unit = SOFT_LIMITS[feat]
        if val < min_soft or val > max_soft:
            val_str = _format_value_with_unit(val, unit)
            r_min = _format_num(min_soft)
            r_max = _format_num(max_soft)
            unit_suffix = f" {unit}" if unit and unit != "%" else unit
            range_str = f"({r_min}–{r_max}{unit_suffix})"
            warnings.append(
                f"{name} {val_str} is outside the range the model was trained on "
                f"{range_str}, so results may be less reliable."
            )

    # 6. Predict crop probabilities
    # DataFrame ensures feature columns match training feature order
    input_df = pd.DataFrame(
        [[validated_values[feat] for feat in FEATURES]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    crop_probs = list(zip(classes, probabilities))
    crop_probs.sort(key=lambda item: item[1], reverse=True)

    # Drop any below 5% confidence, but always keep at least 1
    filtered_probs = [cp for cp in crop_probs if cp[1] >= 0.05]
    if not filtered_probs:
        filtered_probs = [crop_probs[0]]

    # Take top 3 crops
    top_crops = filtered_probs[:3]

    recommendations = []
    for crop_label, prob in top_crops:
        crop_info = CROPS.get(crop_label, {})
        display_name = crop_info.get("name", crop_label.capitalize())
        local_name = crop_info.get("local_name", "")
        season = crop_info.get("season", "")

        explanation = explain_crop(
            crop=crop_label,
            values=validated_values,
            crop_ranges=crop_ranges,
            importances=importances,
            confidence=float(prob),
        )

        recommendations.append({
            "crop": crop_label,
            "name": display_name,
            "local_name": local_name,
            "season": season,
            "confidence": round(float(prob), 2),
            "summary": explanation["summary"],
            "reasons": explanation["reasons"],
            "cautions": explanation["cautions"],
        })

    input_labels = describe_inputs(validated_values)

    return jsonify({
        "recommendations": recommendations,
        "input_labels": input_labels,
        "warnings": warnings,
    }), 200


@app.route("/climate", methods=["GET"])
def climate():
    """Season-average temperature, humidity and rainfall for a place in India.

    Query: season=kharif|rabi|zaid, plus either place=<name> or lat=<..>&lon=<..>.
    """
    season = request.args.get("season", "").strip().lower()
    if season not in SEASON_MONTHS:
        return jsonify({"error": "Choose a season: Kharif, Rabi or Zaid."}), 400

    place = request.args.get("place", "").strip()
    try:
        if place:
            if len(place) > 100:
                return jsonify({"error": "Place name is too long."}), 400
            if is_state(place):
                return jsonify({
                    "error": f"{place} is a state, and weather varies a lot across a state. "
                             "Type a city or district in it instead."
                }), 400
            found = find_place(place)
            if found is None:
                return jsonify({
                    "error": f'Could not find "{place}" in India. Check the spelling (in English), '
                             'or try a nearby city or district.'
                }), 404
            lat, lon, label = found["lat"], found["lon"], found["label"]
        else:
            try:
                lat = float(request.args.get("lat", ""))
                lon = float(request.args.get("lon", ""))
            except ValueError:
                return jsonify({"error": "Type a place name or use your location."}), 400
            if not in_india(lat, lon):
                return jsonify({"error": "That location is outside India."}), 400
            label = "your location"

        values = fetch_season_climate(lat, lon, season)
    except ClimateError as err:
        return jsonify({"error": str(err)}), 502

    return jsonify({"place": label, "season": season, **values}), 200


@app.errorhandler(500)
def handle_500(err):
    logger.error(f"Server error: {err}", exc_info=True)
    return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.errorhandler(Exception)
def handle_unexpected(err):
    if isinstance(err, HTTPException):
        return err
    logger.error(f"Unexpected error: {err}", exc_info=True)
    return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
