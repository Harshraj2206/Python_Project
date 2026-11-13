"""Rule-based explanation layer for AgriSmart crop recommendation.

Provides plain-language descriptions of soil/climate inputs and builds
human-readable explanations (matches, cautions, and summaries) for why
a crop is or is not recommended.
"""

from crops import CROPS

# Feature order defined once here and imported across the application
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Display names for features in explanation sentences
FEATURE_DISPLAY_NAMES = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "pH",
    "rainfall": "Rainfall",
}

# Units for features (°C, %, mm; pH and N/P/K have no unit)
FEATURE_UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "rainfall": "mm",
    "ph": "",
    "N": "",
    "P": "",
    "K": "",
}


def describe_inputs(values: dict) -> dict[str, str]:
    """Converts raw input values into plain-English qualitative labels.

    Uses fixed bands defined in section 7a of the project spec.
    Values exactly on a boundary go to the higher band.
    """
    labels = {}

    for feature in FEATURES:
        if feature not in values:
            continue
        val = values[feature]

        if feature == "ph":
            if val < 5.5:
                labels[feature] = "strongly acidic soil"
            elif val < 6.5:
                labels[feature] = "slightly acidic soil"
            elif val < 7.5:
                labels[feature] = "neutral soil"
            elif val < 8.5:
                labels[feature] = "alkaline soil"
            else:
                labels[feature] = "strongly alkaline soil"

        elif feature == "rainfall":
            if val < 75:
                labels[feature] = "low rainfall"
            elif val < 200:
                labels[feature] = "moderate rainfall"
            else:
                labels[feature] = "high rainfall"

        elif feature == "temperature":
            if val < 18:
                labels[feature] = "cool climate"
            elif val < 28:
                labels[feature] = "mild climate"
            else:
                labels[feature] = "hot climate"

        elif feature == "humidity":
            if val < 40:
                labels[feature] = "dry air"
            elif val < 70:
                labels[feature] = "moderate humidity"
            else:
                labels[feature] = "humid climate"

        elif feature == "N":
            if val < 40:
                labels[feature] = "low nitrogen"
            elif val < 90:
                labels[feature] = "medium nitrogen"
            else:
                labels[feature] = "high nitrogen"

        elif feature == "P":
            if val < 30:
                labels[feature] = "low phosphorus"
            elif val < 70:
                labels[feature] = "medium phosphorus"
            else:
                labels[feature] = "high phosphorus"

        elif feature == "K":
            if val < 30:
                labels[feature] = "low potassium"
            elif val < 80:
                labels[feature] = "medium potassium"
            else:
                labels[feature] = "high potassium"

    return labels


def _format_num(val: float) -> str:
    """Formats a number with at most 1 decimal place, omitting trailing .0 for integers."""
    rounded = round(val, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def _format_value_with_unit(val: float, unit: str) -> str:
    """Formats a value and its unit (e.g., '230 mm', '82%', '8.1')."""
    num_str = _format_num(val)
    if not unit:
        return num_str
    if unit == "%":
        return f"{num_str}%"
    return f"{num_str} {unit}"


def _format_range_with_unit(p25: float, p75: float, unit: str) -> str:
    """Formats an interquartile range with en-dash and unit (e.g., '(191–259 mm)', '(5.6–6.9)')."""
    r25 = _format_num(p25)
    r75 = _format_num(p75)
    if not unit:
        return f"({r25}–{r75})"
    if unit == "%":
        return f"({r25}–{r75}%)"
    return f"({r25}–{r75} {unit})"


def explain_crop(
    crop: str,
    values: dict,
    crop_ranges: dict,
    importances: dict,
    confidence: float = 1.0,
) -> dict:
    """Generates reasons, cautions, and a summary explanation for a crop recommendation.

    Args:
        crop: The crop key (e.g., 'rice').
        values: User input values for each feature.
        crop_ranges: Per-crop 25th and 75th percentiles {crop: {feature: (p25, p75)}}.
        importances: Feature importances {feature: float}.
        confidence: The model's probability for this crop (0–1). Picks the summary
            wording, so a 9% alternative is never called a "strong fit".

    Returns:
        A dictionary with 'summary', 'reasons', and 'cautions'.
    """
    display_name = CROPS.get(crop, {}).get("name", crop.capitalize())
    ranges_for_crop = crop_ranges.get(crop, {})
    input_descriptions = describe_inputs(values)

    matched_features = []
    caution_items = []

    for feat in FEATURES:
        if feat not in values or feat not in ranges_for_crop:
            continue

        val = values[feat]
        p25, p75 = ranges_for_crop[feat]
        feat_name = FEATURE_DISPLAY_NAMES.get(feat, feat.capitalize())
        unit = FEATURE_UNITS.get(feat, "")

        val_str = _format_value_with_unit(val, unit)
        range_str = _format_range_with_unit(p25, p75, unit)

        if p25 <= val <= p75:
            # Inside typical range (inclusive)
            text = f"{feat_name} {val_str} is within {display_name}'s typical range {range_str}"
            matched_features.append((feat, text))
        elif val > p75:
            # Above typical range
            text = f"{feat_name} {val_str} is higher than {display_name}'s typical range {range_str}"
            caution_items.append((feat, text))
        else:
            # Below typical range
            text = f"{feat_name} {val_str} is lower than {display_name}'s typical range {range_str}"
            caution_items.append((feat, text))

    # Sort by feature importance (highest first)
    matched_features.sort(key=lambda item: importances.get(item[0], 0.0), reverse=True)
    caution_items.sort(key=lambda item: importances.get(item[0], 0.0), reverse=True)

    # Return at most 3 matches and at most 2 cautions
    top_matches = matched_features[:3]
    top_cautions = caution_items[:2]

    reasons = [text for _, text in top_matches]
    cautions = [text for _, text in top_cautions]

    # Build summary sentence
    if top_matches:
        top_labels = [
            input_descriptions[feat]
            for feat, _ in top_matches
            if feat in input_descriptions
        ]
        joined_labels = " + ".join(top_labels)
        # Capitalize the first character of the joined sentence
        if joined_labels:
            joined_labels = joined_labels[0].upper() + joined_labels[1:]
        if confidence >= 0.6:
            verdict = "is a strong fit"
        elif confidence >= 0.25:
            verdict = "is a possible fit"
        else:
            verdict = "is a weaker alternative"
        summary = f"{joined_labels} → {display_name} {verdict}"
    else:
        summary = f"→ {display_name} is the closest match, but conditions are not ideal"

    return {
        "summary": summary,
        "reasons": reasons,
        "cautions": cautions,
    }
