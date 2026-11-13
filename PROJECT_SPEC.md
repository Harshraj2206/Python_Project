# AgriSmart — Project Spec

> **For the AI coding agent:** Build exactly what this document describes. Do not add features, pages, databases, logins, or libraries that are not listed here. If something is unclear, ask instead of guessing. Build in the order given in "Build Order" and stop at each checkpoint.

## 1. What it is

AgriSmart is a Python web app for **Indian farmers and students**. The user enters seven soil and climate values. The app recommends the **top 3 crops**, each with a confidence score and a plain-language reason (for example: "High rainfall + slightly acidic soil + humid climate → Rice is a strong fit").

- **Model:** Random Forest classifier (scikit-learn), trained on the Kaggle Crop Recommendation Dataset.
- **Explanation:** a simple rule-based layer. No ML explainability libraries (no SHAP, no LIME).
- **Interface:** one web page, served by Flask. The UI design comes from Stitch.

## 2. Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn, pandas, joblib |
| Web server | Flask |
| Frontend | One HTML page + CSS + vanilla JavaScript (from Stitch). No React, no build step. |
| Tests | pytest |

`requirements.txt` must pin exact versions (`pip freeze` of only these packages).

## 3. Folder structure

```
agrismart/
├── data/
│   └── Crop_recommendation.csv     # downloaded manually from Kaggle
├── model/
│   └── agrismart.joblib            # created by train.py (not committed)
├── templates/
│   └── index.html                  # the single page (Stitch design)
├── static/
│   ├── style.css
│   └── app.js
├── crops.py                        # Indian crop info table (names, seasons)
├── train.py                        # trains and saves the model bundle
├── explain.py                      # rule-based explanation layer
├── app.py                          # Flask app: GET / and POST /predict
├── test_explain.py
├── test_app.py
├── requirements.txt
├── .gitignore                      # model/, __pycache__/, .venv/
└── README.md
```

**Separation rule:** `train.py`, `explain.py`, and `crops.py` must not import Flask or know about the UI. `app.py` only glues them together. The frontend only talks to `/predict`.

## 4. Dataset

- Source: Kaggle "Crop Recommendation Dataset" (Atharva Ingle), file `Crop_recommendation.csv`.
- 2,200 rows, 22 crops, 100 rows per crop.
- Columns: `N, P, K, temperature, humidity, ph, rainfall, label`.

Feature order, used everywhere (define once in `explain.py` and import it elsewhere):

```python
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
```

| Feature | Meaning | Unit | Dataset range |
|---|---|---|---|
| N | Nitrogen ratio in soil | ratio | 0 – 140 |
| P | Phosphorus ratio in soil | ratio | 5 – 145 |
| K | Potassium ratio in soil | ratio | 5 – 205 |
| temperature | Average temperature | °C | 8.8 – 43.7 |
| humidity | Relative humidity | % | 14.3 – 100 |
| ph | Soil pH | 0–14 | 3.5 – 9.9 |
| rainfall | Rainfall | mm | 20.2 – 298.6 |

**Known limitation (put this in the README):** the dataset's N, P, K values are ratios, not the kg/ha values printed on a Government of India Soil Health Card. Values typed directly from a card may give less accurate results.

## 5. `crops.py` — Indian crop info

One dictionary keyed by the dataset label. No logic in this file.

| label | Display name | Local name | Season |
|---|---|---|---|
| rice | Rice | Dhaan | Kharif |
| maize | Maize | Makka | Kharif |
| chickpea | Chickpea | Chana | Rabi |
| kidneybeans | Kidney Beans | Rajma | Rabi |
| pigeonpeas | Pigeon Peas | Arhar / Tur | Kharif |
| mothbeans | Moth Beans | Moth | Kharif |
| mungbean | Mung Bean | Moong | Kharif / Zaid |
| blackgram | Black Gram | Urad | Kharif / Zaid |
| lentil | Lentil | Masoor | Rabi |
| pomegranate | Pomegranate | Anar | Perennial |
| banana | Banana | Kela | Perennial |
| mango | Mango | Aam | Perennial |
| grapes | Grapes | Angoor | Perennial |
| watermelon | Watermelon | Tarbooz | Zaid |
| muskmelon | Muskmelon | Kharbooja | Zaid |
| apple | Apple | Seb | Perennial |
| orange | Orange | Santra | Perennial |
| papaya | Papaya | Papita | Perennial |
| coconut | Coconut | Nariyal | Perennial |
| cotton | Cotton | Kapas | Kharif |
| jute | Jute | Patsan | Kharif |
| coffee | Coffee | Coffee | Perennial |

Season meanings (show these as a tooltip or footnote in the UI):
- **Kharif:** sown with the monsoon (June–July), harvested Sept–Oct.
- **Rabi:** sown in winter (Oct–Nov), harvested March–April.
- **Zaid:** short summer season between Rabi and Kharif (March–June).
- **Perennial:** a tree or plantation crop that grows for many years.

## 6. `train.py`

When run with `python train.py`, it must:

1. Load `data/Crop_recommendation.csv`. If the file is missing, print a clear message with the Kaggle dataset name and exit with code 1.
2. Do a stratified 80/20 train/test split with `random_state=42`.
3. Train `RandomForestClassifier(n_estimators=200, random_state=42)` on the **training set only**.
4. Print:
   - test accuracy
   - `classification_report` on the test set
   - 5-fold cross-validation mean ± std accuracy, run on the training set
   - feature importances, sorted highest first
5. Compute **per-crop typical ranges** from the **training set only**: the 25th and 75th percentile of each feature for each crop.
6. Save one bundle with `joblib` to `model/agrismart.joblib` (create the folder if needed):
   ```python
   {
       "model": <fitted RandomForestClassifier>,
       "features": FEATURES,
       "crop_ranges": {crop: {feature: (p25, p75)}},
       "importances": {feature: float},
   }
   ```

Expected test accuracy is about 99%, because the dataset is very clean. The README must say that real field data would score lower.

## 7. `explain.py` — rule-based explanation layer

Pure functions only. No file reading, no Flask.

### 7a. Descriptive labels: `describe_inputs(values) -> dict[str, str]`

Converts each raw input into plain words, using these fixed bands:

| Feature | Bands |
|---|---|
| ph | < 5.5 "strongly acidic soil" · 5.5–6.5 "slightly acidic soil" · 6.5–7.5 "neutral soil" · 7.5–8.5 "alkaline soil" · > 8.5 "strongly alkaline soil" |
| rainfall (mm) | < 75 "low rainfall" · 75–200 "moderate rainfall" · > 200 "high rainfall" |
| temperature (°C) | < 18 "cool climate" · 18–28 "mild climate" · > 28 "hot climate" |
| humidity (%) | < 40 "dry air" · 40–70 "moderate humidity" · > 70 "humid climate" |
| N | < 40 "low nitrogen" · 40–90 "medium nitrogen" · > 90 "high nitrogen" |
| P | < 30 "low phosphorus" · 30–70 "medium phosphorus" · > 70 "high phosphorus" |
| K | < 30 "low potassium" · 30–80 "medium potassium" · > 80 "high potassium" |

Boundaries: a value exactly on a boundary goes to the **higher** band (for example, pH 5.5 is "slightly acidic soil").

### 7b. Crop fit: `explain_crop(crop, values, crop_ranges, importances) -> dict`

For one crop:
- **matches:** features whose input lies inside that crop's (p25, p75) range, inclusive. Text example: `"Rainfall 230 mm is within Rice's typical range (191–259 mm)"`.
- **cautions:** features outside that range. Text example: `"pH 8.1 is higher than Rice's typical range (5.6–6.9)"`, using "higher than" or "lower than".
- Sort both lists by feature importance, highest first. Return at most 3 matches and at most 2 cautions.
- **summary:** take the descriptive labels (from 7a) of the top matched features, at most 3, join them with " + ", and end with "→ {Display name} {verdict}". The verdict follows the model's confidence for that crop (passed in as `confidence`): ≥ 0.6 "is a strong fit", ≥ 0.25 "is a possible fit", otherwise "is a weaker alternative". If nothing matches, use "→ {Display name} is the closest match, but conditions are not ideal".
- Format numbers with at most 1 decimal place and include units (°C, %, mm; pH and N/P/K have no unit).

Return value:

```python
{"summary": str, "reasons": [str, ...], "cautions": [str, ...]}
```

## 8. `app.py` — Flask

- Load `model/agrismart.joblib` **once at startup**, never per request. If the file is missing, exit with the message "Run `python train.py` first."
- `GET /` renders `templates/index.html`.
- `POST /predict` accepts and returns JSON.

### Request

```json
{"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
```

### Validation (server side; never trust the frontend)

1. Every one of the 7 fields must be present and a real number. Booleans, strings, NaN, and Infinity are rejected. The string `"5"` is rejected too: the frontend must send numbers.
2. **Hard limits.** Reject with 400 if a value is outside:
   - N, P, K: 0 – 300
   - temperature: −10 – 55
   - humidity: 0 – 100
   - ph: 0 – 14
   - rainfall: 0 – 5000
3. **Soft limits.** If a value is inside the hard limits but outside the dataset range (section 4), still predict, but add a warning: `"Temperature 48 °C is outside the range the model was trained on (8.8–43.7 °C), so results may be less reliable."`
4. Reject a body that is not JSON, or larger than 10 KB, with 400.

### Prediction

- Build a one-row `pandas.DataFrame` with columns in `FEATURES` order, then call `predict_proba`.
- Take the top 3 crops by probability. Drop any below 5% confidence, but always keep at least 1.

### Success response (200)

```json
{
  "recommendations": [
    {
      "crop": "rice",
      "name": "Rice",
      "local_name": "Dhaan",
      "season": "Kharif",
      "confidence": 0.87,
      "summary": "High rainfall + humid climate + neutral soil → Rice is a strong fit",
      "reasons": ["..."],
      "cautions": ["..."]
    }
  ],
  "input_labels": {"ph": "neutral soil", "rainfall": "high rainfall", "...": "..."},
  "warnings": []
}
```

### Error response (400)

```json
{"error": "Some values are invalid.", "fields": {"ph": "pH must be between 0 and 14."}}
```

Never return a stack trace to the browser. Unexpected errors return 500 with `{"error": "Something went wrong. Please try again."}`, and the details are logged on the server.

Run with `python app.py`, on `http://127.0.0.1:5000`, with `debug=False` by default.

## 9. Frontend (design in Stitch, wire up in `app.js`)

One page, mobile-friendly, simple enough for a first-time user.

**Header:** "AgriSmart" plus the tagline "Smart crop recommendations for Indian farms".

**Input form:** 7 number fields, each with a visible `<label>`, a unit, and a one-line hint:
- Nitrogen (N), Phosphorus (P), Potassium (K). Hint: "From your Soil Health Card".
- Soil pH (0–14).
- Temperature (°C), Humidity (%), Rainfall (mm).

Buttons:
- **"Get Recommendation"** (primary).
- **"Try an example"**, which fills in: N 90, P 42, K 43, temperature 20.9, humidity 82, pH 6.5, rainfall 202.9.
- **"Reset"**.

**Behaviour:**
- Do basic checks in the browser before sending (required fields, number ranges), and show the error under the field. The server still validates everything.
- While the request is in flight: disable the button and show "Analysing…". This prevents double submits.
- On a 400: show each `fields` message under its input.
- On a network error or 500: show one friendly banner that offers a retry.
- Render all text from the server with `textContent`, never `innerHTML`.

**Results:** one card per recommendation, best first:
- Crop name with its local name: "Rice (Dhaan)".
- Season badge: Kharif / Rabi / Zaid / Perennial.
- Confidence as a percentage, plus a simple bar.
- The summary line in bold, then the reasons as a list, then the cautions in a muted "Keep in mind" list.
- Above the cards: the `input_labels` as small chips (for example "neutral soil", "high rainfall"), and any `warnings` in a yellow notice.

**Accessibility:** every input has a label; errors are linked with `aria-describedby`; colour is never the only signal; the page works by keyboard.

## 10. Tests

`test_explain.py`:
- pH 5.5 → "slightly acidic soil"; pH 5.49 → "strongly acidic soil"; rainfall 250 → "high rainfall".
- `explain_crop` with a hand-made `crop_ranges` returns the correct matches and cautions, sorted by importance, with at most 3 matches and 2 cautions.
- The "no matches" case gives the "closest match" summary.

`test_app.py` (uses the Flask test client and a real trained bundle):
- Valid example input → 200, 1 to 3 recommendations, confidences between 0 and 1, sorted from highest to lowest.
- Missing field, string value, pH 15, and a non-JSON body → each gives 400 with a `fields` or `error` message.
- Temperature 48 → 200 with a warning.

`pytest` must pass.

## 11. README.md

Must include:
- What the project does (2–3 lines).
- Setup steps: create a venv, `pip install -r requirements.txt`, download the dataset into `data/`, `python train.py`, `python app.py`.
- The output of `train.py`: accuracy, cross-validation score, and feature importances.
- How the explanation layer works (one paragraph).
- The limitations: N/P/K are ratios, not Soil Health Card kg/ha values; the dataset is clean, so real-world accuracy will be lower; the model only knows these 22 crops.

## 11b. Location weather (added after the first build)

Optional bar at the top of the form: the user types a city or district in India (or uses the browser's location) and picks a season. The **server** (`climate.py`, `GET /climate`) fetches the last 5 full years of daily weather from Open-Meteo (free, no API key) and returns season averages, which fill the temperature, humidity and rainfall fields.

- Seasons and months: Kharif Jun–Oct, Rabi Nov–Mar, Zaid Mar–Jun.
- Rainfall = average **monthly** rainfall over the season's months (the dataset's 20–300 mm range matches this, not annual rainfall).
- N, P, K and pH are **never** auto-filled: no public service has field-level soil nutrients for India. They stay manual, from the Soil Health Card.
- `GET /climate?season=…&place=…` or `…&lat=…&lon=…`. Errors: 400 bad input or outside India, 404 place not found, 502 weather service unreachable. All errors are JSON with a user-facing `error` message.
- This is the only external call, and it's made by the server, not the browser. Without internet, the rest of the app works; only this bar shows an error.

## 12. Out of scope — do NOT build

User accounts or login · a database · history of past predictions · auto-filling soil values (N, P, K, pH) · Hindi or other-language UI · fertilizer recommendations · multiple ML models or a model comparison page · Docker or deployment · charts or dashboards · any external API calls other than Open-Meteo in section 11b.

## 13. Build order (stop and check after each step)

1. `crops.py`, `train.py`. **Checkpoint:** `python train.py` prints about 99% accuracy and creates `model/agrismart.joblib`.
2. `explain.py` + `test_explain.py`. **Checkpoint:** `pytest test_explain.py` passes.
3. `app.py` + `test_app.py`. **Checkpoint:** `pytest` passes; `curl` or a REST client call to `/predict` with the example input returns rice as the top crop.
4. Stitch design → `templates/index.html`, `static/style.css`, `static/app.js`. **Checkpoint:** the full flow works in a browser, including the error states.
5. `README.md`, `requirements.txt`, `.gitignore`.
