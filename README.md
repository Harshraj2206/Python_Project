# AgriSmart

AgriSmart is an intelligent crop recommendation web application tailored for Indian agriculture that recommends the most suitable crops based on soil nutrients and local climate conditions. Powered by a Random Forest classifier trained on 2,200 agricultural samples across 22 crops, the system delivers top crop recommendations paired with transparent, rule-based agronomic explanations.

---

## Setup Steps (Windows)

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   If PowerShell says "running scripts is disabled", skip activation and use `.\.venv\Scripts\python` instead of `python` in the commands below.

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Download dataset:**
   Download the "Crop Recommendation Dataset" by Atharva Ingle from Kaggle and place `Crop_recommendation.csv` in the `data/` directory:
   ```
   data/Crop_recommendation.csv
   ```

4. **Train the model:**
   ```powershell
   python train.py
   ```
   This generates the trained bundle at `model/agrismart.joblib`.

5. **Start the web application:**
   ```powershell
   python app.py
   ```

6. **Open in browser:**
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Running the Tests

To execute the full automated test suite (32 tests covering web routes, validation rules, error handling, the explanation engine, and the location weather feature; the weather service is faked, so tests run offline):

```powershell
python -m pytest
```

---

## Model Results

Output from running `python train.py`:

```
Loading dataset from 'D:\University Study Material\agrismart\data\Crop_recommendation.csv'...
Dataset split: 1760 training samples, 440 testing samples.
Training RandomForestClassifier(n_estimators=200, random_state=42)...

==================================================
Test Accuracy: 99.55%
==================================================

Classification Report (Test Set):
              precision    recall  f1-score   support

       apple       1.00      1.00      1.00        20
      banana       1.00      1.00      1.00        20
   blackgram       1.00      0.95      0.97        20
    chickpea       1.00      1.00      1.00        20
     coconut       1.00      1.00      1.00        20
      coffee       1.00      1.00      1.00        20
      cotton       1.00      1.00      1.00        20
      grapes       1.00      1.00      1.00        20
        jute       0.95      1.00      0.98        20
 kidneybeans       1.00      1.00      1.00        20
      lentil       1.00      1.00      1.00        20
       maize       0.95      1.00      0.98        20
       mango       1.00      1.00      1.00        20
   mothbeans       1.00      1.00      1.00        20
    mungbean       1.00      1.00      1.00        20
   muskmelon       1.00      1.00      1.00        20
      orange       1.00      1.00      1.00        20
      papaya       1.00      1.00      1.00        20
  pigeonpeas       1.00      1.00      1.00        20
 pomegranate       1.00      1.00      1.00        20
        rice       1.00      0.95      0.97        20
  watermelon       1.00      1.00      1.00        20

    accuracy                           1.00       440
   macro avg       1.00      1.00      1.00       440
weighted avg       1.00      1.00      1.00       440

Running 5-fold cross-validation on the training set...
5-Fold Cross-Validation Accuracy: 99.43% (+/- 0.54%)

Feature Importances (highest first):
  rainfall    : 0.2196
  humidity    : 0.2171
  K           : 0.1808
  P           : 0.1513
  N           : 0.1034
  temperature : 0.0755
  ph          : 0.0523

Model bundle successfully saved to 'D:\University Study Material\agrismart\model\agrismart.joblib'.
```

---

## Explanation Layer

The explanation layer is a deterministic, rule-based agronomic engine that explains why each crop was recommended without relying on ungrounded generation. For each predicted crop, user inputs are compared against the crop's 25th–75th percentile (interquartile) typical ranges extracted directly from the training data. Matches and cautions are ranked by Random Forest feature importance, displaying at most the top 3 matches and top 2 cautions. Each crop's summary sentence pairs qualitative environmental descriptors (e.g., "High rainfall + humid climate + mild climate") with a verdict that scales with model confidence: predictions with confidence ≥ 60% are described as a "strong fit", confidence ≥ 25% as a "possible fit", and lower confidences as a "weaker alternative" (with a graceful fallback when conditions are not ideal).

---

## Location Weather (Optional)

At the top of the form, type a city or district in India (or press "Use my location") and pick a season (Kharif, Rabi or Zaid). The server downloads the last 5 full years of daily weather from [Open-Meteo](https://open-meteo.com/) (free, no API key) and fills in temperature, humidity and rainfall with that season's averages.

- **Rainfall** is filled as the *average monthly* rainfall over the season, because the dataset's rainfall values (20–300 mm) match that scale, not annual totals (Indian annual rainfall is roughly 300–3,000 mm).
- **N, P, K and pH are not auto-filled.** No public service provides field-level soil nutrients for India, and soil varies from field to field, so these still come from a soil test or Soil Health Card.
- **Needs internet.** Without it, this bar shows an error and everything else works normally with manual entry.
- **Sanity check:** using each crop's typical soil values, real weather from its well-known growing region put the expected crop at #1 for 6 of 9 test locations (rice in Kolkata, cotton in Nagpur, pigeon peas in Kalaburagi, chickpea in Indore, coconut in Kozhikode, coffee in Coorg) and #2 for apple in Shimla and moth beans in Jodhpur.

---

## Limitations

- **Nutrient Representation:** N, P, and K values in the dataset represent laboratory nutrient ratio indexes, not direct kg/ha figures found on a Soil Health Card. Entering values directly from a soil card gives approximate recommendations.
- **Dataset Cleanliness:** The Kaggle dataset is curated and well-separated; real-world agricultural conditions exhibit higher noise, regional micro-climates, and weather fluctuations, resulting in lower real-world accuracy.
- **Fixed Crop Vocabulary:** The model is trained on 22 specific crops and cannot recommend crops outside this predefined set.
- **No Irrigation Awareness:** The model only sees rainfall, so irrigated crops are missed. For example, Ludhiana's Kharif rainfall (about 107 mm/month) is too low for rice in the dataset, even though Punjab grows rice with canal and tube-well irrigation.
- **Fixed Season Windows:** Seasons use North-Indian months (Kharif Jun–Oct). Regions with different calendars are handled poorly; for example, Tamil Nadu's main rice season (Samba, Aug–Jan) depends on the north-east monsoon, so Thanjavur in Kharif gets a wrong top crop.
- **Training Range Assumptions:** Input ranges and soft advisory warnings originate directly from the training data distribution.

---

## Project Structure

- `app.py`: Flask web server providing the single-page interface, the `/predict` recommendation API with validation and rule-based explanations, and the `/climate` location weather API.
- `climate.py`: Looks up a place in India and computes season-average temperature, humidity and monthly rainfall from Open-Meteo weather history.
- `crops.py`: Agricultural metadata dictionary with the English name, local (Hindi) name, and Indian cropping season for each of the 22 crops.
- `explain.py`: Rule-based explanation module generating qualitative input labels, matching reasons, cautions, and confidence-scaled summaries.
- `train.py`: Machine learning training pipeline that trains the Random Forest model, computes feature ranges, and saves `model/agrismart.joblib`.
- `requirements.txt`: Pinned Python dependencies for reproducing the environment.
- `.gitignore`: Git exclusion rules for virtual environments, cache directories, and trained model artifacts.
- `templates/index.html`: Accessible, responsive single-page HTML interface with inline SVG icons and zero CDN dependencies.
- `static/style.css`: Pure hand-written CSS implementing the Apple HIG Liquid Glass design system and responsive layout.
- `static/app.js`: Vanilla JavaScript handling client-side form validation, asynchronous API requests, and dynamic DOM rendering.
- `test_app.py`: Integration tests for Flask routes, input validation, soft warnings, error handling, and HTTP status codes.
- `test_climate.py`: Tests for season averaging, India bounds, and the `/climate` endpoint (with the weather service faked).
- `test_explain.py`: Unit tests for qualitative descriptors, boundary conditions, explanation caps, and confidence-scaled summaries.
- `data/Crop_recommendation.csv`: Kaggle crop recommendation dataset containing 2,200 records across 22 crops with 7 soil/weather features.
