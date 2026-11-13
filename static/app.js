/**
 * AgriSmart — Frontend Client Controller
 * Offline-first, vanilla JavaScript implementation strictly adhering to PROJECT_SPEC.md section 9.
 *
 * Key guarantees:
 * 1. Zero external network dependencies or CDNs.
 * 2. 100% of data values from server rendered using textContent (never innerHTML).
 * 3. Client-side validation before sending with per-field errors.
 * 4. Button disabled with "Analysing…" during request.
 * 5. 400 error handling mapping server 'fields' dictionary directly to input errors.
 * 6. Network/500 errors displayed in a single friendly banner with retry.
 * 7. Combined seasons split on ' / ' into individual Apple-style badges.
 * 8. Warnings notice hidden when warnings is empty.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnText = submitBtn.querySelector('.btn-text');
    const exampleBtn = document.getElementById('example-btn');
    const resetBtn = document.getElementById('reset-btn');
    const retryBtn = document.getElementById('retry-btn');

    const emptyState = document.getElementById('empty-state');
    const resultsContent = document.getElementById('results-content');
    const resultsSection = document.getElementById('results-section');
    const errorBanner = document.getElementById('error-banner');
    const errorBannerMessage = document.getElementById('error-banner-message');

    const inputLabelsContainer = document.getElementById('input-labels-container');
    const warningsBanner = document.getElementById('warnings-banner');
    const warningsList = document.getElementById('warnings-list');
    const recommendationsList = document.getElementById('recommendations-list');

    const locationInput = document.getElementById('location-input');
    const seasonSelect = document.getElementById('season-select');
    const locationBtn = document.getElementById('location-btn');
    const geoBtn = document.getElementById('geo-btn');
    const locationStatus = document.getElementById('location-status');

    // Field configuration matching PROJECT_SPEC.md hard limits
    const FIELD_CONFIGS = [
        { name: 'N', label: 'Nitrogen', min: 0, max: 300, unit: 'ratio' },
        { name: 'P', label: 'Phosphorus', min: 0, max: 300, unit: 'ratio' },
        { name: 'K', label: 'Potassium', min: 0, max: 300, unit: 'ratio' },
        { name: 'ph', label: 'pH', min: 0, max: 14, unit: 'scale 0–14' },
        { name: 'temperature', label: 'Temperature', min: -10, max: 55, unit: '°C' },
        { name: 'humidity', label: 'Humidity', min: 0, max: 100, unit: '%' },
        { name: 'rainfall', label: 'Rainfall', min: 0, max: 5000, unit: 'mm' }
    ];

    // Standard Example Data from PROJECT_SPEC.md section 9
    const EXAMPLE_DATA = {
        N: 90,
        P: 42,
        K: 43,
        temperature: 20.9,
        humidity: 82.0,
        ph: 6.5,
        rainfall: 202.9
    };

    /**
     * Clears all client-side and server-side validation error messages
     */
    function clearFieldErrors() {
        FIELD_CONFIGS.forEach(cfg => {
            const errorDiv = document.getElementById(`error-${cfg.name}`);
            const inputWrapper = document.getElementById(`group-${cfg.name}`).querySelector('.input-wrapper');
            if (errorDiv) {
                errorDiv.textContent = '';
            }
            if (inputWrapper) {
                inputWrapper.classList.remove('has-error');
            }
        });
        hideGlobalError();
    }

    /**
     * Sets an error message on a specific input field
     */
    function setFieldError(fieldName, message) {
        const errorDiv = document.getElementById(`error-${fieldName}`);
        const group = document.getElementById(`group-${fieldName}`);
        if (errorDiv && group) {
            errorDiv.textContent = message;
            const inputWrapper = group.querySelector('.input-wrapper');
            if (inputWrapper) {
                inputWrapper.classList.add('has-error');
            }
        }
    }

    /**
     * Displays the friendly network/server error banner
     */
    function showGlobalError(message) {
        errorBannerMessage.textContent = message || 'Unable to connect to the recommendation service. Please check your network and try again.';
        errorBanner.hidden = false;
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Hides the global error banner
     */
    function hideGlobalError() {
        errorBanner.hidden = true;
    }

    /**
     * Reads and validates form inputs in the browser before sending
     * Returns the payload object if valid, or null if errors exist.
     */
    function validateForm() {
        clearFieldErrors();
        let isValid = true;
        let firstInvalidInput = null;
        const payload = {};

        FIELD_CONFIGS.forEach(cfg => {
            const inputEl = document.getElementById(`input-${cfg.name}`);
            const rawValue = inputEl.value.trim();

            if (rawValue === '') {
                setFieldError(cfg.name, `Please enter a value for ${cfg.label}.`);
                isValid = false;
                if (!firstInvalidInput) firstInvalidInput = inputEl;
                return;
            }

            const numValue = Number(rawValue);

            if (Number.isNaN(numValue) || !Number.isFinite(numValue)) {
                setFieldError(cfg.name, `Please enter a valid number for ${cfg.label}.`);
                isValid = false;
                if (!firstInvalidInput) firstInvalidInput = inputEl;
                return;
            }

            if (numValue < cfg.min || numValue > cfg.max) {
                if (cfg.name === 'ph') {
                    setFieldError(cfg.name, 'pH must be between 0 and 14.');
                } else if (cfg.name === 'temperature') {
                    setFieldError(cfg.name, 'Temperature must be between -10 and 55 °C.');
                } else if (cfg.name === 'humidity') {
                    setFieldError(cfg.name, 'Humidity must be between 0 and 100%.');
                } else if (cfg.name === 'rainfall') {
                    setFieldError(cfg.name, 'Rainfall must be between 0 and 5000 mm.');
                } else {
                    setFieldError(cfg.name, `${cfg.label} must be between ${cfg.min} and ${cfg.max}.`);
                }
                isValid = false;
                if (!firstInvalidInput) firstInvalidInput = inputEl;
                return;
            }

            // Valid numeric parameter
            payload[cfg.name] = numValue;
        });

        if (!isValid && firstInvalidInput) {
            firstInvalidInput.focus();
            return null;
        }

        return payload;
    }

    /**
     * Populates input fields with an object of values
     */
    function populateFields(data) {
        clearFieldErrors();
        FIELD_CONFIGS.forEach(cfg => {
            const inputEl = document.getElementById(`input-${cfg.name}`);
            if (inputEl && data[cfg.name] !== undefined) {
                inputEl.value = data[cfg.name];
            }
        });
    }

    /**
     * Returns the appropriate CSS modifier class for a season name
     */
    function getSeasonClass(seasonName) {
        const s = seasonName.toLowerCase().trim();
        if (s.includes('kharif')) return 'season-kharif';
        if (s.includes('rabi')) return 'season-rabi';
        if (s.includes('zaid')) return 'season-zaid';
        if (s.includes('perennial')) return 'season-perennial';
        return 'season-kharif';
    }

    /**
     * Renders a crop recommendation card strictly using DOM methods & textContent
     */
    function createCropCard(rec, index) {
        const card = document.createElement('article');
        card.className = 'card crop-card';

        // --- Card Top Header ---
        const topRow = document.createElement('div');
        topRow.className = 'crop-card-top';

        const titleGroup = document.createElement('div');
        titleGroup.className = 'crop-title-group';

        const title = document.createElement('h3');
        title.className = 'crop-name';
        // Skip the local name when it is the same word (e.g. "Coffee (Coffee)")
        title.textContent = rec.local_name && rec.local_name !== rec.name
            ? `${rec.name} (${rec.local_name})`
            : rec.name;
        titleGroup.appendChild(title);

        // Combined season badge parser: split on " / " and render one pill per season
        const seasonWrap = document.createElement('div');
        seasonWrap.className = 'season-badges-wrap';
        const seasons = rec.season ? rec.season.split(' / ') : [];
        seasons.forEach(seasonText => {
            const pill = document.createElement('span');
            pill.className = `season-pill ${getSeasonClass(seasonText)}`;
            pill.textContent = seasonText.trim();
            seasonWrap.appendChild(pill);
        });
        titleGroup.appendChild(seasonWrap);
        topRow.appendChild(titleGroup);

        // Rank Badge
        const rankBadge = document.createElement('span');
        if (index === 0) {
            rankBadge.className = 'badge badge-rank';
            rankBadge.textContent = '#1 Best Recommendation';
        } else {
            rankBadge.className = 'badge badge-alt';
            rankBadge.textContent = `#${index + 1} Alternative`;
        }
        topRow.appendChild(rankBadge);
        card.appendChild(topRow);

        // --- Model Confidence Bar ---
        const confBlock = document.createElement('div');
        confBlock.className = 'confidence-block';

        const confLabelRow = document.createElement('div');
        confLabelRow.className = 'confidence-label-row';

        const confTitle = document.createElement('span');
        confTitle.textContent = 'Model confidence';

        const confPercent = document.createElement('span');
        const percentage = Math.round(rec.confidence * 100);
        confPercent.textContent = `${percentage}%`;

        confLabelRow.appendChild(confTitle);
        confLabelRow.appendChild(confPercent);
        confBlock.appendChild(confLabelRow);

        const confTrack = document.createElement('div');
        confTrack.className = 'confidence-track';
        const confFill = document.createElement('div');
        confFill.className = 'confidence-fill';
        confFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        confTrack.appendChild(confFill);
        confBlock.appendChild(confTrack);
        card.appendChild(confBlock);

        // --- API Agronomic Summary Statement ---
        const summaryBox = document.createElement('div');
        summaryBox.className = 'summary-box';
        summaryBox.textContent = rec.summary;
        card.appendChild(summaryBox);

        // --- Helper to build the reasons & cautions grid ---
        function buildDetailsGrid() {
            const grid = document.createElement('div');
            grid.className = 'details-grid';

            // Key Strengths Box
            const reasonsBox = document.createElement('div');
            reasonsBox.className = 'reasons-box';

            const reasonsHeader = document.createElement('h4');
            reasonsHeader.className = 'reasons-title';
            reasonsHeader.textContent = 'Key Strengths';
            reasonsBox.appendChild(reasonsHeader);

            const reasonsList = document.createElement('ul');
            reasonsList.className = 'bullet-list';
            if (rec.reasons && rec.reasons.length > 0) {
                rec.reasons.forEach(r => {
                    const li = document.createElement('li');
                    li.className = 'bullet-item';
                    li.textContent = r;
                    reasonsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'bullet-item';
                li.textContent = "None of your values fall inside this crop's typical range.";
                reasonsList.appendChild(li);
            }
            reasonsBox.appendChild(reasonsList);
            grid.appendChild(reasonsBox);

            // Keep in Mind Box (Cautions)
            if (rec.cautions && rec.cautions.length > 0) {
                const cautionsBox = document.createElement('div');
                cautionsBox.className = 'cautions-box';

                const cautionsHeader = document.createElement('h4');
                cautionsHeader.className = 'cautions-title';
                cautionsHeader.textContent = 'Keep in Mind';
                cautionsBox.appendChild(cautionsHeader);

                const cautionsList = document.createElement('ul');
                cautionsList.className = 'bullet-list';
                rec.cautions.forEach(c => {
                    const li = document.createElement('li');
                    li.className = 'bullet-item';
                    li.textContent = c;
                    cautionsList.appendChild(li);
                });
                cautionsBox.appendChild(cautionsList);
                grid.appendChild(cautionsBox);
            }

            return grid;
        }

        // Card #1 shows details directly. Cards #2 and #3 show a "Show details" toggle
        if (index === 0) {
            card.appendChild(buildDetailsGrid());
        } else {
            const toggleBtn = document.createElement('button');
            toggleBtn.type = 'button';
            toggleBtn.className = 'details-toggle-btn';
            toggleBtn.setAttribute('aria-expanded', 'false');

            const btnText = document.createElement('span');
            btnText.textContent = 'Show details';
            toggleBtn.appendChild(btnText);

            // Inline Chevron SVG
            const chevronSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            chevronSvg.setAttribute('class', 'chevron-icon');
            chevronSvg.setAttribute('width', '16');
            chevronSvg.setAttribute('height', '16');
            chevronSvg.setAttribute('viewBox', '0 0 24 24');
            chevronSvg.setAttribute('fill', 'none');
            chevronSvg.setAttribute('stroke', 'currentColor');
            chevronSvg.setAttribute('stroke-width', '2.2');
            chevronSvg.setAttribute('stroke-linecap', 'round');
            chevronSvg.setAttribute('stroke-linejoin', 'round');
            chevronSvg.setAttribute('aria-hidden', 'true');

            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('points', '6 9 12 15 18 9');
            chevronSvg.appendChild(polyline);
            toggleBtn.appendChild(chevronSvg);

            const detailsDrawer = document.createElement('div');
            detailsDrawer.className = 'details-drawer';
            detailsDrawer.hidden = true;
            detailsDrawer.appendChild(buildDetailsGrid());

            toggleBtn.addEventListener('click', () => {
                const isExpanded = toggleBtn.getAttribute('aria-expanded') === 'true';
                toggleBtn.setAttribute('aria-expanded', String(!isExpanded));
                detailsDrawer.hidden = isExpanded;
                btnText.textContent = isExpanded ? 'Show details' : 'Hide details';
            });

            card.appendChild(toggleBtn);
            card.appendChild(detailsDrawer);
        }

        return card;
    }

    /**
     * Renders server response into the UI
     */
    function renderResults(data) {
        // Clear previous results
        inputLabelsContainer.textContent = '';
        warningsList.textContent = '';
        recommendationsList.textContent = '';

        // Render input label chips
        if (data.input_labels && typeof data.input_labels === 'object') {
            Object.values(data.input_labels).forEach(label => {
                const chip = document.createElement('span');
                chip.className = 'input-chip';

                const dot = document.createElement('span');
                dot.className = 'chip-dot';
                dot.setAttribute('aria-hidden', 'true');

                const text = document.createElement('span');
                text.textContent = label;

                chip.appendChild(dot);
                chip.appendChild(text);
                inputLabelsContainer.appendChild(chip);
            });
        }

        // Render advisory warnings notice (Hidden when empty)
        if (data.warnings && Array.isArray(data.warnings) && data.warnings.length > 0) {
            data.warnings.forEach(w => {
                const item = document.createElement('div');
                item.className = 'warning-item';
                item.textContent = w;
                warningsList.appendChild(item);
            });
            warningsBanner.hidden = false;
        } else {
            warningsBanner.hidden = true;
        }

        // Render recommendation cards
        if (data.recommendations && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
            data.recommendations.forEach((rec, idx) => {
                const card = createCropCard(rec, idx);
                recommendationsList.appendChild(card);
            });
        }

        // Switch visible view from empty state to live results
        emptyState.hidden = true;
        resultsContent.hidden = false;

        // Smooth scroll to results on small viewports
        if (window.innerWidth <= 1024) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    /**
     * Submits the prediction request to POST /predict
     */
    async function submitPrediction() {
        const payload = validateForm();
        if (!payload) return;

        // Loading state (example button too, so two requests can't race each other)
        submitBtn.disabled = true;
        exampleBtn.disabled = true;
        submitBtnText.textContent = 'Analysing…';
        hideGlobalError();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json().catch(() => null);

            if (response.ok && data) {
                renderResults(data);
            } else if (response.status === 400 && data) {
                // Map server validation errors back to field containers
                if (data.fields && typeof data.fields === 'object') {
                    let firstErrField = null;
                    Object.entries(data.fields).forEach(([fieldName, msg]) => {
                        setFieldError(fieldName, msg);
                        if (!firstErrField) {
                            firstErrField = document.getElementById(`input-${fieldName}`);
                        }
                    });
                    if (firstErrField) firstErrField.focus();
                } else if (data.error) {
                    showGlobalError(data.error);
                } else {
                    showGlobalError('Invalid parameters provided. Please review input fields.');
                }
            } else {
                // Server 500 or unknown error
                const errorMsg = (data && data.error) ? data.error : 'Something went wrong. Please try again.';
                showGlobalError(errorMsg);
            }
        } catch (err) {
            // Network failure or offline
            showGlobalError('Network connection issue. Please verify your connection and retry.');
        } finally {
            submitBtn.disabled = false;
            exampleBtn.disabled = false;
            submitBtnText.textContent = 'Get Recommendation';
        }
    }

    function setLocationStatus(message, isError) {
        locationStatus.textContent = message;
        locationStatus.classList.toggle('is-error', Boolean(isError));
    }

    /**
     * Asks the server for season-average weather and fills the 3 weather fields.
     * params is either { place } or { lat, lon }.
     */
    async function fillWeather(params) {
        locationBtn.disabled = true;
        geoBtn.disabled = true;
        setLocationStatus('Fetching 5 years of weather…');

        try {
            const query = new URLSearchParams({ season: seasonSelect.value, ...params });
            const response = await fetch(`/climate?${query}`, { headers: { 'Accept': 'application/json' } });
            const data = await response.json().catch(() => null);

            if (!response.ok || !data) {
                setLocationStatus((data && data.error) || 'Could not get the weather right now. Enter the values manually.', true);
                return;
            }

            ['temperature', 'humidity', 'rainfall'].forEach(name => {
                const inputEl = document.getElementById(`input-${name}`);
                inputEl.value = data[name];
                inputEl.dispatchEvent(new Event('input')); // clears any old error on the field
            });
            const seasonName = seasonSelect.options[seasonSelect.selectedIndex].text.split(' ')[0];
            setLocationStatus(
                `Filled with ${seasonName} averages for ${data.place} (${data.start_year}–${data.end_year}, Open-Meteo). ` +
                'Now enter N, P, K and pH from your Soil Health Card.'
            );
        } catch (err) {
            setLocationStatus('No internet connection. Enter the weather values manually.', true);
        } finally {
            locationBtn.disabled = false;
            geoBtn.disabled = false;
        }
    }

    locationBtn.addEventListener('click', () => {
        const place = locationInput.value.trim();
        if (!place) {
            setLocationStatus('Type a city or district first.', true);
            locationInput.focus();
            return;
        }
        fillWeather({ place });
    });

    // Enter in the location box should fill weather, not submit the whole form
    locationInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            locationBtn.click();
        }
    });

    geoBtn.addEventListener('click', () => {
        if (!navigator.geolocation) {
            setLocationStatus('Your browser cannot share its location. Type a place name instead.', true);
            return;
        }
        setLocationStatus('Getting your location…');
        navigator.geolocation.getCurrentPosition(
            (pos) => fillWeather({
                lat: pos.coords.latitude.toFixed(4),
                lon: pos.coords.longitude.toFixed(4)
            }),
            (err) => setLocationStatus(
                err.code === err.PERMISSION_DENIED
                    ? 'Location permission was denied. Type a place name instead.'
                    : 'Could not get your location. Type a place name instead.',
                true
            ),
            { timeout: 10000 }
        );
    });

    // Event Listeners
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        submitPrediction();
    });

    exampleBtn.addEventListener('click', () => {
        populateFields(EXAMPLE_DATA);
        hideGlobalError();
        // Automatically run prediction for immediate demonstration
        submitPrediction();
    });

    resetBtn.addEventListener('click', () => {
        form.reset();
        clearFieldErrors();
        resultsContent.hidden = true;
        emptyState.hidden = false;
        recommendationsList.textContent = '';
        inputLabelsContainer.textContent = '';
        warningsList.textContent = '';
        warningsBanner.hidden = true;
        hideGlobalError();
        setLocationStatus('');
        const firstInput = document.getElementById('input-N');
        if (firstInput) firstInput.focus();
    });

    retryBtn.addEventListener('click', () => {
        submitPrediction();
    });

    // Real-time error dismissal when user types in a field
    FIELD_CONFIGS.forEach(cfg => {
        const inputEl = document.getElementById(`input-${cfg.name}`);
        if (inputEl) {
            inputEl.addEventListener('input', () => {
                const errorDiv = document.getElementById(`error-${cfg.name}`);
                const group = document.getElementById(`group-${cfg.name}`);
                if (errorDiv) errorDiv.textContent = '';
                if (group) {
                    const inputWrapper = group.querySelector('.input-wrapper');
                    if (inputWrapper) inputWrapper.classList.remove('has-error');
                }
            });
        }
    });
});
