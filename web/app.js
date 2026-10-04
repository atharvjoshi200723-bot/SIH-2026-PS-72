/**
 * VajraDrishti ⚡ Forecaster Dashboard Application Logic
 * Integrates Leaflet Map, Doppler Radar Reflectivity canvas overlay,
 * Playback scrubber, Telemetry status, and Forecaster Approval Gate.
 */

// ── Application State ────────────────────────────────────────────────────────
const state = {
  currentStep: 0,
  isPlaying: false,
  playTimer: null,
  cachedFrames: {},
  latestNowcast: null,
  activeAlertId: null,
  districtLayer: null,
  radarOverlay: null,
  map: null,
  bounds: [
    [19.7, 85.2], // South-West (lat_min, lon_min)
    [20.9, 86.4], // North-East (lat_max, lon_max)
  ],
};

// ── Colormap for Radar Reflectivity (dBZ scale) ──────────────────────────────
function getRadarRGBA(val) {
  if (val < 0.15) return [0, 0, 0, 0]; // Transparent
  if (val < 0.35) {
    // Green (Light rain / early convection)
    const t = (val - 0.15) / 0.2;
    return [46, 204, 113, Math.floor(160 + 80 * t)];
  }
  if (val < 0.50) {
    // Yellow (Moderate rain)
    const t = (val - 0.35) / 0.15;
    return [241, 196, 15, Math.floor(200 + 40 * t)];
  }
  if (val < 0.75) {
    // Orange (Severe thunderstorm)
    const t = (val - 0.50) / 0.25;
    return [230, 126, 34, Math.floor(220 + 35 * t)];
  }
  // Red / Magenta (Extreme lightning core)
  return [231, 76, 60, 255];
}

// ── DOM Elements ─────────────────────────────────────────────────────────────
const els = {
  istClock: document.getElementById('ist-clock'),
  tierBadge: document.getElementById('tier-badge'),
  tierText: document.getElementById('tier-text'),
  tierPulse: document.getElementById('tier-pulse'),
  latencyText: document.getElementById('latency-text'),
  btnRunNowcast: document.getElementById('btn-run-nowcast'),
  selectTierForce: document.getElementById('select-tier-force'),
  selectEventPreset: document.getElementById('select-event-preset'),
  mapLeadVal: document.getElementById('map-lead-val'),
  timelineLeadText: document.getElementById('timeline-lead-text'),
  frameCounter: document.getElementById('frame-counter'),
  leadSlider: document.getElementById('lead-slider'),
  btnPlayPause: document.getElementById('btn-play-pause'),
  playIcon: document.getElementById('play-icon'),
  btnStepPrev: document.getElementById('btn-step-prev'),
  btnStepNext: document.getElementById('btn-step-next'),
  hazardCard: document.getElementById('hazard-card'),
  hazardTitle: document.getElementById('hazard-title'),
  hazardIcon: document.getElementById('hazard-icon'),
  impactCount: document.getElementById('impact-count'),
  districtList: document.getElementById('district-list-container'),
  alertGateStatus: document.getElementById('alert-gate-status'),
  inputForecasterId: document.getElementById('input-forecaster-id'),
  inputForecasterNotes: document.getElementById('input-forecaster-notes'),
  btnApproveAlert: document.getElementById('btn-approve-alert'),
  btnRejectAlert: document.getElementById('btn-reject-alert'),
  btnDownloadCap: document.getElementById('btn-download-cap'),
  capXmlPreview: document.getElementById('cap-xml-preview'),
  metricsModal: document.getElementById('metrics-modal'),
  btnOpenMetrics: document.getElementById('btn-open-metrics'),
  btnCloseMetrics: document.getElementById('btn-close-metrics'),
  metricsTableBody: document.getElementById('metrics-table-body'),
};

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initMap();
  initEventListeners();
  loadDistrictsGeoJSON();
  runNowcastCycle(); // Initial run on startup
});

function initClock() {
  const update = () => {
    const now = new Date();
    // Convert to IST
    const istStr = now.toLocaleTimeString('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour12: false,
    });
    els.istClock.textContent = `${istStr} IST`;
  };
  update();
  setInterval(update, 1000);
}

function initMap() {
  // Center on Cuttack / Khordha lightning zone
  state.map = L.map('map', {
    center: [20.32, 85.8],
    zoom: 9,
    zoomControl: false,
  });

  L.control.zoom({ position: 'bottomright' }).addTo(state.map);

  // CartoDB Dark Matter tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap, © CARTO | IMD VajraDrishti',
    subdomains: 'abcd',
    maxZoom: 18,
  }).addTo(state.map);
}

function initEventListeners() {
  els.btnRunNowcast.addEventListener('click', runNowcastCycle);

  els.leadSlider.addEventListener('input', (e) => {
    setForecastStep(parseInt(e.target.value, 10));
  });

  els.btnPlayPause.addEventListener('click', togglePlayPause);
  els.btnStepPrev.addEventListener('click', () => {
    setForecastStep((state.currentStep - 1 + 12) % 12);
  });
  els.btnStepNext.addEventListener('click', () => {
    setForecastStep((state.currentStep + 1) % 12);
  });

  els.btnApproveAlert.addEventListener('click', approveActiveAlert);
  els.btnRejectAlert.addEventListener('click', rejectActiveAlert);
  els.btnDownloadCap.addEventListener('click', downloadCapXmlFile);

  els.btnOpenMetrics.addEventListener('click', openMetricsModal);
  els.btnCloseMetrics.addEventListener('click', () => els.metricsModal.classList.remove('open'));
}

// ── Load Districts GeoJSON ───────────────────────────────────────────────────
async function loadDistrictsGeoJSON() {
  try {
    const res = await fetch('/api/districts');
    const geojson = await res.json();

    state.districtLayer = L.geoJSON(geojson, {
      style: {
        color: '#00f0ff',
        weight: 1.5,
        opacity: 0.8,
        fillColor: '#00f0ff',
        fillOpacity: 0.05,
        dashArray: '4, 4',
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindPopup(`
          <div style="font-family: sans-serif; color: #fff; background: #0f172a; padding: 6px; border-radius: 6px;">
            <b style="color: #00f0ff;">${p.district_name} District</b><br>
            <span style="font-size: 11px; color: #94a3b8;">State: ${p.state_name}</span><br>
            <span style="font-size: 11px; color: #94a3b8;">Population: ${p.population.toLocaleString()}</span>
          </div>
        `);
      },
    }).addTo(state.map);
  } catch (err) {
    console.error('Failed to load districts GeoJSON:', err);
  }
}

// ── Nowcast Execution ────────────────────────────────────────────────────────
async function runNowcastCycle() {
  els.btnRunNowcast.disabled = true;
  els.btnRunNowcast.innerHTML = '<span>⏳</span> Computing...';

  const forcedTier = els.selectTierForce.value || null;
  const eventIdx = parseInt(els.selectEventPreset.value || '0', 10);

  try {
    const res = await fetch('/api/nowcast/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        radar_age_min: forcedTier === 'T1' || forcedTier === 'T2' || forcedTier === 'T3' ? 45.0 : 5.0,
        satellite_age_min: forcedTier === 'T2' || forcedTier === 'T3' ? 60.0 : 12.0,
        aws_age_min: forcedTier === 'T3' ? 90.0 : 10.0,
        nwp_age_min: forcedTier === 'T3' ? 180.0 : 45.0,
        forced_tier: forcedTier,
        event_idx: eventIdx,
      }),
    });

    const data = await res.json();
    state.latestNowcast = data;
    state.cachedFrames = {}; // Invalidate frame cache
    state.activeAlertId = data.alert_id;

    updateDashboardUI(data);
    await setForecastStep(0);
  } catch (err) {
    console.error('Nowcast run failed:', err);
  } finally {
    els.btnRunNowcast.disabled = false;
    els.btnRunNowcast.innerHTML = '<span>⚡</span> Run Nowcast';
  }
}

// ── Update Dashboard Telemetry & Panels ──────────────────────────────────────
function updateDashboardUI(data) {
  // Update Tier badge
  els.tierText.textContent = `TIER ${data.tier_used} : ${data.tier_name.toUpperCase()}`;
  els.latencyText.textContent = `LATENCY: ${data.latency_ms}ms | CONF: ${(data.confidence * 100).toFixed(0)}%`;

  if (data.tier_used === 'T0') {
    els.tierBadge.style.borderColor = 'rgba(0, 240, 255, 0.4)';
    els.tierBadge.style.color = '#00f0ff';
    els.tierPulse.style.background = '#00f0ff';
  } else if (data.tier_used === 'T1') {
    els.tierBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    els.tierBadge.style.color = '#f59e0b';
    els.tierPulse.style.background = '#f59e0b';
  } else {
    els.tierBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    els.tierBadge.style.color = '#ef4444';
    els.tierPulse.style.background = '#ef4444';
  }

  // Update Hazard card
  const riskClass = `hazard-${data.max_overall_risk.toLowerCase()}`;
  els.hazardCard.className = `hazard-badge-card ${riskClass}`;
  els.hazardTitle.textContent = `${data.max_overall_risk.toUpperCase()} WARNING`;

  if (data.max_overall_risk === 'Red') {
    els.hazardIcon.textContent = '⚡';
  } else if (data.max_overall_risk === 'Orange') {
    els.hazardIcon.textContent = '⛈';
  } else {
    els.hazardIcon.textContent = '🌧';
  }

  // Update Impacted Districts list
  els.impactCount.textContent = data.impacted_districts.length;
  els.districtList.innerHTML = '';

  if (data.impacted_districts.length === 0) {
    els.districtList.innerHTML = `
      <div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 12px;">
        No districts currently in severe danger zone.
      </div>
    `;
  } else {
    data.impacted_districts.forEach((d) => {
      const el = document.createElement('div');
      el.className = 'district-item';
      el.innerHTML = `
        <div class="district-info">
          <h4>${d.district_name}</h4>
          <p>${d.impacted_area_sq_km} km² affected (Peak: ${(d.peak_reflectivity * 70).toFixed(0)} dBZ)</p>
        </div>
        <div class="district-arrival">
          T + ${d.earliest_arrival_min}m
        </div>
      `;
      els.districtList.appendChild(el);
    });
  }

  // Update Approval Gate
  if (data.alert_id) {
    els.alertGateStatus.textContent = 'PENDING REVIEW';
    els.alertGateStatus.className = 'gate-badge';
    els.alertGateStatus.style.background = 'rgba(245, 158, 11, 0.2)';
    els.alertGateStatus.style.color = '#f59e0b';
    els.btnApproveAlert.disabled = false;
    els.btnRejectAlert.disabled = false;
    els.btnDownloadCap.disabled = true;
    els.capXmlPreview.textContent = `<!-- Alert ${data.alert_id} drafted. Awaiting forecaster authorization... -->`;
  } else {
    els.alertGateStatus.textContent = 'NO ACTIVE ALERT';
    els.btnApproveAlert.disabled = true;
    els.btnRejectAlert.disabled = true;
    els.btnDownloadCap.disabled = true;
    els.capXmlPreview.textContent = '<!-- Risk levels below threshold. No public alert drafted. -->';
  }
}

// ── Step Navigation & Frame Rendering ─────────────────────────────────────────
async function setForecastStep(stepIdx) {
  state.currentStep = stepIdx;
  els.leadSlider.value = stepIdx;

  const leadMin = (stepIdx + 1) * 15;
  els.mapLeadVal.textContent = `+${leadMin} min (T+${stepIdx + 1})`;
  els.timelineLeadText.textContent = `T + ${leadMin} min`;
  els.frameCounter.textContent = `Frame ${stepIdx + 1} / 12 (1 km Grid)`;

  // Fetch frame data if not cached
  if (!state.cachedFrames[stepIdx]) {
    try {
      const res = await fetch(`/api/nowcast/frame/${stepIdx}`);
      const data = await res.json();
      state.cachedFrames[stepIdx] = data.values;
    } catch (err) {
      console.error('Failed to load frame:', err);
      return;
    }
  }

  renderRadarGrid(state.cachedFrames[stepIdx]);
}

function renderRadarGrid(gridData) {
  const H = gridData.length;
  const W = gridData[0].length;

  // Offscreen canvas for fast generation
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(W, H);

  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const val = gridData[r][c];
      const [red, green, blue, alpha] = getRadarRGBA(val);
      const idx = (r * W + c) * 4;
      imgData.data[idx] = red;
      imgData.data[idx + 1] = green;
      imgData.data[idx + 2] = blue;
      imgData.data[idx + 3] = alpha;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const dataUrl = canvas.toDataURL();

  if (state.radarOverlay) {
    state.radarOverlay.setUrl(dataUrl);
  } else {
    state.radarOverlay = L.imageOverlay(dataUrl, state.bounds, {
      opacity: 0.85,
      interactive: false,
    }).addTo(state.map);
  }
}

// ── Playback Controls ────────────────────────────────────────────────────────
function togglePlayPause() {
  if (state.isPlaying) {
    pausePlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  state.isPlaying = true;
  els.playIcon.textContent = '⏸';
  state.playTimer = setInterval(() => {
    const nextStep = (state.currentStep + 1) % 12;
    setForecastStep(nextStep);
  }, 750);
}

function pausePlayback() {
  state.isPlaying = false;
  els.playIcon.textContent = '▶';
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

// ── Forecaster Approval Actions ──────────────────────────────────────────────
async function approveActiveAlert() {
  if (!state.activeAlertId) return;

  const forecasterId = els.inputForecasterId.value.trim() || 'IMD_MET_DUTY';
  const notes = els.inputForecasterNotes.value.trim() || 'Verified against Cuttack DWR radar sweep.';

  els.btnApproveAlert.disabled = true;
  try {
    const res = await fetch(`/api/alerts/${state.activeAlertId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ forecaster_id: forecasterId, notes: notes }),
    });

    const data = await res.json();
    if (res.ok) {
      els.alertGateStatus.textContent = 'APPROVED & BROADCAST';
      els.alertGateStatus.style.background = 'rgba(16, 185, 129, 0.2)';
      els.alertGateStatus.style.color = '#10b981';
      els.btnDownloadCap.disabled = false;

      // Fetch and show raw XML
      const xmlRes = await fetch(`/api/alerts/${state.activeAlertId}/cap.xml`);
      const xmlText = await xmlRes.text();
      els.capXmlPreview.textContent = xmlText;
    }
  } catch (err) {
    console.error('Failed to approve alert:', err);
  }
}

async function rejectActiveAlert() {
  if (!state.activeAlertId) return;

  const forecasterId = els.inputForecasterId.value.trim() || 'IMD_MET_DUTY';
  const reason = prompt('Please enter reason for suppression / false alarm:') || 'Radar ground clutter';

  try {
    const res = await fetch(`/api/alerts/${state.activeAlertId}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ forecaster_id: forecasterId, reason: reason }),
    });

    if (res.ok) {
      els.alertGateStatus.textContent = 'SUPPRESSED / REJECTED';
      els.alertGateStatus.style.background = 'rgba(239, 68, 68, 0.2)';
      els.alertGateStatus.style.color = '#ef4444';
      els.btnApproveAlert.disabled = true;
      els.btnRejectAlert.disabled = true;
      els.capXmlPreview.textContent = `<!-- Alert ${state.activeAlertId} REJECTED by ${forecasterId}. Reason: ${reason} -->`;
    }
  } catch (err) {
    console.error('Failed to reject alert:', err);
  }
}

function downloadCapXmlFile() {
  if (!state.activeAlertId) return;
  window.open(`/api/alerts/${state.activeAlertId}/cap.xml`, '_blank');
}

// ── Metrics Modal ────────────────────────────────────────────────────────────
async function openMetricsModal() {
  els.metricsModal.classList.add('open');
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    if (data.ai_model && data.advection_baseline) {
      els.metricsTableBody.innerHTML = '';
      const leads = data.lead_times_min || [15, 30, 60, 90, 120, 180];

      leads.forEach((m) => {
        const mStr = String(m);
        const aiCSI = data.ai_model.csi[mStr];
        const baseCSI = data.advection_baseline.csi[mStr];
        const aiPOD = data.ai_model.pod[mStr];
        const basePOD = data.advection_baseline.pod[mStr];
        const aiFAR = data.ai_model.far[mStr];
        const baseFAR = data.advection_baseline.far[mStr];
        const aiFSS = data.ai_model.fss[mStr];
        const baseFSS = data.advection_baseline.fss[mStr];

        const row = document.createElement('tr');
        row.innerHTML = `
          <td style="font-weight: 700; color: #00f0ff;">+${m} min</td>
          <td><b>${aiCSI}</b> vs <span style="color: #94a3b8;">${baseCSI}</span></td>
          <td><b>${aiPOD}</b> vs <span style="color: #94a3b8;">${basePOD}</span></td>
          <td><b>${aiFAR}</b> vs <span style="color: #94a3b8;">${baseFAR}</span></td>
          <td><b>${aiFSS}</b> vs <span style="color: #94a3b8;">${baseFSS}</span></td>
        `;
        els.metricsTableBody.appendChild(row);
      });
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}
