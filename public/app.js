// Real-Time ESP32 IoT Flood Monitor Frontend Controller
// National Flash Flood Early Warning System (NFFEWS) - Government Portal

const POLL_INTERVAL_MS = 2000;
let lastTimestamp = null;
let telemetryChart = null;
const MAX_CHART_POINTS = 30;

// Chart data buffers
const chartLabels = [];
const dataRain = [];
const dataSoil = [];
const dataHumidity = [];
const dataTemp = [];
const dataSlope = [];

// Initialize Real-Time Clock
function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-IN', { hour12: false }) + ' IST';
  const clockEl = document.getElementById('current-clock');
  if (clockEl) clockEl.textContent = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// Initialize Chart.js
function initTelemetryChart() {
  const canvas = document.getElementById('liveTelemetryChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  telemetryChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: 'Rain Rate (mm/h)',
          data: dataRain,
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.08)',
          borderWidth: 2.5,
          tension: 0.35,
          yAxisID: 'yRain',
          pointRadius: 3,
        },
        {
          label: 'Soil Moisture (%)',
          data: dataSoil,
          borderColor: '#059669',
          backgroundColor: 'transparent',
          borderWidth: 2,
          tension: 0.35,
          yAxisID: 'yPct',
          pointRadius: 2,
        },
        {
          label: 'Humidity (%)',
          data: dataHumidity,
          borderColor: '#4f46e5',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          borderDash: [4, 4],
          tension: 0.35,
          yAxisID: 'yPct',
          pointRadius: 0,
        },
        {
          label: 'Temperature (°C)',
          data: dataTemp,
          borderColor: '#d97706',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          tension: 0.35,
          yAxisID: 'yTemp',
          pointRadius: 2,
        },
        {
          label: 'Slope Pitch (°)',
          data: dataSlope,
          borderColor: '#dc2626',
          backgroundColor: 'transparent',
          borderWidth: 2,
          tension: 0.35,
          yAxisID: 'yTemp',
          pointRadius: 2,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(11, 34, 64, 0.95)',
          borderColor: '#e2e8f0',
          borderWidth: 1,
          titleFont: { family: 'Inter', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 11 },
        }
      },
      scales: {
        x: {
          grid: { color: '#e2e8f0' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        },
        yRain: {
          type: 'linear',
          position: 'left',
          min: 0,
          max: 120,
          grid: { color: '#f1f5f9' },
          ticks: { color: '#0284c7', font: { family: 'JetBrains Mono', size: 10, weight: 'bold' } }
        },
        yPct: {
          type: 'linear',
          position: 'right',
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false },
          ticks: { color: '#059669', font: { family: 'JetBrains Mono', size: 10, weight: 'bold' } }
        },
        yTemp: {
          type: 'linear',
          position: 'right',
          display: false,
          min: -10,
          max: 60
        }
      }
    }
  });
}

// Append new live point to chart
function addTelemetryPoint(timeStr, rain, soil, humidity, temp, slope) {
  if (chartLabels.length >= MAX_CHART_POINTS) {
    chartLabels.shift();
    dataRain.shift();
    dataSoil.shift();
    dataHumidity.shift();
    dataTemp.shift();
    dataSlope.shift();
  }

  const shortTime = timeStr.includes(' ') ? timeStr.split(' ')[1] : timeStr;
  chartLabels.push(shortTime);
  dataRain.push(rain);
  dataSoil.push(soil);
  dataHumidity.push(humidity);
  dataTemp.push(temp);
  dataSlope.push(slope);

  if (telemetryChart) {
    telemetryChart.update();
  }
}

// Load initial historical packets to populate chart from database
async function loadTelemetryHistory() {
  try {
    const res = await fetch('/api/history');
    if (res.ok) {
      const histData = await res.json();
      if (histData.history && histData.history.length > 0) {
        chartLabels.length = 0;
        dataRain.length = 0;
        dataSoil.length = 0;
        dataHumidity.length = 0;
        dataTemp.length = 0;
        dataSlope.length = 0;

        const packets = histData.history.slice(-MAX_CHART_POINTS);
        packets.forEach(p => {
          const s = p.sensors || {};
          addTelemetryPoint(
            p.timestamp || '',
            s.rain_rate_mm_h || 0,
            s.soil_moisture_pct || 0,
            s.humidity_pct || 0,
            s.temperature_c || 0,
            s.slope_degrees || 0
          );
        });
      }
    }
  } catch (err) {
    console.warn('History stream init deferred:', err);
  }
}

// Update UI with Live Telemetry Packet
function renderTelemetry(packet) {
  if (!packet) return;

  const nodeBadge = document.getElementById('node-id-badge');
  if (nodeBadge && packet.node_id) {
    nodeBadge.innerHTML = `<span class="node-label">EDGE NODE:</span><span class="node-id">${packet.node_id}</span>`;
  }

  const connPill = document.getElementById('connection-pill');
  const connText = document.getElementById('connection-status-text');
  if (packet.status === 'ONLINE' && packet.is_live !== false) {
    connPill.className = 'official-status-badge status-online';
    connText.textContent = `TELEMETRY ACTIVE (LIVE)`;
  } else if (packet.status === 'OFFLINE' || packet.is_live === false) {
    connPill.className = 'official-status-badge status-alert';
    const elapsed = packet.seconds_since_last_packet ? ` (${Math.round(packet.seconds_since_last_packet)}s ago)` : '';
    connText.textContent = `EDGE NODE OFFLINE${elapsed}`;
  } else {
    connPill.className = 'official-status-badge status-standby';
    connText.textContent = `INITIALIZING TELEMETRY...`;
  }

  const sensors = packet.sensors || {};
  const temp = sensors.temperature_c !== undefined ? sensors.temperature_c : 24.0;
  const humidity = sensors.humidity_pct !== undefined ? sensors.humidity_pct : 65.0;
  const soil = sensors.soil_moisture_pct !== undefined ? sensors.soil_moisture_pct : 40.0;
  const rainRate = sensors.rain_rate_mm_h !== undefined ? sensors.rain_rate_mm_h : 0.0;
  const rainAccum = sensors.rain_accum_mm !== undefined ? sensors.rain_accum_mm : 0.0;
  const slope = sensors.slope_degrees !== undefined ? sensors.slope_degrees : 12.0;
  const vibration = sensors.vibration_g !== undefined ? sensors.vibration_g : 0.0;

  // 1. Update Sensor Values & Badges
  const elTemp = document.getElementById('val-temp');
  if (elTemp) elTemp.textContent = Number(temp).toFixed(1);
  const stateTemp = document.getElementById('state-temp');
  if (stateTemp) {
    if (temp > 38) {
      stateTemp.textContent = 'HIGH HEAT';
      stateTemp.className = 'gov-tag tag-alert';
    } else if (temp > 32) {
      stateTemp.textContent = 'ELEVATED';
      stateTemp.className = 'gov-tag tag-warn';
    } else {
      stateTemp.textContent = 'NORMAL';
      stateTemp.className = 'gov-tag tag-normal';
    }
  }

  const elHum = document.getElementById('val-humidity');
  if (elHum) elHum.textContent = Number(humidity).toFixed(0);
  const stateHum = document.getElementById('state-humidity');
  if (stateHum) {
    if (humidity >= 90) {
      stateHum.textContent = 'SATURATED';
      stateHum.className = 'gov-tag tag-alert';
    } else if (humidity >= 75) {
      stateHum.textContent = 'HIGH';
      stateHum.className = 'gov-tag tag-warn';
    } else {
      stateHum.textContent = 'OPTIMAL';
      stateHum.className = 'gov-tag tag-normal';
    }
  }

  const elSoil = document.getElementById('val-soil');
  if (elSoil) elSoil.textContent = Number(soil).toFixed(1);
  const stateSoil = document.getElementById('state-soil');
  if (stateSoil) {
    if (soil >= 85) {
      stateSoil.textContent = 'WATERLOGGED';
      stateSoil.className = 'gov-tag tag-alert';
    } else if (soil >= 70) {
      stateSoil.textContent = 'HEAVY MOIST';
      stateSoil.className = 'gov-tag tag-warn';
    } else {
      stateSoil.textContent = 'MODERATE';
      stateSoil.className = 'gov-tag tag-normal';
    }
  }

  const elRain = document.getElementById('val-rain');
  if (elRain) elRain.textContent = Number(rainRate).toFixed(1);
  const elRainAccum = document.getElementById('val-rain-accum');
  if (elRainAccum) elRainAccum.textContent = `${Number(rainAccum).toFixed(1)} mm`;
  const stateRain = document.getElementById('state-rain');
  if (stateRain) {
    if (rainRate >= 50) {
      stateRain.textContent = 'CLOUDBURST';
      stateRain.className = 'gov-tag tag-alert';
    } else if (rainRate >= 20) {
      stateRain.textContent = 'HEAVY RAIN';
      stateRain.className = 'gov-tag tag-alert';
    } else if (rainRate > 5) {
      stateRain.textContent = 'MODERATE';
      stateRain.className = 'gov-tag tag-warn';
    } else if (rainRate > 0.5) {
      stateRain.textContent = 'LIGHT RAIN';
      stateRain.className = 'gov-tag tag-normal';
    } else {
      stateRain.textContent = 'DRY';
      stateRain.className = 'gov-tag tag-normal';
    }
  }

  const elSlope = document.getElementById('val-slope');
  if (elSlope) elSlope.textContent = Number(slope).toFixed(1);
  const elVib = document.getElementById('val-vibration');
  if (elVib) elVib.textContent = `${Number(vibration).toFixed(2)}g`;
  const stateSlope = document.getElementById('state-slope');
  if (stateSlope) {
    if (slope >= 25 || vibration >= 0.20) {
      stateSlope.textContent = 'LANDSLIDE HAZARD';
      stateSlope.className = 'gov-tag tag-alert';
    } else if (slope >= 18) {
      stateSlope.textContent = 'HILLSIDE TILT';
      stateSlope.className = 'gov-tag tag-warn';
    } else {
      stateSlope.textContent = 'STABLE';
      stateSlope.className = 'gov-tag tag-normal';
    }
  }

  // 2. Update Hero AI Flash Flood Risk Prediction
  const pred = packet.ai_prediction || {};
  const riskLevel = pred.predicted_risk_level !== undefined ? pred.predicted_risk_level : 0;
  const riskCategory = pred.risk_category || 'Low Risk';
  
  // Official Government Risk Palette
  const govRiskColors = ['#059669', '#d97706', '#ea580c', '#dc2626'];
  const govRiskBgs = ['#ecfdf5', '#fffbeb', '#fff7ed', '#fef2f2'];
  const riskColor = govRiskColors[riskLevel] || '#059669';
  const riskBg = govRiskBgs[riskLevel] || '#ecfdf5';

  const confidence = pred.confidence_score !== undefined ? pred.confidence_score : 98.5;
  const statusCode = pred.status_code || 'NORMAL / ALL CLEAR - BASELINE STATE';
  const riskDesc = pred.risk_description || 'All watershed sensors operating within baseline safety margins.';
  const leadTime = pred.early_warning_lead_time || '72h Routine Surveillance';

  const riskHeroCard = document.getElementById('risk-hero-container');
  if (riskHeroCard) riskHeroCard.style.borderLeftColor = riskColor;

  const meterGlow = document.getElementById('risk-meter-glow');
  if (meterGlow) {
    meterGlow.style.borderColor = riskColor;
    meterGlow.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.07)';
    meterGlow.style.background = riskBg;
  }

  const riskNumEl = document.getElementById('risk-level-num');
  if (riskNumEl) {
    riskNumEl.textContent = `L${riskLevel}`;
    riskNumEl.style.color = riskColor;
  }

  const riskTitleEl = document.getElementById('risk-tier-title');
  if (riskTitleEl) {
    riskTitleEl.textContent = riskCategory.toUpperCase();
    riskTitleEl.style.color = riskColor;
  }

  const confEl = document.getElementById('risk-confidence');
  if (confEl) confEl.textContent = `${Number(confidence).toFixed(1)}%`;

  const bannerEl = document.getElementById('status-code-banner');
  if (bannerEl) {
    bannerEl.textContent = statusCode;
    bannerEl.style.color = riskLevel >= 2 ? riskColor : '#0b2240';
  }

  const descEl = document.getElementById('risk-desc');
  if (descEl) descEl.textContent = riskDesc;

  const leadEl = document.getElementById('lead-time-badge');
  if (leadEl) leadEl.textContent = `⏳ ${leadTime}`;

  // Update Ticker Bulletin with dynamic advisory
  const tickerText = document.getElementById('ticker-text');
  if (tickerText) {
    if (riskLevel === 3) {
      tickerText.textContent = `🚨 CRITICAL EMERGENCY RED ALERT: Flash flood imminent in basin catchment. Immediate evacuation protocols active.`;
    } else if (riskLevel === 2) {
      tickerText.textContent = `⚠️ ORANGE WARNING: Elevated catchment runoff and precipitation surge detected. Deploy rapid response units.`;
    } else if (riskLevel === 1) {
      tickerText.textContent = `🟡 YELLOW ADVISORY: Moderate moisture accumulation and precipitation. District civil defense on standby.`;
    } else {
      tickerText.textContent = `Real-time edge IoT multi-sensor telemetry stream active. Hydrological risk classification algorithms running continuous inference on river basin watershed parameters.`;
    }
  }

  // 3. Actuator Edge States
  const sirenStateEl = document.getElementById('act-siren-state');
  const sirenIconEl = document.getElementById('act-siren-icon');
  const ledStateEl = document.getElementById('act-led-state');
  const ledIconEl = document.getElementById('act-led-icon');

  if (riskLevel >= 2) {
    if (sirenStateEl) { sirenStateEl.textContent = 'ALARM ACTIVE 🚨'; sirenStateEl.style.color = '#EF4444'; }
    if (sirenIconEl) sirenIconEl.textContent = '🚨';
    if (ledStateEl) { ledStateEl.textContent = 'RED (CRITICAL ALERT)'; ledStateEl.style.color = '#EF4444'; }
    if (ledIconEl) ledIconEl.textContent = '🔴';
  } else if (riskLevel === 1) {
    if (sirenStateEl) { sirenStateEl.textContent = 'STANDBY (OFF)'; sirenStateEl.style.color = '#94A3B8'; }
    if (sirenIconEl) sirenIconEl.textContent = '🔇';
    if (ledStateEl) { ledStateEl.textContent = 'YELLOW (ADVISORY)'; ledStateEl.style.color = '#F59E0B'; }
    if (ledIconEl) ledIconEl.textContent = '🟡';
  } else {
    if (sirenStateEl) { sirenStateEl.textContent = 'STANDBY (OFF)'; sirenStateEl.style.color = '#94A3B8'; }
    if (sirenIconEl) sirenIconEl.textContent = '🔇';
    if (ledStateEl) { ledStateEl.textContent = 'GREEN (ALL CLEAR)'; ledStateEl.style.color = '#10B981'; }
    if (ledIconEl) ledIconEl.textContent = '🟢';
  }

  // 4. Probability Distribution Bars
  const probs = pred.probability_distribution || {};
  const pL0 = probs['Low Risk'] !== undefined ? probs['Low Risk'] : 95;
  const pL1 = probs['Moderate Risk'] !== undefined ? probs['Moderate Risk'] : 5;
  const pL2 = probs['High Risk'] !== undefined ? probs['High Risk'] : 0;
  const pL3 = probs['Severe Risk'] !== undefined ? probs['Severe Risk'] : 0;

  const setProb = (idVal, idFill, val) => {
    const vEl = document.getElementById(idVal);
    const fEl = document.getElementById(idFill);
    if (vEl) vEl.textContent = `${Number(val).toFixed(0)}%`;
    if (fEl) fEl.style.width = `${val}%`;
  };

  setProb('prob-l0-val', 'prob-l0-fill', pL0);
  setProb('prob-l1-val', 'prob-l1-fill', pL1);
  setProb('prob-l2-val', 'prob-l2-fill', pL2);
  setProb('prob-l3-val', 'prob-l3-fill', pL3);

  // 5. Contributing Risk Drivers
  const driversList = document.getElementById('drivers-list');
  if (driversList) {
    driversList.innerHTML = '';
    const drivers = pred.primary_risk_drivers || ['All hydrological parameters operating within calibrated baseline safety limits.'];
    drivers.forEach(d => {
      const dDiv = document.createElement('div');
      dDiv.className = 'gov-driver-item';
      dDiv.textContent = `⚡ ${d}`;
      driversList.appendChild(dDiv);
    });
  }

  // 6. Advisories
  const advisoriesList = document.getElementById('advisories-list');
  if (advisoriesList) {
    advisoriesList.innerHTML = '';
    const advisories = pred.advisory_actions || ['Maintain routine telemetry streaming and standard drainage clearances.'];
    advisories.forEach(a => {
      const aDiv = document.createElement('div');
      aDiv.className = 'gov-advisory-item';
      aDiv.textContent = `⚠️ ${a}`;
      advisoriesList.appendChild(aDiv);
    });
  }

  // 7. Update Chart with new point if timestamp changed
  if (packet.timestamp && packet.timestamp !== lastTimestamp) {
    lastTimestamp = packet.timestamp;
    addTelemetryPoint(packet.timestamp, rainRate, soil, humidity, temp, slope);
  }
}

// Fetch and render database table records
async function loadDatabaseRecords() {
  try {
    const res = await fetch('/api/records?limit=30');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('db-table-body');
    if (!tbody) return;

    if (!data.records || data.records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; color: var(--text-dim); padding: 1.5rem;">No sensor readings stored yet in SQLite database. Start sending telemetry from ESP32 or simulator.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.records.map(r => {
      const riskLevel = r.predicted_risk_level !== undefined ? r.predicted_risk_level : 0;
      const riskCategory = r.risk_category || 'Low Risk';
      const timeStr = r.timestamp || '--';
      return `
        <tr>
          <td>#${r.id}</td>
          <td>${timeStr}</td>
          <td><code>${r.node_id || '--'}</code></td>
          <td>${Number(r.temperature_c || 0).toFixed(1)}°C</td>
          <td>${Number(r.humidity_pct || 0).toFixed(0)}%</td>
          <td>${Number(r.soil_moisture_pct || 0).toFixed(1)}%</td>
          <td>${Number(r.rain_rate_mm_h || 0).toFixed(1)}</td>
          <td>${Number(r.rain_accum_mm || 0).toFixed(1)}</td>
          <td>${Number(r.slope_degrees || 0).toFixed(1)}°</td>
          <td>${Number(r.vibration_g || 0).toFixed(2)}g</td>
          <td><span class="risk-tag risk-tag-${riskLevel}">${riskCategory} (L${riskLevel})</span></td>
          <td>${r.status || 'ONLINE'}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.warn('Failed to load database records:', err);
  }
}

// Fetch storage statistics
async function loadStorageStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const stats = await res.json();

    const totalEl = document.getElementById('stat-total-records');
    const badgeEl = document.getElementById('total-stored-badge');
    if (totalEl) totalEl.textContent = stats.total_readings || 0;
    if (badgeEl) badgeEl.textContent = stats.total_readings || 0;

    const avgTempEl = document.getElementById('stat-avg-temp');
    if (avgTempEl) avgTempEl.textContent = `${stats.avg_temperature_c || 0}°C`;

    const avgSoilEl = document.getElementById('stat-avg-soil');
    if (avgSoilEl) avgSoilEl.textContent = `${stats.avg_soil_moisture_pct || 0}%`;

    const maxRainEl = document.getElementById('stat-max-rain');
    if (maxRainEl) maxRainEl.textContent = `${stats.max_rain_rate_mm_h || 0} mm/h`;
  } catch (err) {
    console.warn('Failed to load storage stats:', err);
  }
}

// Live Polling Routine
async function pollLatestTelemetry() {
  try {
    const res = await fetch('/api/latest');
    if (res.ok) {
      const packet = await res.json();
      renderTelemetry(packet);
    }
  } catch (err) {
    const connText = document.getElementById('connection-status-text');
    if (connText) connText.textContent = 'CONNECTING TO SENSOR SERVER...';
  }
}

// Event Listeners and Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initTelemetryChart();
  await loadTelemetryHistory();
  await pollLatestTelemetry();
  await loadDatabaseRecords();
  await loadStorageStats();

  // Polling loops
  setInterval(pollLatestTelemetry, POLL_INTERVAL_MS);
  setInterval(loadStorageStats, 5000);
  setInterval(loadDatabaseRecords, 5000);

  // Refresh DB table button
  const btnRefreshDb = document.getElementById('btn-refresh-db');
  if (btnRefreshDb) {
    btnRefreshDb.addEventListener('click', async () => {
      await loadDatabaseRecords();
      await loadStorageStats();
    });
  }
});

