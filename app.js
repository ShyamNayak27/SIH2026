const API_BASE = 'http://127.0.0.1:8000';

const locations = {
  '5832ea6720e739c3': 'North Sikkim, Sikkim',
  'faa13b60ef2d03ee': 'West Kameng, Arunachal Pradesh',
  '51fd6bd5b303632a': 'North Cachar Hills, Assam',
  '42f02219a21059e1': 'Ukhrul, Manipur',
  '256d963e70da0b7e': 'East Khasi Hills, Meghalaya',
  '8854fae2ae364c2c': 'Lawngtlai, Mizoram',
  '35ae45ac9c84082a': 'Kohima, Nagaland',
  '4ff3ebb1eb0eddc7': 'South Tripura, Tripura'
};

const select = document.querySelector('#locationSelect');

Object.entries(locations).forEach(([id, name]) => {
  select.add(new Option(name, id));
});

function getRiskLevelClass(level) {
  return level.toLowerCase();
}

function renderDrivers(drivers) {
  const container = document.querySelector('#drivers');

  if (!Array.isArray(drivers) || drivers.length === 0) {
    container.innerHTML = `
      <p class="api-note">
        No spatial model factor data available.
      </p>
    `;
    return;
  }

  container.innerHTML = drivers.map(driver => `
    <div class="driver">
      <div>
        <span>
          ${driver.name}
          <small class="driver-direction">
            ${driver.direction === 'increases' ? '↑' : '↓'}
          </small>
        </span>

        <strong>${driver.strength}%</strong>
      </div>

      <div class="driver-track">
        <i
          class="${driver.level}"
          style="width:${driver.strength}%"
        ></i>
      </div>
    </div>
  `).join('');
}

function renderRisk(data) {
  const decision = data.decision;
  const temporal = data.temporal_analysis;

  const finalScore = Math.round(decision.final_risk * 100);

  const locationName =
    `${data.location.district}, ${data.location.state}`;

  document.querySelector('#mapTitle').textContent = locationName;
  document.querySelector('#mapLocation').textContent =
    data.location.district.toUpperCase();

  document.querySelector('#riskScore').textContent = finalScore;
  document.querySelector('#riskBadge').textContent =
    decision.final_alert_level;

  document.querySelector('#riskSummary').textContent =
    decision.explanation?.summary || decision.action;

  document.querySelector('#meterFill').style.left =
    `${finalScore}%`;

  document.querySelector('#rain24').innerHTML =
    `${data.observed_conditions.rain_1d_mm} <small>mm</small>`;

  document.querySelector('#rain72').innerHTML =
    `${data.observed_conditions.rain_3d_mm} <small>mm</small>`;

  document.querySelector('#rain7').innerHTML =
    `${temporal.rainfall_7d} <small>mm</small>`;

  document.querySelector('#rain30').innerHTML =
    `${temporal.rainfall_30d} <small>mm</small>`;

  document.querySelector('#trendLabel').textContent =
    temporal.risk_level;

  document.querySelector('#outlook').innerHTML = `
    Temporal rainfall risk is
    <strong>${Math.round(temporal.temporal_risk)} / 100</strong>.
    ${temporal.drivers?.[0] || 'Rainfall conditions are being monitored.'}
  `;

  document.querySelector('#mapCoordinates').textContent =
    `${Number(data.location.latitude).toFixed(4)}° N · ` +
    `${Number(data.location.longitude).toFixed(4)}° E`;

  document.querySelector('#temporalScore').textContent =
  Math.round(temporal.temporal_risk);

document.querySelector('#temporalRiskLevel').textContent =
  temporal.risk_level;

document.querySelector('#temporalDrivers').innerHTML =
  temporal.drivers.map(driver => `
    <div class="temporal-driver">
      <span>•</span>
      <p>${driver}</p>
    </div>
  `).join('');

document.querySelector('#outlook').innerHTML =
  `Temporal model assessment: <strong>
  ${Math.round(temporal.temporal_risk)} / 100
  </strong>.`;

  const scoreBreakdown = document.querySelector('#scoreBreakdown');

  const temporalRisk = Math.round(
    Number(temporal.temporal_risk)
  );

  // With vision currently unavailable, the fusion engine
  // renormalizes spatial + temporal weights from 45/35.
  const spatialContribution =
    Number(decision.model_contributions?.spatial ?? 0);

  const spatialRisk = Math.round(
    (spatialContribution / 0.5625) * 100
  );

  scoreBreakdown.innerHTML = `
    <div class="score-row">
      <span>Spatial model</span>
      <strong>${spatialRisk} / 100</strong>
    </div>

    <div class="score-row">
      <span>Temporal model</span>
      <strong>${temporalRisk} / 100</strong>
    </div>

    <div class="score-row final">
      <span>Final fused risk</span>
      <strong>${finalScore} / 100 · ${decision.final_alert_level}</strong>
    </div>
  `;

  renderDrivers(data.drivers || []);

  // Update model tag
  document.querySelector('.model-tag').textContent =
    'LIVE API · SPATIAL + TEMPORAL';

  // Update alert message
  const alertBox = document.querySelector('.alert-box p');

  alertBox.innerHTML = `
    <strong>${decision.final_alert_level} RISK</strong><br />
    ${decision.action}
  `;

  // Body state controls existing styling
  document.body.dataset.level =
    getRiskLevelClass(decision.final_alert_level);

  // Show actual model status
  const modelStatus = data.model_status;

  const statusText = `
    Spatial ✓
    Temporal ${modelStatus.temporal === 'connected' ? '✓' : '○'}
    Vision ${modelStatus.vision === 'connected' ? '✓' : '○'}
  `;

  document.querySelector('.map-note').innerHTML = `
    <span class="radar-dot"></span>
    ${statusText}
  `;

  // Update timestamp
  document.querySelector('#updatedAt').textContent =
    `UPDATED ${new Date().toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })}`;
}

async function loadRisk(sampleId) {
  const riskScore = document.querySelector('#riskScore');
  const riskSummary = document.querySelector('#riskSummary');

  riskScore.textContent = '—';
  riskSummary.textContent = 'Loading live risk assessment…';

  try {
    const response = await fetch(
      `${API_BASE}/api/risk/${sampleId}`
    );

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();

    renderRisk(data);

  } catch (error) {
    console.error('Risk API error:', error);

    riskScore.textContent = 'ERR';

    document.querySelector('#riskBadge').textContent =
      'OFFLINE';

    riskSummary.textContent =
      'Unable to connect to the risk engine. Please ensure the FastAPI server is running on port 8000.';

    document.body.dataset.level = 'low';
  }
}

select.addEventListener('change', () => {
  loadRisk(select.value);
});

/*
 * Demo-only interaction.
 * This does NOT alter the backend risk model.
 */
document.querySelector('#refreshBtn').addEventListener('click', () => {
  loadRisk(select.value);
});

// Default location
select.value = '5832ea6720e739c3';
loadRisk(select.value);