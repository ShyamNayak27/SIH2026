// Replace getMockRisk(locationId) with a fetch to Stuti's risk-engine endpoint later.
// The UI consumes this single response shape, keeping integration deliberately small.
const mockRiskResponses = {
  munnar: { name: 'Munnar, Kerala', short: 'MUNNAR', score: 72, level: 'HIGH', summary: 'Elevated risk. Heavy recent rainfall is saturating already vulnerable slopes.', rain24: 84, rain72: 146, soil: 78, forecast: 42, peak: 81, trend: 'RISING', drivers: [['72-hour rainfall accumulation', 91, 'high'], ['Soil moisture saturation', 78, 'high'], ['Terrain susceptibility', 66, 'medium'], ['Forecast precipitation', 58, 'medium']] },
  darjeeling: { name: 'Darjeeling, West Bengal', short: 'DARJEELING', score: 58, level: 'MODERATE', summary: 'Watch conditions. Persistent rain and steep terrain require close monitoring.', rain24: 49, rain72: 102, soil: 69, forecast: 27, peak: 64, trend: 'RISING', drivers: [['72-hour rainfall accumulation', 68, 'high'], ['Terrain susceptibility', 75, 'high'], ['Soil moisture saturation', 61, 'medium'], ['Forecast precipitation', 42, 'low']] },
  shimla: { name: 'Shimla, Himachal Pradesh', short: 'SHIMLA', score: 34, level: 'LOW', summary: 'Conditions are currently stable. Continue routine rainfall monitoring.', rain24: 12, rain72: 31, soil: 42, forecast: 8, peak: 38, trend: 'STABLE', drivers: [['Terrain susceptibility', 59, 'medium'], ['Soil moisture saturation', 42, 'medium'], ['72-hour rainfall accumulation', 25, 'low'], ['Forecast precipitation', 18, 'low']] }
};
const select = document.querySelector('#locationSelect');
Object.entries(mockRiskResponses).forEach(([id, item]) => select.add(new Option(item.name, id)));
const ids = ['riskScore', 'riskBadge', 'riskSummary', 'rain24', 'rain72', 'soil', 'forecast', 'trendLabel', 'mapTitle', 'mapLocation'];
function renderRisk(data) {
  document.querySelector('#mapTitle').textContent = data.name;
  document.querySelector('#mapLocation').textContent = data.short;
  document.querySelector('#riskScore').textContent = data.score;
  document.querySelector('#riskBadge').textContent = data.level;
  document.querySelector('#riskSummary').textContent = data.summary;
  document.querySelector('#rain24').innerHTML = `${data.rain24} <small>mm</small>`;
  document.querySelector('#rain72').innerHTML = `${data.rain72} <small>mm</small>`;
  document.querySelector('#soil').innerHTML = `${data.soil}<small>%</small>`;
  document.querySelector('#forecast').innerHTML = `${data.forecast} <small>mm</small>`;
  document.querySelector('#meterFill').style.left = `${data.score}%`;
  document.querySelector('#trendLabel').textContent = data.trend === 'RISING' ? '↗ RISING' : '→ STABLE';
  document.querySelector('#outlook').innerHTML = data.trend === 'RISING' ? `Risk may peak at <strong>${data.peak} / 100</strong> tomorrow morning if forecast rainfall arrives.` : `Risk is expected to remain <strong>below ${data.peak} / 100</strong> through the next 48 hours.`;
  document.body.dataset.level = data.level.toLowerCase();
  document.querySelector('#drivers').innerHTML = data.drivers.map(([name, value, level]) => `<div class="driver"><div><span>${name}</span><strong>${value}%</strong></div><div class="driver-track"><i class="${level}" style="width:${value}%"></i></div></div>`).join('');
}
select.addEventListener('change', () => renderRisk(mockRiskResponses[select.value]));
document.querySelector('#simulateBtn').addEventListener('click', () => {
  const data = mockRiskResponses[select.value];
  data.score = Math.min(99, data.score + 4); data.rain24 += 7; data.forecast += 6; data.peak = Math.min(99, data.peak + 3);
  renderRisk(data);
  document.querySelector('#simulateBtn').textContent = 'Rainfall added ✓';
  setTimeout(() => document.querySelector('#simulateBtn').textContent = 'Simulate rainfall', 1400);
});
document.querySelector('#updatedAt').textContent = `UPDATED ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
renderRisk(mockRiskResponses.munnar);
