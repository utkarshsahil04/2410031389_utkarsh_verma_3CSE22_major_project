/**
 * Dashboard live-update logic.
 * Polls /state (FL metrics) and /chain (blockchain) every 2 seconds.
 * Vanilla JS only — no framework, no build step.
 */

const POLL_MS = 2000;
const CYAN    = '#00d4ff';
const PURPLE  = '#a855f7';
const GREEN   = '#10b981';
const RED     = '#ef4444';
const DIM     = 'rgba(255,255,255,0.15)';

// ── Mini-chart state (one loss-series per hospital) ───────────────────────
const miniSeries = { hospital_1: [], hospital_2: [], hospital_3: [] };

// ── Main accuracy chart state ─────────────────────────────────────────────
const accSeries  = [];
const lossSeries = [];

// ── Helpers ───────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function fmt(val, decimals = 4) {
  return val == null ? '—' : Number(val).toFixed(decimals);
}

function drawLineChart(canvas, datasets, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.offsetHeight || canvas.height;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);

  if (!datasets.length || !datasets[0].data.length) return;

  const allVals = datasets.flatMap(d => d.data);
  const minV = opts.min ?? Math.min(...allVals) * 0.95;
  const maxV = opts.max ?? Math.max(...allVals) * 1.05;
  const range = maxV - minV || 1;

  const pad = { top: 8, right: 8, bottom: 4, left: 8 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  datasets.forEach(({ data, color }) => {
    const n = data.length;
    if (n < 2) return;

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (n - 1)) * plotW;
      const y = pad.top + (1 - (v - minV) / range) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });

    // Gradient fill under line
    const grad = ctx.createLinearGradient(0, pad.top, 0, H);
    grad.addColorStop(0, color + '33');
    grad.addColorStop(1, color + '00');

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill area
    ctx.lineTo(pad.left + ((n - 1) / (n - 1)) * plotW, H);
    ctx.lineTo(pad.left, H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  });
}

// ── Blockchain audit log ──────────────────────────────────────────────────

let renderedBlocks = 0;

function eventClass(evt) {
  if (evt.includes('genesis'))            return 'ev-genesis';
  if (evt.includes('register'))           return 'ev-register';
  if (evt.includes('fl_round_complete'))  return 'ev-fl';
  if (evt.includes('fl_round_partic'))    return 'ev-fl';
  if (evt.includes('fl_update_hash'))     return 'ev-fl';
  if (evt.includes('grant_access'))       return 'ev-grant';
  if (evt.includes('session_start'))      return 'ev-session';
  return '';
}

function humanLabel(block) {
  const evt = block.data?.event ?? 'unknown';
  const actor = block.data?.actor || block.data?.hospital || '';
  const details = block.data?.details || {};

  if (evt === 'genesis')              return '<strong>Chain initialized</strong> — genesis block';
  if (evt === 'register_record')      return `<strong>${block.data.hospital}</strong> registered dataset — <span class="hash">${block.data.record_hash?.slice(0, 16)}…</span>`;
  if (evt === 'grant_access')         return `<strong>${block.data.from}</strong> granted access to <strong>${block.data.to}</strong>`;
  if (evt === 'fl_session_start')     return `<strong>FL session started</strong> — ${details.rounds} rounds, ${(details.hospitals || []).length} hospitals`;
  if (evt === 'fl_update_hash')       return `<strong>${actor}</strong> committed update hash — <span class="hash">${(details.update_hash || '').slice(0, 16)}…</span> (r${details.round})`;
  if (evt === 'fl_round_participation') return `<strong>${actor}</strong> submitted local update — round ${details.round}, ${details.n_samples} samples`;
  if (evt === 'fl_round_complete')    return `<strong>Round ${details.round} complete</strong> — accuracy <span style="color:var(--cyan)">${fmt(details.accuracy, 4)}</span> — <span class="hash">model: ${details.model_hash}</span>`;
  return `<strong>${evt}</strong> by ${actor}`;
}

function normalizeChain(chain) {
  // Accept legacy list or {blocks, sync} wrapper
  if (Array.isArray(chain)) return chain;
  if (chain && Array.isArray(chain.blocks)) return chain.blocks;
  return [];
}

function updateAuditLog(chainRaw) {
  const chain = normalizeChain(chainRaw);
  if (!chain || !chain.length) return;
  const log = $('audit-log');
  const newBlocks = chain.slice(renderedBlocks);
  if (!newBlocks.length) return;

  if (renderedBlocks === 0) log.innerHTML = '';

  newBlocks.forEach(block => {
    const entry = document.createElement('div');
    entry.className = `audit-entry ${eventClass(block.data?.event ?? '')}`;
    entry.innerHTML = `
      <span class="audit-block-num">#${block.index}</span>
      <span class="audit-event">${humanLabel(block)}</span>
    `;
    log.appendChild(entry);
  });

  renderedBlocks = chain.length;
  $('audit-count').textContent = `${chain.length} blocks`;
  log.scrollTop = log.scrollHeight;
}

// ── Hospital cards ────────────────────────────────────────────────────────

const HOSPITAL_IDS = ['Hospital_A', 'Hospital_B', 'Hospital_C'];
const CARD_PREFIXES = ['h1', 'h2', 'h3'];
const MINI_KEYS = ['hospital_1', 'hospital_2', 'hospital_3'];
const MINI_CANVASES = ['chart-h1', 'chart-h2', 'chart-h3'];

function updateHospitalCards(state) {
  HOSPITAL_IDS.forEach((id, i) => {
    const info = state.hospitals?.[id];
    const prefix = CARD_PREFIXES[i];
    if (!info) return;

    $(prefix + '-name').textContent = id.replace('_', ' ');
    $(prefix + '-records').textContent = info.records ? `${info.records} records` : '— records';

    const regBadge = $(prefix + '-reg');
    if (info.registered) {
      regBadge.textContent = 'Registered ✓';
      regBadge.classList.add('registered');
    }

    if (info.block_index !== undefined) {
      $(prefix + '-block').textContent = `#${info.block_index}`;
    }
  });
}

function updateHospitalLosses(rounds) {
  if (!rounds || !rounds.length) return;
  const last = rounds[rounds.length - 1];

  HOSPITAL_IDS.forEach((id, i) => {
    const key = `hospital_${i + 1}`;
    const loss = last.client_losses?.[key];
    if (loss !== undefined) {
      $(CARD_PREFIXES[i] + '-loss').textContent = fmt(loss, 4);
      miniSeries[key].push(loss);
    }
  });

  MINI_KEYS.forEach((key, i) => {
    drawLineChart(
      $(MINI_CANVASES[i]),
      [{ data: miniSeries[key], color: CYAN }],
      { min: 0 }
    );
  });
}

// ── Global accuracy chart ─────────────────────────────────────────────────

function updateAccuracyChart(rounds) {
  if (!rounds || !rounds.length) return;

  // Rebuild series from all rounds (idempotent)
  accSeries.length  = 0;
  lossSeries.length = 0;
  rounds.forEach(r => {
    accSeries.push(r.accuracy);
    lossSeries.push(r.avg_loss);
  });

  const canvas = $('accuracy-chart');
  canvas.height = 180;
  drawLineChart(
    canvas,
    [
      { data: accSeries,  color: CYAN   },
      { data: lossSeries, color: PURPLE },
    ],
    { min: 0, max: Math.max(...accSeries, ...lossSeries) * 1.1 }
  );
}

// ── Header status ─────────────────────────────────────────────────────────

const STATUS_LABEL = { initializing: 'Initializing…', training: 'Training…', complete: 'Complete ✓', connecting: 'Connecting…' };

function updateHeader(state, chainValid) {
  const statusText = STATUS_LABEL[state.status] ?? state.status;
  $('sim-status-text').textContent = statusText;

  const dot = document.querySelector('.pulse-dot');
  dot.className = 'pulse-dot';
  if (state.status === 'complete')      dot.classList.add('done');
  if (state.status === 'initializing') dot.classList.add('idle');

  const badge = $('chain-badge');
  const valid = chainValid !== false;
  badge.className = 'chain-badge' + (valid ? '' : ' invalid');
  $('chain-validity').textContent = valid ? 'Chain: VALID ✅' : 'Chain: INVALID ❌';
  $('chain-blocks').textContent = `${state.chain_length ?? 0} blocks`;
}

function updateProgress(state) {
  const cur = state.current_round ?? 0;
  const tot = state.total_rounds ?? 10;
  const pct = tot > 0 ? (cur / tot) * 100 : 0;

  $('round-label').textContent = `Round ${cur} / ${tot}`;
  $('progress-fill').style.width = pct + '%';

  const lastRound = state.rounds?.[state.rounds.length - 1];
  if (lastRound) {
    $('global-accuracy').textContent = (lastRound.accuracy * 100).toFixed(1) + '%';
    $('global-loss').textContent = fmt(lastRound.avg_loss, 4);
  }
}

// ── Poll loop ─────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

async function poll() {
  const [state, chain] = await Promise.all([
    fetchJSON('/state'),
    fetchJSON('/chain'),
  ]);

  if (!state) {
    $('sim-status-text').textContent = 'Waiting for simulation…';
    return;
  }

  updateHeader(state, state.chain_valid);
  updateProgress(state);
  updateHospitalCards(state);
  updateHospitalLosses(state.rounds);
  updateAccuracyChart(state.rounds);
  if (chain) updateAuditLog(normalizeChain(chain));

  $('last-update').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

// Start polling immediately then every POLL_MS
poll();
setInterval(poll, POLL_MS);
