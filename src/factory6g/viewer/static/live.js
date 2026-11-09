'use strict';

// Factory6G live monitor. Receives snapshots from `python -m factory6g.viewer monitor`
// over Server-Sent Events (/api/stream) and redraws only what changed.

const METHOD_COLORS = ['#3a56a8', '#cc6677', '#117733', '#aa4499', '#3f8fbf', '#999933', '#882255', '#44aa99'];
const UT_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#6b6b6b', '#882255'];
const TF_SHADES = ['#d55e00', '#e69f00', '#a6501a', '#f0c35e', '#c9302c']; // distinct oranges/reds per TensorFlow run
const TORCH_SHADES = ['#6a4cdb', '#b07cf2', '#3b2a9e', '#8f9cf6', '#d18ae8']; // distinct purples per PyTorch run
const MIN_RESOLVED_BIT_ERRORS = 30; // matches factory6g.sim.stages.common
const CHANNELS = { rayleigh: 'Rayleigh', rician: 'Rician', tr38901: 'TR 38.901 UMi', raytraced: 'Ray-traced hall', awgn: 'AWGN' };
const STAGE_LABEL = { estimators: 'Estimators', resource_managers: 'Schedulers', coupled: 'Coupled' };
const REDUCED_MOTION = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const S = {
  snap: null, filter: 'all', sel: null, selStage: 0, selEb: null, ovStage: null, ovMethod: null,
  versions: new Map(), cards: new Map(), lastMsg: 0, conv: null, convKey: null,
  hall: { run: null, stage: 0, method: null, eb: null, view: 'both', scene: null, key: null, yaw: -0.65, pitch: 0.62, spin: !REDUCED_MOTION, dragging: false },
};
const $ = (id) => document.getElementById(id);

// ---------- DOM helpers (no innerHTML with data) ----------
function h(tag, attrs, ...kids) { const el = document.createElement(tag); setAttrs(el, attrs); append(el, kids); return el; }
function s(tag, attrs, ...kids) { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); setAttrs(el, attrs); append(el, kids); return el; }
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'onclick') el.addEventListener('click', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'text') el.textContent = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
}
function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}
const kids = (...xs) => xs.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false);
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const sup = (n) => String(n).split('').map((c) => SUP[c]).join('');
function sci(x, digits = 1) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= -2 && e <= 2) return x.toFixed(Math.max(0, 3 - Math.max(e, 0)));
  return `${(x / 10 ** e).toFixed(digits)}×10${sup(e)}`;
}
function dur(sec) {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) return '–';
  sec = Math.max(0, Math.round(sec));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  if (hh) return `${hh} h ${String(mm).padStart(2, '0')} min`;
  if (mm) return `${mm} min ${String(ss).padStart(2, '0')} s`;
  return `${ss} s`;
}
const intFmt = (n) => (Number.isFinite(n) ? n.toLocaleString('en-GB') : '–');
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------- run naming ----------
function channelKey(run) {
  const argv = run.argv || [];
  const i = argv.indexOf('--channel');
  if (i >= 0 && argv[i + 1]) return argv[i + 1];
  return Object.keys(CHANNELS).find((c) => run.name.includes(`_${c}_`)) || '';
}
const stackName = (k) => ({ torch: 'PyTorch', tf: 'TensorFlow' }[k] || 'stack not recorded');
function runTitle(run) {
  const stages = run.stages.map((st) => STAGE_LABEL[st.stage] || st.stage).join(' + ');
  const ch = CHANNELS[channelKey(run)] || channelKey(run);
  const seed = run.seed !== null && run.seed !== undefined ? `seed ${run.seed}` : 'no seed recorded';
  return [stages, ch, seed].filter(Boolean).join(' · ');
}
function startedAt(run) {
  const m = run.name.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[3]}/${m[2]} ${m[4]}:${m[5]}` : '';
}
const runVersion = (run) => run.stages.reduce((a, st) => a + st.version, 0);
const methodColor = (methods, m) => METHOD_COLORS[methods.indexOf(m) % METHOD_COLORS.length];
const utColor = (i) => UT_COLORS[i % UT_COLORS.length];

// ---------- BER from summed counts ----------
function cellBer(c) {
  if (!c || !c.bits) return null;
  if (c.bit_errors === 0) return { y: 3 / c.bits, bound: true, resolved: false, c }; // rule of three
  return { y: c.bit_errors / c.bits, bound: false, resolved: c.bit_errors >= MIN_RESOLVED_BIT_ERRORS, c };
}
function seriesOf(stage) {
  const by = {};
  for (const c of stage.cells) (by[c.method] ??= {})[c.ebno_db] = c;
  return stage.methods.map((m) => ({
    label: m, color: methodColor(stage.methods, m),
    points: stage.ebno_db.map((eb) => ({ x: eb, ...(cellBer(by[m]?.[eb]) || { y: null }) })).filter((p) => p.y !== null),
  }));
}

// ---------- charts ----------
function logChart(series, { width = 420, height = 220, xs = null, xlabel = 'Eb/N0 (dB)', ylabel = 'BER', compact = false } = {}) {
  const pad = { l: compact ? 44 : 54, r: 12, t: 10, b: compact ? 34 : 40 };
  const all = series.flatMap((se) => se.points);
  const ysPos = all.map((p) => p.y).filter((y) => y > 0);
  const xVals = xs || [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': `${ylabel} against ${xlabel}` });
  if (!ysPos.length || !xVals.length) {
    svg.append(s('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle' }, 'waiting for the first batch…'));
    return svg;
  }
  const lo = Math.max(-8, Math.floor(Math.log10(Math.min(...ysPos))));
  const hi = Math.min(0, Math.ceil(Math.log10(Math.max(...ysPos))));
  const top = hi === lo ? lo + 1 : hi;
  const bottom = lo - 0.45; // room under the lowest decade for the no-error arrows
  const x0 = Math.min(...xVals), x1 = Math.max(...xVals);
  const plotB = height - pad.b;
  const X = (x) => pad.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const Y = (y) => pad.t + (top - Math.log10(Math.max(y, 10 ** bottom))) / (top - bottom) * (plotB - pad.t);
  const labelEvery = top - lo > 6 ? 2 : 1;
  for (let e = lo; e <= top; e++) {
    svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: Y(10 ** e), y2: Y(10 ** e), class: 'grid' }));
    if ((top - e) % labelEvery === 0) svg.append(s('text', { x: pad.l - 6, y: Y(10 ** e) + 3.5, 'text-anchor': 'end' }, e === 0 ? '1' : `10${sup(e)}`));
  }
  svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: plotB, y2: plotB, class: 'grid' }));
  const step = xVals.length > 8 ? 2 : 1;
  xVals.forEach((x, i) => {
    if (i % step === 0) svg.append(s('text', { x: X(x), y: plotB + 14, 'text-anchor': 'middle' }, String(x)));
  });
  svg.append(s('text', { class: 'axis-title', x: (pad.l + width - pad.r) / 2, y: height - 4, 'text-anchor': 'middle' }, xlabel));
  const yMid = (pad.t + plotB) / 2;
  svg.append(s('text', { class: 'axis-title', x: 11, y: yMid, 'text-anchor': 'middle', transform: `rotate(-90 11 ${yMid})` }, ylabel));
  for (const se of series) {
    // One segment per run of points with errors; a no-error point (upper bound only) breaks the line.
    let d = '', prev = false;
    for (const p of se.points) {
      if (p.bound) { prev = false; continue; }
      d += `${prev ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`;
      prev = true;
    }
    if (d.includes('L')) svg.append(s('path', { class: 'curve', stroke: se.color, d }));
    for (const p of se.points) {
      const title = p.c ? `${se.label}, ${p.x} dB: ${p.bound ? `no errors yet, BER < ${sci(p.y)} (95 %)` : `BER ${sci(p.y, 2)}`} · ${intFmt(p.c.bit_errors)} bit errors in ${intFmt(p.c.bits)} bits, ${p.c.batches} batches`
        : `${se.label}, ${p.x}: ${sci(p.y, 2)}`;
      const filled = p.resolved && !p.bound;
      const cy = Y(p.y);
      svg.append(s('circle', { cx: X(p.x), cy, r: compact ? 2.6 : 3.2, fill: filled ? se.color : 'var(--panel)', stroke: se.color, 'stroke-width': 1.4 }, s('title', {}, title)));
      if (p.bound) svg.append(s('path', { d: `M${X(p.x)},${cy + 4}l0,5m-2.5,-2.5l2.5,2.5l2.5,-2.5`, stroke: se.color, fill: 'none', 'stroke-width': 1.2 }));
    }
  }
  return svg;
}
function linChart(values, { width = 260, height = 46, color = 'var(--brand)', label = 'trend' } = {}) {
  const v = values.map((x) => (Number.isFinite(x) ? x : null));
  const ok = v.filter((x) => x !== null);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'spark', preserveAspectRatio: 'none', role: 'img', 'aria-label': label });
  if (ok.length < 2) return svg;
  const mn = Math.min(...ok), mx = Math.max(...ok);
  const X = (i) => (i / (v.length - 1)) * (width - 2) + 1;
  const Y = (y) => height - 3 - ((y - mn) / (mx - mn || 1)) * (height - 6);
  svg.append(s('path', { d: v.map((y, i) => (y === null ? '' : `${i && v[i - 1] !== null ? 'L' : 'M'}${X(i).toFixed(1)},${Y(y).toFixed(1)}`)).join(''),
    fill: 'none', stroke: color, 'stroke-width': 1.6, 'vector-effect': 'non-scaling-stroke' }));
  return svg;
}

// ---------- run cards ----------
function chip(text, cls) { return h('span', { class: `chip ${cls}` }, text); }
function progressRow(run) {
  if (run.status !== 'running') {
    const traced = Math.max(...run.stages.map((x) => x.batches_traced));
    return h('div', { class: 'prog' }, `${intFmt(traced)} batches traced · last write ${dur(run.age_s)} ago`);
  }
  const p = run.progress;
  if (!p) return h('div', { class: 'prog' }, `${run.stages[0].batches_traced} batches traced`);
  const frac = Math.min(1, (p.batch - 1) / p.total);
  return h('div', { class: 'prog' },
    h('span', {}, `batch ${p.batch}/${p.total}`),
    h('div', { class: 'track', role: 'progressbar', 'aria-label': 'Batches done', 'aria-valuemin': 0, 'aria-valuemax': p.total, 'aria-valuenow': p.batch - 1 },
      h('div', { class: 'fill', style: { width: `${(100 * frac).toFixed(1)}%` } })),
    h('span', {}, p.seconds_per_batch ? `${p.seconds_per_batch.toFixed(0)} s/batch · ${dur(p.eta_s)} left` : ''));
}
function miniLegend(methods) {
  return h('div', { class: 'mini-legend' }, methods.map((m) => h('span', {}, h('i', { style: { background: methodColor(methods, m) } }), m)));
}
function runCard(run) {
  const st = run.stages[Math.min(S.sel === run.id ? S.selStage : 0, run.stages.length - 1)];
  const card = h('div', { role: 'button', tabindex: 0, class: `run-card ${run.status}`, 'aria-pressed': S.sel === run.id ? 'true' : 'false',
    'aria-label': `${runTitle(run)}, ${stackName(run.framework.key)}, ${run.status}. Open details`, onclick: () => select(run.id) },
  h('div', { class: 'title' }, runTitle(run), h('span', { class: 'when' }, startedAt(run))),
  h('div', { class: 'meta' }, chip(run.status, run.status), chip(stackName(run.framework.key), run.framework.key), h('span', {}, `folder: ${run.root}`)),
  progressRow(run),
  logChart(seriesOf(st), { width: 380, height: 180, xs: st.ebno_db, compact: true }),
  miniLegend(st.methods));
  card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(run.id); } });
  return card;
}

// ---------- detail ----------
function select(id) {
  if (S.sel === id) { S.sel = null; renderDetail(); renderRuns(true); return; }
  S.sel = id; S.selStage = 0; S.selEb = null; S.conv = null; S.convKey = null;
  renderRuns(true); renderDetail();
  $('detail').scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'start' });
}
async function loadConvergence(run, st) {
  const key = `${run.id}|${st.dir}|${S.selEb}|${st.version}`;
  if (S.convKey === key) return;
  S.convKey = key;
  try {
    const q = new URLSearchParams({ run: run.id, stage: st.dir, eb: String(S.selEb) });
    const res = await fetch(`/api/convergence?${q}`);
    if (!res.ok) return;
    S.conv = await res.json();
    $('conv-box')?.replaceChildren(convChart(st));
  } catch (_) { /* next update retries */ }
}
function convChart(st) {
  if (!S.conv) return h('p', { class: 'note' }, 'loading…');
  const series = Object.entries(S.conv.series).map(([m, ys]) => ({
    label: m, color: methodColor(st.methods, m), points: ys.map((y, i) => ({ x: i + 1, y })).filter((p) => p.y && p.y > 0),
  }));
  return logChart(series, { width: 520, height: 250, xlabel: 'batches so far', ylabel: 'running BER' });
}
function berTable(st) {
  const by = {};
  for (const c of st.cells) (by[c.method] ??= {})[c.ebno_db] = c;
  return h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'Method'), st.ebno_db.map((eb) => h('th', {}, `${eb} dB`)))),
    h('tbody', {}, st.methods.map((m) => h('tr', {},
      h('td', {}, h('span', { class: 'swatch', style: { background: methodColor(st.methods, m), marginRight: '6px' } }), m),
      st.ebno_db.map((eb) => {
        const b = cellBer(by[m]?.[eb]);
        if (!b) return h('td', {}, '–');
        return h('td', { title: `${intFmt(b.c.bit_errors)} bit errors / ${intFmt(b.c.bits)} bits, ${b.c.batches} batches` },
          b.bound ? `< ${sci(b.y)}` : b.resolved ? h('strong', {}, sci(b.y)) : sci(b.y));
      }))))));
}
function tileEl(k, v, sub) { return h('div', { class: 'tile' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 's' }, sub || '')); }
function renderDetail() {
  const box = $('detail');
  const run = S.snap?.runs.find((r) => r.id === S.sel);
  if (!run) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false;
  const st = run.stages[Math.min(S.selStage, run.stages.length - 1)];
  if (S.selEb === null || !st.ebno_db.includes(S.selEb)) S.selEb = st.ebno_db[Math.floor(st.ebno_db.length / 2)] ?? 0;
  const bits = st.cells.reduce((a, c) => a + c.bits, 0);
  const p = run.progress;
  box.replaceChildren(...kids(
    h('div', { class: 'panel-head' }, h('h1', {}, runTitle(run), h('small', {}, `${stackName(run.framework.key)} · ${run.name}`)),
      h('button', { type: 'button', class: 'ghost', onclick: () => select(run.id) }, 'Close')),
    h('div', { class: 'tiles' },
      tileEl('Status', run.status, `last write ${dur(run.age_s)} ago`),
      tileEl('Stack', run.framework.label, run.config || ''),
      tileEl('Batches traced', intFmt(st.batches_traced), p ? `of ${p.total} · ${p.active_points}/${p.points} points still open` : ''),
      tileEl('Bits simulated', sci(bits, 2), `${st.methods.length} methods × ${st.ebno_db.length} Eb/N0 points`),
      tileEl('Time left', p ? dur(p.eta_s) : '–', p?.seconds_per_batch ? `${p.seconds_per_batch.toFixed(1)} s per batch` : '')),
    run.stages.length > 1 ? h('div', { class: 'seg', role: 'group', 'aria-label': 'Stage' }, run.stages.map((x, i) =>
      h('button', { type: 'button', 'aria-pressed': i === S.selStage ? 'true' : 'false', onclick: () => { S.selStage = i; S.convKey = null; renderDetail(); } }, STAGE_LABEL[x.stage] || x.stage))) : null,
    h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h2', {}, 'BER so far', h('small', {}, 'filled ≥ 30 bit errors · hollow fewer · arrow: none yet (95 % bound)')),
        logChart(seriesOf(st), { width: 520, height: 280, xs: st.ebno_db }), miniLegend(st.methods)),
      h('div', { class: 'card' }, h('h2', {}, 'Convergence', h('small', {}, 'running BER after each batch at one Eb/N0')),
        h('div', { class: 'seg', role: 'group', 'aria-label': 'Eb/N0 point' }, st.ebno_db.map((eb) =>
          h('button', { type: 'button', 'aria-pressed': eb === S.selEb ? 'true' : 'false', onclick: () => { S.selEb = eb; S.convKey = null; S.conv = null; renderDetail(); } }, `${eb}`))),
        h('div', { id: 'conv-box' }, convChart(st)))),
    h('div', { class: 'card' }, h('h2', {}, 'BER table', h('small', {}, 'bold: resolved (≥ 30 bit errors)')), berTable(st))));
  loadConvergence(run, st);
}

// ---------- hall: shared ----------
const HALL_DEFAULT = { room_dimensions: [15, 15, 5], machines: [], bs_position: [0, 0, 4], ut_height: 1 };
function outcomeOf(latest, i) {
  if (!latest || !latest.active?.[i]) return 'idle';
  return (latest.block_errors?.[i] ?? 0) === 0 ? 'delivered' : 'failed';
}
const outcomeColor = (o) => (o === 'delivered' ? cssVar('--ok-line') : o === 'failed' ? cssVar('--bad-line') : cssVar('--muted'));
const hallRuns = () => (S.snap?.runs || []).filter((r) => r.stages.length);
function syncHallControls() {
  const runs = hallRuns();
  if (!runs.find((r) => r.id === S.hall.run)) S.hall.run = (runs.find((r) => r.status === 'running') || runs[0])?.id ?? null;
  fill($('hall-run'), runs.map((r) => [r.id, `${r.status === 'running' ? '● ' : ''}${runTitle(r)} · ${stackName(r.framework.key)}`]), S.hall.run);
  const run = runs.find((r) => r.id === S.hall.run);
  if (!run) return null;
  const st = run.stages[Math.min(S.hall.stage, run.stages.length - 1)];
  if (!st.methods.includes(S.hall.method)) S.hall.method = st.methods[0] ?? null;
  if (!st.ebno_db.includes(S.hall.eb)) S.hall.eb = st.ebno_db[Math.floor(st.ebno_db.length / 2)] ?? null;
  fill($('hall-method'), st.methods.map((m) => [m, m]), S.hall.method);
  fill($('hall-eb'), st.ebno_db.map((e) => [String(e), `${e} dB`]), String(S.hall.eb));
  return { run, st };
}
async function refreshHall(force = false) {
  const sel = syncHallControls();
  if (!sel) { $('hall-body').replaceChildren(h('p', { class: 'empty' }, 'No traced runs yet.')); return; }
  const { run, st } = sel;
  const key = `${run.id}|${st.dir}|${S.hall.method}|${S.hall.eb}|${st.version}`;
  if (!force && key === S.hall.key) return;
  S.hall.key = key;
  try {
    const q = new URLSearchParams({ run: run.id, stage: st.dir, eb: String(S.hall.eb), method: S.hall.method || '' });
    const res = await fetch(`/api/scene?${q}`);
    if (!res.ok) return;
    S.hall.scene = await res.json();
  } catch (_) { return; }
  drawHall();
}
function drawHall() {
  const sc = S.hall.scene;
  const body = $('hall-body');
  if (!sc) return;
  const view = S.hall.view;
  if (!sc.drop) {
    body.replaceChildren(h('p', { class: 'empty' }, sc.large_scale_model === 'none'
      ? 'This run has no terminal positions (large_scale_model "none"): nothing to place in the hall.'
      : 'Waiting for the first batch of this run…'));
    return;
  }
  const panes = [];
  const caption = `batch ${sc.batch + 1} · ${sc.method} · ${sc.ebno_db} dB`;
  if (view !== '3d') panes.push(h('div', { class: 'hall-pane' }, h('h2', {}, '2D plan', h('small', {}, caption)), plan2d(sc)));
  if (view !== '2d') {
    panes.push(h('div', { class: 'hall-pane' }, h('h2', {}, '3D hall', h('small', {}, caption)), h('div', { id: 'hall3d-box' }, hall3d(sc)),
      h('div', { class: 'rot-ctl' }, 'Drag to turn and tilt.',
        h('button', { type: 'button', 'aria-pressed': S.hall.spin ? 'true' : 'false', onclick: (e) => { S.hall.spin = !S.hall.spin; e.target.setAttribute('aria-pressed', String(S.hall.spin)); e.target.textContent = S.hall.spin ? 'Pause rotation' : 'Rotate'; if (S.hall.spin) spin(); } }, S.hall.spin ? 'Pause rotation' : 'Rotate'),
        h('button', { type: 'button', onclick: () => { S.hall.yaw = -0.65; S.hall.pitch = 0.62; redraw3d(); } }, 'Reset view'))));
  }
  body.replaceChildren(
    h('div', { class: `hall-grid${panes.length === 1 ? ' single' : ''}` }, panes),
    hallLegend(),
    utStrip(sc));
  if (view !== '2d' && S.hall.spin) spin();
}
function hallLegend() {
  const dot = (fill, stroke) => s('svg', { width: 12, height: 12, 'aria-hidden': 'true' }, s('circle', { cx: 6, cy: 6, r: 4.5, fill, stroke, 'stroke-width': 1.6 }));
  const line = (c) => s('svg', { width: 18, height: 12, 'aria-hidden': 'true' }, s('line', { x1: 1, y1: 6, x2: 17, y2: 6, stroke: c, 'stroke-width': 3, 'stroke-linecap': 'round' }));
  return h('div', { class: 'legend' },
    h('span', {}, dot(UT_COLORS[0], 'none'), 'terminal, line of sight'),
    h('span', {}, dot(cssVar('--panel'), UT_COLORS[0]), 'terminal, no line of sight'),
    h('span', {}, line(cssVar('--ok-line')), 'scheduled, packet delivered'),
    h('span', {}, line(cssVar('--bad-line')), 'scheduled, packet failed'),
    h('span', {}, dot(cssVar('--muted'), 'none'), 'earlier drops (faint)'),
    h('span', {}, '▲ base station (ceiling, 4 m)'),
    h('span', {}, 'link width: transmit power'));
}
function utStrip(sc) {
  const d = sc.drop, L = sc.latest, P = sc.per_ut;
  const batches = (sc.history || []).length;
  return h('div', { class: 'ut-strip' }, d.positions.map((p, i) => {
    const o = outcomeOf(L, i);
    const sched = P?.scheduled?.[i] ?? 0, blocks = P?.blocks?.[i] ?? 0, errs = P?.block_errors?.[i] ?? 0;
    return h('div', { class: 'ut-card' },
      h('div', { class: 'top' }, h('i', { class: 'swatch', style: { background: utColor(i) } }), `UT${i}`, chip(o === 'idle' ? 'not scheduled' : o, o)),
      h('dl', {},
        h('dt', {}, 'Position'), h('dd', {}, `${p[0].toFixed(1)}, ${p[1].toFixed(1)} m`),
        h('dt', {}, 'Path'), h('dd', {}, d.los?.[i] === false ? 'no line of sight' : 'line of sight'),
        h('dt', {}, 'Large-scale gain'), h('dd', {}, d.gain_db ? `${d.gain_db[i] >= 0 ? '+' : ''}${d.gain_db[i].toFixed(1)} dB` : '–'),
        h('dt', {}, 'Power'), h('dd', {}, L?.active?.[i] ? `${Math.round(100 * (L.power?.[i] ?? 1))} %` : '–'),
        h('dt', {}, 'Scheduled so far'), h('dd', {}, batches ? `${sched} of ${batches} batches` : '–'),
        h('dt', {}, 'Block error rate'), h('dd', {}, blocks ? sci(errs / blocks) : '–')));
  }));
}

// ---------- hall: 2D plan ----------
function plan2d(sc) {
  const hall = sc.hall || HALL_DEFAULT;
  const [L, W] = hall.room_dimensions;
  const size = 560, m = 30, k = size / Math.max(L, W);
  const px = (x) => m + (x + L / 2) * k, py = (y) => m + (W / 2 - y) * k;
  const svg = s('svg', { viewBox: `0 0 ${size + 2 * m} ${size + 2 * m}`, class: 'hall-svg', role: 'img', 'aria-label': 'Plan of the hall with the latest terminal drop and scheduled links' });
  svg.append(s('rect', { x: m, y: m, width: L * k, height: W * k, fill: cssVar('--soft'), stroke: cssVar('--ink'), 'stroke-width': 1.4, rx: 2 }));
  for (let g = -L / 2 + 2.5; g < L / 2; g += 2.5) svg.append(s('line', { x1: px(g), x2: px(g), y1: py(W / 2), y2: py(-W / 2), stroke: cssVar('--line'), 'stroke-width': 0.8 }));
  for (let g = -W / 2 + 2.5; g < W / 2; g += 2.5) svg.append(s('line', { x1: px(-L / 2), x2: px(L / 2), y1: py(g), y2: py(g), stroke: cssVar('--line'), 'stroke-width': 0.8 }));
  for (const [x, label] of [[-L / 2, `${-L / 2}`], [0, '0'], [L / 2, `${L / 2} m`]]) svg.append(s('text', { x: px(x), y: m + W * k + 18, 'text-anchor': 'middle' }, label));
  (hall.machines || []).forEach((mc, i) => {
    svg.append(s('rect', { x: px(mc.x - mc.sx / 2), y: py(mc.y + mc.sy / 2), width: mc.sx * k, height: mc.sy * k, rx: 3, fill: cssVar('--machine'), stroke: cssVar('--machine-line') }));
    svg.append(s('text', { x: px(mc.x), y: py(mc.y) + 4, 'text-anchor': 'middle' }, `M${i + 1}`));
  });
  for (const d of sc.history || []) {
    d.positions.forEach((p, i) => svg.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 2.2, fill: utColor(i), opacity: 0.18 })));
  }
  const bs = hall.bs_position;
  const d = sc.drop, L0 = sc.latest;
  d.positions.forEach((p, i) => {
    const o = outcomeOf(L0, i);
    if (o === 'idle') return;
    const w = 2 + 4 * (L0.power?.[i] ?? 1);
    svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: cssVar('--panel'), 'stroke-width': w + 4, 'stroke-linecap': 'round' }));
    svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: outcomeColor(o), 'stroke-width': w, 'stroke-linecap': 'round' },
      s('title', {}, `UT${i}: ${o}`)));
  });
  svg.append(s('polygon', { points: `${px(bs[0])},${py(bs[1]) - 13} ${px(bs[0]) + 12},${py(bs[1]) + 8} ${px(bs[0]) - 12},${py(bs[1]) + 8}`, fill: cssVar('--ink'), stroke: cssVar('--panel'), 'stroke-width': 2 }));
  svg.append(s('text', { x: px(bs[0]) - 16, y: py(bs[1]) - 6, 'text-anchor': 'end', class: 'ink', style: { fontWeight: 600 } }, 'BS'));
  d.positions.forEach((p, i) => {
    const los = d.los ? d.los[i] : true;
    const g = s('g', { class: 'ut', opacity: L0?.active?.[i] ? 1 : 0.55 });
    g.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 9, fill: los ? utColor(i) : cssVar('--panel'), stroke: los ? cssVar('--panel') : utColor(i), 'stroke-width': los ? 2.5 : 3 }));
    g.append(s('text', { x: px(p[0]) + 13, y: py(p[1]) - 9, class: 'ink', style: { fontWeight: 600 } }, `UT${i}`));
    g.append(s('title', {}, `UT${i}: ${los ? 'line of sight' : 'no line of sight'}, ${d.gain_db ? `${d.gain_db[i].toFixed(1)} dB` : ''}, ${outcomeOf(L0, i)}`));
    svg.append(g);
  });
  return svg;
}

// ---------- hall: 3D (orthographic projection, painter's order) ----------
function project(yaw, pitch, scale, cx, cy) {
  const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return ([x, y, z]) => {
    const xr = x * cyw - y * syw, yr = x * syw + y * cyw;
    return { X: cx + scale * xr, Y: cy - scale * (z * cp - yr * sp), depth: yr * cp + z * sp };
  };
}
function boxFaces(x0, y0, z0, x1, y1, z1) {
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  return [[0, 1, 5, 4, 0.82], [1, 2, 6, 5, 0.7], [2, 3, 7, 6, 0.82], [3, 0, 4, 7, 0.7], [4, 5, 6, 7, 1]].map(([a, b, c, d, shade]) => ({ pts: [v[a], v[b], v[c], v[d]], shade }));
}
function hall3d(sc) {
  const hall = sc.hall || HALL_DEFAULT;
  const [L, W, H] = hall.room_dimensions;
  const width = 600, height = 470;
  // Fit the room's bounding sphere, so the view keeps its size and centre while it turns.
  const radius = Math.hypot(L / 2, W / 2, H / 2);
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const c0 = project(S.hall.yaw, S.hall.pitch, scale, 0, 0)([0, 0, H / 2]);
  const P = project(S.hall.yaw, S.hall.pitch, scale, width / 2 - c0.X, height / 2 - c0.Y);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'hall-svg drag', role: 'img', 'aria-label': '3D view of the hall: machines, ceiling base station, terminals and scheduled links' });
  const poly = (pts) => pts.map((p) => { const q = P(p); return `${q.X.toFixed(1)},${q.Y.toFixed(1)}`; }).join(' ');
  const ink = cssVar('--ink'), line = cssVar('--line'), muted = cssVar('--muted');
  svg.append(s('polygon', { points: poly([[-L / 2, -W / 2, 0], [L / 2, -W / 2, 0], [L / 2, W / 2, 0], [-L / 2, W / 2, 0]]), fill: cssVar('--soft'), stroke: muted, 'stroke-width': 1 }));
  for (let g = -L / 2 + 2.5; g < L / 2; g += 2.5) svg.append(s('polyline', { points: poly([[g, -W / 2, 0], [g, W / 2, 0]]), stroke: line, 'stroke-width': 0.8, fill: 'none' }));
  for (let g = -W / 2 + 2.5; g < W / 2; g += 2.5) svg.append(s('polyline', { points: poly([[-L / 2, g, 0], [L / 2, g, 0]]), stroke: line, 'stroke-width': 0.8, fill: 'none' }));
  const corners = [[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]];
  corners.forEach(([x, y]) => svg.append(s('polyline', { points: poly([[x, y, 0], [x, y, H]]), stroke: muted, 'stroke-width': 0.8, 'stroke-dasharray': '3 3', fill: 'none' })));
  svg.append(s('polygon', { points: poly(corners.map(([x, y]) => [x, y, H])), fill: 'none', stroke: muted, 'stroke-width': 0.8, 'stroke-dasharray': '3 3' }));
  for (const d of sc.history || []) d.positions.forEach((p, i) => { const q = P([p[0], p[1], 0]); svg.append(s('circle', { cx: q.X, cy: q.Y, r: 1.6, fill: utColor(i), opacity: 0.2 })); });
  const items = [];
  (hall.machines || []).forEach((mc, mi) => {
    for (const f of boxFaces(mc.x - mc.sx / 2, mc.y - mc.sy / 2, 0, mc.x + mc.sx / 2, mc.y + mc.sy / 2, mc.sz)) {
      const depth = f.pts.reduce((a, p) => a + P(p).depth, 0) / 4;
      items.push({ depth, el: s('polygon', { points: poly(f.pts), fill: cssVar('--machine'), stroke: cssVar('--machine-line'), 'stroke-width': 0.9, opacity: 0.55 + 0.45 * f.shade }) });
    }
    const top = P([mc.x, mc.y, mc.sz]);
    items.push({ depth: top.depth - 0.5, el: s('text', { x: top.X, y: top.Y + 4, 'text-anchor': 'middle' }, `M${mi + 1}`) });
  });
  const bs = hall.bs_position, uh = hall.ut_height ?? 1;
  const d = sc.drop, L0 = sc.latest;
  d.positions.forEach((p, i) => {
    const o = outcomeOf(L0, i);
    const q = P([p[0], p[1], uh]), f = P([p[0], p[1], 0]);
    const los = d.los ? d.los[i] : true;
    if (o !== 'idle') {
      const a = P(bs);
      items.push({ depth: (a.depth + q.depth) / 2, el: s('line', { x1: a.X, y1: a.Y, x2: q.X, y2: q.Y, stroke: outcomeColor(o), 'stroke-width': 1.5 + 3 * (L0.power?.[i] ?? 1), 'stroke-linecap': 'round', opacity: 0.9 }, s('title', {}, `UT${i}: ${o}`)) });
    }
    items.push({ depth: q.depth - 0.001, el: s('g', { opacity: L0?.active?.[i] ? 1 : 0.55 },
      s('ellipse', { cx: f.X, cy: f.Y, rx: 7, ry: 3, fill: ink, opacity: 0.15 }),
      s('line', { x1: f.X, y1: f.Y, x2: q.X, y2: q.Y, stroke: utColor(i), 'stroke-width': 1.2, 'stroke-dasharray': '2 2' }),
      s('circle', { cx: q.X, cy: q.Y, r: 7.5, fill: los ? utColor(i) : cssVar('--panel'), stroke: los ? cssVar('--panel') : utColor(i), 'stroke-width': los ? 2 : 2.6 }),
      s('text', { x: q.X + 11, y: q.Y - 7, class: 'ink', style: { fontWeight: 600 } }, `UT${i}`),
      s('title', {}, `UT${i}: ${los ? 'line of sight' : 'no line of sight'}, ${outcomeOf(L0, i)}`)) });
  });
  const b = P(bs), c = P([bs[0], bs[1], H]);
  items.push({ depth: b.depth, el: s('g', {},
    s('line', { x1: c.X, y1: c.Y, x2: b.X, y2: b.Y, stroke: ink, 'stroke-width': 2 }),
    s('polygon', { points: `${b.X},${b.Y + 11} ${b.X + 10},${b.Y - 6} ${b.X - 10},${b.Y - 6}`, fill: ink, stroke: cssVar('--panel'), 'stroke-width': 1.5 }),
    s('text', { x: b.X + 14, y: b.Y + 4, class: 'ink', style: { fontWeight: 600 } }, 'BS')) });
  items.sort((u, v) => v.depth - u.depth);
  for (const it of items) svg.append(it.el);
  const o = P([-L / 2, -W / 2, 0]), xa = P([-L / 2 + 2, -W / 2, 0]), ya = P([-L / 2, -W / 2 + 2, 0]), za = P([-L / 2, -W / 2, 2]);
  for (const [t, lab] of [[xa, 'x'], [ya, 'y'], [za, 'z']]) {
    svg.append(s('line', { x1: o.X, y1: o.Y, x2: t.X, y2: t.Y, stroke: muted, 'stroke-width': 1.4 }));
    svg.append(s('text', { x: t.X + (t.X - o.X) * 0.25, y: t.Y + (t.Y - o.Y) * 0.25 + 3, 'text-anchor': 'middle' }, lab));
  }
  svg.append(s('text', { x: 12, y: height - 12 }, `${L} × ${W} × ${H} m hall · axes 2 m`));
  wireDrag(svg);
  return svg;
}
function redraw3d() {
  const box = $('hall3d-box');
  if (box && S.hall.scene?.drop) box.replaceChildren(hall3d(S.hall.scene));
}
function wireDrag(svg) {
  let last = null;
  svg.addEventListener('pointerdown', (e) => { last = [e.clientX, e.clientY]; S.hall.dragging = true; svg.setPointerCapture(e.pointerId); });
  svg.addEventListener('pointermove', (e) => {
    if (!last) return;
    S.hall.yaw += (e.clientX - last[0]) * 0.01;
    S.hall.pitch = Math.max(0.12, Math.min(1.45, S.hall.pitch + (e.clientY - last[1]) * 0.006));
    last = [e.clientX, e.clientY];
    redraw3d();
  });
  const end = () => { last = null; S.hall.dragging = false; };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
}
let spinning = false;
function spin() {
  if (spinning) return;
  spinning = true;
  let t0 = performance.now();
  const stepFn = (t) => {
    if (!S.hall.spin || !$('hall3d-box')) { spinning = false; return; }
    if (!S.hall.dragging && !document.hidden) { S.hall.yaw += (t - t0) * 0.00012; redraw3d(); }
    t0 = t;
    setTimeout(() => requestAnimationFrame(stepFn), 80); // about 12 frames per second keeps the CPU quiet
  };
  requestAnimationFrame(stepFn);
}

// ---------- overlay across runs ----------
function renderOverlay() {
  const runs = S.snap?.runs || [];
  const stageNames = [...new Set(runs.flatMap((r) => r.stages.map((x) => x.stage)))];
  if (!stageNames.includes(S.ovStage)) S.ovStage = stageNames[0] ?? null;
  const withStage = runs.map((r) => ({ r, st: r.stages.find((x) => x.stage === S.ovStage) })).filter((x) => x.st);
  const methods = [...new Set(withStage.flatMap((x) => x.st.methods))];
  if (!methods.includes(S.ovMethod)) S.ovMethod = methods.includes('ls') ? 'ls' : methods[0] ?? null;
  fill($('ov-stage'), stageNames.map((n) => [n, STAGE_LABEL[n] || n]), S.ovStage);
  fill($('ov-method'), methods.map((m) => [m, m]), S.ovMethod);
  const counters = { tf: 0, torch: 0, unknown: 0 };
  const series = withStage.filter((x) => x.st.methods.includes(S.ovMethod)).map(({ r, st }) => {
    const k = r.framework.key;
    const shades = k === 'torch' ? TORCH_SHADES : TF_SHADES;
    const color = k === 'unknown' ? '#8a8f98' : shades[counters[k]++ % shades.length];
    const se = seriesOf(st).find((x) => x.label === S.ovMethod);
    return { ...se, color, label: `${stackName(k)} · ${CHANNELS[channelKey(r)] || channelKey(r)} · ${r.seed !== null && r.seed !== undefined ? `seed ${r.seed}` : startedAt(r)}${r.status === 'running' ? ' · live' : ''}` };
  });
  $('ov-legend').replaceChildren(...series.map((se) => h('span', {}, h('i', { style: { background: se.color } }), se.label)));
  const xs = [...new Set(series.flatMap((se) => se.points.map((p) => p.x)))].sort((a, b) => a - b);
  $('overlay').replaceChildren(series.length ? logChart(series, { width: 1000, height: 300, xs }) : h('p', { class: 'empty' }, 'No run has this method yet.'));
}
function fill(sel, options, value) {
  const cur = [...sel.options].map((o) => `${o.value}=${o.textContent}`).join('|');
  if (cur !== options.map((o) => `${o[0]}=${o[1]}`).join('|')) sel.replaceChildren(...options.map(([v, t]) => h('option', { value: v }, t)));
  if (value !== null && value !== undefined) sel.value = value;
}

// ---------- training ----------
function bars(obj, fmt) {
  if (!obj) return null;
  const entries = Object.entries(obj).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
  const mx = Math.max(...entries.map((e) => e[1]), 1e-12);
  return h('div', { class: 'bars' }, entries.flatMap(([k, v]) => [h('span', {}, `${k} dB`),
    h('div', { class: 'b', style: { width: `${(100 * v / mx).toFixed(1)}%` } }), h('span', {}, fmt(v))]));
}
function trainingCard(t) {
  const frac = Math.min(1, t.iteration / t.total);
  return h('div', { class: `run-card static ${t.status}` },
    h('div', { class: 'title' }, `PPO · ${t.name.replace(/^ppo_/, '')} policy`),
    h('div', { class: 'meta' }, chip(t.status, t.status), h('span', {}, `folder: ${t.root}`)),
    h('div', { class: 'prog' }, h('span', {}, `iteration ${t.iteration}/${t.total}`),
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(100 * frac).toFixed(1)}%` } })),
      h('span', {}, t.seconds_per_iteration ? `${t.seconds_per_iteration.toFixed(0)} s/iter · ${dur(t.eta_s)} left` : '')),
    h('div', { class: 'subhead' }, 'Mean step reward'), linChart(t.reward, { label: 'mean step reward per iteration' }),
    h('div', { class: 'subhead' }, 'Rollout BER'), linChart(t.ber, { color: '#cc6677', label: 'rollout BER per iteration' }),
    t.bler_by_ebno ? [h('div', { class: 'subhead' }, 'Latest rollout BLER per Eb/N0'), bars(t.bler_by_ebno, (v) => sci(v))] : null,
    t.lagrange ? [h('div', { class: 'subhead' }, 'Lagrange multipliers (1 % BLER constraint; 10 = cap)'), bars(t.lagrange, (v) => v.toFixed(2))] : null);
}

// ---------- top-level render ----------
function renderCounts() {
  const runs = S.snap?.runs || [];
  const n = (st) => runs.filter((r) => r.status === st).length;
  $('counts').replaceChildren(chip(`${n('running')} running`, 'running'), chip(`${n('stopped')} stopped`, 'stopped'), chip(`${n('finished')} finished`, 'finished'));
}
function visible(run) { return S.filter === 'all' || run.status === S.filter || (S.filter === 'running' && run.status === 'stopped'); }
function renderRuns(force = false) {
  const grid = $('runs');
  const runs = (S.snap?.runs || []).filter(visible);
  const keep = new Set();
  const order = [];
  for (const run of runs) {
    keep.add(run.id);
    const v = `${runVersion(run)}|${run.status}|${run.progress?.batch}|${S.sel === run.id}|${S.selStage}|${Math.round(run.age_s / 60)}`;
    let el = S.cards.get(run.id);
    if (!el || force || S.versions.get(run.id) !== v) {
      const grew = el && S.versions.get(run.id)?.split('|')[0] !== String(runVersion(run));
      el = runCard(run);
      if (grew && !REDUCED_MOTION) el.classList.add('flash');
      S.cards.set(run.id, el);
      S.versions.set(run.id, v);
    }
    order.push(el);
  }
  for (const id of [...S.cards.keys()]) if (!keep.has(id)) { S.cards.delete(id); S.versions.delete(id); }
  if (!order.length) { grid.replaceChildren(h('p', { class: 'empty' }, 'No traced runs under the watched folders yet. Start a run with simulation.trace on.')); return; }
  grid.replaceChildren(...order);
}
function renderTraining() {
  const list = S.snap?.training || [];
  $('training').replaceChildren(...(list.length ? list.map(trainingCard) : [h('p', { class: 'empty' }, 'No training logs in the watched folders.')]));
}
function onSnapshot(snap) {
  S.snap = snap;
  S.lastMsg = Date.now();
  renderCounts(); renderRuns(); renderOverlay(); renderTraining(); refreshHall();
  if (S.sel) renderDetail();
  updateConn();
}

// ---------- connection ----------
function updateConn() {
  if (!$('conn').classList.contains('on')) return;
  const polled = S.snap?.polled_at ? S.snap.polled_at * 1000 : S.lastMsg;
  const age = (Date.now() - polled) / 1000;
  $('conn-text').textContent = `Live · checked ${age < 4 ? 'just now' : `${Math.round(age)} s ago`}`;
}
function connect() {
  const conn = $('conn');
  if (!('EventSource' in window)) { poll(); return; }
  const es = new EventSource('/api/stream');
  es.onopen = () => { conn.className = 'live-conn on'; updateConn(); };
  es.onmessage = (e) => { conn.className = 'live-conn on'; onSnapshot(JSON.parse(e.data)); };
  es.addEventListener('ping', (e) => { conn.className = 'live-conn on'; if (S.snap) S.snap.polled_at = JSON.parse(e.data).polled_at; updateConn(); });
  es.onerror = () => { conn.className = 'live-conn off'; $('conn-text').textContent = 'reconnecting…'; };
}
async function poll() {
  try { const r = await fetch('/api/runs'); $('conn').className = 'live-conn on'; onSnapshot(await r.json()); } catch (_) { $('conn').className = 'live-conn off'; }
  setTimeout(poll, 3000);
}
setInterval(updateConn, 1000);

function wire() {
  $('filter').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-filter]');
    if (!b) return;
    S.filter = b.dataset.filter;
    for (const x of $('filter').querySelectorAll('button')) x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
    renderRuns(true);
  });
  $('hall-view').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-view]');
    if (!b) return;
    S.hall.view = b.dataset.view;
    for (const x of $('hall-view').querySelectorAll('button')) x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
    drawHall();
  });
  $('hall-run').addEventListener('change', (e) => { S.hall.run = e.target.value; S.hall.method = null; S.hall.eb = null; refreshHall(true); });
  $('hall-method').addEventListener('change', (e) => { S.hall.method = e.target.value; refreshHall(true); });
  $('hall-eb').addEventListener('change', (e) => { S.hall.eb = Number(e.target.value); refreshHall(true); });
  $('ov-stage').addEventListener('change', (e) => { S.ovStage = e.target.value; renderOverlay(); });
  $('ov-method').addEventListener('change', (e) => { S.ovMethod = e.target.value; renderOverlay(); });
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    $('theme-toggle').textContent = dark ? 'Light theme' : 'Dark theme';
    try { localStorage.setItem('f6g-theme', dark ? 'dark' : 'light'); } catch (_) { /* storage may be blocked */ }
    renderRuns(true); renderOverlay(); drawHall();
  });
  try {
    const saved = localStorage.getItem('f6g-theme');
    const theme = saved || (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : null);
    if (theme) { document.documentElement.dataset.theme = theme; $('theme-toggle').textContent = theme === 'dark' ? 'Light theme' : 'Dark theme'; }
  } catch (_) { /* storage may be blocked */ }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.sel) select(S.sel); });
}
wire();
connect();
