'use strict';

// Factory6G live monitor. Receives snapshots from `python -m factory6g.viewer monitor`
// over Server-Sent Events (/api/stream) and redraws only what changed.

const METHOD_COLORS = ['#3a56a8', '#cc6677', '#117733', '#aa4499', '#3f8fbf', '#999933', '#882255', '#44aa99'];
const TF_SHADES = ['#c46a1b', '#d98a3d', '#e3a462', '#b5561a', '#efbd86'];
const TORCH_SHADES = ['#5b3fd1', '#7d66e0', '#9c8aea', '#4630a8', '#b8aaf2'];
const MIN_RESOLVED_BIT_ERRORS = 30; // matches factory6g.sim.stages.common
const CHANNELS = ['rayleigh', 'rician', 'tr38901', 'raytraced', 'awgn'];
const STAGE_LABEL = { estimators: 'Estimators', resource_managers: 'Schedulers', coupled: 'Coupled' };

const S = { snap: null, filter: 'all', sel: null, selStage: 0, selEb: null, ovStage: null, ovMethod: null,
  versions: new Map(), cards: new Map(), lastMsg: 0, conv: null, convKey: null };
const $ = (id) => document.getElementById(id);

// ---------- DOM helpers (no innerHTML with data) ----------
function h(tag, attrs, ...kids) { const el = document.createElement(tag); setAttrs(el, attrs); append(el, kids); return el; }
function s(tag, attrs, ...kids) { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); setAttrs(el, attrs); append(el, kids); return el; }
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'onclick') el.addEventListener('click', v);
    else if (k === 'onchange') el.addEventListener('change', v);
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

// ---------- run naming ----------
function channelOf(run) {
  const argv = run.argv || [];
  const i = argv.indexOf('--channel');
  if (i >= 0 && argv[i + 1]) return argv[i + 1];
  return CHANNELS.find((c) => run.name.includes(`_${c}_`)) || '';
}
function runTitle(run) {
  const stages = run.stages.map((st) => STAGE_LABEL[st.stage] || st.stage).join(' + ');
  const ch = channelOf(run);
  return `${stages}${ch ? ` · ${ch}` : ''}${run.seed !== null && run.seed !== undefined ? ` · seed ${run.seed}` : ''}`;
}
function startedAt(run) {
  const m = run.name.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[3]}/${m[2]} ${m[4]}:${m[5]} UTC` : run.name;
}
const runVersion = (run) => run.stages.reduce((a, st) => a + st.version, 0);
const methodColor = (methods, m) => METHOD_COLORS[methods.indexOf(m) % METHOD_COLORS.length];

// ---------- BER from summed counts ----------
function cellBer(c) {
  if (!c || !c.bits) return null;
  const resolved = c.bit_errors >= MIN_RESOLVED_BIT_ERRORS;
  if (c.bit_errors === 0) return { y: 3 / c.bits, bound: true, resolved: false, c }; // rule of three
  return { y: c.bit_errors / c.bits, bound: false, resolved, c };
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
  const pad = { l: compact ? 34 : 46, r: 10, t: 8, b: compact ? 20 : 32 };
  const all = series.flatMap((se) => se.points);
  const ysPos = all.map((p) => p.y).filter((y) => y > 0);
  const xVals = xs || [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': `${ylabel} against ${xlabel}` });
  if (!ysPos.length || !xVals.length) {
    svg.append(s('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle' }, 'waiting for the first batch…'));
    return svg;
  }
  const lo = Math.max(-7, Math.floor(Math.log10(Math.min(...ysPos))));
  const hi = Math.min(0, Math.ceil(Math.log10(Math.max(...ysPos))));
  const top = hi === lo ? lo + 1 : hi;
  const x0 = Math.min(...xVals), x1 = Math.max(...xVals);
  const X = (x) => pad.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const Y = (y) => pad.t + (top - Math.log10(Math.max(y, 10 ** lo))) / (top - lo) * (height - pad.t - pad.b);
  for (let e = lo; e <= top; e++) {
    svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: Y(10 ** e), y2: Y(10 ** e), class: 'grid' }));
    svg.append(s('text', { x: pad.l - 4, y: Y(10 ** e) + 3, 'text-anchor': 'end' }, e === 0 ? '1' : `10${sup(e)}`));
  }
  const step = xVals.length > 8 ? 2 : 1;
  xVals.forEach((x, i) => { if (i % step === 0) svg.append(s('text', { x: X(x), y: height - pad.b + 13, 'text-anchor': 'middle' }, String(x))); });
  if (!compact) {
    svg.append(s('text', { x: (pad.l + width - pad.r) / 2, y: height - 4, 'text-anchor': 'middle' }, xlabel));
    svg.append(s('text', { x: 10, y: (pad.t + height - pad.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 10 ${(pad.t + height - pad.b) / 2})` }, ylabel));
  }
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
      const title = p.c ? `${se.label}, ${p.x} dB: ${p.bound ? `no errors, BER < ${sci(p.y)} (95 %)` : `BER ${sci(p.y, 2)}`} · ${intFmt(p.c.bit_errors)} bit errors in ${intFmt(p.c.bits)} bits, ${p.c.batches} batches`
        : `${se.label}, ${p.x}: ${sci(p.y, 2)}`;
      const filled = p.resolved && !p.bound;
      svg.append(s('circle', { cx: X(p.x), cy: Y(p.y), r: compact ? 2.4 : 3.2, fill: filled ? se.color : 'var(--panel)', stroke: se.color, 'stroke-width': 1.4 },
        s('title', {}, title)));
      if (p.bound) svg.append(s('path', { d: `M${X(p.x)},${Y(p.y) + 4}l0,6m-3,-3l3,3l3,-3`, stroke: se.color, fill: 'none', 'stroke-width': 1.2 }));
    }
  }
  return svg;
}
function linChart(values, { width = 260, height = 46, color = 'var(--brand)' } = {}) {
  const v = values.map((x) => (Number.isFinite(x) ? x : null));
  const ok = v.filter((x) => x !== null);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'spark', preserveAspectRatio: 'none', role: 'img', 'aria-label': 'trend' });
  if (ok.length < 2) return svg;
  const mn = Math.min(...ok), mx = Math.max(...ok);
  const X = (i) => (i / (v.length - 1)) * (width - 2) + 1;
  const Y = (y) => height - 3 - ((y - mn) / (mx - mn || 1)) * (height - 6);
  svg.append(s('path', { d: v.map((y, i) => (y === null ? '' : `${i && v[i - 1] !== null ? 'L' : 'M'}${X(i).toFixed(1)},${Y(y).toFixed(1)}`)).join(''),
    fill: 'none', stroke: color, 'stroke-width': 1.6, 'vector-effect': 'non-scaling-stroke' }));
  return svg;
}

// ---------- pieces ----------
function chip(text, cls) { return h('span', { class: `chip ${cls}` }, text); }
function progressRow(run) {
  const st = run.stages[0];
  if (run.status !== 'running') {
    const traced = Math.max(...run.stages.map((x) => x.batches_traced));
    return h('div', { class: 'prog' }, `${intFmt(traced)} batches traced · last write ${dur(run.age_s)} ago`);
  }
  const p = run.progress;
  if (!p) return h('div', { class: 'prog' }, `${st.batches_traced} batches traced`);
  const frac = Math.min(1, (p.batch - 1) / p.total);
  return h('div', { class: 'prog' },
    h('span', {}, `${p.batch}/${p.total}`),
    h('div', { class: 'track', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': p.total, 'aria-valuenow': p.batch - 1 },
      h('div', { class: 'fill', style: { width: `${(100 * frac).toFixed(1)}%` } })),
    h('span', {}, p.seconds_per_batch ? `${p.seconds_per_batch.toFixed(0)} s/batch · ETA ${dur(p.eta_s)}` : ''));
}
function miniLegend(methods) {
  return h('div', { class: 'mini-legend' }, methods.map((m) => h('span', {}, h('i', { style: { background: methodColor(methods, m) } }), m)));
}
function runCard(run) {
  const st = run.stages[Math.min(S.sel === run.id ? S.selStage : 0, run.stages.length - 1)];
  const card = h('div', { role: 'button', tabindex: 0, class: `run-card ${run.status}`, 'aria-pressed': S.sel === run.id ? 'true' : 'false',
    'aria-label': `${runTitle(run)}, ${run.status}. Open details`, onclick: () => select(run.id) },
  h('div', { class: 'title' }, runTitle(run)),
  h('div', { class: 'meta' }, chip(run.status, run.status), chip(run.framework.key === 'torch' ? 'PyTorch' : run.framework.key === 'tf' ? 'TensorFlow' : '?', run.framework.key),
    h('span', {}, `${run.root} · started ${startedAt(run)}`)),
  progressRow(run),
  logChart(seriesOf(st), { width: 360, height: 150, xs: st.ebno_db, compact: true }),
  miniLegend(st.methods));
  card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(run.id); } });
  return card;
}

// ---------- detail ----------
function select(id) {
  if (S.sel === id) { S.sel = null; renderDetail(); renderRuns(true); return; }
  S.sel = id; S.selStage = 0; S.selEb = null; S.conv = null; S.convKey = null;
  renderRuns(true); renderDetail();
  $('detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    const box = $('conv-box');
    if (box) box.replaceChildren(convChart(st));
  } catch (_) { /* next update retries */ }
}
function convChart(st) {
  if (!S.conv) return h('p', { class: 'note' }, 'loading…');
  const series = Object.entries(S.conv.series).map(([m, ys]) => ({
    label: m, color: methodColor(st.methods, m), points: ys.map((y, i) => ({ x: i + 1, y })).filter((p) => p.y && p.y > 0),
  }));
  return logChart(series, { width: 460, height: 220, xlabel: 'batches so far (downsampled)', ylabel: 'running BER' });
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
function renderDetail() {
  const box = $('detail');
  const run = S.snap?.runs.find((r) => r.id === S.sel);
  if (!run) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false;
  const st = run.stages[Math.min(S.selStage, run.stages.length - 1)];
  if (S.selEb === null || !st.ebno_db.includes(S.selEb)) S.selEb = st.ebno_db[Math.floor(st.ebno_db.length / 2)] ?? 0;
  const bits = st.cells.reduce((a, c) => a + c.bits, 0);
  const p = run.progress;
  box.replaceChildren(...[
    h('h2', {}, runTitle(run), h('small', {}, run.name), h('button', { type: 'button', class: 'ghost close', onclick: () => select(run.id) }, 'Close')),
    h('div', { class: 'tiles' },
      h('div', { class: 'tile' }, h('div', { class: 'k' }, 'Status'), h('div', { class: 'v' }, run.status), h('div', { class: 's' }, `last write ${dur(run.age_s)} ago`)),
      h('div', { class: 'tile' }, h('div', { class: 'k' }, 'Stack'), h('div', { class: 'v', style: { fontSize: '14px' } }, run.framework.label), h('div', { class: 's' }, run.config || '')),
      h('div', { class: 'tile' }, h('div', { class: 'k' }, 'Batches traced'), h('div', { class: 'v' }, intFmt(st.batches_traced)), h('div', { class: 's' }, p ? `of ${p.total} · ${p.active_points}/${p.points} points still open` : '')),
      h('div', { class: 'tile' }, h('div', { class: 'k' }, 'Bits simulated'), h('div', { class: 'v' }, sci(bits, 2)), h('div', { class: 's' }, `${st.methods.length} methods × ${st.ebno_db.length} Eb/N0 points`)),
      h('div', { class: 'tile' }, h('div', { class: 'k' }, 'Time left'), h('div', { class: 'v' }, p ? dur(p.eta_s) : '–'), h('div', { class: 's' }, p?.seconds_per_batch ? `${p.seconds_per_batch.toFixed(1)} s per batch` : ''))),
    run.stages.length > 1 ? h('div', { class: 'seg', role: 'group', 'aria-label': 'Stage' }, run.stages.map((x, i) =>
      h('button', { type: 'button', 'aria-pressed': i === S.selStage ? 'true' : 'false', onclick: () => { S.selStage = i; S.convKey = null; renderDetail(); } }, STAGE_LABEL[x.stage] || x.stage))) : null,
    h('div', { class: 'cols' },
      h('div', { class: 'card' }, h('h2', {}, 'BER so far', h('small', {}, 'filled: ≥ 30 bit errors · hollow: fewer · arrow: no errors, 95 % bound')),
        logChart(seriesOf(st), { width: 460, height: 240, xs: st.ebno_db }), miniLegend(st.methods)),
      h('div', { class: 'card' }, h('h2', {}, 'Convergence', h('small', {}, 'running BER after each batch')),
        h('div', { class: 'seg', role: 'group', 'aria-label': 'Eb/N0 point' }, st.ebno_db.map((eb) =>
          h('button', { type: 'button', 'aria-pressed': eb === S.selEb ? 'true' : 'false', onclick: () => { S.selEb = eb; S.convKey = null; S.conv = null; renderDetail(); } }, `${eb}`))),
        h('div', { id: 'conv-box' }, convChart(st)))),
    h('div', { class: 'card' }, h('h2', {}, 'BER table', h('small', {}, 'bold: resolved (≥ 30 bit errors)')), berTable(st))].filter(Boolean));
  loadConvergence(run, st);
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
    return { ...se, label: `${k === 'torch' ? 'PyTorch' : k === 'tf' ? 'TF' : 'stack unrecorded'} · ${channelOf(r)} · seed ${r.seed ?? '?'}${r.status === 'running' ? ' (live)' : ''}`,
      color };
  });
  $('ov-legend').replaceChildren(...series.map((se) => h('span', {}, h('i', { class: 'swatch', style: { background: se.color } }), se.label)));
  const xs = [...new Set(series.flatMap((se) => se.points.map((p) => p.x)))].sort((a, b) => a - b);
  $('overlay').replaceChildren(series.length ? logChart(series, { width: 900, height: 260, xs }) : h('p', { class: 'empty' }, 'No run has this method yet.'));
}
function fill(sel, options, value) {
  const cur = [...sel.options].map((o) => o.value).join('|');
  if (cur !== options.map((o) => o[0]).join('|')) sel.replaceChildren(...options.map(([v, t]) => h('option', { value: v }, t)));
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
    h('div', { class: 'title' }, `PPO · ${t.name.replace(/^ppo_/, '')}`),
    h('div', { class: 'meta' }, chip(t.status, t.status), h('span', {}, t.root)),
    h('div', { class: 'prog' }, h('span', {}, `${t.iteration}/${t.total}`),
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(100 * frac).toFixed(1)}%` } })),
      h('span', {}, t.seconds_per_iteration ? `${t.seconds_per_iteration.toFixed(0)} s/iter · ETA ${dur(t.eta_s)}` : '')),
    h('div', { class: 'k note' }, 'Mean step reward'), linChart(t.reward),
    h('div', { class: 'k note' }, 'Rollout BER'), linChart(t.ber, { color: '#cc6677' }),
    t.bler_by_ebno ? [h('div', { class: 'k note' }, 'Latest rollout BLER per Eb/N0'), bars(t.bler_by_ebno, (v) => sci(v))] : null,
    t.lagrange ? [h('div', { class: 'k note' }, 'Lagrange multipliers (BLER constraint)'), bars(t.lagrange, (v) => v.toFixed(2))] : null);
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
    const v = `${runVersion(run)}|${run.status}|${run.progress?.batch}|${S.sel === run.id}|${S.selStage}`;
    let el = S.cards.get(run.id);
    if (!el || force || S.versions.get(run.id) !== v) {
      const grew = el && S.versions.get(run.id)?.split('|')[0] !== String(runVersion(run));
      const fresh = runCard(run);
      if (grew) fresh.classList.add('flash');
      el = fresh;
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
  renderCounts(); renderRuns(); renderOverlay(); renderTraining();
  if (S.sel) renderDetail();
}

// ---------- connection ----------
function connect() {
  const conn = $('conn');
  if (!('EventSource' in window)) { poll(); return; }
  const es = new EventSource('/api/stream');
  es.onmessage = (e) => { conn.className = 'live-conn on'; onSnapshot(JSON.parse(e.data)); };
  es.onopen = () => { conn.className = 'live-conn on'; };
  es.addEventListener('ping', (e) => { conn.className = 'live-conn on'; if (S.snap) S.snap.polled_at = JSON.parse(e.data).polled_at; });
  es.onerror = () => { conn.className = 'live-conn off'; $('conn-text').textContent = 'reconnecting…'; };
}
async function poll() {
  try { const r = await fetch('/api/runs'); onSnapshot(await r.json()); $('conn').className = 'live-conn on'; } catch (_) { $('conn').className = 'live-conn off'; }
  setTimeout(poll, 3000);
}
setInterval(() => {
  if (!S.lastMsg) return;
  const age = (Date.now() - (S.snap?.polled_at ? S.snap.polled_at * 1000 : S.lastMsg)) / 1000;
  if ($('conn').classList.contains('on')) $('conn-text').textContent = `Live · checked ${age < 3 ? 'just now' : `${Math.round(age)} s ago`}`;
}, 1000);

function wire() {
  $('filter').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-filter]');
    if (!b) return;
    S.filter = b.dataset.filter;
    for (const x of $('filter').querySelectorAll('button')) x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
    renderRuns(true);
  });
  $('ov-stage').addEventListener('change', (e) => { S.ovStage = e.target.value; renderOverlay(); });
  $('ov-method').addEventListener('change', (e) => { S.ovMethod = e.target.value; renderOverlay(); });
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    $('theme-toggle').textContent = dark ? 'Light theme' : 'Dark theme';
    try { localStorage.setItem('f6g-theme', dark ? 'dark' : 'light'); } catch (_) { /* storage may be blocked */ }
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
