'use strict';

// Factory6G live monitor: a small multi-page app (hash routes) fed by Server-Sent Events from
// `python -m factory6g.viewer monitor`. Pages: overview, simulations (live list), one page per
// run, factory hall (2D + 3D), compare runs, training.

const METHOD_COLORS = ['#3a56a8', '#cc6677', '#117733', '#aa4499', '#3f8fbf', '#999933', '#882255', '#44aa99'];
const UT_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#6b6b6b', '#882255'];
const TF_SHADES = ['#d55e00', '#e69f00', '#a6501a', '#f0c35e', '#c9302c'];
const TORCH_SHADES = ['#6a4cdb', '#b07cf2', '#3b2a9e', '#8f9cf6', '#d18ae8'];
const MIN_RESOLVED_BIT_ERRORS = 30; // matches factory6g.sim.stages.common
const CHANNELS = { rayleigh: 'Rayleigh', rician: 'Rician', tr38901: 'TR 38.901 UMi', raytraced: 'Ray-traced hall', awgn: 'AWGN' };
const STAGE_LABEL = { estimators: 'Estimators', resource_managers: 'Schedulers', coupled: 'Coupled' };
const REDUCED_MOTION = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const S = {
  snap: null, lastMsg: 0, route: { page: 'overview', id: null },
  runFilter: 'all', runQuery: '',
  detail: { stage: 0, eb: null, conv: null, convKey: null },
  ov: { stage: null, method: null },
  hall: { run: null, method: null, eb: null, view: 'both', scene: null, key: null, yaw: -0.65, pitch: 0.62, spin: !REDUCED_MOTION, dragging: false },
  rowCache: new Map(),
};
const $ = (id) => document.getElementById(id);
let ovHallKey = null; // run and batch the overview's hall preview shows

// ---------- DOM helpers (no innerHTML with data) ----------
function h(tag, attrs, ...kids) { const el = document.createElement(tag); setAttrs(el, attrs); append(el, kids); return el; }
function s(tag, attrs, ...kids) { const el = document.createElementNS('http://www.w3.org/2000/svg', tag); setAttrs(el, attrs); append(el, kids); return el; }
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'onclick') el.addEventListener('click', v);
    else if (k === 'onchange') el.addEventListener('change', v);
    else if (k === 'oninput') el.addEventListener('input', v);
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
const chip = (text, cls) => h('span', { class: `chip ${cls}` }, text);

// ---------- runs: naming and numbers ----------
const runs = () => S.snap?.runs || [];
const training = () => S.snap?.training || [];
const runById = (id) => runs().find((r) => r.id === id);
function channelKey(run) {
  const argv = run.argv || [];
  const i = argv.indexOf('--channel');
  if (i >= 0 && argv[i + 1]) return argv[i + 1];
  return Object.keys(CHANNELS).find((c) => run.name.includes(`_${c}_`)) || '';
}
const channelName = (run) => CHANNELS[channelKey(run)] || channelKey(run) || 'channel not recorded';
const stackName = (k) => ({ torch: 'PyTorch', tf: 'TensorFlow' }[k] || 'stack not recorded');
const stagesName = (run) => run.stages.map((st) => STAGE_LABEL[st.stage] || st.stage).join(' + ');
const seedName = (run) => (run.seed !== null && run.seed !== undefined ? `seed ${run.seed}` : 'no seed');
const runTitle = (run) => `${stagesName(run)} · ${channelName(run)} · ${seedName(run)}`;
function startedAt(run) {
  const m = run.name.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[3]}/${m[2]} ${m[4]}:${m[5]}` : '–';
}
const runVersion = (run) => run.stages.reduce((a, st) => a + st.version, 0);
const methodColor = (methods, m) => METHOD_COLORS[methods.indexOf(m) % METHOD_COLORS.length];
const utColor = (i) => UT_COLORS[i % UT_COLORS.length];
const bitsOf = (run) => run.stages.reduce((a, st) => a + st.cells.reduce((b, c) => b + c.bits, 0), 0);
const runHref = (run) => `#/run/${encodeURIComponent(run.id)}`;
const statusChip = (run) => (run.status === 'running' ? chip('running', 'running live') : chip(run.status, run.status));
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
// Reference point for the "now" column: the highest Eb/N0 at which every method still has errors.
function referenceEb(stage) {
  const by = {};
  for (const c of stage.cells) (by[c.ebno_db] ??= []).push(c);
  const ok = stage.ebno_db.filter((eb) => (by[eb] || []).length === stage.methods.length && by[eb].every((c) => c.bit_errors > 0));
  return ok.length ? ok[ok.length - 1] : stage.ebno_db[0];
}

// ---------- charts ----------
function logChart(series, { width = 640, height = 240, xs = null, xlabel = 'Eb/N0 (dB)', ylabel = 'BER', bare = false } = {}) {
  const pad = bare ? { l: 4, r: 4, t: 4, b: 4 } : { l: 52, r: 14, t: 10, b: 40 };
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
  const bottom = lo - (bare ? 0.1 : 0.45); // room under the lowest decade for the no-error arrows
  const x0 = Math.min(...xVals), x1 = Math.max(...xVals);
  const plotB = height - pad.b;
  const X = (x) => pad.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const Y = (y) => pad.t + (top - Math.log10(Math.max(y, 10 ** bottom))) / (top - bottom) * (plotB - pad.t);
  if (!bare) {
    const labelEvery = top - lo > 6 ? 2 : 1;
    for (let e = lo; e <= top; e++) {
      svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: Y(10 ** e), y2: Y(10 ** e), class: 'grid' }));
      if ((top - e) % labelEvery === 0) svg.append(s('text', { x: pad.l - 7, y: Y(10 ** e) + 3.5, 'text-anchor': 'end' }, e === 0 ? '1' : `10${sup(e)}`));
    }
    svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: plotB, y2: plotB, class: 'grid' }));
    const step = xVals.length > 8 ? 2 : 1;
    xVals.forEach((x, i) => { if (i % step === 0) svg.append(s('text', { x: X(x), y: plotB + 15, 'text-anchor': 'middle' }, String(x))); });
    svg.append(s('text', { class: 'axis-title', x: (pad.l + width - pad.r) / 2, y: height - 5, 'text-anchor': 'middle' }, xlabel));
    const yMid = (pad.t + plotB) / 2;
    svg.append(s('text', { class: 'axis-title', x: 12, y: yMid, 'text-anchor': 'middle', transform: `rotate(-90 12 ${yMid})` }, ylabel));
  }
  for (const se of series) {
    // One segment per run of points with errors; a no-error point (upper bound only) breaks the line.
    let d = '', prev = false;
    for (const p of se.points) {
      if (p.bound) { prev = false; continue; }
      d += `${prev ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`;
      prev = true;
    }
    if (d.includes('L')) svg.append(s('path', { class: 'curve', stroke: se.color, d, 'stroke-width': bare ? 1.5 : 1.9 }));
    if (bare) continue;
    for (const p of se.points) {
      const title = p.c ? `${se.label}, ${p.x} dB: ${p.bound ? `no errors yet, BER < ${sci(p.y)} (95 %)` : `BER ${sci(p.y, 2)}`} · ${intFmt(p.c.bit_errors)} bit errors in ${intFmt(p.c.bits)} bits, ${p.c.batches} batches`
        : `${se.label}, ${p.x}: ${sci(p.y, 2)}`;
      const cy = Y(p.y);
      svg.append(s('circle', { cx: X(p.x), cy, r: 3.2, fill: p.resolved && !p.bound ? se.color : 'var(--panel)', stroke: se.color, 'stroke-width': 1.4 }, s('title', {}, title)));
      if (p.bound) svg.append(s('path', { d: `M${X(p.x)},${cy + 4}l0,5m-2.5,-2.5l2.5,2.5l2.5,-2.5`, stroke: se.color, fill: 'none', 'stroke-width': 1.2 }));
    }
  }
  return svg;
}
function lineChart(values, { width = 560, height = 220, ylabel = '', xlabel = 'iteration', color = 'var(--brand)', yfmt = (v) => v.toFixed(2), log = false } = {}) {
  const pad = { l: 56, r: 14, t: 10, b: 40 };
  const v = values.map((x) => (Number.isFinite(x) && (!log || x > 0) ? (log ? Math.log10(x) : x) : null));
  const ok = v.filter((x) => x !== null);
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': `${ylabel} per ${xlabel}` });
  if (ok.length < 2) { svg.append(s('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle' }, 'waiting for data…')); return svg; }
  let mn = Math.min(...ok), mx = Math.max(...ok);
  if (mx === mn) { mn -= 1; mx += 1; }
  const plotB = height - pad.b;
  const X = (i) => pad.l + (i / (v.length - 1)) * (width - pad.l - pad.r);
  const Y = (y) => pad.t + (mx - y) / (mx - mn) * (plotB - pad.t);
  for (let k = 0; k <= 4; k++) {
    const y = mn + (k / 4) * (mx - mn);
    svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: Y(y), y2: Y(y), class: 'grid' }));
    svg.append(s('text', { x: pad.l - 7, y: Y(y) + 3.5, 'text-anchor': 'end' }, log ? sci(10 ** y) : yfmt(y)));
  }
  const ticks = Math.min(6, v.length - 1);
  for (let k = 0; k <= ticks; k++) {
    const i = Math.round((k / ticks) * (v.length - 1));
    svg.append(s('text', { x: X(i), y: plotB + 15, 'text-anchor': 'middle' }, String(i + 1)));
  }
  svg.append(s('text', { class: 'axis-title', x: (pad.l + width - pad.r) / 2, y: height - 5, 'text-anchor': 'middle' }, xlabel));
  const yMid = (pad.t + plotB) / 2;
  svg.append(s('text', { class: 'axis-title', x: 12, y: yMid, 'text-anchor': 'middle', transform: `rotate(-90 12 ${yMid})` }, ylabel));
  svg.append(s('path', { class: 'curve', stroke: color, d: v.map((y, i) => (y === null ? '' : `${i && v[i - 1] !== null ? 'L' : 'M'}${X(i).toFixed(1)},${Y(y).toFixed(1)}`)).join('') }));
  return svg;
}
const legendOf = (series) => h('div', { class: 'legend' }, series.map((se) => h('span', {}, h('i', { style: { background: se.color } }), se.label)));
function progressBar(run) {
  const p = run.progress;
  if (run.status !== 'running' || !p) {
    const traced = Math.max(...run.stages.map((x) => x.batches_traced));
    return h('div', { class: 'prog' }, `${intFmt(traced)} batches traced`);
  }
  const frac = Math.min(1, (p.batch - 1) / p.total);
  return h('div', { class: 'prog is-running' },
    h('span', {}, `${p.batch - 1}/${p.total}`),
    h('div', { class: 'track', role: 'progressbar', 'aria-label': 'Batches done', 'aria-valuemin': 0, 'aria-valuemax': p.total, 'aria-valuenow': p.batch - 1 },
      h('div', { class: 'fill', style: { width: `${(100 * frac).toFixed(1)}%` } })),
    h('span', {}, `${Math.round(100 * frac)} %`));
}
const kpi = (k, v, sub) => h('div', { class: 'kpi' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 's' }, sub || ''));

// ---------- shell ----------
const PAGES = {
  overview: { title: 'Overview', sub: 'Everything the watched results folders are doing right now.', render: renderOverview },
  runs: { title: 'Simulations', sub: 'Every traced simulation. Running ones first, each with its BER curve building up batch by batch.', render: renderRuns },
  run: { title: 'Simulation', sub: '', render: renderRun },
  hall: { title: 'Factory hall', sub: 'The latest batch of a run in the hall: where the terminals are, which were scheduled, and whether each packet decoded.', render: renderHall },
  compare: { title: 'Compare runs', sub: 'One method from every run that has it, so seeds and the TensorFlow / PyTorch stacks can be compared while they run.', render: renderCompare },
  training: { title: 'Training', sub: 'PPO training jobs, one point per iteration.', render: renderTraining },
};
function parseRoute() {
  const m = location.hash.replace(/^#\/?/, '').split('/');
  const page = PAGES[m[0]] ? m[0] : 'overview';
  return { page, id: page === 'run' && m[1] ? decodeURIComponent(m.slice(1).join('/')) : null };
}
function setHead({ title, sub, crumbs = [], actions = [] }) {
  $('page-title').textContent = title;
  $('page-sub').textContent = sub || '';
  $('page-sub').hidden = !sub;
  $('crumbs').replaceChildren(...crumbs.flatMap((c, i) => kids(i ? h('span', { 'aria-hidden': 'true' }, '/') : null, c.href ? h('a', { href: c.href }, c.text) : h('span', {}, c.text))));
  $('head-actions').replaceChildren(...kids(actions));
  document.title = `${title} · Factory6G live monitor`;
}
function navigate() {
  S.route = parseRoute();
  for (const a of $('side-nav').querySelectorAll('a')) {
    const on = a.dataset.page === S.route.page || (a.dataset.page === 'runs' && S.route.page === 'run');
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  S.rowCache.clear();
  S.hall.key = null;
  ovHallKey = null; // the overview's hall box is rebuilt, so fetch its scene again
  const page = PAGES[S.route.page];
  setHead({ title: page.title, sub: page.sub });
  $('page').replaceChildren();
  page.render(true);
  window.scrollTo(0, 0);
}
// Colours are read from CSS at draw time, so a theme change redraws the current page in place:
// controls, scroll position and the overview's hall preview stay where they are.
function redrawForTheme() {
  S.rowCache.clear();
  ovHallKey = null;
  S.hall.key = null;
  PAGES[S.route.page].render(false);
  if (S.route.page === 'hall') drawHall();
}
function updateBadges() {
  const nr = runs().filter((r) => r.status === 'running').length;
  const nt = training().filter((t) => t.status === 'running').length;
  $('badge-runs').hidden = !nr; $('badge-runs').textContent = String(nr); $('badge-runs').title = `${nr} running`;
  $('badge-train').hidden = !nt; $('badge-train').textContent = String(nt); $('badge-train').title = `${nt} running`;
}

// ---------- page: overview ----------
function renderOverview() {
  const rs = runs();
  const running = rs.filter((r) => r.status === 'running');
  const day = rs.filter((r) => r.status === 'finished' && r.age_s < 86400);
  const next = running.filter((r) => r.progress?.eta_s).sort((a, b) => a.progress.eta_s - b.progress.eta_s)[0];
  const tRun = training().filter((t) => t.status === 'running');
  const totalBits = rs.reduce((a, r) => a + bitsOf(r), 0);
  const hallBox = $('ov-hall');
  $('page').replaceChildren(
    h('div', { class: 'kpis' },
      kpi('Simulations running', String(running.length), `${rs.filter((r) => r.status === 'stopped').length} stopped · ${rs.length} watched`),
      kpi('Next to finish', next ? dur(next.progress.eta_s) : '–', next ? runTitle(next) : 'nothing running'),
      kpi('Finished in 24 h', String(day.length), 'traced runs with final results'),
      kpi('Bits simulated', sci(totalBits, 2), 'across every watched run'),
      kpi('Training jobs', String(tRun.length), tRun.length ? tRun.map((t) => `${t.name.replace(/^ppo_/, '')} ${t.iteration}/${t.total}`).join(' · ') : 'none running')),
    h('section', { class: 'section' },
      h('div', { class: 'section-head' }, h('h2', {}, 'Running now'), h('p', {}, 'progress and BER curve so far'), h('a', { class: 'more', href: '#/runs' }, 'All simulations →')),
      h('div', { class: 'panel' }, running.length ? h('div', { class: 'mini-list' }, running.map(miniRow)) : h('p', { class: 'empty-state' }, 'No simulation is running.'))),
    h('div', { class: 'two-col' },
      h('section', { class: 'section' },
        h('div', { class: 'section-head' }, h('h2', {}, 'Factory hall'), h('p', {}, 'latest drop of the first running run'), h('a', { class: 'more', href: '#/hall' }, 'Open 2D / 3D →')),
        hallBox || h('div', { class: 'panel', id: 'ov-hall' }, h('p', { class: 'empty-state' }, 'loading…'))),
      h('section', { class: 'section' },
        h('div', { class: 'section-head' }, h('h2', {}, 'Training'), h('a', { class: 'more', href: '#/training' }, 'Details →')),
        h('div', { class: 'panel' }, training().length ? h('div', { class: 'mini-list' }, training().map(trainingSummary)) : h('p', { class: 'empty-state' }, 'No training logs.')))));
  overviewHall(running[0] || rs[0]);
}
function miniRow(run) {
  const st = run.stages[0];
  return h('a', { class: 'mini-row', href: runHref(run) },
    h('div', {}, h('div', { class: 't' }, runTitle(run)),
      h('div', { class: 's' }, `${stackName(run.framework.key)} · ${run.progress?.seconds_per_batch ? `${run.progress.seconds_per_batch.toFixed(0)} s per batch · ${dur(run.progress.eta_s)} left` : 'starting'}`)),
    h('div', { class: 'mid' }, progressBar(run)),
    logChart(seriesOf(st), { width: 160, height: 52, xs: st.ebno_db, bare: true }));
}
async function overviewHall(run) {
  const box = $('ov-hall');
  if (!box) return;
  if (!run) { box.replaceChildren(h('p', { class: 'empty-state' }, 'No run to show.')); return; }
  const st = run.stages[0];
  const eb = st.ebno_db[Math.floor(st.ebno_db.length / 2)];
  const key = `${run.id}|${st.version}`;
  if (key === ovHallKey) return; // same batch already shown or being fetched
  ovHallKey = key;
  const sc = await fetchScene(run, st, st.methods[0], eb);
  const target = $('ov-hall');
  if (!target) return;
  target.replaceChildren(...(sc?.drop
    ? [h('h3', {}, runTitle(run), h('small', {}, `batch ${sc.batch + 1} · ${sc.method} · ${eb} dB`)), h('div', { class: 'ov-plan' }, plan2d(sc, { size: 420 }))]
    : [h('p', { class: 'empty-state' }, 'This run has no terminal positions yet.')]));
}
function trainingSummary(t) {
  return h('a', { class: 'mini-row', href: '#/training', style: { gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' } },
    h('div', {}, h('div', { class: 't' }, `PPO · ${t.name.replace(/^ppo_/, '')} policy`),
      h('div', { class: 's' }, t.seconds_per_iteration ? `${t.seconds_per_iteration.toFixed(0)} s per iteration · ${dur(t.eta_s)} left` : t.status)),
    h('div', { class: `prog${t.status === 'running' ? ' is-running' : ''}` }, h('span', {}, `${t.iteration}/${t.total}`),
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(100 * Math.min(1, t.iteration / t.total)).toFixed(1)}%` } }))));
}

// ---------- page: simulations (live list) ----------
function renderRuns(first) {
  if (first) {
    setHead({ title: PAGES.runs.title, sub: PAGES.runs.sub, actions: [
      h('div', { class: 'seg', role: 'group', 'aria-label': 'Show' }, ['all', 'running', 'finished'].map((f) =>
        h('button', { type: 'button', 'aria-pressed': S.runFilter === f ? 'true' : 'false', onclick: (e) => {
          S.runFilter = f; for (const b of e.target.parentNode.children) b.setAttribute('aria-pressed', String(b === e.target)); renderRuns(false);
        } }, f[0].toUpperCase() + f.slice(1)))),
      h('label', { class: 'field' }, h('input', { type: 'search', 'aria-label': 'Filter simulations', placeholder: 'Filter by stage, channel, seed…', value: S.runQuery,
        oninput: (e) => { S.runQuery = e.target.value; renderRuns(false); } }))] });
    $('page').replaceChildren(h('section', { class: 'section', id: 'sec-running' }), h('section', { class: 'section', id: 'sec-done' }));
  }
  const q = S.runQuery.trim().toLowerCase();
  const match = (r) => !q || `${runTitle(r)} ${stackName(r.framework.key)} ${r.name} ${r.root}`.toLowerCase().includes(q);
  const all = runs().filter(match);
  const live = all.filter((r) => r.status !== 'finished');
  const done = all.filter((r) => r.status === 'finished');
  const secR = $('sec-running'), secD = $('sec-done');
  if (!secR || !secD) return;
  secR.hidden = S.runFilter === 'finished'; secD.hidden = S.runFilter === 'running';
  secR.replaceChildren(
    h('div', { class: 'section-head' }, h('h2', {}, 'Running now'), h('p', {}, `${live.length} simulation${live.length === 1 ? '' : 's'} · curves redraw after every batch`)),
    live.length ? h('div', { class: 'run-list' }, live.map((r) => cached(`row:${r.id}`, rowKey(r), () => runRow(r)))) : h('p', { class: 'empty-state' }, q ? 'No running simulation matches.' : 'No simulation is running.'));
  secD.replaceChildren(
    h('div', { class: 'section-head' }, h('h2', {}, 'Finished'), h('p', {}, `${done.length} run${done.length === 1 ? '' : 's'} in the watched window`)),
    done.length ? h('div', { class: 'run-grid' }, done.map((r) => cached(`tile:${r.id}`, rowKey(r), () => runTile(r)))) : h('p', { class: 'empty-state' }, q ? 'No finished run matches.' : 'No finished runs in the window.'));
}
const rowKey = (r) => `${runVersion(r)}|${r.status}|${r.progress?.batch}|${Math.round(r.age_s / 60)}`;
function cached(key, version, make) {
  const hit = S.rowCache.get(key);
  if (hit && hit.v === version) return hit.el;
  const el = make();
  if (hit && !REDUCED_MOTION) el.classList.add('flash');
  S.rowCache.set(key, { v: version, el });
  return el;
}
function runRow(run) {
  const st = run.stages[0];
  const p = run.progress;
  const series = seriesOf(st);
  const ref = referenceEb(st);
  const now = st.methods.map((m) => ({ m, b: cellBer(st.cells.find((c) => c.method === m && c.ebno_db === ref)) }))
    .sort((a, b) => (a.b?.y ?? 1) - (b.b?.y ?? 1));
  const open = p ? Math.round(p.active_points / Math.max(1, st.methods.length)) : null;
  return h('a', { class: 'run-row', href: runHref(run), 'aria-label': `${runTitle(run)}, ${stackName(run.framework.key)}, ${run.status}. Open` },
    h('div', { class: 'run-info' },
      h('div', { class: 't' }, runTitle(run)),
      h('div', { class: 'chips' }, statusChip(run), chip(stackName(run.framework.key), run.framework.key)),
      progressBar(run),
      h('dl', { class: 'facts' },
        h('dt', {}, 'Started'), h('dd', {}, startedAt(run)),
        h('dt', {}, 'Time per batch'), h('dd', {}, p?.seconds_per_batch ? `${p.seconds_per_batch.toFixed(1)} s` : '–'),
        h('dt', {}, 'Time left'), h('dd', {}, p ? dur(p.eta_s) : (run.status === 'stopped' ? `stopped ${dur(run.age_s)} ago` : '–')),
        h('dt', {}, 'Eb/N0 points open'), h('dd', {}, open !== null ? `${open} of ${st.ebno_db.length}` : '–'),
        h('dt', {}, 'Folder'), h('dd', {}, run.root))),
    h('div', { class: 'run-plot' }, logChart(series, { width: 520, height: 210, xs: st.ebno_db }), legendOf(series)),
    h('div', { class: 'run-now' },
      h('div', { class: 'h' }, `BER at ${ref} dB, best first`),
      now.map(({ m, b }) => h('div', { class: 'm' }, h('i', { style: { background: methodColor(st.methods, m) } }), h('span', {}, m),
        h('b', {}, b ? (b.bound ? `< ${sci(b.y)}` : sci(b.y)) : '–')))));
}
function runTile(run) {
  const st = run.stages[0];
  return h('a', { class: 'run-tile', href: runHref(run) },
    h('div', { class: 't' }, runTitle(run)),
    h('div', { class: 'run-info' }, h('div', { class: 'chips' }, statusChip(run), chip(stackName(run.framework.key), run.framework.key), h('span', { class: 'prog' }, `started ${startedAt(run)}`))),
    logChart(seriesOf(st), { width: 420, height: 180, xs: st.ebno_db }));
}

// ---------- page: one run ----------
function renderRun(first) {
  const run = runById(S.route.id);
  if (!run) {
    setHead({ title: 'Simulation not found', crumbs: [{ text: 'Simulations', href: '#/runs' }] });
    $('page').replaceChildren(h('p', { class: 'empty-state' }, 'This run is no longer in the watched window.'));
    return;
  }
  if (first) S.detail = { stage: 0, eb: null, conv: null, convKey: null };
  const st = run.stages[Math.min(S.detail.stage, run.stages.length - 1)];
  if (S.detail.eb === null || !st.ebno_db.includes(S.detail.eb)) S.detail.eb = referenceEb(st);
  setHead({ title: runTitle(run), sub: `${run.framework.label} · folder ${run.root} · started ${startedAt(run)}`,
    crumbs: [{ text: 'Simulations', href: '#/runs' }, { text: runTitle(run) }],
    actions: [statusChip(run), chip(stackName(run.framework.key), run.framework.key),
      h('a', { class: 'btn', href: '#/hall', onclick: () => { S.hall.run = run.id; } }, 'Open in factory hall')] });
  const p = run.progress;
  const series = seriesOf(st);
  const open = p ? Math.round(p.active_points / Math.max(1, st.methods.length)) : null;
  $('page').replaceChildren(...kids(
    h('div', { class: 'kpis' },
      kpi('Progress', p ? `${Math.round(100 * (p.batch - 1) / p.total)} %` : run.status, p ? `batch ${p.batch - 1} of ${p.total}` : `${st.batches_traced} batches traced`),
      kpi('Time left', p ? dur(p.eta_s) : '–', p?.seconds_per_batch ? `${p.seconds_per_batch.toFixed(1)} s per batch` : `last write ${dur(run.age_s)} ago`),
      kpi('Bits simulated', sci(bitsOf(run), 2), `${st.methods.length} methods × ${st.ebno_db.length} Eb/N0 points`),
      kpi('Eb/N0 points open', open !== null ? `${open} of ${st.ebno_db.length}` : '–', 'a point closes once it has enough errors'),
      kpi('Seed', String(run.seed ?? '–'), run.config || '')),
    run.stages.length > 1 ? h('div', { class: 'tabs', role: 'tablist' }, run.stages.map((x, i) =>
      h('button', { type: 'button', role: 'tab', 'aria-selected': i === S.detail.stage ? 'true' : 'false', onclick: () => { S.detail.stage = i; S.detail.convKey = null; renderRun(false); } }, STAGE_LABEL[x.stage] || x.stage))) : null,
    h('div', { class: 'two-col' },
      h('div', { class: 'panel' }, h('h3', {}, 'BER so far', h('small', {}, 'filled ≥ 30 bit errors · hollow fewer · arrow: none yet')),
        logChart(series, { width: 620, height: 300, xs: st.ebno_db }), legendOf(series)),
      h('div', { class: 'panel' }, h('h3', {}, 'Convergence', h('small', {}, 'running BER after each batch')),
        h('div', { class: 'seg', role: 'group', 'aria-label': 'Eb/N0 point' }, st.ebno_db.map((eb) =>
          h('button', { type: 'button', 'aria-pressed': eb === S.detail.eb ? 'true' : 'false', onclick: () => { S.detail.eb = eb; S.detail.convKey = null; S.detail.conv = null; renderRun(false); } }, `${eb}`))),
        h('div', { id: 'conv-box' }, convChart(st)))),
    h('div', { class: 'panel' }, h('h3', {}, 'BER table', h('small', {}, 'bold: resolved (≥ 30 bit errors) · hover a cell for counts')), berTable(st))));
  loadConvergence(run, st);
}
async function loadConvergence(run, st) {
  const key = `${run.id}|${st.dir}|${S.detail.eb}|${st.version}`;
  if (S.detail.convKey === key) return;
  S.detail.convKey = key;
  try {
    const res = await fetch(`/api/convergence?${new URLSearchParams({ run: run.id, stage: st.dir, eb: String(S.detail.eb) })}`);
    if (!res.ok) return;
    S.detail.conv = await res.json();
    $('conv-box')?.replaceChildren(convChart(st));
  } catch (_) { /* next update retries */ }
}
function convChart(st) {
  if (!S.detail.conv) return h('p', { class: 'empty-state' }, 'loading…');
  const series = Object.entries(S.detail.conv.series).map(([m, ys]) => ({
    label: m, color: methodColor(st.methods, m), points: ys.map((y, i) => ({ x: i + 1, y })).filter((p) => p.y && p.y > 0),
  }));
  if (!series.some((se) => se.points.length)) return h('p', { class: 'empty-state' }, `No bit errors yet at ${S.detail.eb} dB.`);
  return logChart(series, { width: 620, height: 270, xlabel: 'batches so far', ylabel: 'running BER' });
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

// ---------- page: factory hall ----------
const HALL_DEFAULT = { room_dimensions: [15, 15, 5], machines: [], bs_position: [0, 0, 4], ut_height: 1 };
function outcomeOf(latest, i) {
  if (!latest || !latest.active?.[i]) return 'idle';
  return (latest.block_errors?.[i] ?? 0) === 0 ? 'delivered' : 'failed';
}
const outcomeColor = (o) => (o === 'delivered' ? cssVar('--ok-line') : o === 'failed' ? cssVar('--bad-line') : cssVar('--muted'));
async function fetchScene(run, st, method, eb) {
  try {
    const res = await fetch(`/api/scene?${new URLSearchParams({ run: run.id, stage: st.dir, eb: String(eb), method: method || '' })}`);
    return res.ok ? await res.json() : null;
  } catch (_) { return null; }
}
function fill(sel, options, value) {
  if (!sel) return;
  const cur = [...sel.options].map((o) => `${o.value}=${o.textContent}`).join('|');
  if (cur !== options.map((o) => `${o[0]}=${o[1]}`).join('|')) sel.replaceChildren(...options.map(([v, t]) => h('option', { value: v }, t)));
  if (value !== null && value !== undefined) sel.value = value;
}
function renderHall(first) {
  if (first) {
    setHead({ title: PAGES.hall.title, sub: PAGES.hall.sub, actions: [
      h('label', { class: 'field' }, 'Run', h('select', { id: 'hall-run', onchange: (e) => { S.hall.run = e.target.value; S.hall.method = null; S.hall.eb = null; renderHall(false); } })),
      h('label', { class: 'field' }, 'Method', h('select', { id: 'hall-method', onchange: (e) => { S.hall.method = e.target.value; renderHall(false); } })),
      h('label', { class: 'field' }, 'Eb/N0', h('select', { id: 'hall-eb', onchange: (e) => { S.hall.eb = Number(e.target.value); renderHall(false); } })),
      h('div', { class: 'seg', role: 'group', 'aria-label': 'View' }, [['both', '2D + 3D'], ['2d', '2D plan'], ['3d', '3D']].map(([v, t]) =>
        h('button', { type: 'button', 'aria-pressed': S.hall.view === v ? 'true' : 'false', onclick: (e) => {
          S.hall.view = v; for (const b of e.target.parentNode.children) b.setAttribute('aria-pressed', String(b === e.target)); drawHall();
        } }, t)))] });
    $('page').replaceChildren(h('p', { class: 'empty-state' }, 'loading…'));
  }
  const rs = runs();
  if (!rs.find((r) => r.id === S.hall.run)) S.hall.run = (rs.find((r) => r.status === 'running') || rs[0])?.id ?? null;
  fill($('hall-run'), rs.map((r) => [r.id, `${r.status === 'running' ? '● ' : ''}${runTitle(r)} · ${stackName(r.framework.key)}`]), S.hall.run);
  const run = runById(S.hall.run);
  if (!run) { $('page').replaceChildren(h('p', { class: 'empty-state' }, 'No traced runs yet.')); return; }
  const st = run.stages[0];
  if (!st.methods.includes(S.hall.method)) S.hall.method = st.methods[0] ?? null;
  if (!st.ebno_db.includes(S.hall.eb)) S.hall.eb = st.ebno_db[Math.floor(st.ebno_db.length / 2)] ?? null;
  fill($('hall-method'), st.methods.map((m) => [m, m]), S.hall.method);
  fill($('hall-eb'), st.ebno_db.map((e) => [String(e), `${e} dB`]), String(S.hall.eb));
  const key = `${run.id}|${st.dir}|${S.hall.method}|${S.hall.eb}|${st.version}`;
  if (key === S.hall.key) return;
  S.hall.key = key;
  fetchScene(run, st, S.hall.method, S.hall.eb).then((sc) => { if (sc && S.route.page === 'hall') { S.hall.scene = sc; drawHall(); } });
}
function drawHall() {
  const sc = S.hall.scene;
  if (!sc || S.route.page !== 'hall') return;
  if (!sc.drop) {
    $('page').replaceChildren(h('p', { class: 'empty-state' }, sc.large_scale_model === 'none'
      ? 'This run has no terminal positions (large_scale_model "none"): nothing to place in the hall.' : 'Waiting for the first batch of this run…'));
    return;
  }
  const caption = `batch ${sc.batch + 1} · ${sc.method} · ${sc.ebno_db} dB`;
  const panes = [];
  if (S.hall.view !== '3d') panes.push(h('div', { class: 'panel' }, h('h3', {}, '2D plan', h('small', {}, caption)), plan2d(sc)));
  if (S.hall.view !== '2d') {
    panes.push(h('div', { class: 'panel' }, h('h3', {}, '3D hall', h('small', {}, caption)), h('div', { id: 'hall3d-box' }, hall3d(sc)),
      h('div', { class: 'rot-ctl' }, 'Drag to turn and tilt.',
        h('button', { type: 'button', class: 'btn', 'aria-pressed': S.hall.spin ? 'true' : 'false', onclick: (e) => { S.hall.spin = !S.hall.spin; e.target.textContent = S.hall.spin ? 'Pause rotation' : 'Rotate'; e.target.setAttribute('aria-pressed', String(S.hall.spin)); if (S.hall.spin) spin(); } }, S.hall.spin ? 'Pause rotation' : 'Rotate'),
        h('button', { type: 'button', class: 'btn', onclick: () => { S.hall.yaw = -0.65; S.hall.pitch = 0.62; redraw3d(); } }, 'Reset view'))));
  }
  $('page').replaceChildren(
    h('div', { class: `hall-grid${panes.length === 1 ? ' single' : ''}` }, panes),
    hallLegend(),
    h('section', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', {}, 'Terminals'), h('p', {}, `batch ${sc.batch + 1}, with totals over the run so far at ${sc.ebno_db} dB`)), utStrip(sc)));
  if (S.hall.view !== '2d' && S.hall.spin) spin();
}
function hallLegend() {
  const dot = (fill, stroke) => s('svg', { width: 12, height: 12, 'aria-hidden': 'true' }, s('circle', { cx: 6, cy: 6, r: 4.5, fill, stroke, 'stroke-width': 1.6 }));
  const line = (c) => s('svg', { width: 18, height: 12, 'aria-hidden': 'true' }, s('line', { x1: 1, y1: 6, x2: 17, y2: 6, stroke: c, 'stroke-width': 3, 'stroke-linecap': 'round' }));
  return h('div', { class: 'legend' },
    h('span', {}, dot(UT_COLORS[0], 'none'), ' terminal, line of sight'),
    h('span', {}, dot(cssVar('--panel'), UT_COLORS[0]), ' terminal, no line of sight'),
    h('span', {}, line(cssVar('--ok-line')), ' scheduled, packet delivered'),
    h('span', {}, line(cssVar('--bad-line')), ' scheduled, packet failed'),
    h('span', {}, dot(cssVar('--muted'), 'none'), ' earlier drops (faint)'),
    h('span', {}, '▲ base station on the ceiling (4 m)'),
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
function plan2d(sc, { size = 560 } = {}) {
  const hall = sc.hall || HALL_DEFAULT;
  const [L, W] = hall.room_dimensions;
  const m = 30, k = size / Math.max(L, W);
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
  for (const d of sc.history || []) d.positions.forEach((p, i) => svg.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 2.2, fill: utColor(i), opacity: 0.18 })));
  const bs = hall.bs_position;
  const d = sc.drop, L0 = sc.latest;
  d.positions.forEach((p, i) => {
    const o = outcomeOf(L0, i);
    if (o === 'idle') return;
    const w = 2 + 4 * (L0.power?.[i] ?? 1);
    svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: cssVar('--panel'), 'stroke-width': w + 4, 'stroke-linecap': 'round' }));
    svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: outcomeColor(o), 'stroke-width': w, 'stroke-linecap': 'round' }, s('title', {}, `UT${i}: ${o}`)));
  });
  svg.append(s('polygon', { points: `${px(bs[0])},${py(bs[1]) - 13} ${px(bs[0]) + 12},${py(bs[1]) + 8} ${px(bs[0]) - 12},${py(bs[1]) + 8}`, fill: cssVar('--ink'), stroke: cssVar('--panel'), 'stroke-width': 2 }));
  svg.append(s('text', { x: px(bs[0]) - 16, y: py(bs[1]) - 6, 'text-anchor': 'end', class: 'ink', style: { fontWeight: 600 } }, 'BS'));
  d.positions.forEach((p, i) => {
    const los = d.los ? d.los[i] : true;
    const g = s('g', { opacity: L0?.active?.[i] ? 1 : 0.55 });
    g.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 9, fill: los ? utColor(i) : cssVar('--panel'), stroke: los ? cssVar('--panel') : utColor(i), 'stroke-width': los ? 2.5 : 3 }));
    g.append(s('text', { x: px(p[0]) + 13, y: py(p[1]) - 9, class: 'ink', style: { fontWeight: 600 } }, `UT${i}`));
    g.append(s('title', {}, `UT${i}: ${los ? 'line of sight' : 'no line of sight'}, ${d.gain_db ? `${d.gain_db[i].toFixed(1)} dB` : ''}, ${outcomeOf(L0, i)}`));
    svg.append(g);
  });
  return svg;
}
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
  const width = 600, height = 480;
  // Fit the room's bounding sphere, so the view keeps its size and centre while it turns.
  const radius = Math.hypot(L / 2, W / 2, H / 2);
  const scale = (Math.min(width, height) / 2 - 14) / radius;
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

// ---------- page: compare runs ----------
function renderCompare(first) {
  const rs = runs();
  const stageNames = [...new Set(rs.flatMap((r) => r.stages.map((x) => x.stage)))];
  if (!stageNames.includes(S.ov.stage)) S.ov.stage = stageNames.includes('estimators') ? 'estimators' : stageNames[0] ?? null;
  // Runs without provenance (no stack or seed recorded) cannot be placed in a comparison; leave them out.
  const withStage = rs.filter((r) => r.framework.key !== 'unknown').map((r) => ({ r, st: r.stages.find((x) => x.stage === S.ov.stage) })).filter((x) => x.st);
  const methods = [...new Set(withStage.flatMap((x) => x.st.methods))];
  if (!methods.includes(S.ov.method)) S.ov.method = methods.includes('ls') ? 'ls' : methods[0] ?? null;
  if (first) {
    setHead({ title: PAGES.compare.title, sub: PAGES.compare.sub, actions: [
      h('label', { class: 'field' }, 'Stage', h('select', { id: 'ov-stage', onchange: (e) => { S.ov.stage = e.target.value; renderCompare(false); } })),
      h('label', { class: 'field' }, 'Method', h('select', { id: 'ov-method', onchange: (e) => { S.ov.method = e.target.value; renderCompare(false); } }))] });
  }
  fill($('ov-stage'), stageNames.map((n) => [n, STAGE_LABEL[n] || n]), S.ov.stage);
  fill($('ov-method'), methods.map((m) => [m, m]), S.ov.method);
  const counters = { tf: 0, torch: 0, unknown: 0 };
  const rows = withStage.filter((x) => x.st.methods.includes(S.ov.method)).map(({ r, st }) => {
    const k = r.framework.key;
    const color = k === 'unknown' ? '#8a8f98' : (k === 'torch' ? TORCH_SHADES : TF_SHADES)[counters[k]++ % 5];
    const se = seriesOf(st).find((x) => x.label === S.ov.method);
    return { r, se: { ...se, color, label: `${stackName(k)} · ${channelName(r)} · ${r.seed !== null && r.seed !== undefined ? `seed ${r.seed}` : startedAt(r)}${r.status === 'running' ? ' · live' : ''}` } };
  });
  const xs = [...new Set(rows.flatMap((x) => x.se.points.map((p) => p.x)))].sort((a, b) => a - b);
  $('page').replaceChildren(...(rows.length ? [
    h('div', { class: 'panel' }, h('h3', {}, `${S.ov.method} · ${STAGE_LABEL[S.ov.stage] || S.ov.stage}`, h('small', {}, `${rows.length} runs · live runs update as they go`)),
      logChart(rows.map((x) => x.se), { width: 1100, height: 340, xs }), legendOf(rows.map((x) => x.se))),
    h('div', { class: 'panel' }, h('h3', {}, 'BER per run', h('small', {}, 'bold: resolved (≥ 30 bit errors)')),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', {}, 'Status'), xs.map((eb) => h('th', {}, `${eb} dB`)))),
        h('tbody', {}, rows.map(({ r, se }) => h('tr', {},
          h('td', {}, h('span', { class: 'swatch', style: { background: se.color, marginRight: '6px' } }), h('a', { href: runHref(r) }, se.label)),
          h('td', {}, statusChip(r)),
          xs.map((eb) => { const p = se.points.find((q) => q.x === eb); return h('td', {}, !p ? '–' : p.bound ? `< ${sci(p.y)}` : p.resolved ? h('strong', {}, sci(p.y)) : sci(p.y)); }))))))),
  ] : [h('p', { class: 'empty-state' }, 'No run has this method yet.')]));
}

// ---------- page: training ----------
function bars(obj, fmt, cap = null) {
  if (!obj) return null;
  const entries = Object.entries(obj).map(([k, v]) => [Number(k), v]).sort((a, b) => a[0] - b[0]);
  const mx = Math.max(...entries.map((e) => e[1]), 1e-12);
  return h('div', { class: 'bars' }, entries.flatMap(([k, v]) => [h('span', {}, `${k} dB`),
    h('div', { class: `b${cap !== null && v >= cap ? ' cap' : ''}`, style: { width: `${(100 * v / mx).toFixed(1)}%` } }), h('span', {}, fmt(v))]));
}
function renderTraining() {
  const list = training();
  $('page').replaceChildren(...(list.length ? list.map((t) => h('section', { class: 'section' },
    h('div', { class: 'section-head' }, h('h2', {}, `PPO · ${t.name.replace(/^ppo_/, '')} policy`), t.status === 'running' ? chip('running', 'running live') : chip(t.status, t.status), h('p', {}, `folder: ${t.root}`)),
    h('div', { class: 'kpis' },
      h('div', { class: 'kpi' }, h('div', { class: 'k' }, 'Progress'), h('div', { class: 'v' }, `${t.iteration}/${t.total}`),
        h('div', { class: `prog${t.status === 'running' ? ' is-running' : ''}` }, h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(100 * Math.min(1, t.iteration / t.total)).toFixed(1)}%` } })))),
      kpi('Time left', dur(t.eta_s), t.seconds_per_iteration ? `${t.seconds_per_iteration.toFixed(0)} s per iteration` : ''),
      kpi('Latest reward', t.reward.length ? t.reward[t.reward.length - 1].toFixed(3) : '–', 'mean step reward'),
      kpi('Latest rollout BER', t.ber.length ? sci(t.ber[t.ber.length - 1]) : '–', 'over the iteration\'s rollout')),
    h('div', { class: 'two-col' },
      h('div', { class: 'panel' }, h('h3', {}, 'Mean step reward'), lineChart(t.reward, { ylabel: 'reward' })),
      h('div', { class: 'panel' }, h('h3', {}, 'Rollout BER'), lineChart(t.ber, { ylabel: 'BER', color: '#cc6677', log: true }))),
    h('div', { class: 'two-col' }, ...kids(
      t.bler_by_ebno ? h('div', { class: 'panel' }, h('h3', {}, 'Latest rollout BLER per Eb/N0', h('small', {}, 'target 1 %')), bars(t.bler_by_ebno, (v) => sci(v))) : null,
      t.lagrange ? h('div', { class: 'panel' }, h('h3', {}, 'Lagrange multipliers', h('small', {}, 'red: at the cap of 10, so the 1 % target is out of reach there')), bars(t.lagrange, (v) => v.toFixed(2), 10)) : null))))
    : [h('p', { class: 'empty-state' }, 'No training logs in the watched folders.')]));
}

// ---------- connection ----------
function updateConn() {
  if (!$('conn').classList.contains('on')) return;
  const polled = S.snap?.polled_at ? S.snap.polled_at * 1000 : S.lastMsg;
  const age = (Date.now() - polled) / 1000;
  $('conn-text').textContent = `Live · checked ${age < 4 ? 'just now' : `${Math.round(age)} s ago`}`;
}
function onSnapshot(snap) {
  const first = !S.snap;
  S.snap = snap;
  S.lastMsg = Date.now();
  if (first) navigate(); else PAGES[S.route.page].render(false);
  updateBadges();
  updateConn();
}
function connect() {
  const conn = $('conn');
  if (!('EventSource' in window)) { poll(); return; }
  const es = new EventSource('/api/stream');
  es.onopen = () => { conn.className = 'conn on'; updateConn(); };
  es.onmessage = (e) => { conn.className = 'conn on'; onSnapshot(JSON.parse(e.data)); };
  es.addEventListener('ping', (e) => { conn.className = 'conn on'; if (S.snap) S.snap.polled_at = JSON.parse(e.data).polled_at; updateConn(); });
  es.onerror = () => { conn.className = 'conn off'; $('conn-text').textContent = 'reconnecting…'; };
}
async function poll() {
  try { const r = await fetch('/api/runs'); $('conn').className = 'conn on'; onSnapshot(await r.json()); } catch (_) { $('conn').className = 'conn off'; }
  setTimeout(poll, 3000);
}
setInterval(updateConn, 1000);

function wire() {
  window.addEventListener('hashchange', () => { if (S.snap) navigate(); });
  // "Skip to content" moves focus without touching the hash, which the router owns.
  document.querySelector('.skip').addEventListener('click', (e) => { e.preventDefault(); $('page').focus(); });
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    $('theme-toggle').textContent = dark ? 'Light theme' : 'Dark theme';
    try { localStorage.setItem('f6g-theme', dark ? 'dark' : 'light'); } catch (_) { /* storage may be blocked */ }
    if (S.snap) redrawForTheme();
  });
  try {
    const saved = localStorage.getItem('f6g-theme');
    const theme = saved || (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : null);
    if (theme) { document.documentElement.dataset.theme = theme; $('theme-toggle').textContent = theme === 'dark' ? 'Light theme' : 'Dark theme'; }
  } catch (_) { /* storage may be blocked */ }
  const page = parseRoute();
  setHead({ title: PAGES[page.page].title, sub: 'Connecting to the monitor…' });
}
wire();
connect();
