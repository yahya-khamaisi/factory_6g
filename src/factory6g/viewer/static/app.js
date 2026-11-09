'use strict';

// Factory6G run viewer. Reads data.json written by `python -m factory6g.viewer export`.
// Everything is computed in the browser from the per-batch trace.

const UT_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#6b6b6b', '#882255'];
const METHOD_COLORS = ['#233369', '#cc6677', '#117733', '#aa4499', '#3f8fbf', '#999933', '#882255', '#44aa99'];
const MIN_RESOLVED_BIT_ERRORS = 30; // matches factory6g.sim.stages.common
const VIEWS = { floor: 'Floor', timeline: 'Timeline', compare: 'Compare', run: 'Run and checks' };

const S = { data: null, stage: 0, method: null, compare: null, eb: 0, batch: 0, ut: 0, view: 'floor', playing: false, speed: 1, timer: null };
const $ = (id) => document.getElementById(id);

// ---------- DOM helpers (no innerHTML with data) ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
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
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function sci(x, digits = 1) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= -2 && e <= 2) return x.toFixed(Math.max(0, 3 - Math.max(e, 0)));
  const m = x / 10 ** e;
  return `${m.toFixed(digits)}×10${String(e).split('').map((c) => SUP[c]).join('')}`;
}
const tick = (e) => (e === 0 ? '1' : `10${String(e).split('').map((c) => SUP[c]).join('')}`);
const pct = (x) => (Number.isFinite(x) ? `${(100 * x).toFixed(0)} %` : '–');
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------- data access ----------
const stage = () => S.data.stages[S.stage];
const methods = () => stage().meta.methods;
const ebnos = () => stage().meta.ebno_db_range;
const curEb = () => ebnos()[S.eb];
const numBatches = () => stage().num_batches;
const methodColor = (m) => METHOD_COLORS[methods().indexOf(m) % METHOD_COLORS.length];
const utColor = (i) => UT_COLORS[i % UT_COLORS.length];

function indexStage(st) {
  if (st._idx) return;
  st._idx = {};
  for (const r of st.records) {
    const byEb = (st._idx[r.method] ??= {});
    (byEb[r.ebno_db] ??= [])[r.batch] = r;
  }
}
const rec = (m, eb, b) => stage()._idx[m]?.[eb]?.[b];
const ctx = (b, eb) => stage().contexts[`${b}|${Number(eb)}`];

function sumActive(r, key) {
  if (!r) return 0;
  return r[key].reduce((acc, v) => acc + (v ?? 0), 0);
}
function batchSeries(m, eb, upto) {
  const out = [];
  for (let b = 0; b <= upto; b++) {
    const r = rec(m, eb, b);
    if (!r) continue;
    const be = sumActive(r, 'bit_errors'), bits = sumActive(r, 'bits');
    const ble = sumActive(r, 'block_errors'), blocks = sumActive(r, 'blocks');
    out.push({ b, be, bits, ble, blocks, ber: bits ? be / bits : NaN, bler: blocks ? ble / blocks : NaN });
  }
  return out;
}
function meanCi(values) {
  const v = values.filter(Number.isFinite);
  const n = v.length;
  if (n === 0) return { mean: NaN, lo: NaN, hi: NaN, n };
  const mean = v.reduce((a, x) => a + x, 0) / n;
  if (n < 2) return { mean, lo: NaN, hi: NaN, n };
  const sd = Math.sqrt(v.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1));
  const half = 1.96 * sd / Math.sqrt(n);
  return { mean, lo: mean - half, hi: mean + half, n };
}
function pointStats(m, eb, upto) {
  const ser = batchSeries(m, eb, upto);
  const be = ser.reduce((a, x) => a + x.be, 0), bits = ser.reduce((a, x) => a + x.bits, 0);
  const ble = ser.reduce((a, x) => a + x.ble, 0), blocks = ser.reduce((a, x) => a + x.blocks, 0);
  const ci = meanCi(ser.map((x) => x.ber));
  const zeroBound = bits ? -Math.log(0.05) / bits : NaN;
  return {
    batches: ser.length, be, bits, ble, blocks,
    ber: bits ? be / bits : NaN, bler: blocks ? ble / blocks : NaN,
    ciLo: Math.max(0, ci.lo), ciHi: ci.hi,
    resolved: be >= MIN_RESOLVED_BIT_ERRORS, zeroBound, series: ser,
  };
}
function paired(m, c, eb, upto) {
  const a = batchSeries(m, eb, upto), bByBatch = new Map(batchSeries(c, eb, upto).map((x) => [x.b, x]));
  const d = [], base = [];
  for (const x of a) {
    const y = bByBatch.get(x.b);
    if (!y) continue;
    d.push(x.bler - y.bler);
    base.push(y.bler);
  }
  const ci = meanCi(d);
  const baseMean = meanCi(base).mean;
  let verdict = 'none';
  if (ci.n >= 2 && Number.isFinite(ci.lo)) verdict = ci.hi < 0 ? 'better' : ci.lo > 0 ? 'worse' : 'tie';
  return { ...ci, baseMean, verdict };
}
function utCumulative(m, eb, ut, upto) {
  let sched = 0, n = 0, ble = 0, blocks = 0, be = 0, bits = 0;
  for (let b = 0; b <= upto; b++) {
    const r = rec(m, eb, b);
    if (!r) continue;
    n++;
    if (r.active[ut]) {
      sched++;
      ble += r.block_errors[ut] ?? 0; blocks += r.blocks[ut] ?? 0;
      be += r.bit_errors[ut] ?? 0; bits += r.bits[ut] ?? 0;
    }
  }
  return { share: n ? sched / n : NaN, sched, n, ble, blocks, be, bits };
}

// ---------- shared pieces ----------
function verdictChip(p) {
  if (p.verdict === 'better') return h('span', { class: 'chip ok', text: '✓ fewer errors' });
  if (p.verdict === 'worse') return h('span', { class: 'chip bad', text: '✕ more errors' });
  if (p.verdict === 'tie') return h('span', { class: 'chip muted', text: '= no clear difference' });
  return h('span', { class: 'chip muted', text: 'not enough batches' });
}
function card(title, note, ...kids) {
  return h('section', { class: 'card' }, h('h2', {}, title, note ? h('small', { text: note }) : null), ...kids);
}
function tile(k, v, sub, subClass) {
  return h('div', { class: 'tile' }, h('div', { class: 'k', text: k }), h('div', { class: 'v' }, v), h('div', { class: `s ${subClass || ''}`, style: subClass ? { color: `var(--${subClass})` } : null }, sub));
}

// ---------- Floor ----------
function hallSvg() {
  const hall = S.data.hall;
  const [L, W] = hall ? hall.room_dimensions : [15, 15];
  const size = 600, m = 26, k = size / Math.max(L, W);
  const px = (x) => m + (x + L / 2) * k, py = (y) => m + (W / 2 - y) * k;
  const svg = s('svg', { viewBox: `0 0 ${size + 2 * m} ${size + 2 * m}`, class: 'hall', role: 'img', 'aria-label': 'Hall plan with terminals and scheduled links for the selected batch' });
  const rm = S.data.radio_map;
  if (rm) {
    const vals = rm.gain_db.flat().filter((v) => v !== null).sort((a, b) => a - b);
    const lo = vals[Math.floor(vals.length * 0.05)], hi = vals[Math.floor(vals.length * 0.95)];
    const c0 = cssVar('--map-lo'), c1 = cssVar('--map-mid'), c2 = cssVar('--map-hi');
    rm.gain_db.forEach((row, r) => row.forEach((v, c) => {
      if (v === null) return;
      const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      const fill = t < 0.5 ? mix(c0, c1, t * 2) : mix(c1, c2, (t - 0.5) * 2);
      const x = rm.x0 + c * rm.cell_size_m, y = rm.y0 + r * rm.cell_size_m;
      svg.append(s('rect', { x: px(x), y: py(y + rm.cell_size_m), width: rm.cell_size_m * k + 0.4, height: rm.cell_size_m * k + 0.4, fill }));
    }));
    svg.dataset.rmLo = lo; svg.dataset.rmHi = hi;
  }
  svg.append(s('rect', { x: m, y: m, width: L * k, height: W * k, fill: 'none', stroke: cssVar('--ink'), 'stroke-width': 1.4 }));
  for (const [x, label] of [[-L / 2, `${-L / 2}`], [0, '0'], [L / 2, `${L / 2} m`]]) svg.append(s('text', { x: px(x), y: m + W * k + 16, 'text-anchor': 'middle', text: label }));
  if (hall) {
    hall.machines.forEach((mc, i) => {
      svg.append(s('rect', { x: px(mc.x - mc.sx / 2), y: py(mc.y + mc.sy / 2), width: mc.sx * k, height: mc.sy * k, rx: 3, fill: cssVar('--machine'), stroke: cssVar('--machine-line') }));
      svg.append(s('text', { x: px(mc.x), y: py(mc.y) + 3, 'text-anchor': 'middle', text: `M${i + 1}` }));
    });
  }
  const bs = hall ? hall.bs_position : [0, 0, 4];
  const r = rec(S.method, curEb(), S.batch), c = ctx(S.batch, curEb());
  const nUt = stage().meta.num_ut;
  if (c && r) {
    c.positions.forEach((p, i) => {
      if (!r.active[i]) return;
      const w = 2 + 5 * (r.power[i] ?? 1);
      svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: cssVar('--panel'), 'stroke-width': w + 4, 'stroke-linecap': 'round' }));
      svg.append(s('line', { x1: px(bs[0]), y1: py(bs[1]), x2: px(p[0]), y2: py(p[1]), stroke: utColor(i), 'stroke-width': w, 'stroke-linecap': 'round' }));
    });
  }
  svg.append(s('polygon', { points: `${px(bs[0])},${py(bs[1]) - 14} ${px(bs[0]) + 13},${py(bs[1]) + 9} ${px(bs[0]) - 13},${py(bs[1]) + 9}`, fill: cssVar('--ink'), stroke: cssVar('--panel'), 'stroke-width': 2 }));
  svg.append(s('text', { x: px(bs[0]), y: py(bs[1]) + 24, 'text-anchor': 'middle', class: 'ink', text: 'BS' }));
  if (c) {
    c.positions.forEach((p, i) => {
      const los = c.los ? c.los[i] : true;
      const active = r ? r.active[i] : true;
      const g = s('g', { opacity: active || i === S.ut ? 1 : 0.45, style: { cursor: 'pointer' }, onclick: () => { S.ut = i; render(); } });
      if (i === S.ut) g.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 16, fill: 'none', stroke: cssVar('--ink'), 'stroke-width': 2 }));
      g.append(s('circle', { cx: px(p[0]), cy: py(p[1]), r: 10, fill: los ? utColor(i) : cssVar('--panel'), stroke: los ? cssVar('--panel') : utColor(i), 'stroke-width': los ? 2.5 : 3.5 }));
      g.append(s('text', { x: px(p[0]) + 15, y: py(p[1]) - 10, class: 'ink', style: { fontWeight: 600, fontSize: '12px' }, text: `UT${i}` }));
      svg.append(g);
    });
  } else {
    svg.append(s('text', { x: m + 10, y: m + 20, text: `No terminal positions in this trace (large_scale_model ${stage().meta.large_scale_model}).` }));
  }
  return { svg, nUt };
}
function mix(a, b, t) {
  const pa = hex(a), pb = hex(b);
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(',')})`;
}
function hex(c) {
  const m = c.replace('#', '');
  const v = m.length === 3 ? m.split('').map((x) => x + x).join('') : m;
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
}

function renderFloor() {
  const eb = curEb();
  const { svg, nUt } = hallSvg();
  const r = rec(S.method, eb, S.batch), c = ctx(S.batch, eb);
  const st = pointStats(S.method, eb, S.batch);
  const pr = S.compare ? paired(S.method, S.compare, eb, S.batch) : null;
  const batchBler = r ? sumActive(r, 'block_errors') / Math.max(1, sumActive(r, 'blocks')) : NaN;
  const rm = S.data.radio_map;
  const legend = h('div', { class: 'legend' },
    h('span', {}, h('span', { class: 'swatch', style: { background: cssVar('--ink') } }), 'LOS'),
    h('span', {}, h('span', { class: 'swatch', style: { border: `2px solid ${cssVar('--ink')}` } }), 'NLOS'),
    h('span', {}, 'faded = not scheduled · link width = power'),
    h('span', { text: rm ? `map: ${rm.source}, path gain at ${rm.height_m} m, ${Number(svg.dataset.rmLo).toFixed(0)} … ${Number(svg.dataset.rmHi).toFixed(0)} dB` : 'no radio map exported (export with --radio-map)' }));

  const tiles = h('div', { class: 'tiles' },
    tile('BLER, this batch', sci(batchBler), `${r ? sumActive(r, 'block_errors') : 0} of ${r ? sumActive(r, 'blocks') : 0} codewords`),
    tile(`Running BER to batch ${S.batch + 1}`, st.be ? sci(st.ber) : `≤ ${sci(st.zeroBound)}`, st.be && Number.isFinite(st.ciHi) ? `batch-level 95 %: ${sci(st.ciLo)} – ${sci(st.ciHi)}` : 'needs 2+ batches'),
    pr ? tile(`ΔBLER vs ${S.compare}`, Number.isFinite(pr.mean) ? sci(pr.mean) : '–', Number.isFinite(pr.lo) ? `paired 95 %: ${sci(pr.lo)} … ${sci(pr.hi)}` : 'needs 2+ batches', pr.verdict === 'better' ? 'ok' : pr.verdict === 'worse' ? 'bad' : null) : null,
    tile('Evidence', st.resolved ? '✓ resolved' : '≈ upper bound', `${st.be} bit errors in ${st.batches} batches`, st.resolved ? 'ok' : 'warn'));

  // terminals table
  const gains = Object.values(stage().contexts).flatMap((x) => x.gain_db || []).filter(Number.isFinite);
  const gMin = Math.min(...gains, -10), gMax = Math.max(...gains, 5), zero = (0 - gMin) / (gMax - gMin);
  const rows = [];
  for (let i = 0; i < nUt; i++) {
    const cum = utCumulative(S.method, eb, i, S.batch);
    const g = c?.gain_db?.[i];
    const gx = Number.isFinite(g) ? (g - gMin) / (gMax - gMin) : zero;
    rows.push(h('tr', { class: `clickable ${i === S.ut ? 'selected' : ''}`, onclick: () => { S.ut = i; render(); } },
      h('td', {}, h('span', { class: 'swatch', style: { background: utColor(i), marginRight: '6px' } }), `UT${i}`),
      h('td', {}, c?.los ? h('span', { class: `chip ${c.los[i] ? 'ok' : 'warn'}`, text: c.los[i] ? 'LOS' : 'NLOS' }) : '–'),
      h('td', {}, h('span', { class: 'bar' }, h('span', { style: { left: `${Math.min(gx, zero) * 100}%`, width: `${Math.abs(gx - zero) * 100}%`, background: utColor(i) } })), ' ', Number.isFinite(g) ? `${g.toFixed(1)} dB` : '–'),
      h('td', { text: r ? (r.active[i] ? (r.power[i] ?? 1).toFixed(2) : '–') : '–' }),
      h('td', { text: r && r.active[i] ? `${r.block_errors[i]} / ${r.blocks[i]}` : '–' }),
      h('td', { text: pct(cum.share) })));
  }
  const table = h('table', {}, h('thead', {}, h('tr', {}, ['Terminal', 'Link', 'Gain re hall average', 'Power', 'Block errors', 'Scheduled'].map((t) => h('th', { text: t })))), h('tbody', {}, rows));

  // inspector
  const i = Math.min(S.ut, nUt - 1);
  const cum = utCumulative(S.method, eb, i, S.batch);
  const p = c?.positions?.[i];
  const kv = (k, v) => [h('dt', { text: k }), h('dd', { text: v })];
  const inspector = card(`UT${i}`, 'selected terminal',
    h('dl', { class: 'kv' },
      kv('Position', p ? `(${p[0].toFixed(1)}, ${p[1].toFixed(1)}, ${p[2].toFixed(1)}) m` : '–'),
      kv('Link', c?.los ? (c.los[i] ? 'LOS' : 'NLOS') : '–'),
      kv('Large-scale gain', Number.isFinite(c?.gain_db?.[i]) ? `${c.gain_db[i].toFixed(2)} dB` : '–'),
      kv('This batch', r ? (r.active[i] ? `scheduled, power ${(r.power[i] ?? 1).toFixed(2)}` : 'not scheduled') : '–'),
      kv('Bit errors this batch', r && r.active[i] ? `${r.bit_errors[i]} of ${r.bits[i]}` : '–'),
      kv(`Scheduled to batch ${S.batch + 1}`, `${cum.sched} of ${cum.n} batches`),
      kv('BLER while scheduled', cum.blocks ? sci(cum.ble / cum.blocks) : '–')),
    h('p', { class: 'note', text: 'Positions and gains are the hall draw shared by every method in this batch.' }));

  return h('div', { class: 'col' }, tiles,
    h('div', { class: 'row' },
      h('div', { class: 'col', style: { flex: '1 1 560px', maxWidth: '680px' } }, card('Hall', `batch ${S.batch + 1} · ${eb} dB · ${S.method}`, svg, legend)),
      h('div', { class: 'col', style: { flex: '1 1 420px' } }, card('Terminals', 'click a row or a marker', table), inspector)));
}

// ---------- Timeline ----------
function renderTimeline() {
  const eb = curEb(), nB = numBatches(), nUt = stage().meta.num_ut;
  const grid = h('div', { class: 'heat', style: { gridTemplateColumns: `52px repeat(${nB}, minmax(2px, 1fr))` } });
  for (let i = 0; i < nUt; i++) {
    grid.append(h('div', { style: { fontSize: '11.5px', alignSelf: 'center' } }, h('span', { class: 'swatch', style: { background: utColor(i), marginRight: '5px' } }), `UT${i}`));
    for (let b = 0; b < nB; b++) {
      const r = rec(S.method, eb, b);
      const on = r && r.active[i];
      const err = on && (r.block_errors[i] ?? 0) > 0;
      grid.append(h('button', {
        type: 'button', class: 'heat-cell', title: `batch ${b + 1}: ${on ? `scheduled, power ${(r.power[i] ?? 1).toFixed(2)}${err ? `, ${r.block_errors[i]} block errors` : ''}` : 'not scheduled'}`,
        'aria-label': `UT${i} batch ${b + 1}`,
        style: { background: on ? utColor(i) : cssVar('--soft'), opacity: on ? 0.35 + 0.65 * (r.power[i] ?? 1) : 1, boxShadow: `${err ? `inset 0 -4px 0 ${cssVar('--bad')}` : ''}${b === S.batch ? `${err ? ',' : ''} inset 0 0 0 2px ${cssVar('--ink')}` : ''}` || 'none' },
        onclick: () => { S.batch = b; render(); },
      }));
    }
  }
  const heat = card('Who got the uplink', `${S.method} · ${eb} dB · red underline = block error`, grid,
    h('p', { class: 'note', text: 'Shade follows allocated power. Click a cell to jump to that batch.' }));

  // convergence chart
  const W = 1000, H = 280, ml = 56, mr = 14, mt = 12, mb = 30;
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, style: { width: '100%', height: 'auto' }, role: 'img', 'aria-label': 'Running BER per method against batch with batch-level 95 % bands' });
  const all = methods().map((m) => ({ m, ser: batchSeries(m, eb, nB - 1) }));
  const cum = all.map(({ m, ser }) => {
    let be = 0, bits = 0;
    const pts = ser.map((x, j) => { be += x.be; bits += x.bits; const ci = meanCi(ser.slice(0, j + 1).map((y) => y.ber)); return { b: x.b, v: bits ? be / bits : NaN, lo: ci.lo, hi: ci.hi, be }; });
    return { m, pts };
  });
  const pos = cum.flatMap((c) => c.pts.flatMap((p) => [p.v, p.hi, p.lo])).filter((v) => Number.isFinite(v) && v > 0);
  if (!pos.length) {
    svg.append(s('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', text: 'No bit errors yet at this Eb/N0: every method sits below the Monte Carlo floor.' }));
  } else {
    const yMin = Math.floor(Math.log10(Math.min(...pos))), yMax = Math.ceil(Math.log10(Math.max(...pos)));
    const X = (b) => ml + (b / Math.max(1, nB - 1)) * (W - ml - mr);
    const Y = (v) => mt + (yMax - Math.log10(Math.max(v, 10 ** yMin))) / Math.max(1, yMax - yMin) * (H - mt - mb);
    for (let e = yMin; e <= yMax; e++) {
      svg.append(s('line', { x1: ml, x2: W - mr, y1: Y(10 ** e), y2: Y(10 ** e), stroke: cssVar('--soft') }));
      svg.append(s('text', { x: ml - 6, y: Y(10 ** e) + 3, 'text-anchor': 'end', text: tick(e) }));
    }
    svg.append(s('text', { x: (W + ml) / 2, y: H - 6, 'text-anchor': 'middle', text: 'batch' }));
    cum.forEach(({ m, pts }) => {
      const col = methodColor(m);
      const band = pts.filter((p) => Number.isFinite(p.lo) && p.hi > 0);
      if (band.length > 1) svg.append(s('polygon', { points: [...band.map((p) => `${X(p.b)},${Y(p.hi)}`), ...band.slice().reverse().map((p) => `${X(p.b)},${Y(Math.max(p.lo, 10 ** yMin))}`)].join(' '), fill: col, opacity: 0.07 }));
      const line = pts.filter((p) => p.be > 0);
      if (line.length) svg.append(s('polyline', { points: line.map((p) => `${X(p.b)},${Y(p.v)}`).join(' '), fill: 'none', stroke: col, 'stroke-width': 2 }));
    });
    svg.append(s('line', { x1: X(S.batch), x2: X(S.batch), y1: mt, y2: H - mb, stroke: cssVar('--ink'), 'stroke-dasharray': '3 3' }));
  }
  const legend = h('div', { class: 'legend' }, cum.map(({ m, pts }) => h('span', {}, h('span', { class: 'swatch', style: { background: methodColor(m) } }), pts.some((p) => p.be > 0) ? m : `${m} (no bit errors at this Eb/N0)`)));
  const conv = card('Running BER as the run converges', `${eb} dB · band = batch-level 95 %`, svg, legend,
    h('p', { class: 'note', text: 'The band comes from the spread of per-batch BER, so it reflects the hall draw changing every batch, unlike a bit-level interval.' }));
  return h('div', { class: 'col' }, heat, conv);
}

// ---------- Compare ----------
function renderCompare() {
  const eb = curEb();
  const rows = methods().map((m) => {
    const st = pointStats(m, eb, S.batch);
    const pr = S.compare && m !== S.compare ? paired(m, S.compare, eb, S.batch) : null;
    return h('tr', { class: `clickable ${m === S.method ? 'selected' : ''}`, onclick: () => { S.method = m; syncControls(); render(); } },
      h('td', {}, h('span', { class: 'swatch', style: { background: methodColor(m), marginRight: '6px' } }), m),
      h('td', { text: st.resolved ? sci(st.ber) : `≤ ${sci(Math.max(st.ber, st.be ? 0 : st.zeroBound))}` }),
      h('td', { text: st.be > 0 && Number.isFinite(st.ciHi) ? `${sci(st.ciLo)} – ${sci(st.ciHi)}` : '–' }),
      h('td', { text: sci(st.bler) }),
      h('td', { text: `${st.ble} / ${st.blocks}` }),
      h('td', { text: String(st.batches) }),
      h('td', { text: pr && Number.isFinite(pr.mean) ? `${sci(pr.mean)} [${sci(pr.lo)}, ${sci(pr.hi)}]` : m === S.compare ? 'baseline' : '–' }),
      h('td', {}, pr ? verdictChip(pr) : m === S.compare ? h('span', { class: 'chip muted', text: 'baseline' }) : null),
      h('td', {}, st.resolved ? h('span', { class: 'chip ok', text: '✓ resolved' }) : h('span', { class: 'chip warn', text: '≈ upper bound' })));
  });
  const table = card('Methods at this Eb/N0', `${eb} dB · batches 1–${S.batch + 1} · ΔBLER paired against ${S.compare || '–'}`,
    h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Method', 'BER', 'Batch-level 95 %', 'BLER', 'Block errors', 'Batches', 'ΔBLER, paired 95 %', 'vs baseline', 'Evidence'].map((t) => h('th', { text: t })))), h('tbody', {}, rows))),
    h('p', { class: 'note', text: 'Every method saw the same channel and noise in each batch, so the difference is tested batch by batch (paired), which is tighter than comparing two separate intervals.' }));

  // BER vs Eb/N0
  const W = 1000, H = 300, ml = 56, mr = 14, mt = 12, mb = 32;
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, style: { width: '100%', height: 'auto' }, role: 'img', 'aria-label': 'BER against Eb/N0 per method with batch-level intervals; hollow markers are upper bounds' });
  const curves = methods().map((m) => ({ m, pts: ebnos().map((e) => ({ e, st: pointStats(m, e, S.batch) })) }));
  const vals = curves.flatMap((c) => c.pts.flatMap((p) => [p.st.ber, p.st.ciHi, p.st.zeroBound])).filter((v) => Number.isFinite(v) && v > 0);
  if (vals.length) {
    const yMin = Math.floor(Math.log10(Math.min(...vals))), yMax = Math.ceil(Math.log10(Math.max(...vals)));
    const e0 = ebnos()[0], e1 = ebnos()[ebnos().length - 1];
    const X = (e) => ml + (e - e0) / Math.max(1e-9, e1 - e0) * (W - ml - mr);
    const Y = (v) => mt + (yMax - Math.log10(Math.max(v, 10 ** yMin))) / Math.max(1, yMax - yMin) * (H - mt - mb);
    for (let e = yMin; e <= yMax; e++) {
      svg.append(s('line', { x1: ml, x2: W - mr, y1: Y(10 ** e), y2: Y(10 ** e), stroke: cssVar('--soft') }));
      svg.append(s('text', { x: ml - 6, y: Y(10 ** e) + 3, 'text-anchor': 'end', text: tick(e) }));
    }
    ebnos().forEach((e) => svg.append(s('text', { x: X(e), y: H - 14, 'text-anchor': 'middle', text: `${e}` })));
    svg.append(s('text', { x: (W + ml) / 2, y: H - 2, 'text-anchor': 'middle', text: 'Eb/N0 (dB)' }));
    svg.append(s('line', { x1: X(curEb()), x2: X(curEb()), y1: mt, y2: H - mb, stroke: cssVar('--ink'), 'stroke-dasharray': '3 3' }));
    curves.forEach(({ m, pts }) => {
      const col = methodColor(m);
      const res = pts.filter((p) => p.st.resolved);
      if (res.length > 1) svg.append(s('polyline', { points: res.map((p) => `${X(p.e)},${Y(p.st.ber)}`).join(' '), fill: 'none', stroke: col, 'stroke-width': 2 }));
      pts.forEach((p) => {
        if (p.st.resolved) {
          if (Number.isFinite(p.st.ciHi)) svg.append(s('line', { x1: X(p.e), x2: X(p.e), y1: Y(p.st.ciHi), y2: Y(Math.max(p.st.ciLo, 10 ** yMin)), stroke: col, 'stroke-width': 1.4 }));
          svg.append(s('circle', { cx: X(p.e), cy: Y(p.st.ber), r: 3.5, fill: col }));
        } else {
          const ub = p.st.be ? p.st.ber : p.st.zeroBound;
          svg.append(s('circle', { cx: X(p.e), cy: Y(ub), r: 4, fill: cssVar('--panel'), stroke: col, 'stroke-width': 1.6 }));
          svg.append(s('line', { x1: X(p.e), x2: X(p.e), y1: Y(ub) + 5, y2: Y(ub) + 13, stroke: col, 'stroke-width': 1.4 }));
        }
      });
    });
  }
  const legend = h('div', { class: 'legend' }, methods().map((m) => h('span', {}, h('span', { class: 'swatch', style: { background: methodColor(m) } }), m)), h('span', { text: 'hollow ↓ = fewer than 30 bit errors, upper bound only' }));
  return h('div', { class: 'col' }, table, card('BER against Eb/N0', `batches 1–${S.batch + 1} · whiskers = batch-level 95 %`, svg, legend));
}

// ---------- Run and checks ----------
function renderRun() {
  const d = S.data;
  const checks = [];
  const add = (ok, label, note) => checks.push(h('li', { style: { display: 'flex', gap: '8px', alignItems: 'baseline', margin: '6px 0' } },
    h('span', { class: `chip ${ok === true ? 'ok' : ok === false ? 'warn' : 'muted'}`, text: ok === true ? '✓' : ok === false ? '!' : '–' }), h('span', {}, h('strong', { text: label }), note ? h('span', { class: 'note', text: ` ${note}` }) : null)));
  for (const st of d.stages) {
    indexStage(st);
    let match = null;
    if (st.results) {
      match = st.meta.methods.every((m) => st.meta.ebno_db_range.every((e, j) => {
        const total = (st._idx[m]?.[e] || []).reduce((a, r) => a + (r ? r.bit_errors.reduce((x, v) => x + (v ?? 0), 0) : 0), 0);
        return Math.abs(total - (st.results[m]?.bit_errors?.[j] ?? NaN)) < 0.5;
      }));
    }
    add(match, `${st.id}: trace matches stage results`, match === null ? 'no stage_results_v2.json next to the trace' : match ? 'per-terminal bit errors sum to the reported totals at every point' : 'totals differ: the trace and the results come from different runs');
  }
  const st0 = d.stages[0];
  add(st0.num_batches >= 100, `${st0.num_batches} batches per point`, st0.num_batches >= 100 ? '' : 'thesis runs use at least 100; intervals here are wide');
  add(false, `Seed ${st0.meta.seed}, one evaluation seed`, 'Phase 1 adds five seeds; until then intervals cover batches, not seeds');
  add(d.hall ? true : false, 'Hall geometry', d.hall ? `from ${d.hall.source}` : 'manifest not found; terminals drawn without machines');
  add(d.radio_map ? true : null, 'Radio map', d.radio_map ? `${d.radio_map.source}, depth ${d.radio_map.max_depth}, ${d.radio_map.cell_size_m} m cells` : 'not exported (add --radio-map)');

  const rows = d.stages.map((st, i) => h('tr', { class: `clickable ${i === S.stage ? 'selected' : ''}`, onclick: () => { selectStage(i); render(); } },
    h('td', { text: st.id }), h('td', { text: st.stage }), h('td', { text: st.meta.methods.join(', ') }), h('td', { text: String(st.num_batches) }),
    h('td', { text: `${st.meta.ebno_db_range[0]}–${st.meta.ebno_db_range[st.meta.ebno_db_range.length - 1]} dB` }), h('td', { text: String(st.meta.num_ut) }), h('td', { text: st.meta.channel_model_type })));
  return h('div', { class: 'col' },
    card('Stages in this run', `${d.run} · exported ${d.exported_at}`,
      h('table', {}, h('thead', {}, h('tr', {}, ['Stage folder', 'Stage', 'Methods', 'Batches', 'Eb/N0', 'Terminals', 'Channel'].map((t) => h('th', { text: t })))), h('tbody', {}, rows))),
    card('Evidence checks', null, h('ul', { style: { listStyle: 'none', padding: 0, margin: 0 } }, checks)));
}

// ---------- shell ----------
function selectStage(i) {
  S.stage = i;
  indexStage(stage());
  const ms = methods();
  if (!ms.includes(S.method)) S.method = ms[0];
  if (!ms.includes(S.compare) || S.compare === S.method) S.compare = ms.find((m) => m === 'round_robin' && m !== S.method) || ms.find((m) => m !== S.method) || null;
  S.eb = Math.min(S.eb, ebnos().length - 1);
  S.batch = numBatches() - 1;
  S.ut = 0;
  syncControls();
}
function fillSelect(sel, options, value) {
  sel.replaceChildren(...options.map(([v, label]) => h('option', { value: v, text: label })));
  sel.value = value ?? '';
}
function syncControls() {
  fillSelect($('sel-stage'), S.data.stages.map((st, i) => [String(i), st.id]), String(S.stage));
  fillSelect($('sel-method'), methods().map((m) => [m, m]), S.method);
  fillSelect($('sel-compare'), [['', 'none'], ...methods().filter((m) => m !== S.method).map((m) => [m, m])], S.compare || '');
  $('method-swatch').style.background = methodColor(S.method);
  $('eb-label').textContent = `${curEb()} dB`;
  const slider = $('batch');
  slider.max = String(numBatches() - 1);
  slider.value = String(S.batch);
  $('batch-label').textContent = `${S.batch + 1} / ${numBatches()}`;
}
function render() {
  syncControls();
  $('view-title').textContent = VIEWS[S.view];
  document.querySelectorAll('.nav a[data-view]').forEach((a) => a.setAttribute('aria-current', a.dataset.view === S.view ? 'page' : 'false'));
  const fn = { floor: renderFloor, timeline: renderTimeline, compare: renderCompare, run: renderRun }[S.view];
  $('view').replaceChildren(fn());
}
function setPlaying(on) {
  S.playing = on;
  $('play').textContent = on ? '❚❚' : '▶';
  $('play').setAttribute('aria-label', on ? 'Pause' : 'Play');
  clearInterval(S.timer);
  if (on) S.timer = setInterval(() => { S.batch = (S.batch + 1) % numBatches(); render(); }, 1000 / S.speed);
}
function routeFromHash() {
  const v = location.hash.replace('#/', '');
  S.view = VIEWS[v] ? v : 'floor';
}
function wire() {
  $('sel-stage').addEventListener('change', (e) => { selectStage(Number(e.target.value)); render(); });
  $('sel-method').addEventListener('change', (e) => { S.method = e.target.value; if (S.compare === S.method) S.compare = null; render(); });
  $('sel-compare').addEventListener('change', (e) => { S.compare = e.target.value || null; render(); });
  $('eb-down').addEventListener('click', () => { S.eb = Math.max(0, S.eb - 1); render(); });
  $('eb-up').addEventListener('click', () => { S.eb = Math.min(ebnos().length - 1, S.eb + 1); render(); });
  $('batch').addEventListener('input', (e) => { S.batch = Number(e.target.value); render(); });
  $('prev').addEventListener('click', () => { S.batch = Math.max(0, S.batch - 1); render(); });
  $('next').addEventListener('click', () => { S.batch = Math.min(numBatches() - 1, S.batch + 1); render(); });
  $('play').addEventListener('click', () => setPlaying(!S.playing));
  document.querySelectorAll('[data-speed]').forEach((b) => b.addEventListener('click', () => {
    S.speed = Number(b.dataset.speed);
    document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    if (S.playing) setPlaying(true);
  }));
  $('theme-toggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    $('theme-toggle').textContent = dark ? 'Light theme' : 'Dark theme';
    try { localStorage.setItem('f6g-theme', dark ? 'dark' : 'light'); } catch (_) { /* storage may be blocked */ }
    render();
  });
  window.addEventListener('hashchange', () => { routeFromHash(); render(); });
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea') && e.key !== ' ') return;
    if (e.key === ' ') { e.preventDefault(); setPlaying(!S.playing); }
    else if (e.key === 'ArrowRight') { S.batch = Math.min(numBatches() - 1, S.batch + 1); render(); }
    else if (e.key === 'ArrowLeft') { S.batch = Math.max(0, S.batch - 1); render(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); S.eb = Math.min(ebnos().length - 1, S.eb + 1); render(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); S.eb = Math.max(0, S.eb - 1); render(); }
    else if (['1', '2', '3', '4'].includes(e.key)) location.hash = `#/${['floor', 'timeline', 'compare', 'run'][Number(e.key) - 1]}`;
  });
}
async function main() {
  try {
    const saved = localStorage.getItem('f6g-theme');
    if (saved) { document.documentElement.dataset.theme = saved; $('theme-toggle').textContent = saved === 'dark' ? 'Light theme' : 'Dark theme'; }
  } catch (_) { /* storage may be blocked */ }
  try {
    const res = await fetch('data.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`data.json: HTTP ${res.status}`);
    S.data = await res.json();
  } catch (err) {
    $('view').replaceChildren(h('div', { class: 'empty', text: `Could not load data.json (${err.message}). Run "python -m factory6g.viewer export <run_dir>" and serve that folder.` }));
    $('status-text').textContent = 'no data';
    return;
  }
  $('run-name').textContent = S.data.run;
  $('status-text').textContent = `${S.data.stages.length} stage trace${S.data.stages.length > 1 ? 's' : ''} · ${S.data.radio_map ? 'radio map' : 'no radio map'}`;
  const rmStage = S.data.stages.findIndex((st) => st.stage === 'resource_managers' || st.stage === 'coupled');
  selectStage(rmStage >= 0 ? rmStage : 0);
  S.eb = Math.floor(ebnos().length / 2);
  routeFromHash();
  wire();
  render();
}
main();
