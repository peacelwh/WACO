/* WACO project page — interactions, benchmark explorer and charts (no dependencies). */
(function () {
  'use strict';
  var D = window.WACO_DATA;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function f1(v) { return typeof v === 'number' ? v.toFixed(1) : v; }

  /* ---------------- nav ---------------- */
  var nav = $('#nav');
  function onScroll() { nav.classList.toggle('scrolled', window.scrollY > 24); }
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  $('#navToggle').addEventListener('click', function () { $('#navLinks').classList.toggle('open'); });
  $$('#navLinks a').forEach(function (a) { a.addEventListener('click', function () { $('#navLinks').classList.remove('open'); }); });

  /* ---------------- reveal + lazy video playback ---------------- */
  var pending = $$('.reveal');
  function revealCheck() {
    var vh = window.innerHeight || document.documentElement.clientHeight;
    pending = pending.filter(function (n) {
      var r = n.getBoundingClientRect();
      if (r.top < vh * 0.94 && r.bottom > 0) { n.classList.add('in'); return false; }
      return true;
    });
  }
  window.addEventListener('scroll', revealCheck, { passive: true });
  window.addEventListener('resize', revealCheck);
  window.addEventListener('load', revealCheck);
  revealCheck();

  var vio = ('IntersectionObserver' in window) ? new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      var v = e.target;
      if (e.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else { v.pause(); }
    });
  }, { threshold: 0.15 }) : null;
  function watch(v) { if (vio) vio.observe(v); }

  /* ---------------- hero: overview video, then the real-robot demo ---------------- */
  var demoA = $('#demoA'), demoB = $('#demoB');
  var film = $('#demoFilm'), pair = $('#demoPair'), ov = $('#overviewVideo'), chip = $('#demoChip');
  var filmChip = chip.textContent;
  ov.addEventListener('loadedmetadata', function () {
    if (!isFinite(ov.duration)) return;
    var s = Math.round(ov.duration);
    filmChip = Math.floor(s / 60) + ' min ' + (s % 60) + ' s · with sound';
    if (!film.hidden) chip.textContent = filmChip;
  });
  ov.addEventListener('play', function () { film.classList.add('started'); });
  $('#filmPlay').addEventListener('click', function () { var p = ov.play(); if (p && p.catch) p.catch(function () {}); });
  function setDemo(task) {
    var isFilm = task === 'overview';
    film.hidden = !isFilm; pair.hidden = isFilm;
    $('#demoRestart').hidden = isFilm;
    chip.textContent = isFilm ? filmChip : '3× speed';
    if (isFilm) { demoA.pause(); demoB.pause(); return; }
    ov.pause();
    demoA.poster = 'static/videos/real/' + task + '_nominal.jpg'; demoA.src = 'static/videos/real/' + task + '_nominal.mp4';
    demoB.poster = 'static/videos/real/' + task + '_base.jpg'; demoB.src = 'static/videos/real/' + task + '_base.mp4';
    [demoA, demoB].forEach(function (v) { var p = v.play(); if (p && p.catch) p.catch(function () {}); });
  }
  $$('#demoTabs .demo-tab').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('#demoTabs .demo-tab').forEach(function (x) { x.classList.remove('is-on'); });
      b.classList.add('is-on'); setDemo(b.dataset.task);
    });
  });
  $('#demoRestart').addEventListener('click', function () {
    [demoA, demoB].forEach(function (v) { v.currentTime = 0; var p = v.play(); if (p && p.catch) p.catch(function () {}); });
  });

  /* ---------------- benchmark explorer ---------------- */
  var KIND = { cam: 'Camera shift', base: 'Base shift', both: 'Compound shift' };
  var KIND_TEXT = {
    clean: 'The calibration pose: the camera and the robot base are exactly where they were during data collection.',
    cam: 'The third-person camera is translated and rotated. The whole third-person image is displaced; the objects, the correct actions and the wrist view are unchanged.',
    base: 'The robot is moved in the floor plane and rotated about the vertical; table, objects and third-person camera stay fixed. The scene stays in place in the image, the arm moves in it, and state and actions are expressed in the moved base frame.',
    both: 'The camera shift and the base shift are applied independently, the hardest condition: only an estimate that separates the two handles it.'
  };
  var LV = ['clean', 'l1', 'l2', 'l3'];
  var state = { task: D.bench.tasks[0].id, kind: 'cam', level: 3 };
  var expTask = $('#expTask');
  D.bench.tasks.forEach(function (t, i) {
    var b = h('button', i === 0 ? 'is-on' : '', t.label); b.dataset.task = t.id; b.title = t.instruction;
    b.addEventListener('click', function () { state.task = t.id; $$('button', expTask).forEach(function (x) { x.classList.toggle('is-on', x === b); }); renderExp(); });
    expTask.appendChild(b);
  });
  $$('#expKind button').forEach(function (b) {
    b.addEventListener('click', function () { state.kind = b.dataset.kind; $$('#expKind button').forEach(function (x) { x.classList.toggle('is-on', x === b); }); renderExp(); });
  });
  $('#expLevel').addEventListener('input', function (e) { state.level = +e.target.value; renderExp(); });
  var expVid = $('#expVid'), expRef = $('#expRef');
  watch(expVid); watch(expRef);
  function setSrc(v, src) { if (v.getAttribute('src') !== src) { v.setAttribute('src', src); var p = v.play(); if (p && p.catch) p.catch(function () {}); } }

  function renderExp() {
    var lv = LV[state.level], clean = state.level === 0;
    var cond = clean ? 'clean' : state.kind + '_' + lv;
    var t = D.bench.tasks.filter(function (x) { return x.id === state.task; })[0];
    setSrc(expRef, 'static/videos/bench/' + t.id + '_clean.mp4');
    setSrc(expVid, 'static/videos/bench/' + t.id + '_' + cond + '.mp4');
    var title = clean ? 'Clean' : KIND[state.kind] + ' · ' + lv.toUpperCase();
    $('#condTitle').textContent = title;
    $('#condText').textContent = KIND_TEXT[clean ? 'clean' : state.kind];
    var lab = $('#expLabel'); lab.className = 'vid-label ' + (clean ? '' : (state.kind === 'cam' ? 'cam' : 'shift'));
    lab.innerHTML = '<i></i>' + title;
    var L = D.bench.levels[lv], kv = $('#condKv'); kv.innerHTML = '';
    function row(k, v) { kv.appendChild(h('dt', '', k)); kv.appendChild(h('dd', '', v)); }
    row('task', t.instruction);
    if (!clean) {
      if (state.kind !== 'base') row('camera', '≤ ' + L.cam_t + ' cm, ± ' + L.cam_r + '° per axis');
      if (state.kind !== 'cam') row('base', '≤ ' + L.base_t + ' cm, ± ' + L.base_yaw + '° yaw');
      if (state.kind !== 'base') row('image', 'a target object moves up to ' + L.px + ' px (translation / yaw) in the 256² view');
    }
    drawTop(t, clean ? null : t.shifts[state.kind + '/' + lv]);
  }

  /* schematic top view of the sampled shift (translations magnified) */
  function drawTop(t, s) {
    var svg = $('#topView'); svg.innerHTML = '';
    var MAG = 260;                                   // px per metre for the shifts (schematic, magnified)
    el('rect', { x: 62, y: 22, width: 92, height: 76, rx: 6, fill: '#fff', stroke: '#C9D4E3' }, svg);
    el('text', { x: 108, y: 64, 'text-anchor': 'middle', 'font-size': 10, fill: '#8C96A8', 'font-family': 'Inter,sans-serif' }, svg, 'table');
    var bx = 34, by = 60, cx = 182, cy = 60;
    // nominal poses
    el('circle', { cx: bx, cy: by, r: 9, fill: 'none', stroke: '#B8C3D4', 'stroke-dasharray': '3 3' }, svg);
    el('path', { d: 'M' + (cx - 8) + ' ' + (cy - 6) + ' h12 l5 -3 v18 l-5 -3 h-12 z', fill: 'none', stroke: '#B8C3D4', 'stroke-dasharray': '3 3', transform: 'rotate(180 ' + cx + ' ' + cy + ')' }, svg);
    var db = s ? s.base_dpos : [0, 0, 0], yaw = s ? s.base_dyaw : 0, dc = s ? s.cam_dpos : [0, 0, 0], cyaw = s ? s.cam_drot_rpy[2] : 0;
    var gB = el('g', { transform: 'translate(' + (bx + db[0] * MAG) + ' ' + (by - db[1] * MAG) + ') rotate(' + (-yaw) + ')' }, svg);
    el('circle', { r: 9, fill: (s && (db[0] || db[1] || yaw)) ? '#EB6834' : '#5A6577' }, gB);
    el('path', { d: 'M0 0 L15 0', stroke: '#fff', 'stroke-width': 2.4, 'stroke-linecap': 'round' }, gB);
    var gC = el('g', { transform: 'translate(' + (cx + dc[0] * MAG) + ' ' + (cy - dc[1] * MAG) + ') rotate(' + (180 - cyaw) + ')' }, svg);
    el('path', { d: 'M-8 -6 h12 l5 -3 v18 l-5 -3 h-12 z', fill: (s && (dc[0] || dc[1] || cyaw)) ? '#2A78D6' : '#5A6577' }, gC);
    el('text', { x: bx, y: 108, 'text-anchor': 'middle', 'font-size': 10, fill: '#5A6577', 'font-family': 'Inter,sans-serif' }, svg, 'robot base');
    el('text', { x: cx, y: 108, 'text-anchor': 'middle', 'font-size': 10, fill: '#5A6577', 'font-family': 'Inter,sans-serif' }, svg, 'camera');
    var txt;
    if (!s) txt = 'Top view (schematic). Dashed outlines mark the calibration pose of the robot base and of the third-person camera.';
    else {
      var parts = [];
      var ct = Math.hypot(dc[0], dc[1], dc[2]) * 100, bt = Math.hypot(db[0], db[1]) * 100;
      if (ct > 0) parts.push('camera moved ' + ct.toFixed(1) + ' cm and rotated ' + s.cam_drot_rpy.map(function (v) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + '°'; }).join(' / ') + ' (roll / pitch / yaw)');
      if (bt > 0) parts.push('base moved ' + bt.toFixed(1) + ' cm and yawed ' + (yaw >= 0 ? '+' : '−') + Math.abs(yaw).toFixed(1) + '°');
      txt = 'Sampled shift of this episode: ' + parts.join('; ') + '. Top view, schematic; dashed outlines mark the calibration pose.';
    }
    $('#topText').textContent = txt;
  }
  renderExp();

  /* ---------------- chart: success vs shift level ---------------- */
  var SERIES = [
    { key: 'waco', name: 'WACO (W1 + runtime)', short: 'WACO', color: '#2A78D6', row: 'WACO (W1 + runtime)', w: 2.6 },
    { key: 'rt', name: 'WACO runtime on B1', short: 'Runtime on B1', color: '#1BAF7A', row: 'WACO runtime on B1', w: 2 },
    { key: 'b1', name: 'B1 (official few-shot recipe)', short: 'B1', color: '#EB6834', row: 'B1 (official few-shot recipe)', w: 2 },
    { key: 'ref', name: 'GT oracle on W1 (upper bound)', short: 'GT oracle', color: '#7C8799', row: 'GT oracle on W1 (upper bound)', w: 2, dash: '5 4' }
  ];
  var KINDS = [{ k: 'cam', name: 'Camera shift', color: '#2A78D6', off: 1 }, { k: 'base', name: 'Base shift', color: '#EB6834', off: 4 }, { k: 'both', name: 'Camera + base', color: '#6250D6', off: 7 }];
  var XL = ['Clean', 'L1', 'L2', 'L3'];
  function rowOf(name) { return D.main.filter(function (r) { return r.name === name; })[0]; }
  function vals(name, off) { var c = rowOf(name).cells; return [c[0], c[off], c[off + 1], c[off + 2]]; }
  var tip = $('#chartTip');

  function buildPanels() {
    var host = $('#vizPanels'); host.innerHTML = '';
    var W = 380, H = 300, m = { l: 44, r: 58, t: 14, b: 34 }, y0 = 40, y1 = 100;
    function X(i) { return m.l + i * (W - m.l - m.r) / 3; }
    function Y(v) { return m.t + (y1 - v) * (H - m.t - m.b) / (y1 - y0); }
    KINDS.forEach(function (kd) {
      var p = h('div', 'panel');
      p.appendChild(h('div', 'panel-title', '<i style="background:' + kd.color + '"></i>' + kd.name));
      var svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', kd.name + ': success rate from the clean condition to level 3 for B1, the WACO runtime on B1, WACO and the ground-truth oracle');
      [40, 60, 80, 100].forEach(function (g) {
        el('line', { x1: m.l, x2: W - m.r, y1: Y(g), y2: Y(g), stroke: '#E6ECF4', 'stroke-width': 1 }, svg);
        el('text', { x: m.l - 8, y: Y(g) + 4, 'text-anchor': 'end' }, svg, g);
      });
      XL.forEach(function (lab, i) { el('text', { x: X(i), y: H - 12, 'text-anchor': 'middle' }, svg, lab); });
      var ends = [];
      SERIES.slice().reverse().forEach(function (s) {
        var v = vals(s.row, kd.off);
        var d = v.map(function (val, i) { return (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(val).toFixed(1); }).join(' ');
        var a = { d: d, fill: 'none', stroke: s.color, 'stroke-width': s.w, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' };
        if (s.dash) a['stroke-dasharray'] = s.dash;
        el('path', a, svg);
        if (!s.dash) v.forEach(function (val, i) { el('circle', { cx: X(i), cy: Y(val), r: i === 3 ? 4.5 : 3, fill: s.color, stroke: '#fff', 'stroke-width': 2 }, svg); });
        ends.push({ s: s, v: v[3], y: Y(v[3]) });
      });
      // direct labels at the L3 end, pushed apart so they never collide
      ends.sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      var over = ends[ends.length - 1].y - (H - m.b - 2);
      if (over > 0) ends.forEach(function (e) { e.y -= over; });
      ends.forEach(function (e) {
        el('circle', { cx: W - m.r + 12, cy: e.y, r: 3.5, fill: e.s.color }, svg);
        el('text', { x: W - m.r + 20, y: e.y + 4, 'class': 'val' }, svg, e.v.toFixed(1));
      });
      // hover layer: crosshair + tooltip, the whole plot is the hit target
      var cross = el('line', { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: '#0F1728', 'stroke-width': 1, opacity: 0 }, svg);
      var hit = el('rect', { x: m.l - 20, y: m.t, width: W - m.l - m.r + 40, height: H - m.t - m.b, fill: 'transparent' }, svg);
      function move(ev) {
        var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
        var i = Math.max(0, Math.min(3, Math.round((px - m.l) / ((W - m.l - m.r) / 3))));
        cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('opacity', .35);
        var html = '<b>' + kd.name + ' · ' + XL[i] + '</b>';
        SERIES.forEach(function (s) { html += '<div class="row"><span><i style="background:' + s.color + '"></i>' + s.short + '</span><span>' + vals(s.row, kd.off)[i].toFixed(1) + '%</span></div>'; });
        tip.innerHTML = html; tip.style.opacity = 1;
        var tx = ev.pageX + 16, tw = tip.offsetWidth;
        if (tx + tw > window.scrollX + document.documentElement.clientWidth - 8) tx = ev.pageX - tw - 16;
        tip.style.left = tx + 'px'; tip.style.top = (ev.pageY - 20) + 'px';
      }
      hit.addEventListener('mousemove', move);
      hit.addEventListener('mouseleave', function () { tip.style.opacity = 0; cross.setAttribute('opacity', 0); });
      hit.addEventListener('touchstart', function (ev) { move(ev.touches[0]); }, { passive: true });
      p.appendChild(svg); host.appendChild(p);
    });
    // table view of the same data
    var t = '<table class="tbl"><thead><tr><th>Method</th>' + D.cond.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead><tbody>';
    SERIES.forEach(function (s) { var r = rowOf(s.row); t += '<tr' + (s.key === 'waco' ? ' class="ours"' : '') + '><td>' + s.name + '</td>' + r.cells.map(function (c) { return '<td>' + c.toFixed(1) + '</td>'; }).join('') + '</tr>'; });
    $('#vizTable').innerHTML = t + '</tbody></table>';
  }
  buildPanels();
  $('#vizTableBtn').addEventListener('click', function (e) {
    var on = $('#vizShift').classList.toggle('show-table');
    e.target.setAttribute('aria-pressed', on); e.target.textContent = on ? 'View as chart' : 'View as table';
  });

  /* ---------------- gap to the oracle ---------------- */
  (function () {
    var host = $('#gapRows');
    D.gap.forEach(function (g) {
      var row = h('div', 'gap-row');
      row.appendChild(h('div', 'gap-name', g.name + '<small>' + g.sub + '</small>'));
      var tr = h('div', 'gap-track'), W = 600, H = 44, lo = 30, hi = 100, pl = 8, pr = 8;
      var svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('preserveAspectRatio', 'none');
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', g.name + ': standard recipe ' + g.b1 + '%, WACO ' + g.waco + '%, ground-truth oracle ' + g.oracle + '%');
      function X(v) { return pl + (v - lo) * (W - pl - pr) / (hi - lo); }
      el('line', { x1: X(lo), x2: X(hi), y1: 20, y2: 20, stroke: '#E6ECF4', 'stroke-width': 1 }, svg);
      [40, 60, 80, 100].forEach(function (v) { el('line', { x1: X(v), x2: X(v), y1: 16, y2: 24, stroke: '#C9D4E3', 'stroke-width': 1 }, svg); el('text', { x: X(v), y: 41, 'text-anchor': 'middle' }, svg, v); });
      el('line', { x1: X(g.b1), x2: X(g.oracle), y1: 20, y2: 20, stroke: '#C9D4E3', 'stroke-width': 4, 'stroke-linecap': 'round' }, svg);
      el('line', { x1: X(g.b1), x2: X(g.waco), y1: 20, y2: 20, stroke: '#2A78D6', 'stroke-width': 4, 'stroke-linecap': 'round' }, svg);
      [['b1', '#EB6834', 'Standard recipe'], ['oracle', '#7C8799', 'GT oracle'], ['waco', '#2A78D6', 'WACO']].forEach(function (k) {
        var c = el('circle', { cx: X(g[k[0]]), cy: 20, r: 6, fill: k[1], stroke: '#fff', 'stroke-width': 2 }, svg);
        el('title', {}, c, k[2] + ': ' + g[k[0]].toFixed(1) + '%');
      });
      el('text', { x: X(g.b1), y: 9, 'text-anchor': 'middle', 'class': 'val' }, svg, g.b1.toFixed(1));
      el('text', { x: X(g.waco) - 8, y: 9, 'text-anchor': 'end', 'class': 'val' }, svg, g.waco.toFixed(1));
      el('text', { x: X(g.oracle) + 8, y: 9, 'text-anchor': 'start', 'class': 'val' }, svg, g.oracle.toFixed(1));
      tr.appendChild(svg); row.appendChild(tr);
      row.appendChild(h('div', 'gap-pct', g.pct + '%<small>of the gap recovered</small>'));
      host.appendChild(row);
    });
  })();

  /* ---------------- tables ---------------- */
  function gain(v) { return v == null ? '' : '<span class="gain">' + (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + '</span>'; }
  (function () {
    var b1 = rowOf('B1 (official few-shot recipe)').avg;
    var t = '<table class="tbl"><thead><tr><th>Method</th>' + D.cond.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '<th class="avg">Avg</th></tr></thead><tbody>';
    D.main.forEach(function (r) {
      var ours = r.name.indexOf('WACO (') === 0, ref = r.runtime === 'GT';
      var name = r.name.replace(' (upper bound)', ' †');
      t += '<tr class="' + (ours ? 'ours' : ref ? 'ref' : '') + '"><td>' + name + '</td>' + r.cells.map(function (c) { return '<td>' + c.toFixed(1) + '</td>'; }).join('') +
        '<td class="avg">' + r.avg.toFixed(1) + (ref || r.avg === b1 ? '' : gain(r.avg - b1)) + '</td></tr>';
    });
    $('#tblMain').innerHTML = t + '</tbody></table>';

    t = '<table class="tbl"><thead><tr><th>Method</th><th>Perturbed data</th><th>Released</th><th>Corrected</th></tr></thead><tbody>';
    D.liberoplus.forEach(function (r, i) { t += '<tr' + (i === 2 ? ' class="ours"' : '') + '><td>' + r[0].replace('pi0.5', 'π<sub>0.5</sub>') + '</td><td>' + (r[1] === '-' ? '–' : r[1]) + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td></tr>'; });
    $('#tblLp').innerHTML = t + '</tbody></table>';

    var rb = D.real[0].avg;
    t = '<table class="tbl"><thead><tr><th>Method</th><th>Nominal</th><th>Camera</th><th>Base</th><th>Both</th><th class="avg">Avg</th></tr></thead><tbody>';
    D.real.forEach(function (r) {
      var ours = r.name.indexOf('WACO (') === 0, ref = r.name.indexOf('oracle') >= 0;
      t += '<tr class="' + (ours ? 'ours' : ref ? 'ref' : '') + '"><td>' + r.name.replace('pi0.5', 'π<sub>0.5</sub>').replace(' (upper bound)', ' †') + '</td>' + r.cells.map(function (c) { return '<td>' + c.toFixed(1) + '</td>'; }).join('') +
        '<td class="avg">' + r.avg.toFixed(1) + (ref || r.avg === rb ? '' : gain(r.avg - rb)) + '</td></tr>';
    });
    $('#tblReal').innerHTML = t + '</tbody></table>';

    t = '<table class="tbl"><thead><tr><th>Demonstrations per task</th><th>10</th><th>20</th><th>50</th></tr></thead><tbody>';
    D.fewshot.forEach(function (r) { t += '<tr' + (r[0].indexOf('WACO (') === 0 ? ' class="ours"' : '') + '><td>' + r[0] + '</td><td>' + r[1].toFixed(1) + '</td><td>' + r[2].toFixed(1) + '</td><td>' + r[3].toFixed(1) + '</td></tr>'; });
    var w = D.fewshot[3], b = D.fewshot[0];
    t += '<tr><td>gain of WACO over B1</td>' + [1, 2, 3].map(function (i) { return '<td>' + gain(w[i] - b[i]) + '</td>'; }).join('') + '</tr>';
    $('#tblFew').innerHTML = t + '</tbody></table>';

    t = '<table class="tbl"><thead><tr><th>Block</th><th>Variant</th><th>Clean</th><th>Cam L3</th><th>Base L3</th><th class="avg">Avg</th></tr></thead><tbody>';
    D.ablation.forEach(function (r) { t += '<tr' + (r[1] === 'WACO (full)' ? ' class="ours"' : '') + '><td>' + (r[0] === '-' ? '–' : r[0]) + '</td><td style="text-align:left">' + r[1].replace('dT_c', 'ΔT<sub>c</sub>').replace('dT_b', 'ΔT<sub>b</sub>') + '</td><td>' + r[2].toFixed(1) + '</td><td>' + r[3].toFixed(1) + '</td><td>' + r[4].toFixed(1) + '</td><td class="avg">' + r[5].toFixed(1) + '</td></tr>'; });
    $('#tblAbl').innerHTML = t + '</tbody></table>';
  })();
  $$('#resTabs .tab').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('#resTabs .tab').forEach(function (x) { x.classList.toggle('is-on', x === b); });
      $$('.tabpane').forEach(function (p) { p.classList.toggle('is-on', p.id === b.dataset.pane); });
    });
  });

  /* ---------------- videos ---------------- */
  (function () {
    var g = $('#simGrid');
    D.sim.forEach(function (c) {
      var card = h('div', 'sim-card');
      card.innerHTML = '<div class="sim-pair">' +
        '<div class="sim-cell"><video muted loop playsinline preload="metadata" src="static/videos/sim/' + c.id + '_b1.mp4"></video><span class="vid-label fail"><i></i>Standard recipe</span></div>' +
        '<div class="sim-cell"><video muted loop playsinline preload="metadata" src="static/videos/sim/' + c.id + '_waco.mp4"></video><span class="vid-label"><i></i>WACO</span></div></div>' +
        '<h4>' + c.title + '</h4><p>' + c.text + '</p>';
      g.appendChild(card);
    });
    var TASKS = [['lemon', 'Lemon to Basket'], ['color', 'Color Matching'], ['toy', 'Toy on Shelf'], ['toast', 'Toast into Toaster']];
    [['realNominal', 'nominal', 'calibration pose'], ['realBase', 'base', 'base shift']].forEach(function (cfg) {
      var host = $('#' + cfg[0]);
      TASKS.forEach(function (t) {
        var card = h('div', 'real-card');
        card.innerHTML = '<div class="real-vid"><video muted loop playsinline preload="metadata" poster="static/videos/real/' + t[0] + '_' + cfg[1] + '.jpg" src="static/videos/real/' + t[0] + '_' + cfg[1] + '.mp4"></video></div>' +
          '<div class="cap"><b>' + t[1] + '</b><span>WACO · ' + cfg[2] + '</span></div>';
        host.appendChild(card);
      });
    });
    $$('#videos video').forEach(watch);
  })();

  /* ---------------- citation + lightbox ---------------- */
  $('#bibCopy').addEventListener('click', function (e) {
    var txt = $('#bibText').textContent;
    function done() { e.target.textContent = 'Copied'; setTimeout(function () { e.target.textContent = 'Copy'; }, 1600); }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done, done);
    else { var ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch (err) {} document.body.removeChild(ta); done(); }
  });
  var lb = $('#lightbox'), lbImg = $('img', lb);
  $$('img[data-zoom]').forEach(function (im) { im.addEventListener('click', function () { lbImg.src = im.currentSrc || im.src; lbImg.alt = im.alt; lb.classList.add('open'); lb.setAttribute('aria-hidden', 'false'); }); });
  lb.addEventListener('click', function () { lb.classList.remove('open'); lb.setAttribute('aria-hidden', 'true'); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.click(); });
})();
