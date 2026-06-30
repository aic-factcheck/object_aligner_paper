(function () {
  "use strict";

  const STORAGE_KEY = "oa_report_state_v2__" + (DATA.run_label || "default");
  const MODES = ["raw", "aligned", "goldids"];
  const SCHEMAS = DATA.schema_names && DATA.schema_names.length ? DATA.schema_names : ["score"];
  const MAIN_SCHEMA = SCHEMAS[0];

  function loadState() {
    let st = {
      cand: DATA.best_idx != null ? DATA.best_idx : 0,
      sample: 0,
      mode: "aligned",
      sortKey: MAIN_SCHEMA,
      sortDir: -1,
      wrap: true,
      sidebarCollapsed: false,
    };
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const p = JSON.parse(raw);
        if (typeof p.cand === "number") st.cand = p.cand;
        if (typeof p.sample === "number") st.sample = p.sample;
        if (MODES.indexOf(p.mode) >= 0) st.mode = p.mode;
        if (typeof p.sortKey === "string") st.sortKey = p.sortKey;
        if (p.sortDir === 1 || p.sortDir === -1) st.sortDir = p.sortDir;
        if (typeof p.wrap === "boolean") st.wrap = p.wrap;
        if (typeof p.sidebarCollapsed === "boolean") st.sidebarCollapsed = p.sidebarCollapsed;
      }
    } catch (e) { /* ignore */ }
    return st;
  }
  function saveState(st) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(st)); } catch (e) { /* ignore */ }
  }
  const state = loadState();

  // ---------- elements ----------
  const $ = (id) => document.getElementById(id);
  const candHead = $("cand-head");
  const candBody = $("cand-body");
  const candSubtitle = $("cand-subtitle");
  const sampleSelect = $("sample-select");
  const samplePrev = $("sample-prev");
  const sampleNext = $("sample-next");
  const samplePos = $("sample-pos");
  const modeRaw = $("mode-raw");
  const modeAligned = $("mode-aligned");
  const modeGoldids = $("mode-goldids");
  const scoreBlock = $("score-block");
  const paretoBadge = $("pareto-badge");
  const paretoIter = $("pareto-iter");
  const titleEl = $("ctx-title");
  const ctxBodyEl = $("ctx-body");
  const sysPromptEl = $("sys-prompt-text");
  const sysPromptSummary = $("sys-prompt-summary");
  const feedbackBlock = $("feedback-block");
  const feedbackText = $("feedback-text");
  const bannerEl = $("banner");
  const goldPane = $("gold-pane");
  const predPane = $("pred-pane");
  const layout = $("layout");
  const sidebarToggle = $("sidebar-toggle");
  const wrapToggle = $("wrap-toggle");
  const panes = $("panes");

  // ---------- helpers ----------
  function fmt(v) { return (v == null) ? "—" : Number(v).toFixed(3); }
  function statsFor(candIdx) {
    return (DATA.candidate_stats || []).find(c => c.idx === candIdx) || null;
  }
  function pairOf(candIdx, sampleIdx) { return DATA.pairs[candIdx + "," + sampleIdx]; }

  // ---------- top bar ----------
  (function fillTopBar() {
    const t = $("task-lm-chip");
    t.textContent = "task_lm: " + (DATA.task_lm_model || "—");
    const r = $("reflection-lm-chip");
    if (DATA.reflection_lm_model) r.textContent = "reflection_lm: " + DATA.reflection_lm_model;
    else r.style.display = "none";

    $("config-text").innerHTML = DATA.config
      ? highlightJson(JSON.stringify(DATA.config, null, 2)) : "(no config.json)";

    const chip = $("holdout-chip");
    const ho = DATA.holdout && DATA.holdout.scores;
    if (ho && Object.keys(ho).length) {
      const parts = SCHEMAS.filter(s => ho[s]).map(s =>
        '<span class="ho-item"><span class="ho-name">' + esc(s) + '</span> '
        + '<b>' + fmt(ho[s].mean_score) + '</b></span>');
      const n = (ho[SCHEMAS[0]] && ho[SCHEMAS[0]].n) || "";
      chip.innerHTML = '<span class="ho-label">holdout (test' + (n ? ", n=" + n : "") + "):</span> "
        + parts.join("");
    } else {
      chip.style.display = "none";
    }
  })();

  // ---------- candidate table ----------
  function sortedCandidates() {
    const rows = (DATA.candidate_stats || []).slice();
    const key = state.sortKey;
    rows.sort((a, b) => {
      if (key === "idx") return (a.idx - b.idx) * state.sortDir;
      const av = a.scores[key], bv = b.scores[key];
      if (av == null && bv == null) return a.idx - b.idx;
      if (av == null) return 1;        // nulls last regardless of dir
      if (bv == null) return -1;
      if (av === bv) return a.idx - b.idx;
      return (av - bv) * state.sortDir;
    });
    return rows;
  }
  let candOrder = [];  // candidate idxs in current display order

  function buildCandTable() {
    // header
    candHead.innerHTML = "";
    const mk = (label, key, cls) => {
      const th = document.createElement("th");
      th.textContent = label;
      if (cls) th.className = cls;
      if (key) {
        th.classList.add("sortable");
        if (state.sortKey === key) th.classList.add(state.sortDir < 0 ? "sort-desc" : "sort-asc");
        th.addEventListener("click", () => {
          if (state.sortKey === key) state.sortDir = -state.sortDir;
          else { state.sortKey = key; state.sortDir = key === "idx" ? 1 : -1; }
          saveState(state); buildCandTable();
        });
      }
      return th;
    };
    candHead.appendChild(mk("#", "idx"));
    SCHEMAS.forEach((s, i) => candHead.appendChild(mk(s, s, "num" + (i === 0 ? " main-col" : ""))));

    // body
    candBody.innerHTML = "";
    const rows = sortedCandidates();
    candOrder = rows.map(r => r.idx);
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.dataset.cand = String(r.idx);
      if (r.idx === state.cand) tr.classList.add("selected");
      if (r.is_best) tr.classList.add("best");

      const idxTd = document.createElement("td");
      idxTd.className = "idx-cell";
      idxTd.innerHTML = (r.is_best ? '<span class="star">★</span>' : "") + r.idx
        + (r.n != null ? '<span class="ncells">n=' + r.n + "</span>" : "");
      tr.appendChild(idxTd);

      SCHEMAS.forEach((s, i) => {
        const td = document.createElement("td");
        td.className = "num" + (i === 0 ? " main-col" : "");
        const v = r.scores[s];
        td.appendChild(scoreBar(v));
        tr.appendChild(td);
      });

      tr.addEventListener("click", () => selectCand(r.idx));
      candBody.appendChild(tr);
    });
    candSubtitle.textContent = "valset mean" + (rows.length ? " · " + rows.length + " candidates" : "");
  }

  // a numeric cell: filled bar (0..1) behind the number
  function scoreBar(v) {
    const wrap = document.createElement("span");
    wrap.className = "bar-wrap";
    if (v != null) {
      const bar = document.createElement("span");
      bar.className = "bar-fill";
      bar.style.width = Math.max(0, Math.min(1, v)) * 100 + "%";
      wrap.appendChild(bar);
    }
    const num = document.createElement("span");
    num.className = "bar-num";
    num.textContent = fmt(v);
    wrap.appendChild(num);
    return wrap;
  }

  function selectCand(idx) {
    state.cand = idx;
    saveState(state);
    repaint();
  }

  // ---------- sample selector ----------
  function rebuildSampleOptions() {
    sampleSelect.innerHTML = "";
    DATA.samples.forEach(s => {
      const opt = document.createElement("option");
      opt.value = String(s.idx);
      const pair = pairOf(state.cand, s.idx);
      let scoreTxt = "—";
      let marker = "";
      if (pair) {
        if (pair.score != null) scoreTxt = pair.score.toFixed(3);
        if (pair.pareto_iter != null) marker = " ★";
        else if (!pair.available) marker = " ·";
        if (pair.parse_error || pair.oa_error) scoreTxt = "err";
      } else {
        marker = " ·";
      }
      opt.textContent = s.idx + " — " + (s.sample_id || s.title || ("#" + s.idx))
        + "  [" + scoreTxt + "]" + marker;
      sampleSelect.appendChild(opt);
    });
  }

  function stepSample(delta) {
    const n = DATA.samples.length;
    if (!n) return;
    state.sample = Math.max(0, Math.min(n - 1, state.sample + delta));
    saveState(state);
    repaint();
  }
  function stepCand(delta) {
    if (!candOrder.length) return;
    let pos = candOrder.indexOf(state.cand);
    if (pos < 0) pos = 0;
    pos = Math.max(0, Math.min(candOrder.length - 1, pos + delta));
    selectCand(candOrder[pos]);
  }

  // ---------- pretty-printer (unchanged core) ----------
  function esc(s) {
    return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
  // Syntax-highlight a pretty-printed JSON string into spans (reuses the
  // json-* color classes). Tokenises strings/keys/numbers/booleans/null.
  function highlightJson(jsonStr) {
    const s = esc(jsonStr);
    return s.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (m) => {
        let cls = "json-num";
        if (/^"/.test(m)) cls = /:$/.test(m) ? "json-key" : "json-str";
        else if (/true|false/.test(m)) cls = "json-bool";
        else if (/null/.test(m)) cls = "json-bool";
        return '<span class="' + cls + '">' + m + "</span>";
      }
    );
  }
  function jsonString(s) { return '"' + esc(JSON.stringify(s).slice(1, -1)) + '"'; }
  function renderPrimitive(v) {
    if (v === null) return '<span class="json-bool">null</span>';
    if (typeof v === "string") return '<span class="json-str">' + jsonString(v) + '</span>';
    if (typeof v === "number") return '<span class="json-num">' + v + '</span>';
    if (typeof v === "boolean") return '<span class="json-bool">' + v + '</span>';
    return esc(JSON.stringify(v));
  }
  function isMissing(row) { return row && typeof row === "object" && row.__missing__ === true; }
  function renderRow(row, rowMeta) {
    if (row == null) return '<span class="json-bool">null</span>';
    if (typeof row !== "object" || Array.isArray(row)) return renderPrimitive(row);
    if (isMissing(row)) return '<span class="row-block missing">·</span>';
    const parts = [];
    for (const k of Object.keys(row)) {
      let valHtml = renderPrimitive(row[k]);
      if (rowMeta && rowMeta[k]) valHtml = '<span class="' + esc(rowMeta[k]) + '">' + valHtml + '</span>';
      parts.push('<span class="json-key">"' + esc(k) + '"</span><span class="json-punct">: </span>' + valHtml);
    }
    return '<span class="json-punct">{</span>' + parts.join('<span class="json-punct">, </span>') + '<span class="json-punct">}</span>';
  }
  function rowClass(thisRow, otherRow) {
    if (isMissing(thisRow)) return "missing";
    if (isMissing(otherRow)) return "solo";
    return "matched";
  }
  function renderGraph(obj, otherObj, meta) {
    const lines = [];
    lines.push('<span class="json-punct">{</span>');
    const keys = Object.keys(obj);
    keys.forEach((k, ki) => {
      // Scalar top-level fields (e.g. AMR `root`, a `ref` to a node id) are
      // not arrays — render them inline as a primitive rather than forcing an
      // empty `[ ]`. Comparison colour is deliberately omitted: under RA the
      // pred keeps its own id namespace, so a literal gold↔pred root compare
      // would be spuriously "mismatched" even when referentially correct.
      if (!Array.isArray(obj[k])) {
        const trailing = ki < keys.length - 1 ? '<span class="json-punct">,</span>' : '';
        lines.push('  <span class="section-key">"' + esc(k) + '"</span><span class="json-punct">: </span>'
          + renderPrimitive(obj[k]) + trailing);
        return;
      }
      lines.push('  <span class="section-key">"' + esc(k) + '"</span><span class="json-punct">: [</span>');
      const arr = obj[k];
      const other = otherObj && Array.isArray(otherObj[k]) ? otherObj[k] : null;
      const metaList = meta && Array.isArray(meta[k]) ? meta[k] : null;
      arr.forEach((row, i) => {
        const o = other ? other[i] : null;
        const cls = otherObj ? rowClass(row, o) : null;
        const rowMeta = metaList ? metaList[i] : null;
        const html = renderRow(row, rowMeta);
        const trailing = i < arr.length - 1 ? '<span class="json-punct">,</span>' : '';
        if (cls && !isMissing(row)) lines.push('    <span class="row-block ' + cls + '">' + html + '</span>' + trailing);
        else lines.push('    ' + html + trailing);
      });
      const closer = ki < keys.length - 1 ? ']<span class="json-punct">,</span>' : ']';
      lines.push('  <span class="json-punct">' + closer + '</span>');
    });
    lines.push('<span class="json-punct">}</span>');
    return lines.join("\n");
  }

  // ---------- score line ----------
  function renderScoreLine(pair) {
    scoreBlock.innerHTML = "";
    SCHEMAS.forEach((s, i) => {
      let v = null;
      if (pair) v = (i === 0) ? pair.score : (pair.cross_scores ? pair.cross_scores[s] : null);
      const el = document.createElement("span");
      el.className = "score-item" + (i === 0 ? " score-main" : "");
      el.innerHTML = '<span class="score-name">' + esc(s) + '</span>'
        + '<span class="score-val">' + fmt(v) + "</span>";
      scoreBlock.appendChild(el);
    });
  }

  // ---------- repaint ----------
  function setMode(mode) {
    state.mode = mode;
    modeRaw.classList.toggle("active", mode === "raw");
    modeAligned.classList.toggle("active", mode === "aligned");
    modeGoldids.classList.toggle("active", mode === "goldids");
    saveState(state);
    repaint();
  }

  function applyWrap() {
    panes.classList.toggle("wrap", state.wrap);
    panes.classList.toggle("nowrap", !state.wrap);
    wrapToggle.classList.toggle("active", state.wrap);
  }
  function setWrap(on) { state.wrap = on; saveState(state); applyWrap(); }

  function applySidebar() {
    layout.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
    sidebarToggle.classList.toggle("active", !state.sidebarCollapsed);
  }
  function setSidebar(collapsed) { state.sidebarCollapsed = collapsed; saveState(state); applySidebar(); }

  function repaint() {
    buildCandTable();
    rebuildSampleOptions();
    sampleSelect.value = String(state.sample);
    samplePos.textContent = (state.sample + 1) + " / " + DATA.samples.length;

    const cand = DATA.candidates[state.cand];
    sysPromptEl.textContent = cand ? cand.system_prompt : "";
    sysPromptSummary.textContent = "System prompt — candidate " + state.cand + (cand && cand.is_best ? " (best)" : "");

    const sample = DATA.samples[state.sample];
    titleEl.textContent = sample.title ? (sample.sample_id + " — " + sample.title) : sample.sample_id;
    if (sample.image_data_url) {
      ctxBodyEl.innerHTML = "";
      const img = document.createElement("img");
      img.src = sample.image_data_url;
      img.alt = sample.sample_id;
      img.className = "context-image";
      ctxBodyEl.appendChild(img);
    } else {
      ctxBodyEl.textContent = sample.context || "";
    }

    const pair = pairOf(state.cand, state.sample);
    renderScoreLine(pair);

    bannerEl.textContent = "";
    bannerEl.className = "banner";
    bannerEl.style.display = "none";
    feedbackBlock.style.display = "none";
    paretoBadge.style.display = "none";

    if (pair && pair.pareto_iter != null) {
      paretoBadge.style.display = "";
      paretoIter.textContent = String(pair.pareto_iter);
    }

    let goldObj = sample.gold;
    let predObj = pair && pair.raw ? pair.raw : null;

    if (!pair || !pair.available) {
      bannerEl.textContent = "No prediction stored for candidate " + state.cand + " on sample " + state.sample + ".";
      bannerEl.classList.add("warn");
      bannerEl.style.display = "block";
    } else if (pair.parse_error) {
      bannerEl.textContent = "Prediction parse error: " + pair.parse_error;
      bannerEl.classList.add("err");
      bannerEl.style.display = "block";
    } else if (pair.oa_error) {
      bannerEl.textContent = "OA alignment error: " + pair.oa_error + ". Showing raw mode only.";
      bannerEl.classList.add("warn");
      bannerEl.style.display = "block";
    } else if (pair.feedback) {
      feedbackBlock.style.display = "block";
      feedbackText.textContent = pair.feedback;
    }

    let goldView, predView, classifyPairs, predMeta;
    if (state.mode === "goldids" && pair && pair.aligned_gold && pair.aligned_pred_goldids) {
      goldView = pair.aligned_gold; predView = pair.aligned_pred_goldids;
      classifyPairs = true; predMeta = pair.aligned_pred_goldids_meta || null;
    } else if ((state.mode === "aligned" || state.mode === "goldids") && pair && pair.aligned_gold && pair.aligned_pred) {
      goldView = pair.aligned_gold; predView = pair.aligned_pred;
      classifyPairs = true; predMeta = null;
    } else {
      goldView = goldObj; predView = predObj; classifyPairs = false; predMeta = null;
    }

    goldPane.innerHTML = goldView
      ? renderGraph(goldView, classifyPairs ? predView : null, null)
      : '<span class="row-block missing">no data</span>';
    predPane.innerHTML = predView
      ? renderGraph(predView, classifyPairs ? goldView : null, predMeta)
      : '<span class="row-block missing">no prediction</span>';
  }

  // ---------- wire up ----------
  sampleSelect.addEventListener("change", () => {
    state.sample = parseInt(sampleSelect.value, 10);
    saveState(state); repaint();
  });
  samplePrev.addEventListener("click", () => stepSample(-1));
  sampleNext.addEventListener("click", () => stepSample(1));
  modeRaw.addEventListener("click", () => setMode("raw"));
  modeAligned.addEventListener("click", () => setMode("aligned"));
  modeGoldids.addEventListener("click", () => setMode("goldids"));
  wrapToggle.addEventListener("click", () => setWrap(!state.wrap));
  sidebarToggle.addEventListener("click", () => setSidebar(!state.sidebarCollapsed));

  // Keep the two panes horizontally aligned when wrapping is off.
  let _syncing = false;
  function linkScroll(src, dst) {
    src.addEventListener("scroll", () => {
      if (_syncing) return;
      _syncing = true;
      dst.scrollLeft = src.scrollLeft;
      _syncing = false;
    });
  }
  linkScroll(goldPane, predPane);
  linkScroll(predPane, goldPane);

  function typingTarget(el) {
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || el.isContentEditable;
  }
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const typing = typingTarget(document.activeElement);
    // When the sample <select> has focus, let it handle arrows natively.
    if (typing && document.activeElement === sampleSelect
        && e.key.indexOf("Arrow") === 0) return;
    switch (e.key) {
      case "ArrowLeft": stepSample(-1); e.preventDefault(); break;
      case "ArrowRight": stepSample(1); e.preventDefault(); break;
      case "ArrowUp": stepCand(-1); e.preventDefault(); break;
      case "ArrowDown": stepCand(1); e.preventDefault(); break;
      case "r": if (!typing) setMode("raw"); break;
      case "a": if (!typing) setMode("aligned"); break;
      case "g": if (!typing) setMode("goldids"); break;
      case "w": if (!typing) setWrap(!state.wrap); break;
    }
  });

  applyWrap();
  applySidebar();
  setMode(state.mode);
})();
