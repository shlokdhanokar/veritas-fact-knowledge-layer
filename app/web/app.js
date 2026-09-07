/* Veritas UI.
 *
 * Plain JS, no framework. The page has one job — show a judgement next to the
 * evidence that produced it — and a build step would add friction to running
 * this from a clean clone without buying anything back.
 */

const MARK = {
  CONTRADICTS: 'var(--contradict)',
  LIKELY_CONTRADICTS: 'var(--likely)',
  RECONCILED: 'var(--reconcile)',
  CORROBORATES: 'var(--corroborate)',
};

const PAGE = 15;

const state = {
  view: 'RECONCILED',
  docId: null,
  crossDoc: false,
  search: '',
  offset: 0,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

/* ------------------------------------------------------------------ header */

async function loadStats() {
  const s = await api('/api/stats');
  $('#n-docs').textContent = s.documents;
  $('#n-pages').textContent = s.pages ?? 0;
  $('#n-facts').textContent = s.facts.toLocaleString();
  const total = Object.values(s.verdicts || {}).reduce((a, b) => a + b, 0);
  $('#n-rels').textContent = total.toLocaleString();

  document.querySelectorAll('.tab[data-view]').forEach((tab) => {
    const n = s.verdicts?.[tab.dataset.view];
    const slot = tab.querySelector('.n');
    if (slot) slot.textContent = n ? n.toLocaleString() : '';
  });
}

async function loadDocs() {
  const docs = await api('/api/documents');
  const host = $('#docs');
  host.innerHTML = '';

  if (!docs.length) {
    host.append(el('p', 'empty', 'No documents yet.'));
    return;
  }

  docs.forEach((d) => {
    const row = el('div', 'doc' + (state.docId === d.doc_id ? ' active' : ''));
    row.append(el('div', 'doc-name', d.filename));
    const rate = d.stats?.kept && d.stats?.proposed
      ? ` · ${Math.round((d.stats.kept / d.stats.proposed) * 100)}% grounded`
      : '';
    row.append(el('div', 'doc-meta', `${d.page_count}pp · ${d.fact_count} facts${rate}`));
    row.onclick = () => {
      state.docId = state.docId === d.doc_id ? null : d.doc_id;
      state.offset = 0;
      loadDocs();
      render();
    };
    host.append(row);
  });
}

/* --------------------------------------------------------- the signature */

function contextRow(key, left, right, kind) {
  const row = el('div', `key ${kind}`);
  row.append(el('span', 'k', key));
  const value = kind === 'differ'
    ? `${left || '—'} ⁄ ${right || '—'}`
    : (left || right || '—');
  row.append(el('span', 'v', value));
  return row;
}

function buildSpine(rel) {
  const { left, right } = rel;
  const spine = el('div', 'spine');

  spine.append(factSide(left));

  const keys = el('div', 'keys');
  keys.append(el('div', 'keys-title', 'context'));

  const allKeys = new Set([
    ...Object.keys(left.context || {}),
    ...Object.keys(right.context || {}),
  ]);

  // Differing keys first: they are the reason the verdict is what it is.
  const ordered = [...allKeys].sort((a, b) => {
    const rank = (k) =>
      rel.differing_keys?.includes(k) ? 0 : rel.unstated_keys?.includes(k) ? 1 : 2;
    return rank(a) - rank(b) || a.localeCompare(b);
  });

  if (!ordered.length) {
    keys.append(el('div', 'key match', 'no qualifiers stated'));
  }

  ordered.forEach((k) => {
    const l = left.context?.[k];
    const r = right.context?.[k];
    let kind = 'match';
    if (rel.differing_keys?.includes(k)) kind = 'differ';
    else if (!l || !r) kind = 'unstated';
    keys.append(contextRow(k, l, r, kind));
  });

  // valid_time is a differing "key" without living in context.
  if (rel.differing_keys?.includes('valid_time')) {
    const fmt = (f) => `${f.valid_from || '?'} → ${f.valid_to || 'open'}`;
    keys.append(contextRow('valid time', fmt(left), fmt(right), 'differ'));
  }

  spine.append(keys);
  spine.append(factSide(right));
  return spine;
}

function describeValue(f) {
  const q = f.quantity;
  if (q && q.canonical_value != null) {
    const v = Math.abs(q.canonical_value) >= 1e5
      ? q.canonical_value.toExponential(4).replace('e+', 'e')
      : q.canonical_value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    return `${v} ${q.canonical_unit || ''}`.trim();
  }
  return f.value_text || '—';
}

function factSide(f) {
  const side = el('div', 'side');
  side.append(el('div', 'claim', describeValue(f)));
  side.append(el('div', 'subject-metric', `${f.subject} · ${f.metric}`));

  const ev = el('div', 'evidence');
  ev.textContent = (f.evidence?.quote || '').trim();
  side.append(ev);

  side.append(el('div', 'provenance',
    `${f.evidence?.filename || '?'} · p${f.evidence?.page} · chars ${f.evidence?.start}–${f.evidence?.end}`));

  if (f.notes) side.append(el('div', 'flag', `⚠ ${f.notes}`));
  return side;
}

function judgementCard(rel) {
  const card = el('article', 'judgement');
  card.style.setProperty('--mark', MARK[rel.verdict] || 'var(--rule)');

  const head = el('div', 'judgement-head');
  head.append(el('span', 'verdict', rel.verdict.replace(/_/g, ' ')));
  head.append(el('span', 'method', `via ${rel.method}`));
  head.append(el('span', 'confidence', `confidence ${rel.confidence.toFixed(2)}`));
  card.append(head);

  card.append(buildSpine(rel));

  const why = el('div', 'reasoning');
  why.textContent = rel.reasoning;
  card.append(why);

  return card;
}

/* ------------------------------------------------------------------ facts */

function factRow(f) {
  const row = el('div', 'fact-row');
  const main = el('div');
  main.append(el('div', 'lbl', `${f.subject} · ${f.metric}`));
  const ctx = Object.entries(f.context || {})
    .map(([k, v]) => `${k}=${v}`).join('  ');
  if (ctx) main.append(el('div', 'ctx', ctx));
  main.append(el('div', 'ctx', `${f.evidence.filename} p${f.evidence.page}`));
  if (f.notes) main.append(el('div', 'flag', `⚠ ${f.notes}`));
  row.append(main);
  row.append(el('div', 'val', describeValue(f)));
  return row;
}

/* ----------------------------------------------------------------- render */

async function render() {
  const host = $('#results');
  const isFacts = state.view === 'FACTS';
  $('#search').hidden = !isFacts;

  if (state.offset === 0) host.innerHTML = '';

  try {
    if (isFacts) {
      const params = new URLSearchParams({ limit: PAGE * 3, offset: state.offset });
      if (state.docId) params.set('doc_id', state.docId);
      if (state.search) params.set('search', state.search);
      const facts = await api(`/api/facts?${params}`);
      if (!facts.length && state.offset === 0) {
        host.append(el('p', 'empty', 'Nothing here yet. Upload a PDF to begin.'));
        $('#more').hidden = true;
        return;
      }
      facts.forEach((f) => host.append(factRow(f)));
      $('#more').hidden = facts.length < PAGE * 3;
    } else {
      const params = new URLSearchParams({
        verdict: state.view, limit: PAGE, offset: state.offset,
      });
      if (state.crossDoc) params.set('cross_document', 'true');
      const rels = await api(`/api/relations?${params}`);
      if (!rels.length && state.offset === 0) {
        host.append(el('p', 'empty',
          state.crossDoc
            ? 'No judgements of this kind across documents. Try turning that filter off.'
            : 'No judgements of this kind yet.'));
        $('#more').hidden = true;
        return;
      }
      rels.forEach((r) => host.append(judgementCard(r)));
      $('#more').hidden = rels.length < PAGE;
    }
  } catch (err) {
    host.append(el('p', 'empty', `Could not load results: ${err.message}`));
  }
}

/* ----------------------------------------------------------------- upload */

async function upload(file) {
  const job = $('#job');
  job.hidden = false;
  job.className = 'job';
  job.textContent = `reading ${file.name}…`;

  const body = new FormData();
  body.append('file', file);

  let res;
  try {
    res = await api('/api/documents', { method: 'POST', body });
  } catch (err) {
    job.className = 'job failed';
    job.textContent = `Upload failed: ${err.message}`;
    return;
  }

  if (res.status === 'already_ingested') {
    job.className = 'job done';
    job.textContent = 'Already in the knowledge layer — same content, so nothing to re-read.';
    return;
  }

  poll(res.job_id);
}

function poll(jobId) {
  const job = $('#job');
  const tick = async () => {
    let s;
    try {
      s = await api(`/api/jobs/${jobId}`);
    } catch {
      job.className = 'job failed';
      job.textContent = 'Lost track of that job.';
      return;
    }

    if (s.status === 'done') {
      const r = s.result;
      job.className = 'job done';
      job.textContent =
        `${r.filename}\n${r.facts} facts from ${r.windows} passages\n` +
        `${Math.round(r.grounding_rate * 100)}% grounded · ${r.flagged_ambiguous} flagged`;
      state.offset = 0;
      loadStats(); loadDocs(); render();
      return;
    }
    if (s.status === 'failed') {
      job.className = 'job failed';
      job.textContent = s.error;
      return;
    }
    job.textContent = `reading ${s.filename || ''}… this takes a few minutes for a large filing`;
    setTimeout(tick, 2000);
  };
  tick();
}

/* ------------------------------------------------------------------ wiring */

document.querySelectorAll('.tab[data-view]').forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll('.tab[data-view]')
      .forEach((t) => t.setAttribute('aria-pressed', String(t === tab)));
    state.view = tab.dataset.view;
    state.offset = 0;
    render();
  };
});

$('#crossdoc').onchange = (e) => {
  state.crossDoc = e.target.checked;
  state.offset = 0;
  render();
};

let searchTimer;
$('#search').oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.search = e.target.value.trim();
    state.offset = 0;
    render();
  }, 220);
};

$('#more').onclick = () => {
  state.offset += state.view === 'FACTS' ? PAGE * 3 : PAGE;
  render();
};

$('#file').onchange = (e) => {
  if (e.target.files[0]) upload(e.target.files[0]);
};

const dz = $('#dropzone');
['dragenter', 'dragover'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
dz.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});

loadStats();
loadDocs();
render();
