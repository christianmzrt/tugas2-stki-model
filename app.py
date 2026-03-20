import os
import streamlit as st
import pandas as pd
from preprocessing import load_corpus
from indexing import build_vocabulary, build_incidence_matrix, build_inverted_index_full
from ir_model import compute_tf_normalized, search

st.set_page_config(
    page_title="STKI — Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

:root {
    --bg: #0f1115;
    --card: #16191f;
    --accent: #3b82f6;
    --text-main: #e2e8f0;
    --text-dim: #94a3b8;
    --border: rgba(255,255,255,0.08);
    --cyan: #00e5ff;
    --violet: #a78bfa;
    --rose: #ff6b9d;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: var(--bg) !important; color: var(--text-main); }

/* ── Sembunyikan label & toggle sidebar ── */
.stTextInput label { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Hero area ── */
.search-container {
    max-width: 800px;
    margin: 2rem auto 1.5rem auto;
    text-align: center;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: #475569;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    margin-bottom: 1.6rem;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 100px !important;
    padding: 14px 25px !important;
    font-size: 0.95rem !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(59,130,246,0.4) !important;
    box-shadow: 0 4px 28px rgba(59,130,246,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #374151 !important; }

/* ── Stats ribbon ── */
.stats-ribbon {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
}

/* ── Contoh query chips ── */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    border-radius: 100px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.02) !important;
    color: var(--text-dim) !important;
    padding: 5px 14px !important;
    font-weight: 400 !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: rgba(59,130,246,0.3) !important;
    color: #60a5fa !important;
    background: rgba(59,130,246,0.05) !important;
}

/* ── Tombol CARI ── */
.btn-cari > div > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 100px !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.04em !important;
    padding: 12px 10px !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.25) !important;
    transition: all 0.2s !important;
}
.btn-cari > div > button:hover {
    box-shadow: 0 6px 28px rgba(59,130,246,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Meta info hasil ── */
.result-meta {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.meta-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 5px;
    padding: 2px 10px;
    color: var(--text-dim);
}
.meta-pill-blue {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 5px;
    padding: 2px 10px;
    color: #60a5fa;
    font-weight: 600;
}
.meta-pill-violet {
    background: rgba(167,139,250,0.07);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 5px;
    padding: 2px 10px;
    color: var(--violet);
}

/* ── Kartu hasil ── */
.res-card {
    background: var(--card);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border: 1px solid var(--border);
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
    position: relative;
    transition: border-color 0.2s, transform 0.2s;
}
.res-card::before {
    content: '';
    position: absolute;
    top: 0; left: 28px; right: 28px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.15), transparent);
}
.res-card:hover {
    border-color: rgba(59,130,246,0.25);
    transform: translateY(-2px);
}
.res-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 6px;
}
.res-docid {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #60a5fa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.res-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.57rem;
    color: #374151;
}
.res-title {
    color: #93c5fd;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 8px;
    line-height: 1.4;
    letter-spacing: -0.01em;
    display: block;
}
.res-snippet {
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.75;
    margin-bottom: 14px;
    font-weight: 300;
}
.res-score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.res-score-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #374151;
    text-transform: uppercase;
    min-width: 64px;
}
.res-bar-bg {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    overflow: hidden;
}
.res-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.res-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #60a5fa;
    font-weight: 600;
    min-width: 36px;
    text-align: right;
}

/* ── Chip term ── */
.chip-hit {
    display: inline-block;
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.2);
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    padding: 2px 9px;
    border-radius: 5px;
    margin: 2px;
}
.chip-miss {
    display: inline-block;
    background: rgba(255,107,157,0.06);
    border: 1px solid rgba(255,107,157,0.18);
    color: var(--rose);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    padding: 2px 9px;
    border-radius: 5px;
    margin: 2px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 0;
    color: #374151;
}
.empty-icon { font-size: 2.8rem; margin-bottom: 12px; opacity: 0.3; }
.empty-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #374151;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center !important;
    background: #16191f !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-dim) !important;
    border-radius: 8px !important;
    padding: 7px 24px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.1) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
}

/* ── Misc ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }
.stCaption { font-family: 'JetBrains Mono', monospace !important; font-size: 0.62rem !important; color: var(--text-dim) !important; }
.stAlert { background: rgba(255,255,255,0.02) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    corpus     = load_corpus("corpus")
    vocab      = build_vocabulary(corpus)
    inc_matrix = build_incidence_matrix(corpus, vocab)
    inv_idx    = build_inverted_index_full(corpus)
    tf_norm    = compute_tf_normalized(corpus)
    return corpus, vocab, inc_matrix, inv_idx, tf_norm

corpus, vocab, inc_matrix, inv_idx, tf_norm = load_all()


# ══════════════════════════════════════════════════════════════════
#  HERO + SEARCH BAR (terpusat)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="search-container">
    <div class="hero-title">STKI Search</div>
    <div class="hero-sub">EXTENDED BOOLEAN MODEL &nbsp;·&nbsp; INFORMATION RETRIEVAL SYSTEM</div>
</div>
""", unsafe_allow_html=True)

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    query_input = st.text_input(
        "",
        placeholder="Ketik query... contoh: jaringan AND saraf AND NOT robot",
        label_visibility="collapsed",
        key="query_main"
    )
    st.markdown(f"""
    <div class="stats-ribbon">
        <span>📂 {len(corpus)} Dokumen</span>
        <span>🔑 {len(vocab)} Term Unik</span>
        <span>⚙️ Extended Boolean</span>
    </div>
    """, unsafe_allow_html=True)

# ── Tombol CARI (terpusat) ─────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    st.markdown('<div class="btn-cari">', unsafe_allow_html=True)
    tombol_cari = st.button("🔍  Cari Dokumen", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Contoh Query (chip horizontal) ────────────────────────────
st.markdown("<div style='text-align:center; margin: 0.6rem 0 1.4rem; color:#374151; font-family:JetBrains Mono,monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.12em;'>Contoh Query</div>", unsafe_allow_html=True)

contoh_list = [
    "pembelajaran AND data",
    "jaringan AND saraf AND NOT robot",
    "bahasa OR visi",
    "kecerdasan AND NOT privasi",
    "(algoritma AND model) AND NOT etika",
]
chip_cols = st.columns(len(contoh_list))
for i, c in enumerate(contoh_list):
    with chip_cols[i]:
        if st.button(c, key=f"chip_{i}", use_container_width=True):
            query_input = c
            tombol_cari = True

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HASIL PENCARIAN
# ══════════════════════════════════════════════════════════════════
if tombol_cari and query_input:
    hasil = search(query_input, corpus, tf_norm)

    if hasil is None:
        st.warning("⚠️ Sintaks query salah. Gunakan AND, OR, atau NOT.")
    else:
        terms, ops, dokumen = hasil

        # Meta bar
        term_pills = "".join(
            f'<span class="meta-pill-violet">{t}</span>' for t in terms
        )
        st.markdown(f"""
        <div class="result-meta">
            <span class="meta-pill-blue">{len(dokumen)} hasil ditemukan</span>
            {term_pills}
            <span class="meta-pill">{query_input}</span>
        </div>
        """, unsafe_allow_html=True)

        if not dokumen:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <div class="empty-txt">Tidak ada dokumen yang cocok</div>
            </div>""", unsafe_allow_html=True)
        else:
            badges = ["🥇", "🥈", "🥉"]
            for rank, (doc, skor) in enumerate(dokumen, start=1):
                path = os.path.join("corpus", doc)
                with open(path, 'r', encoding='utf-8') as f:
                    isi = f.read().strip()

                tok      = corpus[doc]
                snippet  = isi[:200].replace('\n', ' ') + "..."
                pct      = int(skor * 100)
                nama_doc = doc.replace('.txt', '').upper()
                badge    = badges[rank - 1] if rank <= 3 else f"#{rank}"

                # Chip status term
                chips = ""
                for ti, term in enumerate(terms):
                    frek = tok.count(term)
                    op   = ops[ti] if ti < len(ops) else None
                    if op == 'NOT':
                        chips += (
                            f'<span class="chip-hit">✓ NOT {term}</span>'
                            if frek == 0 else
                            f'<span class="chip-miss">✗ NOT {term} ({frek}×)</span>'
                        )
                    else:
                        chips += (
                            f'<span class="chip-hit">✓ {term} ({frek}×)</span>'
                            if frek > 0 else
                            f'<span class="chip-miss">✗ {term}</span>'
                        )

                st.markdown(f"""
                <div class="res-card">
                    <div class="res-top">
                        <span class="res-docid">corpus / {doc}</span>
                        <span class="res-rank">{badge} rank #{rank}</span>
                    </div>
                    <span class="res-title">{isi[:65].rstrip()}…</span>
                    <div class="res-snippet">{snippet}</div>
                    <div class="res-score-row">
                        <span class="res-score-lbl">relevance</span>
                        <div class="res-bar-bg">
                            <div class="res-bar-fill" style="width:{pct}%"></div>
                        </div>
                        <span class="res-score-num">{skor}</span>
                    </div>
                    <div>{chips}</div>
                </div>
                """, unsafe_allow_html=True)

elif tombol_cari and not query_input:
    st.warning("⚠️ Masukkan query terlebih dahulu.")
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚡</div>
        <div class="empty-txt">Masukkan query di atas untuk memulai pencarian</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TABS BAWAH — Incidence Matrix & Inverted Index
# ══════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["  📊  Incidence Matrix  ", "  📋  Inverted Index  "])

with tab1:
    st.caption(f"{inc_matrix.shape[0]} term  ×  {inc_matrix.shape[1]} dokumen  —  1 = ada,  0 = tidak ada")
    filter_m = st.text_input("Filter term:", placeholder="cari term...", key="fm")
    filtered = (
        inc_matrix[inc_matrix.index.str.contains(filter_m, case=False)]
        if filter_m else inc_matrix
    )
    styled = filtered.style.map(
        lambda v: (
            'background:#0d1a2d;color:#60a5fa;font-weight:600;text-align:center;'
            if v == 1 else
            'background:#0f1115;color:#1e2d40;text-align:center;'
        )
    )
    st.dataframe(styled, use_container_width=True, height=400)

with tab2:
    st.caption(f"{len(inv_idx)} term unik  —  format: <dok, frekuensi, [posisi]>")
    filter_i = st.text_input("Filter term:", placeholder="cari term...", key="fi")
    rows = []
    for term in sorted(inv_idx.keys()):
        entries = inv_idx[term]
        fmt = [
            f"<{d.replace('.txt','')}, {i['frekuensi']}, {i['posisi']}>"
            for d, i in entries.items()
        ]
        rows.append({
            "Term"         : term,
            "Inverted List": "  |  ".join(fmt),
            "df"           : len(entries)
        })
    df_inv = pd.DataFrame(rows)
    if filter_i:
        df_inv = df_inv[df_inv["Term"].str.contains(filter_i, case=False)]
    st.dataframe(df_inv, use_container_width=True, height=400)
