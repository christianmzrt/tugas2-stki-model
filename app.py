"""
app.py — STKI IR System
Layout: Two-column (panel kiri kontrol, panel kanan hasil) + tabs bawah
"""

import os
import streamlit as st
import pandas as pd
from preprocessing import load_corpus
from indexing import build_vocabulary, build_incidence_matrix, build_inverted_index_full
from ir_model import compute_tf_normalized, search

st.set_page_config(
    page_title="STKI — IR System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

:root {
    --bg:        #07071a;
    --panel:     #0b0b22;
    --surface:   rgba(255,255,255,0.03);
    --border:    rgba(255,255,255,0.07);
    --cyan:      #00e5ff;
    --violet:    #a78bfa;
    --rose:      #ff6b9d;
    --t1:        #f4f7ff;
    --t2:        #b0b8d0;
    --t3:        #5a6285;
    --grad:      linear-gradient(135deg, #00e5ff, #a78bfa, #ff6b9d);
}

/* ── Base ── */
.stApp { background: var(--bg) !important; }
.main .block-container {
    padding: 1.6rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sembunyikan sidebar toggle ── */
[data-testid="collapsedControl"] { display: none !important; }

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--t1);
}
.page-title span {
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--cyan);
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 5px;
    padding: 3px 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.85;
}
.page-meta {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--t2);
    letter-spacing: 0.06em;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 0.85rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--t1) !important;
    caret-color: var(--cyan) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,229,255,0.3) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.06) !important;
    background: rgba(0,229,255,0.02) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--t3) !important; }
.stTextInput label { display: none !important; }

/* ── Semua tombol default (chip query) ── */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.02) !important;
    color: var(--t1) !important;
    padding: 7px 10px !important;
    font-weight: 400 !important;
    transition: all 0.16s ease !important;
    width: 100% !important;
    text-align: left !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    border-color: rgba(0,229,255,0.25) !important;
    color: var(--cyan) !important;
    background: rgba(0,229,255,0.04) !important;
}

/* ── Tombol CARI ── */
.btn-cari > div > button {
    background: rgba(0,229,255,0.06) !important;
    border: 1.5px solid rgba(0,229,255,0.3) !important;
    border-radius: 10px !important;
    color: var(--cyan) !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.06em !important;
    padding: 11px 10px !important;
    text-shadow: 0 0 10px rgba(0,229,255,0.4) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.07) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.btn-cari > div > button:hover {
    background: rgba(0,229,255,0.1) !important;
    box-shadow: 0 0 24px rgba(0,229,255,0.18) !important;
    border-color: var(--cyan) !important;
}

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.56rem;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 9px;
    margin-top: 16px;
    display: block;
}
.sec-label:first-child { margin-top: 0; }

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 18px;
}
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--violet);
    display: block;
    line-height: 1;
    margin-bottom: 5px;
}
.stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.52rem;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Result header ── */
.result-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}
.result-hdr-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--t1);
}
.result-count-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--cyan);
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 5px;
    padding: 2px 8px;
}
.term-tag {
    background: rgba(167,139,250,0.07);
    border: 1px solid rgba(167,139,250,0.18);
    color: var(--violet);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    padding: 2px 9px;
    border-radius: 5px;
    margin: 0 1px;
}
.query-tag {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--t1);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 2px 10px;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Result cards ── */
.result-card {
    background: rgba(255,255,255,0.018);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    position: relative;
    transition: border-color 0.2s, background 0.2s;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 24px; right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.18), transparent);
}
.result-card:hover {
    border-color: rgba(167,139,250,0.2);
    background: rgba(167,139,250,0.025);
}
.rc-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.rc-docid {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--violet);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.rc-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.57rem;
    color: var(--t2);
}
.rc-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--t1);
    margin-bottom: 7px;
    line-height: 1.4;
    letter-spacing: -0.01em;
}
.rc-snippet {
    font-size: 0.78rem;
    color: var(--t2);
    line-height: 1.75;
    margin-bottom: 12px;
    font-weight: 300;
}
.rc-score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.rc-score-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--t2);
    text-transform: uppercase;
    min-width: 60px;
}
.rc-bar-bg {
    flex: 1;
    height: 2px;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    overflow: hidden;
}
.rc-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #00e5ff, #a78bfa);
}
.rc-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--violet);
    font-weight: 600;
    min-width: 32px;
    text-align: right;
}
.chip-hit {
    display: inline-block;
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.18);
    color: var(--cyan);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.57rem; padding: 2px 9px;
    border-radius: 5px; margin: 2px;
}
.chip-miss {
    display: inline-block;
    background: rgba(255,107,157,0.05);
    border: 1px solid rgba(255,107,157,0.18);
    color: var(--rose);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.57rem; padding: 2px 9px;
    border-radius: 5px; margin: 2px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 0;
}
.empty-icon { font-size: 2.4rem; opacity: 0.25; margin-bottom: 14px; }
.empty-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.18em;
}

/* ── Bottom tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--t1) !important;
    border-radius: 8px !important;
    padding: 7px 22px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.07) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(0,229,255,0.18) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 14px 0 !important; }

/* ── Caption ── */
.stCaption {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--t2) !important;
}

/* ── Alert ── */
.stAlert {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Container border ── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: var(--panel) !important;
    padding: 20px !important;
}
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
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-header">
    <div class="page-title">STKI <span>Search</span></div>
    <div class="page-badge">Extended Boolean Model</div>
    <div class="page-meta">{len(corpus)} dok &nbsp;·&nbsp; {len(vocab)} term unik</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LAYOUT DUA KOLOM
# ══════════════════════════════════════════════════════════════════
col_kiri, col_kanan = st.columns([1, 2.4], gap="medium")

# ─────────────────────────────────────────────────────────────────
#  PANEL KIRI
# ─────────────────────────────────────────────────────────────────
with col_kiri:
    with st.container(border=True):

        st.markdown('<span class="sec-label">Query Pencarian</span>', unsafe_allow_html=True)
        query_input = st.text_input(
            label="query",
            placeholder="jaringan AND saraf AND NOT robot",
            key="query_main",
            label_visibility="collapsed"
        )
        st.markdown('<div class="btn-cari">', unsafe_allow_html=True)
        tombol_cari = st.button("⚡  Cari Dokumen", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown('<span class="sec-label">Contoh Query</span>', unsafe_allow_html=True)
        contoh_list = [
            "pembelajaran AND data",
            "jaringan AND saraf AND NOT robot",
            "bahasa OR visi",
            "kecerdasan AND NOT privasi",
            "(algoritma AND model) AND NOT etika",
        ]
        for c in contoh_list:
            if st.button(c, key=f"q_{c}", use_container_width=True):
                query_input = c
                tombol_cari = True

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">
                <span class="stat-val">{len(corpus)}</span>
                <span class="stat-lbl">Dokumen</span>
            </div>
            <div class="stat-pill">
                <span class="stat-val">{len(vocab)}</span>
                <span class="stat-lbl">Term</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PANEL KANAN — HASIL
# ─────────────────────────────────────────────────────────────────
with col_kanan:
    with st.container(border=True):

        if tombol_cari and query_input:
            hasil = search(query_input, corpus, tf_norm)

            if hasil is None:
                st.warning("⚠️ Query tidak valid. Periksa sintaks AND / OR / NOT.")
            else:
                terms, ops, dokumen = hasil

                term_tags = "".join(
                    f'<span class="term-tag">{t}</span>' for t in terms
                )
                st.markdown(f"""
                <div class="result-header">
                    <span class="result-hdr-title">Hasil Pencarian</span>
                    <span class="result-count-badge">{len(dokumen)} dokumen</span>
                    {term_tags}
                    <span class="query-tag">{query_input}</span>
                </div>
                """, unsafe_allow_html=True)

                if not dokumen:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-icon">🔍</div>
                        <div class="empty-txt">Tidak ada dokumen relevan</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    badges = ["🥇", "🥈", "🥉"]
                    for rank, (doc, skor) in enumerate(dokumen, start=1):
                        path = os.path.join("corpus", doc)
                        with open(path, 'r', encoding='utf-8') as f:
                            isi = f.read().strip()

                        tok      = corpus[doc]
                        snippet  = isi[:180].replace('\n', ' ') + "..."
                        pct      = int(skor * 100)
                        nama_doc = doc.replace('.txt', '').upper()
                        badge    = badges[rank - 1] if rank <= 3 else f"#{rank}"

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
                        <div class="result-card">
                            <div class="rc-top">
                                <div class="rc-docid">{nama_doc}</div>
                                <div class="rc-rank">{badge} rank #{rank}</div>
                            </div>
                            <div class="rc-title">{isi[:65].rstrip()}…</div>
                            <div class="rc-snippet">{snippet}</div>
                            <div class="rc-score-row">
                                <span class="rc-score-lbl">relevance</span>
                                <div class="rc-bar-bg">
                                    <div class="rc-bar-fill" style="width:{pct}%"></div>
                                </div>
                                <span class="rc-score-num">{skor}</span>
                            </div>
                            <div>{chips}</div>
                        </div>
                        """, unsafe_allow_html=True)

        elif tombol_cari and not query_input:
            st.warning("⚠️ Masukkan query terlebih dahulu.")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🧠</div>
                <div class="empty-txt">Masukkan query untuk memulai pencarian</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TABS BAWAH — Incidence Matrix & Inverted Index
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  📊  Incidence Matrix  ", "  📋  Inverted Index  "])

with tab1:
    st.caption(f"{inc_matrix.shape[0]} term  ×  {inc_matrix.shape[1]} dokumen  —  1 = ada,  0 = tidak ada")
    filter_m = st.text_input("Filter term:", placeholder="cari term...", key="fm")
    filtered = inc_matrix[inc_matrix.index.str.contains(filter_m, case=False)] if filter_m else inc_matrix
    styled = filtered.style.map(
        lambda v: (
            'background:#0d1f2d;color:#00e5ff;font-weight:600;text-align:center;'
            if v == 1 else
            'background:#07071a;color:#1e2235;text-align:center;'
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
        rows.append({"Term": term, "Inverted List": "  |  ".join(fmt), "df": len(entries)})
    df_inv = pd.DataFrame(rows)
    if filter_i:
        df_inv = df_inv[df_inv["Term"].str.contains(filter_i, case=False)]
    st.dataframe(df_inv, use_container_width=True, height=400)