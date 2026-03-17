"""
ir_model.py
-----------
Implementasi Extended Boolean IR Model dengan P-Norm Scoring.

Formula P-Norm (p = 2):
  AND : score = √[ Σ tfᵢ²  / n ]
  OR  : score = 1 − √[ Σ (1−tfᵢ)² / n ]
  NOT : tf term yang di-NOT dibalik → tf = 1 − tf_asli, lalu hitung AND/OR

Referensi: Materi Pemodelan IR STKI & kode referensi teman.

Alur pencarian:
  1. Boolean retrieval  → saring dokumen yang relevan (nilai 1/0)
  2. Extended scoring   → beri skor 0.0–1.0 pada dokumen relevan
  3. Sort by skor desc  → kembalikan ranking
"""

import math
from preprocessing import load_corpus, preprocess
from indexing import build_vocabulary, build_incidence_matrix, build_inverted_index_full


# ══════════════════════════════════════════════════════════════════
#  BOBOT TF NORMALIZED
# ══════════════════════════════════════════════════════════════════
def compute_tf_normalized(corpus: dict) -> dict:
    """
    Hitung TF normalized untuk setiap (dokumen, term).
    
    tf_norm(t, d) = count(t, d) / total_token(d)
    
    Nilai antara 0.0 – 1.0.
    """
    tf_norm = {}
    for doc, tokens in corpus.items():
        total = len(tokens)
        tf_norm[doc] = {}
        for token in tokens:
            tf_norm[doc][token] = tf_norm[doc].get(token, 0) + 1
        for token in tf_norm[doc]:
            tf_norm[doc][token] = tf_norm[doc][token] / total
    return tf_norm


# ══════════════════════════════════════════════════════════════════
#  OPERASI BOOLEAN (untuk penyaringan awal)
# ══════════════════════════════════════════════════════════════════
def get_postings(term: str, corpus: dict) -> dict:
    """
    Ambil posting list biner untuk satu term.
    Term di-stem terlebih dahulu menggunakan preprocess().
    Return: {doc: 0/1}
    """
    stemmed = preprocess(term)
    if not stemmed:
        return {doc: 0 for doc in corpus}
    t = stemmed[0]
    return {doc: (1 if t in tokens else 0) for doc, tokens in corpus.items()}


def boolean_and(p1: dict, p2: dict) -> dict:
    return {doc: p1.get(doc, 0) & p2.get(doc, 0) for doc in p1}

def boolean_or(p1: dict, p2: dict) -> dict:
    return {doc: p1.get(doc, 0) | p2.get(doc, 0) for doc in p1}

def boolean_not(p1: dict) -> dict:
    return {doc: 1 - v for doc, v in p1.items()}


# ══════════════════════════════════════════════════════════════════
#  PARSER QUERY BOOLEAN  (Recursive Descent)
#  Mendukung: AND, OR, NOT, tanda kurung ()
#
#  Operator precedence (sesuai PPT):
#    () → NOT → AND → OR
# ══════════════════════════════════════════════════════════════════
def tokenize_query(query: str) -> list:
    """Pisahkan query menjadi list token."""
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    return [t for t in query.split() if t]


def parse_query(tokens: list, corpus: dict, pos: int = 0):
    result, pos = parse_or(tokens, corpus, pos)
    return result, pos

def parse_or(tokens, corpus, pos):
    left, pos = parse_and(tokens, corpus, pos)
    while pos < len(tokens) and tokens[pos].upper() == 'OR':
        pos += 1
        right, pos = parse_and(tokens, corpus, pos)
        left = boolean_or(left, right)
    return left, pos

def parse_and(tokens, corpus, pos):
    left, pos = parse_not(tokens, corpus, pos)
    while pos < len(tokens) and tokens[pos].upper() == 'AND' \
          and (pos + 1 >= len(tokens) or tokens[pos + 1].upper() != 'NOT'):
        pos += 1
        right, pos = parse_not(tokens, corpus, pos)
        left = boolean_and(left, right)
    return left, pos

def parse_not(tokens, corpus, pos):
    if pos < len(tokens) and tokens[pos].upper() == 'NOT':
        pos += 1
        operand, pos = parse_primary(tokens, corpus, pos)
        return boolean_not(operand), pos
    result, pos = parse_primary(tokens, corpus, pos)
    # Handle pola: term AND NOT term
    while pos + 1 < len(tokens) \
          and tokens[pos].upper() == 'AND' \
          and tokens[pos + 1].upper() == 'NOT':
        pos += 2
        right, pos = parse_primary(tokens, corpus, pos)
        result = boolean_and(result, boolean_not(right))
    return result, pos

def parse_primary(tokens, corpus, pos):
    if pos >= len(tokens):
        return {doc: 0 for doc in corpus}, pos
    token = tokens[pos]
    if token == '(':
        pos += 1
        result, pos = parse_or(tokens, corpus, pos)
        if pos < len(tokens) and tokens[pos] == ')':
            pos += 1
        return result, pos
    elif token.upper() not in ('AND', 'OR', 'NOT', '(', ')'):
        pos += 1
        return get_postings(token, corpus), pos
    else:
        return {doc: 0 for doc in corpus}, pos


# ══════════════════════════════════════════════════════════════════
#  EXTENDED BOOLEAN SCORING  (P-Norm, p=2)
# ══════════════════════════════════════════════════════════════════
def extended_boolean_score(terms: list, ops: list, tf_norm: dict, doc: str) -> float:
    """
    Hitung skor Extended Boolean untuk satu dokumen.

    Formula P-Norm (p=2):
      AND dominant : skor = √[ Σ tfᵢ²  / n ]
      OR  dominant : skor = 1 − √[ Σ (1−tfᵢ)² / n ]

    NOT : tf term yang ber-operator NOT dibalik → tf = 1 − tf_asli
    sebelum dimasukkan ke rumus AND/OR.

    Catatan:
      - Untuk AND agar relevan, SEMUA tf harus tinggi → pakai rumus biasa
      - Untuk OR  agar relevan, CUKUP SATU tf tinggi → pakai rumus komplemen
    """
    if not terms:
        return 0.0

    values = []
    for i, term in enumerate(terms):
        tf = tf_norm[doc].get(term, 0.0)
        op = ops[i] if i < len(ops) else None
        if op == 'NOT':
            tf = 1.0 - tf          # balik bobot untuk NOT
        values.append(tf)

    n = len(values)

    # Tentukan operator dominan
    op_dominan = 'OR' if 'OR' in [o for o in ops if o] else 'AND'

    if op_dominan == 'AND':
        # AND: geometrik rata-rata tf → skor tinggi jika SEMUA tf tinggi
        skor = math.sqrt(sum(v ** 2 for v in values) / n)
    else:
        # OR: skor tinggi jika MINIMAL SATU tf tinggi
        skor = 1.0 - math.sqrt(sum((1.0 - v) ** 2 for v in values) / n)

    return round(skor, 4)


# ══════════════════════════════════════════════════════════════════
#  EKSTRAK TERMS DAN OPS DARI QUERY
# ══════════════════════════════════════════════════════════════════
def extract_terms_and_ops(query: str) -> tuple:
    """
    Ambil list term (sudah di-stem) dan list operator NOT/OR dari query.
    Digunakan untuk Extended Boolean scoring setelah Boolean filtering.

    Contoh:
        "bukti AND kasus AND NOT korban"
        → terms: ['bukti', 'kasus', 'korban']
           ops  : [None,   None,    'NOT']
    """
    tokens = query.upper().split()
    terms, ops = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i].strip('()')
        if t == 'NOT' and i + 1 < len(tokens):
            next_term = tokens[i + 1].strip('()').lower()
            stemmed = preprocess(next_term)
            if stemmed:
                terms.append(stemmed[0])
                ops.append('NOT')
            i += 2
        elif t == 'OR':
            if ops:
                ops.append('OR')
            i += 1
        elif t not in ('AND', 'OR', 'NOT', '', '(', ')'):
            stemmed = preprocess(t.lower())
            if stemmed:
                terms.append(stemmed[0])
                ops.append(None)
            i += 1
        else:
            i += 1
    return terms, ops


# ══════════════════════════════════════════════════════════════════
#  FUNGSI PENCARIAN UTAMA
# ══════════════════════════════════════════════════════════════════
def search(query_input: str, corpus: dict, tf_norm: dict):
    """
    Jalankan pencarian dua tahap:
      1. Boolean → saring dokumen relevan (0/1)
      2. Extended Boolean → beri skor, urutkan dari tertinggi

    Return: (terms, ops, [(doc, skor), ...])
    atau None jika query tidak valid.
    """
    if not query_input.strip():
        return None

    # ── Tahap 1: Boolean Retrieval ──────────────────────────────
    tokens_query = tokenize_query(query_input)
    try:
        postings, _ = parse_query(tokens_query, corpus)
    except Exception:
        return None

    dokumen_relevan = [doc for doc, val in postings.items() if val == 1]
    terms, ops = extract_terms_and_ops(query_input)

    if not dokumen_relevan:
        return terms, ops, []

    # ── Tahap 2: Extended Boolean Scoring ───────────────────────
    hasil = []
    for doc in dokumen_relevan:
        skor = extended_boolean_score(terms, ops, tf_norm, doc)
        hasil.append((doc, skor))

    hasil = sorted(hasil, key=lambda x: x[1], reverse=True)
    return terms, ops, hasil


# ── Uji mandiri ──────────────────────────────────────────────────
if __name__ == "__main__":
    corpus  = load_corpus("corpus")
    tf_norm = compute_tf_normalized(corpus)

    queries = [
        "bukti AND kasus",
        "bukti AND kasus AND NOT korban",
        "(bukti AND kasus) AND NOT korban",
        "saksi OR korban",
        "ardan AND NOT bunuh",
        "curi AND NOT (bakar OR bunuh)",
    ]

    print("=" * 60)
    print("  DEMO PENCARIAN EXTENDED BOOLEAN")
    print("=" * 60)

    for q in queries:
        hasil = search(q, corpus, tf_norm)
        if hasil:
            terms, ops, docs = hasil
            print(f"\nQuery : {q}")
            print(f"Terms : {terms}  |  Ops: {ops}")
            print(f"Hasil :")
            for doc, skor in docs:
                print(f"  {doc} → skor: {skor}")
        else:
            print(f"\nQuery : {q} → tidak valid")
