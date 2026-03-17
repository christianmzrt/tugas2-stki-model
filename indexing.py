"""
indexing.py
-----------
Membangun dua struktur data indeks IR:

1. Incidence Matrix (Term-Document Matrix)
   → Tabel biner: 1 jika term ada di dokumen, 0 jika tidak.
   → Dikembalikan sebagai pandas DataFrame (term × dokumen).

2. Inverted Index Lengkap
   → Format: {term: {doc: {frekuensi: int, posisi: [int]}}}
   → Notasi tampilan: <doc, frekuensi, [posisi]>
   → Posisi dimulai dari 0 (sesuai enumerate Python).
"""

import pandas as pd
from preprocessing import load_corpus


# ══════════════════════════════════════════════════════════════════
#  1. VOCABULARY
# ══════════════════════════════════════════════════════════════════
def build_vocabulary(corpus: dict) -> list:
    """
    Kumpulkan semua term unik dari seluruh dokumen, kembalikan
    dalam bentuk list yang sudah diurutkan secara alfabetis.
    """
    vocabulary = set()
    for tokens in corpus.values():
        for token in tokens:
            vocabulary.add(token)
    return sorted(vocabulary)


# ══════════════════════════════════════════════════════════════════
#  2. INCIDENCE MATRIX
# ══════════════════════════════════════════════════════════════════
def build_incidence_matrix(corpus: dict, vocabulary: list) -> pd.DataFrame:
    """
    Bangun Incidence Matrix (TFbiner):
      - Baris  = term (vocabulary)
      - Kolom  = nama dokumen
      - Nilai  = 1 jika term ADA di dokumen, 0 jika TIDAK

    Dikembalikan sebagai pandas DataFrame agar mudah ditampilkan
    di Streamlit.
    """
    nama_dokumen = list(corpus.keys())

    matrix = {}
    for term in vocabulary:
        baris = []
        for doc in nama_dokumen:
            if term in corpus[doc]:
                baris.append(1)
            else:
                baris.append(0)
        matrix[term] = baris

    # index = nama dokumen (baris), columns = term
    df = pd.DataFrame(matrix, index=nama_dokumen).T
    return df


# ══════════════════════════════════════════════════════════════════
#  3. INVERTED INDEX LENGKAP
# ══════════════════════════════════════════════════════════════════
def build_inverted_index_full(corpus: dict) -> dict:
    """
    Bangun Inverted Index lengkap dengan frekuensi dan posisi.

    Struktur output:
        {
          'bukti': {
              'doc1.txt': {'frekuensi': 3, 'posisi': [2, 8, 14]},
              'doc4.txt': {'frekuensi': 1, 'posisi': [5]},
          },
          ...
        }

    Posisi dimulai dari 0 (indeks token dalam list hasil preprocess).
    Notasi sesuai PPT: <idj, fij, [O1, O2, ..., Ok]>
    """
    inverted_index = {}

    for nama_doc, tokens in corpus.items():
        for posisi, token in enumerate(tokens):
            # Pastikan entri term ada
            if token not in inverted_index:
                inverted_index[token] = {}

            # Pastikan entri dokumen ada
            if nama_doc not in inverted_index[token]:
                inverted_index[token][nama_doc] = {
                    'frekuensi': 0,
                    'posisi': []
                }

            inverted_index[token][nama_doc]['frekuensi'] += 1
            inverted_index[token][nama_doc]['posisi'].append(posisi)

    return inverted_index


# ══════════════════════════════════════════════════════════════════
#  4. TAMPIL DI TERMINAL (untuk uji mandiri)
# ══════════════════════════════════════════════════════════════════
def format_inverted_index_table(corpus: dict):
    """Cetak Inverted Index ke terminal dengan format tabel."""
    inv_idx = build_inverted_index_full(corpus)

    print("\n=== INVERTED INDEX ===")
    print(f"{'Term':<20} {'Inverted List'}")
    print("-" * 80)

    for term in sorted(inv_idx.keys()):
        entries = inv_idx[term]
        formatted = []
        for doc, info in entries.items():
            nama = doc.replace('.txt', '')
            frek = info['frekuensi']
            pos  = info['posisi']
            formatted.append(f"<{nama},{frek},{pos}>")
        inverted_list = ", ".join(formatted)
        print(f"{term:<20} {inverted_list}")


# ── Uji mandiri ──────────────────────────────────────────────────
if __name__ == "__main__":
    folder_corpus = "corpus"
    corpus = load_corpus(folder_corpus)

    vocab = build_vocabulary(corpus)
    print(f"Total kata unik (vocabulary): {len(vocab)}")
    print(f"Contoh 10 kata pertama      : {vocab[:10]}")

    incidence_matrix = build_incidence_matrix(corpus, vocab)
    print(f"\nUkuran matrix : {incidence_matrix.shape[0]} term × "
          f"{incidence_matrix.shape[1]} dokumen")
    print("\n=== INCIDENCE MATRIX ===")
    print(incidence_matrix)

    format_inverted_index_table(corpus)
