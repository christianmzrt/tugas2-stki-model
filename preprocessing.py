"""
preprocessing.py
----------------
Melakukan pre-processing teks corpus:
  1. Case folding (huruf kecil)
  2. Hapus karakter non-alfabet
  3. Tokenisasi (split by whitespace)
  4. Stemming menggunakan PySastrawi

TANPA stop words removal (sesuai instruksi tugas & struktur teman).

Corpus dibaca dari folder berisi file .txt.
"""

import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Sastrawi sekali saja (supaya efisien)
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def preprocess(teks: str) -> list:
    """
    Memproses satu teks dan mengembalikan list token yang sudah di-stem.

    Langkah:
      1. Ubah ke huruf kecil (case folding)
      2. Hapus semua karakter selain huruf a-z dan spasi
      3. Pisah menjadi token (split)
      4. Stem setiap token dengan Sastrawi

    Contoh:
        preprocess("Algoritma Genetika digunakan untuk Optimasi")
        → ['algoritm', 'genetik', 'guna', 'untuk', 'optimas']
    """
    # 1. Case folding
    teks = teks.lower()

    # 2. Hapus karakter non-alfabet (angka, tanda baca, dll.)
    teks = re.sub(r'[^a-z\s]', '', teks)

    # 3. Tokenisasi
    tokens = teks.split()

    # 4. Stemming
    tokens_stem = [stemmer.stem(token) for token in tokens]

    return tokens_stem


def load_corpus(folder_path: str) -> dict:
    """
    Membaca semua file .txt dari folder_path, memproses setiap file,
    dan mengembalikan dict {nama_file: [token, token, ...]}.

    Contoh struktur corpus/:
        corpus/doc1.txt
        corpus/doc2.txt
        ...
    """
    corpus = {}

    for nama_file in sorted(os.listdir(folder_path)):
        if nama_file.endswith('.txt'):
            path = os.path.join(folder_path, nama_file)

            with open(path, 'r', encoding='utf-8') as f:
                isi = f.read()

            hasil_preprocessing = preprocess(isi)
            corpus[nama_file] = hasil_preprocessing

    return corpus


# ── Uji mandiri ──────────────────────────────────────────────────
if __name__ == "__main__":
    folder_corpus = "corpus"

    print("=" * 55)
    print("  HASIL PRE-PROCESSING CORPUS")
    print("=" * 55)

    corpus = load_corpus(folder_corpus)

    for nama_doc, tokens in corpus.items():
        print(f"\n{'─'*50}")
        print(f"Dokumen : {nama_doc}")
        print(f"Jumlah token : {len(tokens)}")
        print(f"Tokens  : {tokens}")
