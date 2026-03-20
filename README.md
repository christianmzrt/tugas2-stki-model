# STKI Search — Information Retrieval System

## Identitas

| | |
|---|---|
| **Nama** | Richard Christian Mozart Diazoni |
| **NIM** | 2405551019 |
| **Kelas** | Sistem Temu Kembali Informasi D |
| **Dosen** | Dr. I Made Suwija Putra, S.T., M.T. |

---

## Deskripsi Tugas

Tugas ini merupakan implementasi Sistem Temu Kembali Informasi (*Information Retrieval System*) berbasis **Extended Boolean Model** menggunakan bahasa pemrograman Python dengan antarmuka Streamlit.

Sistem menerima query dari pengguna dalam bentuk ekspresi boolean (AND, OR, NOT) kemudian mencocokkannya terhadap koleksi dokumen teks yang telah diindeks. Berbeda dari Boolean klasik yang menghasilkan nilai biner, sistem ini menggunakan Extended Boolean Model sehingga setiap dokumen mendapatkan skor relevansi antara 0.0 hingga 1.0 dan dapat diurutkan berdasarkan derajat kesesuaiannya terhadap query.

Sistem mencakup tiga komponen utama:

- **Preprocessing** — memuat dan memproses seluruh dokumen dalam korpus menjadi token
- **Indexing** — membangun Incidence Matrix dan Inverted Index sebagai struktur indeks pencarian
- **IR Model** — menghitung skor relevansi dan mengembalikan hasil pencarian yang terurut

Antarmuka aplikasi menampilkan panel pencarian, hasil dokumen beserta skor relevansinya, visualisasi Incidence Matrix, serta Inverted Index lengkap dengan frekuensi dan posisi kemunculan setiap term.

## Struktur File
 
```
stki-ir-system/
│
├── corpus/
├── app.py
├── indexing.py
├── ir_model.py
├── preprocessing.py
├── requirements.txt
└── PengertianFlat_StructureGuide_HyperText...
