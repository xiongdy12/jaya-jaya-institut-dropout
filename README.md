# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

- **Nama:** *Tegar Arifin Prasetyo*
- **Email:** *arifintegar12@gmail.com*
- **ID Dicoding:** *tegar_arifin_umpr*

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan
memiliki reputasi baik dalam mencetak lulusan berkualitas. Namun di balik reputasi tersebut,
institusi ini menghadapi persoalan serius: **32,1% dari 4.424 mahasiswa berstatus dropout** —
hampir sepertiga dari total populasi mahasiswa.

Angka dropout setinggi ini merugikan kedua belah pihak. Mahasiswa kehilangan waktu, biaya, dan
peluang karier tanpa memperoleh gelar. Sementara institusi kehilangan pendapatan operasional,
mengalami penurunan tingkat kelulusan, dan berisiko turun peringkat akreditasi. Manajemen Jaya
Jaya Institut ingin dapat **mendeteksi risiko dropout sedini mungkin** sehingga mahasiswa
berisiko dapat diberi bimbingan khusus sebelum benar-benar berhenti kuliah.

### Permasalahan Bisnis

1. **Skala dan sebaran masalah belum terpetakan.** Institusi belum mengetahui program studi,
   jalur masuk, kelompok usia, atau kondisi finansial mana yang menjadi konsentrasi masalah
   dropout, sehingga alokasi sumber daya bimbingan tidak tepat sasaran.
2. **Faktor penyebab dropout belum teridentifikasi.** Belum diketahui variabel apa yang paling
   menentukan seorang mahasiswa akan dropout, sehingga intervensi yang diberikan bersifat
   coba-coba.
3. **Tidak ada mekanisme deteksi dini.** Institusi baru mengetahui seorang mahasiswa dropout
   setelah kejadiannya berlangsung. Tidak ada sistem yang mampu memberi peringatan lebih awal.
4. **Tidak ada alat monitoring bagi manajemen.** Data mahasiswa tersimpan dalam bentuk mentah dan
   sulit dipahami oleh pihak non-teknis, sehingga pengambilan keputusan tidak berbasis data.

### Cakupan Proyek

1. **Data understanding & EDA** — memahami struktur 4.424 record mahasiswa dengan 37 variabel,
   serta menggali pola dropout berdasarkan faktor demografi, finansial, dan akademik.
2. **Data preparation** — perumusan target biner, encoding fitur kategorikal, standardisasi fitur
   numerik, dan pemisahan data latih/uji secara *stratified*.
3. **Modeling** — membandingkan tiga algoritma klasifikasi (Logistic Regression, Random Forest,
   HistGradientBoosting) menggunakan 5-fold cross validation.
4. **Evaluation** — pemilihan model terbaik, penyetelan *threshold* berbasis kebutuhan bisnis, dan
   analisis feature importance untuk menjawab pertanyaan "faktor apa yang menyebabkan dropout".
5. **Business dashboard** — membangun dashboard monitoring performa siswa bagi manajemen.
6. **Deployment** — men-*deploy* prototype machine learning berbasis Streamlit ke Streamlit
   Community Cloud agar dapat diakses secara remote.
7. **Action items** — menyusun rekomendasi konkret berbasis temuan data.

### Persiapan

**Sumber data:** [Students' Performance — Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset berisi 4.424 baris dan 37 kolom, tanpa nilai kosong maupun baris duplikat. Kolom target
`Status` memiliki tiga kelas: `Graduate` (2.209), `Dropout` (1.421), dan `Enrolled` (794).

**Setup environment:**

```bash
# 1. Kloning atau unduh repositori proyek ini
git clone <URL-REPOSITORI-ANDA>
cd <NAMA-FOLDER-PROYEK>

# 2. Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Pasang seluruh dependensi
pip install -r requirements.txt

# 4. Jalankan notebook analisis (opsional)
jupyter notebook notebook.ipynb
```

**Struktur berkas proyek:**

```
.
├── README.md                       # Dokumentasi proyek (berkas ini)
├── notebook.ipynb                  # Proses lengkap: EDA → modeling → evaluasi
├── app.py                          # Prototype Streamlit (deteksi dini dropout)
├── prediction.py                   # Skrip inferensi berbasis CLI
├── requirements.txt                # Daftar dependensi
├── data.csv                        # Dataset mentah
├── model/
│   ├── dropout_model.joblib        # Pipeline model final + threshold
│   └── defaults.json               # Nilai default tiap fitur
└── dashboard/
    ├── dashboard.png               # Tangkapan layar business dashboard
    ├── students_dashboard.csv      # Dataset siap pakai untuk BI tool
    ├── eda_overview.png
    ├── feature_importance.png
    └── model_evaluation.png
```

---

## Business Dashboard

Business dashboard **Student Performance & Dropout Monitoring** dibangun untuk membantu manajemen
Jaya Jaya Institut memahami data dan memantau performa siswa tanpa perlu menjalankan kode.

![Business Dashboard](dashboard/dashboard.png)

**Tautan dashboard:** `ISI_DENGAN_LINK_DASHBOARD_ANDA`

> Jika menggunakan **Metabase**, sertakan kredensial berikut:
> - Email: `root@mail.com`
> - Password: `root123`
>
> Berkas database instance Metabase (`metabase.db.mv.db`) disertakan pada folder proyek.

### Komponen Dashboard

Dashboard terdiri atas empat kartu KPI dan sembilan panel visualisasi:

| Komponen | Informasi yang disajikan |
|---|---|
| **KPI Cards** | Total mahasiswa (4.424), dropout rate (32,1%), jumlah mahasiswa risiko tinggi (1.097), rata-rata tingkat kelulusan mata kuliah (67,9%) |
| **Distribusi Status Mahasiswa** | Proporsi Graduate / Dropout / Enrolled |
| **Dropout Rate per Program Studi** | Peringkat 17 program studi terhadap garis rata-rata institusi |
| **Sebaran Level Risiko** | Jumlah mahasiswa pada kategori risiko Rendah / Sedang / Tinggi |
| **Kondisi Finansial** | Dropout rate berdasarkan status UKT, status debitur, dan kepemilikan beasiswa |
| **Tingkat Kelulusan Mata Kuliah** | Dropout rate pada kategori performa Kritis / Rendah / Sedang / Baik |
| **Kelompok Usia** | Tren dropout rate menurut usia saat mendaftar |
| **Distribusi Nilai** | Perbandingan sebaran rata-rata nilai antar status |
| **Gender & Waktu Kuliah** | Dropout rate menurut jenis kelamin dan kelas reguler/malam |
| **Validasi Level Risiko** | Dropout rate aktual pada tiap level risiko keluaran model |

### Cara Membangun Ulang Dashboard

Dataset siap pakai tersedia pada `dashboard/students_dashboard.csv` (4.424 baris, 32 kolom) dengan
seluruh kode kategorikal telah diterjemahkan ke label yang mudah dibaca serta dilengkapi kolom
turunan `Approval_Rate`, `Kategori_Performa`, `Skor_Risiko`, dan `Level_Risiko`.

**Opsi A — Metabase (Docker):**

```bash
# 1. Jalankan container Metabase
docker run -d -p 3000:3000 --name metabase metabase/metabase

# 2. Buka http://localhost:3000, buat akun root@mail.com / root123

# 3. Unggah students_dashboard.csv melalui menu Settings → Admin → Databases,
#    atau gunakan fitur "Upload CSV" pada koleksi

# 4. Bangun pertanyaan (question) dan susun menjadi dashboard

# 5. Ekspor database instance Metabase
docker cp metabase:/metabase.db/metabase.db.mv.db ./
```

**Opsi B — Looker Studio:**

1. Unggah `dashboard/students_dashboard.csv` ke Google Sheets.
2. Buka [Looker Studio](https://lookerstudio.google.com/) → **Create** → **Report** → hubungkan
   sumber data Google Sheets tersebut.
3. Susun scorecard dan chart mengikuti tabel komponen di atas.
4. Klik **Share** → ubah akses menjadi **Anyone with the link can view**, lalu salin tautannya ke
   bagian *Tautan dashboard* di atas.

---

## Menjalankan Sistem Machine Learning

Prototype machine learning dibangun menggunakan **Streamlit** dan memiliki dua mode operasi:

- **Prediksi Individu** — formulir input profil mahasiswa (profil & pendaftaran, kondisi
  finansial, akademik semester 1, akademik semester 2). Keluarannya berupa skor risiko 0–100,
  level risiko, keputusan berdasarkan threshold, serta daftar rekomendasi tindak lanjut yang
  disesuaikan dengan faktor risiko yang terdeteksi.
- **Prediksi Massal** — unggah berkas CSV berisi banyak mahasiswa sekaligus, lalu sistem
  menghasilkan ringkasan jumlah mahasiswa per level risiko, daftar prioritas intervensi terurut
  dari skor tertinggi, dan tombol unduh hasil dalam format CSV.

### Tautan Prototype

**Streamlit Community Cloud:** `ISI_DENGAN_LINK_STREAMLIT_APP_ANDA`

### Menjalankan Secara Lokal

```bash
# Pastikan dependensi sudah terpasang
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis pada `http://localhost:8501`.

### Menjalankan Melalui Command Line

```bash
# Prediksi satu mahasiswa
python prediction.py --single \
    Curricular_units_1st_sem_approved=0 \
    Curricular_units_2nd_sem_approved=0 \
    Tuition_fees_up_to_date=0 Debtor=1 Age_at_enrollment=30

# Keluaran:
# Skor risiko dropout : 98.3%
# Level risiko        : Tinggi
# Keputusan (thr 0.447) : BERISIKO DROPOUT

# Prediksi massal dari berkas CSV
python prediction.py --input data.csv --output hasil_prediksi.csv
```

### Cara Deploy ke Streamlit Community Cloud

1. Unggah seluruh isi folder proyek ini ke sebuah repositori **GitHub publik** — pastikan
   `app.py`, `requirements.txt`, dan folder `model/` ikut terunggah.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk menggunakan akun GitHub.
3. Klik **New app**, pilih repositori dan branch yang sesuai, lalu isi *Main file path* dengan
   `app.py`.
4. Klik **Deploy**. Proses instalasi dependensi berlangsung sekitar 2–5 menit.
5. Salin URL aplikasi yang dihasilkan ke bagian *Tautan Prototype* di atas.

### Ringkasan Model

| Aspek | Keterangan |
|---|---|
| Algoritma | HistGradientBoostingClassifier (scikit-learn) |
| Target | Biner — `Dropout` (1) vs `Non-Dropout` (0) |
| Jumlah fitur | 36 (17 kategorikal + 19 numerik) |
| Preprocessing | `StandardScaler` + `OneHotEncoder` dalam satu `Pipeline` |
| Threshold operasional | 0,447 (disetel dari prediksi out-of-fold data latih) |
| Accuracy (data uji) | 0,897 |
| Precision (Dropout) | 0,870 |
| Recall (Dropout) | 0,799 |
| F1-Score (Dropout) | 0,833 |
| ROC-AUC | 0,934 |
| PR-AUC | 0,901 |

---

## Conclusion

Proyek ini berhasil menjawab keempat permasalahan bisnis Jaya Jaya Institut.

**1. Skala dan sebaran masalah kini terpetakan.**
Sebanyak **32,1%** dari 4.424 mahasiswa berstatus dropout. Masalah ini tidak merata, melainkan
terkonsentrasi pada program studi tertentu: **Biofuel Production Technologies (66,7%)**,
**Equinculture (55,3%)**, **Informatics Engineering (54,1%)**, **Management kelas malam (50,7%)**,
dan **Basic Education (44,3%)**. Sebaliknya, program **Nursing** merupakan yang paling aman dengan
dropout rate di bawah 16%. Dropout rate juga meningkat tajam seiring usia pendaftaran: dari 21%
pada kelompok ≤20 tahun menjadi 57% pada kelompok 28–35 tahun.

**2. Faktor penentu dropout telah teridentifikasi.**
Hasil EDA dan permutation importance secara konsisten menunjuk dua kelompok faktor dominan:

- **Performa akademik dua semester pertama.** Jumlah mata kuliah yang lulus pada semester 2
  (`Curricular_units_2nd_sem_approved`) adalah prediktor tunggal terkuat, dengan jarak sangat
  lebar dari fitur lain. Mahasiswa dengan tingkat kelulusan mata kuliah **di bawah 25%** memiliki
  dropout rate **81,6%**, sementara yang **di atas 75%** hanya **8,2%**. Rata-rata mahasiswa
  dropout hanya meluluskan 2,55 mata kuliah di semester 1 dan 1,94 di semester 2, dibandingkan
  6,23 dan 6,18 pada kelompok lulusan.
- **Kondisi finansial.** Status pelunasan UKT menempati posisi kedua dalam feature importance.
  Mahasiswa yang **menunggak UKT** dropout pada tingkat **86,6%** dibanding **24,7%** pada yang
  lunas — selisih lebih dari 60 poin persentase. Mahasiswa berstatus **debitur** dropout pada
  tingkat **62,0%**, sedangkan **penerima beasiswa** jauh lebih aman dengan hanya **12,2%**.

Temuan paling penting justru terletak pada apa yang **tidak** berpengaruh: nilai ujian masuk
(`Admission_grade`) hampir tidak membedakan kelompok dropout (124,96) dan lulusan (128,79).
Artinya, **dropout di Jaya Jaya Institut bukan masalah kualitas seleksi calon mahasiswa,
melainkan masalah dukungan selama masa studi** — terutama dukungan finansial dan pendampingan
akademik.

**3. Sistem deteksi dini telah tersedia dan terbukti akurat.**
Model HistGradientBoostingClassifier mencapai **ROC-AUC 0,934**, **akurasi 89,7%**, dan **recall
79,9%** pada data uji. Dari 284 mahasiswa dropout pada data uji, **227 berhasil terdeteksi lebih
awal** dan hanya 57 yang lolos. Level risiko yang dihasilkan juga terkalibrasi dengan baik:
kelompok **Risiko Tinggi memiliki dropout rate aktual 90,2%**, Sedang 40,2%, dan Rendah 8,8% —
sehingga daftar prioritas yang muncul di dashboard benar-benar dapat dipercaya. Karena prediktor
utamanya adalah data akhir semester 2, deteksi dapat dilakukan **sejak tahun pertama**, memberi
institusi waktu intervensi yang memadai.

**4. Alat monitoring telah dibangun.**
Business dashboard dan prototype Streamlit memungkinkan manajemen memantau performa siswa serta
menghasilkan daftar prioritas intervensi secara mandiri, tanpa perlu menjalankan kode.

**Estimasi dampak.** Dengan recall 79,9% terhadap 1.421 mahasiswa dropout, sistem ini mampu
menandai sekitar **1.135 mahasiswa berisiko** untuk diintervensi lebih awal. Apabila program
bimbingan berhasil menyelamatkan 30% saja di antaranya, institusi mempertahankan sekitar **340
mahasiswa**, setara dengan penurunan dropout rate dari 32,1% menjadi sekitar **24,4%**.

### Rekomendasi Action Items

**1. Terapkan sistem deteksi dini di akhir setiap semester.**
Jalankan prediksi massal terhadap seluruh mahasiswa aktif setiap akhir semester menggunakan
prototype Streamlit, lalu terbitkan daftar prioritas berdasarkan skor risiko. Mahasiswa dengan
skor **≥ 60 (Risiko Tinggi)** wajib masuk program bimbingan khusus dalam dua minggu pertama
semester berikutnya. Tetapkan satu dosen wali sebagai penanggung jawab untuk setiap 10 mahasiswa
berisiko tinggi.

**2. Jadikan keringanan biaya sebagai intervensi prioritas utama.**
Karena tunggakan UKT berasosiasi dengan dropout rate 86,6%, sediakan skema cicilan fleksibel,
penundaan pembayaran, atau dana talangan darurat. Lakukan **kontak proaktif** kepada setiap
mahasiswa yang tunggakannya melewati 30 hari — jangan menunggu mahasiswa mengajukan sendiri,
karena mereka yang paling terdampak umumnya justru paling enggan meminta bantuan.

**3. Perluas dan targetkan ulang alokasi beasiswa.**
Penerima beasiswa hanya dropout 12,2% dibanding 38,7% pada non-penerima. Prioritaskan alokasi
beasiswa bagi mahasiswa berskor risiko tinggi yang masih menunjukkan potensi akademik, dan
pertimbangkan menaikkan kuota khusus untuk program studi dengan dropout rate tertinggi.

**4. Terapkan aturan pemicu akademik otomatis (*academic trigger*).**
Tetapkan aturan sederhana yang dapat dijalankan dosen wali tanpa bantuan model: mahasiswa dengan
tingkat kelulusan mata kuliah **di bawah 50%** pada semester berjalan otomatis dijadwalkan
konseling akademik, program remedial, dan peninjauan ulang beban SKS semester berikutnya. Aturan
ini berfungsi sebagai lapisan pengaman apabila sistem prediksi tidak tersedia.

**5. Audit program studi berisiko tinggi.**
Lakukan evaluasi menyeluruh terhadap kurikulum, beban studi, kualitas pengajaran, dan fasilitas
pada **Biofuel Production Technologies, Equinculture, Informatics Engineering, dan Management
kelas malam**. Dropout rate di atas 50% mengindikasikan adanya masalah struktural pada program
tersebut, bukan sekadar kumpulan masalah individual mahasiswa.

**6. Rancang dukungan khusus bagi mahasiswa non-tradisional.**
Mahasiswa yang mendaftar pada usia di atas 25 tahun, masuk melalui jalur *"Over 23 years old"*,
atau mengambil kelas malam menunjukkan risiko yang jauh lebih tinggi (hingga 57% pada kelompok
usia 28–35 tahun). Sediakan kelas hybrid, jadwal fleksibel, layanan penitipan anak, dan
pendampingan manajemen waktu bagi kelompok ini.

**7. Jadikan dashboard sebagai basis rapat evaluasi rutin, dan perbarui model setiap tahun.**
Tetapkan rapat evaluasi bulanan berbasis dashboard dengan tiga indikator utama: jumlah mahasiswa
Risiko Tinggi, dropout rate per program studi, dan proporsi tunggakan UKT. Latih ulang model
setiap tahun ajaran menggunakan data terbaru agar performanya tetap terjaga seiring perubahan
karakteristik mahasiswa.
