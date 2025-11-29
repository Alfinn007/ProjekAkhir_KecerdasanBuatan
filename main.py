import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

z_score_var = ctrl.Antecedent(np.arange(-5, 6, 0.1), 'z_score')
risiko = ctrl.Consequent(np.arange(0, 11, 1), 'status_gizi')

z_score_var['sangat_pendek'] = fuzz.trapmf(z_score_var.universe, [-5, -5, -3.1, -2.9])
z_score_var['pendek'] = fuzz.trimf(z_score_var.universe, [-3.1, -2, -1.9])
z_score_var['normal'] = fuzz.trapmf(z_score_var.universe, [-2.1, -1, 5, 5])

risiko['parah'] = fuzz.trimf(risiko.universe, [0, 0, 4])
risiko['sedang'] = fuzz.trimf(risiko.universe, [2, 4, 6])
risiko['sehat'] = fuzz.trapmf(risiko.universe, [5, 7, 10, 10])

# Rules
rule1 = ctrl.Rule(z_score_var['sangat_pendek'], risiko['parah'])
rule2 = ctrl.Rule(z_score_var['pendek'], risiko['sedang'])
rule3 = ctrl.Rule(z_score_var['normal'], risiko['sehat'])

# Control System
stunting_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
stunting_engine = ctrl.ControlSystemSimulation(stunting_ctrl)

# ==========================================
# BAGIAN 2: LOAD DATASET WHO & PASIEN
# ==========================================

print("Sedang memuat data...")

# 1. Load Standar WHO (LHFA - Length Height For Age)
# Penting: delimiter=';' dan decimal=',' karena format file kamu Eropa/Indo
std_boys = pd.read_csv('archive/z-scores who/lhfa_boys_0-to-5-years_zscores.csv', sep=';', decimal=',')
std_girls = pd.read_csv('archive/z-scores who/lhfa_girls_0-to-5-years_zscores.csv', sep=';', decimal=',')

# Set 'Month' sebagai index biar gampang dicari
if 'c' in std_boys.columns:
    std_boys.rename(columns={'c': 'Month'}, inplace=True)
if 'c' in std_girls.columns:
    std_girls.rename(columns={'c': 'Month'}, inplace=True)
# 2. Load Data Pasien/Dataset Kaggle
data_pasien = pd.read_csv('stunting_wasting_dataset.csv', sep=',') 
# Cek 5 data pertama untuk memastikan terbaca
print("Contoh Data Pasien:")
print(data_pasien.head())

# ==========================================
# BAGIAN 3: PROSES LOOPING (TESTING)
# ==========================================

hasil_prediksi = []

print("\nMemulai Diagnosa AI...")

for index, row in data_pasien.iterrows():
    # Ambil data per baris
    gender = row['Jenis Kelamin']
    umur = int(row['Umur (bulan)'])
    tinggi = float(row['Tinggi Badan (cm)'])
    label_asli = row['Stunting'] # Kunci jawaban dari dataset
    
    # 1. Pilih Tabel Referensi (Laki/Perempuan)
    if gender == 'Laki-laki':
        tabel_ref = std_boys
    else:
        tabel_ref = std_girls
        
    # Validasi umur (karena tabel WHO cuma 0-60 bulan)
    if umur > 60:
        hasil_prediksi.append("Umur > 60 bln (Skip)")
        continue
        
    # 2. Ambil Median (M) dan Standar Deviasi (SD) dari tabel WHO
    # Kita pakai kolom 'M' dan 'SD' dari file CSV WHO kamu
    median_who = tabel_ref.loc[umur, 'M']
    sd_who = tabel_ref.loc[umur, 'SD'] # File kamu punya kolom 'SD' kan?
    
    # 3. Hitung Z-Score Manual
    # Rumus simple: (Tinggi Anak - Median) / Simpangan Baku
    z_score_hitung = (tinggi - median_who) / sd_who
    
    # 4. Masukkan ke Fuzzy Engine
    stunting_engine.input['z_score'] = z_score_hitung
    stunting_engine.compute()
    skor_fuzzy = stunting_engine.output['status_gizi']
    
    # 5. Terjemahkan Skor ke Kategori
    if skor_fuzzy < 3.5:
        kategori_ai = "Severely Stunted"
    elif skor_fuzzy < 5.5:
        kategori_ai = "Stunted"
    else:
        kategori_ai = "Normal" # Bisa dicampur Tall jadi Normal disini
        
    # Simpan hasil untuk dibandingkan
    hasil_prediksi.append({
        'Gender': gender,
        'Umur': umur,
        'Tinggi': tinggi,
        'Z_Score': round(z_score_hitung, 2),
        'AI_Prediksi': kategori_ai,
        'Label_Asli': label_asli,
        'Cocok': "✅" if kategori_ai in label_asli else "❌" 
        # Note: Logika 'Cocok' mungkin perlu disesuaikan karena label dataset mungkin 'Tall' atau beda format
    })

# ==========================================
# BAGIAN 4: HASIL AKHIR
# ==========================================

# Ubah ke DataFrame biar rapi
df_hasil = pd.DataFrame(hasil_prediksi)

# Tampilkan 20 hasil pertama
print("\n=== HASIL DIAGNOSA (20 Data Pertama) ===")
print(df_hasil.head(20))

# Hitung Akurasi Sederhana (Opsional)
jumlah_data = len(df_hasil)
# Filter yang labelnya mirip-mirip (misal dataset bilang Normal, AI bilang Normal)
# Ini butuh penyesuaian string matching nantinya
print(f"\nTotal Data Diproses: {jumlah_data} baris")