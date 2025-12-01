from flask import Flask, render_template, request
from datetime import datetime
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

app = Flask(__name__)

class StuntingAI:
    def __init__(self):
        print("Sedang memuat data WHO...")
        try:

            self.wfa_boys = pd.read_csv('dataset/wfa_boys_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.wfa_girls = pd.read_csv('dataset/wfa_girls_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.lhfa_boys = pd.read_csv('dataset/lhfa_boys_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.lhfa_girls = pd.read_csv('dataset/lhfa_girls_0-to-5-years_zscores.csv', sep=';', decimal=',')
            
            if 'c' in self.lhfa_boys.columns: self.lhfa_boys.rename(columns={'c': 'Month'}, inplace=True)
            if 'c' in self.lhfa_girls.columns: self.lhfa_girls.rename(columns={'c': 'Month'}, inplace=True)
            
            print("Menyiapkan Logika Fuzzy...")
            self.simulasi = self.set_up_fuzzy_system()
            print("Sistem AI SIAP!")
        except FileNotFoundError:
            print("Error: File dataset tidak ditemukan. Pastikan folder 'dataset' ada.")

    def set_up_fuzzy_system(self):
        z_tb  = ctrl.Antecedent(np.arange(-5, 6, 0.1), 'stunting_score')
        z_bb  = ctrl.Antecedent(np.arange(-5, 6, 0.1), 'gizi_score')
        output = ctrl.Consequent(np.arange(0, 101, 1), 'kondisi_anak')

        z_tb['sangat_pendek'] = fuzz.trapmf(z_tb.universe, [-5, -5, -3.1, -2.9])
        z_tb['pendek']        = fuzz.trimf(z_tb.universe, [-3.1, -2.5, -1.9])
        z_tb['normal']        = fuzz.trapmf(z_tb.universe, [-2.1, -1, 3, 5])

        z_bb['gizi_buruk']  = fuzz.trapmf(z_bb.universe, [-5, -5, -3.1, -2.9])
        z_bb['gizi_kurang'] = fuzz.trimf(z_bb.universe, [-3.1, -2.5, -1.9])
        z_bb['gizi_baik']   = fuzz.trapmf(z_bb.universe, [-2.1, 0, 1.9, 2.1])
        z_bb['gizi_lebih']  = fuzz.trapmf(z_bb.universe, [1.9, 2.1, 5, 5])

        output['severely_stunted'] = fuzz.trimf(output.universe, [0, 0, 45])
        output['stunted']          = fuzz.trimf(output.universe, [40, 60, 80])
        output['normal']           = fuzz.trimf(output.universe, [75, 100, 100])

        rule1 = ctrl.Rule(z_tb['sangat_pendek'] | z_bb['gizi_buruk'], output['severely_stunted'])
        rule2 = ctrl.Rule(z_tb['pendek'] & z_bb['gizi_baik'], output['stunted'])
        rule3 = ctrl.Rule(z_tb['normal'] & z_bb['gizi_kurang'], output['stunted'])
        rule4 = ctrl.Rule(z_tb['normal'] & z_bb['gizi_baik'], output['normal'])
        rule5 = ctrl.Rule(z_tb['pendek'] & z_bb['gizi_kurang'], output['severely_stunted'])
        rule6 = ctrl.Rule(z_tb['sangat_pendek'] & z_bb['gizi_lebih'], output['stunted'])
            
        simulasi = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6]))
        return simulasi

    def hitung_z_tb_u(self, gender, umur, tinggi):
        if gender.lower() == 'laki-laki': std_data = self.lhfa_boys
        else: std_data = self.lhfa_girls
        
        row = std_data[std_data['Month'] == umur]
        if row.empty: return 0
        L, M, S = row['L'].values[0], row['M'].values[0], row['S'].values[0]
        return ((tinggi / M) ** L - 1) / (S * L)

    def hitung_z_bb_u(self, gender, umur, berat):
        if gender.lower() == 'laki-laki': 
            std_data = self.wfa_boys
        else: 
            std_data = self.wfa_girls

        row = std_data[std_data['Month'] == umur]
        if row.empty: 
            return 0
        L, M, S = row['L'].values[0], row['M'].values[0], row['S'].values[0]
        return ((berat / M) ** L - 1) / (S * L)

    def simpan_data(self, data_dict):
        data_to_save = data_dict.copy()
        if 'saran' in data_to_save:
            del data_to_save['saran']

        nama_file = 'dataset/laporan_hasil.csv' 
        
        df_baru = pd.DataFrame([data_to_save])

        if not os.path.isfile(nama_file):
            df_baru.to_csv(nama_file, index=False)
        else:
            df_baru.to_csv(nama_file, mode='a', header=False, index=False)

    def analisa_kesehatan(self, nama, gender, umur_bulan, tinggi, berat, umur_input_asli, tipe_umur):
        z_tinggi = self.hitung_z_tb_u(gender, umur_bulan, tinggi)
        z_berat  = self.hitung_z_bb_u(gender, umur_bulan, berat)
        
        z_tinggi_input = max(min(z_tinggi, 5), -5)
        z_berat_input  = max(min(z_berat, 5), -5)
        
        self.simulasi.input['stunting_score'] = z_tinggi_input
        self.simulasi.input['gizi_score']     = z_berat_input
        
        try:
            self.simulasi.compute()
            skor_akhir = self.simulasi.output['kondisi_anak']
        except:
            skor_akhir = 0

        if tipe_umur == 'tahun':
            umur_display = f"{umur_input_asli} Tahun ({umur_bulan:.1f} Bulan)"
        else:
            umur_display = f"{umur_bulan} Bulan"

        saran_list = []
        warna_css = ''
        
        if skor_akhir < 45:
            kesimpulan = 'Severely Stunted (Stunting Berat)'
            warna_css = 'danger' 
            saran_list = [
                "SEGERA rujuk ke Rumah Sakit atau Dokter Spesialis Anak.",
                "Pemberian Pangan Olahan untuk Keperluan Medis Khusus (PKMK) di bawah pengawasan dokter.",
                "Investigasi penyakit penyerta (seperti TBC, infeksi berulang) yang menghambat pertumbuhan.",
                "Pemantauan pertumbuhan secara intensif setiap minggu."
            ]
        elif skor_akhir < 75:
            kesimpulan = 'Stunted (Stunting)'
            warna_css = 'warning'
            saran_list = [
                "Evaluasi pola makan: Wajib tambahkan satu porsi protein hewani (telur, ikan, ayam, daging) setiap kali makan.",
                "Berikan Pemberian Makanan Tambahan (PMT) tinggi kalori dan protein.",
                "Suplementasi mikronutrien (Taburia, Zinc, Vitamin A) sesuai anjuran Puskesmas/Posyandu.",
                "Cek sanitasi lingkungan (air bersih dan jamban) serta perilaku hidup bersih."
            ]
        else:
            kesimpulan = 'Normal'
            warna_css = 'success'
            saran_list = [
                "Pertahankan pola makan gizi seimbang (Isi Piringku).",
                "Lanjutkan pemantauan pertumbuhan rutin di Posyandu setiap bulan.",
                "Pastikan imunisasi dasar dan lanjutan lengkap.",
                "Jaga kebersihan diri dan lingkungan untuk mencegah infeksi."
            ]

        data_hasil = {
            'Tanggal': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'nama': nama,
            'JK': gender,
            'Umur_Display': umur_display,
            'Tinggi_cm': tinggi,
            'Berat_kg': berat,
            'Z_Score_TB': round(z_tinggi, 2),
            'Z_Score_BB': round(z_berat, 2),
            'Skor_Fuzzy': round(skor_akhir, 2),
            'Kesimpulan': kesimpulan,
            'warna': warna_css,    
            'saran': saran_list
        }
        
        self.simpan_data(data_hasil)
        return data_hasil

ai_system = StuntingAI()

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    if request.method == 'POST':
        try:
            nama = request.form['nama']
            gender = request.form['gender']
            tipe_umur = request.form['tipe_umur']
            umur_input = float(request.form['umur'])
            tinggi = float(request.form['tinggi'])
            berat = float(request.form['berat'])

            umur_bulan = umur_input if tipe_umur == 'bulan' else umur_input * 12
            
            if umur_bulan < 0 or umur_bulan > 60:
                error_msg = "Umur harus antara 0 - 60 bulan (5 Tahun)."
                return render_template('index.html', error=error_msg)

            hasil = ai_system.analisa_kesehatan(nama, gender, umur_bulan, tinggi, berat, umur_input, tipe_umur)
            
        except ValueError:
            return render_template('index.html', error="Pastikan input angka valid.")
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan: {e}")

    return render_template('index.html', hasil=hasil)

@app.route('/database')
def lihat_database():
    path_file = 'dataset/laporan_hasil.csv'
    data_pasien = []
    
    if not os.path.exists(path_file):
        print(f"[DEBUG] File tidak ditemukan di: {os.path.abspath(path_file)}")
        return render_template('database.html', data=[]) 
    
    try:
        df = pd.read_csv(path_file)
        print(f"[DEBUG] File ditemukan. Jumlah baris: {len(df)}")

        required_columns = ['warna', 'Kesimpulan', 'Umur_Display']
        for col in required_columns:
            if col not in df.columns:
                print(f"[DEBUG] Kolom '{col}' tidak ada (Mungkin data lama). Mengisi default.")
                df[col] = '-' 
                if col == 'warna': df[col] = 'light' 

        df = df.iloc[::-1]
        data_pasien = df.to_dict(orient='records')
        print("[DEBUG] Data berhasil diproses ke dictionary.")
        
    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV: {e}")
    return render_template('database.html', data=data_pasien)

if __name__ == '__main__':
    app.run(debug=True)