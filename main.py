from datetime import datetime
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

class StuntingAI:
    def __init__(self):
        print("Sedang memuat data WHO...")
        try:
            self.wfa_boys = pd.read_csv('dataset/wfa_boys_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.wfa_girls = pd.read_csv('dataset/wfa_girls_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.lhfa_boys = pd.read_csv('dataset/lhfa_boys_0-to-5-years_zscores.csv', sep=';', decimal=',')
            self.lhfa_girls = pd.read_csv('dataset/lhfa_girls_0-to-5-years_zscores.csv', sep=';', decimal=',')
            
            if 'c' in self.lhfa_boys.columns:
                self.lhfa_boys.rename(columns={'c': 'Month'}, inplace=True)
            if 'c' in self.lhfa_girls.columns:
                self.lhfa_girls.rename(columns={'c': 'Month'}, inplace=True)
            
            print("Menyiapkan Logika Fuzzy...")
            self.simulasi = self.set_up_fuzzy_system()
            print("Sistem SIAP! \n")
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
        if gender.lower() == 'laki-laki': std_data = self.wfa_boys
        else: std_data = self.wfa_girls

        row = std_data[std_data['Month'] == umur]
        if row.empty: return 0
        L, M, S = row['L'].values[0], row['M'].values[0], row['S'].values[0]
        return ((berat / M) ** L - 1) / (S * L)

    def simpan_data(self, data_dict):
        nama_file = 'laporan_hasil.csv'
        df_baru = pd.DataFrame([data_dict])
        
        if not os.path.isfile(nama_file):
            df_baru.to_csv(nama_file, index=False)
        else:
            df_baru.to_csv(nama_file, mode='a', header=False, index=False)
            print(f"Data {data_dict['nama']} telah disimpan")

    def analisa_kesehatan(self, nama, gender, umur, tinggi, berat):
        z_tinggi = self.hitung_z_tb_u(gender, umur, tinggi)
        z_berat  = self.hitung_z_bb_u(gender, umur, berat)
        
        z_tinggi_input = max(min(z_tinggi, 5), -5)
        z_berat_input  = max(min(z_berat, 5), -5)
        
        self.simulasi.input['stunting_score'] = z_tinggi_input
        self.simulasi.input['gizi_score']     = z_berat_input
        
        try:
            self.simulasi.compute()
            skor_akhir = self.simulasi.output['kondisi_anak']
        except:
            skor_akhir = 0
        
        kesimpulan = ''
        sarantxt = ''
        saran = []
        if skor_akhir < 45:
            kesimpulan = 'Severely Stunted (Stunting Berat)'
            saran = [
                "SEGERA rujuk ke Rumah Sakit atau Dokter Spesialis Anak.",
                "Pemberian Pangan Olahan untuk Keperluan Medis Khusus (PKMK) di bawah pengawasan dokter.",
                "Investigasi penyakit penyerta (seperti TBC, infeksi berulang) yang menghambat pertumbuhan.",
                "Pemantauan pertumbuhan secara intensif setiap minggu."
            ]
        elif skor_akhir < 75:
            kesimpulan = 'Stunted (Stunting)'
            saran = [
                "Evaluasi pola makan: Wajib tambahkan satu porsi protein hewani (telur, ikan, ayam, daging) setiap kali makan.",
                "Berikan Pemberian Makanan Tambahan (PMT) tinggi kalori dan protein.",
                "Suplementasi mikronutrien (Taburia, Zinc, Vitamin A) sesuai anjuran Puskesmas/Posyandu.",
                "Cek sanitasi lingkungan (air bersih dan jamban) serta perilaku hidup bersih."
            ]
        else:
            kesimpulan = 'Normal'
            saran = [
                "Pertahankan pola makan gizi seimbang (Isi Piringku).",
                "Lanjutkan pemantauan pertumbuhan rutin di Posyandu setiap bulan.",
                "Pastikan imunisasi dasar dan lanjutan lengkap.",
                "Jaga kebersihan diri dan lingkungan untuk mencegah infeksi."
            ]
        for i, poin in enumerate(saran):
            if i == 0:
                sarantxt += f'- {poin}\n'
            else:
                sarantxt += f'{" "*15}- {poin}\n'

        print(f"--- Hasil Analisa: {nama} ---")
        print(f"Z-Score TB/U : {z_tinggi:.2f} SD")
        print(f"Z-Score BB/U : {z_berat:.2f} SD")
        print(f"Skor Fuzzy   : {skor_akhir:.2f}/100")
        print(f"Status       : {kesimpulan}")
        print(f"Saran        : {sarantxt}")
        print("-------------------------------\n")
        
        data_hasil = {
            'Tanggal_Periksa': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nama': nama,
            'JK': gender,
            'Umur_Bulan': umur,
            'Tinggi_cm': tinggi,
            'Berat_kg': berat,
            'Z_Score_TB': round(z_tinggi, 2),
            'Z_Score_BB': round(z_berat, 2),
            'Skor_Fuzzy': round(skor_akhir, 2),
            'Kesimpulan': kesimpulan
        }
        
        self.simpan_data(data_hasil)
        return skor_akhir
    
def input_user():
    print("\n==============================================")
    print("   SISTEM DETEKSI STUNTING & GIZI (AI)   ")
    print("==============================================")
    
    aplikasi = StuntingAI()
    
    while True:
        print("\n----------------------------------------------")
        print("              MENU UTAMA POSYANDU              ")
        print("----------------------------------------------")
        print("[1] Cek Status Anak (Input Data)")
        print("[2] Keluar Aplikasi")
        print("----------------------------------------------")
        
        pilihan = input("Pilih menu (1/2): ")
        
        if pilihan == '1':
            try:
                print ("\n--- Input Data Anak ---")
                nama   = input("Nama Anak       : ")
                
                gender = ''
                while True:
                    jk = input("Jenis Kelamin (Laki-Laki/Perempuan) [L/P]: ").lower()
                    if jk == 'l': 
                        gender = 'laki-laki'
                        break
                    elif jk == 'p': 
                        gender = 'perempuan'
                        break
                    else: 
                        print("Input tidak valid. Masukkan 'l' atau 'p'.")
                    
                tipe_umur = ''
                while True:
                    tipe = input("Tipe Umur (Bulan/Tahun) [B/T]: ").lower()
                    if tipe == 'b': 
                        tipe_umur = 'bulan'
                        break
                    elif tipe == 't': 
                        tipe_umur = 'tahun'
                        break
                    else: 
                        print("Input tidak valid. Masukkan 'b' atau 't'.")
                
                umur_input = float(input("Umur Anak        : "))
                umur_bulan = umur_input if tipe_umur == 'bulan' else umur_input * 12
                
                if umur_bulan < 0 or umur_bulan > 60:
                    print("Umur harus antara 0-60 bulan")
                    print("Tekan enter untuk kembali ke menu utama")
                    continue
                
                tinggi = float(input("Tinggi Badan (cm): "))
                berat  = float(input("Berat Badan (kg) : "))
                
                aplikasi.analisa_kesehatan(nama, gender, umur_bulan, tinggi, berat)
                input("Tekan enter untuk melanjutkan")
                
            except ValueError:
                print("Input tidak valid (pastikan angka). Silakan coba lagi.")
                input("Tekan enter untuk melanjutkan")
            except Exception as e:
                print(f"Terjadi kesalahan: {e}")
                input("Tekan enter untuk melanjutkan")
                
        elif pilihan == '2':
            print("Terima kasih telah menggunakan aplikasi ini.")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")
            
if __name__ == "__main__":
    input_user()