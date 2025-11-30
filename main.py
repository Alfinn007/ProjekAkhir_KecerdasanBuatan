import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class StuntingAI:
    def __init__(self):
        print("Sedang memuat data WHO...")
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

        output['bahaya']  = fuzz.trimf(output.universe, [0, 0, 45])
        output['waspada'] = fuzz.trimf(output.universe, [40, 60, 80])
        output['sehat']   = fuzz.trimf(output.universe, [75, 100, 100])

        rule1 = ctrl.Rule(z_tb['sangat_pendek'] | z_bb['gizi_buruk'], output['bahaya'])
        rule2 = ctrl.Rule(z_tb['pendek'] & z_bb['gizi_baik'], output['waspada'])
        rule3 = ctrl.Rule(z_tb['normal'] & z_bb['gizi_kurang'], output['waspada'])
        rule4 = ctrl.Rule(z_tb['normal'] & z_bb['gizi_baik'], output['sehat'])
        rule5 = ctrl.Rule(z_tb['pendek'] & z_bb['gizi_kurang'], output['bahaya'])
        
        simulasi = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5]))
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

    def analisa_kesehatan(self, nama, gender, umur, tinggi, berat):
        z_tinggi = self.hitung_z_tb_u(gender, umur, tinggi)
        z_berat  = self.hitung_z_bb_u(gender, umur, berat)
        
        self.simulasi.input['stunting_score'] = z_tinggi
        self.simulasi.input['gizi_score']     = z_berat
        
        self.simulasi.compute()
        skor_akhir = self.simulasi.output['kondisi_anak']
        
        kesimpulan = ''
        if skor_akhir < 45:
            kesimpulan = 'Bahaya: Perlu perhatian medis segera.'
        elif skor_akhir < 75:
            kesimpulan = 'Waspada: Perlu pemantauan gizi.'
        else:
            kesimpulan = 'Sehat: Gizi baik.'
        
        print(f"--- Hasil Analisa: {nama} ---")
        print(f"Z-Score TB/U : {z_tinggi:.2f} SD")
        print(f"Z-Score BB/U : {z_berat:.2f} SD")
        print(f"Skor Fuzzy   : {skor_akhir:.2f}/100")
        print(f"Status       : {kesimpulan}")
        print("-------------------------------\n")
        
        return skor_akhir
    
def input_user():
    print("\n==============================================")
    print("   SISTEM DETEKSI STUNTING & GIZI (AI)   ")
    print("==============================================")
    
    aplikasi = StuntingAI()
    
    while True:
        try:
            nama = input("Masukkan Nama Anak: ")
            if nama.lower() == 'exit':
                print("Terima kasih telah menggunakan sistem ini.")
                break
            jenis_kelamin = input("Jenis Kelamin (Laki-laki/Perempuan): ")
            gender = 'laki-laki' if jenis_kelamin.lower().startswith('laki-laki') else 'perempuan'
            
            print("   (Ketik angka saja, misal: 24)")
            tipe_umur = input("3. Satuan Umur (Bulan/Tahun)? [b/t]: ").lower()
            umur_input = float(input("4. Masukkan Umur       : "))
            
            umur_bulan = umur_input * 12 if tipe_umur == 't' else umur_input
            
            if umur_bulan < 0 or umur_bulan > 60:
                print("   [Error] Umur harus antara 0-60 bulan untuk analisa ini.\n")
                continue
            tinggi = float(input("5. Tinggi Badan (cm)   : "))
            berat  = float(input("6. Berat Badan (kg)    : "))
            print("\nSedang Menganalisa...", end="\r")
            
            aplikasi.analisa_kesehatan(nama, gender, umur_bulan, tinggi, berat)
        except Exception as e:
            print(f"   [Error] Terjadi kesalahan input: {e}\n")
            continue

if __name__ == "__main__":
    input_user()