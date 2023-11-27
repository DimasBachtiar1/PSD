import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Penyakit Ginjal Kronis Menggunakan Model Random Forest Classifier</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Dimas Bachtiar | 210411100022 | PSD - B</h4>", unsafe_allow_html=True
)

# memanggil dataset
data = pd.read_csv('data_baru_bangetz.csv')

# memisahkan kolom fitur dan kolom target
fitur = data.drop(columns=['id','age','cad','pcc','ane','ba','classification'])
target = data['classification']

# melakukan split data training dan data testing
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# memanggil kembali model normalisasi zscore dari file pickle
path = 'zscore_scaler_ckd.pkl'
with open(path, 'rb') as file:
    zscore_scaler = pickle.load(file)

# menerapkan normalisasi zscore pada data training
zscore_training = zscore_scaler.transform(fitur_train)

# menerapkan normalisasi zscore pada data testing
zscore_testing = zscore_scaler.transform(fitur_test)
    
# memanggil file model terbaik menggunakan random forest
with open('best_rf_model_zscore_ckd (1).pkl', 'rb') as model_file:
    model_rf = pickle.load(model_file)


# memanggil file model terbaik menggunakan random forest
# from joblib import dump, load
# model_rf = load('best_rf_model_zscore_ckd.joblib')


model_rf.fit(zscore_training, target_train)
prediksi_target = model_rf.predict(zscore_testing)

# prediksi

hemo = st.slider("Hemoglobin (hemo):", min_value=0.0, max_value=20.0, value=10.0)
pcv = st.slider("Packed Cell Volume (pcv):", min_value=0.0, max_value=100.0, value=40.0)
rc = st.slider("Red Blood Cell Count (rc):", min_value=0.0, max_value=10.0, value=4.5)
sc = st.slider("Serum Creatinine (sc):", min_value=0.0, max_value=10.0, value=1.0)
sg = st.slider("Specific Gravity (sg):", min_value=1.0, max_value=2.0, value=1.020)

st.warning("Tekan 0 untuk 'normal' dan 1 untuk 'abnormal'")
rbc_options = ['none','0', '1']
selected_rbc = st.selectbox("Red Blood Cells (rbc):", rbc_options)

st.warning("Tekan 0 untuk 'tidak' dan 1 untuk 'ya'")
al_options = ['none','0', '1', '2', '3', '4', '5']
selected_al = st.selectbox("Albumin (al):", al_options)

sod = st.slider("Sodium (sod):", min_value=0.0, max_value=200.0, value=140.0)

st.warning("Tekan 0 untuk 'no' dan 1 untuk 'yes'")
htn_options = ['none','0', '1']
selected_htn = st.selectbox("Hypertension (htn):", htn_options)

pot = st.slider("Potassium (pot):", min_value=0.0, max_value=7.0, value=4.0)

bu = st.slider("Blood Urea (bu):", min_value=0.0, max_value=300.0, value=40.0)

wc = st.slider("White Blood Cell Count (wc):", min_value=0.0, max_value=25000.0, value=8000.0)

bgr = st.slider("Blood Glucose Random (bgr):", min_value=0.0, max_value=300.0, value=100.0)

st.warning("Tekan 0 untuk 'abnormal' dan 1 untuk 'normal'")
pc_options = ['none','0', '1']
selected_pc = st.selectbox("Pus Cell (pc):", pc_options)

bp = st.slider("Blood Pressure (bp):", min_value=0, max_value=180, value=120)

st.warning("Tekan 0 untuk 'no' dan 1 untuk 'yes'")
dm_options = ['none','0', '1']
selected_dm = st.selectbox("Diabetes Mellitus (dm):", dm_options)

st.warning("Tekan 0 untuk 'no' dan 1 untuk 'yes'")
su_options = ['none','0', '1', '2', '3', '4', '5']
selected_su = st.selectbox("Sugar (su):", su_options)

st.warning("Tekan 0 untuk 'poor' dan 1 untuk 'good'")
appet_options = ['none','0', '1']
selected_appet = st.selectbox("Appetite (appet):", appet_options)

st.warning("Tekan 0 untuk 'no' dan 1 untuk 'yes'")
pe_options = ['0', '1']
selected_pe = st.selectbox("Pedal Edema (pe):", pe_options)


if st.button('Cek Hasil'):
    if hemo is not 0.0 and pcv is not 0.0 and rc is not 0.0 and sc is not 0.0 and sg is not 0.0 and selected_rbc is not "none" and selected_al is not "none" and sod is not 0.0 and selected_htn is not "none" and pot is not 0.0 and bu is not 0.0 and wc is not 0.0 and bgr is not 0.0 and selected_pc is not "none" and bp is not 0.0 and selected_dm is not "none" and selected_su is not "none" and selected_appet is not "none" and selected_pe is not "none":
        st.text('Prediksi : ')
        prediksi = model_rf.predict([[hemo, pcv, rc, sc, sg, selected_rbc, selected_al, sod, selected_htn, pot,
                                      bu, wc, bgr, selected_pc, bp, selected_dm, selected_su,
                                      selected_appet, selected_pe]])
        if prediksi[0] == 1:
            st.success("Anda diprediksi terinfeksi Penyakit Ginjal Kronis!")
        elif prediksi[0] == 0:
            st.warning("Anda diprediksi Tidak terinfeksi Penyakit Ginjal Kronis!")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')
