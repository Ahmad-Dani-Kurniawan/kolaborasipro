import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime

st.set_page_config(page_title="Prediksi Saham")

st.write("""# Aplikasi Prediksi Saham PT. Gudang Garam Tbk (GGRM.JK)""")
st.write("Oleh: Ahmad Dani Kurniawan (200411100205)")
st.write("      Indra Ramadan Fadilafani (200411100159)")
st.write("----------------------------------------------------------------------------------")

dataset, preprocessing, modeling, implementasi = st.tabs(["Data", "Preprocessing", "Modelling", "Implementation"])

with dataset:
    st.header("Data Set")
    data = pd.read_csv('https://raw.githubusercontent.com/Ahmad-Dani-Kurniawan/kolaborasipro/main/GGRM.csv')
    st.write(data)
    st.subheader("Penjelasan :")
    st.write("Data yang digunakan merupakan data saham PT.Gudang Garam Tbk (GGRM.JK) yang diambil dari website finance.yahoo.com.")
    st.write("Tipe data yang digunakan merupakan tipe data historical.")
    st.write("""Type datanya dari mana""")


with preprocessing:
    st.header("Preprocessing Data")
    st.write("""
         Preprocessing adalah teknik penambangan data yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien.
    """)
    
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    target = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    st.write("Min Max Scaler")
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    st.write(features_scaled)

with modeling:
    st.header("Modelling")
    
    knn_cekbox = st.checkbox("KNN")
    lr_cekbox = st.checkbox("Linear Regression")

    #===================== KNN =======================

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    # Membuat model KNN
    model1 = KNeighborsRegressor(n_neighbors=5)

    # Melatih model KNN
    model1.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = model1.predict(X_test)

    # Menghitung MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    #===================== Linear Reggression =======================

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    # Membuat model KNN
    model2 = LinearRegression()

    # Melatih model KNN
    model2.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = model2.predict(X_test)

    # Menghitung MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)

    #===================== Cek Box ====================

    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Prediksi menggunakan KNN:")
        # st.warning(knn_accuracy)
        st.warning(f"MAPE  =  {mape_knn}")
        st.markdown("---")
    
    if lr_cekbox:
        st.write("##### Random Forest")
        st.warning("Prediksi menggunakan Random Forest:")
        # st.warning(knn_accuracy)
        st.warning(f"MAPE  =  {mape_lr}")
        st.markdown("---")