"""
Dashboard de predicción de enfermedad cardíaca con Random Forest.
Caso 1 - Grupo 4 - Minería de Datos (CRISP-ML, Fase 6: Despliegue)

Ejecución:  streamlit run app.py
Requiere el archivo processed.cleveland.data en la misma carpeta.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed.cleveland.data")
COLUMNAS = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral',
            'fasting_blood_sugar', 'electrocardiographic', 'max_heart_rate',
            'induced_angina', 'ST_depression', 'slope', 'no_of_vessels', 'thal', 'diagnosis']

st.set_page_config(page_title="Predicción de enfermedad cardíaca", page_icon="❤️", layout="wide")


# ---------------------------------------------------------------------------
# Datos y modelo (mismo procedimiento que el notebook)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(RUTA, names=COLUMNAS, header=None, na_values="?")
    # Preprocesamiento: valores faltantes con la moda
    df['no_of_vessels'] = df['no_of_vessels'].fillna(df['no_of_vessels'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    # Transformación: variable objetivo a 0 (sano) y 1 (enfermo)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x > 0 else 0)
    return df


@st.cache_resource
def entrenar_modelo():
    df = cargar_datos()
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2606)

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    modelo = RandomForestClassifier(n_estimators=100, random_state=2606)
    modelo.fit(X_train_esc, y_train)

    pred = modelo.predict(X_test_esc)
    prob = modelo.predict_proba(X_test_esc)[:, 1]
    resultados = {
        "acc_train": accuracy_score(y_train, modelo.predict(X_train_esc)),
        "acc_test": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob),
        "cm": confusion_matrix(y_test, pred),
        "roc": roc_curve(y_test, prob),
        "reporte": pd.DataFrame(classification_report(
            y_test, pred, target_names=['Sano (0)', 'Enfermo (1)'], output_dict=True)).T.round(2),
        "importancias": pd.Series(modelo.feature_importances_, index=X.columns).sort_values(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return modelo, escalador, resultados


def predecir(modelo, escalador, valores):
    fila = pd.DataFrame([valores], columns=COLUMNAS[:-1])
    probabilidad = modelo.predict_proba(escalador.transform(fila))[0, 1]
    return probabilidad


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
if not os.path.exists(RUTA):
    st.error("No se encontró 'processed.cleveland.data'. Colócalo en la misma carpeta que app.py.")
    st.stop()

df = cargar_datos()
modelo, escalador, res = entrenar_modelo()

st.title("❤️ Predicción de enfermedad cardíaca con Random Forest")
st.caption("Caso de estudio: Mishra et al. (2024) · Dataset Cleveland (UCI) · CRISP-ML, Fase 6: Despliegue")

with st.sidebar:
    st.header("Modelo")
    st.write("**Algoritmo:** Random Forest")
    st.write("**Árboles:** 100")
    st.write(f"**Datos de entrenamiento:** {res['n_train']} pacientes")
    st.write(f"**Datos de prueba:** {res['n_test']} pacientes")
    st.write(f"**Exactitud en prueba:** {res['acc_test']:.2%}")
    st.warning("Uso académico. No sustituye un diagnóstico médico.")

tab_pred, tab_rend, tab_var, tab_datos = st.tabs(
    ["Predicción", "Rendimiento del modelo", "Importancia de variables", "Datos"])

# ---- Pestaña 1: predicción de un paciente nuevo ----
with tab_pred:
    st.subheader("Datos clínicos del paciente")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad (años)", min_value=20, max_value=90, value=55)
        sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: {0: "Mujer", 1: "Hombre"}[x])
        chest_pain = st.selectbox(
            "Tipo de dolor de pecho", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1 - Angina típica", 2: "2 - Angina atípica",
                                   3: "3 - Dolor no anginoso", 4: "4 - Asintomático"}[x])
        blood_pressure = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=220, value=130)
        serum_cholestoral = st.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=245)
    with c2:
        fasting_blood_sugar = st.selectbox(
            "Azúcar en ayunas > 120 mg/dl", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
        electrocardiographic = st.selectbox(
            "Electrocardiograma en reposo", options=[0, 1, 2],
            format_func=lambda x: {0: "0 - Normal", 1: "1 - Anomalía ST-T", 2: "2 - Hipertrofia"}[x])
        max_heart_rate = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
        induced_angina = st.selectbox(
            "Angina inducida por ejercicio", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
    with c3:
        ST_depression = st.number_input("Depresión ST por ejercicio", min_value=0.0, max_value=7.0,
                                        value=1.0, step=0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST", options=[1, 2, 3],
            format_func=lambda x: {1: "1 - Ascendente", 2: "2 - Plana", 3: "3 - Descendente"}[x])
        no_of_vessels = st.selectbox("Vasos principales coloreados (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thal", options=[3, 6, 7],
            format_func=lambda x: {3: "3 - Normal", 6: "6 - Defecto fijo", 7: "7 - Defecto reversible"}[x])

    if st.button("Predecir"):
        valores = [age, sex, chest_pain, blood_pressure, serum_cholestoral, fasting_blood_sugar,
                   electrocardiographic, max_heart_rate, induced_angina, ST_depression, slope,
                   no_of_vessels, thal]
        prob = predecir(modelo, escalador, valores)
        st.metric("Probabilidad de enfermedad cardíaca", f"{prob:.1%}")
        st.progress(float(prob))
        if prob >= 0.5:
            st.error("Predicción: **ENFERMO (1)**. Se recomienda evaluación médica.")
        else:
            st.success("Predicción: **SANO (0)**.")

# ---- Pestaña 2: rendimiento ----
with tab_rend:
    st.subheader("Métricas sobre los datos de prueba")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Exactitud (prueba)", f"{res['acc_test']:.2%}")
    m2.metric("Precisión", f"{res['precision']:.2f}")
    m3.metric("Recall", f"{res['recall']:.2f}")
    m4.metric("F1", f"{res['f1']:.2f}")
    m5.metric("AUC", f"{res['auc']:.2f}")
    st.caption(f"Exactitud en entrenamiento: {res['acc_train']:.2%} (la diferencia con la prueba indica sobreajuste).")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Sano', 'Pred: Enfermo'],
                    yticklabels=['Real: Sano', 'Real: Enfermo'])
        ax.set_title("Matriz de confusión")
        st.pyplot(fig)
    with g2:
        fpr, tpr, _ = res['roc']
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {res['auc']:.2f})")
        ax.plot([0, 1], [0, 1], '--', color='gray', label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curva ROC")
        ax.legend()
        st.pyplot(fig)
    st.dataframe(res['reporte'])

# ---- Pestaña 3: importancia de variables ----
with tab_var:
    st.subheader("Importancia de las variables")
    fig, ax = plt.subplots(figsize=(7, 5))
    res['importancias'].plot.barh(ax=ax)
    ax.set_xlabel("Importancia")
    st.pyplot(fig)

# ---- Pestaña 4: datos ----
with tab_datos:
    st.subheader("Conjunto de datos preparado")
    st.write(f"{df.shape[0]} pacientes y {df.shape[1]} columnas (diagnosis: 0 = sano, 1 = enfermo).")
    st.dataframe(df.head(10))
    st.write("Distribución de la variable objetivo:")
    st.bar_chart(df['diagnosis'].value_counts().rename({0: "Sano (0)", 1: "Enfermo (1)"}))
