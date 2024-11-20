import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_card import card
import pydeck as pdk
from streamlit_metrics import metric, metric_row
from streamlit_echarts import st_echarts
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(
    page_title="REGRESIÓN LOGÍSTICA DINÁMICA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox {
        background-color: white;
        border-radius: 5px;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .explanation-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar variables en session_state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.theta_0 = 0.0
    st.session_state.theta_1 = 0.0
    st.session_state.iteration = 0
    st.session_state.precision = 0.0
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.df = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.training_completed = False
    st.session_state.scaler = StandardScaler()

# Menú de navegación con streamlit-option-menu
selected = option_menu(
    menu_title="Navegación",
    options=["Inicio", "Datos", "Preprocesamiento", "Entrenamiento", "Evaluación", "Predicción"],
    icons=["house", "table", "gear", "play-circle", "graph-up", "check-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Funciones auxiliares
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta_0, theta_1):
    return sigmoid(theta_0 + theta_1 * X)

def validate_data_dimensions(X, y):
    if X is None or y is None:
        return False, "Datos no inicializados"
    
    try:
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if X.size == 0 or y.size == 0:
            return False, "Los arrays están vacíos"
        
        if X.shape[0] != y.shape[0]:
            return False, f"Dimensiones inconsistentes: X tiene {X.shape[0]} muestras, y tiene {y.shape[0]} muestras"
        
        return True, (X, y)
    except Exception as e:
        return False, f"Error al validar dimensiones: {str(e)}"

def create_explanation_box(title, content):
    """Crear una caja de explicación con estilo consistente"""
    st.markdown(f"""
    <div class="explanation-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Página de inicio
if selected == "Inicio":
    st.title("🎯 REGRESIÓN LOGÍSTICA DINÁMICA")
    
    # Información introductoria con cards
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            card(
                title="¿Qué es la Regresión Logística?",
                text="La regresión logística es un método estadístico para predecir variables binarias. Es ampliamente utilizado en machine learning para clasificación.",
                image="https://placeholder.com/300x200",
                styles={
                    "card": {"width": "100%", "height": "300px"},
                    "text": {"fontSize": "1rem"}
                }
            )
        with col2:
            card(
                title="Características Principales",
                text="• Clasificación binaria\n• Interpretabilidad\n• Probabilidades como output\n• Eficiente computacionalmente",
                styles={
                    "card": {"width": "100%", "height": "300px"},
                    "text": {"fontSize": "1rem"}
                }
            )
    

# Página de datos
elif selected == "Datos":
    st.title("📊 Carga y Exploración de Datos")
    
    # Subir archivo de datos
    archivo_cargado = st.file_uploader("Sube tu archivo de datos (Excel o CSV)", type=["xlsx", "xls", "csv"])
    
    if archivo_cargado:
        try:
            # Leer archivo
            if archivo_cargado.name.endswith('.csv'):
                df = pd.read_csv(archivo_cargado)
            else:
                df = pd.read_excel(archivo_cargado)
            st.session_state.df = df
            
            # Vista interactiva de datos con st-aggrid
            st.subheader("Vista Previa de Datos Interactiva")
            create_explanation_box(
                "Tabla de Datos",
                "Esta tabla interactiva permite explorar, filtrar y ordenar los datos. "
                "Usa los controles en la parte superior para personalizar la vista."
            )
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection('multiple', use_checkbox=True)
            grid_options = gb.build()
            AgGrid(
                df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                height=400,
                width='100%'
            )

            # Mostrar métricas del dataset
            st.subheader("📋 Métricas del Dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Registros", df.shape[0])
            with col2:
                st.metric("Variables", df.shape[1])
            with col3:
                st.metric("Valores Faltantes", df.isnull().sum().sum())

            # Análisis Exploratorio
            st.subheader("📈 Análisis Exploratorio")
            
            # Distribución de variables numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Distribución de Variables Numéricas**")
                create_explanation_box(
                    "Distribución de Variables Numéricas",
                    "Este gráfico muestra la distribución de las variables numéricas en el dataset. "
                    "Las barras representan la frecuencia de cada valor."
                )
                
                for col in numeric_cols:
                    option = {
                        "title": {"text": f"Distribución de {col}"},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "category"},
                        "yAxis": {"type": "value"},
                        "series": [{
                            "data": df[col].value_counts().sort_index().tolist(),
                            "type": "bar"
                        }]
                    }
                    st_echarts(options=option, height="400px")
            
            # Mapa de calor de correlación
            if len(numeric_cols) > 1:
                st.write("**Mapa de Calor de Correlación**")
                create_explanation_box(
                    "Mapa de Calor de Correlación",
                    "Este gráfico muestra la correlación entre las columnas numéricas. "
                    "Un valor cercano a 1 o -1 indica una relación fuerte positiva o negativa, respectivamente. "
                    "Sirve para identificar relaciones entre variables que pueden ser útiles para el modelo."
                )
                corr_matrix = df[numeric_cols].corr().round(2).values.tolist()
                option = {
                    "tooltip": {"position": "top"},
                    "xAxis": {
                        "type": "category",
                        "data": numeric_cols.tolist(),
                        "splitArea": {"show": True},
                    },
                    "yAxis": {
                        "type": "category",
                        "data": numeric_cols.tolist(),
                        "splitArea": {"show": True},
                    },
                    "visualMap": {
                        "min": -1,
                        "max": 1,
                        "calculable": True,
                        "orient": "horizontal",
                        "left": "center",
                        "bottom": "15%",
                    },
                    "series": [
                        {
                            "name": "Correlación",
                            "type": "heatmap",
                            "data": [
                                [i, j, corr_matrix[i][j]]
                                for i in range(len(corr_matrix))
                                for j in range(len(corr_matrix))
                            ],
                            "label": {"show": True},
                        }
                    ],
                }
                st_echarts(options=option, height="500px")
            
            # Relación entre variables
            st.write("**Relación entre Variables**")
            create_explanation_box(
                "Relación entre Variables",
                "Selecciona dos variables numéricas para explorar gráficamente su relación. "
                "Esto ayuda a identificar patrones o tendencias entre las variables seleccionadas."
            )

            if len(numeric_cols) > 1:
                # Seleccionar dos variables
                selected_vars = st.multiselect(
                    "Selecciona dos variables para comparar",
                    options=numeric_cols,
                    default=numeric_cols[:2],  # Preseleccionar las primeras dos variables
                    max_selections=2  # Limitar la selección a dos variables
                )

                if len(selected_vars) == 2:
                    col1, col2 = selected_vars
                    scatter_data = df[[col1, col2]].dropna().values.tolist()
                    option = {
                        "title": {"text": f"{col1} vs {col2}"},
                        "tooltip": {"trigger": "item"},
                        "xAxis": {"type": "value", "name": col1},
                        "yAxis": {"type": "value", "name": col2},
                        "series": [
                            {
                                "data": scatter_data,
                                "type": "scatter",
                                "symbolSize": 10,
                            }
                        ],
                    }
                    st_echarts(options=option, height="400px")
                else:
                    st.warning("Por favor selecciona exactamente dos variables para graficar.")
            else:
                st.warning("No hay suficientes variables numéricas para graficar relaciones.")
        
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
    else:
        st.info("👆 Sube un archivo de datos para comenzar el análisis.")

# Página de preprocesamiento
elif selected == "Preprocesamiento":
    st.title("⚙️ Preprocesamiento de Datos")

    if st.session_state.df is not None:
        # Selección de variables
        st.subheader("🎯 Selección de Variables")
        
        create_explanation_box(
            "Selección de Variables",
            "Selecciona las variables independientes (X) que serán usadas para predecir "
            "la variable dependiente (y). Las variables independientes deben ser numéricas."
        )

        columnas_numericas = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) > 0:
            # Selección de variables independientes y dependiente
            X_columns = st.multiselect(
                "Variables Independientes (X)",
                columnas_numericas,
                help="Selecciona las variables que usarás para hacer predicciones"
            )
            
            Y_column = st.selectbox(
                "Variable Dependiente (y)",
                st.session_state.df.columns,
                help="Selecciona la variable que quieres predecir"
            )

            if X_columns and Y_column:
                # Preparar datos
                X = st.session_state.df[X_columns].values
                y = st.session_state.df[Y_column].values

                # División de datos
                st.subheader("📊 División de Datos")
                create_explanation_box(
                    "División Entrenamiento/Prueba",
                    "Los datos se dividen en conjuntos de entrenamiento y prueba para evaluar "
                    "el rendimiento del modelo en datos no vistos."
                )
                
                test_size = st.slider(
                    "Proporción de datos de prueba",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05
                )

                # Realizar división de datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Escalar datos
                X_train_scaled = st.session_state.scaler.fit_transform(X_train)
                X_test_scaled = st.session_state.scaler.transform(X_test)

                # Guardar en session_state
                st.session_state.X_train = X_train_scaled
                st.session_state.X_test = X_test_scaled
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.X_columns = X_columns
                st.session_state.Y_column = Y_column

                # Mostrar dimensiones de los datos
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Muestras de Entrenamiento", X_train.shape[0])
                with col2:
                    st.metric("Muestras de Prueba", X_test.shape[0])

                st.success("✅ Datos preparados exitosamente")

                # Gráficos de tendencia logística
                st.subheader("📊 Tendencia Logística por Variable")
                create_explanation_box(
                    "Tendencia Logística",
                    "Estos gráficos muestran cómo cambia la probabilidad de la variable dependiente "
                    "en función de cada variable independiente seleccionada, ajustando una curva logística."
                )

                for i, col in enumerate(X_columns):
                    # Crear modelo de regresión logística para la variable
                    model = LogisticRegression()
                    model.fit(X_train_scaled[:, [i]], y_train)

                    # Crear datos para la curva
                    x_range = np.linspace(
                        X_train_scaled[:, i].min(), 
                        X_train_scaled[:, i].max(), 
                        300
                    )
                    y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]

                    # Crear gráfico interactivo
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=X_train_scaled[:, i],
                        y=y_train,
                        mode='markers',
                        name="Datos Reales",
                        marker=dict(color="blue", opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_prob,
                        mode='lines',
                        name="Tendencia Logística",
                        line=dict(color="red", width=2)
                    ))

                    # Configuración del gráfico
                    fig.update_layout(
                        title=f"Tendencia Logística - {col}",
                        xaxis_title=col,
                        yaxis_title=f"Probabilidad de {Y_column}",
                        height=400
                    )

                    st.plotly_chart(fig)

            else:
                st.warning("Selecciona las variables independientes y dependiente primero.")
        else:
            st.warning("No hay variables numéricas disponibles en el dataset.")
    else:
        st.warning("Carga un archivo de datos primero en la sección 'Datos'.")

# Página de entrenamiento
elif selected == "Entrenamiento":
    st.title("🎯 Entrenamiento del Modelo")

    if st.session_state.X_train is not None:
        # Parámetros de entrenamiento
        st.subheader("⚙️ Parámetros de Entrenamiento")
        
        create_explanation_box(
            "Configuración del Modelo",
            "Ajusta los parámetros del modelo para optimizar su rendimiento. "
            "El tamaño del bootstrap determina cuántos datos se usan en cada iteración. "
            "Puedes configurar el número máximo de iteraciones y ajustar las iteraciones actuales con el control deslizante."
        )

        # Configuración del número máximo de iteraciones
        max_iteraciones = st.number_input(
            "Número Máximo de Iteraciones",
            min_value=1,
            max_value=1000,
            value=50,
            step=1,
            help="Ingresa el número máximo de iteraciones que deseas realizar."
        )

        # Configuración del tamaño del bootstrap
        n_samples_bootstrap = st.slider(
            "Tamaño del Bootstrap",
            min_value=10,
            max_value=len(st.session_state.X_train),
            value=min(100, len(st.session_state.X_train)),
            step=10,
            help="Número de muestras a usar para el entrenamiento en cada iteración."
        )

        # Inicialización del modelo
        if st.button("🚀 Inicializar Modelo"):
            st.session_state.model = LogisticRegression(solver='liblinear', max_iter=1, warm_start=True)
            st.session_state.iteration = 0
            st.session_state.accuracy = 0
            
            # Datos bootstrap iniciales
            indices = np.random.choice(len(st.session_state.X_train), n_samples_bootstrap)
            st.session_state.X_bootstrap = st.session_state.X_train[indices]
            st.session_state.y_bootstrap = st.session_state.y_train[indices]
            
            st.success("✅ Modelo inicializado correctamente")

        # Entrenamiento en tiempo real
        st.subheader("📈 Entrenamiento en Tiempo Real")
        create_explanation_box(
            "Entrenamiento Continuo",
            "El gráfico se actualiza automáticamente mientras ajustas el número de iteraciones. "
            "Esto te permite visualizar cómo cambia la curva de decisión del modelo en tiempo real."
        )

        # Control deslizante para iteraciones actuales
        iteracion_actual = st.slider(
            "Selecciona Iteraciones Actuales",
            min_value=0,
            max_value=max_iteraciones,
            value=0,
            step=1,
            help="Controla el número de iteraciones actuales para entrenar el modelo en tiempo real."
        )

        # Entrenamiento solo si se aumenta el número de iteraciones
        if st.session_state.model is not None and iteracion_actual > st.session_state.iteration:
            for _ in range(iteracion_actual - st.session_state.iteration):
                # Actualizar bootstrap
                indices = np.random.choice(len(st.session_state.X_train), n_samples_bootstrap)
                X_bootstrap = st.session_state.X_train[indices]
                y_bootstrap = st.session_state.y_train[indices]

                # Entrenar el modelo
                st.session_state.model.fit(X_bootstrap, y_bootstrap)

                # Incrementar contador de iteraciones
                st.session_state.iteration += 1

                # Calcular precisión
                y_pred = st.session_state.model.predict(X_bootstrap)
                st.session_state.accuracy = accuracy_score(y_bootstrap, y_pred) * 100

            # Visualización interactiva de la curva de decisión
            coef = st.session_state.model.coef_[0][0]
            intercept = st.session_state.model.intercept_[0]
            x_vals = np.linspace(X_bootstrap[:, 0].min(), X_bootstrap[:, 0].max(), 100)
            y_vals = sigmoid(coef * x_vals + intercept)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curva de Decisión'))
            fig.add_trace(go.Scatter(
                x=X_bootstrap[:, 0],
                y=y_bootstrap,
                mode='markers',
                marker=dict(color=y_bootstrap, colorscale='Viridis', size=10),
                name='Datos de Entrenamiento'
            ))
            fig.add_hline(y=0.5, line=dict(color="red", dash="dash"), name="Límite de Decisión")
            fig.update_layout(
                title=f"Curva de Decisión - Iteraciones: {st.session_state.iteration}",
                xaxis_title="Variable Independiente Escalada",
                yaxis_title="Probabilidad de Clase",
                height=600
            )
            st.plotly_chart(fig)

            # Explicación del gráfico
            create_explanation_box(
                "Curva de Decisión",
                "Esta curva representa cómo el modelo separa las clases basándose en las probabilidades predichas. "
                "El límite de decisión se establece en una probabilidad del 50%, dividiendo las predicciones en dos clases."
            )

        # Finalizar entrenamiento
        if st.button("✅ Finalizar Entrenamiento"):
            st.session_state.training_completed = True
            st.success("Entrenamiento finalizado. Puedes proceder a la Evaluación.")
    else:
        st.warning("⚠️ Asegúrate de preprocesar los datos antes de entrenar el modelo.")

# 4. Evaluación
elif selected == "Evaluación":
    st.title("📊 Evaluación del Modelo")

    if st.session_state.get('training_completed', False):
        # Calcular métricas y gráficos de evaluación
        st.subheader("⚙️ Evaluación del Modelo")
        
        create_explanation_box(
            "¿Por qué evaluar el modelo?",
            "La evaluación permite analizar el rendimiento del modelo en un conjunto de datos que no se usaron durante el entrenamiento. "
            "Esto asegura que el modelo generaliza correctamente y no está sobreajustado a los datos de entrenamiento."
        )

        # Predicciones sobre los datos de prueba
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        y_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]

        # Métricas generales
        st.metric("Precisión", f"{accuracy_score(st.session_state.y_test, y_pred) * 100:.2f} %")
        
        # Matriz de confusión
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        cm = cm.tolist()  # Convertir a lista para compatibilidad con JSON
        cm_labels = ["Clase 0", "Clase 1"]

        create_explanation_box(
            "Matriz de Confusión",
            "La matriz de confusión muestra cómo el modelo clasifica las observaciones. "
            "Indica cuántos verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos predice."
        )

        # Gráfico interactivo de la matriz de confusión
        option_cm = {
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "category", "data": cm_labels},
            "yAxis": {"type": "category", "data": cm_labels},
            "visualMap": {"min": 0, "max": max([max(row) for row in cm])},
            "series": [
                {
                    "type": "heatmap",
                    "data": [[i, j, int(cm[i][j])] for i in range(len(cm)) for j in range(len(cm[0]))],  # Convertir valores a int
                    "label": {"show": True},
                }
            ],
        }
        st_echarts(options=option_cm, height="400px")

        # Curva ROC
        st.subheader("Curva ROC y AUC")
        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
        auc_score = roc_auc_score(st.session_state.y_test, y_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Línea Base'))
        fig_roc.update_layout(
            title=f"Curva ROC (AUC = {auc_score:.2f})",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            height=400
        )
        st.plotly_chart(fig_roc)

        create_explanation_box(
            "Curva ROC",
            "La Curva ROC mide el rendimiento del modelo al variar el umbral de decisión. "
            "Un AUC más alto indica mejor capacidad de predicción."
        )

        # Reporte de clasificación
        st.subheader("Reporte de Clasificación")
        classification_rep = classification_report(
            st.session_state.y_test, y_pred, output_dict=True, target_names=cm_labels
        )
        report_df = pd.DataFrame(classification_rep).transpose()

        create_explanation_box(
            "Reporte de Clasificación",
            "Este reporte incluye métricas como precisión, sensibilidad, y F1-Score para cada clase. "
            "Es útil para evaluar el rendimiento del modelo en cada categoría."
        )
        AgGrid(report_df.reset_index(), height=300, fit_columns_on_grid_load=True)
        
    else:
        st.warning("⚠️ Finaliza el entrenamiento antes de evaluar el modelo.")


# Evaluación y Predicción (secciones restantes del código original)
# Aquí ya puedes pasar a las otras secciones, como Evaluación o Predicción,
# reutilizando las funcionalidades y estructuras introducidas anteriormente.
 
# 5. Predicción Independiente
elif selected == "Predicción":
    st.title("🔮 Predicción Independiente")

    if st.session_state.model is not None and st.session_state.scaler is not None:
        create_explanation_box(
            "¿Cómo realizar predicciones?",
            "Proporciona valores para las variables independientes seleccionadas. "
            "El modelo usará estos valores para predecir la clase y la probabilidad asociada."
        )

        # Campos de entrada dinámicos para predicción
        valores_input = []
        for col in st.session_state.X_columns:
            valor = st.number_input(f"Valor para {col}", value=0.0)
            valores_input.append(valor)

        # Botón para realizar la predicción
        if st.button("📈 Predecir"):
            # Escalar los valores
            valores_scaled = st.session_state.scaler.transform([valores_input])
            
            # Realizar la predicción
            pred = st.session_state.model.predict(valores_scaled)[0]
            prob = st.session_state.model.predict_proba(valores_scaled)[0][1]

            # Mostrar resultados
            resultado = "Clase 1 (Positiva)" if pred == 1 else "Clase 0 (Negativa)"
            st.metric("Resultado de la Predicción", resultado)
            st.metric("Probabilidad de Clase Positiva", f"{prob:.2%}")

            # Visualización de la predicción
            st.subheader("Visualización de la Región de Decisión")
            coef = st.session_state.model.coef_[0]
            intercept = st.session_state.model.intercept_

            # Generar malla para gráficos 2D (asumiendo 2 variables independientes)
            if len(coef) == 2:
                x_min, x_max = st.session_state.X_train[:, 0].min() - 1, st.session_state.X_train[:, 0].max() + 1
                y_min, y_max = st.session_state.X_train[:, 1].min() - 1, st.session_state.X_train[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                z = sigmoid(np.dot(grid, coef) + intercept)
                z = z.reshape(xx.shape)

                # Crear figura interactiva
                fig_pred = go.Figure()

                # Región de decisión
                fig_pred.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=z,
                    showscale=True,
                    colorscale="Viridis",
                    name="Región de Decisión"
                ))

                # Puntos de entrenamiento
                fig_pred.add_trace(go.Scatter(
                    x=st.session_state.X_train[:, 0],
                    y=st.session_state.X_train[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=st.session_state.y_train,
                        colorscale="Rainbow",
                        line=dict(width=1)
                    ),
                    name="Datos de Entrenamiento"
                ))

                # Punto predicho
                fig_pred.add_trace(go.Scatter(
                    x=[valores_scaled[0, 0]],
                    y=[valores_scaled[0, 1]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color="yellow",
                        symbol="x"
                    ),
                    name="Nuevo Punto"
                ))

                fig_pred.update_layout(
                    title="Región de Decisión para la Predicción",
                    xaxis_title=st.session_state.X_columns[0],
                    yaxis_title=st.session_state.X_columns[1],
                    legend=dict(x=0, y=1),
                    height=600
                )
                st.plotly_chart(fig_pred)

                # Explicación adicional del gráfico
                create_explanation_box(
                    "Región de Decisión y Predicción",
                    "El gráfico muestra la región de decisión del modelo. Los colores indican la probabilidad de clasificación "
                    "en cada área. El punto amarillo representa la predicción realizada basándose en los valores ingresados."
                )
            else:
                st.warning("⚠️ Este gráfico solo está disponible para modelos con 2 variables independientes.")
    else:
        st.warning("⚠️ Asegúrate de que el modelo esté entrenado y escalado antes de realizar predicciones.")

# Función sigmoid corregida
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
