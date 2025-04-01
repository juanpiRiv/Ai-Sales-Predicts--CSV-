import streamlit as st
import pandas as pd
import google.generativeai as genai
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st  
# Configurar la API de Gemini
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 
genai.configure(api_key=GOOGLE_API_KEY)# Usar la clave API proporcionada
# --- Añadir manejo de errores para la configuración de Gemini ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash') # Renombrado
    gemini_available = True
except Exception as gemini_error:
    # Usar st.sidebar.error si es posible, o st.error como fallback
    try:
        st.sidebar.error(f"Error Gemini API: {gemini_error}. Funciones IA desactivadas.")
    except:
        st.error(f"Error Gemini API: {gemini_error}. Funciones IA desactivadas.")
    model_gemini = None
    gemini_available = False
# --- Fin manejo de errores Gemini ---


# --- Funciones Auxiliares ---
def clean_data(df):
    """Limpia y prepara los datos: maneja valores nulos."""
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    return df

def generate_gemini_text(prompt):
    """Genera texto usando la API de Gemini."""
    if not gemini_available or model_gemini is None:
         return "La API de Gemini no está disponible o no se configuró correctamente."
    try:
        # --- Añadir manejo de contenido bloqueado ---
        response = model_gemini.generate_content(prompt)
        if response.parts:
             return response.text
        else:
             # Intentar obtener información sobre el bloqueo si está disponible
             block_reason = "Razón desconocida"
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name
             return f"Error: La respuesta de Gemini fue bloqueada (Razón: {block_reason}). Intenta con un prompt diferente o revisa el contenido."
        # --- Fin manejo de contenido bloqueado ---
    except Exception as e:
        return f"Error al contactar la API de Gemini: {e}"

def calculate_metrics(y_true, y_pred):
    """Calcula MAE y RMSE."""
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')
    if y_true_aligned.empty or y_pred_aligned.empty:
        return None, None
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    return mae, rmse

def plot_decomposition(df, date_col, sales_col, model_type='additive', period=None):
    """Realiza y grafica la descomposición de series temporales."""
    try:
        temp_df = df.set_index(date_col)[sales_col].copy()
        if period is None and isinstance(temp_df.index, pd.DatetimeIndex):
             inferred_freq = pd.infer_freq(temp_df.index)
             freq_map = {'D': 7, 'W': 52, 'M': 12, 'MS': 12, 'Q': 4, 'QS': 4, 'A': 1, 'AS': 1}
             if inferred_freq in freq_map:
                 period = freq_map[inferred_freq]
                 st.info(f"Periodo inferido para descomposición: {period} (freq: {inferred_freq})")
             else:
                 st.warning("No se pudo inferir periodo, usando valor por defecto (7).")
                 period = 7
        elif period is None: period = 7

        if period is None or period < 2 or period >= len(temp_df):
             st.warning(f"Periodo inválido ({period}). Se requiere 1 < periodo < {len(temp_df)}.")
             return None

        decomposition = seasonal_decompose(temp_df, model=model_type, period=period, extrapolate_trend='freq')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=('Observado', 'Tendencia', 'Estacionalidad', 'Residuos'))
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observado'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Tendencia'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Estacionalidad'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residuos'), row=4, col=1)
        fig.update_layout(height=600, title_text=f"Descomposición ({model_type.capitalize()}) - Periodo: {period}")
        return fig
    except Exception as e:
        st.error(f"Error en descomposición: {e}")
        return None

# --- Funciones de Modelo ---
def run_prophet_model(df, date_col, sales_col, periods=30, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto'):
    fig = None; forecast_df = None; model_prophet = None; metrics = {'mae': None, 'rmse': None}
    try:
        prophet_df = df[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
        model_prophet = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
        model_prophet.fit(prophet_df)
        future = model_prophet.make_future_dataframe(periods=periods)
        forecast = model_prophet.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        fitted_values = model_prophet.predict(prophet_df)
        metrics['mae'], metrics['rmse'] = calculate_metrics(prophet_df['y'], fitted_values['yhat'])
        fig = px.line(forecast, x='ds', y='yhat', title='Predicciones Prophet')
        fig.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Histórico')
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Ventas Predichas")
    except Exception as e: st.error(f"Error en Prophet: {e}")
    return fig, forecast_df, model_prophet, metrics

def run_arima_model(df, date_col, sales_col, order=(5, 1, 0), periods=30):
    fig = None; predictions_df = None; model_arima_fit = None; metrics = {'mae': None, 'rmse': None}
    try:
        arima_df = df.set_index(date_col)[sales_col]
        freq = None
        try:
            inferred_freq = pd.infer_freq(arima_df.index)
            if inferred_freq:
                arima_df = arima_df.asfreq(inferred_freq); freq = inferred_freq
                st.sidebar.info(f"Freq. ARIMA: {freq}") # Mover a sidebar
            else: st.sidebar.warning("Freq. ARIMA no inferida.")
        except Exception as freq_error: st.sidebar.warning(f"Error infiriendo freq.: {freq_error}")

        arima_df_for_metrics = arima_df.copy()
        try:
            model_arima = ARIMA(arima_df, order=order); model_arima_fit = model_arima.fit()
        except Exception as fit_error:
            st.warning(f"Error ARIMA{order}, reintentando (1,1,1): {fit_error}"); order = (1, 1, 1)
            model_arima = ARIMA(arima_df, order=order); model_arima_fit = model_arima.fit()

        fitted_values = model_arima_fit.fittedvalues
        metrics['mae'], metrics['rmse'] = calculate_metrics(arima_df_for_metrics, fitted_values)

        predictions = model_arima_fit.forecast(steps=periods)
        # --- Inicio Bloque con Indentación Corregida ---
        if freq:
             try:
                 offset = pd.tseries.frequencies.to_offset(freq)
                 if offset: future_index = pd.date_range(start=arima_df.index[-1] + offset, periods=periods, freq=freq)
                 else: raise ValueError(f"Offset inválido para freq: {freq}")
                 predictions_df = pd.DataFrame({'Prediction': predictions}, index=future_index)
             except Exception as idx_error:
                 st.warning(f"Error índice futuro ({freq}): {idx_error}. Usando índice numérico.")
                 predictions_df = pd.DataFrame({'Prediction': predictions.values})
        else:
             # Este bloque else debe estar alineado con el if freq:
             st.warning("Usando índice numérico para predicciones ARIMA.")
             predictions_df = pd.DataFrame({'Prediction': predictions.values})
        # --- Fin Bloque con Indentación Corregida ---
        predictions_df.index.name = date_col

        fig = px.line(predictions_df, x=predictions_df.index, y='Prediction', title=f'Predicciones ARIMA{order}')
        fig.add_scatter(x=arima_df.index, y=arima_df.values, mode='lines', name='Histórico')
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Ventas Predichas")
    except Exception as e: st.error(f"Error en ARIMA: {e}")
    return fig, predictions_df, model_arima_fit, metrics, order # Devolver orden usado

# --- Función Principal de Streamlit ---
def main():
    st.set_page_config(layout="wide", page_title="Dashboard de Predicciones")
    st.title("📈 Dashboard Interactivo de Predicciones de Ventas")
    st.markdown("Carga tus datos, explóralos, configura modelos y obtén predicciones.")

    # Inicializar estado de sesión
    keys_to_init = {
        'df_cleaned': None, 'date_col': None, 'sales_col': None,
        'prophet_results': {'fig': None, 'forecast': None, 'model': None, 'metrics': {'mae': None, 'rmse': None}},
        'arima_results': {'fig': None, 'forecast': None, 'model': None, 'metrics': {'mae': None, 'rmse': None}, 'order': None}, # Añadir orden usado
        'date_format_input': "", 'date_conversion_success': False,
        'prophet_yearly': True, 'prophet_weekly': True, 'prophet_daily': False,
        'decomposition_period': 7
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Inicio Bloque Try Principal ---
    try:
        # --- Barra Lateral ---
        with st.sidebar:
            st.header("1. Carga y Configuración")
            uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

            if uploaded_file is not None:
                # Resetear estado si se carga un nuevo archivo
                # (Podría hacerse más granular si es necesario)
                # for key in st.session_state.keys():
                #     del st.session_state[key] # Comentado para mantener selecciones

                df = pd.read_csv(uploaded_file)
                st.success("Archivo cargado.")
                st.session_state.df_cleaned = clean_data(df.copy())
                cols = st.session_state.df_cleaned.columns
                st.session_state.date_col = st.selectbox("Columna de Fecha:", cols, index=cols.get_loc(st.session_state.date_col) if st.session_state.date_col in cols else 0)
                st.session_state.sales_col = st.selectbox("Columna de Ventas:", cols, index=cols.get_loc(st.session_state.sales_col) if st.session_state.sales_col in cols else 1)

                # Validación y Conversión de Fecha
                date_col = st.session_state.date_col
                df_temp = st.session_state.df_cleaned.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
                    st.warning(f"Convirtiendo '{date_col}'...")
                    try:
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col], dayfirst=True, errors='coerce')
                        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
                             st.error("Conversión auto falló.")
                             st.session_state.date_format_input = st.text_input("Formato fecha (ej. %d/%m/%Y):", value=st.session_state.date_format_input)
                             if st.button("Reintentar conversión"):
                                 if st.session_state.date_format_input:
                                     try:
                                         df_retry = st.session_state.df_cleaned.copy()
                                         df_retry[date_col] = pd.to_datetime(df_retry[date_col], format=st.session_state.date_format_input, errors='coerce')
                                         if pd.api.types.is_datetime64_any_dtype(df_retry[date_col]):
                                             df_temp = df_retry; st.session_state.date_conversion_success = True
                                             st.success("Conversión manual OK!"); st.experimental_rerun()
                                         else: st.error("Formato manual incorrecto.")
                                     except Exception as fmt_e: st.error(f"Error formato: {fmt_e}")
                                 else: st.warning("Ingresa un formato.")
                             st.session_state.date_conversion_success = False
                        else: st.session_state.date_conversion_success = True
                    except Exception as e: st.error(f"Error conversión: {e}"); st.session_state.date_conversion_success = False
                else: st.session_state.date_conversion_success = True

                # Procesar NaNs y ordenar
                if st.session_state.date_conversion_success:
                    if df_temp[date_col].isnull().any():
                        st.warning(f"Eliminando filas con fechas inválidas.")
                        df_temp.dropna(subset=[date_col], inplace=True)
                    if not df_temp.empty:
                        try:
                            df_temp = df_temp.sort_values(by=date_col).reset_index(drop=True)
                            st.session_state.df_cleaned = df_temp; st.success("Datos listos.")
                        except Exception as sort_error: st.error(f"Error al ordenar: {sort_error}"); st.session_state.df_cleaned = None
                    else: st.error("No quedan datos válidos."); st.session_state.df_cleaned = None
                else: st.session_state.df_cleaned = None

                # Validar columna de ventas
                if st.session_state.df_cleaned is not None:
                     sales_col = st.session_state.sales_col
                     if not pd.api.types.is_numeric_dtype(st.session_state.df_cleaned[sales_col]):
                         st.error(f"Columna '{sales_col}' debe ser numérica."); st.session_state.df_cleaned = None

            # Configuración Adicional (solo si hay datos)
            if st.session_state.df_cleaned is not None:
                 st.header("2. Configuración Adicional")
                 st.session_state.decomposition_period = st.number_input("Periodo Descomposición", min_value=2, max_value=len(st.session_state.df_cleaned)//2 if len(st.session_state.df_cleaned) > 4 else 7, value=st.session_state.decomposition_period, step=1, help="Obs. por ciclo estacional (ej. 7 diario, 12 mensual)")

        # --- Cuerpo Principal con Pestañas ---
        if st.session_state.df_cleaned is not None:
            tab_explore, tab_predict, tab_results = st.tabs(["📊 Exploración", "⚙️ Predicción", "📈 Resultados"])

            with tab_explore:
                st.subheader("Vista Previa de Datos Limpios y Ordenados")
                st.dataframe(st.session_state.df_cleaned)

                st.subheader("Descomposición de Series Temporales")
                st.markdown("Visualiza la tendencia, estacionalidad y residuos.")
                decomp_model_type = st.radio("Tipo de Modelo", ('additive', 'multiplicative'), horizontal=True, key="decomp_type")
                decomposition_fig = plot_decomposition(st.session_state.df_cleaned, st.session_state.date_col, st.session_state.sales_col, model_type=decomp_model_type, period=st.session_state.decomposition_period)
                if decomposition_fig: st.plotly_chart(decomposition_fig, use_container_width=True)
                else: st.warning("No se pudo generar gráfico de descomposición.")

                if gemini_available:
                     st.subheader("💡 Análisis con Gemini")
                     if st.button("Generar Descripción de Datos"):
                         with st.spinner("Contactando a Gemini..."):
                             prompt = f"Analiza brevemente estos datos de ventas (primeras 5 filas y descripción general) y menciona cualquier patrón obvio o punto interesante:\n{st.session_state.df_cleaned.head().to_string()}\n{st.session_state.df_cleaned.describe().to_string()}"
                             description = generate_gemini_text(prompt)
                             st.write(description)

            with tab_predict:
                st.subheader("Configuración de Modelos")
                col_params1, col_params2 = st.columns(2)
                with col_params1:
                    with st.expander("🔧 Parámetros Prophet", expanded=True):
                        prophet_periods = st.slider("Periodos predicción (días)", 1, 365, 30, key="prophet_periods")
                        st.session_state.prophet_yearly = st.checkbox("Est. Anual", value=st.session_state.prophet_yearly)
                        st.session_state.prophet_weekly = st.checkbox("Est. Semanal", value=st.session_state.prophet_weekly)
                        st.session_state.prophet_daily = st.checkbox("Est. Diaria", value=st.session_state.prophet_daily)
                with col_params2:
                    with st.expander("🔧 Parámetros ARIMA", expanded=True):
                        arima_p = st.number_input("Orden p (AR)", 0, 10, 5, key="arima_p")
                        arima_d = st.number_input("Orden d (Diff)", 0, 5, 1, key="arima_d")
                        arima_q = st.number_input("Orden q (MA)", 0, 10, 0, key="arima_q")
                        arima_periods = st.slider("Periodos predicción", 1, 365, 30, key="arima_periods")
                        arima_order = (arima_p, arima_d, arima_q)

                if st.button("🚀 Realizar Predicciones", type="primary", use_container_width=True):
                    date_col = st.session_state.date_col; sales_col = st.session_state.sales_col; df_cleaned = st.session_state.df_cleaned
                    with st.spinner("Entrenando Prophet..."):
                        prophet_fig, prophet_fc, prophet_model, prophet_metrics = run_prophet_model(df_cleaned, date_col, sales_col, periods=prophet_periods, yearly_seasonality=st.session_state.prophet_yearly, weekly_seasonality=st.session_state.prophet_weekly, daily_seasonality=st.session_state.prophet_daily)
                        st.session_state.prophet_results = {'fig': prophet_fig, 'forecast': prophet_fc, 'model': prophet_model, 'metrics': prophet_metrics}
                    with st.spinner("Entrenando ARIMA..."):
                         arima_fig, arima_fc, arima_model, arima_metrics, final_arima_order = run_arima_model(df_cleaned, date_col, sales_col, order=arima_order, periods=arima_periods)
                         st.session_state.arima_results = {'fig': arima_fig, 'forecast': arima_fc, 'model': arima_model, 'metrics': arima_metrics, 'order': final_arima_order} # Guardar orden final
                    st.success("Modelos entrenados y predicciones realizadas!")
                    st.info("Ve a la pestaña 'Resultados' para ver los gráficos y métricas.")


            with tab_results:
                st.subheader("Visualización de Predicciones")
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.markdown("**Prophet**")
                    if st.session_state.prophet_results['fig']:
                        st.plotly_chart(st.session_state.prophet_results['fig'], use_container_width=True)
                        with st.expander("📊 Métricas y Resumen IA (Prophet)"):
                            mae = st.session_state.prophet_results['metrics']['mae']; rmse = st.session_state.prophet_results['metrics']['rmse']
                            st.metric("MAE", f"{mae:.2f}" if mae is not None else "N/A"); st.metric("RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")
                            st.caption("Métricas In-Sample (ajuste sobre datos históricos).")
                            if gemini_available and st.button("Generar Resumen IA (Prophet)", key="gemini_prophet"):
                                 with st.spinner("Generando resumen..."):
                                     forecast_tail = st.session_state.prophet_results['forecast'].tail().to_string()
                                     prompt = f"Interpreta brevemente estos resultados del modelo Prophet:\nMétricas In-Sample: MAE={mae:.2f}, RMSE={rmse:.2f}\nÚltimas predicciones:\n{forecast_tail}\n¿Qué indican las métricas sobre el ajuste? ¿Cuál es la tendencia general del pronóstico?"
                                     summary = generate_gemini_text(prompt); st.markdown("**Resumen Gemini:**"); st.write(summary)
                    else: st.info("Ejecuta las predicciones para ver resultados.")

                with col_viz2:
                    order_str = f"ARIMA{st.session_state.arima_results['order']}" if st.session_state.arima_results['order'] else "ARIMA"
                    st.markdown(f"**{order_str}**")
                    if st.session_state.arima_results['fig']:
                        st.plotly_chart(st.session_state.arima_results['fig'], use_container_width=True)
                        with st.expander(f"📊 Métricas y Resumen IA ({order_str})"):
                            mae = st.session_state.arima_results['metrics']['mae']; rmse = st.session_state.arima_results['metrics']['rmse']
                            st.metric("MAE", f"{mae:.2f}" if mae is not None else "N/A"); st.metric("RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")
                            st.caption("Métricas In-Sample (ajuste sobre datos históricos).")
                            if gemini_available and st.button(f"Generar Resumen IA ({order_str})", key="gemini_arima"):
                                 with st.spinner("Generando resumen..."):
                                     forecast_tail = st.session_state.arima_results['forecast'].tail().to_string()
                                     prompt = f"Interpreta brevemente estos resultados del modelo {order_str}:\nMétricas In-Sample: MAE={mae:.2f}, RMSE={rmse:.2f}\nÚltimas predicciones:\n{forecast_tail}\n¿Qué indican las métricas sobre el ajuste? ¿Cuál es la tendencia general del pronóstico?"
                                     summary = generate_gemini_text(prompt); st.markdown("**Resumen Gemini:**"); st.write(summary)
                    else: st.info("Ejecuta las predicciones para ver resultados.")

                st.subheader("Descargar Predicciones")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    if st.session_state.prophet_results['forecast'] is not None:
                        prophet_csv = st.session_state.prophet_results['forecast'].to_csv(index=False).encode('utf-8')
                        st.download_button("Descargar Prophet CSV", prophet_csv, 'prophet_predictions.csv', 'text/csv', key='dl-prophet')
                with col_dl2:
                    if st.session_state.arima_results['forecast'] is not None:
                        arima_csv = st.session_state.arima_results['forecast'].to_csv().encode('utf-8')
                        st.download_button("Descargar ARIMA CSV", arima_csv, 'arima_predictions.csv', 'text/csv', key='dl-arima')

        # Mensajes condicionales fuera de las pestañas
        elif uploaded_file is not None and not st.session_state.date_conversion_success:
            st.warning("Esperando la especificación del formato de fecha o la corrección del archivo en la barra lateral.")
        elif uploaded_file is None:
            st.info("👈 Carga un archivo CSV desde la barra lateral para comenzar.")
        elif st.session_state.df_cleaned is None:
             st.error("Los datos no pudieron ser procesados. Verifica las selecciones en la barra lateral y el archivo.")

    # --- Fin Bloque Try Principal ---
    except Exception as e:
        st.error(f"Ocurrió un error general en la aplicación: {e}")
        st.exception(e) # Mostrar traceback completo para depuración


if __name__ == "__main__":
    main()
