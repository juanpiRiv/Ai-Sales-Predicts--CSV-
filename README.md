# Dashboard Interactivo de Predicciones de Ventas IA

Este proyecto es una aplicación web interactiva construida con Streamlit que permite a los usuarios cargar sus propios datos de ventas (en formato CSV), explorar visualizaciones de los datos, realizar análisis de descomposición de series temporales y generar predicciones utilizando los modelos Prophet y ARIMA.

La aplicación también integra la API de Google Gemini para generar resúmenes e interpretaciones de los datos y los resultados de los modelos (si se configura una clave API).

## Características Principales

*   **Carga de Datos:** Sube archivos CSV con tus datos de ventas.
*   **Selección de Columnas:** Elige fácilmente las columnas correspondientes a fechas y ventas.
*   **Limpieza Automática:** Manejo básico de valores nulos en columnas numéricas.
*   **Conversión de Fechas:** Intenta convertir automáticamente la columna de fecha, con opción de especificar formato manualmente.
*   **Exploración Visual:**
    *   Vista previa de los datos limpios y ordenados.
    *   Gráfico interactivo de descomposición de series temporales (tendencia, estacionalidad, residuos) con modelos aditivo o multiplicativo.
*   **Modelado y Predicción:**
    *   Configura y ejecuta el modelo **Prophet** (ajustando estacionalidades y periodos de predicción).  Prophet es un modelo de Facebook que se utiliza para predecir series temporales con estacionalidad.
    *   Configura y ejecuta el modelo **ARIMA** (ajustando órdenes p, d, q y periodos de predicción). ARIMA es un modelo estadístico que utiliza los valores pasados de una serie temporal para predecir los valores futuros.
*   **Visualización de Resultados:**
# Dashboard Interactivo de Predicciones de Ventas IA

Este proyecto es una aplicación web interactiva construida con Streamlit que permite a los usuarios cargar sus propios datos de ventas (en formato CSV), explorar visualizaciones de los datos, realizar análisis de descomposición de series temporales y generar predicciones utilizando los modelos Prophet y ARIMA.

La aplicación también integra la API de Google Gemini para generar resúmenes e interpretaciones de los datos y los resultados de los modelos (si se configura una clave API).

## Características Principales

*   **Carga de Datos:** Sube archivos CSV con tus datos de ventas.
*   **Selección de Columnas:** Elige fácilmente las columnas correspondientes a fechas y ventas.
*   **Limpieza Automática:** Manejo básico de valores nulos en columnas numéricas.
*   **Conversión de Fechas:** Intenta convertir automáticamente la columna de fecha, con opción de especificar formato manualmente.
*   **Exploración Visual:**
    *   Vista previa de los datos limpios y ordenados.
    *   Gráfico interactivo de descomposición de series temporales (tendencia, estacionalidad, residuos) con modelos aditivo o multiplicativo.
*   **Modelado y Predicción:**
    *   Configura y ejecuta el modelo **Prophet** (ajustando estacionalidades y periodos de predicción).
    *   Configura y ejecuta el modelo **ARIMA** (ajustando órdenes p, d, q y periodos de predicción).
*   **Visualización de Resultados:**
    *   Gráficos interactivos comparando datos históricos y predicciones para ambos modelos.
    *   Métricas de ajuste (MAE, RMSE) calculadas sobre los datos históricos (in-sample).
*   **Integración con IA (Opcional):**
    *   Genera descripciones automáticas de los datos cargados usando Google Gemini.
    *   Genera resúmenes interpretativos de los resultados de los modelos Prophet y ARIMA usando Google Gemini.
*   **Descarga de Resultados:** Descarga las predicciones generadas por cada modelo en formato CSV.

## Requisitos Previos

*   Python 3.7 o superior.
*   `pip` (gestor de paquetes de Python).

## Instalación y Configuración

1.  **Clonar el Repositorio (Opcional):**
    Si obtuviste el código como un repositorio git:
    ```bash
    git clone <url-del-repositorio>
    cd <nombre-del-directorio>
    ```
    Si solo tienes los archivos, navega hasta el directorio donde se encuentran `app.py` y `requirements.txt`.

2.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Activar el entorno:
    # Windows (cmd):
    venv\Scripts\activate
    # Windows (PowerShell):
    # venv\Scripts\Activate.ps1
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    Asegúrate de que tu entorno virtual esté activado y ejecuta:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar la Clave API de Google Gemini (Opcional):**
    Si deseas utilizar las funciones de análisis con IA, necesitas una clave API de Google Gemini.
    *   Crea un directorio llamado `.streamlit` en la raíz del proyecto si no existe.
    *   Dentro de `.streamlit`, crea un archivo llamado `secrets.toml`.
    *   Añade tu clave API al archivo `secrets.toml` de la siguiente manera:
        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "TU_CLAVE_API_AQUI"
        ```
    *   Reemplaza `"TU_CLAVE_API_AQUI"` con tu clave API real. Si no proporcionas una clave, la aplicación funcionará pero las funciones de IA estarán desactivadas.

## Cómo Usar la Aplicación

1.  **Ejecutar la Aplicación Streamlit:**
    Desde el directorio raíz del proyecto (donde está `app.py`), ejecuta el siguiente comando en tu terminal (asegúrate de que el entorno virtual esté activado):
    ```bash
    streamlit run app.py
    ```
    Esto debería abrir la aplicación automáticamente en tu navegador web.

2.  **Cargar Datos:**
    *   Usa el botón "Cargar archivo CSV" en la barra lateral para subir tu archivo de datos.
    *   El archivo debe contener al menos una columna de fechas y una columna numérica de ventas.

3.  **Configurar Columnas:**
    *   Selecciona la columna que contiene las fechas en el desplegable "Columna de Fecha:".
    *   Selecciona la columna que contiene los valores de ventas en el desplegable "Columna de Ventas:".
    *   Si la columna de fecha no se reconoce automáticamente, la aplicación te pedirá que ingreses el formato correcto (ej., `%d/%m/%Y`, `%Y-%m-%d`).

4.  **Explorar Datos (Pestaña 📊 Exploración):**
    *   Revisa la tabla con los datos limpios y ordenados.
    *   Analiza el gráfico de descomposición. Puedes cambiar el tipo de modelo (aditivo/multiplicativo) y el periodo de descomposición en la barra lateral.
    *   Si configuraste la API de Gemini, puedes hacer clic en "Generar Descripción de Datos".

5.  **Realizar Predicciones (Pestaña ⚙️ Predicción):**
    *   Ajusta los parámetros para los modelos Prophet y ARIMA (periodos de predicción, estacionalidades, órdenes ARIMA).
    *   Haz clic en el botón "🚀 Realizar Predicciones". La aplicación entrenará ambos modelos.

6.  **Ver Resultados (Pestaña 📈 Resultados):**
    *   Observa los gráficos de predicción para Prophet y ARIMA.
    *   Expande las secciones de métricas para ver el MAE y RMSE del ajuste.
    *   Si configuraste la API de Gemini, puedes generar resúmenes interpretativos para cada modelo.
    *   Utiliza los botones "Descargar ... CSV" para guardar las predicciones.
