# Proyecto Data Science - Análisis Integral de Empresas Públicas

## Abstracto, Motivación y Audiencia

Este proyecto se centra en el análisis integral de datos de empresas públicas, combinando información de precios históricos, datos corporativos y estados financieros para determinar si, en cada período, las acciones se encuentran sobrevaloradas o infravaloradas. La motivación principal es generar insights objetivos y basados en datos sólidos que faciliten la toma de decisiones estratégicas en entornos de inversión y análisis financiero. Se busca identificar patrones y relaciones entre indicadores fundamentales y el comportamiento del precio, permitiendo optimizar modelos de Machine Learning para predecir el valor de las acciones.

La audiencia beneficiaria de este análisis incluye:
- **Inversionistas y analistas financieros:** para evaluar oportunidades de inversión basadas en la relación precio-fundamentales.
- **Ejecutivos y equipos de estrategia:** que requieran un marco analítico robusto para la toma de decisiones.
- **Equipos de Research y Data Science:** que buscan integrar múltiples fuentes de datos y aplicar técnicas avanzadas para la generación de insights.


---

## 1. Contexto y Metodología

El proyecto parte de la necesidad de **entrenar y optimizar modelos de Machine Learning** para responder a problemáticas de negocio. La estrategia consiste en:
- **Consolidar información** de precios diarios de acciones, datos corporativos y estados financieros.
- **Transformar y limpiar** cada fuente de datos para facilitar su integración.
- **Agrupar los precios** a nivel mensual para mitigar la alta volatilidad diaria y alinear la frecuencia con los balances anuales o trimestrales.
- **Realizar un merge** basado en el ticker y el año fiscal, de modo que cada registro de precio (promedio mensual) se combine únicamente con los fundamentos correspondientes al mismo período.

---

## 2. Fuentes de Datos y Transformaciones Realizadas

### 2.1. Dataset de Precios (`df_prices`)
- **Origen:** Archivo *prices-split-adjusted.csv*.
- **Contenido:** Contiene datos históricos diarios de precios y volúmenes para diversas empresas (columnas: `date`, `symbol`, `open`, `close`, `low`, `high`, `volume`).
- **Transformaciones:**  
  - Conversión de la columna `date` a formato datetime.
  - Creación de la columna `month` para agrupar por periodo mensual.
  - Agrupación por `ticker` y `month` para calcular promedios mensuales de los precios y sumar el volumen.
  - Extracción del año fiscal (`fiscal_year`) a partir de la columna `month`.

### 2.2. Dataset de Información Corporativa (`df_securities`)
- **Origen:** Archivo *securities.csv*.
- **Contenido:** Información descriptiva de las empresas, incluyendo el ticker, nombre de la compañía, sector (GICS Sector), subindustria (GICS Sub Industry), dirección de la sede, y número de identificación de la SEC (CIK).
- **Transformaciones:**  
  - Eliminación de columnas irrelevantes para este análisis (por ejemplo, `Date first added` y `SEC filings`).

### 2.3. Dataset de Fundamentales (`df_fundamentals`)
- **Origen:** Archivo *fundamentals.csv*.
- **Contenido:** Estados financieros y métricas clave de cada empresa, tales como cuentas por pagar y cobrar, gastos, ingresos, márgenes, activos, pasivos, etc.
- **Transformaciones:**  
  - Eliminación de columnas innecesarias (como `Unnamed: 0` y `For Year`).
  - Creación de una columna booleana `has_eps` que indica si se dispone de datos en `Earnings Per Share`.

### 2.4. Integración de Datos
- Se **homogenizó** la columna clave renombrándola a `ticker` en todos los DataFrames.
- Se realizó un **merge secuencial**:
  1. **Primero:** Se unió `df_prices` (agrupado mensualmente) con `df_securities` usando `ticker`.
  2. **Luego:** Se combinó el resultado anterior con `df_fundamentals` utilizando tanto `ticker` como `fiscal_year`, asegurando que cada registro mensual se empareje con el balance correspondiente al mismo año.

El resultado es un DataFrame final (`df_final`) que integra precios mensuales, datos corporativos y fundamentales, permitiendo análisis más robustos y comparaciones directas.

---

## 3. Objetivos del Análisis

### Objetivos Generales
- **Consolidar y transformar** datos de diferentes fuentes (precios, información corporativa y fundamentos financieros) en un único DataFrame.
- **Resolver preguntas de negocio** a través de análisis descriptivos y modelos de Machine Learning.
- **Generar insights valiosos** que permitan identificar si las acciones están sobrevaloradas o infravaloradas en función de sus indicadores financieros.

### Objetivos Específicos
- Formular preguntas e hipótesis accionables basadas en la relación entre el precio (promedio mensual) y los fundamentales.
- Ejecutar un proceso de recolección, limpieza y transformación de datos utilizando técnicas avanzadas de Python.
- Elaborar visualizaciones ejecutivas y reportes interactivos (por ejemplo, usando ydata-profiling) que faciliten el storytelling de los hallazgos.

---





## 4. Conclusión

El flujo de trabajo desarrollado en este proyecto permite analizar de forma integral el comportamiento de las empresas públicas. Al agrupar los precios en promedios mensuales y alinearlos con los balances financieros (por año fiscal), se obtiene una visión más robusta para evaluar si, en cada período, las acciones están baratas o caras en función de sus fundamentos. Este enfoque no solo facilita el análisis descriptivo y exploratorio, sino que también sienta las bases para futuros modelos predictivos que aporten insights estratégicos a la industria.

---
## 5. Complejidades y Resoluciones

Durante el desarrollo del proyecto se enfrentaron diversas complejidades en el manejo e integración de los tres datasets, las cuales se resolvieron de la siguiente manera:

- **Dataset de Precios (`df_prices`):**
  - **Complejidad:**  
    - Alta volatilidad diaria y variabilidad en precios y volúmenes.
    - Necesidad de alinear la frecuencia de estos datos (diaria) con los balances financieros (anuales o trimestrales).
  - **Resolución:**  
    - Conversión de la columna `date` a formato datetime.
    - Creación de la columna `month` para agrupar los datos diarios en promedios mensuales.
    - Extracción del `fiscal_year` a partir del periodo mensual para facilitar la integración con los balances.

- **Dataset de Información Corporativa (`df_securities`):**
  - **Complejidad:**  
    - Presencia de columnas irrelevantes o con información incompleta, como `Date first added` y `SEC filings`.
  - **Resolución:**  
    - Eliminación de las columnas innecesarias, dejando únicamente la información esencial (ticker, nombre, sector, subindustria, etc.) para simplificar el merge.

- **Dataset de Fundamentales (`df_fundamentals`):**
  - **Complejidad:**  
    - Existencia de columnas redundantes (por ejemplo, `Unnamed: 0` y `For Year`).
    - Necesidad de identificar y manejar datos faltantes en indicadores clave como `Earnings Per Share`.
  - **Resolución:**  
    - Eliminación de las columnas superfluas.
    - Creación de una columna booleana `has_eps` para marcar si hay datos en `Earnings Per Share`, permitiendo un manejo flexible de los valores faltantes sin afectar cálculos estadísticos.

- **Integración de Datos:**
  - **Complejidad:**  
    - Diferentes frecuencias en los datasets (datos diarios vs. balances anuales).
    - Posible duplicación de registros al unir datos de múltiples balances financieros por mismo ticker.
  - **Resolución:**  
    - Agrupar los precios a nivel mensual para suavizar la volatilidad diaria.
    - Realizar un merge secuencial utilizando `ticker` y `fiscal_year` como claves, de modo que cada registro mensual se alinee únicamente con el balance correspondiente al mismo período.

Estas soluciones permitieron construir un DataFrame unificado robusto, que integra información de mercado, corporativa y financiera de forma consistente y facilita el análisis para determinar si las acciones están baratas o caras en función de sus fundamentos.


## Cómo Ejecutar el Proyecto

1. Clona este repositorio:
   ```bash
   git clone https://github.com/nicolinolochex/DataScience.git


