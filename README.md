# 📊 Análisis Predictivo de Acciones Públicas

## 🎯 Abstracto, Motivación y Audiencia

Este proyecto se centra en el análisis integral de datos de empresas públicas, combinando información de precios históricos, datos corporativos y estados financieros para determinar si, en cada período, las acciones se encuentran **sobrevaloradas o infravaloradas**.

https://colab.research.google.com/drive/1ImaNwvrIdj9_NZdkpzpyQ4AZez4CkUSV#scrollTo=X6fywxY-U1cH&uniqifier=2

### 🎯 Motivación
Generar *insights* objetivos y basados en datos sólidos que faciliten la toma de decisiones estratégicas en entornos de inversión y análisis financiero.

### 👥 Audiencia
- Inversionistas y analistas financieros.
- Ejecutivos de estrategia y finanzas.
- Equipos de Research y Data Science.

---

## 🔍 Contexto y Metodología

1. Consolidación de precios, fundamentales y datos corporativos.
2. Transformación, limpieza e integración de fuentes heterogéneas.
3. Agrupación mensual de precios y alineación con año fiscal.
4. Creación del dataset final (`df_final`) con merge secuencial.

---

## 🧾 Fuentes de Datos y Procesamiento

### 📈 Precios (`df_prices`)
- **Origen**: `prices-split-adjusted.csv`
- **Transformaciones**:
  - Conversión de fechas.
  - Agrupación mensual por ticker.
  - Extracción de `fiscal_year`.

### 🏢 Corporativo (`df_securities`)
- **Origen**: `securities.csv`
- **Transformaciones**:
  - Eliminación de columnas irrelevantes.
  - Retención de datos clave (sector, industria, dirección).

### 📉 Fundamentales (`df_fundamentals`)
- **Origen**: `fundamentals.csv`
- **Transformaciones**:
  - Limpieza de columnas redundantes.
  - Cálculo de bandera `has_eps` para EPS disponible.

### 🔗 Integración
- Homogeneización del campo `ticker`.
- Merge por `ticker` y `fiscal_year`.
- Alineación temporal entre precios y balances.

---

## 🔎 Feature Engineering

- Imputación de nulos y codificación de variables categóricas (`get_dummies`).
- Normalización con `StandardScaler`.
- Selección de variables con `SelectKBest (f_regression)`.
- Reducción de dimensionalidad con **PCA** (10 componentes).

### 📌 Variables más relevantes para explicar el precio de apertura (`open`)
1. `Earnings Per Share`: ~61.6%
2. `Revenue per Share`: ~12.1%
3. `ROA`, `Volume`, `Cash Ratio`, entre otras.

---

## 🤖 Entrenamiento y Testeo

### Algoritmos evaluados:
- 🔹 **Regresión Lineal** (baseline)
- 🔹 **Random Forest Regressor** (modelo principal)
- 🔹 **Gradient Boosting Regressor** (evaluado pero descartado)

### Resultados en testeo:
| Modelo              | MSE     | R² Score |
|---------------------|---------|----------|
| Regresión Lineal    | 0.8851  | 0.0808   |
| Random Forest       | 0.0283  | 0.9706   |
| Gradient Boosting   | 0.0548  | 0.9431   |

---

## 🔁 Validación Cruzada (K-Fold, k=3)

- **R² promedio**: 0.9078  
- **MSE promedio**: 0.0919

---

## ⚙️ Optimización de Hiperparámetros

Se utilizó `RandomizedSearchCV` para optimizar el modelo de Random Forest.

### 🔍 Mejores hiperparámetros encontrados:
```python
{
  'n_estimators': 200,
  'min_samples_split': 5,
  'min_samples_leaf': 1,
  'max_features': 'log2',
  'max_depth': 30
}
🧪 Evaluación final del modelo optimizado:
📉 MSE: 0.0257

📈 R² Score: 0.9733

✅ Conclusión Final
El modelo final basado en Random Forest Regressor optimizado fue el que mostró mejor desempeño, alcanzando una capacidad explicativa del 97% sobre el precio de apertura de las acciones.
El pipeline completo demuestra cómo el uso de técnicas avanzadas de procesamiento de datos, selección de variables, reducción de dimensionalidad y tuning de modelos permite generar valor concreto a partir de datos financieros crudos.
Este enfoque puede ser directamente reutilizado o extendido por áreas de estrategia, inversión y ciencia de datos.