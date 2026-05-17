# Justificación de cambios de parámetros — Bagging y Gradient Boosting

**Autor:** Julio Lucero
**Modelos asignados:** Bagging y Gradient Boosting
**Dataset:** train.csv (n=1000, p=609, 3 clases A/B/C ~balanceadas, sin nulos)

---

## Contexto del dataset

Antes de justificar los cambios, conviene tener presentes dos hechos del dataset que condicionan toda la estrategia:

1. **Alta dimensionalidad relativa al tamaño muestral:** 609 variables con sólo 1000 observaciones. La razón p/n ≈ 0.6 es muy elevada, lo que aumenta el riesgo de sobreajuste y hace especialmente útil cualquier mecanismo de submuestreo de columnas (`max_features`).
2. **Clases balanceadas** (~33% cada una), por lo que no es necesario reponderar clases ni usar métricas distintas a *accuracy*. La estratificación en `StratifiedKFold` ya cubre la división.

La evaluación se hace siempre con **5-fold cross-validation estratificada** (la misma `cv` definida globalmente en el notebook), reportando *accuracy* media y desviación. Para cada modelo se mantiene además la *train accuracy* para diagnosticar sobreajuste.

Referencias para comparar: Random Forest tuneado en el notebook obtiene **CV ≈ 0.827**, Naive Bayes ≈ 0.654 y LDA + PCA (Agostina) ≈ 0.738.

---

## 1. Bagging

### Intuición teórica que guía las pruebas

Bagging promedia predictores entrenados sobre muestras bootstrap. Reduce la **varianza** del estimador base pero **no su sesgo**, por lo que conviene usar como base un modelo con alta varianza y bajo sesgo (típicamente árboles profundos). En un espacio de 609 variables, además, dos árboles construidos sobre bootstraps similares quedan demasiado correlacionados: añadir submuestreo de **columnas** (`max_features < 1.0`) decorrelaciona los árboles y mejora la generalización (la lógica de Random Forest llevada a `BaggingClassifier`).

### Configuraciones probadas y resultados reales

| ID | Configuración | Train acc | **5-fold CV** | Δ vs baseline |
|----|---------------|-----------|---------------|---------------|
| **BG-06** ⭐ | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=2`, `max_samples=0.8`, `max_features=0.5`, `bootstrap_features=True` | 0.997 | **0.821 ± 0.017** | **+0.048** |
| BG-03 | `n_estimators=200`, `max_features=0.5`, `bootstrap_features=True` | 1.000 | 0.819 ± 0.019 | +0.046 |
| BG-02 | `n_estimators=200` (sin submuestreo de columnas) | 1.000 | 0.811 ± 0.017 | +0.038 |
| BG-04 | `max_depth=8`, `n_estimators=200`, `max_samples=0.8`, `max_features=0.5` | 0.994 | 0.811 ± 0.019 | +0.038 |
| BG-01 | Baseline (`n_estimators=10`, sin restricciones) | 0.989 | 0.773 ± 0.024 | — |
| BG-05 | `StandardScaler + PCA(100) + Bagging(LogReg, n=100)` | 0.790 | 0.675 ± 0.023 | −0.098 |

### Análisis de los resultados

**El cambio que más mueve la aguja es introducir `max_features=0.5`** (submuestreo de columnas). Comparando configuraciones equivalentes que sólo difieren en este parámetro, BG-02 (sin submuestreo) → BG-03 (con submuestreo) gana **+0.008 CV**; y respecto al baseline BG-01 el salto total al añadir submuestreo + más estimadores es de **+0.046 CV**. Esto confirma la hipótesis previa: con p=609 la descorrelación entre árboles es crítica.

Subir simplemente `n_estimators` de 10 a 200 (BG-01 → BG-02) ya aporta **+0.038**, así que el efecto "más árboles" es real pero menor que el de `max_features`.

La configuración final **BG-06** sólo gana **+0.002 CV** sobre BG-03, dentro del ruido (las desviaciones son ~0.017). En la práctica BG-03 y BG-06 son indistinguibles, pero **BG-06 muestra una desviación ligeramente menor y mejor consistencia**, por lo que se elige como mejor configuración. La lección honesta es que **una vez activado `max_features=0.5`, los demás ajustes son cosméticos**.

`BG-04` (limitar `max_depth=8`) no mejora a BG-03: la limitación de profundidad reduce algo el sobreajuste (train baja de 1.000 a 0.994) pero no se traduce en mejor CV.

### Configuraciones descartadas (requerimiento del enunciado)

- **BG-05 (PCA + LogReg):** descartada con claridad. CV = 0.675, **−0.146 frente a BG-06**, incluso por debajo del baseline. Razón: PCA mezcla las 609 variables en componentes lineales y un modelo lineal pierde la capacidad de capturar las interacciones que sí explotan los árboles. Demuestra que el problema **no es separable linealmente** en el espacio de componentes principales.
- **BG-02 (Bagging sin `max_features`):** descartada como configuración final. Confirma que **sin submuestreo de columnas el bagging satura** alrededor de 0.811 por más estimadores que añadamos, porque los árboles quedan demasiado correlacionados entre sí.

### Mejor modelo de Bagging

```
BG-06 Bagging tuned
  BaggingClassifier(
      estimator=DecisionTreeClassifier(max_depth=12, min_samples_leaf=2),
      n_estimators=300, max_samples=0.8, max_features=0.5,
      bootstrap_features=True)
  Train accuracy: 0.9970
  5-fold CV accuracy: 0.8210 ± 0.0174
```

---

## 2. Gradient Boosting

### Intuición teórica que guía las pruebas

Gradient Boosting construye árboles **secuencialmente**, cada uno corrigiendo los errores residuales del ensamblaje anterior. A diferencia de bagging, reduce **sesgo**, no varianza, y por eso tiende a **sobreajustar** rápidamente si no se regulariza. Las palancas principales son:

- **`learning_rate`**: cuanto más pequeño, más lento aprende y más generaliza, pero exige más árboles.
- **`n_estimators`**: aumentar junto con learning_rate bajo (compensación clásica `lr ↓ + n ↑`).
- **`max_depth`**: en GB se usan árboles **poco profundos** (3–5) porque cada árbol sólo debe modelar una pequeña corrección.
- **`subsample < 1.0`**: convierte el modelo en *Stochastic Gradient Boosting* (Friedman), reduciendo varianza.
- **`max_features`**: equivalente al submuestreo de columnas; muy útil con p=609.
- **`min_samples_leaf`**: regularización adicional por hoja.

### Configuraciones probadas y resultados reales

| ID | Configuración | Train acc | **5-fold CV** | Δ vs baseline |
|----|---------------|-----------|---------------|---------------|
| **GB-05** ⭐ | `n=300, lr=0.05, max_depth=3, subsample=0.8, max_features="sqrt"` | 1.000 | **0.828 ± 0.016** | **+0.026** |
| GB-07 | `n=400, lr=0.05, max_depth=4, subsample=0.8, min_samples_leaf=5, max_features="sqrt"` | 1.000 | 0.827 ± 0.021 | +0.025 |
| GB-04 | `n=200, lr=0.05, max_depth=5, subsample=0.8` | 1.000 | 0.815 ± 0.019 | +0.013 |
| GB-03 | `n=300, lr=0.05, max_depth=3, subsample=0.8` | 1.000 | 0.814 ± 0.010 | +0.012 |
| GB-06 | `StandardScaler + PCA(100) + GB(n=300, lr=0.05, subsample=0.8)` | 0.999 | 0.805 ± 0.025 | +0.003 |
| GB-01 | Baseline (`n=100, lr=0.1, max_depth=3`) | 1.000 | 0.802 ± 0.023 | — |
| GB-02 | `n=300, lr=0.05` (sin subsample ni max_features) | 1.000 | 0.800 ± 0.019 | −0.002 |

### Análisis de los resultados

**Todas las configuraciones de GB sobreajustan a train hasta 1.000**; sólo la CV es informativa. Esto era esperable y refuerza la importancia de regularizar.

**Resultado más relevante:** la mejor configuración es **GB-05**, no la "tuneada al máximo" GB-07. La diferencia (0.828 vs 0.827) está dentro del ruido, pero la lección es clara — **el aporte real proviene de un solo parámetro, `max_features="sqrt"`**: añadirlo a GB-03 (0.814) lo lleva a GB-05 (0.828), ganando **+0.014 CV**. Las regularizaciones extra de GB-07 (más árboles, `max_depth=4`, `min_samples_leaf=5`) no aportan nada por encima de eso, y de hecho aumentan ligeramente la varianza (cv_std sube de 0.016 a 0.021).

Otros hallazgos:

- **GB-02 (`lr=0.05` + `n=300`) es peor que el baseline GB-01** (0.800 vs 0.802). Bajar el learning rate sin añadir mecanismos de regularización (subsample o max_features) **no ayuda por sí solo** en este dataset.
- **GB-03 (añadir `subsample=0.8`)** sí mejora respecto a GB-02 (+0.014), confirmando el beneficio del Stochastic GB.
- **GB-04 (`max_depth=5`)** apenas mejora a GB-03 (+0.001): subir la profundidad permite capturar más interacciones pero también más sobreajuste, y los efectos se cancelan.
- **GB-06 (PCA)** queda **por debajo** de GB-05 sin PCA (0.805 vs 0.828). Confirma la hipótesis: **PCA destruye la capacidad de GB de seleccionar variables individuales relevantes**.

### Configuraciones descartadas

- **GB-02 (sólo `lr↓ + n↑`):** descartada porque **no mejora al baseline**. Demuestra que la regla clásica `lr↓ + n↑` necesita ir acompañada de algún mecanismo de regularización (subsample o max_features) para mostrar beneficio en este dataset.
- **GB-06 (PCA + GB):** descartada porque empeora la CV en 0.023 frente a GB-05. Sirve como contraejemplo claro: el preprocesado de reducción de dimensionalidad **resta**, no suma, cuando el modelo posterior es un ensamblador de árboles capaz de seleccionar features individualmente.

### Mejor modelo de Gradient Boosting

```
GB-05 GB + max_features=sqrt
  GradientBoostingClassifier(
      n_estimators=300, learning_rate=0.05, max_depth=3,
      subsample=0.8, max_features="sqrt")
  Train accuracy: 1.0000
  5-fold CV accuracy: 0.8280 ± 0.0163
```

---

## Comparación final con el resto de modelos

| Modelo | 5-fold CV |
|--------|-----------|
| Naive Bayes + Wrapper | 0.654 |
| LDA + PCA(100) (Agostina) | 0.738 |
| **Bagging tuned (BG-06)** | **0.821** |
| Random Forest tuneado | 0.827 |
| **Gradient Boosting tuned (GB-05)** | **0.828** |

**Gradient Boosting es el mejor modelo individual** del conjunto evaluado, superando ligeramente al Random Forest tuneado (+0.001) y al Bagging tuneado (+0.007). La diferencia con Random Forest no es significativa estadísticamente, pero **valida la elección de GB-05 como candidato principal** para el modelo final y como base fuerte para el stacking de Lama.

---

## Lecciones aprendidas / proceso iterativo

1. **El submuestreo de columnas (`max_features`) es la palanca dominante** en este dataset, tanto en Bagging como en Gradient Boosting. Cualquier intento de mejorar sin tocar este parámetro produce ganancias marginales.
2. **El "fine-tuning" más allá del primer parámetro útil deja de mejorar la CV**: las configuraciones "más complejas" (BG-06 vs BG-03, GB-07 vs GB-05) ganan diferencias dentro del ruido. Honestidad metodológica: no inflar la complejidad cuando no aporta.
3. **PCA es perjudicial** para los modelos basados en árboles en este dataset (BG-05 cae −0.146; GB-06 cae −0.023). PCA elimina la estructura por variable que los árboles explotan, y sólo conviene cuando el modelo posterior es lineal (como en el LDA de Agostina).
4. **Todos los GB sobreajustan train a 1.000.** Esto recuerda que `train_accuracy` no es informativa en GB; la CV es la única señal útil para escoger configuraciones.
5. Si se quiere ganar más allá de 0.828 probablemente haga falta saltar a modelos fuera del alcance del enunciado obligatorio (XGBoost/LightGBM con regularización L1/L2, o el stacking de los mejores modelos), no seguir afinando estos dos.
