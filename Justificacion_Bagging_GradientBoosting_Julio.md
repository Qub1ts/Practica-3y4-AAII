# Justificación de cambios de parámetros — Bagging y Gradient Boosting

**Autor:** Julio Lucero
**Modelos asignados:** Bagging y Gradient Boosting
**Dataset:** train.csv (n=1000, p=609, 3 clases A/B/C ~balanceadas, sin nulos)

---

## Contexto del dataset

Antes de justificar los cambios, conviene tener presentes dos hechos del dataset que condicionan toda la estrategia:

1. **Alta dimensionalidad relativa al tamaño muestral:** 609 variables con sólo 1000 observaciones. La razón p/n ≈ 0.6 es elevada, lo que aumenta el riesgo de sobreajuste y hace especialmente útil cualquier mecanismo de submuestreo de columnas (`max_features`).
2. **Clases balanceadas** (~33% cada una), por lo que no es necesario reponderar clases ni usar métricas distintas a *accuracy*. La estratificación en `StratifiedKFold` ya cubre la división.

La evaluación se hace siempre con **5-fold cross-validation estratificada** (la misma `cv` definida globalmente), reportando *accuracy* media y desviación. Para cada modelo se mantiene además la *train accuracy* para diagnosticar sobreajuste.

---

## 1. Bagging

### Intuición teórica que guía las pruebas

Bagging promedia predictores entrenados sobre muestras bootstrap. Reduce la **varianza** del estimador base pero **no su sesgo**, por lo que conviene usar como base un modelo con alta varianza y bajo sesgo (típicamente árboles profundos). En un espacio de 609 variables, además, dos árboles construidos sobre el mismo bootstrap son demasiado parecidos: añadir submuestreo de **columnas** (`max_features < 1.0`) decorrela los árboles y mejora la generalización (la lógica de Random Forest llevada a `BaggingClassifier`).

### Configuraciones probadas y por qué

| ID | Cambio respecto al baseline | Motivación |
|----|----------------------------|------------|
| **BG-01** | Baseline (`n_estimators=10`, sin restricciones) | Línea de base obligatoria para medir mejoras. |
| **BG-02** | `n_estimators=200` | Bagging tiene rendimiento monótono creciente con el nº de árboles hasta saturar; 10 árboles es claramente insuficiente. |
| **BG-03** | `max_features=0.5`, `bootstrap_features=True` | Cada árbol ve sólo la mitad de las 609 columnas, decorrelando árboles (analogo a Random Forest). Esperamos que sea el cambio que más mueva la CV. |
| **BG-04** | Limita profundidad: `max_depth=8`, `max_samples=0.8` | Reduce el sobreajuste extremo del árbol base (que de origen crece hasta hojas puras → 100% train accuracy). |
| **BG-05** | Pipeline `StandardScaler + PCA(100) + Bagging(LogisticRegression)` | Probar un estimador base lineal sobre el espacio reducido: si el problema es razonablemente separable linealmente, debería ser competitivo y mucho más estable. |
| **BG-06** | Combinación final: `n_estimators=300`, `max_depth=12`, `min_samples_leaf=2`, `max_samples=0.8`, `max_features=0.5`, `bootstrap_features=True` | Une las dos palancas que más ayudan: muchos árboles + submuestreo de filas y columnas, con árboles ligeramente regularizados por `min_samples_leaf`. |

### Hipótesis sobre los resultados

- BG-01 y BG-02 quedarán cerca del Random Forest baseline pero por debajo del Random Forest **tuneado** del notebook (CV ≈ 0.827), porque carecen de submuestreo de columnas.
- BG-03 debería acercarse al rendimiento del Random Forest, ya que conceptualmente reproduce su receta.
- BG-05 (LogReg + PCA) probablemente quede por debajo si el problema tiene fronteras no lineales, pero sirve como "configuración descartada" que pide el enunciado.
- BG-06 es la candidata a mejor CV; la limitación de profundidad relativa (12 en vez de ilimitada) más `min_samples_leaf=2` reduce la varianza por árbol sin perder demasiada capacidad.

### Configuraciones descartadas (cumplir requerimiento del enunciado)

El enunciado exige describir **al menos dos configuraciones descartadas**:

- **BG-05 (LogReg + PCA):** descartada si su CV cae claramente por debajo de los baggings con árboles. Justificación: PCA mezcla todas las variables y un modelo lineal pierde la capacidad de capturar interacciones que sí captan los árboles. Aun así, sirve como punto de referencia útil.
- **Bagging sin `max_features` (BG-01/BG-02):** descartado como configuración final por la alta dimensionalidad. Sin submuestreo de columnas los árboles quedan demasiado correlacionados y el beneficio del bagging satura rápido.

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

### Configuraciones probadas y por qué

| ID | Cambio respecto al baseline | Motivación |
|----|----------------------------|------------|
| **GB-01** | Baseline (`n_estimators=100, lr=0.1, max_depth=3`) | Línea de base. |
| **GB-02** | `lr=0.05`, `n_estimators=300` | Aplicar la regla clásica `lr ↓ + n ↑`. Suele ser la primera mejora segura en GB. |
| **GB-03** | + `subsample=0.8` | Introducir aleatoriedad por filas (Stochastic GB) → menos sobreajuste sin coste de sesgo apreciable. |
| **GB-04** | `max_depth=5`, `n_estimators=200` | Probar si conviene capturar interacciones de mayor orden; arriesgado porque aumenta el sobreajuste. |
| **GB-05** | + `max_features="sqrt"` | Con 609 columnas, sqrt(609)≈25. Reduce drásticamente la varianza y, además, **acelera mucho** el entrenamiento. |
| **GB-06** | `StandardScaler + PCA(100) + GB` | Comprobar si reducir ruido del espacio original ayuda. En GB con árboles la escala no importa, así que aquí lo que se aporta es la denoising vía PCA. |
| **GB-07** | Combinación final: `n_estimators=400, lr=0.05, max_depth=4, subsample=0.8, min_samples_leaf=5, max_features="sqrt"` | Reúne todas las regularizaciones que ayudan individualmente: lr bajo + muchos árboles + profundidad moderada + submuestreo de filas y columnas + hojas mínimas. |

### Hipótesis sobre los resultados

- GB-01 servirá de referencia y casi con seguridad llegará a `train_accuracy=1.0` (sobreajuste típico).
- GB-02 y GB-03 deberían mover la CV hacia arriba en pequeños incrementos.
- GB-05 (con `max_features="sqrt"`) es donde se espera el mayor salto, porque ataca específicamente la alta dimensionalidad.
- GB-06 (con PCA) probablemente quede por **debajo** de GB-05: PCA destruye la estructura por variable que GB explota cuando hay features relevantes individuales. Es una configuración interesante de descartar.
- GB-07 es el candidato a mejor CV. Si supera al Random Forest tuneado (CV ≈ 0.827) tenemos un buen modelo final.

### Configuraciones descartadas

- **GB-06 (PCA antes de GB):** previsiblemente peor o igual que GB-05. PCA mezcla las 609 variables en componentes lineales y le quita a GB su capacidad de seleccionar features individuales relevantes; no aporta nada que `max_features` no haga mejor.
- **GB-04 (`max_depth=5`):** si rinde peor que GB-03/05, se descarta. Árboles más profundos en GB suelen sobreajustar antes de aportar capacidad útil, especialmente con n=1000.

---

## Resumen del proceso iterativo (estilo "lessons learned")

1. **Primer paso (baseline):** confirmar que ambos modelos funcionan razonablemente con configuración por defecto. Sirve como anclaje numérico.
2. **Segundo paso (escalar el ensemble):** subir `n_estimators` en bagging, bajar `learning_rate` en GB. Mejoras seguras y baratas.
3. **Tercer paso (atacar la alta dimensionalidad):** introducir `max_features` en ambos. Es la mejora más importante en este dataset por su razón p/n.
4. **Cuarto paso (regularizar):** `max_samples`/`subsample` (filas), `min_samples_leaf` y `max_depth` controlados. Importante en GB para frenar el sobreajuste secuencial.
5. **Quinto paso (preprocesamiento alternativo):** PCA o estimadores lineales como base. Aquí actúan como **configuraciones discutidas y descartadas**, no como ganadoras: confirman que el camino correcto es trabajar sobre el espacio original con árboles.
6. **Configuración final:** combinar todas las palancas que individualmente mejoraron la CV.

---

## Comparación esperada con el resto de modelos del notebook

| Modelo | CV esperada (referencia) |
|--------|--------------------------|
| Naive Bayes + Wrapper | ~0.65 |
| Random Forest tuneado | ~0.83 |
| LDA + PCA (Agostina) | ~0.74 |
| **Bagging tuneado (BG-06)** | esperado **~0.80–0.84** |
| **Gradient Boosting tuneado (GB-07)** | esperado **~0.83–0.86** |

Si GB-07 supera al Random Forest, el equipo tendrá un candidato fuerte para el modelo final del stacking; si no, Random Forest sigue siendo competitivo. En cualquier caso, ambos modelos confirman la hipótesis de que **los métodos basados en árboles con submuestreo de columnas son la familia más adecuada** para este dataset.
