<img width="800" height="800" alt="XGBSculpt_Logo" src="https://github.com/user-attachments/assets/0ad03268-ac45-4b43-870e-b565c640c3be" />

🛠️  **Proyecto siempre bajo construcción**

Módulo Experimental Python para entrenamiento, evaluación y refinamiento progresivo de pipelines XGBoost mediante selección sistemática de features. Combina nested cross-validation, splits por grupos, ponderación inversa de muestras e importancias múltiples para reducir el número de variables en los modelos.

Los datos debe contener una feature que señale a lo ensayos repetidos en -aproximadamente las mismas condiciones- a una muestra ('sample_id'). Durante la carga de datos se asiginará un peso (weight) a la variable objetivo inversamente proporcional al número de repeticiones en la muestra.

Realiza un target encoder sobre una variable categórica (v.g. sexo, día de la semana, tipo de aleación...).


---

## Tabla de contenidos

- [Dependencias](#dependencias)
- [Instalación](#instalación)
- [Arquitectura general](#arquitectura-general)
- [Constantes globales](#constantes-globales)
- [Flujo de uso recomendado](#flujo-de-uso-recomendado)
- [API — Referencia completa](#api--referencia-completa)
  - [Configuración](#configuración)
  - [Datos](#datos)
  - [Pipeline principal](#pipeline-principal)
  - [Importancias de features](#importancias-de-features)
  - [Selección de features](#selección-de-features)
  - [Utilidades de pipeline](#utilidades-de-pipeline)
  - [SHAP](#shap)
  - [Visualización](#visualización)
- [Decisiones de diseño y advertencias críticas](#decisiones-de-diseño-y-advertencias-críticas)
- [Logging](#logging)
- [Checkpointing y reanudación](#checkpointing-y-reanudación)
- [Guía de selección de función](#guía-de-selección-de-función)

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---|---|---|
| `scikit-learn` | 1.3 | Metadata routing, nested CV, HalvingGridSearchCV |
| `xgboost` | 1.7 | Modelo base |
| `shap` | 0.42 | Explicabilidad |
| `pandas` | 1.5 | Manipulación de datos |
| `numpy` | 1.23 | Operaciones numéricas |
| `matplotlib` | 3.6 | Visualización |
| `seaborn` | 0.12 | Visualización |
| `adjustText` | 0.8 | Etiquetas sin solapamiento en scatter |

> **Crítico:** sklearn >= 1.3 es obligatorio. El módulo activa `enable_metadata_routing=True` para enrutar `groups` y `sample_weight` a través del pipeline. Versiones anteriores lanzarán `AttributeError` al llamar a `set_fit_request` o `set_split_request`.

---

## Instalación

```bash
pip install scikit-learn>=1.3 xgboost shap pandas numpy matplotlib seaborn adjustText
```

```python
import XGBSculpt as xs
```

Para activar mensajes de log (recomendado en notebooks y scripts):

```python
import logging
logging.basicConfig(level=logging.INFO)
```

El módulo **nunca configura el logger raíz** al importarse. La configuración es responsabilidad exclusiva del cliente.

---

## Arquitectura general

```
CSV
 │
 ▼
data_read()  ──────────────────────────────────────────────────────────────────
 │  Devuelve: y, X, num_cols, cat_cols, groups, weights
 │
 ▼
run_pipe()  ────────────────────────────────────────────────────────────────────
 │  Nested CV (GroupKFold 10×10) + HalvingGridSearchCV
 │  Devuelve: models (ensemble 10 folds), cv_results, final_model (100% datos)
 │
 ├── get_metrics(cv_results)          → RMSE train / val / test por fold
 ├── get_hyperparams(cv_results, ...) → mejores params por fold
 ├── predict_ensemble(models, X)      → predicción con banda de incertidumbre
 │
 ▼   [Selección de features — usar [final_model] como entrada]
 │
 ├── drop_column_cv_pipeline_clean([final_model], X, y, ...)
 │       → importancias drop-column, evaluación independiente
 │
 ├── itera_perm_mod(X, y, ...)
 │       → eliminación batch iterativa con reentrenamiento en cada ronda
 │
 ├── sequential_drop_column_cv([final_model], X, y, ...)
 │       → eliminación secuencial dinámica (baseline evoluciona)
 │
 └── stable_backward_selection_with_oof_permutation([final_model], X, y, ...)
         → selección backward OOF con permutation + drop importance
         → con reintroducción, checkpoint y callback
         → Devuelve: history, remaining, X_, num_cols_, cat_cols_,
                     models_, metrics_, final_model_
 │
 ▼
exhaustive_feature_combinations([final_model], X, y, ...)
 │       → búsqueda exhaustiva de combinaciones óptimas
 │
 ▼
shap_explainer(models_, X_)   → valores SHAP del ensemble final
shap_values(models_, X_)      → SHAP sobre muestra aleatoria
 │
 ▼
feat_corr(X_)          → mapa de correlaciones
hist_track(history)    → tablas de importancia por step
steps_plot(history)    → evolución gráfica de importancias
drop_perm_plot(df)     → scatter drop vs permutation importance
```

---

## Constantes globales

### `RANDOM_STATE = 42`

Semilla de aleatoriedad usada por defecto en todas las funciones que la requieren. Se puede sobreescribir pasando `random_state=N` en las firmas que lo aceptan.

### `PARAM_HGS`

Grid de hiperparámetros por defecto para `HalvingGridSearchCV`:

```python
PARAM_HGS = {
    "model__max_depth":        [2, 3, 4],
    "model__learning_rate":    [0.05, 0.15, 0.25],
    "model__subsample":        [0.5, 0.75],
    "model__colsample_bytree": [0.6, 0.8],
    "model__min_split_loss":   [0.01, 0.1],
    "model__reg_lambda":       [1, 5],
}
```

Se puede sustituir pasando `param_grid=mi_grid` a `run_pipe()`. Centralizar el grid aquí evita duplicarlo en el código cliente.

---

## Flujo de uso (ejemplo)

###

```python
import logging
import xgb_sculpt as xs

logging.basicConfig(level=logging.INFO)

# 1. Carga y preparación de datos
y, X, num_cols, cat_cols, groups, weights = xs.data_read(
    csv_name="datos.csv",
    sample_id="paciente_id",   # None si no hay estructura de grupos
    target_name="objetivo",
    drop_cols=["fecha", "notas"]
)

# 2. Entrenamiento inicial con todas las features
models, cv_results, final_model = xs.run_pipe(
    X, y, num_cols, cat_cols, groups, weights
)

# 3. Evaluación del pipeline
metrics = xs.get_metrics(cv_results)
print(metrics)

# 4. Selección de features — final_model ya es lista, usarlo directamente
history, remaining, X_, num_cols_, cat_cols_, models_, metrics_, final_model_ = \
    xs.stable_backward_selection_with_oof_permutation(
        final_model, X, y,          # ← sin corchetes, run_pipe ya devuelve lista
        cv_splits=5,
        groups=groups,
        weights=weights,
        threshold=0.0,
        checkpoint_path="selection_state.json",
        log_path="selection.log",
    )

# 5. Evaluar el modelo de producción con CV externo (sin inner CV)
rmse_df = xs.eval_final_model(final_model_, X_, y, groups=groups, weights=weights)
print(rmse_df)

# 5. Visualización del proceso
records_df, rmse_steps = xs.hist_track(history)
xs.steps_plot(history)
xs.drop_perm_plot(records_df)

# 6. Explicabilidad del modelo final
sv = xs.shap_values(models_, X_, size_sample=500)

# 7. Predicción en producción con el modelo final
mean_preds, std_preds = xs.predict_ensemble([final_model_], X_new)
# O directamente:
preds = final_model_.predict(X_new)
```

### Patrón para selección de features: `[final_model]` vs ensemble

Todas las funciones de selección aceptan tanto el ensemble completo como una lista con un único `[final_model]`. **El patrón recomendado es `[final_model]`** por dos razones:

1. **Coste computacional** — el ensemble tiene 10 modelos. Pasar `[final_model]` reduce el coste en un factor 10 sin pérdida de calidad en la selección.
2. **Consistencia del baseline** — con el ensemble, el baseline se promedia sobre 10 modelos entrenados en 10 subconjuntos distintos (cada uno con el 90% de los datos) y con hiperparámetros potencialmente distintos. Esto introduce ruido en el RMSE. `final_model` tiene un único baseline limpio, con hiperparámetros seleccionados sobre una cantidad mayor de datos 100%.

El ensemble completo se requiere para **estimar la varianza de las importancias entre folds (En general, importancia en este contexto hace referencia a variación del RMSE cuando se eliminar o cuando se permutan las características)**.

```python
# Patrón posible — obtener final_model y usarlo directamente
models, cv_results, final_model = xs.run_pipe(X, y, ...)
importance = xs.drop_column_cv_pipeline_clean(final_model, X, y, ...)

# Patrón alternativo (análisis de varianza entre folds eliminando características una a una)
importance = xs.drop_column_cv_pipeline_clean(models, X, y, ...)
```

---

## API — Referencia completa

### Configuración

#### `set_display_options()`

Aplica opciones de presentación de pandas, NumPy y Jupyter. Debe llamarse explícitamente; el módulo no altera el entorno global al importarse.

```python
xs.set_display_options()
```

---

### Datos

#### `data_read(csv_name, sample_id, target_name, drop_cols=None)`

Lee un CSV y prepara los datos para modelado. Detecta la estructura de grupos, calcula pesos inversos por frecuencia de muestra y clasifica columnas por tipo.

```python
y, X, num_cols, cat_cols, groups, weights = xs.data_read(
    csv_name="datos.csv",
    sample_id="id_muestra",   # None si no hay grupos
    target_name="target",
    drop_cols=["col_extra"]
)
```

**Comportamiento con `sample_id`:**

Si `sample_id` no es `None`, la función detecta cuántas filas existen por cada muestra y asigna un peso `1/n_medidas` a cada fila. Esto compensa el sesgo que producirían muestras con más registros durante el entrenamiento. Los pesos se pasan a XGBoost vía `sample_weight`.

> **'sample_id'** no debe ser None. Si no hay muestras repetidas puede asignarse al índice ordinal por fila (0, 1, 2, 3, ...). En este caso todos los weights serán iguales a 1.

**Returns:** `(y, X, num_cols, cat_cols, groups, weights)`

- Columnas booleanas se incluyen en `num_cols` y se castean a `float64` para compatibilidad con SHAP.
- `X` es siempre una copia independiente del DataFrame original (`.copy()`), nunca una vista.

---

### Pipeline principal

#### `run_pipe(X, y, num_cols, cat_cols, groups, weights, random_state=RANDOM_STATE, param_grid=PARAM_HGS)`

Construye y ejecuta el pipeline completo con nested cross-validation.

**Estructura interna del pipeline:**

```
Pipeline
 ├── "prep": ColumnTransformer
 │    ├── "num": StandardScaler         (columnas numéricas + booleanas)
 │    └── "cat": TargetEncoder          (columnas categóricas, smoothing=20)
 └── "model": XGBRegressor             (objective="reg:squarederror")
```

**Proceso de entrenamiento:**

```
HalvingGridSearchCV (inner CV, 10 folds)
        ↓  encuentra mejores hiperparámetros
cross_validate (outer CV, 10 folds)
        ↓  evalúa el proceso completo (nested CV imparcial)
        ↓  devuelve 10 best_estimator_ → ensemble
HalvingGridSearchCV.fit(X_completa, cv=5 folds)
        ↓  refit automático (refit=True por defecto)
        ↓  devuelve final_model (100% datos)
```

**Returns:** `(models, cv_results, final_model)`

```python
models, cv_results, final_model = xs.run_pipe(X, y, num_cols, cat_cols, groups, weights)
```

| Elemento | Tipo | Descripción |
|---|---|---|
| `models` | `list[Pipeline]` | 10 pipelines, uno por fold outer, cada uno entrenado en ≈90% de datos |
| `cv_results` | `dict` | Resultado de `cross_validate`: métricas y estimadores por fold |
| `final_model` | `list[Pipeline]` | Lista de un único elemento — modelo de producción con hiperparámetros óptimos entrenado en el 100% de los datos. Se devuelve como lista para ser directamente compatible con las funciones de selección |

---

#### `predict_ensemble(models, X)`

Predicción agregada del ensemble. Devuelve la media y la desviación estándar de las predicciones de todos los modelos. La desviación estándar es un proxy de incertidumbre entre modelos.

```python
mean_preds, std_preds = xs.predict_ensemble(models, X_test)
```

---

#### `get_metrics(cv_results)`

Extrae las métricas RMSE de train (inner CV), validación (inner CV) y test (outer CV) para cada fold. Requiere que `cv_results` provenga de `run_pipe()` — los estimadores deben ser instancias de `HalvingGridSearchCV`.

```python
metrics_df = xs.get_metrics(cv_results)
# fold  train_rmse  val_rmse  test_rmse
```

---

#### `eval_final_model(final_model, X, y, groups=None, weights=None, cv_splits=10)`

Estima el RMSE de generalización del modelo de producción mediante CV externo puro — sin ningún inner CV ni búsqueda de hiperparámetros. En cada fold clona el pipeline, reajusta los árboles con los datos de train del fold y evalúa sobre el test.

> **Diferencia con el nested CV de `run_pipe`:** el nested CV evalúa el *proceso completo* de búsqueda de hiperparámetros. `eval_final_model` evalúa el *modelo concreto* con sus hiperparámetros ya fijados. El resultado de `eval_final_model` suele ser ligeramente más optimista porque los hiperparámetros se seleccionaron viendo todos los datos.

> **Coste:** `O(cv_splits)` fits — frente a `O(N_configs × cv_splits²)` del nested CV. Ideal para evaluar el modelo final tras la selección de features.

```python
models, cv_results, final_model = xs.run_pipe(X, y, num_cols, cat_cols, groups, weights)

# Evaluación rápida del modelo de producción
rmse_df = xs.eval_final_model(final_model, X, y, groups=groups, weights=weights)
print(rmse_df)
#    fold    rmse
# 0     1  0.1823
# ...
# 10 mean  0.1791
# 11  std  0.0042

# Comparar con el nested CV
nested_rmse = -cv_results["test_score"].mean()
final_rmse  = rmse_df.loc[rmse_df["fold"] == "mean", "rmse"].iloc[0]
print(f"Nested CV RMSE:    {nested_rmse:.4f}")
print(f"Final model RMSE:  {final_rmse:.4f}")
```

---

#### `get_hyperparams(cv_results, X, y, weights)`

Extrae los mejores hiperparámetros de cada fold del outer CV. También calcula el RMSE in-sample del ensemble sobre los datos proporcionados.

> **Advertencia:** el RMSE del ensemble se calcula sobre los datos de entrada, que pueden incluir datos de entrenamiento. Usar solo como referencia complementaria, no como estimación de generalización.

```python
params_df = xs.get_hyperparams(cv_results, X, y, weights)
```

---

### Importancias de features

#### `permutation_ensemble(models, X, y, random_state=RANDOM_STATE)`

Calcula la importancia por permutación promediada sobre el ensemble. Mide cuánto empeora el RMSE al permutar aleatoriamente cada columna.

> **Nota metodológica:** la permutation importance subestima la importancia de features correlacionadas — si una se permuta, otra correlacionada compensa. Usar `drop_column_cv_pipeline_clean` cuando haya multicolinealidad sospechada.

```python
importance_df = xs.permutation_ensemble(models, X_test, y_test)
```

---

#### `drop_column_cv_pipeline_clean(models, X, y, cv_splits=5, groups=None, weights=None)`

Calcula la importancia de features mediante drop-column CV. Para cada feature, elimina la columna del pipeline, re-entrena con CV y mide el delta RMSE respecto al baseline.

**Estructura en dos fases:**

1. **Fase 1 — baselines:** se calcula una vez por modelo antes del bucle de features. No se repite `n_features` veces.
2. **Fase 2 — drop-importance:** para cada feature, promedia el delta sobre todos los modelos reutilizando los baselines de la fase 1.

**Coste:** `O(n_models × (1 + n_features) × cv_splits)` evaluaciones de `cross_val_score`.

> **Patrón recomendado:** pasar `[final_model]` reduce el coste en un factor `n_models` y produce un baseline más consistente.

```python
importance_df = xs.drop_column_cv_pipeline_clean(
    [final_model], X, y,
    cv_splits=5,
    groups=groups,
    weights=weights
)
```

Un valor positivo indica feature relevante (eliminarla empeora el modelo). Un valor negativo o cercano a cero indica feature redundante o perjudicial.

---

### Selección de features

#### `select_feat(source, X, umbral=0.001)`

Filtra features cuya importancia supere el umbral. Usado internamente por `itera_perm_mod`. Lanza `ValueError` si ninguna feature supera el umbral — comportamiento que `itera_perm_mod` captura para detener el bucle limpiamente.

```python
X_sel, num_cols, cat_cols = xs.select_feat(importance_df, X, umbral=0.005)
```

---

#### `itera_perm_mod(X, y, num_cols, cat_cols, groups, weights, models=None, cv_results=None, cv_splits=5, umbral=0.0)`

Selección iterativa batch con reentrenamiento en cada ronda. En cada iteración:

1. Calcula importancias drop-column con `[final_model]`
2. Elimina todas las features por debajo del umbral
3. Re-entrena el pipeline completo con `run_pipe` sobre las features restantes
4. Repite hasta convergencia

A diferencia de las funciones backward, esta puede eliminar varias features en una sola iteración y **reoptimiza los hiperparámetros** con cada reentrenamiento. Es la más costosa pero la más completa metodológicamente para eliminación batch.

**Returns:** `(importance, X, num_cols, cat_cols, models, cv_results, final_model)`

```python
importance, X_sel, num_cols, cat_cols, models, cv_results, final_model = xs.itera_perm_mod(
    X, y, num_cols, cat_cols, groups, weights, umbral=0.001
)
```

---

#### `sequential_drop_column_cv(models, X, y, groups, weights, cv_splits=5, threshold=0.0, verbose=True)`

Selección secuencial dinámica: recorre las features una a una y elimina cada una si su impacto en el RMSE cae por debajo del umbral. Cada eliminación modifica el baseline de las evaluaciones siguientes.

**Diferencia con `drop_column_cv_pipeline_clean`:** evalúa features en secuencia respecto a un baseline dinámico (no fijo). El orden de evaluación importa.

> **Nota:** no reoptimiza hiperparámetros durante la selección — los hereda del modelo de entrada. Solo llama a `run_pipe` una vez al final, con las features seleccionadas.

```python
X_, num_, cat_, models_, metrics_ = xs.sequential_drop_column_cv(
    [final_model], X, y, groups, weights, threshold=0.0
)
```

---

#### `stable_backward_selection_with_oof_permutation([final_model], X, y, cv_splits=3, groups=None, weights=None, threshold=0.0, n_repeats=3, random_state=RANDOM_STATE, reintroduction_every=5, max_reintroduce=2, model_selection="best", checkpoint_path=None, log_path=None, on_step_end=None)`

La función de selección canónica del módulo. En cada step:

1. **Baseline OOF:** entrena el modelo en cada fold y calcula el RMSE out-of-fold.
2. **Permutation importance:** sobre los modelos cacheados de cada fold (sin reentrenar). Se guarda en `history` para análisis posterior pero **no influye en la decisión de eliminación**.
3. **Drop importance:** para cada feature candidata, reentrena sin ella y mide el delta RMSE vs baseline.
4. **Reintroducción:** cada `reintroduction_every` steps, reevalúa features eliminadas previamente. Si añadirlas mejora el baseline, se reintroducen.
5. **Decisión:** elimina la feature con menor `drop_importance`. Se para cuando la mejor candidata a eliminar supera el `threshold`.

**Splits fijos:** los folds se generan una sola vez antes del bucle. Todas las rondas usan exactamente los mismos folds, garantizando que los RMSEs de distintos steps sean comparables entre sí.

**`model_selection`:**
- `"all"` → usa todos los modelos en cada step.
- `"best"` → en el step 0 identifica el modelo con menor RMSE y lo usa para el resto.
- Si se pasa `[final_model]`, este parámetro no tiene efecto.

**Returns:** `(history, remaining, X_, num_cols_, cat_cols_, models_, metrics_, final_model_)`

```python
history, remaining, X_, num_cols_, cat_cols_, models_, metrics_, final_model_ = \
    xs.stable_backward_selection_with_oof_permutation(
        [final_model], X, y,
        cv_splits=5,
        groups=groups,
        weights=weights,
        threshold=0.005,
        reintroduction_every=5,
        max_reintroduce=2,
        checkpoint_path="state.json",
        log_path="selection.log",
    )
```

**Estructura de `history`:** lista de dicts, uno por step:

```python
{
    "step":                     int,
    "remaining":                list[str],
    "baseline_score":           float,        # RMSE medio OOF
    "baseline_scores_per_fold": dict,         # {model_i: [rmse_fold_0, ...]}
    "drop_importance":          dict[str, float],
    "permutation_importance":   dict[str, float],
    "removed":                  str | None,   # None si se para sin eliminar
    "reintroduced":             list[str],
}
```

---

#### `exhaustive_feature_combinations(models, X, y, cv_splits=5, groups=None, weights=None, min_features=4, max_features=6, mandatory_features=None, warn_threshold=15, checkpoint_path=None, checkpoint_every=50)`

Evalúa todas las combinaciones posibles de features en el rango `[min_features, max_features]`. Las `mandatory_features` se incluyen en todas las combinaciones y no se combinan entre sí.

> **Advertencia de escala:** el número de combinaciones crece combinatoriamente. Para `n=20` features opcionales y `max_features=6` son `C(20,4)+C(20,5)+C(20,6) = 54264` combinaciones. La función muestra una advertencia y pide confirmación si se supera `warn_threshold`.

> **Patrón recomendado:** pasar `[final_model]` reduce el coste por combinación en un factor `n_models`.

```python
results_df = xs.exhaustive_feature_combinations(
    [final_model], X, y,
    min_features=3,
    max_features=5,
    mandatory_features=["grupo"],
    checkpoint_path="exhaustive_ckpt.json",
    checkpoint_every=100,
)
# results_df: DataFrame con columnas 'features', 'n_features', 'rmse'
# ordenado por RMSE ascendente, índice 1-based
```

---

### Utilidades de pipeline

#### `disable_columns_in_pipeline(pipeline, columns_to_disable, prep_step="prep")`

Clona el pipeline y elimina las columnas indicadas de todos los transformers del `ColumnTransformer`. Versión canónica — siempre clona (no muta el original), maneja correctamente transformers con `.cols` (como `TargetEncoder`) y strings como `"passthrough"` o `"drop"`.

> **Uso interno:** esta función es el núcleo de todas las evaluaciones drop-column. Se llama miles de veces durante la selección de features. Por eso es crítico que nunca mute el pipeline original.

```python
pipeline_sin_col = xs.disable_columns_in_pipeline(pipeline, ["feature_a", "feature_b"])
```

---

#### `make_checkpoint_callback(path)`

Crea un callable compatible con el parámetro `on_step_end` de `stable_backward_selection_with_oof_permutation`. Equivalente funcional a `checkpoint_path`, expresado como callback.

```python
cb = xs.make_checkpoint_callback("state.json")
history, remaining, *_ = xs.stable_backward_selection_with_oof_permutation(
    [final_model], X, y, on_step_end=cb
)
```

---

### SHAP

#### `shap_explainer(models, X)`

Calcula valores SHAP del ensemble sobre todos los datos de `X`. Cada modelo usa su propio preprocesador tanto para el background como para la evaluación, garantizando coherencia interna.

> **Nota de eficiencia:** si se pasa `[final_model]`, la transformación del preprocesador se ejecuta una sola vez. Con el ensemble completo, se ejecuta una vez por modelo.

```python
sv = xs.shap_explainer(models_, X_)
shap.plots.beeswarm(sv)
```

#### `shap_values(models, X, size_sample=500, random_state=RANDOM_STATE)`

Envuelve `shap_explainer` aplicando un muestreo previo. Reduce el coste en datasets grandes.

```python
sv = xs.shap_values(models_, X_, size_sample=300)
```

---

### Visualización

#### `feat_corr(X, threshold=0.8)`

Mapa de calor de correlaciones. Las zonas por encima del umbral se resaltan con colores cálidos; las inferiores con fríos.

```python
xs.feat_corr(X, threshold=0.8)
```

#### `hist_track(history)`

Transforma el historial de selección en estructuras tabulares. Devuelve un DataFrame longitudinal con importancias por feature y step, y una lista con la evolución del RMSE baseline.

```python
records_df, rmse_steps = xs.hist_track(history)
```

#### `steps_plot(history, save_path=None)`

Gráfico de líneas con la evolución de la drop-importance por feature a lo largo del proceso. El eje X muestra la feature eliminada en cada step.

```python
xs.steps_plot(history, save_path="steps.png")
```

#### `drop_perm_plot(df_track)`

Scatter plot de drop-importance vs permutation-importance promediadas sobre todas las rondas. Las cuatro zonas del gráfico permiten identificar features ruidosas, redundantes y genuinamente informativas.

```python
xs.drop_perm_plot(records_df)
```

---

## Decisiones de diseño y advertencias críticas

### El modelo de producción y el ensemble son artefactos distintos

`run_pipe` devuelve dos artefactos con propósitos diferentes que **no deben confundirse**:

| Artefacto | Entrenado en | Propósito |
|---|---|---|
| `models` (ensemble) | ≈90% datos (por fold) | Estimación de varianza, `predict_ensemble`, `shap_explainer` |
| `final_model` | 100% datos | **Predicción en producción** |

El RMSE fiable es el del nested CV (`get_metrics(cv_results)`). El `final_model` no tiene RMSE propio imparcial — ese es el del outer CV, calculado antes del refit.

### Por qué no se usa `cross_val_score` para obtener el modelo de producción

Una alternativa tentadora sería llamar a `search.fit(X, y)` con el grid completo para obtener los mejores hiperparámetros sobre todos los datos. Esta opción fue descartada porque el CV interno ya ha ejecutado 10 veces el inner CV durante el nested CV. Re-ejecutarlo sobre `X` completa sería redundante y costoso. En su lugar, `run_pipe` reduce el inner CV a 2 folds (mínimo válido en sklearn) solo para que `HalvingGridSearchCV` seleccione hiperparámetros antes del refit automático — sin ningún coste de búsqueda adicional significativo.

### Los hiperparámetros no se reoptimizaban durante la selección (intencional)

Las funciones de selección (`drop_column_cv_pipeline_clean`, `sequential_drop_column_cv`, `stable_backward_selection_with_oof_permutation`) heredan los hiperparámetros del modelo de entrada y los mantienen fijos durante todo el proceso de selección. Solo `itera_perm_mod` reoptimiza hiperparámetros en cada iteración vía `run_pipe`. Esta decisión es intencional:

- La selección de features es una búsqueda de subconjunto, no una búsqueda de hiperparámetros.
- Reoptimizar en cada step multiplicaría el coste por el número de combinaciones evaluadas.
- La hipótesis de que los hiperparámetros óptimos no cambian drásticamente al eliminar features poco informativas es razonable en la práctica.

### `sample_weight=None` nunca se pasa explícitamente

Con metadata routing activo (sklearn >= 1.3), pasar `sample_weight=None` explícitamente en el dict `params` le dice al router que enrute `None` al estimador. En ciertas versiones de sklearn esto provoca `ValueError`. El módulo construye `params` condicionalmente: `sample_weight` solo se incluye si `weights is not None`.

### `disable_columns_in_pipeline` siempre clona

Versiones anteriores de esta función mutaban el pipeline original. Esto causaba que los modelos del ensemble compartieran el mismo `ColumnTransformer` internamente, produciendo resultados incorrectos en evaluaciones paralelas. La versión actual siempre clona con `sklearn.base.clone` antes de modificar cualquier transformer.

### Splits fijos en `stable_backward_selection_with_oof_permutation`

Los folds se generan antes del bucle principal y permanecen fijos durante toda la selección. Esto garantiza que el RMSE del step N sea directamente comparable con el del step N+1 — ambos se evalúan sobre exactamente los mismos datos de validación. Sin splits fijos, la varianza entre splits podría enmascarar el efecto real de eliminar una feature.

---

## Logging

El módulo usa un logger con el nombre del módulo (`xgb_sculpt`) y añade un `NullHandler` al importarse. Esto sigue la práctica recomendada de Python para librerías: delegar la configuración al cliente.

```python
# Activar en scripts
import logging
logging.basicConfig(level=logging.INFO)

# Activar solo el logger del módulo
logging.getLogger("xgb_sculpt").setLevel(logging.DEBUG)

# Logger específico de la función de selección
# (usa un logger interno con nombre xgb_sculpt.selection_<id>)
logging.getLogger("xgb_sculpt.selection").setLevel(logging.INFO)
```

`stable_backward_selection_with_oof_permutation` crea un logger dedicado por ejecución (con `_setup_logger`) y lo destruye al terminar (con `_teardown_logger`). Esto evita la acumulación de handlers si la función se llama varias veces en la misma sesión.

---

## Checkpointing y reanudación

`stable_backward_selection_with_oof_permutation` y `exhaustive_feature_combinations` soportan guardado y reanudación automática.

### Escritura atómica

El checkpoint se escribe en un fichero temporal y luego se renombra sobre el destino. Esto garantiza que una interrupción durante la escritura no deje el fichero corrupto.

### Resume en `stable_backward_selection_with_oof_permutation`

```python
# Primera ejecución
history, remaining, *_ = xs.stable_backward_selection_with_oof_permutation(
    [final_model], X, y, checkpoint_path="state.json"
)

# Si se interrumpe, la siguiente llamada reanuda desde el último step
history, remaining, *_ = xs.stable_backward_selection_with_oof_permutation(
    [final_model], X, y, checkpoint_path="state.json"
)
```

Los splits se leen del checkpoint para garantizar que el resume usa exactamente los mismos folds que la ejecución original. `X`, `models` y los parámetros deben ser idénticos.

### Resume en `exhaustive_feature_combinations`

El checkpoint registra las combinaciones ya evaluadas. En reanudaciones, `newly_evaluated` (contador de combinaciones evaluadas en la sesión actual) es independiente de `already_evaluated` (cargado del checkpoint), lo que garantiza que el primer checkpoint de la sesión de reanudación se escriba tras la primera nueva combinación evaluada.

---

## Guía de selección de función

| Objetivo | Función recomendada |
|---|---|
| Entrenar pipeline inicial y obtener métricas | `run_pipe` + `get_metrics` |
| Evaluar el modelo de producción con CV externo | `eval_final_model` |
| Ver qué features son relevantes sin eliminar ninguna | `drop_column_cv_pipeline_clean` o `permutation_ensemble` |
| Eliminar features claramente no informativas de forma rápida | `itera_perm_mod` |
| Selección secuencial conservadora con control de umbral | `sequential_drop_column_cv` |
| Selección backward robusta con reintroducción y checkpoint | `stable_backward_selection_with_oof_permutation` |
| Encontrar el subconjunto exactamente óptimo (hasta n features) | `exhaustive_feature_combinations` |
| Explicar el modelo final | `shap_values` + `shap.plots.beeswarm` |
| Detectar multicolinealidad antes de seleccionar | `feat_corr` |
| Analizar el proceso de selección a posteriori | `hist_track` + `steps_plot` + `drop_perm_plot` |
