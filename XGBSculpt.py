"""
xgb_sculpt
Python module for training, evaluating and progressively refining XGBoost
regression pipelines through a suite of feature selection methods.
It combines nested cross-validation, group-aware splitting, inverse-frequency
sample weighting and multiple importance metrics.

Use this only when there is a 'sample_id' attribute (vg. repeated trials) and 
a category feature to apply Target_Encode

import xgb_sculpt as xg
logging.basicConfig(level=logging.INFO)
xg.set_display_options()

drop_cols = ['feat_1', 'feat_2']

y, X, num_cols, cat_cols, groups, weights = xg.data_read(
    csv_name    ='grouped_database_1u.csv',
    sample_id   ='sample',       
    target_name ='target',
    average_by_sample = False,     
    drop_cols   = drop_cols
)


# Change PARAM_HGS for use your own hyperparams grid (see PARAM_HGS 120 rows below) 
models_, cv_results_, final_model_ = xg.run_pipe(
    X          = X_,
    y          = y,
    num_cols   = num_cols_,
    cat_cols   = cat_cols_,
    groups     = groups,
    weights    = weights
)

metrics_ = xg.get_metrics(cv_results_)

hyper = xg.get_hyperparams(
    cv_results = cv_results_,
    X          = X_,
    y          = y,
    weights    = weights
)

 fold  train_rmse  val_rmse  test_rmse
    1       0.177     0.349      0.405
    2       0.186     0.367      0.320
    3       0.185     0.364      0.350
    4       0.188     0.363      0.311
    5       0.183     0.348      0.346
    6       0.177     0.352      0.366
    7       0.184     0.349      0.303
    8       0.183     0.362      0.352
    9       0.185     0.360      0.268
   10       0.178     0.341      0.384
INFO:xgb_sculpt_07:Agregado (media / std):
      train_rmse  val_rmse  test_rmse
mean       0.183     0.355      0.341
std        0.004     0.009      0.041

importance_1 = xg.permutation_ensemble(
    models = models_,
    X      = X_,
    y      = y
)

X_1, num_cols_1, cat_cols_1 = xg.select_feat(
    source = importance_1,
    X      = X_,
    umbral = 0.006
)

print(f"\nFeatures antes de la selección: {X_.shape[1]}")
print(f"Features tras  la selección:    {X_1.shape[1]}")
print(f"Features eliminadas:            {X_.shape[1] - X_1.shape[1]}")


........
........
........

"""
# =============================================================================
# Logging
# =============================================================================
import logging

# El módulo delega la configuración de logging al cliente importador.
# Para activar mensajes en scripts/notebooks basta con:
#   logging.basicConfig(level=logging.INFO)
# o configurar un handler específico sobre el logger del módulo:
#   logging.getLogger("xgb_sculpt").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# Librería estándar
# =============================================================================
import sys
import json
import pathlib
import warnings

# =============================================================================
# Análisis y visualización
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text  # etiquetas sin solapamiento en scatter

# =============================================================================
# Machine learning
# =============================================================================
import xgboost as xgb
import shap

from sklearn.base               import clone
from sklearn                    import set_config
from sklearn.inspection         import permutation_importance
from sklearn.model_selection    import GroupKFold, KFold, cross_validate, cross_val_score
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler
from sklearn.preprocessing      import TargetEncoder
from sklearn.pipeline           import Pipeline
from sklearn.experimental       import enable_halving_search_cv  # noqa: activa HalvingGridSearchCV
from sklearn.model_selection    import HalvingGridSearchCV
from sklearn.metrics            import root_mean_squared_error

# =============================================================================
# Semilla global de aleatoriedad
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# Grid de hiperparámetros para HalvingGridSearchCV
# =============================================================================
PARAM_HGS_ = {
    "model__max_depth":        [2, 3, 4],
    "model__n_estimators":     [100, 200, 300],
    "model__learning_rate":    [0.05, 0.15, 0.25],
    "model__subsample":        [0.5, 0.75],
    "model__colsample_bytree": [0.6, 0.8],
    "model__min_split_loss":   [0.01, 0.1],
    "model__reg_lambda":       [1, 5],
    "prep__cat__smooth":       ['auto', 10, 50],   # regularización del TargetEncoder
}

PARAM_HGS = {
    "model__max_depth":        [2],
    "model__n_estimators":     [140],
    "model__learning_rate":    [0.25],
    "model__subsample":        [0.6],
    "model__colsample_bytree": [0.6],
    "model__min_split_loss":   [0.1],
    "model__reg_lambda":       [15],
    "prep__cat__smooth":       [20],
}

__all__ = [
    # Configuración
    "set_display_options",
    # Datos
    "data_read",
    # Pipeline principal
    "run_pipe",
    "predict_ensemble",
    "get_metrics",
    "get_hyperparams",
    "eval_final_model",
    # Importancias de features
    "permutation_ensemble",
    "drop_column_cv_pipeline_clean",
    # Selección de features
    "select_feat",
    "itera_perm_mod",
    "sequential_drop_column_cv",
    "stable_backward_selection_with_oof_permutation",
    "exhaustive_feature_combinations",
    # Utilidades de pipeline
    "disable_columns_in_pipeline",
    "make_checkpoint_callback",
    # SHAP
    "shap_explainer",
    "shap_values",
    # Visualización
    "feat_corr",
    "hist_track",
    "steps_plot",
    "drop_perm_plot",
]

# --- AUTO-DETECCIÓN DE GPU ---
try:
    # Intenta instanciar y entrenar un modelo dummy en la GPU
    xgb.XGBRegressor(tree_method="hist", device="cuda").fit(np.zeros((1,1)), np.zeros(1))
    GPU_AVAILABLE = True
    logger.info("🟢 CUDA/GPU detectada y a disposición de XGBoost.")
except Exception:
    GPU_AVAILABLE = False
    logger.warning("🟡 No se detectó GPU válida. XGBoost usará la CPU.")
    
    
def set_display_options():
    """Aplica opciones de visualización de pandas, NumPy y Jupyter (opcional).

    Configura formatos de presentación de DataFrames y arrays. Debe llamarse
    explícitamente desde el cliente (script o notebook); el módulo no altera
    el entorno global al importarse.

    Effects:
        - pandas: float a 3 decimales, hasta 1500 filas y 300 columnas,
          sin partir DataFrames anchos en varias líneas.
        - NumPy: precisión de 3 decimales en ``print``.
        - Jupyter: inyecta CSS para evitar el truncado de texto en celdas
          (solo si ``ipykernel`` está cargado).

    Example:
        >>> import xgb_sculpt as xs
        >>> xs.set_display_options()
    """
    pd.options.display.float_format       = '{:,.3f}'.format
    pd.options.display.max_rows           = 1500
    pd.options.display.max_columns        = 300
    pd.set_option('display.expand_frame_repr', False)
    np.set_printoptions(precision=3)

    if 'ipykernel' in sys.modules:
        from IPython.display import display, HTML
        display(HTML("<style>pre { white-space: pre !important; }</style>"))

#%% Lectura
def data_read(csv_name, sample_id, target_name, drop_cols=None,
              sep=",", encoding="utf-8", average_by_sample=False):
    """Lee un CSV y prepara los datos para modelado.

    Carga el fichero, separa la variable objetivo de las features, clasifica
    las columnas por tipo y calcula pesos inversos para muestras con múltiples
    medidas del target.

    Args:
        csv_name (str): Ruta al fichero CSV de entrada.
        sample_id (str): Nombre de la columna identificadora de muestra.
            Se usa para detectar múltiples medidas del target por muestra
            y calcular los pesos correspondientes. ``None`` si no hay
            estructura de grupos.
        target_name (str): Nombre de la columna objetivo (variable a predecir).
        drop_cols (list, optional): Columnas adicionales a eliminar de las
            features. Por defecto [].
        sep (str, optional): Separador de columnas del CSV. Por defecto
            ``","``; usar ``";"`` para exportaciones europeas de Excel.
        encoding (str, optional): Codificación del fichero. Por defecto
            ``"utf-8"``; usar ``"latin-1"`` o ``"windows-1252"`` si el CSV
            contiene caracteres especiales del español y falla con UTF-8.
        average_by_sample (bool, optional): Si es ``True``, agrupa el dataset
            por ``sample_id`` promediando el target. Para el resto de variables
            toma el primer valor que encuentra (supone que dichas features son
            iguales entre samples).

    Returns:
        tuple:
            - y (pd.Series): Variable objetivo.
            - X (pd.DataFrame): Features sin la columna objetivo ni el
              identificador de muestra.
            - num_cols (list[str]): Columnas numéricas y booleanas casteadas
              a float64 (compatibilidad con SHAP).
            - cat_cols (list[str]): Columnas categóricas (dtype object).
            - groups (pd.Series): Serie con el identificador de muestra,
              útil para GroupKFold (evitar presencia en train y test).
            - weights (pd.Series): Pesos inversos al número de medidas por
              muestra, para evitar que muestras con más registros dominen
              el entrenamiento.

    Example:
        >>> y, X, num_cols, cat_cols, groups, weights = data_read(
        ...     csv_name="datos.csv",
        ...     sample_id="paciente_id",
        ...     target_name="diagnostico",
        ...     drop_cols=["fecha", "notas"]
        ... )
    """
    if drop_cols is None:
        drop_cols = []
        
    df = pd.read_csv(csv_name, sep=sep, encoding=encoding)

    if sample_id:
        if average_by_sample:
            # En dos pasos (en un único paso convierte sample_id(str) en tupla)
            # 1-Obtiene la media
            target_mean = df.groupby(sample_id)[target_name].transform('mean')
            df[target_name] = target_mean
            # 2-Elimina duplicados (supone: misma sample_id igual valor feature)
            df = df.drop_duplicates(subset=[sample_id]).reset_index(drop=True)
        
        # Detecta cuántas medidas (filas) existen por muestra y asigna
        # peso inverso para compensar el desbalance en el entrenamiento
        # if average_target: counts and weights = 1 
        groups = df[sample_id].copy()
        counts = df.groupby(sample_id)[sample_id].transform('count')
        weights = 1.0 / counts
    else:
        groups = None
        weights = None

    # Construye la lista completa de columnas a excluir de las features
    id_cols = [sample_id] if sample_id is not None else []
    drop_cols = drop_cols + id_cols + [target_name]

    y = df[target_name]
    X = df.drop(drop_cols, axis=1).copy()

    # Castea numéricas y booleanas a float64 para evitar errores en SHAP
    num_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    X[num_cols] = X[num_cols].astype('float64')

    cat_cols = X.select_dtypes(include='object').columns.tolist()

    logger.debug("Estadísticas descriptivas del dataset:\n%s",
                 df.describe(include='all').to_string())

    return y, X, num_cols, cat_cols, groups, weights

#%% Pipeline
def run_pipe(X, y, num_cols, cat_cols, groups, weights,
             force_cpu=True,
             random_state=RANDOM_STATE,
             param_grid=PARAM_HGS):
    """Construye y ejecuta un pipeline de ML con nested cross-validation.

    Configura un pipeline completo que incluye preprocesado (escalado +
    target encoding), un regresor XGBoost y búsqueda de hiperparámetros
    acelerada (HalvingGridSearchCV). La evaluación se realiza mediante
    nested CV con GroupKFold para respetar la estructura de grupos en los
    datos.

    Args:
        X (pd.DataFrame): Features de entrada, con columnas numéricas y
            categóricas ya separadas.
        y (pd.Series): Variable objetivo continua.
        num_cols (list[str]): Nombres de las columnas numéricas a escalar.
        cat_cols (list[str]): Nombres de las columnas categóricas a codificar
            con TargetEncoder.
        groups (pd.Series): Identificador de grupo por muestra, usado en
            GroupKFold para evitar data leakage entre splits.
        weights (pd.Series): Pesos por muestra, pasados al fit de XGBoost
            para compensar muestras con múltiples medidas del target.
        random_state (int, optional): Semilla de aleatoriedad. Por defecto
            ``RANDOM_STATE``.
        param_grid (dict, optional): Grid de hiperparámetros para
            ``HalvingGridSearchCV``. Por defecto ``PARAM_HGS``.

    Returns:
        tuple:
            - models (list[Pipeline]): Lista de ``best_estimator_`` de cada fold
              del outer CV, que forman el ensemble (≈90% datos por fold).
            - cv_results (dict): Diccionario devuelto por ``cross_validate``,
              con métricas (``test_score``) y estimadores (``estimator``)
              de cada fold outer.
            - final_model (list[Pipeline]): Lista de un único elemento — el
              modelo de producción entrenado sobre el 100% de los datos con
              los hiperparámetros del fold outer de mejor rendimiento.
              Se devuelve como lista para ser directamente compatible con
              las funciones de selección de features sin necesidad de
              envolver en ``[...]``.

    Notes:
        - Requiere ``enable_metadata_routing=True`` (sklearn >= 1.3) para
          enrutar ``groups`` y ``sample_weight`` a través del pipeline.
        - El inner CV optimiza hiperparámetros con ``HalvingGridSearchCV``
          usando RMSE negativo como métrica.
        - El outer CV estima el rendimiento generalizado del proceso de
          selección completo, no solo del modelo final.
        - ``final_model`` se construye tomando los hiperparámetros del fold
          outer con mejor ``test_score`` (menor RMSE). Esos hiperparámetros
          nunca vieron el test de ese fold — no hay leakage. El pipeline
          se clona, se fijan los hiperparámetros y se hace un único ``fit``
          sobre el 100% de los datos, sin ningún CV adicional.
        - ``Pipeline.fit()`` no acepta ``groups``; ese argumento es solo
          para los splitters de CV. El fit final usa únicamente
          ``sample_weight`` si está disponible.

    Example:
        >>> models, cv_results, final_model = run_pipe(
        ...     X, y, num_cols, cat_cols, groups, weights
        ... )
        >>> rmse_scores = -cv_results["test_score"]
        >>> print(f"RMSE medio: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
        >>> # Usar directamente en selección de features:
        >>> importance = drop_column_cv_pipeline_clean(final_model, X, y, ...)
    """

    # =========================================================
    # Activar metadata routing (sklearn >= 1.3)
    # Necesario para propagar groups y sample_weight a través del pipeline
    # =========================================================
    set_config(enable_metadata_routing=True)

    # =========================================================
    # Target encoding para variables categóricas
    # smoothing=20 controla el balance entre la media global y la media
    # por categoría (mayor valor → más regularización)
    # =========================================================
    target_encoder = TargetEncoder(
        target_type="continuous",
        smooth="auto" # Será optimizado por el grid
    )

    # =========================================================
    # Preprocesado: escalado de numéricas + encoding de categóricas
    # sample_weight=False en el scaler para que no intente recibirlo
    # =========================================================
    scaler = StandardScaler().set_fit_request(sample_weight=False)

    preprocess = ColumnTransformer([
        ("num", scaler, num_cols),
        ("cat", target_encoder, cat_cols)
    ])

    if force_cpu or not GPU_AVAILABLE:
        target_device = "cpu"
    else:
        target_device = "cuda"
    # =========================================================
    # Modelo: XGBoost regresor con MSE como función de pérdida
    # sample_weight=True para que el pipeline le pase los pesos al fit
    # =========================================================
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,       # valor base; HalvingGridSearchCV lo varía
        tree_method="hist",
        device=target_device,
        random_state=random_state,       
        n_jobs=4
    )
    model.set_fit_request(sample_weight=True)

    # =========================================================
    # Pipeline: encadena preprocesado y modelo
    # =========================================================
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    # =========================================================
    # Cross-validation con GroupKFold (inner y outer)
    # GroupKFold garantiza que un mismo grupo no aparezca en train y test
    # =========================================================
    if groups is not None:
        inner_cv = GroupKFold(n_splits=10, shuffle=True, random_state=random_state)
        outer_cv = GroupKFold(n_splits=10, shuffle=True, random_state=random_state)
        inner_cv.set_split_request(groups=True)
        outer_cv.set_split_request(groups=True)
    else:
        inner_cv = KFold(n_splits=10)
        outer_cv = KFold(n_splits=10)

    # =========================================================
    # Búsqueda de hiperparámetros con Successive Halving
    # Descarta progresivamente configuraciones malas usando menos recursos,
    # lo que acelera la búsqueda respecto a GridSearchCV estándar
    # =========================================================
    search = HalvingGridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        factor=3,                   # en cada ronda elimina 2/3 de candidatos
        resource="n_samples",      # <-- Aquí lo declaras de forma explícita
        min_resources="exhaust",   # Calcula el tamaño ideal de la 1ª muestra
        max_resources="auto",
        return_train_score=True,    # necesario para get_metrics
        verbose=0,
        random_state=random_state,
        n_jobs=4
    )

    # =========================================================
    # Nested CV: evalúa el proceso completo de selección de hiperparámetros
    # Devuelve métricas por fold y el estimador entrenado en cada fold
    # =========================================================
    if groups is not None:
        params = {"groups": groups}
        if weights is not None:
            params["sample_weight"] = weights
    else:
        params = None if weights is None else {"sample_weight": weights}

    cv_results = cross_validate(
        search,
        X,
        y,
        cv=outer_cv,
        scoring="neg_root_mean_squared_error",
        return_estimator=True,
        params=params,
    )

    # Extrae el mejor estimador de cada fold outer para formar el ensemble
    models = [est.best_estimator_ for est in cv_results["estimator"]]

    # =========================================================
    # Modelo de producción — entrenado sobre el 100% de los datos
    #
    # Los hiperparámetros ya fueron determinados por el nested CV (10×10 folds)
    # — la estimación más fiable posible. No tiene sentido volver a buscarlos
    # con otro inner CV. Se toman del fold con mejor test_score outer (menor RMSE)
    # y se hace un único fit sobre el 100% de los datos.
    # Se devuelve como lista de un elemento para compatibilidad con las funciones
    # de selección de features.
    # =========================================================
    best_fold_idx = int(np.argmax(-cv_results["test_score"]))  # menor RMSE
    best_params   = cv_results["estimator"][best_fold_idx].best_params_

    final_pipe = clone(pipe)
    final_pipe.set_params(**best_params)

    # Pipeline.fit() NO acepta `groups` — ese argumento es exclusivo de los
    # splitters de CV. Se construye fit_kwargs independientemente de `params`,
    # incluyendo solo sample_weight si está disponible.
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    final_pipe.fit(X, y, **fit_kwargs)
    final_model = [final_pipe]

    return models, cv_results, final_model

#%%
def predict_ensemble(models, X):
    """Genera predicciones agregadas a partir de un ensemble de modelos.

    Obtiene la predicción de cada modelo del ensemble y calcula la media
    y la desviación estándar por muestra, ofreciendo así una estimación
    puntual y una medida de incertidumbre entre modelos.

    Args:
        models (list): Lista de modelos entrenados con método ``predict``,
            habitualmente los ``best_estimator_`` devueltos por
            ``run_pipe``.
        X (pd.DataFrame): Features de entrada con la misma estructura
            usada durante el entrenamiento.

    Returns:
        tuple:
            - mean_preds (np.ndarray): Media de las predicciones de todos
              los modelos, shape ``(n_samples,)``.
            - std_preds (np.ndarray): Desviación estándar de las
              predicciones entre modelos, shape ``(n_samples,)``.
              Puede interpretarse como proxy de incertidumbre del ensemble.

    Example:
        >>> mean_preds, std_preds = predict_ensemble(models, X_test)
        >>> print(f"Predicción: {mean_preds[0]:.3f} ± {std_preds[0]:.3f}")
    """
    # Matriz de predicciones: shape (n_samples, n_models)
    if not models:
        raise ValueError("La lista de modelos está vacía.")
        
    preds = np.column_stack([m.predict(X) for m in models])

    return preds.mean(axis=1), preds.std(axis=1)

#%%
def get_metrics(cv_results):
    """Extrae y resume las métricas RMSE de train, validación y test del nested CV.

    Recorre los estimadores del outer CV y, para cada fold, recupera el
    RMSE de entrenamiento e inner-validación del mejor índice encontrado
    por ``HalvingGridSearchCV``, junto con el RMSE de test del outer CV.
    Imprime el detalle por fold y un resumen agregado (media y desviación
    estándar).

    Args:
        cv_results (dict): Diccionario devuelto por ``cross_validate`` con
            ``return_estimator=True``. Debe contener las claves
            ``'estimator'`` (lista de ``HalvingGridSearchCV`` ajustados)
            y ``'test_score'`` (scores del outer CV en negativo).

    Returns:
        pd.DataFrame: Tabla con una fila por fold y las columnas:

            - ``fold`` (int): Número de fold (1-indexado).
            - ``train_rmse`` (float): RMSE medio en train del inner CV
              para el mejor conjunto de hiperparámetros.
            - ``val_rmse`` (float): RMSE medio en validación del inner CV
              para el mejor conjunto de hiperparámetros.
            - ``test_rmse`` (float): RMSE en test del outer CV.

    Notes:
        - Los scores de sklearn se almacenan en negativo
          (``neg_root_mean_squared_error``); la función aplica el signo
          negativo para devolverlos como valores positivos.
        - ``mean_train_score`` no se calcula por defecto en
          ``HalvingGridSearchCV``; es necesario instanciarlo con
          ``return_train_score=True``.

    Example:
        >>> metrics_df = get_metrics(cv_results)
           fold  train_rmse  val_rmse  test_rmse
        1     1      0.312     0.351      0.368
        ...
        >>> metrics_df["test_rmse"].mean()
        0.371
    """
    if 'estimator' not in cv_results:
        raise KeyError("cv_results debe generarse con return_estimator=True.")

    resumen = []
    for i, est in enumerate(cv_results['estimator']):
        if not isinstance(est, HalvingGridSearchCV):
            raise TypeError(
                f"El estimador del fold {i + 1} es {type(est).__name__!r}, "
                f"pero se esperaba HalvingGridSearchCV. "
                f"cv_results debe provenir de run_pipe()."
            )
        # Índice del mejor conjunto de hiperparámetros en el inner CV
        idx = est.best_index_
        resumen.append({
            'fold':       i + 1,
            'train_rmse': -est.cv_results_['mean_train_score'][idx],  # inner CV train
            'val_rmse':   -est.cv_results_['mean_test_score'][idx],   # inner CV validación
            'test_rmse':  -cv_results['test_score'][i]                # outer CV test
        })

    metrics_res = pd.DataFrame(resumen)

    logger.info("Métricas por fold:\n%s", metrics_res.to_string(index=False))
    logger.info(
        "Agregado (media / std):\n%s",
        metrics_res.agg(['mean', 'std']).drop('fold', axis=1).to_string()
    )

    return metrics_res

#%%
def get_hyperparams(cv_results, X, y, weights):
    """Extrae los mejores hiperparámetros del nested CV y evalúa el ensemble.

    Recorre los estimadores del outer CV para recopilar los hiperparámetros
    óptimos de cada fold, evalúa el ensemble completo sobre el conjunto
    proporcionado y muestra el RMSE medio del outer CV.

    Args:
        cv_results (dict): Diccionario devuelto por ``cross_validate`` con
            ``return_estimator=True``. Debe contener las claves
            ``'estimator'`` y ``'test_score'``.
        X (pd.DataFrame): Features con la misma estructura usada en el
            entrenamiento, sobre las que se evalúa el ensemble.
        y (pd.Series): Variable objetivo real, usada para calcular el RMSE
            del ensemble.
        weights (pd.Series): Pesos por muestra aplicados al cálculo del
            RMSE (mismo vector que en el entrenamiento).

    Returns:
        pd.DataFrame: Tabla con una fila por fold y una columna por cada
            hiperparámetro optimizado, facilitando la comparación de los
            valores seleccionados en cada split del outer CV.

    Notes:
        - El RMSE del ensemble se calcula sobre **todos** los datos de
          entrada (``X``, ``y``), no sobre un conjunto de test separado.
          Esto puede producir una estimación optimista si ``X`` incluye
          datos de entrenamiento.
        - Requiere que ``predict_ensemble`` esté definida en el mismo
          ámbito.

    Example:
        >>> best_params_df = get_hyperparams(cv_results, X, y, weights)
           model__learning_rate  model__reg_lambda  ...
        0                  0.05                0.3
        1                  0.10                0.4
        ...
    """
    # Extrae los mejores hiperparámetros de cada fold del outer CV
    best_params = []
    for i, est in enumerate(cv_results['estimator']):
        best_params.append(est.best_params_)

    # Ensemble con los best_estimator_ de todos los folds del outer CV
    models = [est.best_estimator_ for est in cv_results["estimator"]]
    y_pred, _ = predict_ensemble(models, X)

    # RMSE ponderado del ensemble sobre los datos proporcionados
    rmse_ensemble = root_mean_squared_error(y, y_pred, sample_weight=weights)
    
    logger.info("RMSE del ensemble sobre todos los datos (in-sample): %.4f", rmse_ensemble)
    # RMSE medio y dispersión del outer CV (estimación generalizada)
    rmse_outer = -cv_results["test_score"]
    logger.info("RMSE medio outer CV:  %.4f", rmse_outer.mean())
    logger.info("RMSE std  outer CV: ±%.4f", rmse_outer.std())

    return pd.DataFrame(best_params)

#%%
def eval_final_model(final_model, X, y, groups=None, weights=None,
                     cv_splits=10, random_state=RANDOM_STATE):
    """Estima el RMSE de generalización del modelo de producción mediante CV externo.

    A diferencia del nested CV de ``run_pipe`` — que evalúa el proceso completo
    de búsqueda de hiperparámetros —, esta función evalúa directamente el
    ``final_model`` con sus hiperparámetros ya fijados. En cada fold, clona el
    pipeline, reajusta los árboles con los datos de train del fold y evalúa
    sobre el test. No hay ningún inner CV ni búsqueda de hiperparámetros.

    Args:
        final_model (list[Pipeline]): Lista de un único pipeline, tal como
            devuelve ``run_pipe`` en su tercer elemento.
        X (pd.DataFrame): Features completas, con la misma estructura usada
            en el entrenamiento.
        y (pd.Series): Variable objetivo.
        groups (pd.Series, optional): Identificador de grupo por muestra para
            ``GroupKFold``. Por defecto None.
        weights (pd.Series, optional): Pesos por muestra. Por defecto None.
        cv_splits (int, optional): Número de folds del CV externo. Por defecto 10.
        random_state (int, optional): No usado directamente pero mantenido por
            consistencia de firma con el resto del módulo.

    Returns:
        pd.DataFrame: DataFrame con columnas ``fold``, ``rmse`` y una fila de
            resumen con ``fold='mean'`` y ``fold='std'``.

    Notes:
        - El RMSE de este CV externo es ligeramente más optimista que el del
          nested CV de ``run_pipe``, porque los hiperparámetros se seleccionaron
          viendo todos los datos. La diferencia suele ser pequeña en la práctica.
        - El coste es ``O(cv_splits)`` fits — sin inner CV — frente a
          ``O(N_configs_halving × cv_splits × cv_splits)`` del nested CV.
        - Usar ``groups`` si el dataset tiene estructura de grupos para evitar
          data leakage entre folds.

    Example:
        >>> models, cv_results, final_model = run_pipe(X, y, num_cols, cat_cols, groups, weights)
        >>> rmse_df = eval_final_model(final_model, X, y, groups=groups, weights=weights)
        >>> print(rmse_df)
        >>>
        >>> # Comparar nested CV vs eval_final_model:
        >>> nested_rmse = -cv_results["test_score"].mean()
        >>> final_rmse  = rmse_df.loc[rmse_df["fold"] == "mean", "rmse"].iloc[0]
        >>> print(f"Nested CV RMSE : {nested_rmse:.4f}")
        >>> print(f"Final model RMSE: {final_rmse:.4f}")
    """
    set_config(enable_metadata_routing=True)

    if groups is not None:
        outer_cv = GroupKFold(n_splits=cv_splits)
        outer_cv.set_split_request(groups=True)
        params = {"groups": groups}
    else:
        outer_cv = KFold(n_splits=cv_splits)
        params = {}

    if weights is not None:
        params["sample_weight"] = weights

    scores = cross_val_score(
        final_model[0],
        X, y,
        cv=outer_cv,
        scoring="neg_root_mean_squared_error",
        params=params if params else None,
        n_jobs=-1,
    )

    rmse_per_fold = -scores
    rows = [{"fold": i + 1, "rmse": float(r)} for i, r in enumerate(rmse_per_fold)]
    rows.append({"fold": "mean", "rmse": float(rmse_per_fold.mean())})
    rows.append({"fold": "std",  "rmse": float(rmse_per_fold.std())})

    result = pd.DataFrame(rows)

    logger.info(
        "eval_final_model | RMSE: %.4f ± %.4f  (cv_splits=%d)",
        rmse_per_fold.mean(), rmse_per_fold.std(), cv_splits,
    )

    return result

#%%
def permutation_ensemble(models, X, y, random_state=RANDOM_STATE):
    """Calcula la importancia de features por permutación promediada sobre el ensemble.

    Para cada modelo del ensemble obtiene la importancia por permutación
    (``permutation_importance``) y promedia los resultados, ofreciendo una
    estimación más robusta que la de un único modelo.

    Args:
        models (list): Lista de modelos entrenados con método ``predict``,
            habitualmente los ``best_estimator_`` devueltos por ``run_pipe``.
        X (pd.DataFrame): Features sobre las que calcular la importancia.
            Idealmente un conjunto de test no visto durante el entrenamiento.
        y (pd.Series): Variable objetivo real asociada a ``X``.
        random_state (int, optional): Semilla para la permutación aleatoria.
            Por defecto 42.

    Returns:
        pd.DataFrame: Tabla ordenada de mayor a menor importancia con las
            columnas:

            - ``feature`` (str): Nombre de la feature.
            - ``importance`` (float): Media de ``importances_mean`` de cada
              modelo del ensemble.

    Notes:
        - La importancia por permutación mide cuánto empeora el RMSE al
          permutar aleatoriamente una columna; valores cercanos a 0 o
          negativos indican features poco o nada informativas.
        - Calcular la importancia sobre datos de entrenamiento puede
          sobreestimar la relevancia de features; se recomienda usar un
          conjunto de test separado.
        - ``n_repeats=5`` ofrece un balance entre estabilidad y coste
          computacional; aumentarlo reduce la varianza de la estimación.

    Example:
        >>> importance_df = permutation_ensemble(models, X_test, y_test)
        >>> print(importance_df.head(10))
              feature  importance
        0   feature_a      0.142
        1   feature_b      0.098
        ...
    """
    importances = []
    for m in models:
        r = permutation_importance(
            m,
            X,
            y,
            n_repeats=5,        # aumentar para estimaciones más estables
            random_state=random_state,
            n_jobs=-1
        )
        importances.append(r.importances_mean)

    # Promedia la importancia de cada feature a lo largo del ensemble
    mean_importances = np.mean(importances, axis=0)

    importance = pd.DataFrame({
        "feature":    X.columns,
        "importance": mean_importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance

#%%
def select_feat(source, X, umbral=0.001):
    """Filtra features según su importancia por permutación.

    Selecciona las columnas de ``X`` cuya importancia en ``source`` supera
    el umbral indicado y reclasifica las columnas resultantes en numéricas
    y categóricas.

    Args:
        source (pd.DataFrame): Tabla de importancias devuelta por
            ``permutation_ensemble``. Debe contener las columnas
            ``'feature'`` e ``'importance'``.
        X (pd.DataFrame): Features originales de las que se hará la
            selección.
        umbral (float, optional): Importancia mínima para que una feature
            sea seleccionada. Por defecto 0.001.

    Returns:
        tuple:
            - X (pd.DataFrame): DataFrame filtrado con solo las features
              seleccionadas.
            - num_cols (list[str]): Columnas numéricas y booleanas del
              DataFrame filtrado.
            - cat_cols (list[str]): Columnas categóricas del DataFrame
              filtrado.

    Raises:
        ValueError: Si ``source`` no contiene las columnas ``'feature'``
            e ``'importance'``.
        ValueError: Si ninguna feature supera el umbral indicado.

    Example:
        >>> X_sel, num_cols, cat_cols = select_feat(importance_df, X, umbral=0.005)
        >>> print(X_sel.shape)
        (1000, 12)
    """
    # Validación de columnas requeridas en source
    required_cols = {'feature', 'importance'}
    if not required_cols.issubset(source.columns):
        raise ValueError(
            f"'source' debe contener las columnas {required_cols}. "
            f"Encontradas: {set(source.columns)}"
        )

    selected = source.loc[source['importance'] > umbral, 'feature'].tolist()

    if not selected:
        raise ValueError(
            f"Ninguna feature supera el umbral de importancia ({umbral}). "
            "Considera reducir el valor de 'umbral'."
        )

    X = X[selected]
    num_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    logger.info("Features seleccionadas (%d): %s", len(selected), selected)
    logger.info("Numéricas: %s", num_cols)
    logger.info("Categóricas: %s", cat_cols)

    return X, num_cols, cat_cols


def drop_column_cv_pipeline_clean(models, X, y, cv_splits=5, groups=None, weights=None):
    """Calcula la importancia de features mediante drop-column CV matricial optimizado."""
    
    # 1. Configuración de splits explícita
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=cv_splits)
        splits = list(cv.split(X, y))

    # Diccionario para acumular el delta de RMSE por cada feature
    imp = {col: [] for col in X.columns}
    
    if len(X.columns) == 1:
        col = X.columns[0]
        logger.info("Solo queda 1 feature ('%s'), no se puede evaluar su eliminación.", col)
        return pd.DataFrame({"feature": [col], "importance": [0.0]})

    # 2. Bucle de modelos y folds (Preprocesamos UNA VEZ por fold)
    for model_idx, model in enumerate(models):
        logger.info("Calculando drop-importance para el modelo %d/%d", model_idx + 1, len(models))
        
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            w_train = weights.iloc[train_idx] if weights is not None else None
            w_val   = weights.iloc[val_idx] if weights is not None else None
            fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}

            # Extraer preprocesador y modelo XGBoost por separado
            prep = clone(model.named_steps["prep"])
            xgb_base = clone(model.named_steps["model"])

            # --- AHORRO MASIVO ---
            # Transformamos los datos completos del fold UNA SOLA VEZ
            X_train_t = prep.fit_transform(X_train, y_train)
            X_val_t   = prep.transform(X_val)
            
            # Mapear nombres originales a índices de la matriz numpy resultante
            feat_names_out = prep.get_feature_names_out()

            # Baseline del fold con todas las features
            xgb_base.fit(X_train_t, y_train, **fit_kwargs)
            preds_base = xgb_base.predict(X_val_t)
            rmse_base = root_mean_squared_error(y_val, preds_base, sample_weight=w_val)

            # 3. Bucle de ablación rápida (Operación NumPy)
            for orig_col in X.columns:
                # Encontrar qué columna de X_train_t corresponde a la original
                # Sklearn devuelve nombres como "num__variable" o "cat__variable"
                col_idx = [i for i, name in enumerate(feat_names_out) if name.endswith(f"__{orig_col}")]
                
                if not col_idx:
                    continue 
                    
                idx_to_drop = col_idx[0]

                # Eliminar la columna de la matriz numpy ya transformada
                X_train_dropped = np.delete(X_train_t, idx_to_drop, axis=1)
                X_val_dropped   = np.delete(X_val_t, idx_to_drop, axis=1)

                # Entrenar el XGBoost en la matriz reducida
                xgb_drop = clone(model.named_steps["model"])
                xgb_drop.fit(X_train_dropped, y_train, **fit_kwargs)
                preds_drop = xgb_drop.predict(X_val_dropped)
                rmse_drop = root_mean_squared_error(y_val, preds_drop, sample_weight=w_val)

                # Guardar el delta (cuánto empeoró respecto al baseline)
                imp[orig_col].append(rmse_drop - rmse_base)

    # 4. Agregación final
    # Promediamos los deltas a través de todos los folds y todos los modelos
    mean_imp = {col: float(np.mean(deltas)) for col, deltas in imp.items()}

    importances = (
        pd.Series(mean_imp)
        .sort_values(ascending=False)
        .to_frame(name='importance')
        .rename_axis('feature')
        .reset_index()
    )
    
    logger.info("Importancias drop-column calculadas con éxito.")
    return importances

def drop_column_cv_pipeline_clean_(models, X, y, cv_splits=5, groups=None, weights=None):
    """Calcula la importancia de features mediante drop-column CV.

    Para cada modelo de la lista y cada feature, elimina la columna del
    pipeline, re-entrena con CV y evalúa el incremento de RMSE respecto
    al baseline. Promedia los deltas sobre todos los modelos y devuelve
    las importancias ordenadas.

    Un valor positivo indica que eliminar esa feature empeora el modelo
    (feature relevante); un valor negativo o cercano a cero indica que
    la feature no aporta o perjudica.

    Args:
        models (list): Lista de pipelines entrenados. Acepta tanto el
            ensemble completo devuelto por ``run_pipe`` como una lista
            con un único ``final_model``.

            **Patrón recomendado** — modelo único, baseline consistente
            y coste computacional ×N_models menor::

                models, cv_results, final_model = run_pipe(X, y, ...)
                importance = drop_column_cv_pipeline_clean(final_model, X, y, ...)

            **Patrón alternativo** — ensemble, útil para estimar la
            varianza de importancias entre folds::

                importance = drop_column_cv_pipeline_clean(models, X, y, ...)

        X (pd.DataFrame): Features con la misma estructura usada en el
            entrenamiento.
        y (pd.Series): Variable objetivo.
        cv_splits (int, optional): Número de folds para el CV de evaluación.
            Por defecto 5.
        groups (pd.Series, optional): Identificador de grupo por muestra
            para ``GroupKFold``. Por defecto None.
        weights (pd.Series, optional): Pesos por muestra pasados al fit.
            Por defecto None.

    Returns:
        pd.DataFrame: Tabla con columnas ``'feature'`` e ``'importance'``,
            ordenada de mayor a menor importancia, donde ``importance``
            representa el incremento medio de RMSE al eliminar la feature.

    Notes:
        - El coste computacional es ``O(n_models × n_features × cv_splits)``;
          puede ser elevado con muchas features o modelos.
        - El baseline se recalcula para cada modelo del ensemble, lo que
          garantiza una comparación justa pero aumenta el tiempo de ejecución.
        - ``results_pd`` e ``importances`` se recalculan en cada iteración
          del bucle de modelos.

    Example:
        >>> importance_df = drop_column_cv_pipeline_clean(
        ...     models, X, y, groups=groups, weights=weights
        ... )
        >>> print(importance_df.head())
              feature  importance
        0   feature_a      0.031
        1   feature_b      0.018
        ...
    """
    set_config(enable_metadata_routing=True)

    if groups is not None:
        outer_cv = GroupKFold(n_splits=cv_splits)
        outer_cv.set_split_request(groups=True)
        params = {"groups": groups}
    else:
        outer_cv = KFold(n_splits=cv_splits)
        params = {}

    if weights is not None:
        params["sample_weight"] = weights

    cv_kwargs = dict(
        cv=outer_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        params=params
    )

    # ── Fase 1: baselines ──────────────────────────────────────────────
    # Una evaluación por modelo, calculada antes del bucle de features
    # para no repetirla n_features veces.
    baselines = []
    for model in models:
        b = -cross_val_score(model, X, y, **cv_kwargs).mean()
        baselines.append(b)
        logger.info("Baseline RMSE modelo %d: %.4f", len(baselines), b)
    logger.info("Baseline RMSE ensemble: %.4f", float(np.mean(baselines)))

    # ── Fase 2: drop-importance ────────────────────────────────────────
    # Para cada feature, promedia el delta RMSE sobre todos los modelos
    # reutilizando los baselines ya calculados.
    imp = {}
    if len(X.columns) == 1:
        col = X.columns[0]
        logger.info("Solo queda 1 feature ('%s'), no se puede evaluar su eliminación.", col)
        imp[col] = 0.0
    else:
        for col in X.columns:
            col_scores = []
            for model, baseline in zip(models, baselines):
                model_mod = disable_columns_in_pipeline(model, [col])
                score     = -cross_val_score(model_mod, X, y, **cv_kwargs).mean()
                col_scores.append(score - baseline)
            imp[col] = float(np.mean(col_scores))
            logger.debug("Dropped '%s': delta RMSE = %.4f", col, imp[col])

    importances = (
        pd.Series(imp)
        .sort_values(ascending=False)
        .to_frame(name='importance')
        .rename_axis('feature')
        .reset_index()
    )
    logger.info("Importancias drop-column (top 10):\n%s",
                importances.head(10).to_string(index=False))
    return importances
# %%
def itera_perm_mod(X, y, num_cols, cat_cols, groups, weights,
                   models=None, cv_results=None, cv_splits=5,
                   umbral=0.0, tolerance=0.015):
    """Selección iterativa de features mediante drop-column importance.
 
    En cada iteración calcula importancias drop-column, elimina en batch
    todas las features por debajo del umbral y re-entrena el pipeline. Se
    detiene cuando ninguna feature adicional puede descartarse, cuando la
    convergencia se alcanza, o cuando el RMSE empeora más de ``tolerance``
    respecto a la iteración anterior — en cuyo caso restaura el estado de
    la iteración previa.
 
    Args:
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variable objetivo.
        num_cols (list[str]): Columnas numéricas iniciales.
        cat_cols (list[str]): Columnas categóricas iniciales.
        groups (pd.Series): Identificador de grupo para ``GroupKFold``.
            ``None`` si no hay estructura de grupos.
        weights (pd.Series): Pesos por muestra para el entrenamiento.
            ``None`` si no hay pesos.
        models (list, optional): Pipelines ya entrenados. Si se proporciona
            junto con ``cv_results``, se omite el entrenamiento inicial.
            Acepta el ensemble completo (len > 1) o ``final_model`` (len == 1)
            devuelto por ``run_pipe``. Por defecto ``None``.
        cv_results (dict, optional): Resultados del nested CV de ``run_pipe``
            asociados a ``models``. Debe contener la clave ``'test_score'``
            con valores negativos (neg_RMSE). Por defecto ``None``.
        cv_splits (int, optional): Número de folds para el CV de
            drop-column. Por defecto 5.
        umbral (float, optional): Importancia mínima para conservar una
            feature. Features con importancia <= umbral se eliminan.
            Por defecto 0.0.
        tolerance (float, optional): Incremento máximo de RMSE permitido
            entre iteraciones consecutivas. Si el RMSE del nuevo modelo
            supera al anterior en más de ``tolerance``, se restaura el
            estado de la iteración anterior y se detiene el proceso.
            ``tolerance=0.0`` detiene ante cualquier empeoramiento.
            Por defecto 0.0.
 
    Returns:
        tuple:
            - importance (pd.DataFrame): Importancias drop-column de la
              última iteración válida, con columnas ``'feature'`` e
              ``'importance'``.
            - X (pd.DataFrame): Features seleccionadas tras el proceso.
            - num_cols (list[str]): Columnas numéricas del DataFrame filtrado.
            - cat_cols (list[str]): Columnas categóricas del DataFrame filtrado.
            - ensemble (list[Pipeline]): Los 10 pipelines del último
              ``run_pipe`` válido, útil para ``predict_ensemble`` y análisis
              de varianza entre folds.
            - cv_results (dict): Resultados del nested CV del último
              ``run_pipe`` válido, útil para ``get_metrics``.
            - output_models (list[Pipeline]): Modelo de producción o ensemble
              según la intención del usuario: ``final_model`` (len == 1) si
              se pasó ``final_model`` o ``None``; ensemble completo si se pasó
              el ensemble (len > 1).
 
    Notes:
        - La comparación de RMSE usa ``-cv_results["test_score"].mean()``
          del outer CV de cada ``run_pipe``. Ambos valores son comparables
          porque provienen del mismo proceso de evaluación nested CV.
        - ``tolerance=0.0`` es conservador: para ante cualquier empeoramiento.
          Valores como ``0.005`` permiten una degradación de hasta 0.5% en
          RMSE, útil cuando la varianza entre folds es alta.
        - La función respeta la elección del usuario sobre qué modelos usar
          para la selección de features: ensemble (len > 1) o
          ``final_model`` (len == 1).
        - Cuando el criterio de parada se activa por degradación de RMSE,
          el estado devuelto corresponde a la última iteración **válida**
          — no a la que degradó.
 
    Example:
        >>> importance, X_sel, num_cols, cat_cols, ensemble, cv_results, final_model = (
        ...     itera_perm_mod(X, y, num_cols, cat_cols, groups, weights,
        ...                    umbral=0.001, tolerance=0.005)
        ... )
        >>> rmse_df = eval_final_model(final_model, X_sel, y, groups=groups)
    """
    # Registrar la intención del usuario según lo que pasó como entrada:
    # - models=None o cv_results=None → sin preferencia; default a final_model
    # - len(models) > 1              → usuario eligió ensemble
    # - len(models) == 1             → usuario eligió final_model
    n_input_models = len(models) if models is not None else 0
 
    if models is None or cv_results is None:
        _models, cv_results, final_model = run_pipe(X, y, num_cols, cat_cols, groups, weights)
        _models_for_selection = final_model   # default: más eficiente
        logger.info("Métricas iniciales:\n%s", get_metrics(cv_results).to_string(index=False))
    else:
        if "test_score" not in cv_results:
            raise KeyError(
                "cv_results debe contener la clave 'test_score'. "
                "Usar el cv_results devuelto por run_pipe()."
            )
        _models = models
        _models_for_selection = models
        final_model = models   # fallback si el bucle no llama a run_pipe
 
    # RMSE de referencia — media del outer CV del modelo actual.
    # cv_results["test_score"] contiene neg_RMSE → invertir signo.
    best_rmse = float(-cv_results["test_score"].mean())
    logger.info("RMSE inicial: %.4f", best_rmse)
 
    importance = None
    iteracion  = 0
 
    while True:
        iteracion += 1
        logger.info("Iteración %d — features activas: %d", iteracion, len(X.columns))
 
        importance = drop_column_cv_pipeline_clean(
            models=_models_for_selection,
            X=X,
            y=y,
            cv_splits=cv_splits,
            groups=groups,
            weights=weights
        )
        logger.info("Importancias (top 10):\n%s",
                    importance.head(10).to_string(index=False))
 
        # Intentar eliminar features — preservar estado anterior en vars separadas
        try:
            X_new, num_cols_new, cat_cols_new = select_feat(importance, X, umbral=umbral)
        except ValueError:
            logger.info(
                "Iteración %d — ninguna feature supera umbral=%.4f. "
                "Proceso convergido con %d features.",
                iteracion, umbral, len(X.columns)
            )
            break
 
        # Convergencia: select_feat no eliminó ninguna feature
        delta = len(X_new.columns) - len(X.columns)
        logger.info("Cambio en el número de features: %+d", delta)
        if delta >= 0:
            break
 
        # Re-entrenar con el conjunto reducido y evaluar RMSE
        _models_new, cv_results_new, final_model_new = run_pipe(
            X_new, y, num_cols_new, cat_cols_new, groups, weights
        )
        new_rmse = float(-cv_results_new["test_score"].mean())
 
        logger.info(
            "RMSE iteración %d: %.4f (anterior: %.4f, delta: %+.4f, tolerance: %.4f)",
            iteracion, new_rmse, best_rmse, new_rmse - best_rmse, tolerance
        )
 
        # Criterio de parada por degradación: restaurar estado anterior y parar
        if new_rmse - best_rmse > tolerance:
            logger.warning(
                "Iteración %d — RMSE empeoró %.4f > tolerance %.4f. "
                "Restaurando estado anterior (%d features) y deteniendo.",
                iteracion, new_rmse - best_rmse, tolerance, len(X.columns)
            )
            break
 
        # Confirmación: actualizar todo el estado con la nueva iteración
        X           = X_new
        num_cols    = num_cols_new
        cat_cols    = cat_cols_new
        _models     = _models_new
        cv_results  = cv_results_new
        final_model = final_model_new
        best_rmse   = new_rmse
        _models_for_selection = _models if n_input_models > 1 else final_model
 
    # Posición 6: ensemble si el usuario pasó >1 modelos, final_model en otro caso
    # output_models = _models if n_input_models > 1 else final_model
 
    return importance, X, num_cols, cat_cols, _models, cv_results, final_model
#%%
def itera_perm_mod_(X, y, num_cols, cat_cols, groups, weights,
                   models=None, cv_results=None, cv_splits=5, umbral=0.0):
    """Selección iterativa de features mediante drop-column importance.

    Si no se proporcionan modelos previos, entrena el pipeline completo con
    ``run_pipe``. A continuación, en cada iteración: calcula la importancia
    drop-column, re-entrena el pipeline con las features actuales y elimina
    las menos relevantes según el umbral. El bucle se detiene cuando ninguna
    feature adicional es descartada o se agotan las iteraciones.

    Args:
        X (pd.DataFrame): Features de entrada.
        y (pd.Series): Variable objetivo.
        num_cols (list[str]): Columnas numéricas iniciales.
        cat_cols (list[str]): Columnas categóricas iniciales.
        groups (pd.Series): Identificador de grupo por muestra para
            ``GroupKFold``.
        weights (pd.Series): Pesos por muestra para el entrenamiento.
        models (list, optional): Lista de pipelines ya entrenados. Si se
            proporciona junto con ``cv_results``, se omite el entrenamiento
            inicial. Por defecto None.
        cv_results (dict, optional): Resultados del nested CV asociados a
            ``models``. Por defecto None.
        cv_splits (int, optional): Número de folds para el CV de
            drop-column. Por defecto 5.
        umbral (float, optional): Importancia mínima para conservar una
            feature en cada iteración. Por defecto 0.0.

    Returns:
        tuple:
            - importance (pd.DataFrame): Tabla de importancias de la última
              iteración, con columnas ``'feature'`` e ``'importance'``.
            - X (pd.DataFrame): Features seleccionadas tras el proceso
              iterativo.
            - num_cols (list[str]): Columnas numéricas del DataFrame
              filtrado.
            - cat_cols (list[str]): Columnas categóricas del DataFrame
              filtrado.
            - ensemble (list[Pipeline]): Los 10 pipelines del último
              ``run_pipe``, uno por fold outer. Útil para estimar la
              varianza de importancias entre folds y para
              ``predict_ensemble``.
            - cv_results (dict): Resultados del nested CV del último
              ``run_pipe``, útil para ``get_metrics``.
            - final_model (list[Pipeline]): Modelo de producción del último
              ``run_pipe``, entrenado sobre el 100% de los datos con
              las features seleccionadas. Listo para usar directamente
              en funciones de selección o en ``eval_final_model``.

    Notes:
        - En cada iteración se re-entrena el pipeline con ``run_pipe``,
          garantizando que los modelos estén siempre sincronizados con las
          features activas en ese momento.
        - El número máximo de iteraciones es el número inicial de features,
          aunque en la práctica el bucle converge mucho antes.
        - La función respeta la elección del usuario sobre qué modelos usar
          para la selección:

            - Si se pasan modelos externos con ``len > 1`` (ensemble),
              ``drop_column_cv_pipeline_clean`` usa el ensemble y el
              return devuelve el ensemble actualizado en posición 6.
            - Si se pasan modelos externos con ``len == 1`` (``final_model``)
              o no se pasan, la función usa ``final_model`` para la selección
              (más eficiente) y lo devuelve en posición 6.

          En ambos casos, el ensemble completo del último ``run_pipe``
          está disponible en posición 4.

    Example:
        >>> importance, X_sel, num_cols, cat_cols, ensemble, cv_results, final_model = (
        ...     itera_perm_mod(X, y, num_cols, cat_cols, groups, weights, umbral=0.001)
        ... )
        >>> rmse_df = eval_final_model(final_model, X_sel, y, groups=groups)
    """
    # Registrar la intención del usuario según lo que pasó como entrada:
    # - models=None o cv_results=None → sin preferencia; usamos final_model para selección
    # - len(models) > 1 → usuario pasó el ensemble; la selección y el return respetan eso
    # - len(models) == 1 → usuario pasó final_model; la selección y el return lo respetan
    n_input_models = len(models) if models is not None else 0

    if models is None or cv_results is None:
        _models, cv_results, final_model = run_pipe(X, y, num_cols, cat_cols, groups, weights)
        # run_pipe devuelve final_model como lista de un elemento.
        # Por defecto se usa final_model para la selección (más eficiente).
        # El ensemble (_models) se devuelve en posición 4 para análisis posterior.
        _models_for_selection = final_model
        logger.info("Métricas iniciales:\n%s", get_metrics(cv_results).to_string(index=False))
    else:
        # El usuario pasó sus propios modelos — respetar su elección.
        # Si pasó el ensemble (>1), la selección usa el ensemble.
        # Si pasó final_model (1), la selección usa final_model.
        _models = models
        _models_for_selection = models
        final_model = models   # fallback para el return si el bucle no llama a run_pipe

    n_features = len(X.columns)
    importance = None

    for i in range(n_features):
        logger.info("Iteración %d — features activas: %d", i + 1, len(X.columns))

        importance = drop_column_cv_pipeline_clean(
            models=_models_for_selection,   # ← respeta la elección del usuario
            X=X,
            y=y,
            cv_splits=cv_splits,
            groups=groups,
            weights=weights
        )

        logger.info("Importancias (top 10):\n%s",
                    importance.head(10).to_string(index=False))

        try:
            X, num_cols, cat_cols = select_feat(importance, X, umbral=umbral)
        except ValueError:
            logger.info(
                "Iteración %d — ninguna feature supera umbral=%.4f. "
                "Proceso convergido con %d features.",
                i + 1, umbral, len(X.columns)
            )
            break

        # Comprobar convergencia ANTES de re-entrenar:
        # si select_feat no eliminó ninguna feature, no tiene sentido
        # volver a llamar a run_pipe con el mismo conjunto.
        new_n_features = len(X.columns)
        delta = new_n_features - n_features
        logger.info("Cambio en el número de features: %+d", delta)

        if delta >= 0:
            break

        n_features = new_n_features

        # Re-entrena con las features actuales.
        # _models_for_selection se actualiza según la intención original:
        # si el usuario quería ensemble → actualizar _models (ensemble);
        # si quería final_model → actualizar final_model.
        _models, cv_results, final_model = run_pipe(X, y, num_cols, cat_cols, groups, weights)
        _models_for_selection = _models if n_input_models > 1 else final_model

    # Construir el return de posición 6 según la intención original del usuario:
    # ensemble si pasó >1 modelos, final_model en cualquier otro caso.
    output_models = _models if n_input_models > 1 else final_model

    return importance, X, num_cols, cat_cols, _models, cv_results, output_models
 
#%%
def sequential_drop_column_cv(
    models, X, y, groups, weights,
    cv_splits=5,
    threshold=0.0,
    verbose=True
):
    """Selección secuencial de features mediante drop-column CV matricial (Optimizado).

    Evalúa el impacto de eliminar cada feature secuencialmente usando matrices 
    Numpy cacheadas, evitando el overhead masivo de re-entrenar preprocesadores 
    en cada iteración de validación cruzada.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=cv_splits)
        splits = list(cv.split(X, y))

    remaining = list(X.columns)

    # =========================================================
    # 1. CACHÉ DE PREPROCESAMIENTO MATRICIAL
    # Entrenamos el StandardScaler y TargetEncoder una sola vez 
    # por modelo y por fold, y guardamos los arrays de Numpy.
    # =========================================================
    if verbose:
        logger.info("Precalculando transformaciones matriciales por fold...")

    cached_folds_per_model = []
    
    for model in models:
        prep_base = clone(model.named_steps["prep"])
        model_folds = []
        
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = weights.iloc[train_idx] if weights is not None else None
            w_val = weights.iloc[val_idx] if weights is not None else None

            # Entrenar preprocesador del fold UNA VEZ
            X_train_t = prep_base.fit_transform(X_train, y_train)
            X_val_t   = prep_base.transform(X_val)
            feat_names = prep_base.get_feature_names_out()

            # Mapeo rápido: qué índices de la matriz corresponden a qué columna original
            col_mapping = {c: [] for c in X.columns}
            for idx, name in enumerate(feat_names):
                orig_col = name.split("__")[-1] # Extrae el nombre quitando 'num__' o 'cat__'
                if orig_col in col_mapping:
                    col_mapping[orig_col].append(idx)

            model_folds.append({
                "X_tr_t": X_train_t,
                "X_vl_t": X_val_t,
                "y_tr": y_train,
                "y_vl": y_val,
                "w_tr": w_train,
                "w_vl": w_val,
                "col_mapping": col_mapping
            })
            
        cached_folds_per_model.append(model_folds)

    # =========================================================
    # 2. BUCLE SECUENCIAL DE ABLACIÓN 
    # Operamos exclusivamente sobre punteros a las matrices Numpy
    # =========================================================
    i = 0
    cached_baselines = {}
    cached_remaining_key = None

    while i < len(remaining):
        col = remaining[i]

        if verbose:
            logger.info("Evaluando: '%s' | Features activas: %d", col, len(remaining))

        # Recalcular el baseline solo si la lista 'remaining' cambió en la iteración anterior
        current_key = tuple(remaining)
        if current_key != cached_remaining_key:
            cached_remaining_key = current_key
            cached_baselines = {}

            for model_idx, model in enumerate(models):
                xgb_base = clone(model.named_steps["model"])
                fold_rmses = []

                for fold_data in cached_folds_per_model[model_idx]:
                    # Recopilar índices activos de la matriz
                    col_indices = []
                    for c in remaining:
                        col_indices.extend(fold_data["col_mapping"][c])

                    # Filtrar la matriz transformada
                    X_tr_f = fold_data["X_tr_t"][:, col_indices]
                    X_vl_f = fold_data["X_vl_t"][:, col_indices]
                    
                    fit_kwargs = {"sample_weight": fold_data["w_tr"]} if fold_data["w_tr"] is not None else {}
                    
                    xgb_clone = clone(xgb_base)
                    xgb_clone.fit(X_tr_f, fold_data["y_tr"], **fit_kwargs)
                    
                    preds = xgb_clone.predict(X_vl_f)
                    rmse = root_mean_squared_error(fold_data["y_vl"], preds, sample_weight=fold_data["w_vl"])
                    fold_rmses.append(rmse)

                cached_baselines[model_idx] = np.mean(fold_rmses)

        # Evaluar impacto de eliminar la variable actual ('col')
        scores = []
        for model_idx, model in enumerate(models):
            baseline = cached_baselines[model_idx]
            xgb_base = clone(model.named_steps["model"])
            fold_rmses = []
            
            # Simulamos que 'col' ya no está
            temp_remaining = [c for c in remaining if c != col]

            for fold_data in cached_folds_per_model[model_idx]:
                col_indices = []
                for c in temp_remaining:
                    col_indices.extend(fold_data["col_mapping"][c])

                X_tr_f = fold_data["X_tr_t"][:, col_indices]
                X_vl_f = fold_data["X_vl_t"][:, col_indices]
                
                fit_kwargs = {"sample_weight": fold_data["w_tr"]} if fold_data["w_tr"] is not None else {}
                
                xgb_clone = clone(xgb_base)
                xgb_clone.fit(X_tr_f, fold_data["y_tr"], **fit_kwargs)
                
                preds = xgb_clone.predict(X_vl_f)
                rmse = root_mean_squared_error(fold_data["y_vl"], preds, sample_weight=fold_data["w_vl"])
                fold_rmses.append(rmse)

            # Delta RMSE (Negativo o ~0 significa que eliminarla mejoró/no afectó al modelo)
            scores.append(np.mean(fold_rmses) - baseline)

        impact = np.mean(scores)

        if verbose:
            logger.info("Impacto de '%s': %.4f", col, impact)

        if impact < threshold:
            if verbose:
                logger.info("Eliminando '%s'", col)
            remaining.remove(col)
            # No incrementamos 'i' porque la siguiente variable ocupa ahora este índice
        else:
            i += 1

        if len(remaining) <= 1:
            logger.warning("Solo queda %d feature, deteniendo el bucle.", len(remaining))
            break

    # =========================================================
    # 3. RECONSTRUCCIÓN FINAL
    # =========================================================
    X_sel = X[remaining]
    num_cols = X_sel.select_dtypes(include=['number', 'bool']).columns.tolist()
    cat_cols = X_sel.select_dtypes(include='object').columns.tolist()

    # Re-entrena el pipeline completo y oficial con las features seleccionadas
    # Usamos la función run_pipe para devolver los objetos tal como los espera la API
    models_out, cv_results, final_model = run_pipe(X_sel, y, num_cols, cat_cols, groups, weights)
    metrics = get_metrics(cv_results)

    return X_sel, num_cols, cat_cols, models_out, metrics, final_model

#%%
def sequential_drop_column_cv_(
    models, X, y, groups, weights,
    cv_splits=5,
    threshold=0.0,
    verbose=True
):
    """Selección secuencial de features mediante drop-column CV con decisión dinámica.

    Recorre las features una a una y evalúa el impacto de eliminar cada una
    sobre el RMSE del modelo o modelos recibidos. Si el impacto es negativo
    (eliminar la feature mejora o no empeora el modelo), la descarta
    dinámicamente y continúa con las features restantes. Al finalizar,
    re-entrena el pipeline con las features seleccionadas.

    A diferencia de ``drop_column_cv_pipeline_clean``, que evalúa todas las
    features en paralelo respecto a un baseline fijo, este enfoque es
    secuencial: cada eliminación afecta al baseline de las evaluaciones
    siguientes.

    Args:
        models (list): Lista de pipelines entrenados. Acepta tanto el
            ensemble completo como una lista con un único ``final_model``.

            **Patrón recomendado**::

                models, cv_results, final_model = run_pipe(X, y, ...)
                X_, num_, cat_, models_, metrics_ = sequential_drop_column_cv(
                    final_model, X, y, groups, weights
                )

        X (pd.DataFrame): Features con la misma estructura usada en el
            entrenamiento.
        y (pd.Series): Variable objetivo.
        groups (pd.Series): Identificador de grupo por muestra para
            ``GroupKFold``.
        weights (pd.Series): Pesos por muestra pasados al fit.
        cv_splits (int, optional): Número de folds para el CV de evaluación.
            Por defecto 5.
        threshold (float, optional): Umbral de impacto por debajo del cual
            una feature se elimina. Un valor de 0.0 elimina features que no
            empeoran el RMSE; valores positivos son más agresivos. Por
            defecto 0.0.
        verbose (bool, optional): Si True, delega el logging en nivel INFO.
            Por defecto True.

    Returns:
        tuple:
            - X (pd.DataFrame): Features seleccionadas tras el proceso.
            - num_cols (list[str]): Columnas numéricas del DataFrame
              filtrado.
            - cat_cols (list[str]): Columnas categóricas del DataFrame
              filtrado.
            - models (list[Pipeline]): Ensemble re-entrenado con las
              features seleccionadas.
            - metrics (pd.DataFrame): Métricas del pipeline re-entrenado,
              devueltas por ``get_metrics``.
            - final_model (list[Pipeline]): Modelo de producción entrenado
              sobre el 100% de los datos con las features seleccionadas.

    Notes:
        - El orden de evaluación de las features influye en el resultado
          final, ya que cada eliminación modifica el baseline de las
          siguientes. Considerar ordenar las features por importancia
          previa antes de llamar a esta función.
        - El baseline se recalcula para cada feature y cada modelo, lo que
          hace que el coste sea ``O(n_features × n_models × cv_splits)``.

    Example:
        >>> X_sel, num_cols, cat_cols, models, metrics, final_model = (
        ...     sequential_drop_column_cv(
        ...         final_model, X, y, groups, weights, threshold=0.005
        ...     )
        ... )
        >>> print(f"Features seleccionadas: {X_sel.columns.tolist()}")
    """
    set_config(enable_metadata_routing=True)

    if groups is not None:
        outer_cv = GroupKFold(n_splits=cv_splits)
        outer_cv.set_split_request(groups=True)
        params = {"groups": groups}
    else:
        outer_cv = KFold(n_splits=cv_splits)
        params = {}

    # Incluir sample_weight solo si se proporcionó; pasarlo como None
    # con metadata routing activo puede provocar errores en algunos estimadores
    if weights is not None:
        params["sample_weight"] = weights

    cv_kwargs = dict(
        cv=outer_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        params=params,
    )

    remaining = list(X.columns)
    i = 0

    # Cache de baselines por modelo: se invalida solo cuando `remaining` cambia.
    # Sin caché, el baseline se recalcularía n_features veces con los mismos datos
    # mientras no se elimine ninguna feature — coste O(n_features × n_models × cv_splits)
    # en lugar de O((1 + n_features) × n_models × cv_splits).
    # La clave es el id de `remaining` como tupla para detectar cambios.
    cached_remaining_key = None
    cached_baselines     = {}   # {model_id: baseline_rmse}

    while i < len(remaining):
        col = remaining[i]

        if verbose:
            logger.info("Evaluando: '%s' | Features activas: %d", col, len(remaining))
            logger.debug("Features activas: %s", remaining)

        # Recalcula baselines solo si `remaining` ha cambiado desde la última vez
        current_key = tuple(remaining)
        if current_key != cached_remaining_key:
            cached_remaining_key = current_key
            cached_baselines = {}
            for model in models:
                model_current = disable_columns_in_pipeline(
                    model,
                    [c for c in X.columns if c not in remaining]
                )
                cached_baselines[id(model)] = (
                    model_current,
                    -cross_val_score(model_current, X, y, **cv_kwargs).mean()
                )
            logger.debug(
                "Baseline recalculado para %d modelos (remaining cambió).",
                len(models)
            )

        scores = []
        for model in models:
            model_current, baseline = cached_baselines[id(model)]
            # Evalúa el impacto de eliminar la feature actual
            model_mod = disable_columns_in_pipeline(model_current, [col])
            score = -cross_val_score(model_mod, X, y, **cv_kwargs).mean()
            scores.append(score - baseline)

        impact = np.mean(scores)

        if verbose:
            logger.info("Impacto de '%s': %.4f", col, impact)

        if impact < threshold:
            # Eliminar la feature mejora o no penaliza el modelo
            if verbose:
                logger.info("Eliminando '%s'", col)
            remaining.remove(col)
            # No se incrementa i: la siguiente feature ocupa la posición actual
        else:
            i += 1

        if len(remaining) <= 1:
            logger.warning("Solo queda %d feature, deteniendo el bucle.", len(remaining))
            break

    X = X[remaining]

    num_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    # Re-entrena el pipeline con las features seleccionadas
    models, cv_results, final_model = run_pipe(X, y, num_cols, cat_cols, groups, weights)
    metrics = get_metrics(cv_results)

    return X, num_cols, cat_cols, models, metrics, final_model

#%%
def disable_columns_in_pipeline(pipeline, columns_to_disable, prep_step="prep"):
    """Devuelve un clon del pipeline con las columnas indicadas eliminadas del ColumnTransformer.

    Versión canónica del helper de desactivación de columnas. Soporta
    transformers con atributo ``.cols`` (como ``TargetEncoder``), siempre
    clona todos los transformers para evitar mutación compartida entre folds,
    y maneja correctamente columnas string como ``"passthrough"`` o ``"drop"``.

    Args:
        pipeline (Pipeline): sklearn Pipeline con un paso ``ColumnTransformer``
            identificado por ``prep_step``.
        columns_to_disable (list[str]): Lista de columnas a excluir.
        prep_step (str, optional): Nombre del paso ``ColumnTransformer`` en el
            pipeline. Por defecto ``"prep"``.

    Returns:
        Pipeline: Clon del pipeline con las columnas desactivadas. El pipeline
            original no se modifica.

    Notes:
        - Si el ``ColumnTransformer`` tiene ``remainder`` con un transformer
          custom, las columnas que caen en él NO serán filtradas. Se emite un
          ``UserWarning``.
        - Si un transformer queda sin columnas tras el filtro, se omite entero.
        - Todos los transformers se clonan para evitar mutación compartida
          entre folds o llamadas concurrentes.

    Example:
        >>> pipe_mod = disable_columns_in_pipeline(pipeline, ['feature_a'])
    """
    columns_to_disable = set(columns_to_disable)
    pipe = clone(pipeline)
    ct = pipe.named_steps[prep_step]

    new_transformers = []

    for name, trans, cols in ct.transformers:

        # "passthrough" y "drop" son strings, no listas de columnas
        if isinstance(cols, str):
            new_transformers.append((name, trans, cols))
            continue

        cols_new = [c for c in cols if c not in columns_to_disable]

        # Transformer sin columnas tras el filtro → se omite
        if not cols_new:
            continue

        # Siempre clonar el transformer para evitar mutación compartida
        trans = clone(trans)

        # Soporte para transformers con atributo .cols (e.g. category_encoders)
        if hasattr(trans, "cols"):
            trans.cols = cols_new

        new_transformers.append((name, trans, cols_new))

    ct.transformers = new_transformers

    # Advertir si remainder es un transformer custom no filtrable
    if ct.remainder not in ("drop", "passthrough"):
        warnings.warn(
            "ColumnTransformer tiene 'remainder' con transformer custom. "
            "Las columnas en 'remainder' no serán filtradas por "
            "disable_columns_in_pipeline.",
            UserWarning,
            stacklevel=2,
        )

    return pipe

# ===========================================================================
# LOGGING HELPER
# ===========================================================================

def _setup_logger(log_path=None):
    """
    Crea y configura un logger dedicado para una ejecución de la selección.

    El logger recibe un nombre único basado en su id de objeto para evitar
    colisiones si la función se llama varias veces en la misma sesión o en
    entornos con múltiples ejecuciones simultáneas.

    Siempre añade un StreamHandler (consola). Si log_path no es None, añade
    además un FileHandler en modo append, de modo que reanudar una ejecución
    continúa el log existente en lugar de sobreescribirlo.

    Formato de cada línea:
        2024-01-15 10:23:41 | [Step   3] Features: 12 | Baseline RMSE: 0.1423 | ...

    Returns:
        logger: el objeto logger configurado.
    """
    logger = logging.getLogger(f"xgb_sculpt.selection_{id(object())}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # no contaminar el root logger del entorno

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger

def _teardown_logger(logger):
    """
    Cierra y elimina todos los handlers del logger para liberar recursos
    y evitar acumulación de handlers si la función se llama varias veces.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

# ===========================================================================
# CHECKPOINT HELPERS
# ===========================================================================

def _to_serializable(obj):
    """
    Convierte recursivamente tipos numpy a tipos nativos de Python para que
    json.dumps pueda serializarlos sin errores.
    """
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_checkpoint(path, step, remaining, removed_features, history, splits):
    """
    Persiste el estado completo de la selección en un fichero JSON.

    El checkpoint contiene:
      - step:             índice del próximo step a ejecutar.
      - remaining:        features activas en ese momento.
      - removed_features: lista de dicts con metadatos de cada eliminación.
      - history:          lista de dicts con el estado de cada step completado.
      - splits:           índices de los folds fijos, serializado para garantizar
                          que el resume usa exactamente los mismos folds que la
                          ejecución original.

    Usa escritura atómica (write a .tmp → rename) para que una interrupción
    durante la escritura no deje el fichero en estado corrupto.
    """
    state = {
        "step": int(step),
        "remaining": list(remaining),
        "removed_features": _to_serializable(removed_features),
        "history": _to_serializable(history),
        "splits": [
            [train_idx.tolist(), val_idx.tolist()]
            for train_idx, val_idx in splits
        ],
    }
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)

def _load_checkpoint(path):
    """
    Carga el estado desde un checkpoint JSON y reconstruye los splits como
    arrays de numpy (formato esperado por el bucle principal).

    Returns:
        step, remaining, removed_features, history, splits
    """
    state = json.loads(pathlib.Path(path).read_text())
    splits = [
        (np.array(train), np.array(val))
        for train, val in state["splits"]
    ]
    return (
        state["step"],
        state["remaining"],
        state["removed_features"],
        state["history"],
        splits,
    )

def make_checkpoint_callback(path):
    """
    Devuelve un callable compatible con el parámetro `on_step_end` que
    guarda el estado de la selección en un fichero JSON al término de
    cada step.

    Es el equivalente funcional de `checkpoint_path`, expresado como
    callback. Ambos mecanismos pueden coexistir en la misma ejecución.

    Args:
        path: ruta del fichero JSON de checkpoint (str o Path).

    Returns:
        callback con firma:
            fn(step, history, remaining, removed_features, splits, **kwargs)

    Ejemplo:
        from xgb_sculpt import make_checkpoint_callback
        cb = make_checkpoint_callback("selection_state.json")
        history, selected = stable_backward_selection_with_oof_permutation(
            ..., on_step_end=cb
        )
    """
    def _callback(step, history, remaining, removed_features, splits, **kwargs):
        _save_checkpoint(path, step, remaining, removed_features, history, splits)

    return _callback

# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def stable_backward_selection_with_oof_permutation(
    models,
    X,
    y,
    cv_splits=3,
    groups=None,
    weights=None,
    threshold=0.0,
    n_repeats=3,
    random_state=RANDOM_STATE,
    reintroduction_every=5,
    max_reintroduce=2,
    model_selection="best", # 'best' | 'all'
    checkpoint_path=None,
    log_path=None,
    on_step_end=None,
):
    """
    Backward feature selection estable con OOF permutation importance.

    En cada step:
      1. Entrena todos los modelos (o el mejor si model_selection="best") en
         los folds fijos y calcula el baseline RMSE.
      2. Calcula permutation importance sobre validación (para análisis posterior).
      3. Calcula drop importance (ablación): coste de eliminar cada feature.
      4. Cada `reintroduction_every` steps, reevalúa features eliminadas
         previamente y las reintroduce si mejoran el baseline.
      5. Elimina la feature con menor contribución marginal.
      6. Para cuando ninguna feature candidata a ser eliminada mejora el modelo
         más allá de `threshold`.

    Args:
        models:                Lista de pipelines sklearn (con paso "prep"
                               ColumnTransformer). Acepta tanto el ensemble
                               completo como una lista con un único
                               ``final_model``.

                               **Patrón recomendado** — modelo único::

                                   models, cv_results, final_model = run_pipe(...)
                                   history, remaining, ... = \
                                       stable_backward_selection_with_oof_permutation(
                                           final_model, X, y, ...
                                       )

                               **Patrón alternativo** — ensemble completo,
                               útil para estimar varianza entre folds::

                                   history, remaining, ... = \
                                       stable_backward_selection_with_oof_permutation(
                                           models, X, y, ...
                                       )

        X:                     DataFrame de features.
        y:                     Series de target.
        cv_splits:             Número de splits utilizados en cv.
        groups:                Array de grupos para CV estratificado por grupo
                               (opcional).
        weights:               Series de sample weights (opcional).
        threshold:             Umbral de drop importance. Si el mejor candidato
                               a eliminar tiene drop_importance > threshold,
                               se para.
        n_repeats:             Repeticiones de permutation importance.
        random_state:          Semilla para permutation importance.
        reintroduction_every:  Cada cuántos steps se reevalúan features
                               eliminadas.
        max_reintroduce:       Máximo de features eliminadas (las más recientes
                               sin reintroducir) que se reevalúan en cada ronda.
        model_selection:       "all" → se usan todos los modelos en cada step.
                               "best" → tras el paso 1 del step 0 se identifica
                               el modelo con menor RMSE medio y se usa solo ese
                               para el resto de la selección.
                               Si se pasa ``final_model`` (lista de un elemento
                               devuelta por ``run_pipe``) este parámetro
                               no tiene efecto — solo hay un modelo disponible.
        checkpoint_path:       Ruta del fichero JSON de checkpoint (str o Path).
                               - None → no se guarda ni se carga nada.
                               - Si el fichero existe al llamar a la función,
                                 la ejecución se reanuda desde el último step
                                 guardado, usando los mismos folds fijos que
                                 la ejecución original (los splits se leen del
                                 checkpoint, no se regeneran).
                               - Al término de cada step, y al parar, el estado
                                 completo se sobreescribe en el fichero.
                               Ejemplo: checkpoint_path="selection_state.json"
        log_path:              Ruta del fichero de log (str o Path).
                               - None → solo salida por consola.
                               - Si el fichero existe, el log se añade al final
                                 (modo append), de modo que un resume continúa
                                 el log existente sin sobreescribirlo.
                               Ejemplo: log_path="selection.log"
        on_step_end:           Callable opcional invocado al término de cada
                               step (incluyendo el step final de parada).
                               Firma esperada:
                                 fn(step, history, remaining, removed_features,
                                    splits, **kwargs)
                               Se le pasan copias de remaining y removed_features
                               para que no pueda mutar el estado interno.
                               Se invoca antes del checkpoint interno
                               (checkpoint_path), de modo que si el callback
                               falla el checkpoint todavía se guarda.
                               Ejemplo: on_step_end=make_checkpoint_callback(p)

    Returns:
        history:     Lista de dicts con el estado de cada step:
                       - step
                       - remaining
                       - baseline_score          escalar (media de modelos y folds)
                       - baseline_scores_per_fold dict {model_i: [rmse_fold_0, ...]}
                       - drop_importance
                       - permutation_importance
                       - removed       str o None si se paró sin eliminar
                       - reintroduced  lista de features reintroducidas ese step
        remaining:   Lista de features seleccionadas al finalizar.
        X_:          DataFrame filtrado con las features seleccionadas.
        num_cols_:   Columnas numéricas de X_.
        cat_cols_:   Columnas categóricas de X_.
        models_:     Ensemble re-entrenado con las features seleccionadas.
        metrics_:    Métricas del ensemble final (ver ``get_metrics``).
        final_model_: Modelo de producción entrenado sobre el 100% de los
                      datos con las features seleccionadas (ver ``run_pipe``).

    Notes:
        - Permutation importance se guarda en history para análisis posterior
          pero NO influye en la decisión de eliminación. La decisión se basa
          en drop_importance (ablación), que captura mejor la contribución
          marginal en presencia de multicolinealidad.
        - removed_features mantiene metadatos completos de cada eliminación.
          Estructura de cada entrada:
            {
              "feature":                    str,
              "step":                       int,
              "drop_importance_at_removal": float,
              "n_features_at_removal":      int,
              "reintroduced_at_step":       int,  # solo si fue reintroducida
            }
          Si una feature se elimina, se reintroduce y se vuelve a eliminar,
          aparece dos veces con entradas independientes.
        - El checkpoint usa escritura atómica (write→rename) para que una
          interrupción durante la escritura no deje el fichero corrupto.
        - El log se escribe en modo append: reanudar una ejecución continúa
          el fichero de log existente sin sobreescribirlo.
    """
    
    if model_selection not in ("best", "all"):
        raise ValueError(f"model_selection debe ser 'best' o 'all',\
                         recibido: {model_selection!r}")

    # Inicialización defensiva: si _setup_logger lanza una excepción, el bloque
    # finally necesita que `logger` esté definido para llamar a _teardown_logger.
    logger = None
    try:
        logger = _setup_logger(log_path)
        # =========================
        # INICIO / RESUME
        # Si existe checkpoint, restaurar estado y continuar desde el step
        # guardado. Los splits se leen del checkpoint para garantizar que el
        # resume usa exactamente los mismos folds que la ejecución original.
        # Si no existe (o checkpoint_path es None), inicializar desde cero.
        # =========================
        if checkpoint_path is not None and pathlib.Path(checkpoint_path).exists():
            # metadata routing debe activarse también en el path de resume,
            # antes de cualquier operación que use el pipeline con metadata
            set_config(enable_metadata_routing=True)
            step, remaining, removed_features, history, splits = _load_checkpoint(
                checkpoint_path
            )
            logger.info(
                f"[Checkpoint] Resuming from step {step} — "
                f"{len(remaining)} features remaining: {remaining}"
            )
        else:
            set_config(enable_metadata_routing=True)

            remaining = list(X.columns)
            removed_features = []
            history = []

            # SPLITS FIJOS
            # Todos los experimentos usan los mismos folds para que las
            # comparaciones de RMSE entre configuraciones de features sean
            # estadísticamente válidas.
            if groups is not None:
                outer_cv = GroupKFold(n_splits=cv_splits)
                outer_cv.set_split_request(groups=True)
                splits = list(outer_cv.split(X, y, groups))
            else:
                splits = list(KFold(n_splits=cv_splits).split(X, y))

            step = 0

            if checkpoint_path is not None:
                logger.info(
                    f"[Checkpoint] New run — state will be saved to "
                    f"'{checkpoint_path}' after each step."
                )

        # Flag para model_selection="best": debe aplicarse exactamente una vez,
        # en el primer step que se ejecute — sea step 0 en una ejecución nueva
        # o step N en una reanudación desde checkpoint.
        model_selection_applied = False

        # =========================
        # BUCLE PRINCIPAL
        # =========================
        while len(remaining) > 1:

            # =========================
            # 1. CACHE DE MODELOS Y BASELINE
            # Se entrenan todos los modelos con el feature set actual.
            # Los modelos entrenados se guardan para reutilizarlos en el
            # paso 2 (permutation importance), evitando reentrenar.
            # models_cache guarda también la lista completa de fold_scores
            # (no solo el promedio) para exponerlos en history.
            # =========================
            models_cache = []
            baseline_scores = []

            for model in models:

                model_current = disable_columns_in_pipeline(
                    model,
                    [c for c in X.columns if c not in remaining],
                )

                fold_models = []
                fold_scores = []

                for train_idx, val_idx in splits:

                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    w_train = weights.iloc[train_idx] if weights is not None else None
                    w_val = weights.iloc[val_idx] if weights is not None else None
                    
                    m = clone(model_current)
                    fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}
                    m.fit(X_train, y_train, **fit_kwargs)

                    y_pred = m.predict(X_val)
                    rmse = root_mean_squared_error(y_val, y_pred, sample_weight=w_val)

                    fold_models.append((m, val_idx))
                    fold_scores.append(rmse)

                models_cache.append({
                    "models":      fold_models,
                    "baseline":    np.mean(fold_scores),
                    "fold_scores": fold_scores,   # lista completa, un valor por fold
                })
                baseline_scores.append(np.mean(fold_scores))

            baseline = np.mean(baseline_scores)

            # =========================
            # MODEL SELECTION
            # Solo se ejecuta en el primer step de la ejecución actual
            # (sea step 0 en una ejecución nueva o step N tras un resume),
            # y solo si model_selection="best". Se identifica el modelo con
            # menor RMSE medio entre folds y se reduce `models` y
            # `models_cache` a ese único elemento. A partir de este punto
            # el bucle continúa sin cambios: los pasos 2, 3 y 4 simplemente
            # iteran sobre una lista de un elemento. El baseline se recalcula
            # con el único modelo restante para coherencia con los scores
            # que se usarán en el resto del bucle.
            # =========================
            if not model_selection_applied and model_selection == "best":
                model_selection_applied = True
                best_idx = int(np.argmin(baseline_scores))
                logger.info(
                    f"[Model selection] Best model: index {best_idx} "
                    f"(RMSE {baseline_scores[best_idx]:.4f} vs mean "
                    f"{baseline:.4f}). Using only this model from next step onwards."
                )
                models = [models[best_idx]]
                models_cache = [models_cache[best_idx]]
                baseline = baseline_scores[best_idx]
            elif model_selection == "best":
                model_selection_applied = True  # ya aplicado, no hacer nada

            # =========================
            # 2. PERMUTATION IMPORTANCE (sobre OOF, reutilizando cache)
            # Mide cuánto empeora el RMSE al permutar aleatoriamente cada
            # feature. Se guarda en history para análisis posterior; NO se
            # usa para decidir qué eliminar.
            # Nota: subestima importancia de features correlacionadas (si una
            # se permuta, otra correlacionada compensa). Usar con cautela.
            #
            # Se pasan solo las columnas activas (remaining) a
            # permutation_importance. Pasar X_val completo forzaría a permutar
            # también columnas inactivas que el modelo ignora internamente,
            # multiplicando el coste por (n_total / n_remaining) sin aporte.
            # =========================
            perm_values = {col: [] for col in remaining}

            for cache in models_cache:
                for m, val_idx in cache["models"]:

                    X_val = X.iloc[val_idx][remaining]   # solo columnas activas
                    y_val = y.iloc[val_idx]

                    perm = permutation_importance(
                        m,
                        X_val,
                        y_val,
                        n_repeats=n_repeats,
                        random_state=random_state,
                        scoring="neg_root_mean_squared_error",
                    )

                    for i, col in enumerate(remaining):
                        # neg_RMSE → invertir signo: positivo = feature útil
                        perm_values[col].append(-perm.importances_mean[i])

            perm_importance = {c: np.mean(v) for c, v in perm_values.items()}

            # =========================
            # 3. DROP IMPORTANCE (ablación)
            # Para cada feature, reentrena sin ella y mide delta RMSE vs
            # baseline.
            #   > 0: eliminarla empeora el modelo  → feature útil
            #   ~ 0: eliminarla es neutral          → feature redundante
            #   < 0: eliminarla mejora el modelo    → feature perjudicial
            # Se elimina la feature con el valor más bajo.
            # =========================
            drop_importance = {}

            for col_to_drop in remaining:

                temp_remaining = [c for c in remaining if c != col_to_drop]
                drop_scores = []

                for model in models:

                    model_mod = disable_columns_in_pipeline(
                        model,
                        [c for c in X.columns if c not in temp_remaining],
                    )

                    fold_scores = []

                    for train_idx, val_idx in splits:

                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        w_train = weights.iloc[train_idx] if weights is not None else None
                        w_val = weights.iloc[val_idx] if weights is not None else None

                        m = clone(model_mod)
                        fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}
                        m.fit(X_train, y_train, **fit_kwargs)

                        y_pred = m.predict(X_val)
                        rmse = root_mean_squared_error(y_val, y_pred, sample_weight=w_val)
                        fold_scores.append(rmse)

                    drop_scores.append(np.mean(fold_scores))

                drop_importance[col_to_drop] = np.mean(drop_scores) - baseline

            # =========================
            # 4. REINTRODUCCIÓN
            # Cada `reintroduction_every` steps se reevalúan features
            # eliminadas previamente. Solo se consideran las que aún no han
            # sido reintroducidas (sin clave "reintroduced_at_step").
            # Si una feature mejora el baseline al añadirla, se reintroduce.
            # Su entrada en removed_features se marca con
            # "reintroduced_at_step" preservando el historial completo.
            # =========================
            reintroduced = []

            if step > 0 and step % reintroduction_every == 0 and removed_features:

                # Candidatas: eliminadas que aún no han sido reintroducidas
                active_removed = [
                    r for r in removed_features
                    if "reintroduced_at_step" not in r
                ]
                candidates = [r["feature"] for r in active_removed[-max_reintroduce:]]

                for col in candidates:

                    temp_remaining = remaining + [col]
                    re_scores = []

                    for model in models:

                        model_mod = disable_columns_in_pipeline(
                            model,
                            [c for c in X.columns if c not in temp_remaining],
                        )

                        fold_scores = []

                        for train_idx, val_idx in splits:

                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                            w_train = weights.iloc[train_idx] if weights is not None else None
                            w_val = weights.iloc[val_idx] if weights is not None else None

                            m = clone(model_mod)
                            fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}
                            m.fit(X_train, y_train, **fit_kwargs)

                            y_pred = m.predict(X_val)
                            rmse = root_mean_squared_error(y_val, y_pred, sample_weight=w_val)
                            fold_scores.append(rmse)

                        re_scores.append(np.mean(fold_scores))

                    re_score = np.mean(re_scores)

                    if re_score < baseline:
                        remaining.append(col)
                        reintroduced.append(col)

                # Marcar reintroducidas en sus metadatos en lugar de eliminar.
                # Si más adelante se vuelven a eliminar, nueva entrada independiente.
                for r in removed_features:
                    if r["feature"] in reintroduced and "reintroduced_at_step" not in r:
                        r["reintroduced_at_step"] = step

            # =========================
            # 5+6. CANDIDATO A ELIMINAR
            # Las features recién reintroducidas no pueden eliminarse en el
            # mismo step en que se reintrodujeron.
            # =========================
            droppable = {
                c: v for c, v in drop_importance.items()
                if c not in reintroduced
            }

            # Guard: si todas las features activas se acaban de reintroducir
            # no hay candidata a eliminar en este step; avanzar al siguiente.
            if not droppable:
                logger.warning(
                    "[Step %d] Todas las features activas fueron reintroducidas "
                    "en este step — no hay candidata a eliminar. Continuando.",
                    step,
                )
                step += 1
                continue

            worst_col = min(droppable, key=droppable.get)

            # =========================
            # LOG DE PROGRESO
            # =========================
            reintroduced_str = ", ".join(reintroduced) if reintroduced else "—"
            logger.info(
                f"[Step {step:>3}] "
                f"Features: {len(remaining):>3} | "
                f"Baseline RMSE: {baseline:.4f} | "
                f"Drop: '{worst_col}' ({drop_importance[worst_col]:+.4f}) | "
                f"Reintroduced: {reintroduced_str}"
            )

            # =========================
            # CONSTRUCCIÓN DEL ENTRY DE HISTORY
            # baseline_scores_per_fold expone la distribución completa de RMSE
            # por fold de cada modelo activo, permitiendo análisis de varianza,
            # intervalos de confianza y tests estadísticos entre steps.
            # El campo baseline_score se mantiene como escalar para no romper
            # código existente que consuma el history.
            # =========================
            entry = {
                "step":                     step,
                "remaining":                remaining.copy(),
                "baseline_score":           baseline,
                "baseline_scores_per_fold": {
                    f"model_{i}": cache["fold_scores"]
                    for i, cache in enumerate(models_cache)
                },
                "drop_importance":          drop_importance.copy(),
                "permutation_importance":   perm_importance.copy(),
                "reintroduced":             reintroduced,
            }

            # =========================
            # STOP ANTES DE GUARDAR EN HISTORY
            # Si ninguna feature mejora el modelo más allá del threshold,
            # se para. Se registra removed=None: la feature identificada como
            # worst_col NO fue eliminada realmente.
            # =========================
            if drop_importance[worst_col] > threshold:
                entry["removed"] = None
                history.append(entry)
                logger.info(
                    f"[Step {step:>3}] Stopping — "
                    f"best drop ({drop_importance[worst_col]:+.4f}) "
                    f"exceeds threshold ({threshold}). "
                    f"Final features: {len(remaining)}"
                )
                _fire_step_end(
                    on_step_end, step + 1,
                    history, remaining, removed_features, splits, logger,
                )
                if checkpoint_path is not None:
                    _save_checkpoint(
                        checkpoint_path, step + 1,
                        remaining, removed_features, history, splits,
                    )
                    logger.info(f"[Checkpoint] Final state saved to '{checkpoint_path}'")
                break

            # =========================
            # GUARDAR HISTORY Y ELIMINAR
            # Los metadatos de eliminación se añaden a removed_features ANTES
            # de modificar `remaining`, de modo que n_features_at_removal
            # refleja el tamaño del feature set en el momento de la decisión.
            # =========================
            entry["removed"] = worst_col
            history.append(entry)

            removed_features.append({
                "feature":                    worst_col,
                "step":                       step,
                "drop_importance_at_removal": drop_importance[worst_col],
                "n_features_at_removal":      len(remaining),
            })

            remaining.remove(worst_col)
            step += 1

            # =========================
            # CALLBACK Y CHECKPOINT AL TÉRMINO DE CADA STEP
            # El callback se invoca primero: si falla, el checkpoint todavía
            # se guarda. `step` ya está incrementado al guardarse, de modo que
            # un resume arranca en el step correcto con el estado actualizado.
            # =========================
            _fire_step_end(
                on_step_end, step,
                history, remaining, removed_features, splits, logger,
            )
            if checkpoint_path is not None:
                _save_checkpoint(
                    checkpoint_path, step,
                    remaining, removed_features, history, splits,
                )

        X_ = X[remaining]
        num_cols_ = X_.select_dtypes(include=['number', 'bool']).columns.tolist()
        cat_cols_ = X_.select_dtypes(include='object').columns.tolist()

        # Si el bucle no ejecutó ninguna iteración (X tenía una sola feature
        # desde el inicio), se avisa pero igualmente se entrena el modelo final
        # con esa única feature para que el return siempre sea consistente.
        if not history:
            logger.warning(
                "El dataset tiene una sola feature desde el inicio — "
                "no se ejecutó ningún step de selección."
            )

        models_, cv_results_, final_model_ = run_pipe(X_, y, num_cols_, cat_cols_, groups, weights)
        metrics_ = get_metrics(cv_results_)
    
    finally:
        if logger is not None:
            _teardown_logger(logger)
        
    return history, remaining, X_, num_cols_, cat_cols_, models_, metrics_, final_model_

# ===========================================================================
# HELPER INTERNO — INVOCACIÓN DEL CALLBACK
# ===========================================================================

def _fire_step_end(on_step_end, step, history, remaining, removed_features,
                   splits, logger):
    """
    Invoca on_step_end si está definido. Pasa copias de remaining y
    removed_features para que el callback no pueda mutar el estado interno.
    Captura cualquier excepción del callback, la loguea y continúa, de modo
    que un callback defectuoso no interrumpe la ejecución ni impide que el
    checkpoint se guarde.
    """
    if on_step_end is None:
        return
    try:
        on_step_end(
            step=step,
            history=history,
            remaining=remaining.copy(),
            removed_features=[r.copy() for r in removed_features],
            splits=splits,
        )
    except Exception as exc:
        logger.warning(f"[on_step_end] Callback raised an exception: {exc}")

#%%
def hist_track(history):
    """Transforma el historial de selección en estructuras tabulares para análisis.

    Convierte la lista de rondas devuelta por
    ``stable_backward_selection_with_oof_permutation`` en un DataFrame
    longitudinal con las importancias por feature y paso, y en una lista
    con la evolución del RMSE baseline a lo largo del proceso.

    Args:
        history (list[dict]): Historial devuelto por
            ``stable_backward_selection_with_oof_permutation``. Cada
            entrada debe contener las claves ``'baseline_score'``,
            ``'remaining'``, ``'drop_importance'`` y
            ``'permutation_importance'``.

    Returns:
        tuple:
            - records_df (pd.DataFrame): DataFrame longitudinal con una
              fila por feature activa en cada ronda y las columnas:

                - ``step`` (int): Índice de la ronda (0-indexado).
                - ``feature`` (str): Nombre de la feature.
                - ``drop`` (float): Drop-column importance de esa feature
                  en esa ronda. ``None`` si no está disponible.
                - ``perm`` (float): Permutation importance de esa feature
                  en esa ronda. ``None`` si no está disponible.

            - rmse_steps (list[float]): RMSE baseline OOF de cada ronda,
              en el mismo orden que ``history``. Útil para visualizar la
              evolución del rendimiento a lo largo de la selección.

    Example:
        >>> records_df, rmse_steps = hist_track(history)
        >>> print(records_df.head())
           step    feature   drop   perm
        0     0  feature_a  0.031  0.045
        1     0  feature_b -0.002  0.001
        ...
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(rmse_steps)
        >>> plt.xlabel("Ronda")
        >>> plt.ylabel("RMSE baseline")
    """
    records = []
    rmse_steps = []

    for i, step in enumerate(history):
        rmse_steps.append(step["baseline_score"])

        for col in step["remaining"]:
            records.append({
                "step":    i,
                "feature": col,
                "drop":    step["drop_importance"].get(col),
                "perm":    step["permutation_importance"].get(col)
            })

    return pd.DataFrame(records), rmse_steps

#%%
def steps_plot(history, save_path=None):
    """Visualiza la evolución de la drop-importance a lo largo del proceso de selección.

    Genera un gráfico de líneas con una serie por feature, mostrando cómo
    evoluciona su drop-importance en cada ronda. Marca con un punto el
    momento exacto en que cada feature fue eliminada. El eje X muestra la
    variable eliminada en cada ronda en lugar del índice numérico.

    Args:
        history (list[dict]): Historial devuelto por
            ``stable_backward_selection_with_oof_permutation``. Cada
            entrada debe contener las claves ``'removed'`` y
            ``'drop_importance'``.

    Returns:
        None: La función genera y muestra el gráfico directamente con
            ``plt.show()``. Para guardar el gráfico antes de mostrarlo
            usar ``save_path``.

    Notes:
        - Las features se ordenan por su drop-importance media absoluta
          a lo largo de todas las rondas, de mayor a menor.
        - La leyenda se coloca fuera del área del gráfico para no tapar
          las series. Si hay muchas features puede solaparse con el eje X;
          en ese caso aumentar el valor de ``bottom`` en
          ``plt.subplots_adjust``.
        - La línea horizontal en y=0 sirve de referencia: features por
          debajo de 0 no penalizan al modelo al ser eliminadas.

    Example:
        >>> steps_plot(history)

    """
    # -------------------------------------------------------
    # Preparación de datos
    # -------------------------------------------------------

    # Etiquetas del eje X: feature eliminada en cada ronda.
    # El último entry del history puede tener removed=None (step de parada,
    # donde ninguna feature superó el umbral y no se eliminó nada). Se filtra
    # para construir steps_labels, y df_track también se limita a esas rondas
    # para que los índices de línea y de etiqueta queden alineados.
    active_history = [h for h in history if h.get("removed") is not None]
    steps_labels   = [h["removed"] for h in active_history]

    # DataFrame longitudinal con drop-importance por feature y ronda
    # (solo rondas con eliminación efectiva, no el step de parada)
    rows = []
    for i, h in enumerate(active_history):
        for col, val in h["drop_importance"].items():
            rows.append({
                "step":    i,
                "feature": col,
                "drop":    val
            })
    df_track = pd.DataFrame(rows)

    # Paso en que cada feature fue eliminada (índice en steps_labels)
    removal_step = {var: i for i, var in enumerate(steps_labels)}

    # Ordena features por importancia media absoluta (mayor a menor)
    top_features = (
        df_track.groupby("feature")["drop"]
        .mean()
        .abs()
        .sort_values(ascending=False)
        .index
    )

    # -------------------------------------------------------
    # Gráfico
    # -------------------------------------------------------
    plt.figure(figsize=(12, 6))

    for col in top_features:
        sub = df_track[df_track["feature"] == col]
        plt.plot(sub["step"], sub["drop"], label=col)

        # Marca el momento de eliminación con un punto
        if col in removal_step:
            x = removal_step[col]
            y = sub.loc[sub["step"] == x, "drop"].values
            if len(y) > 0:
                plt.scatter(x, y[0], s=50, zorder=5)

    # Eje X: nombre de la feature eliminada en cada ronda
    plt.xticks(
        ticks=range(len(steps_labels)),
        labels=steps_labels,
        rotation=45,
        ha="right"
    )

    plt.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    plt.xlabel("Variable eliminada")
    plt.ylabel("Drop importance")
    plt.title("Evolución de importancia y momento de eliminación")

    # Leyenda fuera del área del gráfico para no tapar las series
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=5,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

#%%
def drop_perm_plot(df_track):
    """Scatter plot comparativo de drop-importance vs permutation-importance.

    Visualiza la relación entre ambas métricas de importancia promediadas
    sobre todas las rondas del proceso de selección. Permite identificar
    features redundantes (alta permutation, baja drop), features clave
    (ambas altas) y features prescindibles (ambas bajas o negativas).

    Las etiquetas de cada punto se ajustan automáticamente para evitar
    solapamientos usando ``adjustText``.

    Args:
        df_track (pd.DataFrame): DataFrame longitudinal devuelto por
            ``hist_track``, con las columnas ``'feature'``, ``'drop'``
            y ``'perm'``.

    Returns:
        None: La función genera y muestra el gráfico directamente con
            ``plt.show()``.

    Notes:
        - Requiere la librería ``adjustText`` para el ajuste automático
          de etiquetas (``pip install adjustText``).
        - Las líneas guía en x=0 e y=0 dividen el gráfico en cuatro
          cuadrantes interpretables:

            - Superior derecho: features importantes para el modelo y con
              información real → conservar.
            - Inferior derecho: alta permutation pero baja drop → posible
              redundancia con otras features.
            - Superior izquierdo: baja permutation pero alta drop → la
              feature aporta información no captada por permutación.
            - Inferior izquierdo: ambas bajas o negativas → candidatas
              claras a eliminar.
        - Para guardar el gráfico antes de mostrarlo, añadir
          ``plt.savefig(path, bbox_inches='tight')`` antes de
          ``plt.show()``.

    Example:
        >>> records_df, rmse_steps = hist_track(history)
        >>> drop_perm_plot(records_df)
    """
    # Promedia drop e importancia por permutación sobre todas las rondas
    summary = df_track.groupby("feature").agg(
        drop=("drop", "mean"),
        perm=("perm", "mean")
    )

    # -------------------------------------------------------
    # Gráfico
    # -------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.scatter(summary["perm"], summary["drop"], zorder=5)

    # Etiquetas ajustadas automáticamente para evitar solapamientos
    texts = [
        plt.text(
            summary.loc[feature, "perm"],
            summary.loc[feature, "drop"],
            feature,
            ha='center', va='center',
            fontsize=9
        )
        for feature in summary.index
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    # Líneas guía que dividen los cuatro cuadrantes de interpretación
    plt.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    plt.axvline(0, linestyle="--", color="gray", linewidth=0.8)

    plt.xlabel("Permutation importance (uso del modelo)")
    plt.ylabel("Drop importance (información real)")
    plt.title("Drop vs Permutation importance")

    plt.tight_layout()
    plt.show()
    
#%%
def exhaustive_feature_combinations(
    models,
    X,
    y,
    cv_splits=5,
    groups=None,
    weights=None,
    min_features=4,
    max_features=6,
    mandatory_features=None,
    warn_threshold=15,
    checkpoint_path=None,
    checkpoint_every=50,
):
    """
    Evalúa todas las combinaciones posibles de features (de min_features hasta
    max_features) y devuelve un DataFrame con el RMSE de cada combinación,
    ordenado de menor a mayor RMSE.

    Reutiliza disable_columns_in_pipeline y la misma lógica de folds fijos que
    stable_backward_selection_with_oof_permutation para que los resultados sean
    comparables entre sí y con los de la selección backward.

    Soporta features obligatorias en todas las combinaciones, rango mínimo de
    features, checkpoint automático cada ``checkpoint_every`` combinaciones y
    reanudación automática desde el último checkpoint guardado.

    Args:
        models (list): Lista de pipelines sklearn. Acepta tanto el
            ensemble completo como una lista con un único ``final_model``.

            **Patrón recomendado** — modelo único::

                models, cv_results, final_model = run_pipe(X, y, ...)
                results = exhaustive_feature_combinations(final_model, X, y, ...)
        X (pd.DataFrame): DataFrame de features.
        y (pd.Series): Series de target.
        cv_splits (int, optional): Número de folds para el CV. Por defecto 5.
        groups (pd.Series, optional): Array de grupos para CV estratificado.
            Por defecto None.
        weights (pd.Series, optional): Series de sample weights. Por defecto
            None.
        min_features (int, optional): Número mínimo de features opcionales a
            incluir en cada combinación, sin contar las ``mandatory_features``.
            Por defecto 1.
        max_features (int, optional): Número máximo de features opcionales a
            combinar, sin contar las ``mandatory_features``. None evalúa hasta
            todas las opcionales. Por defecto None.
        mandatory_features (list[str], optional): Lista de features que deben
            aparecer en todas las combinaciones evaluadas. Se añaden a cada
            combinación antes de evaluar y no se combinan entre sí. Si None
            no se fija ninguna feature obligatoria. Por defecto None.
        warn_threshold (int, optional): Si el número de features opcionales
            supera este valor se muestra una advertencia con el total de
            combinaciones y se pide confirmación. Por defecto 15.
        checkpoint_path (str | pathlib.Path, optional): Ruta al fichero JSON
            donde guardar y leer el checkpoint. Si el fichero existe al iniciar
            la función reanuda automáticamente desde el último estado guardado.
            Si es None no se guarda ni se reanuda. Por defecto None.
        checkpoint_every (int, optional): Número de combinaciones evaluadas
            entre cada guardado del checkpoint. Por defecto 50.

    Returns:
        pd.DataFrame: Tabla con columnas ``'features'``, ``'n_features'`` y
            ``'rmse'``, ordenada por RMSE ascendente con índice 1-based.
            ``'n_features'`` refleja el total de features en la combinación
            (opcionales + obligatorias).

    Notes:
        - ``min_features`` y ``max_features`` se aplican sobre las features
          **opcionales** (``X.columns`` menos ``mandatory_features``). El
          total de features evaluado en cada combinación es
          ``k + len(mandatory_features)``.
        - La reanudación reconstruye el conjunto de combinaciones ya evaluadas
          comparando la tupla completa de features (opcionales + obligatorias)
          con las entradas del checkpoint. El orden de columnas de ``X`` y la
          lista ``mandatory_features`` deben ser idénticos en todas las
          ejecuciones.
        - Si ``checkpoint_path`` apunta a un fichero existente pero ``X``,
          ``mandatory_features`` o ``max_features`` han cambiado, los
          resultados pueden ser inconsistentes. En ese caso borrar el
          checkpoint antes de reiniciar.

    Example:
        >>> # Primera ejecución (o reanudación automática si existe checkpoint)
        >>> results = exhaustive_feature_combinations(
        ...     models=models,
        ...     X=X, y=y,
        ...     min_features=2,
        ...     max_features=6,
        ...     mandatory_features=["feature_a", "feature_b"],
        ...     checkpoint_path="checkpoint_exhaustive.json",
        ...     checkpoint_every=100,
        ... )
        >>> print(results.head(10))
    """
    import itertools

    mandatory_features = mandatory_features or []

    # Valida que las features obligatorias existen en X
    missing = [c for c in mandatory_features if c not in X.columns]
    if missing:
        raise ValueError(
            f"Las siguientes mandatory_features no están en X: {missing}"
        )

    # Pool de features opcionales (excluye las obligatorias)
    optional_cols = [c for c in X.columns if c not in mandatory_features]
    n_optional    = len(optional_cols)
    max_k         = min(max_features, n_optional) if max_features else n_optional

    if min_features > max_k:
        raise ValueError(
            f"min_features ({min_features}) no puede ser mayor que "
            f"max_features efectivo ({max_k})."
        )

    # Total de combinaciones opcionales a evaluar
    total = sum(
        len(list(itertools.combinations(optional_cols, k)))
        for k in range(min_features, max_k + 1)
    )

    # -------------------------------------------------------
    # Advertencia si el número de combinaciones es elevado
    # -------------------------------------------------------
    if n_optional > warn_threshold:
        logger.warning(
            "%d features opcionales → %s combinaciones a evaluar "
            "(tamaño %d a %d, + %d obligatorias). "
            "Esto puede tardar mucho tiempo.",
            n_optional, f"{total:,}", min_features, max_k,
            len(mandatory_features)
        )
        answer = input("Continuar? [s/N]: ").strip().lower()
        if answer != "s":
            logger.info("Evaluación cancelada por el usuario.")
            return pd.DataFrame(columns=["features", "n_features", "rmse"])

    # -------------------------------------------------------
    # Reanudación desde checkpoint
    # -------------------------------------------------------
    rows = []
    already_evaluated = set()
    splits = None   # se asignará tras cargar checkpoint o generar splits

    if checkpoint_path is not None:
        cp = pathlib.Path(checkpoint_path)
        if cp.exists():
            checkpoint_data = json.loads(cp.read_text())
            rows = checkpoint_data["rows"]
            already_evaluated = {tuple(r["features"]) for r in rows}
            # Restaurar splits para garantizar comparabilidad entre ejecuciones
            if "splits" in checkpoint_data:
                splits = [
                    (np.array(train), np.array(val))
                    for train, val in checkpoint_data["splits"]
                ]
            logger.info(
                "Checkpoint cargado desde '%s': %d combinaciones ya evaluadas.",
                checkpoint_path, len(already_evaluated)
            )

    # -------------------------------------------------------
    # Configuración del CV (folds fijos)
    # -------------------------------------------------------
    set_config(enable_metadata_routing=True)

    if splits is None:
        # Generar splits solo si no se cargaron del checkpoint
        if groups is not None:
            outer_cv = GroupKFold(n_splits=cv_splits)
            outer_cv.set_split_request(groups=True)
            splits = list(outer_cv.split(X, y, groups))
        else:
            splits = list(KFold(n_splits=cv_splits).split(X, y))

    # -------------------------------------------------------
    # Bucle principal
    # -------------------------------------------------------
    mandatory_str = f" | Obligatorias: {mandatory_features}" if mandatory_features else ""
    logger.info(
        "\n%s\n%s\n%s",
        f"{'#':<8} {'N':>3}  {'Features':<50}  {'RMSE':>10}{mandatory_str}",
        "-" * 78,
        ""
    )

    counter          = len(already_evaluated)
    newly_evaluated  = 0   # combinaciones evaluadas en ESTA ejecución (no en reanudaciones)

    for k in range(min_features, max_k + 1):
        for combo in itertools.combinations(optional_cols, k):

            # Combinación completa: obligatorias + opcionales de esta iteración
            full_combo = tuple(mandatory_features) + combo

            # Salta combinaciones ya evaluadas en una ejecución anterior
            if full_combo in already_evaluated:
                continue

            cols_to_drop = [c for c in X.columns if c not in full_combo]
            combo_scores = []

            for model in models:
                model_mod   = disable_columns_in_pipeline(model, cols_to_drop)
                fold_scores = []

                for train_idx, val_idx in splits:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    w_train = weights.iloc[train_idx] if weights is not None else None
                    w_val = weights.iloc[val_idx] if weights is not None else None

                    m = clone(model_mod)
                    fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}
                    m.fit(X_train, y_train, **fit_kwargs)

                    pred = m.predict(X_val)
                    fold_scores.append(root_mean_squared_error(y_val, pred, sample_weight=w_val))

                combo_scores.append(np.mean(fold_scores))

            rmse            = np.mean(combo_scores)
            counter        += 1
            newly_evaluated += 1

            rows.append({
                "features":   list(full_combo),
                "n_features": len(full_combo),  # total = opcionales + obligatorias
                "rmse":       float(rmse),
            })

            features_str = ", ".join(full_combo)
            logger.info(
                "%s %s  %-50s  %10.4f",
                f"{counter:<8}", f"{len(full_combo):>3}", features_str, rmse
            )

            # -------------------------------------------------------
            # Checkpoint — primera escritura al evaluar la 1ª combinación
            # nueva de esta ejecución (funciona también en resume) y luego
            # cada checkpoint_every combinaciones nuevas.
            # -------------------------------------------------------
            if checkpoint_path is not None and (
                newly_evaluated == 1 or newly_evaluated % checkpoint_every == 0
            ):
                pathlib.Path(checkpoint_path).write_text(
                    json.dumps({
                        "rows": rows,
                        "splits": [
                            [train_idx.tolist(), val_idx.tolist()]
                            for train_idx, val_idx in splits
                        ],
                    }, indent=2)
                )
                logger.info(
                    "Checkpoint guardado en '%s' (%d/%d combinaciones).",
                    checkpoint_path, counter, total
                )

    # -------------------------------------------------------
    # Checkpoint final al terminar
    # -------------------------------------------------------
    if checkpoint_path is not None:
        pathlib.Path(checkpoint_path).write_text(
            json.dumps({
                "rows": rows,
                "splits": [
                    [train_idx.tolist(), val_idx.tolist()]
                    for train_idx, val_idx in splits
                ],
            }, indent=2)
        )
        logger.info(
            "Checkpoint final guardado en '%s' (%d combinaciones).",
            checkpoint_path, counter
        )

    # -------------------------------------------------------
    # Tabla final ordenada por RMSE
    # -------------------------------------------------------
    results = (
        pd.DataFrame(rows)
        .assign(features=lambda df: df["features"].apply(tuple))
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    results.index += 1  # ranking 1-based

    logger.info(
        "\n%s\nRANKING FINAL — %s combinaciones (opcionales %d–%d%s), ordenadas por RMSE\n%s\n%s",
        "=" * 78,
        f"{total:,}",
        min_features, max_k,
        f" + {len(mandatory_features)} obligatorias" if mandatory_features else "",
        "=" * 78,
        results.to_string()
    )

    return results

#%%
def shap_explainer(models, X):
    """Calcula los valores SHAP promediados sobre el ensemble de modelos.

    Transforma ``X`` con el preprocesador del primer modelo, obtiene los
    valores SHAP de cada modelo del ensemble y promedia tanto los valores
    SHAP como los valores base, devolviendo un único objeto
    ``shap.Explanation`` representativo del ensemble.

    Args:
        models (list): Lista de pipelines entrenados con pasos ``'prep'``
            y ``'model'``, habitualmente los ``best_estimator_`` devueltos
            por ``run_pipe``.
        X (pd.DataFrame): Features de entrada. Todos los modelos del ensemble
            deben haber sido entrenados con el mismo esquema de columnas.

    Returns:
        shap.Explanation: Objeto con los valores SHAP promediados del
            ensemble, con las siguientes propiedades:

            - ``values``: Media de los valores SHAP de todos los modelos,
              shape ``(n_samples, n_features)``.
            - ``base_values``: Media de los valores base de todos los
              modelos, shape ``(n_samples,)``.
            - ``data``: Datos transformados por el preprocesador del primer
              modelo, shape ``(n_samples, n_features)``.
            - ``feature_names``: Nombres de features limpios, sin el
              prefijo del ``ColumnTransformer`` (e.g. ``'num__'``).

    Notes:
        - Cada modelo del ensemble transforma ``X`` con su propio
          preprocesador, garantizando que el background y los datos de
          evaluación del ``shap.Explainer`` sean coherentes entre sí.
        - El objeto ``shap.Explanation`` final usa los datos transformados
          por el preprocesador del primer modelo para el campo ``data``,
          y los nombres de features se extraen también del primer modelo.
        - Compatible con modelos basados en árboles (XGBoost, LightGBM,
          etc.) donde ``shap.Explainer`` selecciona automáticamente el
          algoritmo ``TreeExplainer``.

    Example:
        >>> sv = shap_explainer(models, X_sample)
        >>> shap.plots.beeswarm(sv)
    """
    # Nombres de features del primer modelo (referencia para el objeto Explanation)
    names = models[0].named_steps["prep"].get_feature_names_out()
    names = [name.split("__")[-1] for name in names]

    vals, base = [], []
    Xt_ref = None
    for m in models:
        # Cada modelo usa su propio preprocesador: background y evaluación coherentes
        Xt_m = m.named_steps["prep"].transform(X)
        if Xt_ref is None:
            Xt_ref = Xt_m   # se guarda la transformación del primer modelo para data
        e = shap.Explainer(m.named_steps["model"], Xt_m)(Xt_m)
        vals.append(e.values)
        base.append(e.base_values)

    return shap.Explanation(
        values=np.mean(vals, axis=0),
        base_values=np.mean(base, axis=0),
        data=Xt_ref,
        feature_names=names
    )


def shap_values(models, X, size_sample=500, random_state=RANDOM_STATE):
    """Calcula los valores SHAP del ensemble sobre una muestra aleatoria de X.

    Envuelve ``shap_explainer`` aplicando un muestreo previo para reducir
    el coste computacional en datasets grandes.

    Args:
        models (list): Lista de pipelines entrenados, habitualmente los
            ``best_estimator_`` devueltos por ``run_pipe``.
        X (pd.DataFrame): Features completas de las que se extrae la
            muestra.
        size_sample (int, optional): Número de filas a muestrear. Si es
            mayor que ``len(X)`` se usarán todas las filas disponibles.
            Por defecto 500.

    Returns:
        shap.Explanation: Objeto devuelto por ``shap_explainer`` calculado
            sobre la muestra. Ver ``shap_explainer`` para el detalle de
            propiedades.

    Notes:
        - El muestreo es aleatorio sin reemplazamiento usando
          ``random_state`` para reproducibilidad.
        - Si ``size_sample >= len(X)`` se recomienda llamar directamente
          a ``shap_explainer`` para evitar el overhead del muestreo.

    Example:
        >>> sv = shap_values(models, X, size_sample=300)
        >>> shap.plots.beeswarm(sv)
    """
    # Limita el muestreo al tamaño real del dataset si size_sample es mayor
    n = min(size_sample, len(X))
    X_sample = X.sample(n, random_state=random_state)

    return shap_explainer(models, X_sample)

#%%
def feat_corr(X, threshold=0.8):
    """Visualiza la matriz de correlación resaltando correlaciones fuertes.

    Usa una paleta de tres zonas discretas: azul para correlaciones por
    debajo de ``-threshold``, blanco para el rango intermedio y rojo para
    correlaciones por encima de ``+threshold``. La transición es abrupta,
    lo que hace inmediatamente visibles las correlaciones fuertes.

    Args:
        X (pd.DataFrame): Features de entrada. Solo se consideran las
            columnas numéricas.
        threshold (float, optional): Valor absoluto a partir del cual se
            activa el color. Por defecto 0.8.

    Returns:
        None: Genera y muestra el gráfico con ``plt.show()``.

    Example:
        >>> feat_corr(X)               # resalta correlaciones > |0.8|
        >>> feat_corr(X, threshold=0.6)
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    corr = X.select_dtypes(include=['number']).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Cinco zonas discretas: frío | neutro | cálido
    # mid es el punto intermedio entre threshold y 1.0 para zonas simétricas
    mid = round((1.0 + threshold) / 2, 4)
    boundaries = [-1.0, -mid, -threshold, threshold, mid, 1.0]
    colors     = ["#1a3a6b", "#3a7ebf", "#ffffff", "#e8543a", "#7b1a1a"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        norm=norm,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    ax.set_title(
        f"Correlación entre features numéricas  "
        f"(colores activos fuera de ±{threshold})"
    )
    plt.tight_layout()
    plt.show()
