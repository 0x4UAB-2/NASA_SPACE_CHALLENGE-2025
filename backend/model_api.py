import pandas as pd
import pickle
import importlib

features_of_interest = [
    "koi_score", "koi_fpflag_nt", "koi_max_mult_ev", "koi_dicco_msky", "koi_fpflag_co", "koi_fwm_stat_sig",
    "koi_fpflag_ss", "koi_dikco_msky", "koi_model_snr", "koi_prad", "koi_smet_err2", "koi_ror",
    "koi_fwm_sdec_err", "koi_duration_err1", "koi_dor", "koi_duration_err2", "koi_prad_err2", "koi_fwm_sra_err",
    "koi_time0bk_err1", "koi_dicco_msky_err", "koi_time0_err1", "koi_steff_err2", "koi_fwm_srao_err", "koi_count",
    "koi_prad_err1", "koi_dikco_mra_err", "koi_time0bk_err2", "koi_fwm_sdeco_err", "koi_dikco_msky_err", "koi_steff_err1",
    "koi_max_sngle_ev", "koi_fpflag_ec", "koi_dikco_mdec_err", "koi_srho_err2", "koi_incl", "koi_dicco_mdec_err",
    "koi_insol_err1", "koi_dor_err1", "koi_sma", "koi_dor_err2", "koi_insol_err2", "koi_num_transits",
    "koi_period", "koi_ror_err2", "koi_dicco_mdec", "koi_bin_oedp_sig", "koi_teq", "koi_fwm_pdeco_err",
    "koi_period_err1", "koi_dikco_mdec", "koi_period_err2", "koi_time0_err2", "koi_duration", "koi_srho_err1",
    "koi_ror_err1", "koi_srho", "koi_depth", "koi_impact", "koi_dikco_mra", "koi_insol",
    "koi_dicco_mra", "koi_fwm_prao_err", "koi_fwm_sdeco", "koi_srad_err1", "koi_dicco_mra_err", "koi_fwm_sdec",
    "dec", "koi_srad_err2", "koi_time0bk", "koi_depth_err2", "koi_slogg_err2", "koi_fwm_srao",
    "koi_impact_err1", "ra", "koi_impact_err2", "koi_smass_err1", "koi_time0", "koi_smet",
    "koi_zmag", "koi_smet_err1", "koi_gmag", "koi_smass_err2", "koi_fwm_prao", "koi_imag",
    "koi_srad", "koi_depth_err1", "koi_kmag", "koi_slogg", "koi_jmag", "koi_fwm_sra",
    "koi_fwm_pdeco", "koi_hmag", "koi_steff", "koi_rmag", "koi_kepmag", "koi_ldm_coeff1",
    "koi_smass", "koi_ldm_coeff2", "koi_slogg_err1", "koi_tce_plnt_num"
]

SCALER_ENDPOINT = "scaler.pkl"

SCALER = None

MODELS_ENDPOINTS = {
    0: "random_forest.pkl",
    1: "cat.pkl",
    2: "xgb.pkl",
    3: "ensembled_rf_nb_lgb_svc.pkl"
}

MODELS_API = {
    0: None,
    1: None,
    2: None,
    3: None
}

def load_models():
    global SCALER
    global MODELS_API

    scaler_path = f'models/{SCALER_ENDPOINT}'
    try:
        with open(scaler_path, 'rb') as f:
            SCALER = pickle.load(f)
    except ModuleNotFoundError as e:
        raise ValueError("Scaler model could not be loaded.") from e

    for model_id in MODELS_ENDPOINTS.keys():
        model_path = f'models/{MODELS_ENDPOINTS[model_id]}'
        try:
            with open(model_path, 'rb') as f:
                MODELS_API[model_id] = pickle.load(f)
        except ModuleNotFoundError as e:
            # Common when loading pickles that reference sklearn classes but sklearn isn't installed
            missing = e.name if hasattr(e, 'name') else str(e)
            raise ModuleNotFoundError(
                f"{missing} is required to unpickle {model_path}.\n"
                "Install it in your environment, for example: `pip install scikit-learn`"
            ) from e
        except Exception:
            # Re-raise other exceptions with model path context
            raise

def pipeline_predict(df: pd.DataFrame, model_id: int):
    # Normalize incoming column names: strip whitespace and lowercase
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    df = df[features_of_interest]

    # Scale features
    if SCALER is not None:
        X_scaled = SCALER.transform(df.values)
        df = pd.DataFrame(X_scaled, columns=df.columns)

    # Drop nans
    df = df.dropna()

    print("DATAFRAME SHAPE AFTER PREPROCESSING: ", df.shape, df.iloc[0])
    model = MODELS_API.get(model_id, -1)
    if model == -1:
        raise ValueError("Model not found")

    # Some sklearn models were trained on numpy arrays (no feature names).
    # Passing a DataFrame with column names can produce a warning. Use .values
    # to pass a plain array to the model to avoid the "feature names" warning.
    X = df.values
    predictions = model.predict(X)

    print("PREDICTIONS SAMPLE: ", predictions[:5])

    # Use assign to avoid DataFrame fragmentation warnings and return a new frame
    return df.assign(prediction=predictions)

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

import lightgbm as lgb

import catboost as cb

from xgboost import XGBClassifier

import numpy as np


def pipeline_fit_and_predict(to_predict, dataset_path: str, model_names):

    """

    

    Entrena un VotingClassifier (soft) con los modelos especificados

    y predice sobre 'to_predict'.

    model_names: lista de strings, ej. ['rf', 'xgb', 'lr', 'svc']

    """
    # --- Filtrar dataset ---

    print(to_predict.columns)
    dataset = pd.read_csv(dataset_path, comment="#", engine="python")

    mask = dataset["koi_disposition"] != "CANDIDATE"

    dataset = dataset[mask].copy()

    features_to_predict = to_predict.columns.tolist()

    final_features = list(set(features_to_predict) & set(features_of_interest))

    if final_features == []:
        raise ValueError("No hay features válidas para predecir")

    dataset = dataset[final_features + ["koi_disposition"]]
    # --- Separar features y target ---

    dataset = dataset.fillna(dataset.mean(numeric_only=True))

    to_predict = to_predict[final_features]



    

    X = dataset[final_features]
    y = dataset["koi_disposition"].values

    # --- Escalado global ---

    scaler = StandardScaler()


    X_scaled = scaler.fit_transform(X)

    to_predict_scaled = scaler.transform(to_predict)

    # --- Modelos base ---

    rf_clf = RandomForestClassifier(

        n_estimators=1000,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=5,
        random_state=42
    )


    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


    cat_clf = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )


    xgb_clf = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )


    svc_pipe = Pipeline([('svc', SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42))])


    nb_pipe = Pipeline([
        ('nb', GaussianNB())
    ])


    lr_pipe = Pipeline([
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])


    knn_pipe = Pipeline([
        ('knn', KNeighborsClassifier(n_neighbors=10))
    ])


    # --- Diccionario de modelos disponibles ---

    available_models = {
        'rf': rf_clf,
        'lgb': lgb_clf,
        'cat': cat_clf,
        'xgb': xgb_clf,
        'svc': svc_pipe,
        'nb': nb_pipe,
        'lr': lr_pipe,
        'knn': knn_pipe
    }


    # --- Crear ensemble con los modelos seleccionados ---

    estimators = []

    for name in model_names:
        if name not in available_models:
            raise ValueError(f"Modelo '{name}' no está definido. Usa uno de: {list(available_models.keys())}")
        estimators.append((name, available_models[name]))
    
    voting = VotingClassifier(estimators=estimators, voting='soft')

    # --- Entrenar y predecir ---
    voting.fit(X_scaled, y)
    preds = voting.predict(to_predict_scaled)
    preds = np.array(['CONFIRMED' if p == 1 else 'FALSE POSITIVE' for p in preds])

    return pd.DataFrame(preds)