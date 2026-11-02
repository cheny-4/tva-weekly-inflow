from pathlib import Path
import hashlib, json
import numpy as np

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    if len(obs) < 2:
        return {"n": int(len(obs)), "nse": None, "kge": None, "r": None,
                "rmse": None, "mae": None, "pbias_percent": None}
    den = np.sum((obs - obs.mean()) ** 2)
    r = float(np.corrcoef(obs, pred)[0, 1]) if obs.std() > 0 and pred.std() > 0 else None
    beta = float(pred.mean() / obs.mean()) if obs.mean() != 0 else None
    alpha = float(pred.std() / obs.std()) if obs.std() > 0 else None
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if all(
        v is not None for v in (r, alpha, beta)) else None
    return {"n": int(len(obs)), "nse": float(1 - np.sum((pred - obs)**2) / den) if den > 0 else None,
            "kge": float(kge) if kge is not None else None, "r": r,
            "rmse": float(np.sqrt(np.mean((pred - obs)**2))),
            "mae": float(np.mean(np.abs(pred - obs))),
            "pbias_percent": float(100 * (pred.sum() / obs.sum() - 1)) if obs.sum() else None}


