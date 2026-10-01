"""Shared setup for experiment scripts. Run every script from the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)


def load_apipop():
    """apipop districts with >= 5 schools, covariates cleaned. Needs data/apipop.pkl
    (python experiments/fetch_apipop.py)."""
    import pandas as pd
    path = DATA / 'apipop.pkl'
    if not path.exists():
        raise SystemExit('data/apipop.pkl missing: run  python experiments/fetch_apipop.py')
    d = pd.read_pickle(path)
    d = d[d.groupby('dnum').dnum.transform('size') >= 5].copy()
    base = ['meals', 'ell', 'mobility', 'not.hsg', 'hsg', 'some.col', 'col.grad', 'grad.sch',
            'avg.ed', 'full', 'emer', 'enroll', 'pct.resp']
    d['yr_rnd'] = (d['yr.rnd'].astype(str) == 'Yes').astype(float)
    d = pd.concat([d, pd.get_dummies(d.stype.astype(str), prefix='st', dtype=float)], axis=1)
    extra = ['yr_rnd'] + [c for c in d.columns if str(c).startswith('st_')]
    for c in base + ['api99', 'api00']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d[base] = d[base].fillna(d[base].median())
    feats = {'socio': base + extra, 'api99': base + extra + ['api99']}
    return d, feats


def apipop_replication(d, feats, n, rng, n_train=110, n_calib=110):
    """One replication: split districts, sample n schools SRSWOR in train/calib districts,
    fit Ridge h on train-sample schools; centre = population mean of h in a district.
    Returns calibration scores V, their design variances D, calibration latent W_c,
    and eval-district latent scores W_e."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeCV
    groups = {k: g for k, g in d.groupby('dnum')}
    p = rng.permutation(np.array(list(groups)))
    T, C, E = p[:n_train], p[n_train:n_train + n_calib], p[n_train + n_calib:]
    samp = {c: groups[c].sample(n, random_state=int(rng.integers(1e9))) for c in np.r_[T, C]}
    tr = pd.concat([samp[c] for c in T])
    h = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(tr[feats].values, tr.api00.values)
    V, D, Wc = [], [], []
    for c in C:
        g, s = groups[c], samp[c]
        res_s = s.api00.values - h.predict(s[feats].values)
        V.append(res_s.mean()); D.append((1 - n / len(g)) * res_s.var(ddof=1) / n)
        Wc.append((g.api00.values - h.predict(g[feats].values)).mean())
    We = np.array([(groups[c].api00.values - h.predict(groups[c][feats].values)).mean() for c in E])
    return np.array(V), np.maximum(np.array(D), 1e-6), np.array(Wc), We
