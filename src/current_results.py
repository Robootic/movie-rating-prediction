"""Persistência da rodada atual; não usa receitas ou resultados históricos."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RepeatedKFold


def save_run(root, notebook, frame, target, predictions, by_fold,
             n_splits, n_repeats, seed, clip, selections=None):
    root = Path(root)
    destination = root / 'results' / 'rodada_atual'
    destination.mkdir(parents=True, exist_ok=True)
    partitions = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed).split(frame))
    coverage = np.zeros((n_repeats, len(frame)), dtype=int)
    for i, (train, valid) in enumerate(partitions):
        assert not np.intersect1d(train, valid).size
        coverage[i // n_splits, valid] += 1
    assert (coverage == 1).all()
    records = []
    arrays = {'target': np.asarray(target)}
    for i, (name, values) in enumerate(predictions.items()):
        values = np.asarray(values)
        assert values.shape == (n_repeats, len(frame)) and np.isfinite(values).all()
        if clip:
            assert ((values >= 0) & (values <= 5)).all()
        scores = by_fold.loc[by_fold['modelo'].eq(name)]
        assert len(scores) == n_splits * n_repeats
        recomputed = [float(np.sqrt(np.mean((target[valid] - values[j // n_splits, valid]) ** 2)))
                      for j, (_, valid) in enumerate(partitions)]
        np.testing.assert_allclose(recomputed, scores['E_CV'], atol=1e-12)
        key = f'model_{i}'
        arrays[key] = values
        records.append({'name': name, 'array_key': key,
                        'rmse_oof': float(np.sqrt(np.mean((values - target) ** 2))),
                        'rmse_repeat': np.sqrt(np.mean((values - target) ** 2, axis=1)).tolist(),
                        'rmse_valid': recomputed,
                        'rmse_train': scores['E_in'].tolist()})
    source = root / 'notebooks' / (notebook + '.ipynb')
    cells = json.loads(source.read_text(encoding='utf-8'))['cells']
    code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    payload = {
        'notebook': notebook, 'data_sha256': sha256((root / 'data/train.csv').read_bytes()).hexdigest(),
        'code_sha256': sha256(code.encode()).hexdigest(),
        'partitions_sha256': sha256(b''.join(valid.astype('<i8').tobytes() for _, valid in partitions)).hexdigest(),
        'protocol': dict(n_splits=n_splits, n_repeats=n_repeats, seed=seed, clip=clip),
        'versions': dict(python=platform.python_version(), numpy=np.__version__,
                         pandas=pd.__version__, sklearn=sklearn.__version__),
        'records': records,
        'selections': json.loads(pd.DataFrame(selections or []).to_json(orient='records'))}
    (destination / (notebook + '.json')).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    np.savez_compressed(destination / (notebook + '.npz'), **arrays)
    print('Resultados verificados e salvos em', (destination / (notebook + '.json')).relative_to(root))
