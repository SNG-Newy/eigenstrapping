import sys
from pathlib import Path
import numpy as np
from scipy import stats as sstats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from eigenstrapping import stats


def test_gpd_inference_tail_selection():
    perms = np.linspace(0, 1, 100)
    stat = 0.95
    per = 0.1

    threshold = np.quantile(perms, 1 - per)
    tail_values = perms[perms > threshold]
    params = sstats.genpareto.fit(tail_values)
    expected = 1 - sstats.genpareto.cdf(stat, *params)

    result = stats.gpd_inference(perms, stat=stat, per=per)
    assert np.isclose(result, expected)
