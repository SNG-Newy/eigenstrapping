import sys
from pathlib import Path
import numpy as np
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))
from eigenstrapping.datasets import get_surface_examples


def test_get_surface_examples_download(tmp_path, monkeypatch):
    sample_data = "1 2 3\n4 5 6\n"
    calls = []

    def fake_download(url, output, quiet=True, fuzzy=False):
        assert fuzzy is True
        with open(output, "w") as f:
            f.write(sample_data)
        calls.append(url)
        return output

    monkeypatch.setattr("eigenstrapping.datasets.base.gdown", types.SimpleNamespace(download=fake_download))

    def fake_get_dataset_info(name):
        url = "https://drive.google.com/uc?id=dummy"
        if name == "surfaces":
            return [
                {"space": "fsaverage", "den": "10k", "hemi": "lh", "fname": "lh.surf.gii", "rel_path": "", "checksum": None, "url": url},
                {"space": "fsaverage", "den": "10k", "hemi": "rh", "fname": "rh.surf.gii", "rel_path": "", "checksum": None, "url": url},
            ]
        elif name == "eigenmodes":
            return [
                {"space": "fsaverage", "den": "10k", "hemi": "lh", "format": "emodes", "fname": "lh_emodes.txt", "rel_path": "", "checksum": None, "url": url},
                {"space": "fsaverage", "den": "10k", "hemi": "lh", "format": "evals", "fname": "lh_evals.txt", "rel_path": "", "checksum": None, "url": url},
                {"space": "fsaverage", "den": "10k", "hemi": "rh", "format": "emodes", "fname": "rh_emodes.txt", "rel_path": "", "checksum": None, "url": url},
                {"space": "fsaverage", "den": "10k", "hemi": "rh", "format": "evals", "fname": "rh_evals.txt", "rel_path": "", "checksum": None, "url": url},
            ]
        else:
            return []

    monkeypatch.setattr("eigenstrapping.datasets.base.get_dataset_info", fake_get_dataset_info)
    monkeypatch.setenv("EIGEN_DATA", str(tmp_path))

    def fake_nib_load(path, *_args, **_kwargs):
        if "mask" in str(path):
            data = np.array([True, False])
        else:
            data = np.array([1.0, 2.0])
        class FakeImg:
            def __init__(self, arr):
                self._arr = arr
            def agg_data(self):
                return self._arr
        return FakeImg(data)

    monkeypatch.setattr("eigenstrapping.datasets.base.nib.load", fake_nib_load)
    monkeypatch.setattr("eigenstrapping.datasets.base.datasets.fetch_annotation", lambda **_kw: tmp_path / "ann.gii")

    class Medial:
        L = tmp_path / "maskL.gii"
        R = tmp_path / "maskR.gii"

    class FsAvg:
        medial = Medial()

    monkeypatch.setattr("eigenstrapping.datasets.base.datasets.fetch_fsaverage", lambda density="10k": FsAvg())

    result = get_surface_examples()
    assert calls, "gdown download not invoked"
    assert len(result) == 6
    for arr in result:
        assert isinstance(arr, np.ndarray)
