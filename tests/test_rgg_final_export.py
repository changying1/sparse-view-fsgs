import os
from argparse import Namespace

from train import save_final_rgg_diagnostics


class _Gaussians:
    def __init__(self):
        self.calls = []

    def save_rgg_diagnostics(self, directory, observation_end_iter=None):
        self.calls.append((directory, observation_end_iter))


def test_final_rgg_diagnostics_export_enabled():
    args = Namespace(enable_rgg_diagnostics=True)
    opt = Namespace(iterations=900)
    scene = Namespace(model_path="output/run")
    gaussians = _Gaussians()

    save_final_rgg_diagnostics(args, opt, scene, gaussians)

    assert gaussians.calls == [(os.path.join("output/run", "rgg_diagnostics_final"), 900)]


def test_final_rgg_diagnostics_export_disabled():
    args = Namespace(enable_rgg_diagnostics=False)
    opt = Namespace(iterations=900)
    scene = Namespace(model_path="output/run")
    gaussians = _Gaussians()

    save_final_rgg_diagnostics(args, opt, scene, gaussians)

    assert gaussians.calls == []
