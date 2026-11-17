"""Tests for determinism settings and environment recording."""

import json

import torch

from tbm_rop.evaluation import save_report
from tbm_rop.utils import collect_environment, set_seed


def test_set_seed_enables_deterministic_cudnn():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_collect_environment_reports_versions():
    env = collect_environment()

    assert "python_version" in env
    assert "torch_version" in env
    assert "packages" in env and "scikit-learn" in env["packages"]
    assert "gpu_model" in env  # "N/A" on CPU-only machines


def test_save_report_embeds_environment(tmp_path):
    path = save_report({"stage": "test"}, tmp_path / "report.json")

    payload = json.loads(path.read_text())
    assert payload["stage"] == "test"
    assert "environment" in payload
    assert "python_version" in payload["environment"]
