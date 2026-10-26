"""
GATE TESTS: testing the control itself.

If the gate has a bug, every model it approves is suspect. So the gate gets
its own tests, using made-up metrics where we already know the right answer.
The first four should pass from the start. The last one is yours (Lab 5).

Run:   python -m pytest -q tests/test_lab5_gate.py
"""
import pytest

from src.gate import run_gate
from src.prepare import load_policy

POLICY = load_policy()


def fake_metrics(**overrides):
    m = {
        "model_version": "test",
        "features": ["years_licensed", "prior_claims_3y"],
        "overall_auc": 0.85,
        "groups": {
            "urban": {"n": 100, "auc": 0.85, "approval_rate": 0.80, "tpr": 0.90, "fpr": 0.40},
            "rural": {"n": 50, "auc": 0.84, "approval_rate": 0.75, "tpr": 0.88, "fpr": 0.38},
        },
        "approval_disparity_ratio": 0.9375,
        "tpr_ratio": 0.978,
        "fpr_ratio": 0.95,
    }
    m.update(overrides)
    return m


def test_gate_passes_a_good_model():
    assert run_gate(fake_metrics(), POLICY)["decision"] == "PASS"


def test_gate_blocks_low_disparity_ratio():
    assert run_gate(fake_metrics(approval_disparity_ratio=0.60), POLICY)["decision"] == "BLOCKED"


def test_gate_blocks_prohibited_feature():
    m = fake_metrics(features=["years_licensed", "area"])
    assert run_gate(m, POLICY)["decision"] == "BLOCKED"


def test_gate_blocks_weak_subgroup_even_if_average_is_fine():
    groups = {
        "urban": {"n": 100, "auc": 0.90, "approval_rate": 0.80, "tpr": 0.90, "fpr": 0.40},
        "rural": {"n": 50, "auc": 0.55, "approval_rate": 0.75, "tpr": 0.88, "fpr": 0.38},
    }
    assert run_gate(fake_metrics(groups=groups), POLICY)["decision"] == "BLOCKED"


def test_gate_blocks_fpr_gap():
    """LAB 5: prove your new check fires.
    Replace the skip with a test that builds metrics with an fpr_ratio BELOW
    your policy's min_fpr_ratio and asserts the decision is BLOCKED. Then add a
    second assert showing that an fpr_ratio at or above the threshold PASSES."""
    threshold = POLICY["min_fpr_ratio"]
    assert run_gate(fake_metrics(fpr_ratio=threshold - 0.10),
                    POLICY)["decision"] == "BLOCKED"
    assert run_gate(fake_metrics(fpr_ratio=threshold),
                    POLICY)["decision"] == "PASS"
