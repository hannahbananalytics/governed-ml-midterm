"""
STAGE 5: THE DEPLOYMENT GATE

This is the governance control. It compares the evaluation metrics against the
policy thresholds and returns PASS or BLOCKED. In CI, BLOCKED means a non-zero
exit code, which turns the build red and stops the model from shipping.

Three properties make this a real control:
  OWNER    : the policy names its owner and approver
  TRIGGER  : runs on every pull request and every push, automatically
  EVIDENCE : writes artifacts/gate_report.json and appends to the audit log

The gate does NOT decide what the thresholds are. Those come from
governance/policy.json, set by people, in advance.

In this repo you ARE allowed to change this file (Lab 5), because this time
you are also the governance team. The CI integrity check will report the
change for owner review, and it will refuse any change that LOOSENS a
threshold. Tightening is allowed. Loosening needs the committee.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def check(name, value, threshold, passed, why):
    return {"check": name, "value": value, "threshold": threshold, "passed": bool(passed), "why": why}


def run_gate(metrics, policy):
    results = []

    # 1. Prohibited features: a PREVENTIVE check on the model's inputs
    used_prohibited = sorted(set(metrics["features"]) & set(policy["prohibited_features"]))
    results.append(check(
        "no_prohibited_features", used_prohibited or "none", "none",
        not used_prohibited,
        "Model may not use sensitive attributes or identifiers as inputs."))

    # 2. Overall performance
    results.append(check(
        "overall_auc", metrics["overall_auc"], policy["min_overall_auc"],
        metrics["overall_auc"] >= policy["min_overall_auc"],
        "Model must be useful overall."))

    # 3. Performance for every group, not just on average
    for g, m in metrics["groups"].items():
        results.append(check(
            f"subgroup_auc[{g}]", m["auc"], policy["min_subgroup_auc"],
            m["auc"] >= policy["min_subgroup_auc"],
            "Model must be useful for every group, not just the largest."))
        results.append(check(
            f"subgroup_tpr[{g}]", m["tpr"], policy["min_subgroup_tpr"],
            m["tpr"] >= policy["min_subgroup_tpr"],
            "Claim-free drivers in every group must mostly be offered the standard rate."))

    # 4. Outcome fairness: DETECTIVE checks that catch proxies
    results.append(check(
        "approval_disparity_ratio", metrics["approval_disparity_ratio"], policy["min_approval_disparity_ratio"],
        metrics["approval_disparity_ratio"] >= policy["min_approval_disparity_ratio"],
        "Approval rates across groups must not diverge beyond the four-fifths screen."))
    results.append(check(
        "tpr_ratio", metrics["tpr_ratio"], policy["min_tpr_ratio"],
        metrics["tpr_ratio"] >= policy["min_tpr_ratio"],
        "Equally safe drivers must have similar chances across groups."))

    # 5. LAB 5: add a check on metrics["fpr_ratio"] against a NEW policy key
    #    (suggested name: min_fpr_ratio). Make it conditional on the key being
    #    present, so a policy without it still works:
    #
    #        if "min_fpr_ratio" in policy:
    #            results.append(check( ... ))
    #
    #    Then choose the threshold in governance/policy.json, defend it in
    #    MIDTERM.md, and prove the check fires in tests/test_lab5_gate.py.

    if "min_fpr_ratio" in policy:
        results.append(check(
            "fpr_ratio",
            metrics["fpr_ratio"],
            policy["min_fpr_ratio"],
            metrics["fpr_ratio"] >= policy["min_fpr_ratio"],
            "Claim-filing drivers must be let through at"
            " similar rates across groups."))
        
    passed = all(r["passed"] for r in results)
    return {"decision": "PASS" if passed else "BLOCKED", "policy_version": policy["policy_version"], "checks": results}


def print_report(report, metrics):
    line = "=" * 78
    print(line)
    print(f"  DEPLOYMENT GATE   model {metrics['model_version']}   policy v{report['policy_version']}")
    print(line)
    print(f"  {'check':<32}{'value':>14}{'threshold':>12}   result")
    print("-" * 78)
    for r in report["checks"]:
        val = r["value"] if not isinstance(r["value"], float) else f"{r['value']:.4f}"
        thr = r["threshold"] if not isinstance(r["threshold"], float) else f"{r['threshold']:.2f}"
        mark = "pass" if r["passed"] else "FAIL  <--"
        print(f"  {r['check']:<32}{str(val):>14}{str(thr):>12}   {mark}")
    print(line)
    if report["decision"] == "PASS":
        print("  DECISION: PASS. This model may proceed to deployment review.")
    else:
        print("  DECISION: BLOCKED. This model may NOT be deployed.")
        for r in report["checks"]:
            if not r["passed"]:
                print(f"    - {r['check']}: {r['why']}")
    print(line)


def save_report(report, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "gate_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
