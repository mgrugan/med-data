"""Generates (and optionally executes) the research walkthrough notebook.

    python notebooks/build_notebook.py --execute
"""
import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = [
    nbf.v4.new_markdown_cell("""# RecoveryAI: research walkthrough

**Primary question.** To what extent can patient characteristics available near hospital admission predict length of
stay in real, de-identified hospital data (UCI Diabetes 130-US Hospitals), and how reliably can modern ML models quantify uncertainty in those predictions?

**Secondary question.** Do advanced nonlinear models provide statistically meaningful improvement over simple
statistical and naive baselines?

This notebook reads the artefacts written by `python -m training.run_pipeline --budget full`. It does not retrain
anything, and it never touches the test set again."""),
    nbf.v4.new_code_cell("""import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
R = json.loads((ROOT / "validation/results/validation_report.json").read_text())
pd.set_option("display.precision", 3)
print(R["research_questions"]["primary"])"""),
    nbf.v4.new_markdown_cell("## 1. Data: the LOS distribution\nLength_of_Stay is `time_in_hospital` (1–14 days). One stay per patient (the first); deaths and hospice transfers removed."),
    nbf.v4.new_code_cell("""d = R["data"]
print({k: d[k] for k in ["rows_raw", "repeat_encounters_removed", "death_or_hospice_removed", "n_dev", "n_test", "los_mean", "los_median", "los_sd", "los_skew"]})
pd.DataFrame({k: pd.Series(v) for k, v in d["group_means"].items()}).T"""),
    nbf.v4.new_markdown_cell("![LOS](../validation/figures/los_distribution.png)"),
    nbf.v4.new_markdown_cell("## 2. Nested cross-validation (headline generalisation estimate)"),
    nbf.v4.new_code_cell("""rows = []
for k, v in R["nested_cv"].items():
    s = v["summary"]
    rows.append({"model": k, "MAE mean": s["mae"]["mean"], "MAE sd": s["mae"]["sd"], "MAE median": s["mae"]["median"],
                 "IQR": s["mae"]["iqr"], "RMSE": s["rmse"]["mean"], "R2": s["r2"]["mean"],
                 "repeated-CV MAE": R["repeated_cv"][k]["summary"]["mae"]["mean"]})
pd.DataFrame(rows).set_index("model")"""),
    nbf.v4.new_markdown_cell("## 3. Model selection (CV only, locked before testing)"),
    nbf.v4.new_code_cell("""sel = R["selection"]; print(sel["reason"])
pd.DataFrame(sel["comparisons_vs_leader"]).T"""),
    nbf.v4.new_markdown_cell("## 4. Final test set: bootstrap CIs and paired differences"),
    nbf.v4.new_code_cell("""T = R["test"]
pd.DataFrame({k: {"MAE": b["metrics"]["mae"]["estimate"], "BCa low": b["metrics"]["mae"]["ci_bca"][0], "BCa high": b["metrics"]["mae"]["ci_bca"][1],
                  "RMSE": b["metrics"]["rmse"]["estimate"], "R2": b["metrics"]["r2"]["estimate"], "bias": b["metrics"]["bias"]["estimate"]}
              for k, b in T["bootstrap"].items()}).T"""),
    nbf.v4.new_code_cell("""pd.DataFrame({k: {"MAE diff": v["mae"]["difference"], "CI low": v["mae"]["ci_percentile"][0], "CI high": v["mae"]["ci_percentile"][1],
                  "p": v["mae"]["p_value"]} for k, v in T["paired_differences"].items() if k.endswith("median_baseline")}).T"""),
    nbf.v4.new_markdown_cell("![comparison](../validation/figures/model_comparison.png)"),
    nbf.v4.new_markdown_cell("## 5. Shuffled-target negative control and permutation p-values"),
    nbf.v4.new_code_cell("""pd.DataFrame({k: {kk: p[kk] for kk in ["observed_mae", "null_mean", "null_sd", "improvement_over_shuffled", "p_value", "n_permutations"]}
              for k, p in R["permutation_tests"].items()}).T"""),
    nbf.v4.new_markdown_cell("![shuffled](../validation/figures/shuffled_target.png)"),
    nbf.v4.new_markdown_cell("## 6. Statistical baseline: Gamma GLM (AIC / AICc / BIC are valid only here)"),
    nbf.v4.new_code_cell("""sm = R["statistical_model"]; print(sm["information_criteria"]); print(sm["null_comparison"])
pd.DataFrame(sm["coefficients"]).set_index("term")[["coef", "std_err", "ci_low", "ci_high", "exp_coef", "p_value"]]"""),
    nbf.v4.new_markdown_cell("## 7. Calibration, learning curves and conformal coverage"),
    nbf.v4.new_markdown_cell("![calib](../validation/figures/calibration_learning_coverage.png)"),
    nbf.v4.new_markdown_cell("## 8. Positive control and quality gate"),
    nbf.v4.new_code_cell("""pc = ROOT / "validation/results/positive_control.json"
if pc.exists():
    P = json.loads(pc.read_text()); print(P["conclusion"]); display(pd.DataFrame({k: {"MAE": v["mae"]["mean"], "R2": v["r2"]["mean"]} for k, v in P["cv"].items()}).T)
g = R["quality_gate"]; print(g["headline"])
pd.DataFrame([{"check": c["label"], "failed": c["failed"]} for c in g["checks"]])"""),
    nbf.v4.new_markdown_cell("""## Conclusion

Read the quality gate together with the shuffled-target control, the set A vs set B comparison and the positive control.
On real data, admission-time variables carry a real but modest signal: models beat the median baseline and the shuffled
target, but explain only a small share of the variance. Most of the apparent accuracy of richer feature sets comes from
information gathered during the stay, which would leak the answer at admission time. The earlier synthetic dataset, kept
in `validation/results/synthetic_reference/`, contained no signal at all."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true"); a = ap.parse_args()
    out = HERE / "01_research_walkthrough.ipynb"
    if a.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, out)
    print("wrote", out)
