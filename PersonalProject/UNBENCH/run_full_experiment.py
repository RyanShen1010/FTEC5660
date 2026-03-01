"""
FTEC5660 Reproducibility — Full Experiment Pipeline
Runs: (1) OpenAI baseline, (2) DeepSeek, (3) Qwen, (4) Parameter modifications.
Outputs: results/experiments.json + comparison table.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    # --- Baseline: OpenAI (paper reproduction) ---
    {"name": "openai_temp0", "provider": "openai", "model": "gpt-4o-mini", "temp": 0, "trials": 1, "env": "OPENAI_API_KEY"},
    # --- Multi-model comparison ---
    {"name": "deepseek_temp0", "provider": "deepseek", "model": "deepseek-chat", "temp": 0, "trials": 1, "env": "DEEPSEEK_API_KEY"},
    {"name": "qwen_temp0", "provider": "qwen", "model": "qwen-plus", "temp": 0, "trials": 1, "env": "DASHSCOPE_API_KEY"},
    # --- Modification 1: Temperature 0 → 0.7 ---
    {"name": "openai_temp07", "provider": "openai", "model": "gpt-4o-mini", "temp": 0.7, "trials": 3, "env": "OPENAI_API_KEY"},
    # --- Modification 2: max_tokens 5 → 50 ---
    {"name": "openai_max50", "provider": "openai", "model": "gpt-4o-mini", "temp": 0, "trials": 1, "max_tokens": 50, "env": "OPENAI_API_KEY"},
]


def run_experiment(exp: dict):
    """Run one experiment, return metrics dict or None if skipped."""
    if not os.environ.get(exp["env"]):
        print(f"  [SKIP] {exp['name']} — set {exp['env']}")
        return None

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_task3.py"),
        "--provider", exp["provider"],
        "--model", exp["model"],
        "--temperature", str(exp["temp"]),
        "--trials", str(exp["trials"]),
        "--output-json", str(RESULTS_DIR / f"{exp['name']}.json"),
    ]
    if exp.get("max_tokens"):
        cmd.extend(["--max-tokens", str(exp["max_tokens"])])

    print(f"  Running: {exp['name']} ...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:300]}")
        return None

    out_path = RESULTS_DIR / f"{exp['name']}.json"
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        return data
    return None


def main():
    print("=" * 60)
    print("FTEC5660 UNBench Task 3 — Full Reproducibility Experiment")
    print("=" * 60)

    all_results = []
    for exp in EXPERIMENTS:
        print(f"\n[{exp['name']}]")
        data = run_experiment(exp)
        if data:
            all_results.append({"experiment": exp["name"], **data})

    # Save aggregated results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiments": all_results,
    }
    out_path = RESULTS_DIR / "experiments.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (fill report.html)")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Accuracy':>10} {'Bal.ACC':>10} {'Mac.F1':>10} {'AUC':>10}")
    print("-" * 65)
    for r in all_results:
        m = r.get("metrics", {})
        acc = m.get("accuracy", 0)
        bal = m.get("balanced_acc", 0)
        f1 = m.get("macro_f1", 0)
        auc = m.get("auc", 0)
        mean_std = r.get("mean_std", {})
        if mean_std:
            acc_s = f"{mean_std.get('accuracy', {}).get('mean', acc):.4f}±{mean_std.get('accuracy', {}).get('std', 0):.2f}"
            bal_s = f"{mean_std.get('balanced_acc', {}).get('mean', bal):.4f}±{mean_std.get('balanced_acc', {}).get('std', 0):.2f}"
            f1_s = f"{mean_std.get('macro_f1', {}).get('mean', f1):.4f}±{mean_std.get('macro_f1', {}).get('std', 0):.2f}"
            auc_s = f"{mean_std.get('auc', {}).get('mean', auc):.4f}±{mean_std.get('auc', {}).get('std', 0):.2f}"
        else:
            acc_s = f"{acc:.4f}"
            bal_s = f"{bal:.4f}"
            f1_s = f"{f1:.4f}"
            auc_s = f"{auc:.4f}"
        print(f"{r['experiment']:<25} {acc_s:>10} {bal_s:>10} {f1_s:>10} {auc_s:>10}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
