"""
UNBench Task 3: Resolution Adoption Prediction
Reproducibility script - OpenAI, DeepSeek, and Qwen.
"""
import json
import os
import argparse
from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, confusion_matrix, precision_recall_curve
)
import numpy as np

try:
    from imblearn.metrics import geometric_mean_score
except ImportError:
    def geometric_mean_score(y_true, y_pred):
        from sklearn.metrics import recall_score
        r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        return np.sqrt(r0 * r1) if (r0 * r1) > 0 else 0


def get_client(provider="openai", api_key=None, base_url=None):
    """Create client for OpenAI, DeepSeek, or Qwen."""
    from openai import OpenAI
    key = api_key or (
        os.environ.get("DASHSCOPE_API_KEY") if provider == "qwen" else
        os.environ.get("DEEPSEEK_API_KEY") if provider == "deepseek" else
        os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError("Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or DASHSCOPE_API_KEY (for Qwen)")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return ("openai", OpenAI(**kwargs))


def _get_base_url(provider, base_url):
    if provider == "deepseek" and not base_url:
        return "https://api.deepseek.com"
    if provider == "qwen" and not base_url:
        return os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return base_url


def _call_openai(client, model, prompt, temperature, max_tokens=5):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip().lower()


def _parse_result(result):
    if result.startswith("yes") or result == "1":
        return 1
    elif result.startswith("no") or result == "0":
        return 0
    return 1 if "yes" in result or "adopt" in result else 0


def classify_texts(client_info, texts, model, temperature=0.0, max_tokens=5, metadata_list=None):
    """Classify draft resolutions. metadata_list: optional list of {date, authors} dicts."""
    provider, client = client_info
    results = []
    for i, text in enumerate(tqdm(texts, desc="Classifying")):
        meta = (metadata_list[i] if metadata_list and i < len(metadata_list) else {}) or {}
        meta_str = ""
        if meta.get("date"):
            meta_str += f"\nDate: {meta['date']}"
        if meta.get("authors"):
            meta_str += f"\nSponsoring countries: {meta['authors']}"
        if meta_str:
            meta_str = "\nContext:" + meta_str + "\n"

        user_prompt = f"""
The provided document is a United Nations Security Council's draft resolution. Predict whether the draft resolution will be adopted or not. Consider: adoption requires at least 9 votes in favour and no veto from permanent members (China, France, Russia, UK, US). Answer with 'yes' (1) or 'no' (0) without any explanation.
{meta_str}
Text: "{text[:8000]}"

Answer:
"""
        try:
            result = _call_openai(client, model, user_prompt, temperature, max_tokens)
            results.append(_parse_result(result))
        except Exception as e:
            print(f"Error: {e}")
            results.append(0)
    return results


def calculate_metrics(pred, labels, verbose=True):
    """Compute metrics (matches UNBench paper: Accuracy, Bal. ACC, AUC, F1, etc.)."""
    pred = [1 - x for x in pred]
    labels = [1 - x for x in labels]
    acc = accuracy_score(labels, pred)
    try:
        roc_auc = roc_auc_score(labels, pred)
    except ValueError:
        roc_auc = 0
    balanced_acc = balanced_accuracy_score(labels, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    macro_f1 = precision_recall_fscore_support(labels, pred, average='macro', zero_division=0)[2]
    precision, recall, _ = precision_recall_curve(labels, pred)
    from sklearn.metrics import auc
    pr_auc = auc(recall, precision)
    mcc = matthews_corrcoef(labels, pred)
    g_mean = geometric_mean_score(labels, pred)
    cm = confusion_matrix(labels, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    if verbose:
        print("\n=== Metrics (UNBench Task 3) ===")
        print(f'Accuracy: {acc:.4f}')
        print(f'AUC: {roc_auc:.4f}')
        print(f'Balanced Accuracy: {balanced_acc:.4f}')
        print(f'Macro F1: {macro_f1:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall: {rec:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'PR AUC: {pr_auc:.4f}')
        print(f'MCC: {mcc:.4f}')
        print(f'G-Mean: {g_mean:.4f}')
        print(f'Specificity: {specificity:.4f}')
    return {
        "accuracy": acc, "auc": roc_auc, "balanced_acc": balanced_acc,
        "macro_f1": macro_f1, "f1": f1, "precision": prec, "recall": rec,
        "pr_auc": pr_auc, "mcc": mcc, "g_mean": g_mean, "specificity": specificity
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "deepseek", "qwen"], default="openai")
    parser.add_argument("--model", default=None, help="Model name (auto if not set)")
    parser.add_argument("--base-url", default=None, help="API base URL (for DeepSeek)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--data", default="data/task3.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1, help="Number of runs for mean/variance (temp>0)")
    parser.add_argument("--metadata", action="store_true", help="Use metadata if available in data")
    parser.add_argument("--max-tokens", type=int, default=5, help="Max tokens for LLM output (modification experiment)")
    parser.add_argument("--output-json", default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    model = args.model or {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "qwen": "qwen-plus",
    }[args.provider]

    base_url = _get_base_url(args.provider, args.base_url)

    data_path = Path(__file__).parent / args.data
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return 1

    with open(data_path) as f:
        data = json.load(f)
    texts = data["drafts"]
    labels = data["labels"]
    metadata_list = data.get("metadata") if args.metadata else None
    if args.max_samples:
        texts = texts[:args.max_samples]
        labels = labels[:args.max_samples]
        if metadata_list:
            metadata_list = metadata_list[:args.max_samples]

    print(f"Provider: {args.provider}, Model: {model}, temp={args.temperature}, max_tokens={args.max_tokens}, n={len(texts)}, trials={args.trials}")

    client_info = get_client(args.provider, base_url=base_url)
    all_metrics = []
    for trial in range(args.trials):
        if args.trials > 1:
            print(f"\n--- Trial {trial+1}/{args.trials} ---")
        pred = classify_texts(client_info, texts, model, args.temperature, args.max_tokens, metadata_list)
        m = calculate_metrics(pred, labels, verbose=(args.trials == 1))
        all_metrics.append(m)

    if args.trials > 1:
        print("\n=== Mean ± Std across trials ===")
        for k in ["accuracy", "balanced_acc", "macro_f1", "auc"]:
            vals = [x[k] for x in all_metrics]
            print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if args.output_json:
        out = {
            "provider": args.provider, "model": model, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "n_samples": len(texts), "trials": args.trials,
            "metrics": all_metrics[-1] if all_metrics else {},
            "mean_std": {k: {"mean": float(np.mean([x[k] for x in all_metrics])), "std": float(np.std([x[k] for x in all_metrics]))}
                        for k in ["accuracy", "balanced_acc", "macro_f1", "auc"]} if args.trials > 1 else {}
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved metrics to {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
