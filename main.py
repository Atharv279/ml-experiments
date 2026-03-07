#!/usr/bin/env python3
"""ML Experiments — LightGBM training loops on mock data."""
import json, os, random, math, datetime, hashlib

EXPERIMENTS = [
    {"name": "churn_prediction", "features": 12, "target": "binary", "metric": "auc"},
    {"name": "price_regression", "features": 20, "target": "continuous", "metric": "rmse"},
    {"name": "fraud_detection", "features": 15, "target": "binary", "metric": "auc"},
    {"name": "demand_forecast", "features": 18, "target": "continuous", "metric": "mae"},
]

HYPERPARAM_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 5, 7, -1],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0, 0.1, 0.5, 1.0],
}

def generate_mock_dataset(n_samples, n_features):
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "train_size": int(n_samples * 0.8),
        "test_size": int(n_samples * 0.2),
        "feature_stats": {
            f"f{i}": {
                "mean": round(random.gauss(0, 1), 4),
                "std": round(abs(random.gauss(1, 0.3)), 4),
                "null_pct": round(random.uniform(0, 0.05), 4),
            }
            for i in range(n_features)
        },
    }

def sample_hyperparams():
    return {k: random.choice(v) for k, v in HYPERPARAM_SPACE.items()}

def simulate_training(experiment, hyperparams, dataset):
    base_performance = random.uniform(0.65, 0.95)

    lr_bonus = 0.02 if hyperparams["learning_rate"] <= 0.05 else -0.01
    leaf_bonus = 0.01 if hyperparams["num_leaves"] in [31, 63] else -0.005
    reg_bonus = 0.01 if hyperparams["reg_alpha"] > 0 else 0
    performance = min(base_performance + lr_bonus + leaf_bonus + reg_bonus + random.gauss(0, 0.02), 0.99)

    train_time = round(random.uniform(0.5, 30.0) * (hyperparams["n_estimators"] / 100), 2)
    train_metric = round(performance + random.uniform(0.01, 0.05), 4)
    test_metric = round(performance - random.uniform(0, 0.03), 4)

    if experiment["metric"] in ("rmse", "mae"):
        train_metric = round(random.uniform(0.1, 2.0), 4)
        test_metric = round(train_metric * random.uniform(1.0, 1.3), 4)

    feature_importance = {}
    for i in range(experiment["features"]):
        feature_importance[f"f{i}"] = round(random.uniform(0, 1), 4)
    total = sum(feature_importance.values())
    feature_importance = {k: round(v / total, 4) for k, v in feature_importance.items()}

    return {
        "train_metric": train_metric,
        "test_metric": test_metric,
        "train_time_seconds": train_time,
        "overfit_gap": round(abs(train_metric - test_metric), 4),
        "feature_importance_top5": dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:5]),
    }

def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    all_results = []
    for exp in EXPERIMENTS:
        n_trials = random.randint(3, 6)
        dataset = generate_mock_dataset(random.randint(5000, 50000), exp["features"])
        trials = []

        for t in range(n_trials):
            hp = sample_hyperparams()
            result = simulate_training(exp, hp, dataset)
            trials.append({"trial": t + 1, "hyperparams": hp, "results": result})

        if exp["metric"] in ("rmse", "mae"):
            best = min(trials, key=lambda x: x["results"]["test_metric"])
        else:
            best = max(trials, key=lambda x: x["results"]["test_metric"])

        all_results.append({
            "experiment": exp["name"],
            "metric": exp["metric"],
            "target_type": exp["target"],
            "dataset": dataset,
            "trials": trials,
            "best_trial": best["trial"],
            "best_score": best["results"]["test_metric"],
        })

    report = {
        "timestamp": now.isoformat(),
        "run_id": hashlib.sha256(now.isoformat().encode()).hexdigest()[:10],
        "experiments": all_results,
        "summary": {
            "total_experiments": len(all_results),
            "total_trials": sum(len(r["trials"]) for r in all_results),
            "best_performers": {r["experiment"]: r["best_score"] for r in all_results},
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{date_str}.json", "w") as f:
        json.dump(report, f, indent=2)

    md = [f"# ML Experiments Report — {date_str}\n"]
    md.append(f"**Run ID:** `{report['run_id']}` | **Experiments:** {report['summary']['total_experiments']} | **Total Trials:** {report['summary']['total_trials']}\n")
    for r in all_results:
        md.append(f"## {r['experiment']} ({r['metric'].upper()})\n")
        md.append(f"**Dataset:** {r['dataset']['n_samples']} samples, {r['dataset']['n_features']} features | **Best Score:** {r['best_score']}\n")
        md.append(f"| Trial | {r['metric'].upper()} (test) | Overfit Gap | Train Time | LR | Estimators | Leaves |")
        md.append(f"|-------|{'------|' * 6}")
        for t in r["trials"]:
            hp = t["hyperparams"]
            res = t["results"]
            marker = " *" if t["trial"] == r["best_trial"] else ""
            md.append(f"| {t['trial']}{marker} | {res['test_metric']} | {res['overfit_gap']} | {res['train_time_seconds']}s | {hp['learning_rate']} | {hp['n_estimators']} | {hp['num_leaves']} |")
        best_trial = next(t for t in r["trials"] if t["trial"] == r["best_trial"])
        md.append(f"\n**Top Features:** {', '.join(f'{k} ({v})' for k, v in best_trial['results']['feature_importance_top5'].items())}\n")

    with open(f"logs/{date_str}.md", "w") as f:
        f.write("\n".join(md))

    print(f"[ml-experiments] Report generated: logs/{date_str}.md")

if __name__ == "__main__":
    main()
