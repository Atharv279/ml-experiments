#!/usr/bin/env python3
"""ML Experiments — LightGBM training simulator with loss curves and visual analytics."""
import json, os, random, math, datetime, hashlib, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = [
    {"name": "churn_prediction", "features": 12, "target": "binary", "metric": "auc"},
    {"name": "price_regression", "features": 20, "target": "continuous", "metric": "rmse"},
    {"name": "fraud_detection", "features": 15, "target": "binary", "metric": "auc"},
    {"name": "demand_forecast", "features": 18, "target": "continuous", "metric": "mae"},
]

HYPERPARAM_SPACE = {
    "num_leaves": [15, 31, 63, 127], "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 1000], "max_depth": [3, 5, 7, -1],
    "min_child_samples": [5, 10, 20, 50], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0], "reg_lambda": [0, 0.1, 0.5, 1.0],
}

def sample_hyperparams():
    return {k: random.choice(v) for k, v in HYPERPARAM_SPACE.items()}

def generate_loss_curve(n_estimators, lr, is_lower_better):
    train_curve, val_curve = [], []
    base = random.uniform(0.5, 2.0) if is_lower_better else random.uniform(0.3, 0.6)
    for i in range(min(n_estimators, 100)):
        decay = math.exp(-lr * i * 0.5)
        noise_t = random.gauss(0, 0.005)
        noise_v = random.gauss(0, 0.01)
        if is_lower_better:
            train_curve.append(round(base * decay + noise_t, 4))
            val_curve.append(round(base * decay * random.uniform(1.0, 1.2) + noise_v, 4))
        else:
            train_curve.append(round(1 - base * decay + noise_t, 4))
            val_curve.append(round(1 - base * decay * random.uniform(1.0, 1.2) + noise_v, 4))
    return train_curve, val_curve

def simulate_training(experiment, hyperparams):
    is_lower_better = experiment["metric"] in ("rmse", "mae")
    train_curve, val_curve = generate_loss_curve(hyperparams["n_estimators"], hyperparams["learning_rate"], is_lower_better)

    test_metric = val_curve[-1] if val_curve else 0.5
    train_metric = train_curve[-1] if train_curve else 0.5
    train_time = round(random.uniform(0.5, 30.0) * (hyperparams["n_estimators"] / 100), 2)

    feature_importance = {}
    for i in range(experiment["features"]):
        feature_importance[f"f{i}"] = round(random.uniform(0, 1), 4)
    total = sum(feature_importance.values()) or 1
    feature_importance = {k: round(v / total, 4) for k, v in feature_importance.items()}

    return {
        "train_metric": train_metric, "test_metric": test_metric,
        "train_time_seconds": train_time, "overfit_gap": round(abs(train_metric - test_metric), 4),
        "train_curve": train_curve, "val_curve": val_curve,
        "feature_importance_top5": dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:5]),
    }

def load_yesterday(date_str):
    yesterday = (datetime.datetime.strptime(date_str, "%Y-%m-%d") - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    path = f"logs/{yesterday}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def compute_delta(today_results, yesterday_data):
    if not yesterday_data:
        return {"status": "no_previous_data"}
    y_map = {e["experiment"]: e["best_score"] for e in yesterday_data.get("experiments", [])}
    deltas = {}
    for r in today_results:
        y_score = y_map.get(r["experiment"])
        if y_score is not None:
            change = round(((r["best_score"] - y_score) / max(abs(y_score), 0.001)) * 100, 1)
            deltas[r["experiment"]] = {"today": r["best_score"], "yesterday": y_score, "change_pct": change}
    return {"status": "compared", "deltas": deltas}

def generate_charts(all_results, date_str):
    n_exp = len(all_results)
    fig, axes = plt.subplots(2, n_exp, figsize=(5 * n_exp, 8))
    if n_exp == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle(f"ML Experiments Dashboard — {date_str}", fontsize=14, fontweight="bold")

    colors = plt.cm.Set2.colors

    for i, r in enumerate(all_results):
        best_trial = next(t for t in r["trials"] if t["trial"] == r["best_trial"])
        train_c = best_trial["results"]["train_curve"]
        val_c = best_trial["results"]["val_curve"]

        # Row 1: Loss curves
        axes[0][i].plot(train_c, label="Train", color="#3498db", linewidth=1.5)
        axes[0][i].plot(val_c, label="Val", color="#e74c3c", linewidth=1.5)
        axes[0][i].fill_between(range(len(train_c)), train_c, val_c, alpha=0.1, color="#e74c3c")
        axes[0][i].set_title(f"{r['experiment']}\n({r['metric'].upper()})", fontsize=10)
        axes[0][i].set_xlabel("Iteration")
        axes[0][i].legend(fontsize=8)
        axes[0][i].grid(True, alpha=0.3)

        # Row 2: Feature importance
        fi = best_trial["results"]["feature_importance_top5"]
        axes[1][i].barh(list(fi.keys()), list(fi.values()), color=colors[i % len(colors)])
        axes[1][i].set_xlabel("Importance")
        axes[1][i].set_title("Top 5 Features", fontsize=10)

    plt.tight_layout()
    path = f"logs/{date_str}_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # Historical best scores trend
    history_files = sorted(glob.glob("logs/*.json"))[-14:]
    if len(history_files) >= 2:
        fig2, ax = plt.subplots(figsize=(12, 4))
        exp_trends = {e["name"]: {"dates": [], "scores": []} for e in EXPERIMENTS}
        for hf in history_files:
            with open(hf) as f:
                h = json.load(f)
            d = os.path.basename(hf).replace(".json", "")
            for exp in h.get("experiments", []):
                if exp["experiment"] in exp_trends:
                    exp_trends[exp["experiment"]]["dates"].append(d)
                    exp_trends[exp["experiment"]]["scores"].append(exp["best_score"])
        for name, data in exp_trends.items():
            if data["dates"]:
                ax.plot(data["dates"], data["scores"], "o-", label=name, linewidth=2)
        ax.set_ylabel("Best Score")
        ax.set_title("14-Day Experiment Score Trend")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"logs/{date_str}_trend.png", dpi=150, bbox_inches="tight")
        plt.close()

    return path

def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    all_results = []
    for exp in EXPERIMENTS:
        n_trials = random.randint(3, 6)
        trials = []
        for t in range(n_trials):
            hp = sample_hyperparams()
            result = simulate_training(exp, hp)
            trials.append({"trial": t + 1, "hyperparams": hp, "results": result})
        is_lower = exp["metric"] in ("rmse", "mae")
        best = min(trials, key=lambda x: x["results"]["test_metric"]) if is_lower else max(trials, key=lambda x: x["results"]["test_metric"])
        all_results.append({"experiment": exp["name"], "metric": exp["metric"], "target_type": exp["target"],
                            "trials": trials, "best_trial": best["trial"], "best_score": best["results"]["test_metric"]})

    yesterday = load_yesterday(date_str)
    delta = compute_delta(all_results, yesterday)

    report = {
        "timestamp": now.isoformat(), "run_id": hashlib.sha256(now.isoformat().encode()).hexdigest()[:10],
        "experiments": all_results, "delta": delta,
        "summary": {"total_experiments": len(all_results), "total_trials": sum(len(r["trials"]) for r in all_results),
                     "best_performers": {r["experiment"]: r["best_score"] for r in all_results}},
    }

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{date_str}.json", "w") as f:
        json.dump(report, f, indent=2)

    chart_path = generate_charts(all_results, date_str)

    md = [f"# ML Experiments Report — {date_str}\n"]
    md.append(f"**Run ID:** `{report['run_id']}` | **Experiments:** {report['summary']['total_experiments']} | **Trials:** {report['summary']['total_trials']}\n")
    md.append(f"![Dashboard]({os.path.basename(chart_path)})\n")
    if os.path.exists(f"logs/{date_str}_trend.png"):
        md.append(f"![Trend]({date_str}_trend.png)\n")
    if delta.get("status") == "compared":
        md.append("## Delta vs Yesterday\n")
        md.append("| Experiment | Today | Yesterday | Change |")
        md.append("|-----------|-------|-----------|--------|")
        for exp, d in delta["deltas"].items():
            arrow = "📈" if d["change_pct"] > 0 else "📉"
            md.append(f"| {exp} | {d['today']} | {d['yesterday']} | {arrow} {d['change_pct']}% |")
        md.append("")
    for r in all_results:
        md.append(f"## {r['experiment']} ({r['metric'].upper()})\n")
        md.append(f"**Best Score:** {r['best_score']} (Trial {r['best_trial']})\n")
        md.append(f"| Trial | Score | Overfit Gap | Time | LR | Trees | Leaves |")
        md.append(f"|-------|-------|-------------|------|-----|-------|--------|")
        for t in r["trials"]:
            hp, res = t["hyperparams"], t["results"]
            marker = " ⭐" if t["trial"] == r["best_trial"] else ""
            md.append(f"| {t['trial']}{marker} | {res['test_metric']} | {res['overfit_gap']} | {res['train_time_seconds']}s | {hp['learning_rate']} | {hp['n_estimators']} | {hp['num_leaves']} |")
        md.append("")

    with open(f"logs/{date_str}.md", "w") as f:
        f.write("\n".join(md))
    print(f"[ml-experiments] v2.0 report + charts generated")

if __name__ == "__main__":
    main()
