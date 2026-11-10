"""
實驗 1：迷你版 Table 4 ——「不確定就不採用」到底有沒有用？

對應論文：Robust Vision-Based Cheat Detection in Competitive Gaming
          §5.3 Avoiding false positives、Table 4
論文做法：只有在偵測器「有信心」時才檢舉作弊，用來避免冤枉正常玩家。

本實驗：用不確定度當門檻，不確定度 > 0.5 就「不採用」這次預測，然後看：
  1. 被採用的預測，是不是比較準？
  2. 被擋下來的，是不是剛好就是模型會答錯的？
  3. 對照組（softmax，用 1 − 最大機率 當不確定度）和實驗組（EDL，用 u）
     誰比較會「知道自己不知道」？

測試資料：
  - 乾淨 MNIST 測試集（跟訓練資料同分布）
  - 旋轉 MNIST：每張圖隨機旋轉 0°～180°（模擬沒見過的新情境）

執行：python exp1_threshold.py
輸出：results/exp1_threshold.png，以及終端機上的表格
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from exp_common import (
    COLOR_CORRECT, COLOR_EDL, COLOR_SOFTMAX, COLOR_WRONG, DEFAULT_U_THRESHOLD,
    TEXT_SECONDARY, accuracy_coverage_curve, error_detection_auroc, evaluate_both,
    fmt_num, fmt_pct, get_device, load_both_models, make_rotation_fn,
    pixel_space_from_loader, print_table, safe_mean, save_figure, setup_plot_style,
)


def confidence_breakdown(correct, uncertainty, threshold):
    """仿照論文 Table 4，把預測分成「有信心／沒信心」×「答對／答錯」四格。"""
    correct = np.asarray(correct, dtype=bool)
    confident = np.asarray(uncertainty) <= threshold
    return {
        "conf_right": int((confident & correct).sum()),
        "conf_wrong": int((confident & ~correct).sum()),   # 最危險：有信心卻答錯
        "unsure_right": int((~confident & correct).sum()),
        "unsure_wrong": int((~confident & ~correct).sum()),
        "coverage": float(confident.mean()),
        "acc_accepted": safe_mean(correct[confident]),
        "acc_rejected": safe_mean(correct[~confident]),
    }


def analyze(result, threshold):
    """回傳兩個模型的統計結果。不確定度：softmax = 1 − 最大機率；EDL = u。"""
    y = result["labels"]
    stats = {}
    for name, model_result, uncertainty in [
        ("softmax", result["softmax"], 1.0 - result["softmax"]["conf"]),
        ("edl", result["edl"], result["edl"]["u"]),
    ]:
        correct = model_result["pred"] == y
        s = confidence_breakdown(correct, uncertainty, threshold)
        s["accuracy"] = float(correct.mean())
        s["auroc"] = error_detection_auroc(uncertainty, ~correct)
        s["correct"] = correct
        s["uncertainty"] = uncertainty
        stats[name] = s
    return stats


def print_results(set_name, stats, threshold):
    print(f"\n=== {set_name}（門檻：不確定度 > {threshold} 就不採用）===")
    headers = ["模型", "準確率", "信心高-對", "信心高-錯", "信心低-對", "信心低-錯",
               "採用比例", "採用準確率", "擋下準確率", "AUROC"]
    rows = []
    for name, label in [("softmax", "對照組 softmax"), ("edl", "實驗組 EDL")]:
        s = stats[name]
        rows.append([
            label, fmt_pct(s["accuracy"]), s["conf_right"], s["conf_wrong"],
            s["unsure_right"], s["unsure_wrong"], fmt_pct(s["coverage"]),
            fmt_pct(s["acc_accepted"]), fmt_pct(s["acc_rejected"]), fmt_num(s["auroc"]),
        ])
    print_table(headers, rows)


def plot_results(all_stats, rotated_name, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("實驗 1：不確定就不採用（迷你版 Table 4）", fontsize=15, fontweight="bold")

    # (a) 準確率 vs 採用比例：只採用最有信心的前 x%，準確率多少
    ax = axes[0, 0]
    line_styles = ["-", "--"]
    for (set_name, stats), style in zip(all_stats.items(), line_styles):
        for name, label, color in [("edl", "EDL", COLOR_EDL), ("softmax", "softmax", COLOR_SOFTMAX)]:
            s = stats[name]
            cov, acc = accuracy_coverage_curve(1.0 - s["uncertainty"], s["correct"])
            ax.plot(cov * 100, acc * 100, style, color=color, label=f"{label}｜{set_name}")
    ax.set_title("(a) 只採用最有信心的前 x%，準確率是多少？")
    ax.set_xlabel("採用比例（%）")
    ax.set_ylabel("被採用部分的準確率（%）")
    ax.legend(fontsize=8.5, loc="best")

    # (b) EDL 門檻掃描（旋轉資料）
    ax = axes[0, 1]
    s = all_stats[rotated_name]["edl"]
    thresholds = np.linspace(0.05, 1.0, 20)
    acc_acc, acc_rej, coverage = [], [], []
    for t in thresholds:
        b = confidence_breakdown(s["correct"], s["uncertainty"], t)
        acc_acc.append(b["acc_accepted"])
        acc_rej.append(b["acc_rejected"])
        coverage.append(b["coverage"])
    ax.plot(thresholds, np.array(acc_acc) * 100, "-o", color=COLOR_EDL, markersize=5, label="被採用的準確率")
    ax.plot(thresholds, np.array(acc_rej) * 100, "-s", color=COLOR_WRONG, markersize=5, label="被擋下的準確率")
    ax.plot(thresholds, np.array(coverage) * 100, "--", color=TEXT_SECONDARY, label="採用比例")
    ax.axvline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.text(threshold, 3, f" 門檻 {threshold}", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"(b) EDL 門檻掃描｜{rotated_name}")
    ax.set_xlabel("門檻：u 大於此值就不採用")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=9, loc="center left")

    # (c)(d) 答對 vs 答錯的不確定度分布（旋轉資料）
    bins = np.linspace(0, 1, 21)
    for ax, name, title in [
        (axes[1, 0], "edl", "(c) EDL：u 的分布"),
        (axes[1, 1], "softmax", "(d) softmax：1 - 最大機率 的分布"),
    ]:
        s = all_stats[rotated_name][name]
        unc, correct = s["uncertainty"], s["correct"]
        for mask, color, label in [(correct, COLOR_CORRECT, "答對"), (~correct, COLOR_WRONG, "答錯")]:
            ax.hist(unc[mask], bins=bins, color=color, alpha=0.35, histtype="stepfilled")
            ax.hist(unc[mask], bins=bins, color=color, histtype="step", linewidth=2,
                    label=f"{label}（{mask.sum()} 張）")
        ax.axvline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax.set_title(f"{title}｜{rotated_name}")
        ax.set_xlabel("不確定度（越右邊越不確定）")
        ax.set_ylabel("圖片張數")
        ax.legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, "exp1_threshold.png")


def main():
    parser = argparse.ArgumentParser(description="實驗 1：不確定度門檻（迷你版 Table 4）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_U_THRESHOLD,
                        help="不確定度大於此值就不採用（預設 0.5）")
    parser.add_argument("--seed", type=int, default=0, help="隨機旋轉角度的亂數種子")
    args = parser.parse_args()

    setup_plot_style()
    device = get_device()
    from data import dataloaders  # 匯入時會載入 MNIST

    val_loader = dataloaders["val"]
    space = pixel_space_from_loader(val_loader)
    baseline, edl = load_both_models(device)

    clean_name = "乾淨 MNIST"
    rotated_name = "旋轉 MNIST（0°～180°）"
    print("評估中：乾淨 MNIST ...")
    clean = evaluate_both(baseline, edl, val_loader, device)
    print("評估中：旋轉 MNIST ...")
    rotated = evaluate_both(baseline, edl, val_loader, device,
                            transform_fn=make_rotation_fn(space, seed=args.seed))

    all_stats = {clean_name: analyze(clean, args.threshold),
                 rotated_name: analyze(rotated, args.threshold)}
    for set_name, stats in all_stats.items():
        print_results(set_name, stats, args.threshold)

    plot_results(all_stats, rotated_name, args.threshold)

    r = all_stats[rotated_name]
    print("\n--- 解讀提示 ---")
    print(f"「信心高-錯」最危險：模型很有信心卻答錯，對應論文 Table 4 的 FP（冤枉玩家）。"
          f"旋轉資料上：softmax {r['softmax']['conf_wrong']} 張、EDL {r['edl']['conf_wrong']} 張。")
    print(f"AUROC：不確定度能不能分辨對錯，越接近 1 越好（0.5 = 亂猜）。"
          f"旋轉資料上：softmax {fmt_num(r['softmax']['auroc'])}、EDL {fmt_num(r['edl']['auroc'])}。")


if __name__ == "__main__":
    main()
