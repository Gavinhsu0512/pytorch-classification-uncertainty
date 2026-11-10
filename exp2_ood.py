"""
實驗 2：模擬「出現新外掛」——模型能察覺自己該重新訓練了嗎？

對應論文：Robust Vision-Based Cheat Detection in Competitive Gaming
          §5.4 Detecting the need for re-training、Table 5
論文做法：
  LR（likelihood ratio）= 主導類別的 α ÷ 另一類的 α（Eq. 8）
  先在訓練資料上算出平均 LR 當基準；部署後再算實際資料的平均 LR。
  只要 LR 掉到基準的 50% 以下，就發出「需要重新訓練」的警報。

本實驗：
  論文是二分類，LR = max(α) / min(α)。MNIST 有 10 類，我們推廣成
  「第一名的 α ÷ 第二名的 α」，二分類時兩者完全相同。
  把模型沒看過的資料當成「新外掛」：
    - 旋轉 90° 的 MNIST ：同一款遊戲，但畫面變了
    - Fashion-MNIST     ：換了一款遊戲（衣服、鞋子、包包）
    - 隨機雜訊          ：完全沒有意義的畫面
  另外也比較對照組 softmax：它會不會對沒看過的東西照樣很有信心？

執行：python exp2_ood.py
輸出：results/exp2_ood.png、results/exp2_fashion_examples.png，以及終端機上的表格
注意：第一次執行會自動下載 Fashion-MNIST（約 30 MB）到 data/FashionMNIST/
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from exp_common import (
    COLOR_EDL, COLOR_SOFTMAX, DEFAULT_U_THRESHOLD, TEXT_PRIMARY, TEXT_SECONDARY,
    edl_outputs, evaluate_both, fmt_num, fmt_pct, get_device, get_transform,
    load_both_models, make_noise_fn, make_rotation_fn, pixel_space_from_loader,
    print_table, save_figure, setup_plot_style, softmax_outputs,
)

FASHION_CLASSES = ["T恤", "褲子", "套頭衫", "洋裝", "外套", "涼鞋", "襯衫", "運動鞋", "包包", "短靴"]
RETRAIN_RATIO = 0.5  # 論文的規則：LR 掉到訓練時的 50% 以下就重新訓練


def load_fashion_loader(transform, batch_size):
    dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def summarize(result, has_labels):
    y = result["labels"]
    sm, ed = result["softmax"], result["edl"]
    return {
        "edl_mean_u": float(ed["u"].mean()),
        "edl_mean_lr": float(ed["lr"].mean()),
        "edl_confident_rate": float((ed["u"] <= DEFAULT_U_THRESHOLD).mean()),
        "softmax_mean_conf": float(sm["conf"].mean()),
        "softmax_very_confident_rate": float((sm["conf"] >= 0.9).mean()),
        "softmax_acc": float((sm["pred"] == y).mean()) if has_labels else float("nan"),
        "edl_acc": float((ed["pred"] == y).mean()) if has_labels else float("nan"),
        "edl_u": ed["u"],
        "softmax_unc": 1.0 - sm["conf"],
    }


def print_fashion_breakdown(fashion_result):
    """Fashion-MNIST 每一類衣服，被模型認成哪個數字？"""
    y = fashion_result["labels"]
    sm, ed = fashion_result["softmax"], fashion_result["edl"]
    rows = []
    for c, name in enumerate(FASHION_CLASSES):
        mask = y == c
        preds = sm["pred"][mask]
        digit = int(np.bincount(preds, minlength=10).argmax())
        share = float((preds == digit).mean())
        rows.append([name, digit, fmt_pct(share), fmt_pct(float(sm["conf"][mask].mean())),
                     fmt_num(float(ed["u"][mask].mean()))])
    print("\n=== Fashion-MNIST：每一類衣服被認成哪個數字？ ===")
    print_table(["衣服類別", "softmax 最常認成", "占比", "softmax 平均信心", "EDL 平均 u"], rows)


def plot_ood(summaries, train_lr):
    names = list(summaries.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("實驗 2：遇到沒看過的資料，模型知道嗎？（迷你版 Table 5）", fontsize=15, fontweight="bold")

    # (a) 兩個模型自己回報的不確定度（盒狀圖）
    ax = axes[0]
    positions = np.arange(len(names))
    width = 0.34
    for offset, key, color, label in [
        (-width / 2 - 0.02, "edl_u", COLOR_EDL, "EDL：u"),
        (width / 2 + 0.02, "softmax_unc", COLOR_SOFTMAX, "softmax：1 - 最大機率"),
    ]:
        data = [summaries[n][key] for n in names]
        box = ax.boxplot(data, positions=positions + offset, widths=width, patch_artist=True,
                         showfliers=False, medianprops={"color": TEXT_PRIMARY, "linewidth": 2})
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
        for part in ("whiskers", "caps"):
            for line in box[part]:
                line.set_color(color)
        ax.plot([], [], color=color, linewidth=8, alpha=0.45, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("模型回報的不確定度（越高越不確定）")
    ax.set_title("(a) 各資料集上，模型覺得自己有多不確定？")
    ax.legend(fontsize=9, loc="upper left")

    # (b) LR 相對訓練時的比例 → 是否需要重新訓練
    ax = axes[1]
    field_names = [n for n in names if n != "訓練資料（基準）"]
    ratios = [summaries[n]["edl_mean_lr"] / train_lr for n in field_names]
    bars = ax.bar(np.arange(len(field_names)), ratios, color=COLOR_EDL, width=0.55)
    ax.axhline(RETRAIN_RATIO, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(len(field_names) - 0.5, RETRAIN_RATIO, "  重新訓練門檻（50%）", va="bottom", ha="right",
            color=TEXT_SECONDARY, fontsize=9)
    for bar, ratio in zip(bars, ratios):
        verdict = "需要重新訓練" if ratio < RETRAIN_RATIO else "正常"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio:.2f}\n{verdict}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xticks(np.arange(len(field_names)))
    ax.set_xticklabels(field_names, fontsize=9.5)
    ax.set_ylim(0, max(1.2, max(ratios) + 0.3))
    ax.set_ylabel("平均 LR ÷ 訓練時的平均 LR")
    ax.set_title("(b) EDL 的 LR 比例：該重新訓練了嗎？")

    fig.tight_layout()
    return save_figure(fig, "exp2_ood.png")


@torch.no_grad()
def plot_fashion_examples(baseline, edl, fashion_loader, space, device):
    """每一類衣服挑一張，看兩個模型怎麼說。"""
    dataset = fashion_loader.dataset
    picks = {}
    for i in range(len(dataset)):
        label = int(dataset.targets[i])
        if label not in picks:
            picks[label] = i
        if len(picks) == len(FASHION_CLASSES):
            break

    fig, axes = plt.subplots(2, 5, figsize=(13, 6.8))
    fig.suptitle("Fashion-MNIST：把衣服丟給只學過數字的模型", fontsize=14, fontweight="bold")
    for ax, c in zip(axes.ravel(), range(len(FASHION_CLASSES))):
        x, _ = dataset[picks[c]]
        x = x.unsqueeze(0).to(device)
        b = softmax_outputs(baseline(x))
        e = edl_outputs(edl(x))
        image = space.to_pixel(x)[0, 0].cpu().numpy()
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{FASHION_CLASSES[c]}\n"
            f"softmax：{int(b['pred'][0])}（{float(b['conf'][0]) * 100:.0f}%）\n"
            f"EDL：{int(e['pred'][0])}，u = {float(e['u'][0]):.2f}",
            fontsize=10, fontweight="normal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    return save_figure(fig, "exp2_fashion_examples.png")


def main():
    parser = argparse.ArgumentParser(description="實驗 2：分布外資料與重新訓練警報")
    parser.add_argument("--train-samples", type=int, default=10000,
                        help="計算訓練基準時使用的訓練圖片數（預設 10000）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_plot_style()
    device = get_device()
    from data import dataloaders  # 匯入時會載入 MNIST

    val_loader = dataloaders["val"]
    train_loader = dataloaders["train"] if "train" in dataloaders else val_loader
    space = pixel_space_from_loader(val_loader)
    baseline, edl = load_both_models(device)

    print("下載／載入 Fashion-MNIST ...")
    fashion_loader = load_fashion_loader(get_transform(val_loader), val_loader.batch_size or 1000)

    print("評估中（共 5 組資料）...")
    results = {
        "訓練資料（基準）": (evaluate_both(baseline, edl, train_loader, device,
                                         max_samples=args.train_samples), True),
        "MNIST 測試集": (evaluate_both(baseline, edl, val_loader, device), True),
        "旋轉 90° MNIST": (evaluate_both(baseline, edl, val_loader, device,
                                       transform_fn=make_rotation_fn(space, fixed_angle=90)), True),
        "Fashion-MNIST": (evaluate_both(baseline, edl, fashion_loader, device), False),
        "隨機雜訊": (evaluate_both(baseline, edl, val_loader, device,
                                transform_fn=make_noise_fn(space, seed=args.seed)), False),
    }
    summaries = {name: summarize(r, has_labels) for name, (r, has_labels) in results.items()}
    train_lr = summaries["訓練資料（基準）"]["edl_mean_lr"]

    print("\n=== 實驗組 EDL：仿照論文 Table 5 的重新訓練警報 ===")
    rows = []
    for name, s in summaries.items():
        ratio = s["edl_mean_lr"] / train_lr
        verdict = "（基準）" if name == "訓練資料（基準）" else (
            "需要重新訓練" if ratio < RETRAIN_RATIO else "正常")
        rows.append([name, fmt_pct(s["edl_acc"]), fmt_num(s["edl_mean_u"]), fmt_num(s["edl_mean_lr"], 2),
                     fmt_num(ratio, 2), fmt_pct(s["edl_confident_rate"]), verdict])
    print_table(["資料", "準確率", "平均 u", "平均 LR", "LR 比例", "有信心比例（u≤0.5）", "判斷"], rows)

    print("\n=== 對照組 softmax：對沒看過的資料，它還是很有信心嗎？ ===")
    rows = []
    for name, s in summaries.items():
        rows.append([name, fmt_pct(s["softmax_acc"]), fmt_pct(s["softmax_mean_conf"]),
                     fmt_pct(s["softmax_very_confident_rate"])])
    print_table(["資料", "準確率", "平均最大機率", "非常有信心比例（≥90%）"], rows)

    print_fashion_breakdown(results["Fashion-MNIST"][0])

    plot_ood(summaries, train_lr)
    plot_fashion_examples(baseline, edl, fashion_loader, space, device)

    print("\n--- 解讀提示 ---")
    print("LR 比例 < 0.5 → 仿照論文規則發出「需要重新訓練」警報。MNIST 測試集應該是「正常」，")
    print("其他沒看過的資料若被判「需要重新訓練」，代表模型能自己察覺環境變了。")
    print("softmax 的「非常有信心比例」若在 Fashion-MNIST 或雜訊上還很高，代表它對沒看過的東西照樣很自信。")


if __name__ == "__main__":
    main()
