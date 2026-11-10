"""
實驗 3：FGSM 對抗攻擊——作弊者能騙過偵測器嗎？

對應論文：Robust Vision-Based Cheat Detection in Competitive Gaming
          §6 Adversarial attack、Fig. 2、Fig. 4、Table 6
論文擔心：作弊軟體可能在畫面上加一層「肉眼幾乎看不出」的雜訊，騙偵測器把作弊畫面判成正常。

FGSM（Fast Gradient Sign Method）：
    x_adv = clip( x + ε · sign(∇x Loss), 0, 1 )
  只算一次梯度，把每個像素往「讓模型更錯」的方向推 ε。ε 越大，攻擊越強，畫面也越髒。

本實驗的三種情境：
  1. softmax（白箱攻擊）：攻擊者知道 softmax 模型的權重，直接攻擊它
  2. EDL（白箱攻擊）    ：攻擊者知道 EDL 模型的權重，直接攻擊它
  3. EDL（遷移攻擊）    ：用 softmax 模型算出來的攻擊，拿去打 EDL 模型
                          （論文提到：攻擊者實務上常常只有一個「替代模型」）

衡量：
  - 準確率
  - 「被騙且有信心」的比例：答錯，而且模型還很有信心
      softmax：最大機率 ≥ 0.5；EDL：u ≤ 0.5
    對反作弊系統來說，這種錯誤會直接通過信心篩選，最危險
  - 平均不確定度：EDL 用 u；softmax 用 1 − 最大機率

執行：python exp3_fgsm.py
輸出：results/exp3_fgsm.png、results/exp3_examples.png，以及終端機上的表格
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exp_common import (
    COLOR_EDL, COLOR_EXTRA, COLOR_SOFTMAX, DEFAULT_U_THRESHOLD, TEXT_SECONDARY,
    edl_outputs, fmt_num, fmt_pct, get_device, load_both_models,
    pixel_space_from_loader, print_table, save_figure, setup_plot_style, softmax_outputs,
)

DEFAULT_EPS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

SCENARIOS = [
    # (名稱, 被攻擊的模型種類, 攻擊方向來自哪個模型, 顏色, 線型)
    ("softmax（白箱攻擊）", "softmax", "softmax", COLOR_SOFTMAX, "-o"),
    ("EDL（白箱攻擊）", "edl", "edl", COLOR_EDL, "-s"),
    ("EDL（遷移攻擊）", "edl", "softmax", COLOR_EXTRA, "--^"),
]


def attack_loss(model, x, y, kind):
    """攻擊者想讓這個 loss 變大（讓正確答案的機率變小）。"""
    logits = model(x)
    if kind == "softmax":
        return F.cross_entropy(logits, y, reduction="sum")
    # EDL 的機率是 p = α / S，攻擊目標是讓正確類別的 p 變小
    p_true = edl_outputs(logits)["probs"].gather(1, y.unsqueeze(1)).squeeze(1)
    return -torch.log(p_true + 1e-12).sum()


def fgsm_direction(model, x, y, kind, space):
    """回傳 sign(∇x Loss)（在 0～1 像素空間計算）與原始像素圖。"""
    x_pix = space.to_pixel(x).detach().requires_grad_(True)
    with torch.enable_grad():
        loss = attack_loss(model, space.from_pixel(x_pix), y, kind)
        (grad,) = torch.autograd.grad(loss, x_pix)
    return grad.sign(), x_pix.detach()


@torch.no_grad()
def predict(model, kind, x):
    """回傳 (預測, 不確定度)。softmax 的不確定度 = 1 − 最大機率；EDL = u。"""
    logits = model(x)
    if kind == "softmax":
        out = softmax_outputs(logits)
        return out["pred"], 1.0 - out["conf"]
    out = edl_outputs(logits)
    return out["pred"], out["u"]


def run_attacks(models, loader, device, space, eps_list, max_samples):
    stats = {name: {"n": np.zeros(len(eps_list)), "correct": np.zeros(len(eps_list)),
                    "fooled_confident": np.zeros(len(eps_list)), "confident": np.zeros(len(eps_list)),
                    "sum_unc": np.zeros(len(eps_list))}
             for name, *_ in SCENARIOS}
    seen = 0
    for x, y in loader:
        if seen >= max_samples:
            break
        x, y = x[: max_samples - seen].to(device), y[: max_samples - seen].to(device)
        directions = {}
        for kind in ("softmax", "edl"):
            directions[kind], x_pix = fgsm_direction(models[kind], x, y, kind, space)

        for i, eps in enumerate(eps_list):
            for name, target_kind, source_kind, *_ in SCENARIOS:
                x_adv = space.from_pixel((x_pix + eps * directions[source_kind]).clamp(0, 1))
                pred, unc = predict(models[target_kind], target_kind, x_adv)
                wrong = pred != y
                confident = unc <= DEFAULT_U_THRESHOLD
                s = stats[name]
                s["n"][i] += len(y)
                s["correct"][i] += (~wrong).sum().item()
                s["fooled_confident"][i] += (wrong & confident).sum().item()
                s["confident"][i] += confident.sum().item()
                s["sum_unc"][i] += unc.sum().item()
        seen += len(y)
        print(f"  已處理 {seen} 張", end="\r")
    print()

    for s in stats.values():
        s["accuracy"] = s["correct"] / s["n"]
        s["fooled_confident_rate"] = s["fooled_confident"] / s["n"]
        s["confident_rate"] = s["confident"] / s["n"]
        s["mean_unc"] = s["sum_unc"] / s["n"]
    return stats


def print_results(stats, eps_list):
    for name, *_ in SCENARIOS:
        s = stats[name]
        print(f"\n=== {name} ===")
        rows = [[f"{eps:.2f}", fmt_pct(s["accuracy"][i]), fmt_pct(s["confident_rate"][i]),
                 fmt_pct(s["fooled_confident_rate"][i]), fmt_num(s["mean_unc"][i])]
                for i, eps in enumerate(eps_list)]
        print_table(["ε", "準確率", "有信心比例", "被騙且有信心", "平均不確定度"], rows)


def plot_curves(stats, eps_list):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("實驗 3：FGSM 對抗攻擊（迷你版 Fig. 4）", fontsize=15, fontweight="bold")
    panels = [
        ("accuracy", "(a) 準確率", "準確率（%）", 100),
        ("fooled_confident_rate", "(b) 被騙且有信心的比例（最危險）", "比例（%）", 100),
        ("mean_unc", "(c) 模型回報的平均不確定度", "不確定度（越高越不確定）", 1),
    ]
    for ax, (key, title, ylabel, scale) in zip(axes, panels):
        for name, _, _, color, style in SCENARIOS:
            ax.plot(eps_list, stats[name][key] * scale, style, color=color, markersize=6, label=name)
        ax.set_title(title)
        ax.set_xlabel("攻擊強度 ε（像素值 0～1）")
        ax.set_ylabel(ylabel)
        if scale == 100:
            ax.set_ylim(-2, 102)
        else:
            ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return save_figure(fig, "exp3_fgsm.png")


@torch.no_grad()
def _first_confident_correct(models, x, y):
    """找一張兩個模型都答對、而且都有信心的圖片當範例。"""
    pred_s, unc_s = predict(models["softmax"], "softmax", x)
    pred_e, unc_e = predict(models["edl"], "edl", x)
    good = (pred_s == y) & (pred_e == y) & (unc_s <= DEFAULT_U_THRESHOLD) & (unc_e <= DEFAULT_U_THRESHOLD)
    idx = torch.nonzero(good).flatten()
    return int(idx[0]) if len(idx) else 0


def plot_examples(models, loader, device, space, eps_list):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    k = _first_confident_correct(models, x, y)
    x, y = x[k:k + 1], y[k:k + 1]

    rows = [("softmax", "攻擊 softmax"), ("edl", "攻擊 EDL")]
    fig, axes = plt.subplots(len(rows), len(eps_list), figsize=(1.9 * len(eps_list), 5.2), squeeze=False)
    fig.suptitle(f"同一張數字 {int(y[0])}，攻擊越來越強", fontsize=14, fontweight="bold")
    for r, (kind, row_label) in enumerate(rows):
        direction, x_pix = fgsm_direction(models[kind], x, y, kind, space)
        for c, eps in enumerate(eps_list):
            adv_pix = (x_pix + eps * direction).clamp(0, 1)
            pred, unc = predict(models[kind], kind, space.from_pixel(adv_pix))
            ax = axes[r, c]
            ax.imshow(adv_pix[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            if kind == "softmax":
                info = f"預測 {int(pred[0])}\n信心 {(1 - float(unc[0])) * 100:.0f}%"
            else:
                info = f"預測 {int(pred[0])}\nu = {float(unc[0]):.2f}"
            ax.set_title(f"ε = {eps}\n{info}", fontsize=9.5, fontweight="normal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, color=TEXT_SECONDARY)
    fig.tight_layout()
    return save_figure(fig, "exp3_examples.png")


def main():
    parser = argparse.ArgumentParser(description="實驗 3：FGSM 對抗攻擊")
    parser.add_argument("--n", type=int, default=10000, help="使用幾張測試圖片（預設 10000）")
    parser.add_argument("--eps", type=float, nargs="+", default=DEFAULT_EPS, help="攻擊強度 ε 的列表")
    args = parser.parse_args()

    setup_plot_style()
    device = get_device()
    from data import dataloaders  # 匯入時會載入 MNIST

    val_loader = dataloaders["val"]
    space = pixel_space_from_loader(val_loader)
    baseline, edl = load_both_models(device)
    models = {"softmax": baseline, "edl": edl}

    print(f"攻擊中：{args.n} 張圖片 × {len(args.eps)} 種強度 × 3 種情境 ...")
    stats = run_attacks(models, val_loader, device, space, args.eps, args.n)
    print_results(stats, args.eps)
    plot_curves(stats, args.eps)
    plot_examples(models, val_loader, device, space, args.eps)

    print("\n--- 解讀提示 ---")
    print("(a) 準確率：兩個模型被攻擊後都會下降，EDL 不一定比較不會被騙。")
    print("(b) 被騙且有信心：這才是重點。EDL 被騙時若 u 跟著升高，就會被信心篩選擋下，不會直接誤判。")
    print("(c) 不確定度：EDL 的 u 隨 ε 上升，代表它察覺到「這張圖怪怪的」。")
    print("遷移攻擊：攻擊者只有替代模型時，攻擊效果通常比白箱攻擊弱。")


if __name__ == "__main__":
    main()
