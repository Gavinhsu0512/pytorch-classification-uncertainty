"""
實驗 4：把 MNIST 改造成「作弊偵測」——在縮小版上重現整篇論文

對應論文：Robust Vision-Based Cheat Detection in Competitive Gaming
  §3 資料集      → exp4_data.py 合成的作弊畫面（外掛 A、B，各有完整／中等／最少三種等級）
  §4 網路設計    → CheatDetector：4 層卷積 + global head（是否作弊）+ local head（作弊熱力圖）
  Table 2        → local head 有沒有幫助？（UL_combined vs UL_global）
  Table 3        → 不同作弊等級的偵測率
  Table 4        → 信心篩選（dropout 的 VR、uncertainty loss 的 u）能不能減少誤判
  Table 5        → 出現新外掛時，LR 比例能不能發出「需要重新訓練」警報
  Fig. 1         → local head 的作弊熱力圖

會訓練 4 個偵測器（每個約 1～2 分鐘，CPU）：
  mse_A ：MSE_combined，只學過外掛 A（論文 Table 4 的對照組）
  ul_A  ：UL_combined， 只學過外掛 A（uncertainty loss + local head）
  ulg_A ：UL_global，   只學過外掛 A，拿掉 local head（Table 2 的消融實驗）
  ul_AB ：UL_combined， 學過外掛 A＋B（Table 5 的「重新訓練」）

執行：python exp4_cheat_detection.py
      python exp4_cheat_detection.py --reuse   （直接載入上次訓練好的模型，只重新評估和畫圖）
      python exp4_cheat_detection.py --evidence relu   （UL 模型改用 ReLU 證據，重現「證據死掉」的問題）
      python exp4_cheat_detection.py --new-cheat diamond   （外掛 B 改用菱形＋骨架版本）
輸出：results/exp4_dataset*.png、results/exp4_heatmaps*.png、results/exp4_results*.png、
      results/exp4_summary*.md（所有表格的 Markdown 版本，可以直接貼進 README）
      檔名後綴：外掛 B 為發光版加 _glow，證據函數為 ReLU 加 _relu；菱形＋exp 沒有後綴
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exp4_data import IMG_SIZE, LEVEL_NAMES, LEVELS, STYLE_NAMES, dilate, make_dataset, set_b_variant
from exp_common import (
    COLOR_EDL, COLOR_EXTRA, COLOR_SOFTMAX, RESULTS_DIR, TEXT_PRIMARY, TEXT_SECONDARY,
    fmt_num, fmt_pct, get_device, print_table, save_figure, setup_plot_style,
)
from losses import exp_evidence, relu_evidence
from losses import mse_loss as evidential_mse_loss  # Sensoy Eq. 5（直接輸入 α 的版本）

# 把網路輸出轉成證據的函數
#   exp ：論文 Eq. 5 的做法 e = exp(l)，永遠大於 0，梯度永遠存在（實驗 4 預設）
#   relu：本專案 losses.py 原本的做法 e = ReLU(l)；某個類別的輸出一旦對所有畫面都是負的，
#         證據和梯度都會變成 0，這個類別就「死掉」學不回來（可用 --evidence relu 重現）
EVIDENCE_FUNCTIONS = {"exp": exp_evidence, "relu": relu_evidence}

U_THRESHOLD = 0.5    # u > 0.5 視為沒信心
RETRAIN_RATIO = 0.5  # 論文規則：任一 LR 比例 < 0.5 就需要重新訓練

MODEL_SPECS = {
    "mse_A": {"name": "MSE_combined", "loss": "mse", "local": True, "train": "train_A", "color": COLOR_SOFTMAX},
    "ul_A": {"name": "UL_combined", "loss": "ul", "local": True, "train": "train_A", "color": COLOR_EDL},
    "ulg_A": {"name": "UL_global（無 local head）", "loss": "ul", "local": False, "train": "train_A",
              "color": COLOR_EXTRA},
    "ul_AB": {"name": "UL_combined（重新訓練 A＋B）", "loss": "ul", "local": True, "train": "train_AB",
              "color": COLOR_EDL},
}


# ---------------------------------------------------------------------------
# 網路：縮小版的論文架構（§4 Network design）
# ---------------------------------------------------------------------------
class CheatDetector(nn.Module):
    """論文：輸入 1080×1920，4 層卷積（48/48/32/16 個濾波器）+ leaky ReLU，再分成兩個輸出頭。
    這裡：輸入 28×28，4 層卷積（24/24/16/16 個濾波器，同樣逐層遞減）+ leaky ReLU。

    global head：平均池化 → dropout → 全連接 16×2 → 是否作弊
                 （論文 Table 2：平均池化最好；dropout 用於 §5.1 的 variation ratio）
    local head ：1×1 卷積 → sigmoid → 14×14 的作弊熱力圖（論文 Fig. 1 的小圖）
    """

    def __init__(self, dropout=0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.LeakyReLU(0.1),             # 28×28
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.LeakyReLU(0.1),  # → 14×14
            nn.Conv2d(24, 16, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(0.1),
        )
        self.local_head = nn.Conv2d(16, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.global_fc = nn.Linear(16, 2)

    def forward(self, x):
        f = self.features(x)                                         # (B, 16, 14, 14)
        local_map = torch.sigmoid(self.local_head(f)).squeeze(1)     # (B, 14, 14)
        pooled = f.mean(dim=(2, 3))                                  # 平均池化
        logits = self.global_fc(self.dropout(pooled))                # (B, 2)
        return logits, local_map


def detector_loss(spec, logits, local_map, y, epoch, num_epochs, device, evidence="exp"):
    """論文的損失函數。

    MSE_combined = MSE_global + MSE_local                 （Table 4 的對照組）
    UL_combined  = UL_global  + MSE_local                 （Eq. 7）
    UL_global    = 只有 UL_global（拿掉 local head）

    local head 沒有像素級的標準答案，論文直接把「整張畫面的標籤」當成每個位置的目標，
    並做標籤平滑：作弊畫面 0.9、正常畫面 0.1（Eq. 2）。
    """
    y_onehot = F.one_hot(y, num_classes=2).float()
    if spec["loss"] == "mse":
        loss_global = F.mse_loss(F.softmax(logits, dim=1), y_onehot)
    else:
        # 使用本專案 losses.py 的 evidential loss（Sensoy Eq. 5），KL 權重在前一半 epoch 逐漸升到 1
        alpha = EVIDENCE_FUNCTIONS[evidence](logits) + 1
        loss_global = torch.mean(evidential_mse_loss(y_onehot, alpha, epoch, 2, max(1, num_epochs // 2), device))
    if not spec["local"]:
        return loss_global
    target_map = (0.1 + 0.8 * y.float()).view(-1, 1, 1).expand_as(local_map)
    return loss_global + F.mse_loss(local_map, target_map)


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------
def load_mnist_digits():
    """從 data.py 取得 MNIST 原始像素（0～1），回傳 (訓練用數字, 測試用數字)。"""
    from data import dataloaders

    def base(loader):
        dataset = loader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def to_numpy(dataset):
        data = dataset.data
        data = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        return data.astype(np.float32) / 255.0

    test_ds = base(dataloaders["val"])
    train_ds = base(dataloaders["train"]) if "train" in dataloaders else None
    test_digits = to_numpy(test_ds)
    if train_ds is None or train_ds is test_ds:  # 沒有獨立的訓練集時，自己切一半
        half = len(test_digits) // 2
        return test_digits[half:], test_digits[:half]
    return to_numpy(train_ds), test_digits


def to_tensor_dataset(d):
    return TensorDataset(torch.from_numpy(d["x"]).unsqueeze(1), torch.from_numpy(d["y"]))


def subset(d, n):
    return {k: v[:n] for k, v in d.items()}


# ---------------------------------------------------------------------------
# 訓練與預測
# ---------------------------------------------------------------------------
def train_model(key, dataset, device, epochs, batch_size, seed, evidence):
    spec = MODEL_SPECS[key]
    torch.manual_seed(seed)
    model = CheatDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(to_tensor_dataset(dataset), batch_size=batch_size, shuffle=True)
    evidence_note = f"，證據函數 {evidence}" if spec["loss"] == "ul" else ""
    print(f"\n訓練 {key}：{spec['name']}，資料 {spec['train']}{evidence_note}")
    for epoch in range(epochs):
        model.train()
        start = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, local_map = model(x)
            loss = detector_loss(spec, logits, local_map, y, epoch, epochs, device, evidence)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total_loss / seen:.4f}  "
              f"訓練準確率 {correct / seen * 100:.1f}%  （{time.time() - start:.0f} 秒）")
    model.eval()
    return model


def model_path(key, evidence, variant):
    # UL 模型的檔名包含證據函數；有學過外掛 B 的模型再加上外掛 B 的版本，彼此不會互相覆蓋。
    # 只學過外掛 A 的模型跟外掛 B 的版本無關，可以直接沿用。
    suffix = f"_{evidence}" if MODEL_SPECS[key]["loss"] == "ul" else ""
    if MODEL_SPECS[key]["train"] == "train_AB" and variant != "diamond":
        suffix += f"_{variant}"
    return os.path.join(RESULTS_DIR, f"exp4_{key}{suffix}.pt")


def output_suffix(variant, evidence):
    """輸出檔名的後綴：菱形 + exp 沒有後綴（最早的版本），其餘組合加上後綴，避免覆蓋。"""
    return ("" if variant == "diamond" else f"_{variant}") + ("" if evidence == "exp" else f"_{evidence}")


def get_model(key, sets, device, args):
    path = model_path(key, args.evidence, args.new_cheat)
    if args.reuse and os.path.exists(path):
        model = CheatDetector().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"載入已訓練的模型：{path}")
        return model
    model = train_model(key, sets[MODEL_SPECS[key]["train"]], device, args.epochs, args.batch_size, args.seed,
                        args.evidence)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def predict(model, kind, dataset, device, mc_passes=0, batch_size=1000, seed=0, evidence="exp"):
    """回傳每張畫面的：預測、作弊機率、u、LR、VR（dropout 的 variation ratio）、熱力圖。"""
    model.eval()
    torch.manual_seed(seed)
    x_all = torch.from_numpy(dataset["x"]).unsqueeze(1)
    keys = ("pred", "p_cheat", "u", "lr", "vr", "local_map")
    out = {k: [] for k in keys}
    for start in range(0, len(x_all), batch_size):
        x = x_all[start:start + batch_size].to(device)
        features = model.features(x)
        local_map = torch.sigmoid(model.local_head(features)).squeeze(1)
        pooled = features.mean(dim=(2, 3))
        logits = model.global_fc(pooled)  # 測試時不使用 dropout
        pred = logits.argmax(dim=1)
        if kind == "ul":
            alpha = EVIDENCE_FUNCTIONS[evidence](logits) + 1
            S = alpha.sum(dim=1)
            p_cheat = alpha[:, 1] / S
            u = 2 / S                                                      # K = 2
            lr = alpha.max(dim=1).values / alpha.min(dim=1).values         # 論文 Eq. 8
        else:
            p_cheat = F.softmax(logits, dim=1)[:, 1]
            u = torch.full_like(p_cheat, float("nan"))
            lr = torch.full_like(p_cheat, float("nan"))
        if mc_passes > 0:
            # 論文 §5.1：同一張圖、開著 dropout 跑 T 次，看預測有多不一致
            # VR = 1 - (最常出現的類別次數 / T)；VR = 0 代表每次都一樣，最有把握
            votes = torch.zeros(len(x), 2, device=device)
            rows = torch.arange(len(x), device=device)
            for _ in range(mc_passes):
                mc_logits = model.global_fc(F.dropout(pooled, p=model.dropout.p, training=True))
                votes[rows, mc_logits.argmax(dim=1)] += 1
            vr = 1 - votes.max(dim=1).values / mc_passes
        else:
            vr = torch.zeros(len(x), device=device)
        for k, v in zip(keys, (pred, p_cheat, u, lr, vr, local_map)):
            out[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def rate(mask_true, mask_all):
    n = int(np.sum(mask_all))
    return float(np.sum(mask_true & mask_all)) / n if n else float("nan")


def confusion(pred, y, keep=None):
    keep = np.ones(len(y), dtype=bool) if keep is None else keep
    return {
        "TP": int(((pred == 1) & (y == 1) & keep).sum()),
        "FN": int(((pred == 0) & (y == 1) & keep).sum()),
        "FP": int(((pred == 1) & (y == 0) & keep).sum()),
        "TN": int(((pred == 0) & (y == 0) & keep).sum()),
    }


def summary_rates(c):
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    sens = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    acc = (tp + tn) / (tp + fn + fp + tn) if tp + fn + fp + tn else float("nan")
    return sens, prec, acc


def detection_by_level(pred, d):
    """Table 3：各作弊等級被抓到的比例，以及正常畫面被正確判為正常的比例。"""
    out = {level: rate(pred == 1, d["level"] == level) for level in LEVELS}
    out["clean"] = rate(pred == 0, d["y"] == 0)
    out["overall"] = float(np.mean(pred == d["y"]))
    return out


def lr_means(p):
    """論文 Table 5 的三種平均 LR：全部、預測為作弊、預測為正常。"""
    pos = p["pred"] == 1
    return {
        "tot": float(np.mean(p["lr"])),
        "pos": float(np.mean(p["lr"][pos])) if pos.any() else float("nan"),
        "neg": float(np.mean(p["lr"][~pos])) if (~pos).any() else float("nan"),
    }


def localization(p, d):
    """local head 的熱力圖最亮的位置，有沒有落在外掛像素上（容許 1 像素誤差）。"""
    maps = np.repeat(np.repeat(p["local_map"], 2, axis=1), 2, axis=2)  # 14×14 → 28×28
    flat = maps.reshape(len(maps), -1).argmax(axis=1)
    rows, cols = np.unravel_index(flat, (IMG_SIZE, IMG_SIZE))
    target = dilate(d["mask"], 1)
    hit = target[np.arange(len(target)), rows, cols]
    chance = target.reshape(len(target), -1).mean(axis=1)  # 隨便猜一個位置會命中的機率
    return {level: (float(hit[d["level"] == level].mean()), float(chance[d["level"] == level].mean()))
            for level in LEVELS}


def md_table(headers, rows):
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 畫圖
# ---------------------------------------------------------------------------
def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_dataset(sets, suffix=""):
    rows = [("正常\n（含隊友標記）", sets["test_A"], (sets["test_A"]["y"] == 0) & sets["test_A"]["teammate"])]
    for style, name in (("A", "test_A"), ("B", "test_B")):
        for level in LEVELS:
            rows.append((f"外掛 {style}\n{LEVEL_NAMES[level]}", sets[name], sets[name]["level"] == level))
    fig, axes = plt.subplots(len(rows), 6, figsize=(8.5, 11.5))
    fig.suptitle("實驗 4：合成的作弊偵測資料（數字 = 敵人）", fontsize=14, fontweight="bold")
    for r, (label, d, select) in enumerate(rows):
        for c, i in enumerate(np.where(select)[0][:6]):
            axes[r, c].imshow(d["x"][i], cmap="gray", vmin=0, vmax=1)
        for c in range(6):
            _clean_axes(axes[r, c])
        axes[r, 0].set_ylabel(label, fontsize=10, color=TEXT_PRIMARY, rotation=0, ha="right", va="center")
    fig.tight_layout()
    return save_figure(fig, f"exp4_dataset{suffix}.png")


def plot_heatmaps(sets, preds, suffix=""):
    """仿照論文 Fig. 1：畫面 + local head 熱力圖 + 作弊機率與不確定度。"""
    picks = [("正常（隊友標記）", "test_A", (sets["test_A"]["y"] == 0) & sets["test_A"]["teammate"])]
    for style, name, levels in (("A", "test_A", LEVELS), ("B", "test_B", ("full", "minimal"))):
        for level in levels:
            picks.append((f"外掛 {style} {LEVEL_NAMES[level]}", name, sets[name]["level"] == level))
    model_rows = [("ul_A", "熱力圖\n只學過外掛 A"), ("ul_AB", "熱力圖\n重新訓練 A＋B")]

    fig, axes = plt.subplots(1 + len(model_rows), len(picks), figsize=(2.1 * len(picks), 7.6))
    fig.suptitle("實驗 4：local head 的作弊熱力圖（迷你版 Fig. 1）", fontsize=14, fontweight="bold")
    for c, (title, name, select) in enumerate(picks):
        i = int(np.where(select)[0][0])
        ax = axes[0, c]
        ax.imshow(sets[name]["x"][i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        _clean_axes(ax)
        for r, (key, _) in enumerate(model_rows, start=1):
            p = preds[(key, name)]
            ax = axes[r, c]
            ax.imshow(p["local_map"][i], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlabel(f"P(作弊) = {p['p_cheat'][i] * 100:.0f}%\nu = {p['u'][i]:.2f}",
                          fontsize=9, color=TEXT_PRIMARY)
            _clean_axes(ax)
    axes[0, 0].set_ylabel("遊戲畫面", fontsize=10, rotation=0, ha="right", va="center")
    for r, (_, label) in enumerate(model_rows, start=1):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center")
    fig.text(0.5, 0.005, "熱力圖越深＝該位置越像外掛；論文的 local head 從沒看過外掛位置的標準答案，只靠整張畫面的標籤學會定位",
             ha="center", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save_figure(fig, f"exp4_heatmaps{suffix}.png")


def plot_results(level_stats, table5_rows, suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    fig.suptitle("實驗 4：迷你版作弊偵測（Table 2、3、5）", fontsize=15, fontweight="bold")
    categories = list(LEVELS) + ["clean"]
    labels = [LEVEL_NAMES[c] for c in LEVELS] + ["正常畫面\n（判為正常）"]
    keys = ["mse_A", "ul_A", "ulg_A"]
    width = 0.26
    x = np.arange(len(categories))
    for ax, test_name, title in [(axes[0], "test_A", "(a) 偵測率｜測試：外掛 A（學過）"),
                                 (axes[1], "test_B", "(b) 偵測率｜測試：外掛 B（沒學過）")]:
        for j, key in enumerate(keys):
            values = [level_stats[(key, test_name)][c] * 100 for c in categories]
            ax.bar(x + (j - 1) * width, values, width=width * 0.92, color=MODEL_SPECS[key]["color"],
                   label=MODEL_SPECS[key]["name"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title(title)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.34, 0.93))

    ax = axes[2]
    names = [r["label"] for r in table5_rows]
    ratios = [r["min_ratio"] for r in table5_rows]
    bars = ax.bar(np.arange(len(names)), ratios, color=COLOR_EDL, width=0.55)
    ax.axhline(RETRAIN_RATIO, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2)
    for bar, r in zip(bars, table5_rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{r['min_ratio']:.2f}\n{'需要重新訓練' if r['retrain'] else '正常'}",
                ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, max(1.3, max(ratios) + 0.35))
    ax.set_ylabel("三種 LR 比例中的最小值")
    ax.set_title("(c) 迷你版 Table 5：該重新訓練了嗎？\n（虛線 = 重新訓練門檻 0.5）")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, f"exp4_results{suffix}.png")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="實驗 4：MNIST 作弊偵測（迷你版論文重現）")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=20000, help="每個訓練集的畫面數")
    parser.add_argument("--n-test", type=int, default=6000, help="每個測試集的畫面數（6 的倍數可讓各等級數量相同）")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mc", type=int, default=64, help="dropout variation ratio 的取樣次數（論文 T = 64）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse", action="store_true", help="載入上次訓練好的模型，不重新訓練")
    parser.add_argument("--evidence", choices=["exp", "relu"], default="exp",
                        help="UL 模型的證據函數：exp（論文 Eq. 5，預設）或 relu（本專案 losses.py 原本的做法）")
    parser.add_argument("--new-cheat", choices=["glow", "diamond"], default="glow",
                        help="外掛 B 的版本：glow（發光透視，跟外掛 A 風格完全不同，預設）"
                             "或 diamond（菱形＋骨架，跟外掛 A 一樣是細亮線）")
    args = parser.parse_args()

    set_b_variant(args.new_cheat)
    suffix = output_suffix(args.new_cheat, args.evidence)
    setup_plot_style()
    device = get_device()
    md = ["# 實驗 4 結果摘要（自動產生）", "",
          f"設定：每個模型訓練 {args.epochs} 個 epoch，訓練集 {args.n_train} 張，測試集 {args.n_test} 張，"
          f"UL 模型的證據函數：{args.evidence}，外掛 B：{STYLE_NAMES['B']}。", ""]

    # 1. 資料
    print("產生合成資料 ...")
    train_digits, test_digits = load_mnist_digits()
    sets = {
        "train_A": make_dataset(train_digits, args.n_train, styles=("A",), seed=args.seed),
        "train_AB": make_dataset(train_digits, args.n_train, styles=("A", "B"), seed=args.seed + 1),
        "test_A": make_dataset(test_digits, args.n_test, styles=("A",), balanced=True, seed=args.seed + 2),
        "test_B": make_dataset(test_digits, args.n_test, styles=("B",), balanced=True, seed=args.seed + 3),
    }
    for name, d in sets.items():
        print(f"  {name}：{len(d['y'])} 張，作弊 {int(d['y'].sum())} 張，有隊友標記 {int(d['teammate'].sum())} 張")
    plot_dataset(sets, suffix)

    # 2. 訓練
    if not args.reuse:
        print(f"\n開始訓練 4 個偵測器（每個 {args.epochs} 個 epoch），CPU 上每個約 1～3 分鐘 ...")
    models = {key: get_model(key, sets, device, args) for key in MODEL_SPECS}

    # 3. 預測
    print("\n評估中 ...")
    preds = {}
    for key, model in models.items():
        kind = MODEL_SPECS[key]["loss"]
        for name in ("test_A", "test_B"):
            preds[(key, name)] = predict(model, kind, sets[name], device, mc_passes=args.mc, seed=args.seed,
                                         evidence=args.evidence)
        train_subset = subset(sets[MODEL_SPECS[key]["train"]], min(5000, args.n_train))
        preds[(key, "train")] = predict(model, kind, train_subset, device, seed=args.seed, evidence=args.evidence)

    report(sets, preds, md, suffix)


def report(sets, preds, md, suffix=""):
    """把預測結果整理成論文的各個表格，畫圖，並存成 Markdown 摘要。"""
    # 4. Table 2、3：各等級偵測率
    level_stats = {(key, name): detection_by_level(preds[(key, name)]["pred"], sets[name])
                   for key in ("mse_A", "ul_A", "ulg_A", "ul_AB") for name in ("test_A", "test_B")}
    headers3 = ["模型", "測試資料", "完整", "中等", "最少", "正常畫面（判為正常）", "整體準確率"]
    rows3 = []
    for key in ("mse_A", "ul_A", "ulg_A"):
        for name in ("test_A", "test_B"):
            s = level_stats[(key, name)]
            rows3.append([MODEL_SPECS[key]["name"], STYLE_NAMES[name[-1]], fmt_pct(s["full"]), fmt_pct(s["medium"]),
                          fmt_pct(s["minimal"]), fmt_pct(s["clean"]), fmt_pct(s["overall"])])
    print("\n=== 迷你版 Table 2、3：各作弊等級的偵測率（三個模型都只學過外掛 A）===")
    print_table(headers3, rows3)
    md += ["## 迷你版 Table 2、3：各作弊等級的偵測率", "", "三個模型都只學過外掛 A。", "", md_table(headers3, rows3), ""]

    # 5. Table 4：信心篩選
    y_a = sets["test_A"]["y"]
    p_mse, p_ul = preds[("mse_A", "test_A")], preds[("ul_A", "test_A")]
    filters = [
        ("MSE_combined", "無", p_mse, None),
        ("MSE_combined", "VR = 0", p_mse, p_mse["vr"] == 0),
        ("UL_combined", "無", p_ul, None),
        ("UL_combined", "VR = 0", p_ul, p_ul["vr"] == 0),
        ("UL_combined", f"VR = 0 且 u ≤ {U_THRESHOLD}", p_ul, (p_ul["vr"] == 0) & (p_ul["u"] <= U_THRESHOLD)),
    ]
    headers4 = ["損失函數", "信心篩選", "有信心 TP", "有信心 FN", "有信心 FP", "有信心 TN",
                "沒信心 TP", "沒信心 FN", "沒信心 FP", "沒信心 TN"]
    rows4 = []
    for loss_name, filter_name, p, keep in filters:
        c = confusion(p["pred"], y_a, keep)
        row = [loss_name, filter_name, c["TP"], c["FN"], c["FP"], c["TN"]]
        if keep is None:
            row += ["—"] * 4
        else:
            n = confusion(p["pred"], y_a, ~keep)
            row += [n["TP"], n["FN"], n["FP"], n["TN"]]
        rows4.append(row)
    print("\n=== 迷你版 Table 4：信心篩選能減少誤判嗎？（測試：外掛 A）===")
    print_table(headers4, rows4)
    md += ["## 迷你版 Table 4：信心篩選能減少誤判嗎？", "",
           f"測試資料：外掛 A，共 {len(y_a)} 張（作弊 {int(y_a.sum())} 張）。FP = 正常畫面被判為作弊（冤枉玩家）。", "",
           md_table(headers4, rows4), ""]

    # 誤報來源：隊友標記
    clean = y_a == 0
    teammate = sets["test_A"]["teammate"]
    headers_fp = ["模型", "有隊友標記的正常畫面誤報率", "沒有隊友標記的正常畫面誤報率"]
    rows_fp = [[MODEL_SPECS[k]["name"], fmt_pct(rate(preds[(k, "test_A")]["pred"] == 1, clean & teammate)),
                fmt_pct(rate(preds[(k, "test_A")]["pred"] == 1, clean & ~teammate))]
               for k in ("mse_A", "ul_A", "ulg_A")]
    print("\n=== 誤報來源：合法的隊友標記 ===")
    print_table(headers_fp, rows_fp)
    md += ["## 誤報來源：合法的隊友標記", "", md_table(headers_fp, rows_fp), ""]

    # 6. Table 5：重新訓練警報
    table5_rows = []
    headers5 = ["訓練資料", "測試資料", "LR_tot 比例", "LR_pos 比例", "LR_neg 比例", "敏感度", "精確度", "準確率", "需要重新訓練？"]
    rows5 = []
    for key, train_label in (("ul_A", "外掛 A"), ("ul_AB", "外掛 A＋B")):
        base = lr_means(preds[(key, "train")])
        for name in ("test_A", "test_B"):
            p = preds[(key, name)]
            field = lr_means(p)
            ratios = {k: field[k] / base[k] if base[k] and not np.isnan(base[k]) else float("nan") for k in base}
            valid = [v for v in ratios.values() if not np.isnan(v)]
            min_ratio = min(valid) if valid else float("nan")
            retrain = bool(valid) and min_ratio < RETRAIN_RATIO
            sens, prec, acc = summary_rates(confusion(p["pred"], sets[name]["y"]))
            rows5.append([train_label, STYLE_NAMES[name[-1]], fmt_num(ratios["tot"], 2), fmt_num(ratios["pos"], 2),
                          fmt_num(ratios["neg"], 2), fmt_pct(sens), fmt_pct(prec), fmt_pct(acc),
                          "是" if retrain else "否"])
            table5_rows.append({"label": f"訓練 {train_label.replace('外掛 ', '')}\n測試 {name[-1]}",
                                "min_ratio": min_ratio, "retrain": retrain})
    print("\n=== 迷你版 Table 5：出現新外掛時，能自動發現該重新訓練嗎？===")
    print_table(headers5, rows5)
    md += ["## 迷你版 Table 5：出現新外掛時，能自動發現該重新訓練嗎？", "",
           "LR 比例 = 測試資料的平均 LR ÷ 訓練資料的平均 LR；任一比例 < 0.5 就需要重新訓練（論文規則）。", "",
           md_table(headers5, rows5), ""]

    # 7. 熱力圖定位
    headers_loc = ["模型", "完整", "中等", "最少"]
    rows_loc = []
    for key in ("mse_A", "ul_A"):
        loc = localization(preds[(key, "test_A")], sets["test_A"])
        rows_loc.append([MODEL_SPECS[key]["name"]] + [f"{loc[lv][0] * 100:.1f}%（隨機猜 {loc[lv][1] * 100:.1f}%）"
                                                     for lv in LEVELS])
    print("\n=== 熱力圖定位：最亮的位置有沒有落在外掛上？（測試：外掛 A）===")
    print_table(headers_loc, rows_loc)
    md += ["## 熱力圖定位：最亮的位置有沒有落在外掛上？", "",
           "測試資料：外掛 A。括號內為隨便猜一個位置會命中的機率。", "", md_table(headers_loc, rows_loc), ""]

    # 8. 圖與摘要
    plot_heatmaps(sets, preds, suffix)
    plot_results(level_stats, table5_rows, suffix)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, f"exp4_summary{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"已儲存表格摘要：{md_path}")

    print("\n--- 解讀提示 ---")
    print("Table 3：論文發現「最少」等級最難偵測，看看這裡是否一樣。")
    print("Table 2：比較 UL_combined 和 UL_global，local head 有沒有幫助偵測「最少」等級？")
    print("Table 4：FP（冤枉玩家）在加上信心篩選後有沒有下降？代價是多少 TP 被歸為沒信心？")
    print("Table 5：只學過外掛 A 的模型遇到外掛 B，LR 比例應該掉到 0.5 以下；重新訓練後應該恢復。")


if __name__ == "__main__":
    main()
