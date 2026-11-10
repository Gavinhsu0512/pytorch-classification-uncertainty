"""
實驗共用工具（exp1、exp2、exp3 都會用到）

功能：
1. 讀取已經訓練好的兩個模型
   - 對照組：results/model.pt（一般 softmax）
   - 實驗組：results/model_uncertainty_mse.pt（EDL，--uncertainty --mse）
2. 從 data.py 找出資料前處理方式，讓新做出來的資料（旋轉、Fashion-MNIST、雜訊、
   對抗攻擊）都跟訓練時使用同一套前處理
3. 計算 softmax 的機率，以及 EDL 的機率、不確定度 u、likelihood ratio
4. 畫圖設定（中文字型、配色）與表格輸出
"""
import os
import unicodedata

import matplotlib

matplotlib.use("Agg")  # 只把圖存成檔案，不跳出視窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from helpers import get_device  # noqa: E402,F401  （讓實驗程式可以從這裡匯入）
from lenet import LeNet  # noqa: E402
from losses import relu_evidence  # noqa: E402

RESULTS_DIR = "./results"
BASELINE_CKPT = os.path.join(RESULTS_DIR, "model.pt")
EDL_CKPT = os.path.join(RESULTS_DIR, "model_uncertainty_mse.pt")

# 「有信心 / 沒信心」的分界：不確定度 > 0.5 就視為沒信心（不採用這次預測）
DEFAULT_U_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 配色（淺色模式）
# ---------------------------------------------------------------------------
COLOR_EDL = "#2a78d6"      # 藍：實驗組 EDL
COLOR_SOFTMAX = "#eb6834"  # 橘：對照組 softmax
COLOR_EXTRA = "#1baf7a"    # 綠：第三條線（例如遷移攻擊）
COLOR_CORRECT = "#1baf7a"  # 答對
COLOR_WRONG = "#e34948"    # 答錯
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def setup_plot_style():
    """設定中文字型與圖表樣式。Windows 會使用微軟正黑體。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
            "Noto Sans CJK TC", "Noto Sans TC", "Arial Unicode MS", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,  # 讓負號在中文字型下正常顯示
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "axes.axisbelow": True,  # 格線畫在資料後面
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
    })


def save_figure(fig, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已儲存圖片：{path}")
    return path


# ---------------------------------------------------------------------------
# 讀取模型
# ---------------------------------------------------------------------------
def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}。請先訓練模型：\n"
            "  對照組：python main.py --train --dropout --epochs 10\n"
            "  實驗組：python main.py --train --dropout --uncertainty --mse --epochs 10"
        )
    model = LeNet()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # 測試模式：關閉 dropout
    return model


def load_both_models(device):
    """回傳 (對照組 softmax 模型, 實驗組 EDL 模型)。"""
    return load_model(BASELINE_CKPT, device), load_model(EDL_CKPT, device)


# ---------------------------------------------------------------------------
# 資料前處理：讓新資料跟訓練時用同一套前處理
# ---------------------------------------------------------------------------
def _base_dataset(loader):
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):  # 處理 torch.utils.data.Subset 這類包裝
        dataset = dataset.dataset
    return dataset


def get_transform(loader):
    """取得 data.py 裡設定的前處理（例如 ToTensor、Resize、Normalize）。"""
    return getattr(_base_dataset(loader), "transform", None)


def get_normalization(transform):
    """找出前處理中的 Normalize(mean, std)。沒有的話，代表輸入本來就是 0～1 的像素值。"""
    mean, std = 0.0, 1.0
    if transform is None:
        return mean, std
    steps = getattr(transform, "transforms", [transform])
    for step in steps:
        if isinstance(step, T.Normalize):
            mean = float(np.ravel(step.mean)[0])
            std = float(np.ravel(step.std)[0])
    return mean, std


class PixelSpace:
    """在「模型的輸入值」和「0～1 的像素值」之間轉換。

    旋轉、加雜訊、對抗攻擊都在像素空間進行（0 = 黑、1 = 白），
    做完再轉回模型的輸入格式，這樣不管 data.py 有沒有做 Normalize 都正確。
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def to_pixel(self, x):
        return x * self.std + self.mean

    def from_pixel(self, x_pix):
        return (x_pix - self.mean) / self.std


def pixel_space_from_loader(loader):
    mean, std = get_normalization(get_transform(loader))
    return PixelSpace(mean, std)


# ---------------------------------------------------------------------------
# 模型輸出 → 機率、不確定度
# ---------------------------------------------------------------------------
def softmax_outputs(logits):
    """對照組：softmax 機率，最大機率當作「信心」。"""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return {"pred": pred, "conf": conf, "probs": probs}


def edl_outputs(logits):
    """實驗組：跟 losses.py 一樣的算法。

    e_k = ReLU(網路輸出)    證據
    α_k = e_k + 1           Dirichlet 參數
    S   = Σ α_k             總證據強度
    p_k = α_k / S           機率
    u   = K / S             不確定度（K = 類別數）
    LR  = 第一名 α ÷ 第二名 α（論文 Eq. 8 的多類別版本，二分類時等於 max(α)/min(α)）
    """
    evidence = relu_evidence(logits)
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    num_classes = logits.shape[1]
    u = (num_classes / S).squeeze(1)
    # 預測類別用 logits 的最大值決定，跟原作者 train.py / test.py 一致。
    # 原因：證據全為 0（u = 1）時，所有 α 都是 1，用 α 取最大值會平手，
    # PyTorch 會固定回傳第一個類別「0」；logits 仍保留排序資訊，可以打破平手。
    # 因為 α = ReLU(logits) + 1 不會改變大小順序，logits 的最大值一定也是 α 的最大值之一。
    pred = logits.argmax(dim=1)
    conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
    top2 = alpha.topk(2, dim=1).values
    lr = top2[:, 0] / top2[:, 1]
    return {"pred": pred, "conf": conf, "probs": probs, "u": u, "alpha": alpha, "lr": lr}


@torch.no_grad()
def evaluate_both(baseline, edl, loader, device, transform_fn=None, max_samples=None):
    """把「同一批」圖片同時丟給兩個模型，收集預測結果。

    transform_fn：可以先把圖片改造（旋轉、換成雜訊……）再丟給模型。
    回傳的所有數值都是 numpy 陣列，方便後續統計。
    """
    softmax_keys = ["pred", "conf"]
    edl_keys = ["pred", "conf", "u", "lr"]
    out = {
        "labels": [],
        "softmax": {k: [] for k in softmax_keys},
        "edl": {k: [] for k in edl_keys},
    }
    seen = 0
    for x, y in loader:
        if max_samples is not None:
            if seen >= max_samples:
                break
            x, y = x[: max_samples - seen], y[: max_samples - seen]
        x, y = x.to(device), y.to(device)
        if transform_fn is not None:
            x = transform_fn(x)
        b = softmax_outputs(baseline(x))
        e = edl_outputs(edl(x))
        out["labels"].append(y.cpu())
        for k in softmax_keys:
            out["softmax"][k].append(b[k].cpu())
        for k in edl_keys:
            out["edl"][k].append(e[k].cpu())
        seen += x.shape[0]

    out["labels"] = torch.cat(out["labels"]).numpy()
    for model_name in ("softmax", "edl"):
        for k in out[model_name]:
            out[model_name][k] = torch.cat(out[model_name][k]).numpy()
    return out


# ---------------------------------------------------------------------------
# 資料改造：旋轉、雜訊
# ---------------------------------------------------------------------------
def rotate_pixels(x_pix, angles_deg):
    """每張圖各自旋轉不同角度（在像素空間做，旋轉後空出來的地方補 0，也就是黑色背景）。"""
    angles = angles_deg.to(device=x_pix.device, dtype=x_pix.dtype)
    theta = torch.deg2rad(angles)
    cos, sin = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(cos)
    matrix = torch.stack([
        torch.stack([cos, -sin, zero], dim=1),
        torch.stack([sin, cos, zero], dim=1),
    ], dim=1)  # 形狀 (B, 2, 3)：每張圖一個旋轉矩陣
    grid = F.affine_grid(matrix, list(x_pix.shape), align_corners=False)
    return F.grid_sample(x_pix, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def make_rotation_fn(space, fixed_angle=None, max_angle=180.0, seed=0):
    """回傳一個「旋轉圖片」的函式。

    fixed_angle=None：每張圖隨機轉 0°～max_angle
    fixed_angle=90 ：每張圖都轉 90°
    """
    generator = torch.Generator().manual_seed(seed)

    def rotate(x):
        n = x.shape[0]
        if fixed_angle is None:
            angles = torch.rand(n, generator=generator) * max_angle
        else:
            angles = torch.full((n,), float(fixed_angle))
        x_pix = space.to_pixel(x)
        return space.from_pixel(rotate_pixels(x_pix, angles))

    return rotate


def make_noise_fn(space, seed=0):
    """回傳一個函式：把圖片整個換成 0～1 的隨機雜訊（完全沒有意義的畫面）。"""
    generator = torch.Generator().manual_seed(seed)

    def noise(x):
        x_pix = torch.rand(x.shape, generator=generator).to(device=x.device, dtype=x.dtype)
        return space.from_pixel(x_pix)

    return noise


# ---------------------------------------------------------------------------
# 統計工具
# ---------------------------------------------------------------------------
def error_detection_auroc(uncertainty, is_wrong):
    """不確定度能不能分辨「答錯」和「答對」？

    回傳 AUROC：隨機抽一個答錯的樣本和一個答對的樣本，
    答錯那個的不確定度比較高的機率。
    0.5 = 跟亂猜一樣，1.0 = 完美分辨。
    """
    is_wrong = np.asarray(is_wrong, dtype=bool)
    n_pos = int(is_wrong.sum())
    n_neg = int((~is_wrong).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(uncertainty, dtype=float))
    return float((ranks[is_wrong].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy_coverage_curve(confidence, correct, num_points=101):
    """依信心由高到低排序，只採用最有信心的前 x% 時，準確率是多少。

    回傳 (採用比例, 採用部分的準確率)。曲線越高，代表信心越能反映對錯。
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    sorted_correct = correct[order]
    n = len(sorted_correct)
    cum_acc = np.cumsum(sorted_correct) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    idx = np.unique(np.linspace(0, n - 1, num_points).astype(int))
    return coverage[idx], cum_acc[idx]


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


def fmt_pct(value):
    return "—" if value is None or np.isnan(value) else f"{value * 100:.1f}%"


def fmt_num(value, digits=3):
    return "—" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def _display_width(text):
    """中文字在終端機佔兩格寬，計算對齊用的寬度。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def print_table(headers, rows):
    """在終端機印出對齊的表格（支援中文）。"""
    table = [list(map(str, headers))] + [list(map(str, r)) for r in rows]
    widths = [max(_display_width(row[i]) for row in table) for i in range(len(headers))]

    def fmt(row):
        return "  ".join(cell + " " * (w - _display_width(cell)) for cell, w in zip(row, widths))

    print(fmt(table[0]))
    print("  ".join("-" * w for w in widths))
    for row in table[1:]:
        print(fmt(row))
