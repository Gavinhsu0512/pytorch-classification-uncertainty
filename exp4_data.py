"""
實驗 4 的資料合成：把 MNIST 改造成「作弊偵測」資料集（只用 numpy，不需要 PyTorch）

對應論文 §3 Dataset：論文用真實的外掛軟體（AimWare、Iniuria、Artemis）錄製遊戲畫面，
並設定 full / medium / minimal 三種作弊資訊量。這裡用程式合成縮小版。

每一張「遊戲畫面」由三層組成：
  1. MNIST 手寫數字：代表畫面中的敵人
  2. 合法的遊戲介面（HUD）：正常畫面和作弊畫面都可能有
       - 準心：畫面正中央的小十字
       - 自己的血條：左下角的橫條
       - 隊友標記：數字上方的小倒三角形
         （真實遊戲中隊友頭上會有合法標記，外觀跟外掛很像，是誤判的主要來源）
  3. 透視外掛的疊加圖層：只有作弊畫面才有
       外掛 A（方框 ESP）—— 訓練時使用
         完整 full   ：方框 + 血條 + 名字標籤
         中等 medium ：只有方框
         最少 minimal：數字正上方一個 2×2 小點
       外掛 B —— 模擬「沒見過的新外掛」，有兩種版本（用 set_b_variant() 切換）：
         glow（預設，發光透視）：視覺風格跟外掛 A 完全不同，沒有任何細直線或小方塊
           完整 full   ：敵人全身改成單色（chams）+ 外圍柔和發光
           中等 medium ：只有外圍柔和發光
           最少 minimal：敵人腳下一團柔和的小光暈（類似「腳步聲 ESP」）
         diamond（菱形 + 骨架 ESP）：跟外掛 A 一樣是細亮線，模型很容易「舉一反三」
           完整 full   ：菱形框 + 骨架中線 + 小 x 標記
           中等 medium ：只有菱形框
           最少 minimal：數字正上方一個 3×3 小 x

標籤：0 = 正常畫面，1 = 作弊畫面

另外記錄每張圖的外掛像素位置（mask），用來檢查 local head 的熱力圖有沒有找對地方。
"""
import numpy as np

IMG_SIZE = 28
LEVELS = ("full", "medium", "minimal")
LEVEL_NAMES = {"full": "完整", "medium": "中等", "minimal": "最少"}
STYLE_NAMES = {"A": "外掛 A（方框）", "B": "外掛 B（發光）"}  # B 的名稱會隨 set_b_variant() 改變
OVERLAY_INTENSITY = (0.35, 1.0)  # 外掛與隊友標記的亮度範圍，越暗越難偵測
TEAMMATE_PROB = 0.3               # 每張畫面出現隊友標記的機率


# ---------------------------------------------------------------------------
# 基本繪圖
# ---------------------------------------------------------------------------
def _plot_points(img, mask, rows, cols, value):
    """在指定像素畫上亮度 value（超出畫面的部分自動忽略）。mask 不為 None 時同步標記外掛像素。"""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    keep = (rows >= 0) & (rows < IMG_SIZE) & (cols >= 0) & (cols < IMG_SIZE)
    rows, cols = rows[keep], cols[keep]
    img[rows, cols] = np.maximum(img[rows, cols], value)
    if mask is not None:
        mask[rows, cols] = True


def draw_line(img, mask, r0, c0, r1, c1, value):
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rows = np.rint(np.linspace(r0, r1, n)).astype(int)
    cols = np.rint(np.linspace(c0, c1, n)).astype(int)
    _plot_points(img, mask, rows, cols, value)


def draw_rect(img, mask, r0, r1, c0, c1, value):
    draw_line(img, mask, r0, c0, r0, c1, value)
    draw_line(img, mask, r1, c0, r1, c1, value)
    draw_line(img, mask, r0, c0, r1, c0, value)
    draw_line(img, mask, r0, c1, r1, c1, value)


def digit_bbox(img, threshold=0.2):
    """數字（敵人）的外框：(上, 下, 左, 右)。"""
    rows = np.where(img.max(axis=1) > threshold)[0]
    cols = np.where(img.max(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 9, 18, 9, 18
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


# ---------------------------------------------------------------------------
# 合法介面（HUD）
# ---------------------------------------------------------------------------
def add_hud(img, bbox, rng, teammate_prob=TEAMMATE_PROB):
    """畫上合法的遊戲介面，回傳這張圖有沒有隊友標記。"""
    center = IMG_SIZE // 2
    if rng.random() < 0.5:  # 準心
        v = rng.uniform(0.4, 0.8)
        draw_line(img, None, center, center - 2, center, center + 2, v)
        draw_line(img, None, center - 2, center, center + 2, center, v)
    if rng.random() < 0.5:  # 自己的血條
        v = rng.uniform(0.4, 0.8)
        length = int(rng.integers(3, 9))
        draw_line(img, None, IMG_SIZE - 2, 1, IMG_SIZE - 2, 1 + length, v)
    has_teammate = bool(rng.random() < teammate_prob)
    if has_teammate:  # 隊友標記：數字上方的倒三角形
        r0, _, c0, c1 = bbox
        cm = (c0 + c1) // 2
        top = max(r0 - 5, 0)
        v = rng.uniform(*OVERLAY_INTENSITY)
        draw_line(img, None, top, cm - 1, top, cm + 1, v)
        _plot_points(img, None, [top + 1], [cm], v)
    return has_teammate


# ---------------------------------------------------------------------------
# 外掛圖層
# ---------------------------------------------------------------------------
def draw_cheat_a(img, mask, bbox, level, value, rng, digit=None):
    """外掛 A：方框 ESP。"""
    r0, r1, c0, c1 = bbox
    br0, br1, bc0, bc1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1  # 方框比數字大一圈
    cm = (c0 + c1) // 2
    if level in ("full", "medium"):
        draw_rect(img, mask, br0, br1, bc0, bc1, value)
    if level == "full":
        height = br1 - br0
        hp = int(rng.integers(max(2, height // 3), height + 1))       # 敵人血量（隨機）
        draw_line(img, mask, br1 - hp, bc0 - 2, br1, bc0 - 2, value)   # 血條（方框左邊）
        draw_line(img, mask, br0 - 2, cm - 2, br0 - 2, cm + 2, value)  # 名字標籤（方框上方）
    if level == "minimal":
        _plot_points(img, mask, [r0 - 2, r0 - 2, r0 - 1, r0 - 1], [cm, cm + 1, cm, cm + 1], value)


def draw_cheat_b(img, mask, bbox, level, value, rng, digit=None):
    """外掛 B 的菱形版本：菱形框 + 骨架 ESP。"""
    r0, r1, c0, c1 = bbox
    rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
    half_h, half_w = (r1 - r0) / 2 + 3, (c1 - c0) / 2 + 3
    top, bottom = (rc - half_h, cc), (rc + half_h, cc)
    left, right = (rc, cc - half_w), (rc, cc + half_w)
    if level in ("full", "medium"):
        for a, b in [(top, right), (right, bottom), (bottom, left), (left, top)]:
            draw_line(img, mask, a[0], a[1], b[0], b[1], value)
    if level == "full":
        draw_line(img, mask, top[0], top[1], bottom[0], bottom[1], value)  # 骨架中線（上下頂點相連）
    if level in ("full", "minimal"):
        cm, rt = int(round(cc)), r0 - 2  # 小 x 標記
        _plot_points(img, mask, [rt - 1, rt, rt + 1, rt - 1, rt + 1],
                     [cm - 1, cm, cm + 1, cm + 1, cm - 1], value)


def _paint_region(img, mask, region, value):
    """把一整塊區域（布林陣列）畫上亮度，並標記為外掛像素。"""
    img[region] = np.maximum(img[region], value)
    mask |= region


def draw_cheat_b_glow(img, mask, bbox, level, value, rng, digit=None):
    """外掛 B 的發光版本：沒有任何細直線或小方塊，跟外掛 A 的視覺風格完全不同。

    digit：還沒畫上合法介面前的原始數字圖，用來找出敵人本身的筆畫
           （避免準心、隊友標記也被當成敵人一起發光）
    """
    stroke = (digit if digit is not None else img) > 0.15  # 敵人的筆畫
    glow = 0.7 * value                                      # 發光比線條柔和
    if level in ("full", "medium"):
        # 外圍發光：距離筆畫 1、2、3 像素的位置，亮度逐漸變淡
        inner = stroke
        for strength in (0.8, 0.5, 0.25):
            outer = dilate(inner[None], 1)[0]
            _paint_region(img, mask, outer & ~inner, glow * strength)
            inner = outer
    if level == "full":
        # chams：敵人全身改成同一種亮度，失去原本筆畫的明暗變化
        img[stroke] = rng.uniform(0.55, 0.8)
        mask |= stroke
    if level == "minimal":
        # 敵人腳下的柔和小光暈（高斯分布，沒有清楚的邊緣）
        _, r1, c0, c1 = bbox
        center_r = min(r1 + 2.5, IMG_SIZE - 1.5)
        center_c = (c0 + c1) / 2
        rows, cols = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        blob = 0.75 * value * np.exp(-((rows - center_r) ** 2 + (cols - center_c) ** 2) / (2 * 1.4 ** 2))
        region = blob > 0.06
        img[region] = np.maximum(img[region], blob[region])
        mask |= region


CHEAT_DRAWERS = {"A": draw_cheat_a, "B": draw_cheat_b_glow}

B_VARIANTS = {
    "glow": ("外掛 B（發光）", draw_cheat_b_glow),
    "diamond": ("外掛 B（菱形＋骨架）", draw_cheat_b),
}


def set_b_variant(variant):
    """切換外掛 B 的版本：glow（發光透視，預設）或 diamond（菱形＋骨架）。"""
    name, drawer = B_VARIANTS[variant]
    STYLE_NAMES["B"] = name
    CHEAT_DRAWERS["B"] = drawer


# ---------------------------------------------------------------------------
# 產生資料集
# ---------------------------------------------------------------------------
def make_dataset(digits, n, styles=("A",), cheat_ratio=0.5, balanced=False, seed=0):
    """把 MNIST 數字做成作弊偵測資料集。

    digits     ：(N, 28, 28) 的數字圖，像素值 0～1
    n          ：要產生幾張畫面
    styles     ：作弊畫面使用哪些外掛，例如 ("A",) 或 ("A", "B")
    cheat_ratio：作弊畫面的比例
    balanced   ：True 時，各外掛 × 各等級的數量完全相同（測試集用）

    回傳 dict：
      x         (n, 28, 28) 畫面
      y         (n,)        標籤，1 = 作弊
      style     (n,)        "A"、"B"，正常畫面為 ""
      level     (n,)        "full"、"medium"、"minimal"，正常畫面為 ""
      mask      (n, 28, 28) 外掛像素的位置
      teammate  (n,)        這張畫面有沒有合法的隊友標記
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(digits), size=n, replace=n > len(digits))
    x = np.array(digits[idx], dtype=np.float32, copy=True)
    y = np.zeros(n, dtype=np.int64)
    style = np.full(n, "", dtype="<U1")
    level = np.full(n, "", dtype="<U7")
    mask = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=bool)
    teammate = np.zeros(n, dtype=bool)

    n_cheat = int(round(n * cheat_ratio))
    cheat_idx = rng.permutation(n)[:n_cheat]
    if balanced:
        k = np.arange(n_cheat)
        cheat_levels = np.array(LEVELS)[k % len(LEVELS)]
        cheat_styles = np.array(styles)[(k // len(LEVELS)) % len(styles)]
        order = rng.permutation(n_cheat)
        cheat_levels, cheat_styles = cheat_levels[order], cheat_styles[order]
    else:
        cheat_levels = rng.choice(LEVELS, size=n_cheat)
        cheat_styles = rng.choice(styles, size=n_cheat)
    y[cheat_idx] = 1
    level[cheat_idx] = cheat_levels
    style[cheat_idx] = cheat_styles

    for i in range(n):
        digit = x[i].copy()  # 還沒畫上合法介面前的原始數字
        bbox = digit_bbox(digit)
        teammate[i] = add_hud(x[i], bbox, rng)
        if y[i] == 1:
            value = rng.uniform(*OVERLAY_INTENSITY)
            CHEAT_DRAWERS[style[i]](x[i], mask[i], bbox, level[i], value, rng, digit=digit)

    return {"x": x, "y": y, "style": style, "level": level, "mask": mask, "teammate": teammate}


def dilate(mask, radius=1):
    """把遮罩往外擴大 radius 像素（判斷熱力圖有沒有命中時，容許一點誤差）。"""
    h, w = mask.shape[-2:]
    padded = np.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(radius, radius), (radius, radius)])
    out = np.zeros_like(mask)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            out |= padded[..., dr:dr + h, dc:dc + w]
    return out
