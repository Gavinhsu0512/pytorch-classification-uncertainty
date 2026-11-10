# 以證據式深度學習量化分類不確定度（Evidential Deep Learning to Quantify Classification Uncertainty）

[![arXiv](http://img.shields.io/badge/arXiv-1806.01768-B31B1B.svg)](https://arxiv.org/abs/1806.01768)

> **關於這個 fork**
> 本專案 fork 自 [dougbrion/pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty)，原作者為 Douglas Brion，採 MIT 授權。
> 此 fork 由 [Gavinhsu0512](https://github.com/Gavinhsu0512) 維護，內容包含：
> 1. README 全文翻譯為繁體中文
> 2. 修正原始程式碼在新版 PyTorch 上無法執行的 bug（見 [已知問題與修正](#-已知問題與修正)）
> 3. 補充原理說明與實際重現結果（見 [原理說明](#-原理說明) 與 [重現結果](#-重現結果)）
> 4. 延伸實驗：在 MNIST 上重現作弊偵測論文的三大技術（見 [延伸實驗](#-延伸實驗)）

本專案的目的，是以 PyTorch 提供一個**容易執行、運算需求低**的 demo，示範論文 *Evidential Deep Learning to Quantify Classification Uncertainty* 所提出的方法。論文作者原本是以 TensorFlow 實作。

論文連結：<http://arxiv.org/abs/1806.01768>

收錄於：[Advances in Neural Information Processing Systems 31 (NIPS 2018)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

## 📝 目錄

- [關於](#-關於)
- [論文摘要](#-論文摘要)
- [作者](#️-作者)
- [原理說明](#-原理說明)
- [示範](#-示範)
- [重現結果](#-重現結果)
- [延伸實驗](#-延伸實驗)
- [使用方式](#-使用方式)
- [已知問題與修正](#-已知問題與修正)
- [授權](LICENSE)
- [套件需求](requirements.txt)

## 🧐 關於

本專案的目的，是以 PyTorch 提供一個容易執行、運算需求低的 demo，示範論文 *Evidential Deep Learning to Quantify Classification Uncertainty* 所提出的方法。論文作者原本是以 TensorFlow 實作。

論文連結：<http://arxiv.org/abs/1806.01768>

收錄於：[Advances in Neural Information Processing Systems 31 (NIPS 2018)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

## 📚 論文摘要

確定性神經網路（deterministic neural nets）已被證明能在各式各樣的機器學習問題上學到有效的預測器。然而，由於標準做法是訓練網路去最小化預測誤差，所得到的模型並不知道自己的預測有多可信。有別於貝氏神經網路透過權重的不確定性間接推論預測的不確定性，我們提出以主觀邏輯（subjective logic）理論直接對不確定性建模。我們在類別機率上放置一個 Dirichlet 分布，把神經網路的預測視為主觀意見（subjective opinions），並以確定性神經網路從資料中學習「收集證據、形成這些意見」的函數。對多類別分類問題而言，所得到的預測器是另一個 Dirichlet 分布，其參數由神經網路的連續輸出決定。我們初步分析了新損失函數的特性如何帶來更好的不確定度估計，並觀察到此方法在偵測分布外（out-of-distribution）查詢以及抵抗對抗擾動上，都取得了前所未有的成果。

## ✍️ 作者

論文原作者：

- [Murat Sensoy](https://papers.nips.cc/author/murat-sensoy-11083)
- [Lance Kaplan](https://papers.nips.cc/author/lance-kaplan-11084)
- [Melih Kandemir](https://papers.nips.cc/author/melih-kandemir-11085)

本 repo 程式碼：

- [Douglas Brion](https://github.com/dougbrion)

繁體中文翻譯、bug 修正與原理補充：

- [Gavinhsu0512](https://github.com/Gavinhsu0512)

## 💡 原理說明

### 1. 一般分類器的問題

一般的 CNN 分類器最後會接 softmax，輸出 K 個類別的機率，總和必定為 1。因此，就算輸入一張**完全沒看過**的圖片，模型也**一定得在已知類別中選一個**，而且常常給出很高的機率。

換句話說，softmax 沒辦法說「我不知道」，而**輸出機率也不是可靠的信心指標**：模型可能很有信心地答錯。

### 2. 證據式深度學習（Evidential Deep Learning, EDL）的做法

網路不直接輸出機率，而是輸出每個類別的**證據量（evidence）**，再由證據推算出機率與不確定度。對應程式碼 `losses.py`：

```python
evidence = relu_evidence(output)   # e_k = ReLU(網路輸出)，證據必須 ≥ 0
alpha = evidence + 1               # α_k = e_k + 1（Dirichlet 分布的參數）
```

由 α 可以算出三個量（K 為類別數，MNIST 中 K = 10）：

| 量 | 公式 | 意義 |
|---|---|---|
| 總證據強度 S | $S = \sum_k \alpha_k$ | 模型總共收集到多少證據 |
| 預測機率 | $p_k = \alpha_k / S$ | 預測屬於哪一類 |
| **不確定度** | $u = K / S$ | **證據越少，u 越接近 1，代表「我不知道」** |

當模型對某張圖片完全沒有證據時，所有 α 都等於 1，此時 S = K、u = 1，每個類別的機率都是 1/K，模型等於在說「都不是，我不知道」。

### 3. 損失函數

以 `--mse` 選項使用的 `edl_mse_loss`（論文 Eq. 5）為例，損失由兩部分組成。

**① 預測誤差與變異數（`loglikelihood_loss`）**

```python
loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)          # 預測機率與正確答案的差距
loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)  # 預測的變異數
```

**② KL 散度懲罰項（讓模型學會說「不知道」）**

```python
kl_alpha = (alpha - 1) * (1 - y) + 1    # 移除正確類別的證據，只保留錯誤類別的證據
kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
```

這一項的意思是：**放在錯誤類別上的證據要受罰**。模型因此學到，沒把握時寧可不給證據、讓 u 升高，而不是硬猜一個答案。

**③ 退火（annealing）**

```python
annealing_coef = min(1.0, epoch_num / annealing_step)
```

KL 懲罰的權重會隨 epoch 從 0 逐漸增加到 1。原因是訓練初期模型什麼都還不會，若一開始就重罰，模型會傾向把所有輸入都判為「不知道」。

### 4. 三種損失函數

| 指令參數 | 損失函數 | 對應論文公式 |
|---|---|---|
| `--mse` | 期望均方誤差（Expected Mean Square Error） | Eq. 5 |
| `--digamma` | 期望交叉熵（Expected Cross Entropy） | Eq. 4 |
| `--log` | 期望概似的負對數（Negative Log of the Expected Likelihood） | Eq. 3 |

三者都會加上相同的 KL 懲罰項。

### 5. 實驗設計

- **訓練**：只使用正常的 MNIST 手寫數字。
- **測試分布外資料**的兩種方式：
  - **旋轉數字 1**：從 0° 轉到 180°，讓圖片從「像訓練資料」逐漸變成「沒看過的樣子」，觀察不確定度如何隨之變化。
  - **尤達大師照片**：與手寫數字完全無關的圖片，是最極端的分布外資料。
- **對照設計**：對照組（softmax）與實驗組（EDL）使用相同的網路（LeNet）、資料與優化器，**唯一的差別是損失函數**，因此結果的差異完全來自損失函數的設計。

### 6. 專案結構

```
main.py      ← 程式入口：讀取參數，決定訓練或測試
├── data.py     ← 下載、載入 MNIST
├── lenet.py    ← 網路架構（小型 CNN：LeNet）
├── losses.py   ← ★ 三種 evidential loss（Eq. 3、4、5）與 KL 懲罰項
├── train.py    ← 訓練迴圈
├── test.py     ← 旋轉實驗、單張圖片測試
└── helpers.py  ← 旋轉圖片、one-hot 編碼等工具

延伸實驗（本 fork 新增，見「延伸實驗」一節）
├── exp_common.py      ← 三個實驗共用：讀模型、計算 u 與 α、資料前處理、畫圖設定
├── exp1_threshold.py  ← 實驗 1：不確定就不採用（迷你版 Table 4）
├── exp2_ood.py        ← 實驗 2：自動偵測何時需要重新訓練（迷你版 Table 5）
└── exp3_fgsm.py       ← 實驗 3：FGSM 對抗攻擊（迷你版 Fig. 4）
```

### 7. 應用：視覺式遊戲作弊偵測

論文 [*Robust Vision-Based Cheat Detection in Competitive Gaming*](https://doi.org/10.1145/3451259)（Jonnalagadda et al., I3D 2021）在 §5.2 直接採用本方法（Sensoy Eq. 5 的 MSE 版本）估計作弊偵測器的信心。對應關係如下：

| 本 demo | 作弊偵測論文 |
|---|---|
| 旋轉後的數字 1、尤達照片 | 沒見過的新外掛、新遊戲版本（分布外資料） |
| 答錯時 u 升高 | 只在高信心時才檢舉，避免誤判正常玩家（Table 4：FP 從 96 降到 2） |
| 大量樣本 u 偏高 | 實際部署時信心整體下降，代表需要重新訓練（§5.4） |
| KL 權重隨 epoch 從 0 升到 1 | §6 前 150 個 epoch 將 KL 權重從 0 升到 1 |

對反作弊系統而言，冤枉正常玩家的代價非常高。EDL 讓模型在沒把握時說「我不知道」，系統就能只在模型有把握時才採取行動。

## 🏁 示範

> 本節圖片為原作者訓練 50 個 epoch 的結果。

### 使用 Softmax 交叉熵損失的分類網路

目前深度神經網路的標準做法，是用 softmax 將輸出層的連續值轉換成類別機率。

以下示範 softmax 型的深度神經網路在遇到樣本外（out-of-sample）查詢時如何失效。

網路訓練 50 個 epoch 後，測試準確率約為 98.9%。接著我們對 MNIST 中一個旋轉中的數字進行分類，觀察網路面對不屬於訓練資料分布的樣本時表現如何。

旋轉角度小時，圖片能以很高的機率被正確分類為「1」。然而當旋轉角度變大，即使樣本已被分錯，softmax 算出的分類機率仍然很高。

[![一般網路（無不確定度）](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate.jpg)

**一般網路（無不確定度）**

如上圖所示，以 softmax 機率訓練的神經網路在遇到與訓練樣本不同的輸入時，會嚴重失效。softmax 強迫神經網路選出一個類別，即使該物件屬於未知類別也一樣。當數字 1 旋轉到 60° 至 130° 之間時就出現這種情況：網路在這個範圍內把它誤判為 5 或 7，而且分類機率很高。

### 以期望均方誤差進行具不確定度的分類

理想上，我們希望網路在分類這些屬於未知類別的物件時能表現出「不確定」。要讓網路說出「我不知道」，可以將所有信念質量（belief mass）分配給整個框架（whole frame）。

如論文所述，神經網路可以訓練來學習 Dirichlet 分布的參數，而非 softmax 機率。參數 `α ≥ 1` 的 Dirichlet 分布可視為 softmax 機率（類別分布）的生成模型，它會為每個類別分布給出一個概似值。

[![期望均方誤差](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_mse.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_mse.jpg)

**期望均方誤差**

上圖顯示，此方法對誤判樣本產生的證據遠少於正確分類的樣本。無論在訓練集或測試集，誤判樣本的不確定度都很高（約 1.0），正確分類的樣本則約為 0.2。這表示神經網路對誤判的樣本非常不確定，只對正確分類的樣本給出有把握的預測。換句話說，神經網路也能透過對錯誤預測給出高不確定度，預告自己何時會失敗。

上述網路是以論文中的 Eq. 5 訓練。

### 以期望交叉熵進行具不確定度的分類

本節以論文 Eq. 4 所描述的損失函數訓練神經網路。此損失函數是對預測的 Dirichlet 分布取交叉熵損失的期望值而推導出來的。

[![期望交叉熵](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_digamma.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_digamma.jpg)

**期望交叉熵**

上圖顯示，神經網路對正確分類的樣本產生了多得多的證據，因此正確分類樣本的不確定度非常低（接近 0），而誤判樣本的不確定度則非常高（約 0.7）。

### 以期望概似的負對數進行具不確定度的分類

本節以論文 Eq. 3 的損失函數重複上述實驗。

[![期望概似的負對數](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_log.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/rotate_uncertainty_log.jpg)

**期望概似的負對數**

### 比較樣本內與樣本外的分類

下表顯示網路對一張完全隨機的圖片（尤達大師）的反應：網路的不確定度為 1.0，且對所有類別給出相同的機率。相較之下，對樣本內的數字「1」，不確定度低得多，約 0.15，並以約 0.86 的分類機率有把握地判斷它是 1。

| MNIST 數字 1 | 隨機圖片：尤達大師 |
|---|---|
| [![](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/one.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/one.jpg) | [![](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/yoda.jpg)](https://github.com/dougbrion/pytorch-classification-uncertainty/raw/master/results/yoda.jpg) |

## 🔬 重現結果

以下為本 fork 在新版環境上實際重現的結果，兩組皆訓練 **10 個 epoch**（原作者為 50 個）。

**環境**：Windows、Python 3.13.2、PyTorch 2.15.0（CPU）、torchvision 0.29.2

### 訓練結果

| | 對照組（softmax） | 實驗組（EDL，`--mse`） |
|---|---|---|
| 驗證集準確率 | **98.75%** | **95.10%** |
| 訓練時間（CPU） | 5 分 32 秒 | 5 分 12 秒 |

EDL 的準確率較低，主要原因是 KL 懲罰讓模型在不確定時傾向少給證據，加上只訓練 10 個 epoch，KL 權重才剛升到完整值訓練就結束了。這是**以少許準確率換取不確定度估計**的取捨。

### 旋轉實驗

| 對照組：`results/rotate.jpg` | 實驗組：`results/rotate_uncertainty_mse.jpg` |
|---|---|
| ![對照組旋轉實驗](results/rotate.jpg) | ![實驗組旋轉實驗](results/rotate_uncertainty_mse.jpg) |

各角度（0°、10°、…、180°）的預測結果：

```
對照組：[1, 1, 1, 1, 1, 1, 5, 7, 7, 7, 7, 4, 4, 4, 4, 1, 1, 1, 1]
實驗組：[1, 1, 1, 1, 1, 3, 3, 3, 7, 4, 4, 4, 5, 5, 5, 8, 1, 1, 1]
```

**對照組**：轉到 70°～100° 時被判為 7，**90° 附近的機率約 0.88**；轉到 110°～140° 時被判為 4，機率約 0.4～0.5。也就是說，**模型很有信心地答錯了**。

**實驗組**：

| 旋轉角度 | 數字 1 的機率 | 不確定度 u |
|---|---|---|
| 0°～30° | 約 0.78～0.80 | 約 0.21～0.26（低） |
| 40° | 0.50 | 0.55 |
| **50°～170°** | **0.10** | **約 1.0（完全不確定）** |
| 180° | 0.29 | 0.78 |

在答錯的區間，u 全部升到約 1.0，此時 10 個類別的機率都約為 0.10，代表模型沒有給任何類別證據。因此這個區間的預測結果（3、7、4、5、8）其實沒有意義，模型真正表達的是「我不知道」。

### 單張圖片測試

| 輸入 | 對照組 | 實驗組 |
|---|---|---|
| 數字 1（`one.jpg`） | 預測 1，機率 0.9992 | 預測 1，機率 0.7746，**u = 0.2504** |
| 尤達照片（`yoda.jpg`） | 預測 0，機率 0.2000 | 預測 0，機率 0.4050，**u = 0.6612** |

可以用 $u = K/S$ 反推證據量，驗證 KL 懲罰項的效果：

- **數字 1**：$S = 10 / 0.2504 \approx 40$，類別 1 的 $\alpha \approx 0.7746 \times 40 \approx 31$，即**證據約 30**；其餘 9 類 $\alpha = 0.025 \times 40 = 1$，即**證據為 0**。
- **尤達**：$S = 10 / 0.6612 \approx 15$，類別 0 的**證據約 5**；其餘 9 類**證據為 0**。

錯誤類別的證據被壓到 0，模型只對有把握的類別給證據。數字 1 拿到約 30 份證據，尤達只有約 5 份，因此尤達的不確定度明顯較高。

### 對照組與實驗組比較

| 測試情境 | 對照組（softmax） | 實驗組（EDL） |
|---|---|---|
| 正常數字 1 | 機率 99.9%，正確 | u = 0.25，正確且有把握 |
| 轉 90° 的數字 1 | 判為 7，機率約 88%，**很有信心地答錯** | u ≈ 1.0，**知道自己不知道** |
| 尤達照片 | 機率 20%，沒有不確定度可參考 | u = 0.66，明顯偏高 |
| 能否判斷「該不該相信這次預測」 | ❌ 機率時高時低，沒有規律 | ✅ u 高就不採用 |

若設定門檻「u > 0.5 時不採用預測」：正常的數字 1 會被採用，轉 50°～170° 的所有錯誤預測與尤達照片都會被擋下。

### 限制

1. **轉到 160°～180° 時過於保守**：數字 1 轉 180° 後仍是一條直線，對照組在 160° 就認回來了，實驗組到 170° 的 u 仍接近 1。推測是訓練 epoch 不足所致。
2. **尤達的 u 只有 0.66**：原作者訓練 50 個 epoch 的結果為 1.0。
3. **準確率下降約 3.6 個百分點**。

## 🧪 延伸實驗

論文 [*Robust Vision-Based Cheat Detection in Competitive Gaming*](https://doi.org/10.1145/3451259)（I3D 2021）有三個核心技術。本節用三個實驗，在 MNIST 上做出迷你版，逐一驗證論文的主張。

| 實驗 | 對應論文 | 想回答的問題 | 程式 |
|---|---|---|---|
| 1 | §5.3、Table 4 | 「不確定就不採用」能減少誤判嗎？ | `exp1_threshold.py` |
| 2 | §5.4、Table 5 | 遇到新情境，模型能自己察覺該重新訓練嗎？ | `exp2_ood.py` |
| 3 | §6、Fig. 4、Table 6 | 對抗攻擊下，偵測器還可靠嗎？ | `exp3_fgsm.py` |

三個實驗都直接載入 [重現結果](#-重現結果) 一節訓練好的兩個模型（各訓練 10 個 epoch），**不需要重新訓練**。

### 執行方式

```powershell
python exp1_threshold.py
python exp2_ood.py       # 第一次執行會自動下載 Fashion-MNIST（約 30 MB）
python exp3_fgsm.py
```

每個實驗在 CPU 上約 1～3 分鐘，結果圖會存到 `results/`。`exp_common.py` 是共用工具，會自動偵測 `data.py` 的前處理方式，確保旋轉、雜訊、攻擊後的圖片都跟訓練時使用同一套前處理。

Fashion-MNIST 會下載到 `data/FashionMNIST/`，已加入 `.gitignore`。

### 共用設定

- **有信心的定義**：不確定度 ≤ 0.5。EDL 使用 u；softmax 使用「1 - 最大機率」。
- **預測類別**：EDL 以 logits 的最大值決定，與原作者 `train.py`、`test.py` 一致（原因見 [實作細節](#實作細節u--1-時的預測類別)）。
- **測試資料**：MNIST 測試集 10,000 張。

### 實驗 1：不確定就不採用（迷你版 Table 4）

**做法**：在兩組資料上，把每個預測依「有信心／沒信心」×「答對／答錯」分成四格，仿照論文 Table 4。

- 乾淨 MNIST：跟訓練資料同分布
- 旋轉 MNIST：每張圖隨機旋轉 0°～180°，模擬沒見過的新情境

![實驗 1 結果](results/exp1_threshold.png)

| 資料 | 模型 | 準確率 | 信心高-對 | **信心高-錯** | 信心低-對 | 信心低-錯 | 採用比例 | 採用準確率 | 擋下準確率 | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| 乾淨 MNIST | softmax | 98.8% | 9847 | **100** | 28 | 25 | 99.5% | 99.0% | 52.8% | 0.980 |
| 乾淨 MNIST | EDL | 95.1% | 7655 | **18** | 1855 | 472 | 76.7% | 99.8% | 79.7% | 0.923 |
| 旋轉 MNIST | softmax | 39.2% | 3560 | **4396** | 364 | 1680 | 79.6% | 44.7% | 17.8% | 0.766 |
| 旋轉 MNIST | EDL | 35.4% | 1384 | **503** | 2155 | 5958 | 18.9% | 73.3% | 26.6% | 0.708 |

- **信心高-錯**：模型很有信心卻答錯，對應論文的 FP（冤枉正常玩家），是最危險的錯誤。
- **AUROC**：不確定度能不能分辨對錯。隨機抽一個答錯和一個答對的樣本，答錯那個不確定度比較高的機率；0.5 等於亂猜，1.0 為完美。

**解讀**

1. **同一個門檻下，危險的錯誤大幅減少**：信心高-錯在乾淨資料從 100 降到 18（約 5.6 倍），旋轉資料從 4396 降到 503（約 8.7 倍）。這與論文 Table 4 的 FP 從 96 降到 2 是同一個現象。
2. **但 EDL 的排序能力沒有比較好**：兩組資料的 AUROC 都是 softmax 較高。從圖 (a) 看，若 softmax 也只採用最有信心的 19%，準確率約 83%，反而高於 EDL 的 73.3%。原因在圖 (c)：旋轉資料中約 4,500 張圖的 u 都落在 1.0，EDL 知道自己不知道，卻無法再分辨「不知道的程度」。
3. **EDL 真正的優勢是門檻有絕對意義**：0.5 這個門檻不需要調整，遇到新情境就自動生效。softmax 要達到同樣效果，必須把門檻調到非常嚴格，而那個門檻要先看過新資料才調得出來；實務上不可能事先拿到明天才出現的新外掛來調門檻。
4. **代價是過於保守**：乾淨資料上 EDL 拒答了 23.3%，被擋下的圖有 79.7% 其實答對，推測與只訓練 10 個 epoch 有關。

### 實驗 2：自動偵測何時需要重新訓練（迷你版 Table 5）

**做法**：仿照論文 §5.4 的 likelihood ratio（Eq. 8）。

- 論文是二分類，LR = max(α) / min(α)。MNIST 有 10 類，推廣為「**第一名的 α ÷ 第二名的 α**」，二分類時兩者完全相同。
- 以訓練資料（隨機抽 10,000 張）的平均 LR 為基準，計算各組資料的平均 LR 比例。
- 依論文規則：**比例低於 0.5，就發出「需要重新訓練」警報**。
- 把模型沒看過的資料當成「新外掛」：旋轉 90° 的 MNIST（同一款遊戲，畫面變了）、Fashion-MNIST（換了一款遊戲）、隨機雜訊（完全沒意義的畫面）。

![實驗 2 結果](results/exp2_ood.png)

| 資料 | LR 比例 | 判斷 |
|---|---|---|
| MNIST 測試集 | 1.02 | 正常 |
| 旋轉 90° MNIST | 0.14 | 需要重新訓練 |
| Fashion-MNIST | 0.11 | 需要重新訓練 |
| 隨機雜訊 | 0.04 | 需要重新訓練 |

訓練基準每次都是隨機抽樣，所以比例會有小幅波動（例如 MNIST 測試集在不同次執行為 1.02～1.03）。

**解讀**

1. **四組全部判斷正確，而且完全不需要標準答案**。與論文 Table 5 的結論一致：同分布的資料比例接近 1，換了分布就掉到 0.5 以下。
2. **softmax 對沒看過的東西照樣很有自信**：圖 (a) 中，EDL 在三組新資料上的 u 中位數都是 1.0；softmax 的「1 - 最大機率」中位數只有約 0.2～0.35，而且它對**純雜訊**的信心，甚至比對旋轉 90° 的數字還高。

#### 把衣服丟給只學過數字的模型

![Fashion-MNIST 範例](results/exp2_fashion_examples.png)

softmax 把套頭衫認成「8」（信心 99%），T 恤認成「8」（信心 94%）。各類別的統計（節錄）：

| 衣服類別 | softmax 最常認成 | 占比 | softmax 平均信心 | EDL 平均 u |
|---|---|---|---|---|
| 外套 | 8 | 81.9% | 65.0% | 0.994 |
| 襯衫 | 8 | 85.4% | 64.2% | 0.994 |
| 包包 | 8 | 59.6% | 65.2% | 0.865 |
| 涼鞋 | 2 | 53.3% | 56.4% | 0.804 |
| 運動鞋 | 2 | 77.0% | 70.5% | 0.792 |
| 短靴 | 2 | 57.9% | 73.5% | 0.818 |

- **模型其實有規律地在找最像的數字**：鞋子類大多被認成 **2**（底部一條橫線，像 2 的筆畫），衣服和包包類大多被認成 **8**（一團封閉的形狀）。
- **EDL 對鞋子的平均 u（約 0.79～0.82）比衣服（約 0.99）低**，代表鞋子在它眼中真的比較像數字。
- **失敗案例**：短靴被 EDL 有信心地認成「2」（u = 0.39）。分布外資料剛好長得像訓練資料時，EDL 也會被騙，這是這類方法已知的弱點。

### 實驗 3：FGSM 對抗攻擊（迷你版 Fig. 4）

**做法**：FGSM（Fast Gradient Sign Method）只算一次梯度，把每個像素往「讓模型更錯」的方向推一小步：

$$x_{adv} = \text{clip}\left(x + \epsilon \cdot \text{sign}(\nabla_x \text{Loss}),\ 0,\ 1\right)$$

攻擊強度 ε 從 0 到 0.3（像素值範圍 0～1），比較三種情境：

| 情境 | 說明 |
|---|---|
| softmax（白箱攻擊） | 攻擊者知道 softmax 模型的權重，直接攻擊它 |
| EDL（白箱攻擊） | 攻擊者知道 EDL 模型的權重，直接攻擊它（論文假設的最壞情況） |
| EDL（遷移攻擊） | 用 softmax 模型算出的攻擊去打 EDL；論文提到攻擊者實務上常常只有替代模型 |

![實驗 3 結果](results/exp3_fgsm.png)

![同一張圖片被攻擊的樣子](results/exp3_examples.png)

**白箱攻擊的比較**（ε = 0 為精確值；標「約」者為從圖上讀取的近似值）

| ε | softmax 準確率 | EDL 準確率 | softmax 被騙且有信心 | EDL 被騙且有信心 | softmax 平均不確定度 | EDL 平均 u |
|---|---|---|---|---|---|---|
| 0 | 98.8% | 95.1% | 1.0% | 0.2% | 約 0.02 | 約 0.42 |
| 0.10 | 約 85% | 約 82% | 約 12% | 約 1.5% | 約 0.14 | 約 0.72 |
| 0.20 | 約 32% | 約 31% | 約 52% | 約 3% | 約 0.32 | 約 0.92 |
| 0.30 | 約 2% | 約 13% | 約 72% | 約 5.5% | 約 0.32 | 約 0.93 |

**EDL（遷移攻擊）**

| ε | 準確率 | 有信心比例 | 被騙且有信心 | 平均 u |
|---|---|---|---|---|
| 0.10 | 82.0% | 41.0% | 1.0% | 0.631 |
| 0.15 | 65.0% | 21.5% | 2.4% | 0.780 |
| 0.20 | 40.0% | 11.3% | 4.0% | 0.866 |
| 0.25 | 21.5% | 7.9% | 5.3% | 0.902 |
| 0.30 | 10.8% | 6.6% | 6.0% | 0.913 |

**解讀**

1. **EDL 並沒有比較不會被騙**：兩個模型被攻擊時，準確率掉得差不多快。
2. **差別在被騙之後**：softmax 被騙時還很有信心（ε = 0.3 時約 72%）；EDL 被騙時 u 升到約 0.93，大多會被信心篩選擋下（約 5.5%）。範例圖中，ε = 0.15 時畫面已佈滿雜訊，softmax 仍說「7，信心 93%」，EDL 的 u 已升到 0.61，ε = 0.2 起完全不確定。這呼應論文 Table 6：只換成 uncertainty loss（UL_combined），面對 FGSM 的防禦比例就從 0.62 提升到 0.81。
3. **漏網之魚**：遷移攻擊在 ε = 0.30 時，有信心的預測只剩 6.6%，其中 6.0% 是被騙的，也就是**還敢說有信心的預測，約 91% 都是錯的**。不確定度篩選只能提供實務上的抵抗力，沒有數學保證；這正是論文還要加入 IBP 的原因：IBP 能算出「在 ε 範圍內保證攻擊不了」的下界（Table 6 的 IBP bound）。
4. **遷移攻擊比白箱攻擊弱**：在中等強度下，遷移攻擊後的準確率較高（ε = 0.15：65.0% vs 約 56%；ε = 0.20：40.0% vs 約 31%），符合「攻擊者只有替代模型時效果較差」。

### 綜合結論

| 論文的主張 | 驗證實驗 | 結果 |
|---|---|---|
| 信心分析能大幅減少誤判（§5.3） | 實驗 1 | ✅ 成立，但優勢來自「門檻有絕對意義」，而不是排序能力較好 |
| LR 能自動偵測何時需要重新訓練（§5.4） | 實驗 2 | ✅ 完全成立，4 組資料全部判斷正確 |
| Uncertainty loss 提升對抗攻擊的實務抵抗力（§6） | 實驗 3 | ✅ 成立，被騙且有信心從約 72% 降到約 5.5% |
| 但沒有數學保證，需要 IBP（§6） | 實驗 3 遷移攻擊 | ✅ 強攻擊下漏網的有信心預測約 91% 是錯的 |

一句話總結：**EDL 的價值不是讓模型更準，而是讓「不確定」成為一個可以直接使用的訊號**，系統因此能在沒把握時選擇不行動。

### 限制

- 使用 MNIST 與 LeNet，與論文的 1080p 遊戲畫面、二分類作弊偵測任務差距很大。
- 兩個模型只訓練 10 個 epoch（原作者為 50 個），EDL 偏保守。
- 每個實驗只執行一次，沒有用多個亂數種子取平均；部分數值為從圖上讀取的近似值。
- 對抗攻擊只測試 FGSM（論文另有 universal attack 與 PGD/Madry），也沒有實作 IBP。

### 實作細節：u = 1 時的預測類別

當模型對一張圖完全沒有證據時，所有 α 都等於 1，十個類別平手。若用 α 取最大值決定類別，PyTorch 會固定回傳第一個類別「0」，導致準確率被低估。本 fork 的實驗程式初版就犯了這個錯，把 EDL 在乾淨資料上的準確率算成約 85%，與訓練時回報的 95.1% 不符才發現。

修正方式是改用 logits 的最大值決定類別，與原作者 `train.py`、`test.py` 的做法一致。因為 α = ReLU(logits) + 1 不會改變大小順序，logits 的最大值一定也是 α 的最大值之一。「有信心」的預測至少有一個類別有證據、不會平手，因此「信心高-錯」、「被騙且有信心」、u、LR 等核心數值都不受影響。

## 🎈 使用方式

> 原 README 標示的測試環境為 Python `v3.6.8`、Torch `v1.3.1`、Torchvision `v0.4.2`。這些版本已無法安裝在新版 Python 上，**請勿執行 `pip install -r requirements.txt`**，改用下方的安裝方式。本 fork 已在 Python 3.13.2、PyTorch 2.15.0 上測試通過。

### 1. 下載專案並建立虛擬環境（Windows PowerShell）

```powershell
git clone https://github.com/Gavinhsu0512/pytorch-classification-uncertainty.git
cd pytorch-classification-uncertainty
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

若出現「已停用指令碼執行」的錯誤，先執行以下指令再重新啟動虛擬環境：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. 安裝套件

```powershell
pip install torch torchvision matplotlib numpy scipy pillow
```

本實驗使用 CPU 即可完成，不需要 GPU。

### 3. 參數說明

訓練與測試網路、顯示範例資料都有對應的參數。使用不確定度訓練或測試時，除了 `--uncertainty` 之外，還要指定以下其中一種損失函數：`--mse`、`--digamma`、`--log`。

```
python main.py --help

usage: main.py [-h] [--train] [--epochs EPOCHS] [--dropout] [--uncertainty]
               [--mse] [--digamma] [--log] [--test] [--examples]

optional arguments:
  -h, --help       顯示說明並離開
  --train          訓練網路
  --epochs EPOCHS  訓練的 epoch 數（預設 10）
  --dropout        是否使用 dropout
  --uncertainty    是否使用不確定度
  --mse            使用不確定度時設定。損失函數為期望均方誤差（Eq. 5）
  --digamma        使用不確定度時設定。損失函數為期望交叉熵（Eq. 4）
  --log            使用不確定度時設定。損失函數為期望概似的負對數（Eq. 3）
  --test           測試網路
  --examples       顯示 MNIST 範例資料
```

### 4. 執行

顯示 MNIST 範例圖（第一次執行會自動下載 MNIST）：

```
python main.py --examples
```

訓練對照組（一般 softmax）並測試：

```
python main.py --train --dropout --epochs 10
python main.py --test
```

訓練實驗組（EDL）並測試：

```
python main.py --train --dropout --uncertainty --mse --epochs 50
python main.py --test --uncertainty --mse
```

| 指令 | 產生的檔案 |
|---|---|
| `--examples` | `images/examples.jpg` |
| 對照組 `--train` | `results/model.pt` |
| 對照組 `--test` | `results/rotate.jpg` |
| 實驗組 `--train` | `results/model_uncertainty_mse.pt`（或 `_digamma`、`_log`） |
| 實驗組 `--test` | `results/rotate_uncertainty_mse.jpg`（或 `_digamma`、`_log`） |

## 🐛 已知問題與修正

原始程式碼撰寫於 2019 年（PyTorch 1.3.1），在新版 PyTorch 上執行測試時會發生以下錯誤，本 fork 已修正於 `test.py`。

### 問題 1：對照組測試時出現 `AttributeError`

**執行指令**

```
python main.py --test
```

**錯誤訊息**

```
File "test.py", line 53, in test_single_image
    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))
AttributeError: 'bool' object has no attribute 'item'
```

**原因**

`test_single_image` 中的變數 `uncertainty` 有兩種身分：

- 使用 `--uncertainty` 時，程式會把它重新設為模型算出的不確定度（tensor），tensor 有 `.item()` 方法，所以正常。
- **不使用** `--uncertainty` 時，它仍是函式參數傳入的布林值 `False`，布林值沒有 `.item()`，因此當掉。

原作者應只測試過不確定度模式，未發現對照組會出錯。

**修正方式**

將第 53 行：

```python
plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))
```

改為：

```python
if hasattr(uncertainty, "item"):
    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))
else:
    plt.title("Classified as: {}".format(preds[0]))
```

若 `uncertainty` 是 tensor（有 `.item()`）就顯示不確定度，否則只顯示分類結果。

### 問題 2：實驗組測試時出現 `RuntimeError`

**執行指令**

```
python main.py --test --uncertainty --mse
```

**錯誤訊息**

```
File "test.py", line 141, in rotating_image_classification
    axs[2].plot(ldeg, lu, marker="<", c="red")
...
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

**原因**

測試時程式**沒有關閉梯度追蹤**，因此記錄不確定度的串列 `lu` 裡存的是仍帶有梯度資訊的 tensor。matplotlib 畫圖時需要將它們轉成 numpy 陣列，新版 PyTorch 不允許直接轉換帶梯度的 tensor，於是報錯。舊版 PyTorch 的檢查較寬鬆，所以原作者當時沒有遇到。

（對照組測試時印出的 `grad_fn=<ViewBackward0>` 也是同一個原因造成的。）

**修正方式**

測試本來就不需要計算梯度。確認 `test.py` 開頭有 `import torch`，然後在兩個測試函式的定義**正上方**加上 `@torch.no_grad()` 裝飾器：

```python
@torch.no_grad()
def rotating_image_classification(...):
    ...

@torch.no_grad()
def test_single_image(...):
    ...
```

`@torch.no_grad()` 會讓整個函式執行期間都不追蹤梯度，同時也能節省記憶體。

### 問題 3：`requirements.txt` 版本過舊

`requirements.txt` 指定的是 PyTorch 1.3.1 等 2019 年的版本，無法安裝在 Python 3.11 以上的環境。請改用 [使用方式](#-使用方式) 中的安裝指令。

## 📄 授權

本專案採用 [MIT 授權](LICENSE)，原作者為 Douglas Brion。
