from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 使用非互動式後端，方便在無螢幕環境（例如伺服器 / Colab）存圖
import matplotlib.pyplot as plt
import numpy as np


# 訓練時使用的 SNR（dB）列表，用來對應各個 .npz 檔
SNR_DB = [5, 10, 15, 20, 25, 30, 35, 40]


def load_loss(npz_path: Path):
    """
    從一個儲存訓練資訊的 .npz 檔載入驗證 loss 曲線。

    參數
    ----
    npz_path : Path
        .npz 檔案的完整路徑。

    回傳
    ----
    loss_history : np.ndarray
        每次測試時的驗證 loss（1 維陣列）。
    test_step : int
        每隔多少 epoch 做一次驗證（對應原始 training code 的 test_step）。
    """
    data = np.load(npz_path, allow_pickle=True)
    loss_history = np.asarray(data["loss_history"]).reshape(-1)
    test_step = int(np.asarray(data["test_step"]).item())
    return loss_history, test_step


def plot_one_group(prefix: str, title: str, output_name: str):
    """
    繪製一組（有 CP 或無 CP）DNN 通道估測器在不同 SNR 下的訓練曲線。

    會依序讀取像是：
      dnn_ce / {prefix}{SNR}dB.npz
    的檔案，例如：
      dnn_ce / CE_DNN_4QAM_SNR_5dB.npz

    參數
    ----
    prefix : str
        檔名前綴字串，不含 SNR 與副檔名，例如：
        - "CE_DNN_4QAM_SNR_"
        - "CE_DNN_CPFREE_4QAM_SNR_"
    title : str
        圖片標題。
    output_name : str
        輸出的 PNG 檔名。
    """
    plt.figure(figsize=(9, 6))

    found = False  # 確認至少找到一個對應的 .npz 檔案
    for snr in SNR_DB:
        npz_path = Path("dnn_ce") / f"{prefix}{snr}dB.npz"
        if not npz_path.exists():
            print(f"Skip missing file: {npz_path}")
            continue

        loss_history, test_step = load_loss(npz_path)
        # x 軸為 epoch 編號：每 test_step 個 epoch 有一個 loss
        epochs = np.arange(len(loss_history)) * test_step
        plt.plot(epochs, loss_history, linewidth=2, label=f"SNR = {snr} dB")
        found = True

    if not found:
        print(f"No files found for prefix: {prefix}")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_name)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path.resolve()}")
    plt.close()


def main():
    # ① 含 CP 的 DNN 通道估測器訓練曲線
    plot_one_group(
        prefix="CE_DNN_4QAM_SNR_",  # 對應 dnn_ce/CE_DNN_4QAM_SNR_5dB.npz 等
        title="Training Curves of DNN Channel Estimator (with CP)",
        output_name="training_curves_dnn_with_cp.png",
    )

    # ② 無 CP 的 DNN 通道估測器訓練曲線
    plot_one_group(
        prefix="CE_DNN_CPFREE_4QAM_SNR_",  # 對應 dnn_ce/CE_DNN_CPFREE_4QAM_SNR_5dB.npz 等
        title="Training Curves of DNN Channel Estimator (without CP)",
        output_name="training_curves_dnn_without_cp.png",
    )


if __name__ == "__main__":
    main()
