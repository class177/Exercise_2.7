import argparse
import logging
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from tools import networks, raputil

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 常數定義
K = 64
MU = 2
DEFAULT_SNRS = [5, 10, 15, 20, 25, 30, 35, 40]
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50

def parse_args():
    parser = argparse.ArgumentParser(description="Exercise 2.7 Runner - DNN Channel Estimation")
    parser.add_argument("--ce-type", choices=["mmse", "dnn"], default="dnn", help="估測器類型")
    parser.add_argument("--mode", choices=["train", "test"], default="test", help="訓練或測試模式")
    parser.add_argument("--no-cp", action="store_true", help="若設置則不使用 CP (Cyclic Prefix)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--snrs", type=int, nargs="*", default=DEFAULT_SNRS, help="SNR 值 (dB)")
    return parser.parse_args()

def get_save_path(ce_type, cp_flag, snr):
    """ 生成權重存檔路徑 """
    prefix = "CPFREE_" if not cp_flag else ""
    return Path("dnn_ce") / f"CE_DNN_{prefix}{2**MU}QAM_SNR_{snr}dB.npz"

def run(args):
    # 初始化參數
    cp_flag = not args.no_cp
    is_test = (args.mode == "test")
    
    # 建立目錄
    Path("dnn_ce").mkdir(parents=True, exist_ok=True)

    results = {
        "mse_t": [],
        "mse_f": []
    }

    # 設置隨機種子確保可複現性 (TF2 寫法)
    tf.random.set_seed(1)
    np.random.seed(1)

    for snr in args.snrs:
        logging.info(f"Processing SNR: {snr} dB")
        
        # 建立網路
        # 註：這裡假設你的 networks.build_ce_dnn 內部已處理好 TF1/TF2 相容性
        # 如果要完全轉為 TF2，建議將 build_ce_dnn 改寫為 tf.keras.Model
        savefile = str(get_save_path(args.ce_type, cp_flag, snr))
        
        sess, input_holder, output = networks.build_ce_dnn(
            K, snr, 
            training_epochs=args.epochs, 
            batch_size=args.batch_size,
            savefile=savefile, 
            test_flag=is_test, 
            cp_flag=cp_flag,
            nh1=500, nh2=250
        )

        if is_test:
            mse_t, mse_f = raputil.test_ce(
                sess, input_holder, output, snr,
                est_type=args.ce_type, 
                CP_flag=cp_flag, 
                num_trail=args.trials
            )
            results["mse_t"].append(mse_t)
            results["mse_f"].append(mse_f)
            logging.info(f"MSE_F for SNR {snr}: {mse_f}")

        # 必須重置 Graph，避免記憶體洩漏
        tf.compat.v1.reset_default_graph()

    # 輸出最終結果
    if is_test:
        print("\n" + "="*30)
        print(f"Final MSE_F: {results['mse_f']}")
        
        # 存檔
        cp_suffix = "_CP_FREE" if not cp_flag else ""
        mat_file_name = f"MSE_{args.ce_type}_{2**MU}QAM{cp_suffix}"
        sio.savemat(f"{mat_file_name}.mat", {mat_file_name: results["mse_f"]})
        logging.info(f"Results saved to {mat_file_name}.mat")

if __name__ == "__main__":
    run(parse_args())
