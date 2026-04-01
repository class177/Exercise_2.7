import argparse
from pathlib import Path
import numpy as np

def get_pdp(n_taps: int, decay: float) -> np.ndarray:
    """計算歸一化的指數功率延遲譜。"""
    taps = np.arange(n_taps, dtype=np.float64)
    pdp = np.exp(-decay * taps)
    return pdp / pdp.sum()

def gen_rayleigh_channels(n_samples: int, n_taps: int, decay: float, rng: np.random.Generator) -> np.ndarray:
    """生成歸一化的複數瑞利通道樣本。"""
    pdp = get_pdp(n_taps, decay)
    sigma = np.sqrt(pdp / 2.0).astype(np.float32)
    
    # 生成複數隨機樣本並套用 PDP
    h = (rng.standard_normal((n_samples, n_taps), dtype=np.float32) + 
         1j * rng.standard_normal((n_samples, n_taps), dtype=np.float32)) * sigma
    
    # 樣本功率歸一化
    power = np.sum(np.abs(h)**2, axis=1, keepdims=True)
    h_norm = h / np.sqrt(np.maximum(power, 1e-12))
    
    return h_norm.astype(np.complex64)

def save_data(file_path: Path, data: np.ndarray) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, data)

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Exercise 2.7 的通道數據集")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=390000)
    parser.add_argument("--taps", type=int, default=16, dest="n_taps")
    parser.add_argument("--decay", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Generating: Train={args.train_size}, Test={args.test_size}, Taps={args.n_taps}")

    # 生成數據
    h_train = gen_rayleigh_channels(args.train_size, args.n_taps, args.decay, rng)
    h_test = gen_rayleigh_channels(args.test_size, args.n_taps, args.decay, rng)

    # 儲存檔案
    save_data(args.out_dir / "channel_train.npy", h_train)
    save_data(args.out_dir / "channel_test.npy", h_test)

    print(f"Saved to {args.out_dir}. Train shape: {h_train.shape}, Test shape: {h_test.shape}")

if __name__ == "__main__":
    main()
