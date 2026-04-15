# Windows 本地运行说明（DSW/DSWD）

## 1) 环境
建议 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) 运行
```powershell
python train_cifar10_dsw.py --mode DSW --epochs 10 --batch-size 128
python train_cifar10_dsw.py --mode DSWD --epochs 10 --batch-size 128
python train_cifar10_dsw.py --mode DSWD_MIX --epochs 10 --mix-ratio 0.15
python train_cifar10_dsw.py --mode DSWD_TOPK --epochs 10 --topk-ratio 0.5
python train_cifar10_dsw.py --mode DSWD_SN --epochs 10
python train_cifar10_dsw.py --mode DSWD_GP --epochs 10 --gp-lambda 10.0

# 关闭 FID/KID（默认开启）
python train_cifar10_dsw.py --mode DSWD --epochs 10 --no-compute-fid-kid
```

## 3) 输出
默认输出在 `./runs/<mode>/seed_<seed>/`：
- `history.json`：每个 epoch 的指标（`g_sw`, `d_total`, `d_gp`, `selector_obj`, `fid`, `kid_mean`, `kid_std`）
- `samples/epoch_XXX.png`：固定噪声采样图
- `last.pt`：最新 checkpoint

## 4) 与 Notebook 对应关系
- `DSW`: 随机投影，不训练 selector
- `DSWD`: 训练 `TransformNet` 投影
- `DSWD_MIX`: 在 DSWD 基础上加入随机方向残差混合
- `DSWD_TOPK`: 在 DSWD 基础上加入 top-k 投影筛选
- `DSWD_SN`: 在 DSWD 基础上仅加判别器 Spectral Normalization
- `DSWD_GP`: 在 DSWD 基础上仅加判别器 Gradient Penalty
