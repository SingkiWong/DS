@echo off
REM Windows local runner for CIFAR-10 DSW/DSWD variants
python -m pip install -r requirements.txt

REM Example: baseline DSW
python train_cifar10_dsw.py --mode DSW --epochs 10 --batch-size 128 --data-root .\data --output-dir .\runs
REM FID/KID is enabled by default. To disable: add --no-compute-fid-kid

REM Example: DSWD with spectral normalization
REM python train_cifar10_dsw.py --mode DSWD_SN --epochs 10 --batch-size 128 --data-root .\data --output-dir .\runs

REM Example: DSWD with gradient penalty
REM python train_cifar10_dsw.py --mode DSWD_GP --epochs 10 --batch-size 128 --gp-lambda 10.0 --data-root .\data --output-dir .\runs
