import argparse
import json
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except Exception:
    torch_fidelity = None


@dataclass
class Config:
    mode: str = "DSWD"
    seed: int = 42
    data_root: str = "./data"
    output_dir: str = "./runs"

    image_size: int = 64
    num_channels: int = 3
    latent_size: int = 100
    hidden_channels: int = 64

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 10

    lr_g: float = 5e-4
    lr_d: float = 5e-4
    lr_t: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    num_projections: int = 256
    p: int = 2
    selector_steps: int = 1
    use_diversity_reg: bool = True
    diversity_weight: float = 10.0
    mix_ratio: float = 0.15
    topk_ratio: float = 0.5
    gp_lambda: float = 10.0

    # FID / KID evaluation
    compute_fid_kid: bool = True
    eval_every: int = 5
    eval_samples: int = 2000
    kid_subsets: int = 10
    kid_subset_size: int = 100
    fidelity_batch_size: int = 128


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    def __init__(self, latent_size=100, num_channels=3, hidden_channels=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), z.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self, latent_size=100, num_channels=3, hidden_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(nn.Conv2d(hidden_channels * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        feat = self.features(x)
        y = self.out(feat).view(x.size(0), 1)
        return y, feat.view(x.size(0), -1)


class TransformNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, theta):
        return F.normalize(self.linear(theta), dim=1)


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def sample_random_projections(num_projections, dim, device):
    theta = torch.randn(num_projections, dim, device=device)
    return F.normalize(theta, dim=1)


def projectionwise_swd(x, y, theta, p=2):
    proj_x = torch.sort(x @ theta.t(), dim=0)[0]
    proj_y = torch.sort(y @ theta.t(), dim=0)[0]
    return (proj_x - proj_y).abs().pow(p).mean(dim=0).pow(1.0 / p)


def sliced_wasserstein_distance(x, y, theta, p=2):
    return projectionwise_swd(x, y, theta, p=p).mean()


def maybe_select_topk_theta(theta, real_feat, fake_feat, cfg: Config):
    if cfg.mode != "DSWD_TOPK":
        return theta
    with torch.no_grad():
        score = projectionwise_swd(real_feat.detach(), fake_feat.detach(), theta.detach(), p=cfg.p)
        k = max(1, int(round(theta.shape[0] * cfg.topk_ratio)))
        idx = torch.topk(score, k=k, largest=True).indices
    return F.normalize(theta.index_select(0, idx), dim=1)


def selector_forward(selector: Optional[nn.Module], theta0, cfg: Config):
    if selector is None or cfg.mode == "DSW":
        return theta0
    theta = selector(theta0)
    if cfg.mode == "DSWD_MIX":
        theta = F.normalize((1.0 - cfg.mix_ratio) * theta + cfg.mix_ratio * theta0, dim=1)
    return theta


def direction_diversity_penalty(theta: torch.Tensor, eps: float = 1e-8):
    theta = F.normalize(theta, dim=1)
    cos_mat = theta @ theta.t()
    L = theta.shape[0]
    eye = torch.eye(L, device=theta.device, dtype=theta.dtype)
    off = cos_mat - eye
    return off.pow(2).sum() / (L * (L - 1) + eps)


def gradient_penalty(D, real_images, fake_images):
    bs = real_images.shape[0]
    alpha = torch.rand(bs, 1, 1, 1, device=real_images.device)
    interp = alpha * real_images + (1.0 - alpha) * fake_images
    interp.requires_grad_(True)
    pred, _ = D(interp)
    grad = torch.autograd.grad(pred.sum(), interp, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(bs, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def apply_spectral_norm_to_discriminator(D):
    for m in D.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not hasattr(m, "weight_u"):
            nn.utils.spectral_norm(m)


def make_models(cfg: Config, device):
    G = Generator(cfg.latent_size, cfg.num_channels, cfg.hidden_channels).to(device)
    D = Discriminator(cfg.latent_size, cfg.num_channels, cfg.hidden_channels).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    if cfg.mode == "DSWD_SN":
        apply_spectral_norm_to_discriminator(D)

    feat_dim = cfg.hidden_channels * 8 * 4 * 4
    selector = None if cfg.mode == "DSW" else TransformNet(feat_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    opt_t = None if selector is None else torch.optim.Adam(selector.parameters(), lr=cfg.lr_t, betas=(cfg.beta1, cfg.beta2))
    return G, D, selector, opt_g, opt_d, opt_t


def to_uint8_batch(images: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,255] uint8
    x = (images.clamp(-1, 1) + 1.0) * 127.5
    return x.round().to(torch.uint8)


def save_uint8_tensor_dir(images_uint8: torch.Tensor, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images_uint8):
        save_image(img.float() / 255.0, out_dir / f"{i:06d}.png")


def ensure_real_eval_dir(cfg: Config, out_dir: Path):
    real_dir = out_dir / "_fid_cache" / "real"
    marker = real_dir / ".done"
    if marker.exists():
        return real_dir

    if real_dir.exists():
        shutil.rmtree(real_dir)
    real_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_ds = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=tfm)

    imgs = []
    for i in range(min(len(test_ds), cfg.eval_samples)):
        x, _ = test_ds[i]
        imgs.append(x)
    images = torch.stack(imgs, dim=0)
    save_uint8_tensor_dir(to_uint8_batch(images), real_dir)
    marker.write_text("ok", encoding="utf-8")
    return real_dir


@torch.no_grad()
def evaluate_fid_kid(cfg: Config, G: nn.Module, device: torch.device, out_dir: Path, epoch: int):
    if not cfg.compute_fid_kid:
        return None
    if torch_fidelity is None:
        warnings.warn("torch-fidelity not installed, skip FID/KID evaluation.")
        return None

    real_dir = ensure_real_eval_dir(cfg, out_dir)
    fake_dir = out_dir / "_fid_cache" / "fake" / f"epoch_{epoch:03d}"
    if fake_dir.exists():
        shutil.rmtree(fake_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)

    G.eval()
    collected = 0
    chunks = []
    while collected < cfg.eval_samples:
        bs = min(256, cfg.eval_samples - collected)
        z = torch.randn(bs, cfg.latent_size, device=device)
        chunks.append(G(z).detach().cpu())
        collected += bs
    fake_images = torch.cat(chunks, dim=0)
    save_uint8_tensor_dir(to_uint8_batch(fake_images), fake_dir)

    metrics = torch_fidelity.calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=(device.type == "cuda"),
        fid=True,
        kid=True,
        kid_subsets=cfg.kid_subsets,
        kid_subset_size=cfg.kid_subset_size,
        batch_size=cfg.fidelity_batch_size,
        verbose=False,
    )
    return {
        "fid": float(metrics["frechet_inception_distance"]),
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
    }


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    out_dir = Path(cfg.output_dir) / cfg.mode / f"seed_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=tfm)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    G, D, selector, opt_g, opt_d, opt_t = make_models(cfg, device)
    bce = nn.BCELoss()

    history = []
    fixed_z = torch.randn(64, cfg.latent_size, device=device)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        meters = {k: [] for k in ["g_sw", "d_total", "d_gp", "selector_obj"]}

        for real_images, _ in loader:
            real_images = real_images.to(device, non_blocking=True)
            bs = real_images.shape[0]

            # D step
            z = torch.randn(bs, cfg.latent_size, device=device)
            fake_images = G(z).detach()
            real_label = torch.ones((bs, 1), device=device)
            fake_label = torch.zeros((bs, 1), device=device)

            opt_d.zero_grad(set_to_none=True)
            y_real, _ = D(real_images)
            y_fake, _ = D(fake_images)
            loss_d = bce(y_real, real_label) + bce(y_fake, fake_label)
            gp_val = torch.tensor(0.0, device=device)
            if cfg.mode == "DSWD_GP":
                gp_val = gradient_penalty(D, real_images.detach(), fake_images.detach())
                loss_d = loss_d + cfg.gp_lambda * gp_val
            loss_d.backward()
            opt_d.step()

            # G + selector step
            z = torch.randn(bs, cfg.latent_size, device=device)
            fake_images = G(z)
            with torch.no_grad():
                _, real_feat = D(real_images)
            _, fake_feat = D(fake_images)

            selector_obj = torch.tensor(0.0, device=device)
            if selector is not None and opt_t is not None:
                for _ in range(cfg.selector_steps):
                    theta0 = sample_random_projections(cfg.num_projections, real_feat.shape[1], device)
                    theta = selector_forward(selector, theta0, cfg)
                    theta = maybe_select_topk_theta(theta, real_feat, fake_feat, cfg)
                    sw = sliced_wasserstein_distance(real_feat.detach(), fake_feat.detach(), theta, p=cfg.p)
                    selector_obj = sw
                    loss_t = -sw
                    if cfg.use_diversity_reg:
                        loss_t = loss_t + cfg.diversity_weight * direction_diversity_penalty(theta)
                    opt_t.zero_grad(set_to_none=True)
                    loss_t.backward()
                    opt_t.step()

            with torch.no_grad():
                theta0 = sample_random_projections(cfg.num_projections, real_feat.shape[1], device)
                theta = selector_forward(selector, theta0, cfg)
                theta = maybe_select_topk_theta(theta, real_feat, fake_feat, cfg)

            g_sw = sliced_wasserstein_distance(real_feat, fake_feat, theta, p=cfg.p)
            opt_g.zero_grad(set_to_none=True)
            g_sw.backward()
            opt_g.step()

            meters["g_sw"].append(float(g_sw.item()))
            meters["d_total"].append(float(loss_d.item()))
            meters["d_gp"].append(float(gp_val.item()))
            meters["selector_obj"].append(float(selector_obj.item()))

        with torch.no_grad():
            samples = G(fixed_z)
            save_image((samples + 1) * 0.5, out_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 2),
            **{k: float(np.mean(v)) for k, v in meters.items()},
        }
        eval_metrics = None
        if cfg.compute_fid_kid and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            eval_metrics = evaluate_fid_kid(cfg, G, device, out_dir, epoch)
        if eval_metrics is not None:
            row.update(eval_metrics)
        else:
            row.update({"fid": None, "kid_mean": None, "kid_std": None})
        history.append(row)
        if row["fid"] is None:
            print(f"[{cfg.mode}] epoch {epoch:03d}/{cfg.epochs} | g_sw={row['g_sw']:.4f} d={row['d_total']:.4f} gp={row['d_gp']:.4f}")
        else:
            print(
                f"[{cfg.mode}] epoch {epoch:03d}/{cfg.epochs} | "
                f"g_sw={row['g_sw']:.4f} d={row['d_total']:.4f} gp={row['d_gp']:.4f} "
                f"FID={row['fid']:.3f} KID={row['kid_mean']:.6f}±{row['kid_std']:.6f}"
            )

        torch.save({
            "epoch": epoch,
            "cfg": asdict(cfg),
            "G": G.state_dict(),
            "D": D.state_dict(),
            "selector": None if selector is None else selector.state_dict(),
        }, out_dir / "last.pt")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 DSW/DSWD variants on local Windows/Linux")
    p.add_argument("--mode", type=str, default="DSWD", choices=["DSW", "DSWD", "DSWD_MIX", "DSWD_TOPK", "DSWD_SN", "DSWD_GP"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--lr-g", type=float, default=5e-4)
    p.add_argument("--lr-d", type=float, default=5e-4)
    p.add_argument("--lr-t", type=float, default=1e-4)
    p.add_argument("--num-projections", type=int, default=256)
    p.add_argument("--selector-steps", type=int, default=1)
    p.add_argument("--mix-ratio", type=float, default=0.15)
    p.add_argument("--topk-ratio", type=float, default=0.5)
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--compute-fid-kid", action="store_true", default=True)
    p.add_argument("--no-compute-fid-kid", action="store_false", dest="compute_fid_kid")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--kid-subsets", type=int, default=10)
    p.add_argument("--kid-subset-size", type=int, default=100)
    p.add_argument("--fidelity-batch-size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        output_dir=args.output_dir,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lr_t=args.lr_t,
        num_projections=args.num_projections,
        selector_steps=args.selector_steps,
        mix_ratio=args.mix_ratio,
        topk_ratio=args.topk_ratio,
        gp_lambda=args.gp_lambda,
        compute_fid_kid=args.compute_fid_kid,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        fidelity_batch_size=args.fidelity_batch_size,
    )
    train(cfg)


if __name__ == "__main__":
    main()
