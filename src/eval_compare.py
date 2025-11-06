# -*- coding: utf-8 -*-
"""
FAST Evaluation & Visualization for 3D voxel-wise predictions.

What this does (single pass over indices):
  - Tripanel maps for a small random subset (for quick visual sanity checks)
  - Global statistics (NO per-index tables):
      * RMSE, MAE, Pearson r  (global, voxel-wise)
      * log10(rho_pred/rho_true): global mean (bias) and std
      * Kolmogorov–Smirnov statistic + KS sample count (adjacent columns)
  - Aggregates and figures:
      * Joint PDF (truth vs pred) with 1σ/2σ/3σ contours
      * Density PDF (truth vs pred)
      * Two-point correlation ξ(r): median + 1σ band (truth & pred)
      * Optional loss curve if --loss_csv is provided
  - Split summary tables:
      * tables/summary_overall.{csv,tex}
      * tables/summary_metrics.{csv,tex}
      * tables/summary_logratio.{csv,tex}

Author: Mingyeong Yang (UST-KASI)
Last-Modified: 2025-11-03
"""

from __future__ import annotations
import os, re, glob, argparse, yaml, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp

# ----------------------------
# IO helpers
# ----------------------------
def _s(a): return np.squeeze(np.asarray(a))

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _collect_indices(pred_dir: str) -> list[int]:
    out = []
    for fn in os.listdir(pred_dir):
        m = re.match(r"(\d+)\.hdf5$", fn)
        if m: out.append(int(m.group(1)))
    return sorted(out)

def _read_truth(yaml_cfg: dict, idx: int):
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]
    if "*" in test_rel:
        test_path = os.path.join(base, test_rel.replace("*", str(idx)))
    else:
        test_path = os.path.join(base, test_rel, f"{idx}.hdf5")
    with h5py.File(test_path, "r") as f:
        arr = _s(f["output_rho"][:])
    return arr, test_path

def _read_pred(pred_dir: str, idx: int):
    path = os.path.join(pred_dir, f"{idx}.hdf5")
    with h5py.File(path, "r") as f:
        arr = _s(f["prediction"][:])
        # pull model attrs if present (used later for metadata)
        attrs = dict(f.attrs)
    return arr, path, attrs

def _read_alex(alex_tpl: str, idx: int):
    pats = sorted(glob.glob(alex_tpl.format(idx=idx)))
    if not pats:
        raise FileNotFoundError(f"No Alex npy for idx={idx} using pattern={alex_tpl}")
    return _s(np.load(pats[0])), pats[0]

def _maybe_delta_to_rho(arr: np.ndarray, force: str|None):
    """
    force: "true" | "false" | None
    """
    if force == "true":
        return arr + 1.0, True
    if force == "false":
        return arr, False
    if arr.min() < 0 or arr.max() < 1.0:
        return arr + 1.0, True
    return arr, False

# ----------------------------
# math helpers
# ----------------------------
def get_slice(vol3d: np.ndarray, axis: int, idx: int|str="center"):
    if isinstance(idx, str) and idx == "center":
        idx = vol3d.shape[axis] // 2
    if axis == 0:   return vol3d[idx, :, :]
    if axis == 1:   return vol3d[:, idx, :]
    if axis == 2:   return vol3d[:, :, idx]
    raise ValueError("axis must be 0/1/2")

def log1p10(a: np.ndarray):
    return np.log10(1.0 + np.clip(a, 0, None))

def autocorr_fft(delta: np.ndarray):
    F = np.fft.fftn(delta)
    xi = np.fft.ifftn(np.abs(F)**2).real / delta.size
    return np.fft.fftshift(xi)

def radial_profile(vol: np.ndarray, voxel_size: float, r_max: float, n_bins: int):
    nz, ny, nx = vol.shape
    cz, cy, cx = (np.array(vol.shape)//2)
    z = (np.arange(nz)-cz)*voxel_size
    y = (np.arange(ny)-cy)*voxel_size
    x = (np.arange(nx)-cx)*voxel_size
    Z, Y, X = np.meshgrid(z,y,x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    edges = np.linspace(0.0, r_max, n_bins+1)
    r = 0.5*(edges[1:]+edges[:-1])
    inds = np.digitize(R.ravel(), edges)-1
    prof = np.zeros(n_bins); v = vol.ravel()
    for i in range(n_bins):
        m = inds==i
        prof[i] = np.mean(v[m]) if np.any(m) else np.nan
    return r, prof

def contour_levels_from_hist2d(H: np.ndarray, probs=(0.68, 0.95, 0.997)):
    H = np.asarray(H, dtype=float)
    s = H.sum()
    if s <= 0 or not np.isfinite(s):
        return []

    Hn = H / s  # 확률 스케일
    flat = np.sort(Hn.ravel())[::-1]      # 내림차순
    cumsum = np.cumsum(flat)

    levels_prob = []
    for p in probs:
        idx = np.searchsorted(cumsum, p)
        t = flat[min(idx, flat.size - 1)] # 해당 커버리지의 최소 bin 높이(확률)
        levels_prob.append(t)

    # 1) 카운트 스케일로 환원
    levels = (np.asarray(levels_prob) * s)
    # 2) contour 요구사항: 엄밀히 증가하도록 정렬 + 중복 제거
    levels = np.unique(np.sort(levels))
    # 3) 레벨이 하나면 작은 epsilon로 분리
    if levels.size < 2:
        eps = np.finfo(float).eps * (levels[0] if levels.size==1 else 1.0)
        levels = np.array([levels[0] - eps, levels[0] + eps])
    return levels


def median_and_band(X: np.ndarray):
    med = np.nanmedian(X, axis=0)
    lo  = np.nanpercentile(X, 16, axis=0)
    hi  = np.nanpercentile(X, 84, axis=0)
    return med, lo, hi

# ----------------------------
# plotting helpers
# ----------------------------
def save_tripanel_maps(idx: int, truth: np.ndarray, alex: np.ndarray, pred: np.ndarray,
                       out_png: str, axis: int, slice_idx: int|str):
    t = get_slice(truth, axis, slice_idx)
    a = get_slice(alex,  axis, slice_idx)
    p = get_slice(pred,  axis, slice_idx)
    t_img, a_img, p_img = log1p10(t), log1p10(a), log1p10(p)
    vmin = min(t_img.min(), a_img.min(), p_img.min())
    vmax = max(t_img.max(), a_img.max(), p_img.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = [
        "Truth — ρ : log$_{10}$(1+ρ)",
        "Alex — $\\hat{\\rho}$ : log$_{10}$(1+ρ)",
        "Ours — $\\hat{\\rho}$ : log$_{10}$(1+ρ)"
    ]
    for ax, img, title in zip(axes, [t_img, a_img, p_img], titles):
        im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log$_{10}$(1+ρ)")
    plt.savefig(out_png, dpi=200); plt.close(fig)

def save_loss_curve_from_csv(csv_path: str, out_png: str, title: str = "Training and Validation Loss"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("epoch", "train_loss", "val_loss", "lr"):
        if col not in df.columns:
            raise ValueError(f"[loss-curve] Missing column '{col}' in {csv_path}")
    df = df.sort_values("epoch").reset_index(drop=True)

    best_idx = int(df["val_loss"].idxmin())
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_loss"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"],   label="Validation Loss", linewidth=2, linestyle="--")
    ax1.axvline(best_ep, color="gray", linestyle=":", lw=1.5)
    ax1.scatter([best_ep], [best_val], s=30, color="gray", zorder=3)
    ax1.text(best_ep + 0.3, best_val, f"Best val={best_val:.3f}\n@ epoch {best_ep}", fontsize=9)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3); ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["lr"], label="Learning Rate", alpha=0.7)
    ax2.set_yscale("log"); ax2.set_ylabel("Learning Rate (log)")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")

    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[DONE] Loss curve → {out_png} (best epoch = {best_ep}, best val = {best_val:.4f})")

def save_aggregate_plots(cache: dict, out_dir: str, label_pred: str):
    rho_edges   = cache["rho_edges"]
    logt_edges  = cache["logt_edges"]; logp_edges = cache["logp_edges"]
    Hm          = cache["Hm"]
    hT, hP      = cache["hT"], cache["hP"]
    r           = cache["r"]
    xiT, xiP    = cache["xiT"], cache["xiP"]

    rho_centers = 0.5*(rho_edges[1:]+rho_edges[:-1])

    # 1) Joint PDF + contours (Truth vs Pred)
    levels = contour_levels_from_hist2d(Hm, probs=(0.68, 0.95, 0.997))
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    Hm_plot = np.ma.masked_where(Hm <= 0, Hm)
    pcm = ax.pcolormesh(10**logt_edges, 10**logp_edges, Hm_plot.T, shading='auto',
                        norm=plt.matplotlib.colors.LogNorm(), cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax); cb.set_label("counts (log)")
    CS = ax.contour(10**logt_edges[:-1], 10**logp_edges[:-1], Hm.T,
                    levels=levels, colors='white', linewidths=1.5)
    ax.clabel(CS, inline=True, fontsize=8, fmt={levels[0]:"1σ", levels[1]:"2σ", levels[2]:"3σ"})
    xmin, xmax = 10**logt_edges[0], 10**logt_edges[-1]
    ax.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\rho_{\rm truth}/\rho_0$")
    ax.set_ylabel(r"$\rho_{\rm pred}/\rho_0$")
    ax.set_title(f"Joint PDF: Truth vs {label_pred}")
    plt.savefig(os.path.join(out_dir, "joint_pdf_pred.png"), dpi=200); plt.close(fig)

    # 2) Density PDF
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.plot(rho_centers, hT, color='black', lw=2, label='Truth')
    ax.plot(rho_centers, hP, color='tab:blue', lw=2, label=label_pred)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\rho/\rho_0$")
    ax.set_ylabel(r"$df/d\log_{10}\rho$")
    ax.set_title("Density PDF")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "pdf_density.png"), dpi=200); plt.close(fig)

    # 3) ξ(r): median + 1σ band
    t_med, t_lo, t_hi = median_and_band(xiT)
    p_med, p_lo, p_hi = median_and_band(xiP)
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.fill_between(r, t_lo, t_hi, color='gray', alpha=0.3, label='Truth 1σ')
    ax.plot(r, t_med, color='black', lw=2, label='Truth median')
    ax.fill_between(r, p_lo, p_hi, color='tab:blue', alpha=0.25, label=f'{label_pred} 1σ')
    ax.plot(r, p_med, color='tab:blue', lw=2, label=f'{label_pred} median')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\langle \delta(\mathbf{x})\delta(\mathbf{x}+\mathbf{r}) \rangle$")
    ax.set_title("Two-point correlation ξ(r)")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "xi_two_point.png"), dpi=200); plt.close(fig)

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="FAST eval: random few maps + global metrics (single pass).")
    # data/config
    ap.add_argument("--yaml_path",     type=str, required=True)
    ap.add_argument("--pred_dir",      type=str, required=True, help="Prediction dir containing <index>.hdf5")
    ap.add_argument("--alex_tpl",      type=str, required=True, help="Alex npy template, e.g., /.../test_{idx}_final_*.npy")
    ap.add_argument("--out_dir",       type=str, required=True)
    # (optional) model metadata to enrich tables
    ap.add_argument("--model_ckpt",    type=str, default=None, help=".pt path for param count (optional)")
    ap.add_argument("--flops_g",       type=float, default=None, help="FLOPs in G (optional, if measured elsewhere)")
    # slicing for map figures
    ap.add_argument("--slice_axis",    type=int, default=2, choices=[0,1,2])
    ap.add_argument("--slice_index",   type=str, default="center", help='"center" or integer index')
    # speed / sampling
    ap.add_argument("--map_count",     type=int, default=5, help="Number of random indices for map saving")
    ap.add_argument("--rng_seed",      type=int, default=42)
    # alex delta->rho handling
    ap.add_argument("--alex_force",    type=str, choices=["true","false"], default=None)
    # aggregates
    ap.add_argument("--joint_sample",  type=int, default=50000, help="pairs per index for joint hist (max)")
    ap.add_argument("--pdf_bins",      type=int, default=120)
    ap.add_argument("--joint_bins",    type=int, default=120)
    ap.add_argument("--voxel_size",    type=float, default=205.0/250.0, help="h^-1 Mpc per voxel")
    ap.add_argument("--rmax",          type=float, default=10.0)
    ap.add_argument("--n_r_bins",      type=int, default=24)
    # global stats (no per-index)
    ap.add_argument("--ks_global_cap", type=int, default=200_000, help="Max total samples used for global KS")
    # labeling
    ap.add_argument("--label_pred",    type=str, default="Ours")
    # tables
    ap.add_argument("--save_latex",    action="store_true")
    # (optional) loss curve
    ap.add_argument("--loss_csv",      type=str, default=None,
                    help="If provided, plot loss curve from CSV (epoch,train_loss,val_loss,lr)")
    args = ap.parse_args()

    # ---------------- dirs & loss curve
    yaml_cfg = _load_yaml(args.yaml_path)
    _ensure_dir(args.out_dir)
    figs_dir = _ensure_dir(os.path.join(args.out_dir, "figs"))
    maps_dir = _ensure_dir(os.path.join(figs_dir, "maps"))
    aggr_dir = _ensure_dir(os.path.join(figs_dir, "aggregates"))
    tabs_dir = _ensure_dir(os.path.join(args.out_dir, "tables"))

    if args.loss_csv is not None:
        if os.path.isfile(args.loss_csv):
            loss_png = os.path.join(figs_dir, "loss_curve.png")
            save_loss_curve_from_csv(args.loss_csv, loss_png,
                                     title=f"Training and Validation Loss ({args.label_pred})")
        else:
            raise FileNotFoundError(f"--loss_csv specified but not found: {args.loss_csv}")

    # ---------------- indices & sampling for maps
    indices = _collect_indices(args.pred_dir)
    if not indices:
        raise RuntimeError(f"No <index>.hdf5 in {args.pred_dir}")
    rng = np.random.default_rng(args.rng_seed)
    map_indices = sorted(rng.choice(indices, size=min(args.map_count, len(indices)), replace=False).tolist())

    # ---------------- prepare aggregate accumulators (single pass)
    # PDFs and joint PDF bins
    rho_edges  = np.logspace(-5, 4, args.pdf_bins+1)
    logt_edges = np.linspace(-5, 4, args.joint_bins+1)
    logp_edges = np.linspace(-5, 4, args.joint_bins+1)

    Hm = np.zeros((args.joint_bins, args.joint_bins))  # Truth vs Pred
    hT = np.zeros(args.pdf_bins); hP = np.zeros(args.pdf_bins)

    xiT_list, xiP_list = [], []
    r_axis = None

    # --- Global metrics accumulators ---
    # For log10 ratio bias/std (Welford)
    count_lr = 0
    mean_lr  = 0.0
    M2_lr    = 0.0

    # For RMSE/MAE/Pearson (streaming sums)
    n_vox   = 0
    s_p     = 0.0
    s_t     = 0.0
    s_p2    = 0.0
    s_t2    = 0.0
    s_pt    = 0.0
    s_abs   = 0.0
    s_sq    = 0.0

    # KS global samples (cap)
    ks_cap = args.ks_global_cap
    ks_pred_samples = []
    ks_truth_samples = []

    def _welford_update(arr_pred, arr_truth):
        nonlocal count_lr, mean_lr, M2_lr
        eps = 1e-6
        ratio = np.clip(arr_pred, eps, None) / np.clip(arr_truth, eps, None)
        lr = np.log10(ratio)
        m = np.isfinite(lr)
        x = lr[m].ravel()
        n = x.size
        if n == 0: return
        count_new = count_lr + n
        delta = x.mean() - mean_lr
        mean_new = mean_lr + delta * (n / count_new)
        ss_old = M2_lr
        ss_batch = np.sum((x - x.mean())**2)
        M2_new = ss_old + ss_batch + (delta**2) * (count_lr * n / count_new)
        count_lr, mean_lr, M2_lr = count_new, mean_new, M2_new

    def _metrics_update(p: np.ndarray, t: np.ndarray):
        """Update global RMSE/MAE/Pearson accumulators in-place."""
        nonlocal n_vox, s_p, s_t, s_p2, s_t2, s_pt, s_abs, s_sq
        m = np.isfinite(p) & np.isfinite(t)
        if not np.any(m): return
        pv = p[m].ravel()
        tv = t[m].ravel()
        n = pv.size
        n_vox += n
        s_p  += float(pv.sum())
        s_t  += float(tv.sum())
        s_p2 += float(np.dot(pv, pv))
        s_t2 += float(np.dot(tv, tv))
        s_pt += float(np.dot(pv, tv))
        diff = pv - tv
        s_abs += float(np.abs(diff).sum())
        s_sq  += float(np.dot(diff, diff))

    def _hist1d_pos(x):
        x = x[np.isfinite(x) & (x>0)]
        h, _ = np.histogram(x, bins=rho_edges, density=True)
        return h

    def _pick_pairs(t, p, n, _rng):
        t = t.ravel(); p = p.ravel()
        m = np.isfinite(t) & np.isfinite(p) & (t>0) & (p>0)
        t = t[m]; p = p[m]
        if t.size <= n: return t, p
        idxs = _rng.choice(t.size, size=n, replace=False)
        return t[idxs], p[idxs]

    # ---------------- scan once over all indices
    model_meta = {}  # capture from first pred file attrs
    have_meta = False

    for idx in tqdm(indices, desc="Scan indices (single pass)"):
        rho_t, _ = _read_truth(yaml_cfg, idx)
        rho_p, _, attrs = _read_pred(args.pred_dir, idx)
        if not have_meta and isinstance(attrs, dict):
            model_meta = {k: attrs[k] for k in attrs.keys()}
            have_meta = True

        rho_a, _ = _read_alex(args.alex_tpl, idx); rho_a, _ = _maybe_delta_to_rho(rho_a, args.alex_force)

        # 1) tripanel maps for a few random indices (viz only)
        if idx in map_indices:
            out_png = os.path.join(maps_dir, f"{idx:05d}.png")
            try:
                save_tripanel_maps(idx, rho_t, rho_a, rho_p, out_png, args.slice_axis, args.slice_index)
            except Exception as e:
                print(f"[WARN] map save failed @ idx={idx}: {e}")

        # 2) global metrics updates
        _welford_update(rho_p, rho_t)
        _metrics_update(rho_p, rho_t)

        # KS sampling (append until cap)
        if len(ks_pred_samples) < ks_cap:
            need = ks_cap - len(ks_pred_samples)
            t_flat = rho_t.ravel()
            p_flat = rho_p.ravel()
            m = np.isfinite(t_flat) & np.isfinite(p_flat)
            t_flat = t_flat[m]; p_flat = p_flat[m]
            if t_flat.size > 0:
                take = min(need, t_flat.size)
                sel = rng.choice(t_flat.size, size=take, replace=False)
                ks_truth_samples.append(t_flat[sel])
                ks_pred_samples.append(p_flat[sel])

        # 3) aggregates for plots
        # PDFs
        hT += _hist1d_pos(rho_t)
        hP += _hist1d_pos(rho_p)

        # Joint PDF (truth vs pred)
        t_p, p_p = _pick_pairs(rho_t, rho_p, args.joint_sample, rng)
        Hm_ , _, _ = np.histogram2d(np.log10(t_p), np.log10(p_p),
                                    bins=[logt_edges, logp_edges])
        Hm += Hm_

        # ξ(r)
        def to_delta(x): return x/np.mean(x) - 1.0
        xi_t = autocorr_fft(to_delta(rho_t))
        xi_p = autocorr_fft(to_delta(rho_p))
        r, prof_t = radial_profile(xi_t, args.voxel_size, args.rmax, args.n_r_bins)
        _, prof_p = radial_profile(xi_p, args.voxel_size, args.rmax, args.n_r_bins)
        xiT_list.append(prof_t); xiP_list.append(prof_p)
        r_axis = r

    # ---------------- finalize global stats
    # log10 ratio bias/std
    if count_lr > 1:
        var_lr = M2_lr / (count_lr - 1)
        log_std = float(np.sqrt(var_lr))
        log_bias = float(mean_lr)
    elif count_lr == 1:
        log_bias = float(mean_lr); log_std = float("nan")
    else:
        log_bias = float("nan"); log_std = float("nan")

    # RMSE / MAE / Pearson
    if n_vox > 0:
        rmse = np.sqrt(s_sq / n_vox)
        mae  = s_abs / n_vox
        mean_p = s_p / n_vox
        mean_t = s_t / n_vox
        var_p = max(s_p2 / n_vox - mean_p**2, 0.0)
        var_t = max(s_t2 / n_vox - mean_t**2, 0.0)
        cov_pt = s_pt / n_vox - mean_p * mean_t
        pearson_r = cov_pt / np.sqrt(var_p * var_t) if (var_p > 0 and var_t > 0) else np.nan
    else:
        rmse = mae = pearson_r = np.nan

    # KS
    ks_pred = np.concatenate(ks_pred_samples) if len(ks_pred_samples) else np.array([])
    ks_truth = np.concatenate(ks_truth_samples) if len(ks_truth_samples) else np.array([])
    if ks_pred.size > 0 and ks_truth.size > 0:
        ks_stat = float(ks_2samp(ks_pred, ks_truth).statistic)
        ks_n    = int(ks_pred.size)
    else:
        ks_stat = float("nan"); ks_n = 0

    # ---------------- infer params (#) from checkpoint if given
    params_count = None
    if args.model_ckpt is not None and os.path.isfile(args.model_ckpt):
        try:
            import torch
            state = torch.load(args.model_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if isinstance(state, dict):
                params_count = int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
        except Exception as e:
            print(f"[WARN] Param count from ckpt failed: {e}")

    # ---------------- save tables (GLOBAL ONLY, split by category)
    overall_df = pd.DataFrame([{
        "Model": args.label_pred,
        "N_indices": len(indices),
        "Params (#)": f"{params_count:,}" if params_count is not None else "N/A",
        "FLOPs (G)": f"{args.flops_g:.2f}" if args.flops_g is not None else "N/A",
    }])

    metrics_df = pd.DataFrame([{
        "Model": args.label_pred,
        "RMSE": f"{rmse:.3e}" if np.isfinite(rmse) else "N/A",
        "MAE": f"{mae:.3e}" if np.isfinite(mae) else "N/A",
        "Pearson_r": f"{pearson_r:.3f}" if np.isfinite(pearson_r) else "N/A",
        "KS (stat)": f"{ks_stat:.3f}" if np.isfinite(ks_stat) else "N/A",
        "KS_samples": ks_n,
        "⟨log10(ρ_pred/ρ_true)⟩": f"{log_bias:+.3f}" if np.isfinite(log_bias) else "N/A",
        "σ_log10": f"{log_std:.3f}" if np.isfinite(log_std) else "N/A",
    }])

    logratio_df = pd.DataFrame([{
        "Model": args.label_pred,
        "log_bias": f"{log_bias:+.3f}" if np.isfinite(log_bias) else "N/A",
        "log_std": f"{log_std:.3f}" if np.isfinite(log_std) else "N/A",
    }])

    overall_csv = os.path.join(tabs_dir, "summary_overall.csv")
    metrics_csv = os.path.join(tabs_dir, "summary_metrics.csv")
    logratio_csv = os.path.join(tabs_dir, "summary_logratio.csv")
    overall_df.to_csv(overall_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    logratio_df.to_csv(logratio_csv, index=False)

    if args.save_latex:
        overall_df.to_latex(os.path.join(tabs_dir, "summary_overall.tex"), index=False, escape=False)
        metrics_df.to_latex(os.path.join(tabs_dir, "summary_metrics.tex"), index=False, escape=False)
        logratio_df.to_latex(os.path.join(tabs_dir, "summary_logratio.tex"), index=False, escape=False)

    # ---------------- save aggregate plots
    cache = dict(
        rho_edges=rho_edges, logt_edges=logt_edges, logp_edges=logp_edges,
        Hm=Hm, hT=hT, hP=hP,
        r=r_axis, xiT=np.vstack(xiT_list), xiP=np.vstack(xiP_list)
    )
    save_aggregate_plots(cache, aggr_dir, args.label_pred)

    # ---------------- footer logs
    print(f"[DONE] Saved map images → {maps_dir}  (count={len(map_indices)})")
    print(f"[DONE] Aggregate figs → {aggr_dir}")
    print(f"[DONE] Tables → {tabs_dir}")
    if have_meta:
        # a few convenient echoes from pred-file attrs if present
        model_cls = model_meta.get("model_class", "N/A")
        print(f"[META] model_class={model_cls}")
        mp = model_meta.get("model_path", "N/A")
        print(f"[META] model_path={mp}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
FAST Evaluation & Visualization for 3D voxel-wise predictions.

What this does (single pass over indices):
  - Tripanel maps for a small random subset (for quick visual sanity checks)
  - Global statistics (NO per-index tables):
      * RMSE, MAE, Pearson r  (global, voxel-wise)
      * log10(rho_pred/rho_true): global mean (bias) and std
      * Kolmogorov–Smirnov statistic + KS sample count (adjacent columns)
  - Aggregates and figures:
      * Joint PDF (truth vs pred) with 1σ/2σ/3σ contours
      * Density PDF (truth vs pred)
      * Two-point correlation ξ(r): median + 1σ band (truth & pred)
      * Optional loss curve if --loss_csv is provided
  - Split summary tables:
      * tables/summary_overall.{csv,tex}
      * tables/summary_metrics.{csv,tex}
      * tables/summary_logratio.{csv,tex}

Author: Mingyeong Yang (UST-KASI)
Last-Modified: 2025-11-04
"""

from __future__ import annotations
import os, re, glob, argparse, yaml, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp


# ----------------------------
# IO helpers
# ----------------------------
def _s(a): return np.squeeze(np.asarray(a))

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _collect_indices(pred_dir: str) -> list[int]:
    out = []
    for fn in os.listdir(pred_dir):
        m = re.match(r"(\d+)\.hdf5$", fn)
        if m: out.append(int(m.group(1)))
    return sorted(out)

def _read_truth(yaml_cfg: dict, idx: int):
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]
    if "*" in test_rel:
        test_path = os.path.join(base, test_rel.replace("*", str(idx)))
    else:
        test_path = os.path.join(base, test_rel, f"{idx}.hdf5")
    with h5py.File(test_path, "r") as f:
        arr = _s(f["output_rho"][:])
    return arr, test_path

def _read_pred(pred_dir: str, idx: int):
    path = os.path.join(pred_dir, f"{idx}.hdf5")
    with h5py.File(path, "r") as f:
        arr = _s(f["prediction"][:])
        attrs = dict(f.attrs)  # model metadata if present
    return arr, path, attrs

def _read_alex(alex_tpl: str, idx: int):
    """
    alex_tpl can be:
      - a template string containing {idx}, e.g. ".../test_{idx:03d}_*rho*.npy"
      - a glob string (already expanded wildcard), e.g. ".../test_*rho*.npy"
      - a directory path: this function will normalize it to a template

    Robust to:
      * zero-padding differences
      * missing 'rho' keyword (falls back to any *.npy)
      * directory input
      * unpadded filenames like test_0_...npy
    """
    import numpy as _np

    # If a directory is given, normalize to a template by inferring zero-padding.
    if os.path.isdir(alex_tpl):
        def _infer_pad_from_dir(dir_path: str, prefix="test_"):
            cand = sorted(glob.glob(os.path.join(dir_path, f"{prefix}*.npy")))
            for c in cand:
                m = re.search(rf"{re.escape(prefix)}(\d+)", os.path.basename(c))
                if m:
                    return len(m.group(1))
            return 3  # fallback
        pad = _infer_pad_from_dir(alex_tpl, prefix="test_")
        cand_tpl = os.path.join(alex_tpl, f"test_{{idx:0{pad}d}}_*rho*.npy")
        if not glob.glob(cand_tpl.format(idx=idx)):
            cand_tpl = os.path.join(alex_tpl, f"test_{{idx:0{pad}d}}_*.npy")
        # recurse with the new template
        return _read_alex(cand_tpl, idx)

    pats = []
    if "{idx" in alex_tpl:
        # 1) exact template
        try:
            pats = sorted(glob.glob(alex_tpl.format(idx=idx)))
        except Exception:
            pats = []

        # 2) relax suffix: ".npy" → "*.npy"
        if not pats:
            try:
                base = alex_tpl.format(idx=idx)
                base_star = re.sub(r"\.npy$", "*.npy", base)
                pats = sorted(glob.glob(base_star))
            except Exception:
                pats = []

        # 3) try different zero-padding lengths
        if not pats:
            for pad in (1, 2, 3, 4, 5, 6):
                try_pat = re.sub(r"\{idx:0\d+d\}", f"{{idx:0{pad}d}}", alex_tpl)
                try:
                    hit = sorted(glob.glob(try_pat.format(idx=idx)))
                except Exception:
                    hit = []
                if hit:
                    pats = hit
                    break

        # 4) unpadded fallback {idx}
        if not pats:
            plain_tpl = re.sub(r"\{idx:0\d+d\}", "{idx}", alex_tpl)
            try:
                pats = sorted(glob.glob(plain_tpl.format(idx=idx)))
                if not pats:
                    base = plain_tpl.format(idx=idx)
                    base_star = re.sub(r"\.npy$", "*.npy", base)
                    pats = sorted(glob.glob(base_star))
            except Exception:
                pats = []
    else:
        # already glob-like
        pats = sorted(glob.glob(alex_tpl))

    # keep only files
    pats = [p for p in pats if os.path.isfile(p)]
    if not pats:
        raise FileNotFoundError(f"No Alex npy for idx={idx} using pattern={alex_tpl}")
    return _s(_np.load(pats[0])), pats[0]

def _maybe_delta_to_rho(arr: np.ndarray, force: str|None):
    """
    force: "true" | "false" | None
    """
    if force == "true":
        return arr + 1.0, True
    if force == "false":
        return arr, False
    if arr.min() < 0 or arr.max() < 1.0:
        return arr + 1.0, True
    return arr, False


# ----------------------------
# math helpers
# ----------------------------
def get_slice(vol3d: np.ndarray, axis: int, idx: int|str="center"):
    if isinstance(idx, str) and idx == "center":
        idx = vol3d.shape[axis] // 2
    if axis == 0:   return vol3d[idx, :, :]
    if axis == 1:   return vol3d[:, idx, :]
    if axis == 2:   return vol3d[:, :, idx]
    raise ValueError("axis must be 0/1/2")

def log1p10(a: np.ndarray):
    return np.log10(1.0 + np.clip(a, 0, None))

def autocorr_fft(delta: np.ndarray):
    F = np.fft.fftn(delta)
    xi = np.fft.ifftn(np.abs(F)**2).real / delta.size
    return np.fft.fftshift(xi)

def radial_profile(vol: np.ndarray, voxel_size: float, r_max: float, n_bins: int):
    nz, ny, nx = vol.shape
    cz, cy, cx = (np.array(vol.shape)//2)
    z = (np.arange(nz)-cz)*voxel_size
    y = (np.arange(ny)-cy)*voxel_size
    x = (np.arange(nx)-cx)*voxel_size
    Z, Y, X = np.meshgrid(z,y,x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    edges = np.linspace(0.0, r_max, n_bins+1)
    r = 0.5*(edges[1:]+edges[:-1])
    inds = np.digitize(R.ravel(), edges)-1
    prof = np.zeros(n_bins); v = vol.ravel()
    for i in range(n_bins):
        m = inds==i
        prof[i] = np.mean(v[m]) if np.any(m) else np.nan
    return r, prof

def contour_levels_from_hist2d(H: np.ndarray, probs=(0.68, 0.95, 0.997)):
    H = H.astype(float)
    H = H / H.sum() if H.sum() > 0 else H
    flat = np.sort(H.ravel())[::-1]
    cumsum = np.cumsum(flat)
    levels = []
    for p in probs:
        idx = np.searchsorted(cumsum, p)
        t = flat[min(idx, flat.size-1)]
        levels.append(t * H.sum())  # back to counts space
    return levels

def median_and_band(X: np.ndarray):
    med = np.nanmedian(X, axis=0)
    lo  = np.nanpercentile(X, 16, axis=0)
    hi  = np.nanpercentile(X, 84, axis=0)
    return med, lo, hi


# ----------------------------
# plotting helpers
# ----------------------------
def save_tripanel_maps(idx: int, truth: np.ndarray, alex: np.ndarray, pred: np.ndarray,
                       out_png: str, axis: int, slice_idx: int|str):
    t = get_slice(truth, axis, slice_idx)
    a = get_slice(alex,  axis, slice_idx)
    p = get_slice(pred,  axis, slice_idx)
    t_img, a_img, p_img = log1p10(t), log1p10(a), log1p10(p)
    vmin = min(t_img.min(), a_img.min(), p_img.min())
    vmax = max(t_img.max(), a_img.max(), p_img.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = [
        "Truth — ρ : log$_{10}$(1+ρ)",
        "Alex — $\\hat{\\rho}$ : log$_{10}$(1+ρ)",
        "Ours — $\\hat{\\rho}$ : log$_{10}$(1+ρ)"
    ]
    for ax, img, title in zip(axes, [t_img, a_img, p_img], titles):
        im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log$_{10}$(1+ρ)")
    plt.savefig(out_png, dpi=200); plt.close(fig)

def save_loss_curve_from_csv(csv_path: str, out_png: str, title: str = "Training and Validation Loss"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("epoch", "train_loss", "val_loss", "lr"):
        if col not in df.columns:
            raise ValueError(f"[loss-curve] Missing column '{col}' in {csv_path}")
    df = df.sort_values("epoch").reset_index(drop=True)

    best_idx = int(df["val_loss"].idxmin())
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_val = float(df.loc[best_idx, "val_loss"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"],   label="Validation Loss", linewidth=2, linestyle="--")
    ax1.axvline(best_ep, color="gray", linestyle=":", lw=1.5)
    ax1.scatter([best_ep], [best_val], s=30, color="gray", zorder=3)
    ax1.text(best_ep + 0.3, best_val, f"Best val={best_val:.3f}\n@ epoch {best_ep}", fontsize=9)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3); ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["lr"], label="Learning Rate", alpha=0.7)
    ax2.set_yscale("log"); ax2.set_ylabel("Learning Rate (log)")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")

    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
    print(f"[DONE] Loss curve → {out_png} (best epoch = {best_ep}, best val = {best_val:.4f})")

def save_aggregate_plots(cache: dict, out_dir: str, label_pred: str):
    rho_edges   = cache["rho_edges"]
    logt_edges  = cache["logt_edges"]; logp_edges = cache["logp_edges"]
    Hm          = cache["Hm"]
    hT, hP      = cache["hT"], cache["hP"]
    r           = cache["r"]
    xiT, xiP    = cache["xiT"], cache["xiP"]

    rho_centers = 0.5*(rho_edges[1:]+rho_edges[:-1])

    # 1) Joint PDF + contours (Truth vs Pred)
    levels = contour_levels_from_hist2d(Hm, probs=(0.68, 0.95, 0.997))
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    Hm_plot = np.ma.masked_where(Hm <= 0, Hm)
    pcm = ax.pcolormesh(10**logt_edges, 10**logp_edges, Hm_plot.T, shading='auto',
                        norm=plt.matplotlib.colors.LogNorm(), cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax); cb.set_label("counts (log)")
    CS = ax.contour(10**logt_edges[:-1], 10**logp_edges[:-1], Hm.T,
                    levels=levels, colors='white', linewidths=1.5)
    ax.clabel(CS, inline=True, fontsize=8, fmt={levels[0]:"1σ", levels[1]:"2σ", levels[2]:"3σ"})
    xmin, xmax = 10**logt_edges[0], 10**logt_edges[-1]
    ax.plot([xmin, xmax], [xmin, xmax], 'k--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\rho_{\rm truth}/\rho_0$")
    ax.set_ylabel(r"$\rho_{\rm pred}/\rho_0$")
    ax.set_title(f"Joint PDF: Truth vs {label_pred}")
    plt.savefig(os.path.join(out_dir, "joint_pdf_pred.png"), dpi=200); plt.close(fig)

    # 2) Density PDF
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.plot(rho_centers, hT, color='black', lw=2, label='Truth')
    ax.plot(rho_centers, hP, color='tab:blue', lw=2, label=label_pred)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$\rho/\rho_0$")
    ax.set_ylabel(r"$df/d\log_{10}\rho$")
    ax.set_title("Density PDF")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "pdf_density.png"), dpi=200); plt.close(fig)

    # 3) ξ(r): median + 1σ band
    t_med, t_lo, t_hi = median_and_band(xiT)
    p_med, p_lo, p_hi = median_and_band(xiP)
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.fill_between(r, t_lo, t_hi, color='gray', alpha=0.3, label='Truth 1σ')
    ax.plot(r, t_med, color='black', lw=2, label='Truth median')
    ax.fill_between(r, p_lo, p_hi, color='tab:blue', alpha=0.25, label=f'{label_pred} 1σ')
    ax.plot(r, p_med, color='tab:blue', lw=2, label=f'{label_pred} median')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\langle \delta(\mathbf{x})\delta(\mathbf{x}+\mathbf{r}) \rangle$")
    ax.set_title("Two-point correlation ξ(r)")
    ax.legend()
    plt.savefig(os.path.join(out_dir, "xi_two_point.png"), dpi=200); plt.close(fig)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="FAST eval: random few maps + global metrics (single pass).")
    # data/config
    ap.add_argument("--yaml_path",     type=str, required=True)
    ap.add_argument("--pred_dir",      type=str, required=True, help="Prediction dir containing <index>.hdf5")
    ap.add_argument("--alex_tpl",      type=str, required=True,
                    help="Alex NPY template or directory. Examples: "
                         "/.../test_{idx:03d}_*rho*.npy  OR  /path/to/dir (auto-normalized)")
    ap.add_argument("--out_dir",       type=str, required=True)
    # (optional) model metadata to enrich tables
    ap.add_argument("--model_ckpt",    type=str, default=None, help=".pt path for param count (optional)")
    ap.add_argument("--flops_g",       type=float, default=None, help="FLOPs in G (optional, if measured elsewhere)")
    # slicing for map figures
    ap.add_argument("--slice_axis",    type=int, default=2, choices=[0,1,2])
    ap.add_argument("--slice_index",   type=str, default="center", help='"center" or integer index')
    # speed / sampling
    ap.add_argument("--map_count",     type=int, default=5, help="Number of random indices for map saving")
    ap.add_argument("--rng_seed",      type=int, default=42)
    # alex delta->rho handling
    ap.add_argument("--alex_force",    type=str, choices=["true","false"], default=None)
    # aggregates
    ap.add_argument("--joint_sample",  type=int, default=50000, help="pairs per index for joint hist (max)")
    ap.add_argument("--pdf_bins",      type=int, default=120)
    ap.add_argument("--joint_bins",    type=int, default=120)
    ap.add_argument("--voxel_size",    type=float, default=205.0/250.0, help="h^-1 Mpc per voxel")
    ap.add_argument("--rmax",          type=float, default=10.0)
    ap.add_argument("--n_r_bins",      type=int, default=24)
    # global stats (no per-index)
    ap.add_argument("--ks_global_cap", type=int, default=200_000, help="Max total samples used for global KS")
    # labeling
    ap.add_argument("--label_pred",    type=str, default="Ours")
    # tables
    ap.add_argument("--save_latex",    action="store_true")
    # (optional) loss curve
    ap.add_argument("--loss_csv",      type=str, default=None,
                    help="If provided, plot loss curve from CSV (epoch,train_loss,val_loss,lr)")
    args = ap.parse_args()

    # ---- Normalize --alex_tpl when a directory is provided (robust to trailing slashes/whitespace)
    raw_alex = args.alex_tpl.strip()
    alex_dir = raw_alex.rstrip("/")
    if os.path.isdir(alex_dir):
        def _infer_pad_from_dir(dir_path: str, prefix: str = "test_") -> int:
            cand = sorted(glob.glob(os.path.join(dir_path, f"{prefix}*.npy")))
            for c in cand:
                m = re.search(rf"{re.escape(prefix)}(\d+)", os.path.basename(c))
                if m:
                    return len(m.group(1))
            return 3
        pad = _infer_pad_from_dir(alex_dir, prefix="test_")
        tmpl = os.path.join(alex_dir, f"test_{{idx:0{pad}d}}_*rho*.npy")
        if not glob.glob(tmpl.format(idx=0)) and not glob.glob(tmpl.format(idx=1)):
            tmpl = os.path.join(alex_dir, f"test_{{idx:0{pad}d}}_*.npy")
        args.alex_tpl = tmpl
        print(f"[INFO] alex_tpl normalized to template: {args.alex_tpl}")

    # ---------------- dirs & loss curve
    yaml_cfg = _load_yaml(args.yaml_path)
    _ensure_dir(args.out_dir)
    figs_dir = _ensure_dir(os.path.join(args.out_dir, "figs"))
    maps_dir = _ensure_dir(os.path.join(figs_dir, "maps"))
    aggr_dir = _ensure_dir(os.path.join(figs_dir, "aggregates"))
    tabs_dir = _ensure_dir(os.path.join(args.out_dir, "tables"))

    if args.loss_csv is not None:
        if os.path.isfile(args.loss_csv):
            loss_png = os.path.join(figs_dir, "loss_curve.png")
            save_loss_curve_from_csv(args.loss_csv, loss_png,
                                     title=f"Training and Validation Loss ({args.label_pred})")
        else:
            raise FileNotFoundError(f"--loss_csv specified but not found: {args.loss_csv}")

    # ---------------- indices & sampling for maps
    indices = _collect_indices(args.pred_dir)
    if not indices:
        raise RuntimeError(f"No <index>.hdf5 in {args.pred_dir}")
    rng = np.random.default_rng(args.rng_seed)
    map_indices = sorted(rng.choice(indices, size=min(args.map_count, len(indices)), replace=False).tolist())

    # ---------------- prepare aggregate accumulators (single pass)
    rho_edges  = np.logspace(-5, 4, args.pdf_bins+1)
    logt_edges = np.linspace(-5, 4, args.joint_bins+1)
    logp_edges = np.linspace(-5, 4, args.joint_bins+1)

    Hm = np.zeros((args.joint_bins, args.joint_bins))  # Truth vs Pred
    hT = np.zeros(args.pdf_bins); hP = np.zeros(args.pdf_bins)

    xiT_list, xiP_list = [], []
    r_axis = None

    # --- Global metrics accumulators ---
    # For log10 ratio bias/std (Welford)
    count_lr = 0; mean_lr = 0.0; M2_lr = 0.0

    # For RMSE/MAE/Pearson (streaming sums)
    n_vox = 0; s_p = s_t = s_p2 = s_t2 = s_pt = s_abs = s_sq = 0.0

    # KS global samples (cap)
    ks_cap = args.ks_global_cap
    ks_pred_samples = []
    ks_truth_samples = []

    def _welford_update(arr_pred, arr_truth):
        nonlocal count_lr, mean_lr, M2_lr
        eps = 1e-6
        ratio = np.clip(arr_pred, eps, None) / np.clip(arr_truth, eps, None)
        lr = np.log10(ratio)
        m = np.isfinite(lr)
        x = lr[m].ravel()
        n = x.size
        if n == 0: return
        count_new = count_lr + n
        delta = x.mean() - mean_lr
        mean_new = mean_lr + delta * (n / count_new)
        ss_old = M2_lr
        ss_batch = np.sum((x - x.mean())**2)
        M2_new = ss_old + ss_batch + (delta**2) * (count_lr * n / count_new)
        count_lr, mean_lr, M2_lr = count_new, mean_new, M2_new

    def _metrics_update(p: np.ndarray, t: np.ndarray):
        """Update global RMSE/MAE/Pearson accumulators in-place."""
        nonlocal n_vox, s_p, s_t, s_p2, s_t2, s_pt, s_abs, s_sq
        m = np.isfinite(p) & np.isfinite(t)
        if not np.any(m): return
        pv = p[m].ravel(); tv = t[m].ravel()
        n = pv.size
        n_vox += n
        s_p  += float(pv.sum())
        s_t  += float(tv.sum())
        s_p2 += float(np.dot(pv, pv))
        s_t2 += float(np.dot(tv, tv))
        s_pt += float(np.dot(pv, tv))
        diff = pv - tv
        s_abs += float(np.abs(diff).sum())
        s_sq  += float(np.dot(diff, diff))

    def _hist1d_pos(x):
        x = x[np.isfinite(x) & (x>0)]
        h, _ = np.histogram(x, bins=rho_edges, density=True)
        return h

    def _pick_pairs(t, p, n, _rng):
        t = t.ravel(); p = p.ravel()
        m = np.isfinite(t) & np.isfinite(p) & (t>0) & (p>0)
        t = t[m]; p = p[m]
        if t.size <= n: return t, p
        idxs = _rng.choice(t.size, size=n, replace=False)
        return t[idxs], p[idxs]

    # ---------------- scan once over all indices
    model_meta = {}  # capture from first pred file attrs
    have_meta = False

    for idx in tqdm(indices, desc="Scan indices (single pass)"):
        rho_t, _ = _read_truth(yaml_cfg, idx)
        rho_p, _, attrs = _read_pred(args.pred_dir, idx)
        if not have_meta and isinstance(attrs, dict):
            model_meta = {k: attrs[k] for k in attrs.keys()}
            have_meta = True

        # 1) tripanel maps for a few random indices (viz only)
        if idx in map_indices:
            out_png = os.path.join(maps_dir, f"{idx:05d}.png")
            try:
                rho_a, _apath = _read_alex(args.alex_tpl, idx)
                rho_a, _ = _maybe_delta_to_rho(rho_a, args.alex_force)
            except Exception as e:
                print(f"[WARN] Alex not found for idx={idx}: {e}  (using Truth as placeholder)")
                rho_a = rho_t
            try:
                save_tripanel_maps(idx, rho_t, rho_a, rho_p, out_png, args.slice_axis, args.slice_index)
            except Exception as e:
                print(f"[WARN] map save failed @ idx={idx}: {e}")

        # 2) global metrics updates
        _welford_update(rho_p, rho_t)
        _metrics_update(rho_p, rho_t)

        # KS sampling (append until cap)
        if len(ks_pred_samples) < ks_cap:
            need = ks_cap - len(ks_pred_samples)
            t_flat = rho_t.ravel(); p_flat = rho_p.ravel()
            m = np.isfinite(t_flat) & np.isfinite(p_flat)
            t_flat = t_flat[m]; p_flat = p_flat[m]
            if t_flat.size > 0:
                take = min(need, t_flat.size)
                sel = rng.choice(t_flat.size, size=take, replace=False)
                ks_truth_samples.append(t_flat[sel])
                ks_pred_samples.append(p_flat[sel])

        # 3) aggregates for plots
        # PDFs
        hT += _hist1d_pos(rho_t)
        hP += _hist1d_pos(rho_p)

        # Joint PDF (truth vs pred)
        t_p, p_p = _pick_pairs(rho_t, rho_p, args.joint_sample, rng)
        Hm_ , _, _ = np.histogram2d(np.log10(t_p), np.log10(p_p),
                                    bins=[logt_edges, logp_edges])
        Hm += Hm_

        # ξ(r)
        def to_delta(x): return x/np.mean(x) - 1.0
        xi_t = autocorr_fft(to_delta(rho_t))
        xi_p = autocorr_fft(to_delta(rho_p))
        r, prof_t = radial_profile(xi_t, args.voxel_size, args.rmax, args.n_r_bins)
        _, prof_p = radial_profile(xi_p, args.voxel_size, args.rmax, args.n_r_bins)
        xiT_list.append(prof_t); xiP_list.append(prof_p)
        r_axis = r

    # ---------------- finalize global stats
    # log10 ratio bias/std
    if count_lr > 1:
        var_lr = M2_lr / (count_lr - 1)
        log_std = float(np.sqrt(var_lr))
        log_bias = float(mean_lr)
    elif count_lr == 1:
        log_bias = float(mean_lr); log_std = float("nan")
    else:
        log_bias = float("nan"); log_std = float("nan")

    # RMSE / MAE / Pearson
    if n_vox > 0:
        rmse = np.sqrt(s_sq / n_vox)
        mae  = s_abs / n_vox
        mean_p = s_p / n_vox
        mean_t = s_t / n_vox
        var_p = max(s_p2 / n_vox - mean_p**2, 0.0)
        var_t = max(s_t2 / n_vox - mean_t**2, 0.0)
        cov_pt = s_pt / n_vox - mean_p * mean_t
        pearson_r = cov_pt / np.sqrt(var_p * var_t) if (var_p > 0 and var_t > 0) else np.nan
    else:
        rmse = mae = pearson_r = np.nan

    # KS
    ks_pred = np.concatenate(ks_pred_samples) if len(ks_pred_samples) else np.array([])
    ks_truth = np.concatenate(ks_truth_samples) if len(ks_truth_samples) else np.array([])
    if ks_pred.size > 0 and ks_truth.size > 0:
        ks_stat = float(ks_2samp(ks_pred, ks_truth).statistic)
        ks_n    = int(ks_pred.size)
    else:
        ks_stat = float("nan"); ks_n = 0

    # ---------------- infer params (#) from checkpoint if given
    params_count = None
    if args.model_ckpt is not None and os.path.isfile(args.model_ckpt):
        try:
            import torch
            state = torch.load(args.model_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if isinstance(state, dict):
                params_count = int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
        except Exception as e:
            print(f"[WARN] Param count from ckpt failed: {e}")

    # ---------------- save tables (GLOBAL ONLY, split by category)
    overall_df = pd.DataFrame([{
        "Model": args.label_pred,
        "N_indices": len(indices),
        "Params (#)": f"{params_count:,}" if params_count is not None else "N/A",
        "FLOPs (G)": f"{args.flops_g:.2f}" if args.flops_g is not None else "N/A",
    }])

    metrics_df = pd.DataFrame([{
        "Model": args.label_pred,
        "RMSE": f"{rmse:.3e}" if np.isfinite(rmse) else "N/A",
        "MAE": f"{mae:.3e}" if np.isfinite(mae) else "N/A",
        "Pearson_r": f"{pearson_r:.3f}" if np.isfinite(pearson_r) else "N/A",
        "KS (stat)": f"{ks_stat:.3f}" if np.isfinite(ks_stat) else "N/A",
        "KS_samples": ks_n,
        "⟨log10(ρ_pred/ρ_true)⟩": f"{log_bias:+.3f}" if np.isfinite(log_bias) else "N/A",
        "σ_log10": f"{log_std:.3f}" if np.isfinite(log_std) else "N/A",
    }])

    logratio_df = pd.DataFrame([{
        "Model": args.label_pred,
        "log_bias": f"{log_bias:+.3f}" if np.isfinite(log_bias) else "N/A",
        "log_std": f"{log_std:.3f}" if np.isfinite(log_std) else "N/A",
    }])

    overall_csv = os.path.join(tabs_dir, "summary_overall.csv")
    metrics_csv = os.path.join(tabs_dir, "summary_metrics.csv")
    logratio_csv = os.path.join(tabs_dir, "summary_logratio.csv")
    overall_df.to_csv(overall_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    logratio_df.to_csv(logratio_csv, index=False)

    if args.save_latex:
        overall_df.to_latex(os.path.join(tabs_dir, "summary_overall.tex"), index=False, escape=False)
        metrics_df.to_latex(os.path.join(tabs_dir, "summary_metrics.tex"), index=False, escape=False)
        logratio_df.to_latex(os.path.join(tabs_dir, "summary_logratio.tex"), index=False, escape=False)

    # ---------------- save aggregate plots
    cache = dict(
        rho_edges=rho_edges, logt_edges=logt_edges, logp_edges=logp_edges,
        Hm=Hm, hT=hT, hP=hP,
        r=r_axis, xiT=np.vstack(xiT_list), xiP=np.vstack(xiP_list)
    )
    save_aggregate_plots(cache, aggr_dir, args.label_pred)

    # ---------------- footer logs
    print(f"[DONE] Saved map images → {maps_dir}  (count={len(map_indices)})")
    print(f"[DONE] Aggregate figs → {aggr_dir}")
    print(f"[DONE] Tables → {tabs_dir}")
    if have_meta:
        model_cls = model_meta.get("model_class", "N/A")
        print(f"[META] model_class={model_cls}")
        mp = model_meta.get("model_path", "N/A")
        print(f"[META] model_path={mp}")


if __name__ == "__main__":
    main()
