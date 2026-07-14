import os
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Import project-specific modules
try:
    from uwir.models import build_model, parse_model_variant
    from uwir.cli.train import _resolve_physics_extractor, _add_physics_channels
except ImportError:
    print("Error: Could not import model registry and physics extractors.")
    raise

# Import metric functions
from uwir.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ciede2000,
    compute_uciqe,
    compute_uiqm
)

# ==============================================================================
# MATPLOTLIB CONFIGURATION FOR LNCS / SPRINGER PAPERS
# ==============================================================================
def setup_matplotlib_for_paper():
    """Configures matplotlib for LNCS/Springer style."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.grid': False,
        'figure.facecolor': 'w',
        'axes.facecolor': 'w',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
    })

# ==============================================================================
# MODULAR FUNCTIONS
# ==============================================================================

def run_inference(args):
    """Runs inference across given ablation directories."""
    variants = ["3ch", "4ch_t", "4ch_b", "5ch"]
    seeds = [0, 1, 2]

    for ablation_dir in args.ablation_dirs:
        parts = ablation_dir.split("_")
        if len(parts) >= 3:
            dataset_name = parts[1].upper() # UIEB or EUVP
            prior_method = parts[2].lower() # udcp
        else:
            print(f"Skipping {ablation_dir}: format should be ablation_<dataset>_<prior>")
            continue

        print(f"Processing ablation dir: {ablation_dir} (Dataset: {dataset_name}, Prior: {prior_method})")
        physics_extractor = _resolve_physics_extractor(prior_method)

        # Resolve dataset paths
        if dataset_name == "UIEB":
            img_dir = Path(args.data_root) / "UIEB" / "raw-890"
            ref_dir = Path(args.data_root) / "UIEB" / "reference-890"
        elif dataset_name == "EUVP":
            img_dir = Path(args.data_root) / "EUVP" / "test_samples" / "Inp"
            ref_dir = Path(args.data_root) / "EUVP" / "test_samples" / "GTr"
        else:
            print(f"Unknown dataset {dataset_name}")
            continue

        if not img_dir.exists():
            print(f"Data dir {img_dir} not found. Skipping.")
            continue

        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Restrict UIEB to test split (last 90 images)
        if dataset_name == "UIEB":
            image_files = image_files[-90:]

        for variant_short in variants:
            model_variant = f"unet_{variant_short}"
            try:
                _, in_channels, physics_mode = parse_model_variant(model_variant)
            except Exception as e:
                print(f"Failed to parse variant {model_variant}: {e}")
                continue

            for seed in seeds:
                # E.g.: checkpoints/ablation_uieb_udcp/4ch_t/seed 0/best_model.pth
                ckpt_path = Path(args.checkpoint_root) / ablation_dir / variant_short / f"seed {seed}" / "best_model.pth"
                if not ckpt_path.exists():
                    ckpt_path = Path(args.checkpoint_root) / ablation_dir / variant_short / f"seed{seed}" / "best_model.pth"
                    if not ckpt_path.exists():
                        print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
                        continue

                model = build_model(model_variant, pretrained_backbone=False).to(args.device)
                state_dict = torch.load(ckpt_path, map_location=args.device)
                model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
                model.eval()

                # output mapping e.g. figures/restored/UIEB/unet_3ch/seed_0
                out_dir = Path(args.output_dir) / "restored" / dataset_name / model_variant / f"seed_{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)

                print(f"  Inferencing {model_variant} (Seed {seed})...")

                with torch.no_grad():
                    for img_name in image_files:
                        rgb_path = img_dir / img_name
                        rgb_img = Image.open(rgb_path).convert('RGB')
                        orig_size = rgb_img.size

                        # Resize to 256 for inference to match training
                        rgb_img = rgb_img.resize((256, 256))
                        rgb_tensor = TF.to_tensor(rgb_img)

                        input_tensor = _add_physics_channels(rgb_tensor, physics_mode, physics_extractor)
                        input_tensor = input_tensor.unsqueeze(0).to(args.device)

                        output = model(input_tensor)

                        output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
                        output_img = Image.fromarray((output_np * 255).astype(np.uint8))
                        # Resize back to original
                        output_img = output_img.resize(orig_size)
                        output_img.save(out_dir / img_name)

def compute_all_metrics(args):
    """Computes all metrics and saves them to CSVs."""
    datasets = []
    for ablation_dir in args.ablation_dirs:
        parts = ablation_dir.split("_")
        if len(parts) >= 3:
            datasets.append(parts[1].upper())
    datasets = list(set(datasets))

    variants = ["unet_3ch", "unet_4ch_t", "unet_4ch_b", "unet_5ch"]
    seeds = [0, 1, 2]

    all_metrics = []

    for dataset in datasets:
        if dataset == "UIEB":
            ref_dir = Path(args.data_root) / "UIEB" / "reference-890"
        elif dataset == "EUVP":
            ref_dir = Path(args.data_root) / "EUVP" / "test_samples" / "GTr"
        else:
            continue

        if not ref_dir.exists():
            continue

        image_files = sorted([f for f in os.listdir(ref_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if dataset == "UIEB":
            image_files = image_files[-90:]

        for variant in variants:
            for seed in seeds:
                out_dir = Path(args.output_dir) / "restored" / dataset / variant / f"seed_{seed}"
                if not out_dir.exists():
                    continue

                for img_name in image_files:
                    ref_path = ref_dir / img_name
                    pred_path = out_dir / img_name

                    if not pred_path.exists():
                        continue

                    ref_img = np.array(Image.open(ref_path).convert('RGB')) / 255.0
                    pred_img = np.array(Image.open(pred_path).convert('RGB')) / 255.0

                    psnr_val = compute_psnr(pred_img, ref_img)
                    ssim_val = compute_ssim(pred_img, ref_img)
                    ciede_val = compute_ciede2000(pred_img, ref_img)
                    uciqe_val = compute_uciqe(pred_img)
                    uiqm_val = compute_uiqm(pred_img)

                    all_metrics.append({
                        "Dataset": dataset,
                        "Variant": variant,
                        "Seed": seed,
                        "Image": img_name,
                        "PSNR": psnr_val,
                        "SSIM": ssim_val,
                        "CIEDE2000": ciede_val,
                        "UCIQE": uciqe_val,
                        "UIQM": uiqm_val
                    })

    if not all_metrics:
        print("No metrics computed. Check if inference generated outputs.")
        return None, None

    df = pd.DataFrame(all_metrics)
    df.to_csv(Path(args.output_dir) / "metrics_per_image.csv", index=False)

    df_mean_std = df.groupby(["Dataset", "Variant", "Image"]).agg(
        {metric: ['mean', 'std'] for metric in ["PSNR", "SSIM", "CIEDE2000", "UCIQE", "UIQM"]}
    ).reset_index()
    df_mean_std.columns = ['_'.join(col).strip('_') for col in df_mean_std.columns.values]
    df_mean_std.to_csv(Path(args.output_dir) / "metrics_mean_std.csv", index=False)

    df_overall = df.groupby(["Dataset", "Variant", "Seed"]).mean(numeric_only=True).reset_index()
    df_overall_stats = df_overall.groupby(["Dataset", "Variant"]).agg(
        {metric: ['mean', 'std'] for metric in ["PSNR", "SSIM", "CIEDE2000", "UCIQE", "UIQM"]}
    ).reset_index()

    return df, df_overall_stats

def select_representative_images(df, dataset):
    df_3ch = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_3ch")].groupby("Image").mean(numeric_only=True)
    df_5ch = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_5ch")].groupby("Image").mean(numeric_only=True)

    if df_3ch.empty or df_5ch.empty:
        return []

    diff_psnr = df_5ch["PSNR"] - df_3ch["PSNR"]
    diff_ciede = df_3ch["CIEDE2000"] - df_5ch["CIEDE2000"] # positive means 5ch is better

    # 2 with high improvement (picking different ones to see variety)
    img_best_psnr = diff_psnr.nlargest(4).index[2] # 3rd best PSNR improvement
    img_best_ciede = diff_ciede.drop([img_best_psnr], errors='ignore').nlargest(4).index[2] # 3rd best CIEDE improvement

    # 2 with equal performance (improvement close to 0)
    diff_psnr_rem = diff_psnr.drop([img_best_psnr, img_best_ciede], errors='ignore')
    closest_to_zero = diff_psnr_rem.abs().nsmallest(2).index.tolist()

    img_normal_1 = closest_to_zero[0] if len(closest_to_zero) > 0 else img_best_psnr
    img_normal_2 = closest_to_zero[1] if len(closest_to_zero) > 1 else img_best_ciede

    return [img_best_psnr, img_best_ciede, img_normal_1, img_normal_2]

def get_representative_seed(df, dataset, variant):
    df_var = df[(df["Dataset"] == dataset) & (df["Variant"] == variant)]
    if df_var.empty:
        return 0

    mean_psnr = df_var['PSNR'].mean()
    seed_means = df_var.groupby("Seed")['PSNR'].mean()
    best_seed = (seed_means - mean_psnr).abs().idxmin()
    return best_seed

def plot_qualitative_grid(args, df):
    datasets = df["Dataset"].unique()
    variants = ["unet_3ch", "unet_4ch_t", "unet_4ch_b", "unet_5ch"]

    for dataset in datasets:
        images = select_representative_images(df, dataset)
        if not images:
            continue

        fig, axes = plt.subplots(len(images), 6, figsize=(15, 2.5 * len(images)))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        rep_seeds = {v: get_representative_seed(df, dataset, v) for v in variants}
        headers = ["Input", "UNet-3ch", "UNet-4ch-t", "UNet-4ch-B", "UNet-5ch", "Reference"]

        # Need physics extractor to generate t and B maps for visualization
        # We will assume UDCP is used for visualization maps if available
        physics_extractor = _resolve_physics_extractor("udcp")

        for i, img_name in enumerate(images):
            if dataset == "UIEB":
                in_path = Path(args.data_root) / "UIEB" / "raw-890" / img_name
                ref_path = Path(args.data_root) / "UIEB" / "reference-890" / img_name
            else:
                in_path = Path(args.data_root) / "EUVP" / "test_samples" / "Inp" / img_name
                ref_path = Path(args.data_root) / "EUVP" / "test_samples" / "GTr" / img_name

            if in_path.exists():
                in_img_pil = Image.open(in_path).convert('RGB')
                in_img = np.array(in_img_pil)
                in_img_tensor = TF.to_tensor(in_img_pil.resize((256, 256)))
                # Extract maps using the preprocessor
                physics_tensor = _add_physics_channels(in_img_tensor, "tb", physics_extractor)
                t_map = physics_tensor[3].numpy()
                b_map = physics_tensor[4].numpy()
            else:
                in_img = np.zeros((256, 256, 3))
                t_map = np.zeros((256, 256))
                b_map = np.zeros((256, 256))

            ref_img = np.array(Image.open(ref_path)) if ref_path.exists() else np.zeros((256, 256, 3))

            row_images = [in_img]

            for v in variants:
                res_path = Path(args.output_dir) / "restored" / dataset / v / f"seed_{rep_seeds[v]}" / img_name
                res_img = np.array(Image.open(res_path)) if res_path.exists() else np.zeros((256, 256, 3))
                row_images.append(res_img)

            row_images.append(ref_img)

            for j, (ax, img) in enumerate(zip(axes[i], row_images)):
                ax.imshow(img)
                ax.axis('off')
                if i == 0:
                    ax.set_title(headers[j], fontsize=12)

                if j == 0 and ref_path.exists() and in_path.exists():
                    in_psnr = compute_psnr(in_img.astype(np.float32)/255.0, ref_img.astype(np.float32)/255.0)
                    in_ssim = compute_ssim(in_img.astype(np.float32)/255.0, ref_img.astype(np.float32)/255.0)
                    metric_str = f"PSNR/SSIM\n{in_psnr:.2f} / {in_ssim:.3f}"
                    ax.text(0.05, 0.05, metric_str, color='white', fontsize=9,
                            ha='left', va='bottom', transform=ax.transAxes,
                            bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

                elif 1 <= j <= len(variants):
                    v = variants[j-1]
                    row_df = df[(df["Dataset"] == dataset) & (df["Variant"] == v) & (df["Seed"] == rep_seeds[v]) & (df["Image"] == img_name)]
                    if not row_df.empty:
                        psnr = row_df["PSNR"].values[0]
                        ssim = row_df["SSIM"].values[0]
                        metric_str = f"{psnr:.2f} / {ssim:.3f}"
                        ax.text(0.05, 0.05, metric_str, color='white', fontsize=9,
                                ha='left', va='bottom', transform=ax.transAxes,
                                bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

        plt.tight_layout()
        out_pdf = Path(args.output_dir) / f"qualitative_{dataset.lower()}.pdf"
        out_png = Path(args.output_dir) / f"qualitative_{dataset.lower()}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)

    # Now generate the combined 4-row figure
    for idx in range(3):
        plot_combined_qualitative_grid(args, df, config_idx=idx)

def plot_combined_qualitative_grid(args, df, config_idx=0):
    variants = ["unet_3ch", "unet_4ch_t", "unet_4ch_b", "unet_5ch"]
    datasets_to_plot = ["EUVP", "UIEB"]

    selected_rows = [] # Will store tuples of (dataset, img_name)

    for dataset in datasets_to_plot:
        if dataset not in df["Dataset"].unique():
            continue

        rep_seeds = {v: get_representative_seed(df, dataset, v) for v in variants}

        df_3ch = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_3ch") & (df["Seed"] == rep_seeds["unet_3ch"])].groupby("Image").mean(numeric_only=True)
        df_4ch_t = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_4ch_t") & (df["Seed"] == rep_seeds["unet_4ch_t"])].groupby("Image").mean(numeric_only=True)
        df_4ch_b = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_4ch_b") & (df["Seed"] == rep_seeds["unet_4ch_b"])].groupby("Image").mean(numeric_only=True)
        df_5ch = df[(df["Dataset"] == dataset) & (df["Variant"] == "unet_5ch") & (df["Seed"] == rep_seeds["unet_5ch"])].groupby("Image").mean(numeric_only=True)

        if df_3ch.empty or df_5ch.empty:
            continue

        diff_psnr = df_5ch["PSNR"] - df_3ch["PSNR"]
        diff_ciede = df_3ch["CIEDE2000"] - df_5ch["CIEDE2000"]

        # Calculate how much 5ch beats the best of the other variants
        max_others_psnr = pd.concat([df_3ch["PSNR"], df_4ch_t["PSNR"], df_4ch_b["PSNR"]], axis=1).max(axis=1)
        diff_psnr_all = df_5ch["PSNR"] - max_others_psnr

        max_others_ssim = pd.concat([df_3ch["SSIM"], df_4ch_t["SSIM"], df_4ch_b["SSIM"]], axis=1).max(axis=1)
        diff_ssim_all = df_5ch["SSIM"] - max_others_ssim

        # Pick a different image for EUVP (row 1), but keep the previous one for UIEB (row 3)
        if dataset == "EUVP":
            valid_images = diff_psnr_all[(diff_psnr_all > 0) & (diff_ssim_all > 0)]
            if len(valid_images) > config_idx:
                img_best_psnr = valid_images.nlargest(config_idx + 1).index[config_idx]
            elif not valid_images.empty:
                img_best_psnr = valid_images.nlargest(1).index[0]
            else:
                img_best_psnr = diff_psnr_all.nlargest(config_idx + 1).index[config_idx] if len(diff_psnr_all) > config_idx else diff_psnr_all.idxmax()
        else:
            img_best_psnr = diff_psnr.nlargest(4 + config_idx).index[2 + config_idx] if len(diff_psnr) > 2 + config_idx else diff_psnr.idxmax()

        diff_psnr_rem = diff_psnr.drop([img_best_psnr], errors='ignore')
        closest_to_zero = diff_psnr_rem.abs().nsmallest(config_idx + 1).index.tolist()
        img_normal = closest_to_zero[config_idx] if len(closest_to_zero) > config_idx else (closest_to_zero[0] if len(closest_to_zero) > 0 else img_best_psnr)

        selected_rows.append((dataset, img_best_psnr))
        selected_rows.append((dataset, img_normal))

    if not selected_rows:
        return

    fig, axes = plt.subplots(len(selected_rows), 6, figsize=(15, 2.5 * len(selected_rows)))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    headers = ["Input", "UNet-3ch", "UNet-4ch-t", "UNet-4ch-B", "UNet-5ch", "Reference"]

    for i, (dataset, img_name) in enumerate(selected_rows):
        rep_seeds = {v: get_representative_seed(df, dataset, v) for v in variants}

        if dataset == "UIEB":
            in_path = Path(args.data_root) / "UIEB" / "raw-890" / img_name
            ref_path = Path(args.data_root) / "UIEB" / "reference-890" / img_name
        else:
            in_path = Path(args.data_root) / "EUVP" / "test_samples" / "Inp" / img_name
            ref_path = Path(args.data_root) / "EUVP" / "test_samples" / "GTr" / img_name

        if in_path.exists():
            in_img_pil = Image.open(in_path).convert('RGB')
            in_img = np.array(in_img_pil)
        else:
            in_img = np.zeros((256, 256, 3))

        ref_img = np.array(Image.open(ref_path)) if ref_path.exists() else np.zeros((256, 256, 3))
        row_images = [in_img]

        for v in variants:
            res_path = Path(args.output_dir) / "restored" / dataset / v / f"seed_{rep_seeds[v]}" / img_name
            res_img = np.array(Image.open(res_path)) if res_path.exists() else np.zeros((256, 256, 3))
            row_images.append(res_img)

        row_images.append(ref_img)

        for j, (ax, img) in enumerate(zip(axes[i], row_images)):
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(headers[j], fontsize=12)

            if j == 0 and ref_path.exists() and in_path.exists():
                in_psnr = compute_psnr(in_img.astype(np.float32)/255.0, ref_img.astype(np.float32)/255.0)
                in_ssim = compute_ssim(in_img.astype(np.float32)/255.0, ref_img.astype(np.float32)/255.0)
                metric_str = f"PSNR/SSIM\n{in_psnr:.2f} / {in_ssim:.3f}"
                ax.text(0.05, 0.05, metric_str, color='white', fontsize=9,
                        ha='left', va='bottom', transform=ax.transAxes,
                        bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

                # Add row label (i, ii, iii, iv) to the top left of the Input image
                row_labels = ["(i)", "(ii)", "(iii)", "(iv)"]
                ax.text(0.05, 0.95, row_labels[i], color='black', fontsize=11, fontweight='bold',
                        ha='left', va='top', transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))

            elif 1 <= j <= len(variants):
                v = variants[j-1]
                row_df = df[(df["Dataset"] == dataset) & (df["Variant"] == v) & (df["Seed"] == rep_seeds[v]) & (df["Image"] == img_name)]
                if not row_df.empty:
                    psnr = row_df["PSNR"].values[0]
                    ssim = row_df["SSIM"].values[0]
                    metric_str = f"{psnr:.2f} / {ssim:.3f}"
                    ax.text(0.05, 0.05, metric_str, color='white', fontsize=9,
                            ha='left', va='bottom', transform=ax.transAxes,
                            bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

    plt.tight_layout()
    out_pdf = Path(args.output_dir) / f"qualitative_combined_{config_idx + 1}.pdf"
    out_png = Path(args.output_dir) / f"qualitative_combined_{config_idx + 1}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

def plot_metric_bars(args, df_overall_stats):
    metrics = ["PSNR", "SSIM", "CIEDE2000", "UCIQE", "UIQM"]
    datasets = df_overall_stats["Dataset"].unique() if "Dataset" in df_overall_stats.columns else []

    df = df_overall_stats.copy()

    # Check if columns are MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]

    dataset_col = "Dataset"
    variant_col = "Variant"

    for dataset in datasets:
        df_ds = df[df[dataset_col] == dataset]
        if df_ds.empty:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        variants = df_ds[variant_col].values
        x = np.arange(len(variants))
        width = 0.6

        for i, metric in enumerate(metrics):
            means = df_ds[f"{metric}_mean"].values
            stds = df_ds[f"{metric}_std"].values

            axes[i].bar(x, means, width, yerr=stds, capsize=5, color=sns.color_palette("muted")[0:len(variants)], edgecolor='black')
            axes[i].set_title(metric)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(variants, rotation=45, ha="right")

        plt.tight_layout()
        out_pdf = Path(args.output_dir) / f"metrics_{dataset.lower()}.pdf"
        out_png = Path(args.output_dir) / f"metrics_{dataset.lower()}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)

def plot_improvement_heatmap(args, df_overall_stats):
    metrics = ["PSNR", "SSIM", "CIEDE2000", "UCIQE", "UIQM"]
    physics_variants = ["unet_4ch_t", "unet_4ch_b", "unet_5ch"]

    df = df_overall_stats.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]

    dataset_col = "Dataset"
    variant_col = "Variant"

    datasets = df[dataset_col].unique()

    for dataset in datasets:
        df_ds = df[df[dataset_col] == dataset].set_index(variant_col)
        if "unet_3ch" not in df_ds.index:
            continue

        heatmap_data = []
        for var in physics_variants:
            if var not in df_ds.index:
                continue
            row = []
            for metric in metrics:
                base_val = df_ds.loc["unet_3ch", f"{metric}_mean"]
                var_val = df_ds.loc[var, f"{metric}_mean"]

                if metric == "CIEDE2000":
                    diff = base_val - var_val
                else:
                    diff = var_val - base_val

                rel_diff = (diff / base_val) * 100
                row.append(rel_diff)
            heatmap_data.append(row)

        if not heatmap_data:
            continue

        heatmap_df = pd.DataFrame(heatmap_data, index=physics_variants, columns=metrics)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={'label': 'Improvement (%)'})
        ax.set_title(f"Relative Improvement vs UNet-3ch ({dataset})")
        plt.tight_layout()

        out_pdf = Path(args.output_dir) / f"improvement_heatmap_{dataset.lower()}.pdf"
        out_png = Path(args.output_dir) / f"improvement_heatmap_{dataset.lower()}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)

def plot_failure_cases(args, df):
    datasets = df["Dataset"].unique()

    for dataset in datasets:
        df_ds = df[df["Dataset"] == dataset]
        if df_ds.empty:
            continue

        df_avg = df_ds.groupby(["Variant", "Image"]).mean(numeric_only=True).reset_index()
        df_3ch = df_avg[df_avg["Variant"] == "unet_3ch"].set_index("Image")
        df_5ch = df_avg[df_avg["Variant"] == "unet_5ch"].set_index("Image")

        if df_3ch.empty or df_5ch.empty:
            continue

        diff = df_5ch["PSNR"] - df_3ch["PSNR"]
        failure_images = diff.nsmallest(2).index.tolist()

        if not failure_images:
            continue

        fig, axes = plt.subplots(len(failure_images), 6, figsize=(15, 2.5 * len(failure_images)))
        if len(failure_images) == 1:
            axes = [axes]
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        headers = ["Input", "UNet-3ch", "UNet-5ch", "Reference", "Abs Error 3ch", "Abs Error 5ch"]

        rep_seed_3ch = get_representative_seed(df, dataset, "unet_3ch")
        rep_seed_5ch = get_representative_seed(df, dataset, "unet_5ch")

        for i, img_name in enumerate(failure_images):
            if dataset == "UIEB":
                in_path = Path(args.data_root) / "UIEB" / "raw-890" / img_name
                ref_path = Path(args.data_root) / "UIEB" / "reference-890" / img_name
            else:
                in_path = Path(args.data_root) / "EUVP" / "test_samples" / "Inp" / img_name
                ref_path = Path(args.data_root) / "EUVP" / "test_samples" / "GTr" / img_name

            res3_path = Path(args.output_dir) / "restored" / dataset / "unet_3ch" / f"seed_{rep_seed_3ch}" / img_name
            res5_path = Path(args.output_dir) / "restored" / dataset / "unet_5ch" / f"seed_{rep_seed_5ch}" / img_name

            in_img = np.array(Image.open(in_path)) / 255.0
            ref_img = np.array(Image.open(ref_path)) / 255.0
            res3_img = np.array(Image.open(res3_path)) / 255.0
            res5_img = np.array(Image.open(res5_path)) / 255.0

            err3 = np.mean(np.abs(res3_img - ref_img), axis=2)
            err5 = np.mean(np.abs(res5_img - ref_img), axis=2)

            row_images = [in_img, res3_img, res5_img, ref_img, err3, err5]

            for j, (ax, img) in enumerate(zip(axes[i], row_images)):
                if j >= 4:
                    im = ax.imshow(img, cmap='hot', vmin=0, vmax=1.0)
                else:
                    ax.imshow(img)
                ax.axis('off')
                if i == 0:
                    ax.set_title(headers[j], fontsize=12)

        plt.tight_layout()
        out_pdf = Path(args.output_dir) / f"failure_cases_{dataset.lower()}.pdf"
        out_png = Path(args.output_dir) / f"failure_cases_{dataset.lower()}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize and Evaluate Ablation Checkpoints")
    parser.add_argument("--ablation_dirs", type=str, nargs="+", default=["ablation_uieb_udcp", "ablation_euvp_udcp"],
                        help="List of ablation directories in checkpoints/ to evaluate.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/", help="Root directory containing model checkpoints.")
    parser.add_argument("--data_root", type=str, default="datasets/", help="Root directory containing datasets.")
    parser.add_argument("--output_dir", type=str, default="figures/", help="Output directory for restored images and figures.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    args = parser.parse_args()

    setup_matplotlib_for_paper()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Step 1: Running Inference and saving restored images...")
    run_inference(args)

    print("Step 2: Computing metrics...")
    df_metrics, df_stats = compute_all_metrics(args)

    if df_metrics is not None and not df_metrics.empty:
        print("Step 3: Generating paper-ready figures...")
        print("  Generating qualitative grids...")
        plot_qualitative_grid(args, df_metrics)

        print("  Generating metric bar plots...")
        plot_metric_bars(args, df_stats)

        print("  Generating improvement heatmaps...")
        plot_improvement_heatmap(args, df_stats)

        print("  Generating failure cases...")
        plot_failure_cases(args, df_metrics)

        print(f"Done! Results saved to {args.output_dir}")
    else:
        print("Metric computation failed. Check data and checkpoint paths.")

if __name__ == "__main__":
    main()
