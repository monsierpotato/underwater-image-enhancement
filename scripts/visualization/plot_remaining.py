import pandas as pd
import argparse
from scripts.visualization.ablation import plot_improvement_heatmap, plot_failure_cases, setup_matplotlib_for_paper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/")
    parser.add_argument("--output_dir", type=str, default="figures/")
    args = parser.parse_args()

    setup_matplotlib_for_paper()

    df_per_image = pd.read_csv("figures/metrics_per_image.csv")

    df_overall = df_per_image.groupby(["Dataset", "Variant", "Seed"]).mean(numeric_only=True).reset_index()
    df_overall_stats = df_overall.groupby(["Dataset", "Variant"]).agg(
        {metric: ['mean', 'std'] for metric in ["PSNR", "SSIM", "CIEDE2000", "UCIQE", "UIQM"]}
    ).reset_index()
    df_overall_stats.columns = ['_'.join(col).strip('_') for col in df_overall_stats.columns.values]

    print("Generating improvement heatmaps...")
    plot_improvement_heatmap(args, df_overall_stats)

    print("Generating failure cases...")
    plot_failure_cases(args, df_per_image)

    print("Done!")

if __name__ == "__main__":
    main()
