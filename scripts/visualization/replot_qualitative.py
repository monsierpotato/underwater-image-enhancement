import pandas as pd
import argparse
from scripts.visualization.ablation import plot_qualitative_grid, setup_matplotlib_for_paper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/")
    parser.add_argument("--output_dir", type=str, default="figures/")
    args = parser.parse_args()

    setup_matplotlib_for_paper()
    df = pd.read_csv("figures/metrics_per_image.csv")

    print("Generating qualitative grids...")
    plot_qualitative_grid(args, df)
    print("Done!")

if __name__ == "__main__":
    main()
