import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from NOVA_rotation.load_files.load_data_from_npy import load_labels_from_npy

# Path to the directory containing marker subdirectories
BASE_DIR = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/attention_maps/output_for_presentation"

models = [
    'finetuned_model',
    'finetuned_model_classification_with_batch_freeze',
    'pretrained_model'
]

datasets = [
    ("deltaNLS", "batch4", "dNLSB4TDP43SubsetConfigForPresentation"),
    ("neurons", "batch9", "WTB9SubsetConfigForPresentation")
]
# exp = ["deltaNLS", "neurons"]
# batch = "batch4"
# config = "dNLSB4TDP43SubsetConfigForPresentation"

# exp = "neurons"
# batch = "batch9"
# config = "WTB9SubsetConfigForPresentation"


corr_method = "pearsonr" # attn_overlap, pearsonr
settype = "testset"
# DATA_DIR = os.path.join(BASE_DIR, model, "correlations/rollout",corr_method, exp, batch, config)
# LABELS_DIR = os.path.join(BASE_DIR, model, "processed", exp, batch)
# labels_df = load_labels_from_npy(LABELS_DIR, settype)

# Store average correlations for each marker
markers = []
avg_corr_ch0 = []
avg_corr_ch1 = []


def plot_hist(corr, ch, output_path):
    plt.hist(corr, bins=30, alpha=0.7, color='skyblue')
    plt.title(f"Distribution of Correlation Scores ch:{ch}")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"histogram_ch{ch}"), dpi=300)


def nucleus_vs_marker(hist = False):
    # Iterate through each marker's subdirectory
    for marker_name in os.listdir(DATA_DIR):
        marker_dir = os.path.join(DATA_DIR, marker_name)
        if not os.path.isdir(marker_dir):
            continue

        try:
            # Load correlation arrays
            ch0_corrs = np.load(os.path.join(marker_dir, f"{settype}_corrs_ch0.npy"))
            ch1_corrs = np.load(os.path.join(marker_dir, f"{settype}_corrs_ch1.npy"))

            # Calculate mean correlations
            mean_ch0 = np.mean(ch0_corrs)
            mean_ch1 = np.mean(ch1_corrs)

            # Store results
            markers.append(marker_name)
            avg_corr_ch0.append(mean_ch0)
            avg_corr_ch1.append(mean_ch1)

            if hist:
                plot_hist(ch0_corrs, 0, marker_dir)
                plot_hist(ch1_corrs, 1, marker_dir)

    

        except Exception as e:
            print(f"Skipping {marker_name} due to error: {e}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(avg_corr_ch0, avg_corr_ch1)

    # Annotate points with marker names
    for i, name in enumerate(markers):
        plt.annotate(name, (avg_corr_ch0[i], avg_corr_ch1[i]), fontsize=8)

    # Dashed y = x line
    # lims = [min(avg_corr_ch0 + avg_corr_ch1), max(avg_corr_ch0 + avg_corr_ch1)]
    lims = [0.1, 0.65]
    plt.plot(lims, lims, 'k--', alpha=0.5)

    # Labels and title
    plt.xlabel(f"Nucleus")
    plt.ylabel(f"Marker")
    plt.title(f"Average {corr_method} Attention Correlation per Marker")
    plt.xlim(0.1, 0.65)
    plt.ylim(0.1, 0.65)
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(DATA_DIR, "nucleus_vs_marker.png"), dpi=300)

def stress_vs_wt():

    ch0_corrs = np.load(os.path.join(DATA_DIR, f"{settype}_corrs_ch0.npy"))
    ch1_corrs = np.load(os.path.join(DATA_DIR, f"{settype}_corrs_ch1.npy"))
    labels_df['corr_ch0'] = ch0_corrs
    labels_df['corr_ch1'] = ch1_corrs
    grouped = labels_df.groupby(['protein', 'treatment']).agg({
        'corr_ch0': 'mean',
        'corr_ch1': 'mean'
    }).reset_index()


    print(grouped)

    # Plot ch1 (marker) correlation as bars
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=labels_df, x='protein', y='corr_ch1', hue='treatment')
    plt.axhline(0, linestyle='--', color='gray', alpha=0.7)
    plt.ylabel("Average Marker Correlation")
    plt.title(f"Marker {corr_method} Correlation per Treatment")
    plt.xticks(rotation=45)
    plt.ylim(-1, 1)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(DATA_DIR, "stress_vs_wt.png"), dpi=300)

def different_models(BASE_DIR, models, datasets, corr_method="pearsonr"):
    all_data = []

    # Load and aggregate correlation values
    for model in models:
        for exp, batch, config in datasets:
            data_dir = os.path.join(BASE_DIR, model, "correlations/rollout", corr_method, exp, batch, config)
            dataset_label = f"{exp}_{batch}"

            if not os.path.isdir(data_dir):
                continue

            for marker in os.listdir(data_dir):
                marker_dir = os.path.join(data_dir, marker)
                if not os.path.isdir(marker_dir):
                    continue

                ch1_path = os.path.join(marker_dir, "testset_corrs_ch1.npy")
                if not os.path.exists(ch1_path):
                    continue

                ch1_corrs = np.load(ch1_path)

                # Normalize marker name
                if marker in ["TDP43", "TDP43B"]:
                    marker = "TDP43"

                for val in ch1_corrs:
                    all_data.append({
                        "Marker": marker,
                        "Correlation": val,
                        "Model": model,
                        "Dataset": dataset_label
                    })

    if not all_data:
        print("No data found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Map clean names for display
    model_map = {
        "pretrained_model": "Pretrained",
        "finetuned_model": "Finetuned",
        "finetuned_model_classification_with_batch_freeze": "Finetuned + Classifier"
    }
    dataset_map = {
        "deltaNLS_batch4": "ΔNLS - Batch 4",
        "neurons_batch9": "Neurons - Batch 9"
    }

    df["ModelLabel"] = df["Model"].map(model_map)
    df["DatasetLabel"] = df["Dataset"].map(dataset_map)

    model_labels = df["ModelLabel"].dropna().unique()
    dataset_labels = df["DatasetLabel"].dropna().unique()

    # Create grid of subplots
    fig, axes = plt.subplots(
        len(model_labels), len(dataset_labels),
        figsize=(len(dataset_labels) * 5, len(model_labels) * 4),
        sharex=True, sharey=True
    )

    if len(model_labels) == 1:
        axes = axes.reshape(1, -1)
    if len(dataset_labels) == 1:
        axes = axes.reshape(-1, 1)

    for i, model in enumerate(model_labels):
        for j, dataset in enumerate(dataset_labels):
            ax = axes[i, j]
            subset = df[(df["ModelLabel"] == model) & (df["DatasetLabel"] == dataset)]
            if subset.empty:
                ax.axis("off")
                continue

            sns.boxplot(data=subset, x="Marker", y="Correlation", ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title("")  # We'll add custom titles

            # Only show y-label on first column
            if j == 0:
                ax.set_ylabel("Attention Correlation (ch1)")
            else:
                ax.set_ylabel("")

            # Only show x-labels on last row
            if i == len(model_labels) - 1:
                ax.set_xlabel("Marker")
            else:
                ax.set_xlabel("")

    # Add row titles (Model names)
    for i, model in enumerate(model_labels):
        axes[i, 0].text(-0.2, 0.5, model, transform=axes[i, 0].transAxes,
                        rotation=90, ha='center', va='center', fontsize=14, weight='bold')

    # Add column titles (Dataset names)
    for j, dataset in enumerate(dataset_labels):
        axes[0, j].text(0.5, 1.12, dataset, transform=axes[0, j].transAxes,
                        ha='center', va='bottom', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


    plt.savefig(os.path.join(BASE_DIR, "models_compare.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    #nucleus_vs_marker()
    #stress_vs_wt()
    different_models(BASE_DIR, models, datasets)


if __name__ == "__main__":
    main()

