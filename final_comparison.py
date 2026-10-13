import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULTS = ROOT / "results"
OUTPUT = RESULTS / "final_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "resnet50",
    "vgg19",
    "densenet121",
    "proposed"
]

DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "vgg19": "VGG19",
    "densenet121": "DenseNet121",
    "proposed": "Proposed OCTNet"
}


# ============================================================
# READ CLASSIFICATION REPORTS
# ============================================================

results = []

for model in MODELS:

    report_file = (
        RESULTS /
        f"{model}_classification_report.txt"
    )

    if not report_file.exists():

        print(
            f"WARNING: Report not found: "
            f"{report_file}"
        )

        continue

    text = report_file.read_text()

    # --------------------------------------------------------
    # Test Accuracy
    # --------------------------------------------------------

    accuracy_match = re.search(
        r"Test Accuracy:\s*([\d.]+)%",
        text
    )

    # --------------------------------------------------------
    # Macro Average
    # --------------------------------------------------------

    macro_match = re.search(
        r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text
    )

    if accuracy_match and macro_match:

        accuracy = float(
            accuracy_match.group(1)
        )

        precision = float(
            macro_match.group(1)
        ) * 100

        recall = float(
            macro_match.group(2)
        ) * 100

        f1 = float(
            macro_match.group(3)
        ) * 100

        results.append({
            "Model": DISPLAY_NAMES[model],
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


if df.empty:

    print("ERROR: No model results found.")

    exit()


# ============================================================
# READ ROBUSTNESS RESULTS
# ============================================================

robustness_file = (
    RESULTS /
    "robustness_comparison.csv"
)

if robustness_file.exists():

    robustness_df = pd.read_csv(
        robustness_file
    )

    print("\nRobustness file loaded.")

    print(
        robustness_df.head()
    )

else:

    robustness_df = None

    print(
        "\nWARNING: robustness_comparison.csv "
        "not found."
    )


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}%".format,
            "Precision": "{:.2f}%".format,
            "Recall": "{:.2f}%".format,
            "F1-Score": "{:.2f}%".format
        }
    )
)


# ============================================================
# SAVE CSV
# ============================================================

comparison_file = (
    OUTPUT /
    "final_model_comparison.csv"
)

df.to_csv(
    comparison_file,
    index=False
)

print(
    f"\nComparison saved:\n"
    f"{comparison_file}"
)


# ============================================================
# BEST MODEL
# ============================================================

best_accuracy = df.loc[
    df["Accuracy"].idxmax()
]

best_f1 = df.loc[
    df["F1-Score"].idxmax()
]

print("\n")
print("=" * 80)
print("BEST MODELS")
print("=" * 80)

print(
    f"Best Accuracy : "
    f"{best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.2f}%)"
)

print(
    f"Best F1-Score : "
    f"{best_f1['Model']} "
    f"({best_f1['F1-Score']:.2f}%)"
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("OCT-C8 Test Accuracy Comparison")

plt.ylim(
    max(0, df["Accuracy"].min() - 2),
    100
)

for i, value in enumerate(
    df["Accuracy"]
):

    plt.text(
        i,
        value + 0.05,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

accuracy_graph = (
    OUTPUT /
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION / RECALL / F1 GRAPH
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1-Score"
]

x = range(len(df))

width = 0.25

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):

    values = df[metric]

    positions = [
        p + (i - 1) * width
        for p in x
    ]

    plt.bar(
        positions,
        values,
        width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"]
)

plt.ylabel("Score (%)")
plt.xlabel("Model")

plt.title(
    "Precision, Recall and F1-Score Comparison"
)

plt.ylim(90, 100)

plt.legend()

plt.tight_layout()

metrics_graph = (
    OUTPUT /
    "precision_recall_f1_comparison.png"
)

plt.savefig(
    metrics_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROBUSTNESS GRAPH
# ============================================================

if robustness_df is not None:

    # Try to identify model and condition columns
    model_column = None

    for column in robustness_df.columns:

        if column.lower() == "model":

            model_column = column
            break

    if model_column:

        numeric_columns = (
            robustness_df
            .select_dtypes(
                include="number"
            )
            .columns
        )

        if len(numeric_columns) > 0:

            robustness_mean = (
                robustness_df
                .groupby(model_column)[
                    list(numeric_columns)
                ]
                .mean()
            )

            robustness_mean["Average"] = (
                robustness_mean.mean(
                    axis=1
                )
            )

            plt.figure(
                figsize=(10, 6)
            )

            values = []

            labels = []

            for model in MODELS:

                display_name = (
                    DISPLAY_NAMES[model]
                )

                if display_name in robustness_mean.index:

                    values.append(
                        robustness_mean.loc[
                            display_name,
                            "Average"
                        ]
                    )

                    labels.append(
                        display_name
                    )

            if values:

                plt.bar(
                    labels,
                    values
                )

                plt.ylabel(
                    "Average Accuracy (%)"
                )

                plt.xlabel(
                    "Model"
                )

                plt.title(
                    "Robustness Comparison"
                )

                plt.tight_layout()

                robustness_graph = (
                    OUTPUT /
                    "robustness_comparison.png"
                )

                plt.savefig(
                    robustness_graph,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_file = (
    OUTPUT /
    "final_comparison_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "OCT-C8 FINAL MODEL COMPARISON\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Best Accuracy Model: "
        f"{best_accuracy['Model']}\n"
    )

    f.write(
        f"Best Accuracy: "
        f"{best_accuracy['Accuracy']:.2f}%\n"
    )

    f.write(
        f"Best F1 Model: "
        f"{best_f1['Model']}\n"
    )

    f.write(
        f"Best F1-Score: "
        f"{best_f1['F1-Score']:.2f}%\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL COMPARISON COMPLETED")
print("=" * 80)

print(
    f"Results saved in:\n"
    f"{OUTPUT}"
)

print("\nGenerated files:")

print(
    "1. final_model_comparison.csv"
)

print(
    "2. accuracy_comparison.png"
)

print(
    "3. precision_recall_f1_comparison.png"
)

print(
    "4. robustness_comparison.png"
)

print(
    "5. final_comparison_summary.txt"
)
