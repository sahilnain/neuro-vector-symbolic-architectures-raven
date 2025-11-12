import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- Main totals (compute only; excluding Input copy to CUDA) ----
main_totals = pd.DataFrame({
    "Main": ["ResNet18", "VSA frontend", "VSA backend"],
    "Total_ms": [10.44, 2.02, 123.89],
})

# ---- Sub-operation breakdowns from profiling ----
resnet_sub = pd.DataFrame({
    "Main": "ResNet18",
    "Sub": ["2D convolution", "ReLU", "2D Batch Norm", "Matrix addition",
            "Adaptive avg pooling", "Flatten", "Linear Transform", "Tanh"],
    "Time_ms": [2.42, 0.71, 1.67, 0.42, 0.13, 0.02, 0.155, 0.051],
})
resnet_total_profiled = 5.56

frontend_sub = pd.DataFrame({
    "Main": "VSA frontend",
    "Sub": ["Linear Transform", "Matrix Multiplication", "Softmax", "Log", "ReLU", "Normalise"],
    "Time_ms": [0.541, 0.20, 0.264, 0.215, 0.048, 0.622],
})
frontend_total_profiled = 1.89

backend_sub = pd.DataFrame({
    "Main": "VSA backend",
    "Sub": ["Circular Convolution", "Circular Correlation", "Cosine Similarity",
            "Matrix Multiplication", "Linear Transform", "Vector sum", "ArgMax",
            "Normalise", "Log", "Exp", "Cross-Entropy"],
    "Time_ms": [31.413, 27.667, 35.356, 0.12, 1.50, 0.551, 0.086, 1.018, 1.880, 0.173, 0.101],
})
backend_total_profiled = 99.865

# Compute "Data restructuring/copy" (black time) per main
resnet_black = 10.44 - resnet_total_profiled
frontend_black = 2.02 - frontend_total_profiled
backend_black = 123.89 - backend_total_profiled

# Append black time rows
resnet_black_row = pd.DataFrame({"Main": ["ResNet18"], "Sub": ["Data restructuring/copy"], "Time_ms": [resnet_black]})
frontend_black_row = pd.DataFrame({"Main": ["VSA frontend"], "Sub": ["Data restructuring/copy"], "Time_ms": [frontend_black]})
backend_black_row = pd.DataFrame({"Main": ["VSA backend"], "Sub": ["Data restructuring/copy"], "Time_ms": [backend_black]})

subs = pd.concat([resnet_sub, resnet_black_row, frontend_sub, frontend_black_row, backend_sub, backend_black_row], ignore_index=True)

# Total compute time
total_compute_ms = main_totals["Total_ms"].sum()  # 136.35 ms

# Compute % of total for each sub-op
subs["Pct_total"] = subs["Time_ms"] / total_compute_ms * 100.0

# Sort sub-operations by total contribution (including black time)
sub_order = subs.groupby("Sub")["Pct_total"].sum().sort_values(ascending=False).index.tolist()

# Pivot for plotting
pivot = subs.pivot_table(index="Main", columns="Sub", values="Pct_total", fill_value=0.0)
pivot = pivot.reindex(columns=sub_order)  # ensure legend order by contribution

# ---- Plot ----
fig, ax = plt.subplots(figsize=(13, 6))
bottom = np.zeros(len(pivot.index))
x = np.arange(len(pivot.index))

for sub in pivot.columns:
    ax.bar(x, pivot[sub].values, bottom=bottom, label=sub)
    bottom += pivot[sub].values

# Annotate main bar totals as % of total
main_pct = main_totals.set_index("Main")["Total_ms"] / total_compute_ms * 100.0
for i, m in enumerate(pivot.index):
    ax.text(i, main_pct[m] + 1.0, f"{main_pct[m]:.1f}%", ha="center", va="bottom", fontsize=10)

# Styling
ax.set_xticks(x)
ax.set_xticklabels(pivot.index)
ax.set_ylabel("% of Total Compute Time")
ax.set_title(f"Processing time (Total compute = {total_compute_ms:.2f} ms)")
ax.set_ylim(0, 105)

# Legend sorted by contribution
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", title="Sub-operation")

fig.tight_layout()

# Save updated figure
out_png_sorted = "processing_time_stacked_percentage.png"
fig.savefig(out_png_sorted, dpi=200, bbox_inches="tight")
