import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ["#FF6B9D", "#C44BFF", "#4BAAFF", "#FFD93D", "#6BCB77", "#FF6B35", "#00D9C0", "#FF4757"]
BG_COLOR   = "#0D0D1A"
CARD_COLOR = "#161628"
GRID_COLOR = "#252540"
TEXT_COLOR = "#F0F0FF"

TYPE_COLORS = {
    "Normal":   "#A8A878", "Fire":     "#F08030", "Water":    "#6890F0",
    "Electric": "#F8D030", "Grass":    "#78C850", "Ice":      "#98D8D8",
    "Fighting": "#C03028", "Poison":   "#A040A0", "Ground":   "#E0C068",
    "Flying":   "#A890F0", "Psychic":  "#F85888", "Bug":      "#A8B820",
    "Rock":     "#B8A038", "Ghost":    "#705898", "Dragon":   "#7038F8",
    "Dark":     "#705848", "Steel":    "#B8B8D0", "Fairy":    "#EE99AC",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": CARD_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 1.0,
    "grid.alpha": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT_COLOR,
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": CARD_COLOR,
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
})

# -------------------------
# LOAD & CLEAN DATA
# -------------------------
df = pd.read_csv("dataset.csv")
df["Name"]  = df["Name"].str.strip().str.title()
df["Type1"] = df["Type1"].str.strip().str.title()
df["Type2"] = df["Type2"].str.strip().str.title()

# Separate dual-type only (no None)
df_dual = df.dropna(subset=["Type2"]).copy()

print(f"Loaded {len(df):,} Pokemon records ({len(df_dual):,} dual-type).")

# -------------------------
# GRAPH 1: Type 1 Distribution
# -------------------------
type1_counts = df["Type1"].value_counts()
colors = [TYPE_COLORS.get(t, "#888888") for t in type1_counts.index]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)
bars = ax.bar(type1_counts.index, type1_counts.values, color=colors, edgecolor="none", width=0.7)
ax.set_title("Primary Type Distribution", pad=15)
ax.set_xlabel("Type", labelpad=10)
ax.set_ylabel("Count", labelpad=10)
ax.tick_params(axis="x", rotation=45)

for bar, val, color in zip(bars, type1_counts.values, colors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "type1_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: type1_distribution.png")

# -------------------------
# GRAPH 2: Dual-Type Heatmap (no None)
# -------------------------
all_types = sorted(TYPE_COLORS.keys())
matrix = pd.DataFrame(0, index=all_types, columns=all_types)

for _, row in df_dual.iterrows():
    t1, t2 = row["Type1"], row["Type2"]
    if t1 in all_types and t2 in all_types:
        matrix.loc[t1, t2] += 1

fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

cmap = LinearSegmentedColormap.from_list("poke", ["#161628", "#C44BFF", "#FF6B9D"])
im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")

ax.set_xticks(range(len(all_types)))
ax.set_yticks(range(len(all_types)))
ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(all_types, fontsize=9)
ax.set_title("Dual-Type Combination Heatmap", pad=15)
ax.set_xlabel("Type 2", labelpad=10)
ax.set_ylabel("Type 1", labelpad=10)

for i in range(len(all_types)):
    for j in range(len(all_types)):
        val = matrix.values[i, j]
        if val > 0:
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dual_type_heatmap.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: dual_type_heatmap.png")

# -------------------------
# GRAPH 3: Top 15 Dual-Type Combos with Pokemon names
# -------------------------
df_dual["type_combo"] = df_dual["Type1"] + " / " + df_dual["Type2"]
combo_counts = df_dual["type_combo"].value_counts().head(15)

# Get up to 3 example Pokemon names per combo
def get_examples(combo, n=3):
    t1, t2 = combo.split(" / ")
    names = df_dual[(df_dual["Type1"] == t1) & (df_dual["Type2"] == t2)]["Name"].tolist()
    sample = names[:n]
    label = ", ".join(sample)
    if len(names) > n:
        label += f" +{len(names) - n} more"
    return label

example_labels = [get_examples(c) for c in combo_counts.index]
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(combo_counts))]

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)
bars = ax.barh(combo_counts.index, combo_counts.values,
               color=bar_colors, edgecolor="none", height=0.65)
ax.set_title("Top 15 Dual-Type Combinations (no single-type)", pad=15)
ax.set_xlabel("Count", labelpad=10)
ax.invert_yaxis()

# Set x limit with room for labels
ax.set_xlim(0, combo_counts.values.max() + 14)

for bar, val, color, examples in zip(bars, combo_counts.values, bar_colors, example_labels):
    # Count on the right of bar
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val}  |  {examples}",
            va="center", ha="left", fontsize=8,
            color=TEXT_COLOR)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_type_combos.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_type_combos.png")

# -------------------------
# GRAPH 4: Evolution Coverage Pie
# -------------------------
has_evolution = df["Evolution"].notna().sum()
no_evolution  = df["Evolution"].isna().sum()
evo_labels    = ["Has Evolution", "Final / No Evolution"]
evo_values    = [has_evolution, no_evolution]
evo_colors    = ["#C44BFF", "#FF6B9D"]

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

wedges, texts, autotexts = ax.pie(
    evo_values,
    labels=evo_labels,
    autopct="%1.1f%%",
    colors=evo_colors,
    startangle=140,
    pctdistance=0.78,
    wedgeprops=dict(edgecolor=BG_COLOR, linewidth=3),
    textprops=dict(color=TEXT_COLOR, fontsize=12),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color(BG_COLOR)

ax.set_title("Evolution Coverage", pad=20, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "evolution_coverage.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: evolution_coverage.png")

# -------------------------
# GRAPH 5: Single-type vs Dual-type per Type1
# -------------------------
df["is_dual"] = df["Type2"].notna()
dual_breakdown = df.groupby("Type1")["is_dual"].agg(
    Dual=lambda x: x.sum(),
    Single=lambda x: (~x).sum()
).reindex(type1_counts.index)

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)

x = np.arange(len(dual_breakdown))
w = 0.4
ax.bar(x - w/2, dual_breakdown["Single"], width=w, label="Single-type",
       color="#4BAAFF", edgecolor="none")
ax.bar(x + w/2, dual_breakdown["Dual"],   width=w, label="Dual-type",
       color="#FF6B9D", edgecolor="none")

ax.set_title("Single-type vs Dual-type per Primary Type", pad=15)
ax.set_xlabel("Type 1", labelpad=10)
ax.set_ylabel("Count", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(dual_breakdown.index, rotation=45, ha="right", fontsize=9)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "single_vs_dual.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: single_vs_dual.png")

# -------------------------
# GRAPH 6: Rarest Types (bottom 5) with example Pokemon
# -------------------------
rare_types  = type1_counts.tail(5)
rare_colors = [TYPE_COLORS.get(t, "#888888") for t in rare_types.index]

# Get up to 4 example names per rare type
def rare_examples(type_name, n=4):
    names = df[df["Type1"] == type_name]["Name"].tolist()
    sample = names[:n]
    label = ", ".join(sample)
    if len(names) > n:
        label += f" +{len(names) - n} more"
    return label

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG_COLOR)
bars = ax.barh(rare_types.index, rare_types.values,
               color=rare_colors, edgecolor="none", height=0.55)
ax.set_title("5 Rarest Primary Types", pad=15)
ax.set_xlabel("Count", labelpad=10)
ax.invert_yaxis()
ax.set_xlim(0, rare_types.values.max() + 20)

for bar, val, color, t in zip(bars, rare_types.values, rare_colors, rare_types.index):
    examples = rare_examples(t)
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val}  |  {examples}",
            va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rarest_types.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: rarest_types.png")

print("\nAll graphs generated successfully!")