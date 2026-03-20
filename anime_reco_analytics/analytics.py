import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE    = ["#FF6B9D", "#C44BFF", "#4BAAFF", "#FFD93D", "#6BCB77",
              "#FF6B35", "#00D9C0", "#FF4757", "#A29BFE", "#FD79A8"]
BG_COLOR   = "#0D0D1A"
CARD_COLOR = "#161628"
GRID_COLOR = "#252540"
TEXT_COLOR = "#F0F0FF"

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
    "axes.titlesize": 13,
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
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"]         = df["Release_Date"].dt.year
df_rev = df[df["Total_Reviews"] >= 100].copy()

bins   = [0, 40, 60, 70, 80, 90, 95, 100]
labels = ["Overwhelm. Neg", "Mostly Neg", "Mixed",
          "Mostly Pos", "Very Pos", "Overwhelm. Pos", "Perfect"]
df_rev["score_cat"] = pd.cut(df_rev["Review_Score_Pct"], bins=bins, labels=labels)

def best_game(subset, name_len=30):
    if subset.empty:
        return None, None, None
    row  = subset.sort_values(["Review_Score_Pct", "Total_Reviews"],
                               ascending=[False, False]).iloc[0]
    name = row["Name"]
    name = (name[:name_len - 3] + "...") if len(name) > name_len else name
    return name, int(row["Review_Score_Pct"]), int(row["Total_Reviews"])

print(f"Loaded {len(df):,} games | {len(df_rev):,} with 100+ reviews")

# -------------------------
# GRAPH 1: Top 15 Games by Review Score
# -------------------------
top_games  = (df_rev
              .sort_values(["Review_Score_Pct", "Total_Reviews"], ascending=[False, False])
              .head(15))
best_name, best_score, _ = best_game(df_rev)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(top_games))]
bars = ax.barh(top_games["Name"].str[:40], top_games["Review_Score_Pct"],
               color=bar_colors, edgecolor="none", height=0.6)
ax.set_title(f"Top 15 Games by Positive Review %  —  Overall Best: {best_name} ({best_score}%)", pad=15)
ax.set_xlabel("Positive Review %", labelpad=10)
# extra right space so labels don't clip
ax.set_xlim(0, 140)
ax.invert_yaxis()

for bar, score, reviews, color in zip(bars, top_games["Review_Score_Pct"],
                                       top_games["Total_Reviews"], bar_colors):
    ax.text(bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{score}%  |  {int(reviews):,} reviews",
            va="center", ha="left", fontsize=8, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_games_reviews.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_games_reviews.png")

# -------------------------
# GRAPH 2: Price Distribution
# -------------------------
df_paid = df[df["Price_USD"] > 0].copy()

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG_COLOR)
n, bins_p, patches = ax.hist(df_paid["Price_USD"], bins=35, edgecolor="none")
cmap = LinearSegmentedColormap.from_list("steam", ["#4BAAFF", "#C44BFF", "#FF6B9D"])
norm_vals = (bins_p[:-1] - bins_p[:-1].min()) / (bins_p[:-1].max() - bins_p[:-1].min() + 1e-9)
for patch, nv in zip(patches, norm_vals):
    patch.set_facecolor(cmap(nv))

median_p = df_paid["Price_USD"].median()
ax.axvline(median_p, color="#FFD93D", linewidth=1.8, linestyle="--")
ax.text(median_p + 0.5, ax.get_ylim()[1] * 0.88,
        f"Median ${median_p:.2f}", color="#FFD93D", fontsize=9)

ax.set_title(
    f"Price Distribution — {len(df_paid):,} Paid Games  |  {(df['Price_USD']==0).sum()} Free-to-Play",
    pad=15)
ax.set_xlabel("Price (USD)", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: price_distribution.png")

# -------------------------
# GRAPH 3: Top 15 Genres
# — count label on top, best game label BELOW count with a small gap
# -------------------------
genre_counts = df["Primary_Genre"].value_counts().head(15)

def top_game_for_genre(genre):
    subset = df_rev[df_rev["Primary_Genre"] == genre]
    name, score, _ = best_game(subset, name_len=22)
    if name is None:
        return ""
    return f"{name}  {score}%"

top_labels = [top_game_for_genre(g) for g in genre_counts.index]
bar_colors  = [PALETTE[i % len(PALETTE)] for i in range(len(genre_counts))]
max_h       = genre_counts.max()

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG_COLOR)
# Extra top space so labels above bars don't clip
ax.set_ylim(0, max_h * 1.28)
genre_counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="none", width=0.7)
ax.set_title("Top 15 Primary Genres  (best-reviewed game shown above bar)", pad=15)
ax.set_xlabel("Genre", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.tick_params(axis="x", rotation=35)

for p, color, label in zip(ax.patches, bar_colors, top_labels):
    h = p.get_height()
    cx = p.get_x() + p.get_width() / 2
    # count label directly above bar
    ax.text(cx, h + max_h * 0.01,
            f"{int(h):,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
    # best game label above count, smaller, no rotation
    if label:
        ax.text(cx, h + max_h * 0.06,
                label,
                ha="center", va="bottom", fontsize=6.5, color=TEXT_COLOR,
                rotation=40)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_genres.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_genres.png")

# -------------------------
# GRAPH 4: Top 15 Community Tags  +  Multiplayer callout
# -------------------------
tags_split = df["All_Tags"].dropna().str.split(";")
all_tags   = [t.strip() for sub in tags_split for t in sub if t.strip()]
tag_counts = pd.Series(all_tags).value_counts().head(15)
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(tag_counts))]
max_h      = tag_counts.max()

multi_mask  = df_rev["All_Tags"].fillna("").str.contains("Multiplayer", case=False, regex=False)
multi_name, multi_score, multi_reviews = best_game(df_rev[multi_mask], name_len=28)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_ylim(0, max_h * 1.22)
tag_counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="none", width=0.7)
ax.set_title(
    f"Top 15 Community Tags  —  Top Multiplayer: {multi_name} ({multi_score}%)",
    pad=15)
ax.set_xlabel("Tag", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.tick_params(axis="x", rotation=35)

for p, color in zip(ax.patches, bar_colors):
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + max_h * 0.01,
            f"{int(p.get_height()):,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

# Arrow callout on Multiplayer bar — positioned well above count label
if "Multiplayer" in tag_counts.index:
    idx = list(tag_counts.index).index("Multiplayer")
    bar = ax.patches[idx]
    bh  = bar.get_height()
    bx  = bar.get_x() + bar.get_width() / 2
    ax.annotate(
        f"Top: {multi_name}",
        xy=(bx, bh + max_h * 0.02),
        xytext=(bx, bh + max_h * 0.14),
        ha="center", va="bottom", fontsize=7.5, color="#FFD93D",
        arrowprops=dict(arrowstyle="->", color="#FFD93D", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_tags.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_tags.png")

# -------------------------
# GRAPH 5: Review Score Category Breakdown
# -------------------------
cat_counts   = df_rev["score_cat"].value_counts().reindex(labels)
score_colors = ["#FF4757","#FF6B35","#FFD93D","#6BCB77","#4BAAFF","#C44BFF","#FF6B9D"]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_ylim(0, cat_counts.max() * 1.18)
bars = ax.bar(cat_counts.index, cat_counts.values,
              color=score_colors, edgecolor="none", width=0.65)
ax.set_title(f"Review Score Breakdown — {len(df_rev):,} Games with 100+ Reviews", pad=15)
ax.set_xlabel("Review Category", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.tick_params(axis="x", rotation=25)

for bar, val, color in zip(bars, cat_counts.values, score_colors):
    if pd.notna(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{int(val)}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "review_score_breakdown.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: review_score_breakdown.png")

# -------------------------
# GRAPH 6: Top 15 Games by Estimated Owners
# -------------------------
top_owners = df[df["Estimated_Owners"] > 0].nlargest(15, "Estimated_Owners")
bar_colors  = [PALETTE[i % len(PALETTE)] for i in range(len(top_owners))]
max_val     = top_owners["Estimated_Owners"].max() / 1_000_000

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)
bars = ax.barh(top_owners["Name"].str[:40], top_owners["Estimated_Owners"] / 1_000_000,
               color=bar_colors, edgecolor="none", height=0.6)
ax.set_title("Top 15 Games by Estimated Owners", pad=15)
ax.set_xlabel("Estimated Owners (Millions)", labelpad=10)
ax.invert_yaxis()
ax.set_xlim(0, max_val * 1.28)

for bar, val, color in zip(bars, top_owners["Estimated_Owners"], bar_colors):
    ax.text(bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val/1_000_000:.1f}M",
            va="center", ha="left", fontsize=9, fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_estimated_owners.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_estimated_owners.png")

# -------------------------
# GRAPH 7: Games Released Per Year  +  top game per year
# — label shown as a tooltip-style box above each bar instead of rotated inside
# -------------------------
year_counts = df["Year"].dropna().astype(int).value_counts().sort_index()
year_counts = year_counts[year_counts.index >= 2006]
bar_colors  = [PALETTE[i % len(PALETTE)] for i in range(len(year_counts))]
max_h       = year_counts.max()

def top_game_for_year(year):
    subset = df_rev[df_rev["Year"] == year]
    name, score, _ = best_game(subset, name_len=18)
    if name is None:
        return ""
    return f"{name}\n({score}%)"

year_top_labels = {yr: top_game_for_year(yr) for yr in year_counts.index}

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor(BG_COLOR)
# Extra top room for labels
ax.set_ylim(0, max_h * 1.55)
bars = ax.bar(year_counts.index.astype(str), year_counts.values,
              color=bar_colors, edgecolor="none", width=0.7)
ax.set_title("Top-Selling Games Released Per Year  (best-reviewed game shown above bar)", pad=15)
ax.set_xlabel("Year", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.tick_params(axis="x", rotation=40)

for bar, (yr, val), color in zip(bars, year_counts.items(), bar_colors):
    cx = bar.get_x() + bar.get_width() / 2
    # count label just above bar
    ax.text(cx, val + max_h * 0.01,
            str(val), ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=color)
    # top game label above count — only for years with enough games
    label = year_top_labels.get(yr, "")
    if label and val >= 15:
        ax.text(cx, val + max_h * 0.09,
                label,
                ha="center", va="bottom",
                fontsize=6.5, color=TEXT_COLOR,
                linespacing=1.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "releases_per_year.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: releases_per_year.png")

# -------------------------
# GRAPH 8: Discount % Distribution
# -------------------------
df_disc = df[df["Discount_Pct"] > 0].copy()

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG_COLOR)
n, bins_d, patches = ax.hist(df_disc["Discount_Pct"], bins=20, edgecolor="none")
cmap2 = LinearSegmentedColormap.from_list("disc", ["#FFD93D", "#FF6B35", "#FF4757"])
norm2 = (bins_d[:-1] - bins_d[:-1].min()) / (bins_d[:-1].max() - bins_d[:-1].min() + 1e-9)
for patch, nv in zip(patches, norm2):
    patch.set_facecolor(cmap2(nv))

ax.set_title(f"Discount % Distribution — {len(df_disc):,} Discounted Games", pad=15)
ax.set_xlabel("Discount (%)", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "discount_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: discount_distribution.png")

print("\nAll graphs generated successfully!")