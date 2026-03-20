import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import squarify

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
# LOAD & CLEAN
# -------------------------
df = pd.read_csv("dataset.csv")
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"]         = df["Release_Date"].dt.year
df_rev = df[df["Total_Reviews"] >= 100].copy()

bins_s = [0, 40, 60, 70, 80, 90, 95, 100]
labels = ["Overwhelm. Neg", "Mostly Neg", "Mixed",
          "Mostly Pos", "Very Pos", "Overwhelm. Pos", "Perfect"]
df_rev["score_cat"] = pd.cut(df_rev["Review_Score_Pct"], bins=bins_s, labels=labels)

def best_game(subset, name_len=30):
    if subset.empty:
        return None, None, None
    row  = subset.sort_values(["Review_Score_Pct", "Total_Reviews"],
                               ascending=[False, False]).iloc[0]
    name = row["Name"]
    name = (name[:name_len - 3] + "...") if len(name) > name_len else name
    return name, int(row["Review_Score_Pct"]), int(row["Total_Reviews"])

print(f"Loaded {len(df):,} games | {len(df_rev):,} with 100+ reviews")

# ======================================================
# GRAPH 1: HORIZONTAL BAR — Top 15 by Review Score
# (kept closest to original)
# ======================================================
top_games = (df_rev
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
ax.set_xlim(0, 140)
ax.invert_yaxis()

for bar, score, reviews, color in zip(bars, top_games["Review_Score_Pct"],
                                       top_games["Total_Reviews"], bar_colors):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{score}%  |  {int(reviews):,} reviews",
            va="center", ha="left", fontsize=8, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_games_reviews.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_games_reviews.png")

# ======================================================
# GRAPH 2: BOX + STRIP PLOT — Price Distribution
# ======================================================
df_paid = df[df["Price_USD"] > 0].copy()

# Price buckets for strip jitter
price_buckets = {
    "Under $5":   df_paid[df_paid["Price_USD"] < 5]["Price_USD"],
    "$5 - $15":   df_paid[(df_paid["Price_USD"] >= 5)  & (df_paid["Price_USD"] < 15)]["Price_USD"],
    "$15 - $30":  df_paid[(df_paid["Price_USD"] >= 15) & (df_paid["Price_USD"] < 30)]["Price_USD"],
    "$30 - $50":  df_paid[(df_paid["Price_USD"] >= 30) & (df_paid["Price_USD"] < 50)]["Price_USD"],
    "$50+":       df_paid[df_paid["Price_USD"] >= 50]["Price_USD"],
}

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(CARD_COLOR)

bucket_names = list(price_buckets.keys())
for i, (bucket, vals) in enumerate(price_buckets.items()):
    color = PALETTE[i % len(PALETTE)]
    if len(vals) == 0:
        continue
    # box
    bp = ax.boxplot(vals, positions=[i], widths=0.4, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.4, linewidth=0),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.5),
                    flierprops=dict(marker="o", markersize=2,
                                    markerfacecolor=color, alpha=0.3, linewidth=0))
    # strip (jittered dots)
    jitter = np.random.uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(i + jitter, vals, color=color, alpha=0.35, s=10, zorder=3)
    # count label
    ax.text(i, vals.max() + 1.5, f"n={len(vals)}", ha="center",
            fontsize=8, color=color, fontweight="bold")

ax.set_xticks(range(len(bucket_names)))
ax.set_xticklabels(bucket_names, fontsize=10)
ax.set_ylabel("Price (USD)", labelpad=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
median_p = df_paid["Price_USD"].median()
ax.set_title(
    f"Price Distribution by Bucket — Median ${median_p:.2f}  |  "
    f"{(df['Price_USD']==0).sum()} Free-to-Play not shown",
    pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: price_distribution.png")

# ======================================================
# GRAPH 3: LOLLIPOP — Top 15 Genres
# ======================================================
genre_counts = df["Primary_Genre"].value_counts().head(15)

def top_game_for_genre(genre):
    subset = df_rev[df_rev["Primary_Genre"] == genre]
    name, score, _ = best_game(subset, name_len=22)
    return f"{name}  {score}%" if name else ""

top_labels = [top_game_for_genre(g) for g in genre_counts.index]
bar_colors  = [PALETTE[i % len(PALETTE)] for i in range(len(genre_counts))]
max_h       = genre_counts.max()

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_ylim(0, max_h * 1.38)

xs = range(len(genre_counts))
for x, (genre, count), color, label in zip(xs, genre_counts.items(), bar_colors, top_labels):
    # stem line
    ax.plot([x, x], [0, count], color=color, linewidth=2.5, solid_capstyle="round", zorder=2)
    # circle head
    ax.scatter(x, count, color=color, s=120, zorder=3, edgecolors=BG_COLOR, linewidths=1.5)
    # count above circle
    ax.text(x, count + max_h * 0.012, f"{int(count):,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
    # best game label above count
    if label:
        ax.text(x, count + max_h * 0.07, label,
                ha="center", va="bottom", fontsize=6.5,
                color=TEXT_COLOR, rotation=38)

ax.set_xticks(list(xs))
ax.set_xticklabels(genre_counts.index, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.set_title("Top 15 Primary Genres  (best-reviewed game shown above)", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_genres.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_genres.png")

# ======================================================
# GRAPH 4: TREEMAP — Top 20 Community Tags
# ======================================================
tags_split = df["All_Tags"].dropna().str.split(";")
all_tags   = [t.strip() for sub in tags_split for t in sub if t.strip()]
tag_counts = pd.Series(all_tags).value_counts().head(20)

multi_mask = df_rev["All_Tags"].fillna("").str.contains("Multiplayer", case=False, regex=False)
multi_name, multi_score, _ = best_game(df_rev[multi_mask], name_len=26)

# Color: cycle palette, highlight Multiplayer in gold
tree_colors = []
for tag in tag_counts.index:
    if tag == "Multiplayer":
        tree_colors.append("#FFD93D")
    else:
        i = list(tag_counts.index).index(tag)
        tree_colors.append(PALETTE[i % len(PALETTE)])

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_axis_off()

squarify.plot(
    sizes=tag_counts.values,
    label=[f"{t}\n{v:,}" for t, v in zip(tag_counts.index, tag_counts.values)],
    color=tree_colors,
    alpha=0.88,
    ax=ax,
    text_kwargs={"fontsize": 9, "color": "white", "fontweight": "bold",
                 "wrap": True},
    pad=True,
)
ax.set_title(
    f"Top 20 Community Tags (area = count)  —  Top Multiplayer: {multi_name} ({multi_score}%)",
    pad=15, color=TEXT_COLOR)
fig.patch.set_facecolor(BG_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_tags.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_tags.png")

# ======================================================
# GRAPH 5: DIVERGING BAR — Review Score Breakdown
# ======================================================
cat_counts  = df_rev["score_cat"].value_counts().reindex(labels).fillna(0).astype(int)
score_colors = ["#FF4757","#FF6B35","#FFD93D","#6BCB77","#4BAAFF","#C44BFF","#FF6B9D"]

# Split: negative side (left) vs positive side (right) from center
neg_labels = ["Overwhelm. Neg", "Mostly Neg", "Mixed"]
pos_labels = ["Mostly Pos", "Very Pos", "Overwhelm. Pos", "Perfect"]
neg_vals   = [-cat_counts[l] for l in neg_labels]
pos_vals   = [cat_counts[l]  for l in pos_labels]

all_div_labels = neg_labels + pos_labels
all_div_vals   = neg_vals   + pos_vals
all_div_colors = ["#FF4757","#FF6B35","#FFD93D","#6BCB77","#4BAAFF","#C44BFF","#FF6B9D"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)
ys = range(len(all_div_labels))
bars = ax.barh(list(ys), all_div_vals, color=all_div_colors, edgecolor="none", height=0.65)
ax.axvline(0, color=TEXT_COLOR, linewidth=1, alpha=0.4)
ax.set_yticks(list(ys))
ax.set_yticklabels(all_div_labels, fontsize=10)
ax.set_xlabel("Number of Games  (negative side = bad reviews)", labelpad=10)
ax.set_title(f"Review Score Breakdown — {len(df_rev):,} Games  |  Negative left, Positive right", pad=15)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(int(x))}"))

for bar, val, color in zip(bars, all_div_vals, all_div_colors):
    label_x = bar.get_width() + (8 if val >= 0 else -8)
    ha = "left" if val >= 0 else "right"
    ax.text(label_x, bar.get_y() + bar.get_height() / 2,
            str(abs(val)), va="center", ha=ha, fontsize=9,
            fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "review_score_breakdown.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: review_score_breakdown.png")

# ======================================================
# GRAPH 6: BUBBLE CHART — Estimated Owners vs Review Score
# ======================================================
df_bubble = df[(df["Estimated_Owners"] > 0) & (df["Total_Reviews"] >= 100)].copy()
# Size = log of estimated owners for bubble radius
df_bubble["bubble_size"] = np.log1p(df_bubble["Estimated_Owners"]) ** 2.2

# Color by genre
genres_list = df_bubble["Primary_Genre"].unique()
genre_color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(genres_list)}
bcolors = df_bubble["Primary_Genre"].map(genre_color_map)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)

sc = ax.scatter(df_bubble["Review_Score_Pct"], df_bubble["Estimated_Owners"] / 1_000_000,
                s=df_bubble["bubble_size"], c=bcolors,
                alpha=0.55, edgecolors="none")

# Label top 10 by owners
top10 = df_bubble.nlargest(10, "Estimated_Owners")
for _, row in top10.iterrows():
    name = row["Name"][:22]
    ax.text(row["Review_Score_Pct"] + 0.4, row["Estimated_Owners"] / 1_000_000,
            name, fontsize=7, color=TEXT_COLOR, va="center", alpha=0.9)

ax.set_xlabel("Review Score (%)", labelpad=10)
ax.set_ylabel("Estimated Owners (Millions)", labelpad=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax.set_title("Estimated Owners vs Review Score  (bubble size = owner count)", pad=15)

# Genre legend — top 6 genres only
top_genres = df_bubble["Primary_Genre"].value_counts().head(6).index
legend_handles = [mpatches.Patch(color=genre_color_map[g], label=g) for g in top_genres]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          framealpha=0.3, title="Genre", title_fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_estimated_owners.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_estimated_owners.png")

# ======================================================
# GRAPH 7: AREA + LINE — Releases Per Year
# ======================================================
year_counts = df["Year"].dropna().astype(int).value_counts().sort_index()
year_counts = year_counts[year_counts.index >= 2006]
max_h       = year_counts.max()

def top_game_for_year(year):
    subset = df_rev[df_rev["Year"] == year]
    name, score, _ = best_game(subset, name_len=18)
    return f"{name}\n({score}%)" if name else ""

year_top_labels = {yr: top_game_for_year(yr) for yr in year_counts.index}

xs     = np.array(year_counts.index)
ys     = np.array(year_counts.values)
xs_str = [str(x) for x in xs]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_ylim(0, max_h * 1.6)

# Gradient area fill
cmap_area = LinearSegmentedColormap.from_list("yr", ["#4BAAFF", "#C44BFF", "#FF6B9D"])
for i in range(len(xs) - 1):
    color = cmap_area(i / max(len(xs) - 1, 1))
    ax.fill_between([xs_str[i], xs_str[i+1]], [ys[i], ys[i+1]], alpha=0.25, color=color)

# Line
ax.plot(xs_str, ys, color="#C44BFF", linewidth=2.5, zorder=4)

# Dots
dot_colors = [cmap_area(i / max(len(xs) - 1, 1)) for i in range(len(xs))]
for x, y, color in zip(xs_str, ys, dot_colors):
    ax.scatter(x, y, color=color, s=60, zorder=5, edgecolors=BG_COLOR, linewidths=1.2)

# Count + top game labels
for x, y, yr, color in zip(xs_str, ys, xs, dot_colors):
    ax.text(x, y + max_h * 0.02, str(y),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)
    label = year_top_labels.get(yr, "")
    if label and y >= 15:
        ax.text(x, y + max_h * 0.1, label,
                ha="center", va="bottom", fontsize=6.2,
                color=TEXT_COLOR, linespacing=1.3)

ax.set_xlabel("Year", labelpad=10)
ax.set_ylabel("Number of Games", labelpad=10)
ax.set_title("Top-Selling Games Released Per Year  (best-reviewed game shown above)", pad=15)
ax.tick_params(axis="x", rotation=40)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "releases_per_year.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: releases_per_year.png")

# ======================================================
# GRAPH 8: DONUT — Discount Bracket Breakdown
# ======================================================
df_disc = df[df["Discount_Pct"] > 0].copy()

disc_bins   = [0, 20, 40, 60, 80, 100]
disc_labels = ["1-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
df_disc["disc_bracket"] = pd.cut(df_disc["Discount_Pct"],
                                  bins=disc_bins, labels=disc_labels)
bracket_counts = df_disc["disc_bracket"].value_counts().reindex(disc_labels).fillna(0).astype(int)
donut_colors   = ["#FFD93D", "#FF6B35", "#FF6B9D", "#C44BFF", "#FF4757"]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

wedges, texts, autotexts = ax.pie(
    bracket_counts,
    labels=[f"{l}\n({v} games)" for l, v in zip(disc_labels, bracket_counts)],
    colors=donut_colors,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG_COLOR, linewidth=3),
    textprops=dict(color=TEXT_COLOR, fontsize=10),
)
for at, color in zip(autotexts, donut_colors):
    at.set_fontsize(9)
    at.set_fontweight("bold")
    at.set_color(BG_COLOR)

# Center text
ax.text(0, 0, f"{len(df_disc)}\ngames\ndiscounted",
        ha="center", va="center", fontsize=12,
        fontweight="bold", color=TEXT_COLOR, linespacing=1.5)

ax.set_title("Discount Bracket Breakdown  (donut = share of discounted games)", pad=15,
             color=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "discount_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: discount_distribution.png")

print("\nAll graphs generated successfully!")