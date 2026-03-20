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

# Vibrant anime-inspired palette
PALETTE = ["#FF6B9D", "#C44BFF", "#4BAAFF", "#FFD93D", "#6BCB77", "#FF6B35", "#00D9C0", "#FF4757"]
BG_COLOR   = "#0D0D1A"   # deep dark navy background
CARD_COLOR = "#161628"   # slightly lighter for axes
GRID_COLOR = "#252540"   # subtle grid lines
TEXT_COLOR = "#F0F0FF"   # near-white text

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
import html as html_lib

df = pd.read_csv("dataset.csv")

# Fix 1: Decode HTML entities in names (e.g. &#039; -> ', &amp; -> &)
df["name"] = df["name"].apply(html_lib.unescape)

# Fix 2: Convert episodes to numeric; "Unknown" becomes NaN
df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")

# Fix 3: Drop rows with no rating
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["members"] = pd.to_numeric(df["members"], errors="coerce")
df = df.dropna(subset=["rating", "members"])

# Fix 4: Filter out low-member entries to avoid unreliable ratings
#         (e.g. a 10/10 from only 13 votes skews Top 10 charts)
df_rated = df[df["members"] >= 100].copy()

print(f"Loaded {len(df):,} anime records ({len(df_rated):,} with 100+ members).")

# -------------------------
# GRAPH 1: Top 10 Anime by Rating
# -------------------------
top_rating = df_rated.sort_values("rating", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG_COLOR)

# Cycle colors across bars for a rainbow effect
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(top_rating))]
bars = ax.barh(top_rating["name"], top_rating["rating"],
               color=bar_colors, edgecolor="none", height=0.65)

ax.set_xlabel("Average Rating", labelpad=10)
ax.set_title("Top 10 Anime by Rating", pad=18)
ax.set_xlim(left=top_rating["rating"].min() - 0.3, right=10.6)
ax.invert_yaxis()

for bar, val, color in zip(bars, top_rating["rating"], bar_colors):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", fontsize=10,
            fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_rating.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_rating.png")

# -------------------------
# GRAPH 2: Members Distribution (log scale)
# -------------------------
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG_COLOR)

n, bins, patches = ax.hist(df_rated["members"], bins=60, edgecolor="none", log=True)
# Apply gradient colors across the bars
cmap = LinearSegmentedColormap.from_list("anime", ["#4BAAFF", "#C44BFF", "#FF6B9D"])
norm_vals = (bins[:-1] - bins[:-1].min()) / (bins[:-1].max() - bins[:-1].min())
for patch, nv in zip(patches, norm_vals):
    patch.set_facecolor(cmap(nv))

ax.set_title("Distribution of Members (log scale)", pad=15)
ax.set_xlabel("Members", labelpad=10)
ax.set_ylabel("Frequency (log)", labelpad=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "members_distribution.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: members_distribution.png")

# -------------------------
# GRAPH 3: Top 10 Genres
# -------------------------
genres = df_rated["genre"].dropna().str.split(", ")
all_genres = [g.strip() for sublist in genres for g in sublist if g.strip()]
genre_series = pd.Series(all_genres).value_counts().head(10)

# Find the top-rated anime for each genre
def top_anime_for_genre(genre):
    mask = df_rated["genre"].dropna().str.contains(genre, regex=False)
    subset = df_rated.loc[mask.index[mask]].sort_values("rating", ascending=False)
    if subset.empty:
        return ""
    name = subset.iloc[0]["name"]
    rating = subset.iloc[0]["rating"]
    if len(name) > 22:
        name = name[:20] + "..."
    return f"* {name} ({rating:.2f})"

top_labels = [top_anime_for_genre(g) for g in genre_series.index]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG_COLOR)

bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(genre_series))]
bars = genre_series.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="none", width=0.7)
ax.set_title("Top 10 Genres", pad=15)
ax.set_xlabel("Genre", labelpad=10)
ax.set_ylabel("Count", labelpad=10)
ax.tick_params(axis="x", rotation=30)

for p, color, label in zip(ax.patches, bar_colors, top_labels):
    # Count label at the top of bar
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=color)
    # Top anime label inside the bar
    bar_height = p.get_height()
    if bar_height > 80:
        ax.annotate(label,
                    (p.get_x() + p.get_width() / 2, bar_height * 0.5),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold",
                    rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_genres.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: top_genres.png")

# -------------------------
# GRAPH 4: Anime Type Distribution
# -------------------------
if "type" in df_rated.columns:
    type_counts = df_rated["type"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    wedges, texts, autotexts = ax.pie(
        type_counts,
        labels=type_counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(type_counts)],
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor=BG_COLOR, linewidth=3),
        textprops=dict(color=TEXT_COLOR, fontsize=11),
    )
    for at, color in zip(autotexts, PALETTE):
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color(BG_COLOR)

    ax.set_title("Anime by Type", pad=20, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "type_distribution.png"), facecolor=BG_COLOR)
    plt.close()
    print("Saved: type_distribution.png")

# -------------------------
# GRAPH 5: Rating vs Members (scatter)
# -------------------------
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_COLOR)

ax.scatter(
    df_rated["rating"],
    df_rated["members"],
    alpha=0.35,
    s=18,
    c=PALETTE[0],
    edgecolors="none",
)
ax.set_yscale("log")
ax.set_xlabel("Rating", labelpad=10)
ax.set_ylabel("Members (log scale)", labelpad=10)
ax.set_title("Rating vs Members", pad=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Trend line (log space)
valid = df_rated[["rating", "members"]].dropna()
log_members = np.log10(valid["members"])
z = np.polyfit(valid["rating"], log_members, 1)
p = np.poly1d(z)
xs = np.linspace(valid["rating"].min(), valid["rating"].max(), 200)
ax.plot(xs, 10 ** p(xs), color=PALETTE[2], linewidth=2.5,
        linestyle="--", label="Trend", alpha=0.9)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_vs_members.png"), facecolor=BG_COLOR)
plt.close()
print("Saved: rating_vs_members.png")

print("\nAll graphs generated successfully!")