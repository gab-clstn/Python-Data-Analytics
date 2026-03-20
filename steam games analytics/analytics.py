import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

OUT = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUT, exist_ok=True)

PAL  = ["#FF6B9D","#C44BFF","#4BAAFF","#FFD93D","#6BCB77",
        "#FF6B35","#00D9C0","#FF4757","#A29BFE","#FD79A8"]
BG, CARD, GRID, TXT = "#0D0D1A", "#161628", "#252540", "#F0F0FF"

plt.rcParams.update({
    "figure.dpi":150, "figure.facecolor":BG, "axes.facecolor":CARD,
    "axes.grid":True, "grid.color":GRID, "grid.linewidth":1.0,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.spines.left":False, "axes.spines.bottom":False,
    "font.family":"DejaVu Sans", "axes.titlesize":13,
    "axes.titleweight":"bold", "axes.titlecolor":TXT,
    "axes.labelcolor":TXT, "xtick.color":TXT, "ytick.color":TXT,
    "text.color":TXT, "legend.facecolor":CARD, "legend.edgecolor":GRID,
})

def save(name): plt.tight_layout(); plt.savefig(f"{OUT}/{name}", facecolor=BG); plt.close(); print(f"Saved: {name}")
def colors(n): return [PAL[i % len(PAL)] for i in range(n)]
def best(sub, n=30):
    if sub.empty: return None, None, None
    r = sub.sort_values(["Review_Score_Pct","Total_Reviews"], ascending=False).iloc[0]
    nm = r["Name"]; nm = (nm[:n-3]+"...") if len(nm)>n else nm
    return nm, int(r["Review_Score_Pct"]), int(r["Total_Reviews"])

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv")
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"] = df["Release_Date"].dt.year
df_rev = df[df["Total_Reviews"] >= 100].copy()
SCORE_LABELS = ["Overwhelm. Neg","Mostly Neg","Mixed","Mostly Pos","Very Pos","Overwhelm. Pos","Perfect"]
df_rev["score_cat"] = pd.cut(df_rev["Review_Score_Pct"], bins=[0,40,60,70,80,90,95,100], labels=SCORE_LABELS)
print(f"Loaded {len(df):,} games | {len(df_rev):,} with 100+ reviews")

# ── 1. Top 15 Games ───────────────────────────────────────────────────────────
top = df_rev.sort_values(["Review_Score_Pct","Total_Reviews"], ascending=False).head(15)
bn, bs, _ = best(df_rev)
fig, ax = plt.subplots(figsize=(14,8))
bars = ax.barh(top["Name"].str[:40], top["Review_Score_Pct"], color=colors(15), edgecolor="none", height=0.6)
[ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, f"{s}%  |  {int(r):,} reviews",
         va="center", ha="left", fontsize=8, color=c)
 for b,s,r,c in zip(bars, top["Review_Score_Pct"], top["Total_Reviews"], colors(15))]
ax.set(title=f"Top 15 Games by Positive Review %  —  Best: {bn} ({bs}%)", xlabel="Positive Review %", xlim=(0,140))
ax.invert_yaxis()
save("top_games_reviews.png")

# ── 2. Price Distribution ─────────────────────────────────────────────────────
df_paid = df[df["Price_USD"]>0].copy()
df_paid["Bucket"] = pd.cut(df_paid["Price_USD"], bins=[0,5,15,30,50,df_paid["Price_USD"].max()+1],
                            labels=["Under $5","$5-$15","$15-$30","$30-$50","$50+"])
fig, ax = plt.subplots(figsize=(13,6))
sns.violinplot(data=df_paid, x="Bucket", y="Price_USD", palette=PAL[:5], inner=None, linewidth=0, alpha=0.45, ax=ax)
sns.stripplot(data=df_paid, x="Bucket", y="Price_USD", palette=PAL[:5], size=2.5, alpha=0.3, jitter=True, ax=ax)
for i, bkt in enumerate(["Under $5","$5-$15","$15-$30","$30-$50","$50+"]):
    v = df_paid[df_paid["Bucket"]==bkt]["Price_USD"]
    if not v.empty:
        ax.hlines(v.median(), i-.2, i+.2, colors="white", linewidths=2, zorder=5)
        ax.text(i, v.max()+1.5, f"n={len(v)}", ha="center", fontsize=8, color=PAL[i], fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}"))
ax.set(title=f"Price Distribution — Median ${df_paid['Price_USD'].median():.2f}  |  {(df['Price_USD']==0).sum()} F2P not shown",
       xlabel="Price Bucket", ylabel="Price (USD)")
save("price_distribution.png")

# ── 3. Top 15 Genres (lollipop) ───────────────────────────────────────────────
gc = df["Primary_Genre"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(14,7))
ax.set_ylim(0, gc.max()*1.38)
for x,(g,c),col in zip(range(len(gc)), gc.items(), colors(15)):
    ax.plot([x,x],[0,c], color=col, linewidth=2.5, solid_capstyle="round", zorder=2)
    ax.scatter(x, c, color=col, s=120, zorder=3, edgecolors=BG, linewidths=1.5)
    ax.text(x, c+gc.max()*.012, f"{c:,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
    nm, sc, _ = best(df_rev[df_rev["Primary_Genre"]==g], 22)
    if nm: ax.text(x, c+gc.max()*.07, f"{nm}  {sc}%", ha="center", va="bottom", fontsize=6.5, color=TXT, rotation=38)
ax.set_xticks(range(len(gc))); ax.set_xticklabels(gc.index, rotation=35, ha="right", fontsize=10)
ax.set(ylabel="Number of Games", title="Top 15 Primary Genres  (best-reviewed game shown above)")
save("top_genres.png")

# ── 4. Top 20 Tags (horizontal bar) ──────────────────────────────────────────
all_tags = pd.Series([t.strip() for sub in df["All_Tags"].dropna().str.split(";") for t in sub if t.strip()])
tc = all_tags.value_counts().head(20)
mn, ms, _ = best(df_rev[df_rev["All_Tags"].fillna("").str.contains("Multiplayer", case=False)], 26)
tcols = ["#FFD93D" if t=="Multiplayer" else PAL[i%len(PAL)] for i,t in enumerate(tc.index)]
fig, ax = plt.subplots(figsize=(14,9))
bars = ax.barh(tc.index[::-1], tc.values[::-1], color=tcols[::-1], edgecolor="none", height=0.65)
[ax.text(b.get_width()+tc.max()*.005, b.get_y()+b.get_height()/2, f"{v:,}",
         va="center", ha="left", fontsize=8, fontweight="bold", color=c)
 for b,v,c in zip(bars, tc.values[::-1], tcols[::-1])]
ax.set(xlabel="Number of Games", xlim=(0, tc.max()*1.15),
       title=f"Top 20 Community Tags  —  Top Multiplayer: {mn} ({ms}%)")
save("top_tags.png")

# ── 5. Review Score Breakdown (diverging) ────────────────────────────────────
cc = df_rev["score_cat"].value_counts().reindex(SCORE_LABELS).fillna(0).astype(int)
div_labels = ["Overwhelm. Neg","Mostly Neg","Mixed","Mostly Pos","Very Pos","Overwhelm. Pos","Perfect"]
div_vals   = [-cc[l] for l in div_labels[:3]] + [cc[l] for l in div_labels[3:]]
div_cols   = ["#FF4757","#FF6B35","#FFD93D","#6BCB77","#4BAAFF","#C44BFF","#FF6B9D"]
fig, ax = plt.subplots(figsize=(13,6))
bars = ax.barh(range(7), div_vals, color=div_cols, edgecolor="none", height=0.65)
ax.axvline(0, color=TXT, linewidth=1, alpha=0.4)
ax.set_yticks(range(7)); ax.set_yticklabels(div_labels, fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{abs(int(x))}"))
[ax.text(b.get_width()+(8 if v>=0 else -8), b.get_y()+b.get_height()/2, str(abs(v)),
         va="center", ha="left" if v>=0 else "right", fontsize=9, fontweight="bold", color=c)
 for b,v,c in zip(bars, div_vals, div_cols)]
ax.set(xlabel="Number of Games  (negative = bad reviews)",
       title=f"Review Score Breakdown — {len(df_rev):,} Games  |  Negative left, Positive right")
save("review_score_breakdown.png")

# ── 6. Owners vs Review Score (bubble) ───────────────────────────────────────
db = df[(df["Estimated_Owners"]>0)&(df["Total_Reviews"]>=100)].copy()
db["sz"] = np.log1p(db["Estimated_Owners"])**2.2
gcmap = {g: PAL[i%len(PAL)] for i,g in enumerate(db["Primary_Genre"].unique())}
fig, ax = plt.subplots(figsize=(14,8))
ax.scatter(db["Review_Score_Pct"], db["Estimated_Owners"]/1e6, s=db["sz"],
           c=db["Primary_Genre"].map(gcmap), alpha=0.55, edgecolors="none")
[ax.text(r["Review_Score_Pct"]+.4, r["Estimated_Owners"]/1e6, r["Name"][:22], fontsize=7, color=TXT, alpha=.9)
 for _,r in db.nlargest(10,"Estimated_Owners").iterrows()]
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))
ax.legend(handles=[mpatches.Patch(color=gcmap[g], label=g) for g in db["Primary_Genre"].value_counts().head(6).index],
          loc="upper left", fontsize=8, framealpha=0.3, title="Genre", title_fontsize=8)
ax.set(xlabel="Review Score (%)", ylabel="Estimated Owners (Millions)",
       title="Estimated Owners vs Review Score  (bubble size = owner count)")
save("top_estimated_owners.png")

# ── 7. Releases Per Year ──────────────────────────────────────────────────────
yc = df["Year"].dropna().astype(int).value_counts().sort_index()
yc = yc[yc.index>=2006]
xs, ys = [str(x) for x in yc.index], yc.values
cmap = LinearSegmentedColormap.from_list("yr", ["#4BAAFF","#C44BFF","#FF6B9D"])
dcols = [cmap(i/max(len(xs)-1,1)) for i in range(len(xs))]
fig, ax = plt.subplots(figsize=(16,7))
ax.set_ylim(0, yc.max()*1.6)
[ax.fill_between([xs[i],xs[i+1]],[ys[i],ys[i+1]], alpha=.25, color=cmap(i/max(len(xs)-1,1))) for i in range(len(xs)-1)]
ax.plot(xs, ys, color="#C44BFF", linewidth=2.5, zorder=4)
for x,y,yr,c in zip(xs,ys,yc.index,dcols):
    ax.scatter(x,y,color=c,s=60,zorder=5,edgecolors=BG,linewidths=1.2)
    ax.text(x, y+yc.max()*.02, str(y), ha="center", va="bottom", fontsize=8, fontweight="bold", color=c)
    nm, sc, _ = best(df_rev[df_rev["Year"]==yr], 18)
    if nm and y>=15: ax.text(x, y+yc.max()*.1, f"{nm}\n({sc}%)", ha="center", va="bottom", fontsize=6.2, color=TXT, linespacing=1.3)
ax.set(xlabel="Year", ylabel="Number of Games", title="Games Released Per Year  (best-reviewed shown above)")
ax.tick_params(axis="x", rotation=40)
save("releases_per_year.png")

# ── 8. Discount Donut ─────────────────────────────────────────────────────────
dd = df[df["Discount_Pct"]>0].copy()
dd["bracket"] = pd.cut(dd["Discount_Pct"], bins=[0,20,40,60,80,100],
                        labels=["1-20%","21-40%","41-60%","61-80%","81-100%"])
bc = dd["bracket"].value_counts().reindex(["1-20%","21-40%","41-60%","61-80%","81-100%"]).fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(10,8)); ax.set_facecolor(BG)
_, _, auts = ax.pie(bc, labels=[f"{l}\n({v} games)" for l,v in bc.items()],
                    colors=["#FFD93D","#FF6B35","#FF6B9D","#C44BFF","#FF4757"],
                    autopct="%1.1f%%", startangle=140, pctdistance=0.75,
                    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3),
                    textprops=dict(color=TXT, fontsize=10))
[a.set(fontsize=9, fontweight="bold", color=BG) for a in auts]
ax.text(0,0,f"{len(dd)}\ngames\ndiscounted", ha="center", va="center", fontsize=12, fontweight="bold", color=TXT, linespacing=1.5)
ax.set_title("Discount Bracket Breakdown", pad=15, color=TXT)
save("discount_distribution.png")

print("\nAll graphs generated successfully!")