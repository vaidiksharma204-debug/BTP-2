"""
BTP-2 EDA + Labeling Script
Vaidik Sharma · 22MT10063 · IIT Kharagpur · Prof. Pabita Mitra

Run from inside btp2_scraper/:
    python3 eda.py

Outputs:
    data/plots/         → all charts as PNG files
    data/btp2_v1_labeled.parquet
    data/btp2_train.parquet
    data/btp2_val.parquet
    data/btp2_test.parquet
    data/dataset_meta.json
"""
import warnings, json
warnings.filterwarnings('ignore')

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed — saves to files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path('data')
PLOTS_DIR   = DATA_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW         = DATA_DIR / 'btp2_v1.parquet'
OUT_LABELED = DATA_DIR / 'btp2_v1_labeled.parquet'
OUT_TRAIN   = DATA_DIR / 'btp2_train.parquet'
OUT_VAL     = DATA_DIR / 'btp2_val.parquet'
OUT_TEST    = DATA_DIR / 'btp2_test.parquet'

# ── Style ──────────────────────────────────────────────────────────────────
TEAL  = '#0D9488'
NAVY  = '#0F1B3C'
AMBER = '#F59E0B'
PALE  = '#9FE1CB'
sns.set_theme(style='whitegrid', font_scale=1.1)

def save(name):
    path = PLOTS_DIR / f'{name}.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'    saved → data/plots/{name}.png')

def section(title):
    print(f'\n{"═"*55}')
    print(f'  {title}')
    print(f'{"═"*55}')

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════
section('1 · Load dataset')
assert RAW.exists(), f'Dataset not found at {RAW}.\nRun: python3 run.py --seed seed_channels.csv first.'

df = pd.read_parquet(RAW)
print(f'  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'  Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB')
print(f'  Columns: {df.columns.tolist()}')

# ══════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════
section('2 · Data quality audit')

# Missing values chart
num_cols = df.select_dtypes(include='number').columns.tolist()
null_pct = df[num_cols].isnull().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(null_pct.index, null_pct.values * 100,
        color=[TEAL if v < 0.2 else AMBER if v < 0.5 else '#E11D48'
               for v in null_pct.values])
ax.set_xlabel('Missing %')
ax.set_title('Missing value rate per column')
ax.axvline(20, color='gray', linestyle='--', alpha=0.5, label='20% threshold')
ax.legend()
plt.tight_layout()
save('01_missing_values')

# Dedup + filter
before = len(df)
df = df.sort_values('scraped_date').drop_duplicates(subset='video_id', keep='last')
df = df[df['views_lifetime'].notna() & (df['views_lifetime'] > 0)]
df = df[df['title'].notna() & (df['title'].str.len() > 0)]
print(f'  Dropped {before - len(df):,} rows → {len(df):,} clean rows')

# Clip engagement outliers
q99 = df['engagement_rate'].quantile(0.99)
df['engagement_rate'] = df['engagement_rate'].clip(upper=q99)
print(f'  Engagement rate clipped at p99 = {q99:.2f}%')

# ══════════════════════════════════════════════════════════════════════════
# 3. LABELS
# ══════════════════════════════════════════════════════════════════════════
section('3 · Generate training labels')

def label_tier(group, col):
    q33, q66 = group[col].quantile([0.33, 0.66])
    return pd.cut(group[col],
                  bins=[-np.inf, q33, q66, np.inf],
                  labels=['LOW', 'MID', 'HIGH'])

df['engagement_tier'] = (
    df.groupby('sector', group_keys=False)
      .apply(lambda g: label_tier(g, 'engagement_rate'))
)
df['views_tier'] = (
    df.groupby('sector', group_keys=False)
      .apply(lambda g: label_tier(g, 'views_lifetime'))
)

print('  Engagement tier distribution:')
print(df['engagement_tier'].value_counts().to_string())
print(f'  NaN labels (no sector): {df["engagement_tier"].isna().sum()}')

# Label balance chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['engagement_tier'].value_counts()[['LOW', 'MID', 'HIGH']]
axes[0].bar(counts.index, counts.values, color=[NAVY, AMBER, TEAL])
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, f'{v:,}', ha='center')
axes[0].set_title('Engagement tier — overall')
axes[0].set_ylabel('Videos')

sector_tiers = (df.groupby(['sector', 'engagement_tier'])
                  .size().unstack(fill_value=0)[['LOW', 'MID', 'HIGH']])
sector_tiers.plot(kind='bar', stacked=True, ax=axes[1],
                  color=[NAVY, AMBER, TEAL])
axes[1].set_title('Engagement tier — per sector')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Tier')
plt.tight_layout()
save('02_label_balance')

# ══════════════════════════════════════════════════════════════════════════
# 4. SECTOR OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
section('4 · Sector & language overview')

sc = df['sector'].value_counts()
print('  Videos per sector:')
print(sc.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette = [TEAL, AMBER, NAVY, PALE, '#E11D48', '#7C3AED',
           '#F97316', '#06B6D4', '#84CC16']
sc.plot(kind='bar', ax=axes[0], color=palette[:len(sc)])
axes[0].set_title('Videos per sector')
axes[0].tick_params(axis='x', rotation=45)
axes[1].pie(sc.values, labels=sc.index, colors=palette[:len(sc)],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Sector split')
plt.tight_layout()
save('03_sector_distribution')

# ══════════════════════════════════════════════════════════════════════════
# 5. KPIs
# ══════════════════════════════════════════════════════════════════════════
section('5 · Performance KPIs (Group G)')

kpis = [('views_lifetime', 'Views'), ('likes_count', 'Likes'),
        ('comments_count', 'Comments'), ('engagement_rate', 'Engagement rate %'),
        ('likes_per_view', 'Likes / view'), ('comments_per_view', 'Comments / view')]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (col, label) in zip(axes.flat, kpis):
    data = df[col].dropna()
    use_log = col in ('views_lifetime', 'likes_count', 'comments_count')
    if use_log:
        data = np.log1p(data)
        label = f'log1p({label})'
    ax.hist(data, bins=60, color=TEAL, edgecolor='white', linewidth=0.2)
    ax.set_title(label)
    ax.axvline(data.median(), color=AMBER, linestyle='--',
               label=f'median={data.median():.2f}')
    ax.legend(fontsize=8)
plt.suptitle('Group G — KPI distributions', fontsize=14, y=1.01)
plt.tight_layout()
save('04_kpi_distributions')

# Engagement by sector
fig, ax = plt.subplots(figsize=(13, 5))
order = df.groupby('sector')['engagement_rate'].median().sort_values(ascending=False).index
sns.violinplot(data=df, x='sector', y='engagement_rate',
               order=order, inner='quartile', ax=ax, palette='Set2')
ax.set_title('Engagement rate by sector')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save('05_engagement_by_sector')

kpi_stats = df[['views_lifetime', 'likes_count', 'comments_count',
                'engagement_rate', 'likes_per_view']].describe().T
kpi_stats = kpi_stats[['mean', '50%', 'std', 'min', 'max']]
kpi_stats.columns = ['mean', 'median', 'std', 'min', 'max']
print('\n  KPI summary:')
print(kpi_stats.round(4).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 6. TEXT FEATURES
# ══════════════════════════════════════════════════════════════════════════
section('6 · Text features (Group C)')

text_feats = [
    ('title_length_chars', 'Title length'),
    ('title_word_count', 'Title words'),
    ('title_sentiment_polarity', 'Title sentiment'),
    ('title_emoji_count', 'Emoji count'),
    ('flesch_readability', 'Flesch readability'),
    ('combined_sentiment', 'Combined sentiment'),
    ('hashtag_count', 'Hashtag count'),
    ('cta_word_count', 'CTA words'),
]
text_feats = [(c, l) for c, l in text_feats if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (col, label) in zip(axes.flat, text_feats):
    data = df[col].dropna()
    ax.hist(data, bins=50, color=AMBER, edgecolor='white', linewidth=0.2)
    ax.set_title(label, fontsize=10)
    ax.axvline(data.median(), color=NAVY, linestyle='--', alpha=0.8)
plt.suptitle('Group C — Text feature distributions', fontsize=13, y=1.01)
plt.tight_layout()
save('06_text_features')

# Text vs engagement tier
sig = ['title_length_chars', 'combined_sentiment',
       'flesch_readability', 'hashtag_count']
sig = [c for c in sig if c in df.columns]
fig, axes = plt.subplots(1, len(sig), figsize=(4 * len(sig), 5))
if len(sig) == 1:
    axes = [axes]
for ax, col in zip(axes, sig):
    sub = df[[col, 'engagement_tier']].dropna()
    sns.boxplot(data=sub, x='engagement_tier', y=col,
                order=['LOW', 'MID', 'HIGH'],
                palette=[NAVY, AMBER, TEAL], ax=ax, showfliers=False)
    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
plt.suptitle('Text features vs engagement tier', fontsize=13, y=1.01)
plt.tight_layout()
save('07_text_vs_tier')

# Boolean feature rates
bool_cols = [c for c in ['title_has_number', 'title_has_question', 'cta_presence']
             if c in df.columns]
if bool_cols:
    rates = df[bool_cols].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(range(len(rates)), rates.values,
                  color=[TEAL, AMBER, NAVY][:len(rates)])
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(['Has number', 'Has question', 'Has CTA'][:len(rates)])
    for bar, v in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{v:.1f}%', ha='center')
    ax.set_title('Boolean text feature prevalence')
    ax.set_ylabel('% of videos')
    plt.tight_layout()
    save('08_boolean_features')
    print('  Boolean feature rates:')
    for col, v in rates.items():
        print(f'    {col}: {v:.1f}%')

# ══════════════════════════════════════════════════════════════════════════
# 7. THUMBNAILS
# ══════════════════════════════════════════════════════════════════════════
section('7 · Thumbnail features (Group D)')

thumb_cols = [c for c in df.columns
              if c.startswith('thumb_')
              and df[c].dtype in ['float64', 'int64']
              and c not in ('thumb_face_count', 'thumb_text_word_count',
                            'thumb_width', 'thumb_height')]
if thumb_cols:
    fig, axes = plt.subplots(1, len(thumb_cols),
                              figsize=(5 * len(thumb_cols), 4))
    if len(thumb_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, thumb_cols):
        data = df[col].dropna()
        ax.hist(data, bins=50, color=PALE, edgecolor='white')
        ax.set_title(col.replace('thumb_', '').title())
        ax.axvline(data.mean(), color=NAVY, linestyle='--',
                   label=f'mean={data.mean():.2f}')
        ax.legend(fontsize=8)
    plt.suptitle('Group D — Thumbnail features', fontsize=13)
    plt.tight_layout()
    save('09_thumbnail_features')

    # Thumbnail vs engagement
    if 'thumb_brightness' in df.columns and 'thumb_saturation' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, ['thumb_brightness', 'thumb_saturation']):
            sub = df[[col, 'engagement_tier']].dropna()
            sns.violinplot(data=sub, x='engagement_tier', y=col,
                           order=['LOW', 'MID', 'HIGH'],
                           palette=[NAVY, AMBER, TEAL],
                           inner='quartile', ax=ax)
            ax.set_title(f'{col.replace("thumb_","").title()} vs engagement tier')
        plt.tight_layout()
        save('10_thumbnail_vs_tier')

    print(f'  Avg brightness : {df["thumb_brightness"].mean():.3f}' if 'thumb_brightness' in df.columns else '')
    print(f'  Avg saturation : {df["thumb_saturation"].mean():.3f}' if 'thumb_saturation' in df.columns else '')
else:
    print('  No numeric thumbnail features found')

# ══════════════════════════════════════════════════════════════════════════
# 8. TRANSCRIPTS
# ══════════════════════════════════════════════════════════════════════════
section('8 · Transcript features (Group E)')

cap_rate = df['has_captions'].mean() * 100
src = df['transcript_source'].value_counts()
print(f'  Caption coverage : {cap_rate:.1f}%')
print('  Transcript sources:')
print(src.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].pie([cap_rate, 100 - cap_rate],
            labels=['Has captions', 'No captions'],
            colors=[TEAL, '#E5E7EB'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Caption coverage')
src.plot(kind='bar', ax=axes[1], color=palette[:len(src)])
axes[1].set_title('Transcript source')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
save('11_transcript_coverage')

if 'speech_rate_wpm' in df.columns:
    wpm = df['speech_rate_wpm'].dropna().clip(0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(wpm, bins=60, color=TEAL, edgecolor='white')
    axes[0].set_title('Speech rate distribution')
    axes[0].set_xlabel('Words per minute')
    axes[0].axvline(150, color=AMBER, linestyle='--', label='Avg conversational (150 wpm)')
    axes[0].legend()
    sub = df[['speech_rate_wpm', 'engagement_tier']].dropna()
    sub['speech_rate_wpm'] = sub['speech_rate_wpm'].clip(0, 300)
    sns.boxplot(data=sub, x='engagement_tier', y='speech_rate_wpm',
                order=['LOW', 'MID', 'HIGH'],
                palette=[NAVY, AMBER, TEAL], ax=axes[1], showfliers=False)
    axes[1].set_title('Speech rate vs engagement tier')
    plt.tight_layout()
    save('12_speech_rate')

# ══════════════════════════════════════════════════════════════════════════
# 9. TEMPORAL
# ══════════════════════════════════════════════════════════════════════════
section('9 · Temporal patterns (Group F)')

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hour vs engagement
he = df.groupby('upload_hour_utc')['engagement_rate'].mean()
axes[0, 0].bar(he.index, he.values, color=TEAL, alpha=0.8)
axes[0, 0].set_title('Avg engagement by upload hour (UTC)')
bh = int(he.idxmax())
axes[0, 0].axvline(bh, color=AMBER, linestyle='--',
                   label=f'Best: {bh:02d}:00 UTC = {(bh+5)%24:02d}:30 IST')
axes[0, 0].legend(fontsize=9)

# DOW vs engagement
de = df.groupby('upload_dow')['engagement_rate'].mean()
axes[0, 1].bar(de.index, de.values, color=AMBER, alpha=0.8)
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(dow_names)
axes[0, 1].set_title('Avg engagement by day of week')
bd = int(de.idxmax())
axes[0, 1].axvline(bd, color=NAVY, linestyle='--',
                   label=f'Best: {dow_names[bd]}')
axes[0, 1].legend(fontsize=9)

# Monthly volume
mc = df['upload_month'].value_counts().sort_index()
axes[1, 0].bar(mc.index, mc.values, color=NAVY, alpha=0.8)
axes[1, 0].set_title('Upload volume by month')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            rotation=45)

# Season vs engagement
se = df.groupby('season')['engagement_rate'].mean().sort_index()
bars = axes[1, 1].bar(se.index, se.values, color=[TEAL, AMBER, NAVY, PALE])
axes[1, 1].set_title('Avg engagement by season')
for bar, v in zip(bars, se.values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    v + 0.02, f'{v:.2f}%', ha='center', fontsize=9)

plt.suptitle('Group F — Temporal patterns', fontsize=14, y=1.01)
plt.tight_layout()
save('13_temporal_patterns')

# Heatmap
pivot = (df.groupby(['upload_dow', 'upload_hour_utc'])['engagement_rate']
           .mean().unstack(fill_value=0))
pivot.index = dow_names[:len(pivot)]
fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Avg engagement rate (%)'})
ax.set_title('Upload timing heatmap — Day × Hour (UTC)')
plt.tight_layout()
save('14_timing_heatmap')

print(f'  Best upload day  : {dow_names[bd]}')
print(f'  Best upload hour : {bh:02d}:00 UTC  =  {(bh+5)%24:02d}:30 IST')

# ══════════════════════════════════════════════════════════════════════════
# 10. HASHTAGS
# ══════════════════════════════════════════════════════════════════════════
section('10 · Hashtag analysis (Group H)')

all_tags = [t for tags in df['hashtags_normalized'].dropna() for t in tags]
tc = Counter(all_tags)
top30 = pd.Series(dict(tc.most_common(30)))

print(f'  Unique hashtags  : {len(tc):,}')
print(f'  Total occurrences: {sum(tc.values()):,}')
print(f'  Avg per video    : {df["hashtag_count"].mean():.1f}')
print(f'  Videos with ≥1   : {(df["hashtag_count"] > 0).mean()*100:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top30.plot(kind='barh', ax=axes[0], color=TEAL)
axes[0].invert_yaxis()
axes[0].set_title('Top 30 hashtags')
hc = df['hashtag_count'].dropna()
axes[1].hist(hc[hc <= 30], bins=30, color=AMBER, edgecolor='white')
axes[1].set_title('Hashtags per video')
axes[1].axvline(hc.median(), color=NAVY, linestyle='--',
                label=f'median={hc.median():.0f}')
axes[1].legend()
plt.tight_layout()
save('15_hashtag_analysis')

print('\n  Top 10 hashtags per sector:')
for sec in sorted(df['sector'].dropna().unique()):
    tags = [t for tl in df[df['sector'] == sec]['hashtags_normalized'].dropna()
            for t in tl]
    top = [t for t, _ in Counter(tags).most_common(10)]
    print(f'    {sec:<15}: {top}')

# ══════════════════════════════════════════════════════════════════════════
# 11. CORRELATION
# ══════════════════════════════════════════════════════════════════════════
section('11 · Feature correlation with engagement')

feat_cols = ['title_length_chars', 'title_word_count', 'title_sentiment_polarity',
             'title_emoji_count', 'title_caps_ratio', 'description_length_chars',
             'description_url_count', 'combined_sentiment', 'flesch_readability',
             'cta_word_count', 'hashtag_count', 'thumb_brightness',
             'thumb_contrast', 'thumb_saturation', 'transcript_word_count',
             'speech_rate_wpm', 'upload_hour_utc', 'upload_dow', 'upload_month',
             'duration_seconds', 'channel_subscribers', 'channel_age_days']
feat_cols = [c for c in feat_cols if c in df.columns]

corr = df[feat_cols + ['engagement_rate']].corr()['engagement_rate'].drop('engagement_rate')
cs = corr.abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(cs.index, [corr[c] for c in cs.index],
        color=[TEAL if corr[c] > 0 else '#E11D48' for c in cs.index])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature correlation with engagement_rate (Pearson)')
ax.invert_yaxis()
plt.tight_layout()
save('16_feature_correlation')

print('  Top 10 positively correlated:')
print(corr.sort_values(ascending=False).head(10).round(3).to_string())
print('  Top 10 negatively correlated:')
print(corr.sort_values().head(10).round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 12. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
section('12 · Train / Val / Test split')

df_labeled = df[df['engagement_tier'].notna()].copy()
print(f'  Labeled rows: {len(df_labeled):,} / {len(df):,}')

df_labeled['strat_key'] = (df_labeled['sector'].astype(str) + '_' +
                           df_labeled['engagement_tier'].astype(str))

df_train, df_temp = train_test_split(
    df_labeled, test_size=0.30, random_state=42,
    stratify=df_labeled['strat_key']
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42,
    stratify=df_temp['strat_key']
)
for s in (df_train, df_val, df_test):
    s.drop(columns='strat_key', inplace=True)

print(f'  Train : {len(df_train):,}  ({len(df_train)/len(df_labeled)*100:.0f}%)')
print(f'  Val   : {len(df_val):,}   ({len(df_val)/len(df_labeled)*100:.0f}%)')
print(f'  Test  : {len(df_test):,}   ({len(df_test)/len(df_labeled)*100:.0f}%)')

# Split balance chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, split) in zip(axes, [('Train', df_train),
                                      ('Val', df_val),
                                      ('Test', df_test)]):
    counts = split['engagement_tier'].value_counts()[['LOW', 'MID', 'HIGH']]
    ax.bar(counts.index, counts.values, color=[NAVY, AMBER, TEAL])
    ax.set_title(f'{name} ({len(split):,} rows)')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f'{v/len(split)*100:.1f}%', ha='center', fontsize=9)
plt.suptitle('Label distribution across splits', fontsize=13)
plt.tight_layout()
save('17_split_balance')

# ══════════════════════════════════════════════════════════════════════════
# 13. SAVE
# ══════════════════════════════════════════════════════════════════════════
section('13 · Save all outputs')

df_labeled.to_parquet(OUT_LABELED, engine='pyarrow', compression='snappy', index=False)
df_train.to_parquet(OUT_TRAIN,   engine='pyarrow', compression='snappy', index=False)
df_val.to_parquet(OUT_VAL,     engine='pyarrow', compression='snappy', index=False)
df_test.to_parquet(OUT_TEST,    engine='pyarrow', compression='snappy', index=False)

meta = {
    'total_videos'   : int(len(df)),
    'labeled_videos' : int(len(df_labeled)),
    'unique_channels': int(df['channel_id'].nunique()),
    'sectors'        : df['sector'].value_counts().to_dict(),
    'label_dist'     : df_labeled['engagement_tier'].value_counts().to_dict(),
    'caption_pct'    : float(df['has_captions'].mean() * 100),
    'median_er'      : float(df['engagement_rate'].median()),
    'schema_cols'    : int(df.shape[1]),
    'train_rows'     : int(len(df_train)),
    'val_rows'       : int(len(df_val)),
    'test_rows'      : int(len(df_test)),
}
(DATA_DIR / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f'\n{"═"*55}')
print('  EDA COMPLETE')
print(f'{"═"*55}')
print(f'  Videos           : {len(df):,}')
print(f'  Unique channels  : {df["channel_id"].nunique():,}')
print(f'  Schema columns   : {df.shape[1]}')
print(f'  Caption coverage : {df["has_captions"].mean()*100:.1f}%')
print(f'  Median eng. rate : {df["engagement_rate"].median():.2f}%')
print(f'  Sectors          : {", ".join(sorted(df["sector"].dropna().unique()))}')
print(f'  Best upload day  : {dow_names[bd]}')
print(f'  Best upload hour : {bh:02d}:00 UTC = {(bh+5)%24:02d}:30 IST')
print(f'  Train/Val/Test   : {len(df_train):,} / {len(df_val):,} / {len(df_test):,}')
print(f'\n  Charts saved  → data/plots/  ({len(list(PLOTS_DIR.glob("*.png")))} files)')
print(f'  Labeled data  → {OUT_LABELED}')
print(f'  Train split   → {OUT_TRAIN}')
print(f'  Val split     → {OUT_VAL}')
print(f'  Test split    → {OUT_TEST}')
print(f'  Metadata      → data/dataset_meta.json')
print(f'{"═"*55}')
print('\n  Next step: python3 train_roberta.py')
