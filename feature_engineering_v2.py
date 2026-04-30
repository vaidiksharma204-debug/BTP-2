"""
feature_engineering_v2.py
──────────────────────────────────────────────────────────────────────────
Adds four improvements to the labeled dataset before model training:

  1. Location-aware upload timing  (upload_hour_local, is_peak_local)
  2. Optimized hashtag features    (hashtag_optimal, hashtag_quality_score)
  3. Transcript readiness flag     (group_e_available for Phase 2)
  4. Thumbnail completeness check  (face_count status, coverage report)

Run from inside btp2_scraper/:
    python3 feature_engineering_v2.py

Input:  data/btp2_v1_labeled.parquet
Output: data/btp2_v2_features.parquet   ← use this for all model training
"""
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
IN_FILE  = DATA_DIR / 'btp2_v1_labeled.parquet'
OUT_FILE = DATA_DIR / 'btp2_v2_features.parquet'

assert IN_FILE.exists(), f'Run eda.py first — {IN_FILE} not found'

print('Loading dataset...')
df = pd.read_parquet(IN_FILE)
print(f'  {len(df):,} rows × {df.shape[1]} columns')

# ══════════════════════════════════════════════════════════════════════════
# 1. LOCATION-AWARE UPLOAD TIMING
# ══════════════════════════════════════════════════════════════════════════
print('\n[1] Location-aware upload timing...')

# UTC offset (hours) by country code — covers ~95% of YouTube traffic
# Source: standard timezone data, one representative offset per country
COUNTRY_UTC_OFFSET = {
    # Asia-Pacific
    'IN':  5.5,   # India IST
    'PK':  5.0,   # Pakistan PKT
    'BD':  6.0,   # Bangladesh BST
    'LK':  5.5,   # Sri Lanka
    'NP':  5.75,  # Nepal
    'JP':  9.0,   # Japan JST
    'KR':  9.0,   # South Korea KST
    'CN':  8.0,   # China CST
    'HK':  8.0,   # Hong Kong HKT
    'TW':  8.0,   # Taiwan CST
    'SG':  8.0,   # Singapore SGT
    'MY':  8.0,   # Malaysia MYT
    'ID':  7.0,   # Indonesia WIB
    'TH':  7.0,   # Thailand ICT
    'VN':  7.0,   # Vietnam ICT
    'PH':  8.0,   # Philippines PHT
    'AU':  10.0,  # Australia AEST (east coast)
    'NZ':  12.0,  # New Zealand NZST
    # Europe
    'GB':  0.0,   # UK GMT (approximation)
    'DE':  1.0,   # Germany CET
    'FR':  1.0,   # France CET
    'ES':  1.0,   # Spain CET
    'IT':  1.0,   # Italy CET
    'NL':  1.0,   # Netherlands CET
    'SE':  1.0,   # Sweden CET
    'NO':  1.0,   # Norway CET
    'DK':  1.0,   # Denmark CET
    'PL':  1.0,   # Poland CET
    'RU':  3.0,   # Russia MSK
    'TR':  3.0,   # Turkey TRT
    'UA':  2.0,   # Ukraine EET
    # Americas
    'US': -5.0,   # USA EST (dominant timezone)
    'CA': -5.0,   # Canada EST
    'MX': -6.0,   # Mexico CST
    'BR': -3.0,   # Brazil BRT
    'AR': -3.0,   # Argentina ART
    'CO': -5.0,   # Colombia COT
    'CL': -4.0,   # Chile CLT
    'PE': -5.0,   # Peru PET
    # Middle East / Africa
    'AE':  4.0,   # UAE GST
    'SA':  3.0,   # Saudi Arabia AST
    'EG':  2.0,   # Egypt EET
    'ZA':  2.0,   # South Africa SAST
    'NG':  1.0,   # Nigeria WAT
    'KE':  3.0,   # Kenya EAT
    'GH':  0.0,   # Ghana GMT
}

# Peak local hours by region type (when most of the audience is online)
# Based on YouTube internal research and social media studies
PEAK_HOURS_LOCAL = set(range(14, 22))   # 2 PM – 10 PM local time

def get_utc_offset(country_code):
    if pd.isna(country_code):
        return 0.0   # assume UTC if unknown
    return COUNTRY_UTC_OFFSET.get(str(country_code).upper(), 0.0)

def utc_to_local_hour(utc_hour, offset):
    if pd.isna(utc_hour):
        return np.nan
    return int((utc_hour + offset) % 24)

# Apply
df['channel_utc_offset']  = df['channel_country'].apply(get_utc_offset)
df['upload_hour_local']   = df.apply(
    lambda r: utc_to_local_hour(r['upload_hour_utc'], r['channel_utc_offset']),
    axis=1
)
df['is_peak_local']       = df['upload_hour_local'].apply(
    lambda h: bool(h in PEAK_HOURS_LOCAL) if not pd.isna(h) else False
)

# Local hour bucket (morning / afternoon / evening / night)
def hour_bucket(h):
    if pd.isna(h): return 'unknown'
    h = int(h)
    if 5 <= h < 12:  return 'morning'
    if 12 <= h < 17: return 'afternoon'
    if 17 <= h < 22: return 'evening'
    return 'night'

df['upload_time_bucket'] = df['upload_hour_local'].apply(hour_bucket)

# Encode bucket as ordinal (night=0, morning=1, afternoon=2, evening=3)
bucket_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3, 'unknown': -1}
df['upload_time_bucket_enc'] = df['upload_time_bucket'].map(bucket_map)

peak_rate = df['is_peak_local'].mean() * 100
print(f'  channel_utc_offset  → derived from channel_country')
print(f'  upload_hour_local   → UTC + country offset')
print(f'  is_peak_local       → 14:00–22:00 local  ({peak_rate:.1f}% of videos)')
print(f'  upload_time_bucket  → morning/afternoon/evening/night')

# Compare: UTC hour engagement vs local hour engagement
utc_eng  = df.groupby('upload_hour_utc')['engagement_rate'].mean()
local_eng = df.groupby('upload_hour_local')['engagement_rate'].mean()
print(f'\n  UTC   best hour : {utc_eng.idxmax():02.0f}:00  (avg eng={utc_eng.max():.3f}%)')
print(f'  Local best hour : {local_eng.idxmax():02.0f}:00  (avg eng={local_eng.max():.3f}%)')
print(f'  → Local hour is {"more" if local_eng.std() > utc_eng.std() else "less"} '
      f'discriminative (std={local_eng.std():.4f} vs {utc_eng.std():.4f})')

# ══════════════════════════════════════════════════════════════════════════
# 2. HASHTAG FEATURE OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════
print('\n[2] Optimizing hashtag features...')

# Raw correlation is negative because spammers use 20+ hashtags.
# But the REAL question is: what count range maximizes engagement?
# Compute mean engagement by hashtag count bucket.
hc_eng = (df[df['hashtag_count'] <= 30]
            .groupby('hashtag_count')['engagement_rate']
            .agg(['mean', 'count'])
            .reset_index())
hc_eng.columns = ['count', 'mean_er', 'n_videos']

# Find the optimal range (top 3 counts by mean engagement, min 50 videos each)
eligible = hc_eng[hc_eng['n_videos'] >= 50].sort_values('mean_er', ascending=False)
optimal_counts = set(eligible.head(5)['count'].tolist())

print(f'\n  Hashtag count vs engagement rate:')
print(f'  {"Count":<8} {"Mean ER":>8} {"Videos":>8}')
print(f'  {"─"*28}')
for _, row in hc_eng[hc_eng['n_videos'] >= 30].iterrows():
    marker = ' ← OPTIMAL' if row['count'] in optimal_counts else ''
    print(f'  {int(row["count"]):<8} {row["mean_er"]:>8.3f}% {int(row["n_videos"]):>8}{marker}')

# Determine optimal range
if optimal_counts:
    opt_min = min(optimal_counts)
    opt_max = max(optimal_counts)
else:
    opt_min, opt_max = 1, 5

print(f'\n  → Optimal hashtag range: {opt_min}–{opt_max}')

# New features
df['hashtag_optimal']   = df['hashtag_count'].apply(
    lambda n: bool(opt_min <= n <= opt_max) if not pd.isna(n) else False
)
df['hashtag_zero']      = (df['hashtag_count'] == 0)
df['hashtag_spam']      = (df['hashtag_count'] > 10)

# Hashtag quality score:
# Uses corpus frequency — popular hashtags score higher, obscure ones lower.
# Rationale: using well-known hashtags means the video appears in established
# search clusters with existing audience demand.
from collections import Counter
all_tags = [t for tags in df['hashtags_normalized'].dropna() for t in tags]
tag_freq = Counter(all_tags)
total_tags = max(sum(tag_freq.values()), 1)

def hashtag_quality_score(tags, tag_freq, total):
    """
    Score = mean log-frequency of a video's hashtags.
    High score = uses popular, well-established hashtags.
    Low score  = uses rare/obscure hashtags unlikely to drive discovery.
    Returns 0.0 if no hashtags.
    """
    if not tags or len(tags) == 0:
        return 0.0
    scores = [np.log1p(tag_freq.get(t, 0)) for t in tags]
    return float(np.mean(scores))

df['hashtag_quality_score'] = df['hashtags_normalized'].apply(
    lambda tags: hashtag_quality_score(tags if isinstance(tags, list) else [], tag_freq, total_tags)
)

zero_pct    = df['hashtag_zero'].mean() * 100
optimal_pct = df['hashtag_optimal'].mean() * 100
spam_pct    = df['hashtag_spam'].mean() * 100

print(f'\n  hashtag_zero        : {zero_pct:.1f}% of videos have NO hashtags')
print(f'  hashtag_optimal     : {optimal_pct:.1f}% are in optimal range ({opt_min}–{opt_max})')
print(f'  hashtag_spam        : {spam_pct:.1f}% have >10 hashtags (spam territory)')
print(f'  hashtag_quality_score: mean={df["hashtag_quality_score"].mean():.3f}  '
      f'std={df["hashtag_quality_score"].std():.3f}')

# Verify: optimal vs zero vs spam engagement comparison
for label, mask in [('Zero hashtags', df['hashtag_zero']),
                    (f'Optimal ({opt_min}–{opt_max})', df['hashtag_optimal']),
                    ('Spam (>10)', df['hashtag_spam'])]:
    subset = df[mask]['engagement_rate']
    if len(subset) > 10:
        print(f'  {label:<22}: mean={subset.mean():.3f}%  median={subset.median():.3f}%')

# ══════════════════════════════════════════════════════════════════════════
# 3. GROUP E — TRANSCRIPT READINESS
# ══════════════════════════════════════════════════════════════════════════
print('\n[3] Group E — transcript readiness check...')

transcript_available = df['has_captions'].sum()
total = len(df)
pct = transcript_available / total * 100

print(f'  Videos with captions  : {transcript_available:,} / {total:,} ({pct:.1f}%)')
print(f'  Transcript text null  : {df["transcript_text"].isna().sum():,}')
print(f'  speech_rate_wpm null  : {df["speech_rate_wpm"].isna().sum():,}')

# Flag rows where Group E is usable
df['group_e_available'] = df['has_captions'].fillna(False).astype(bool)

print(f'\n  Group E status for each training phase:')
print(f'  Phase 1 (XGBoost/RoBERTa) → EXCLUDE Group E entirely')
print(f'    speech_rate_wpm, transcript_word_count, transcript_qm_count')
print(f'    all excluded — {100-pct:.1f}% null makes them noise not signal')
print(f'\n  Phase 2 (Multimodal)     → FIX transcript extractor first')
print(f'    Then run: python3 retranscribe.py  (retries all 20,308 videos)')
print(f'    Expected coverage after fix: ~70–80% for English videos')
print(f'\n  How to fix transcripts (root cause: youtube-transcript-api v1.x)')
print(f'    The _fetch_raw_v1 function needs cookies for non-public captions.')
print(f'    Fix: python3 fix_transcripts.py  (provided separately)')

# ══════════════════════════════════════════════════════════════════════════
# 4. THUMBNAIL FEATURE COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════
print('\n[4] Thumbnail feature completeness check...')

thumb_cols = {
    'thumb_brightness'      : 'Basic visual  (computed during harvest)',
    'thumb_contrast'        : 'Basic visual  (computed during harvest)',
    'thumb_saturation'      : 'Basic visual  (computed during harvest)',
    'thumb_dominant_colors' : 'Color palette (computed during harvest)',
    'thumb_face_count'      : 'Face detection (DEFERRED — needs mtcnn)',
    'thumb_has_text'        : 'OCR text flag  (DEFERRED — needs easyocr)',
    'thumb_text_word_count' : 'OCR word count (DEFERRED — needs easyocr)',
}

print(f'\n  {"Column":<28} {"Coverage":>10}  {"Status"}')
print(f'  {"─"*70}')
for col, note in thumb_cols.items():
    if col in df.columns:
        null_pct = df[col].isna().mean() * 100
        coverage = 100 - null_pct
        status = '✅ Ready' if coverage > 90 else '⚠️ Partial' if coverage > 20 else '❌ Empty'
        print(f'  {col:<28} {coverage:>9.1f}%  {status}  {note}')
    else:
        print(f'  {col:<28} {"N/A":>10}  ❌ Not in schema  {note}')

# Face count + OCR are NOT populated because mtcnn/easyocr weren't installed
face_null = df['thumb_face_count'].isna().mean() * 100 if 'thumb_face_count' in df.columns else 100
ocr_null  = df['thumb_has_text'].isna().mean() * 100 if 'thumb_has_text' in df.columns else 100

print(f'\n  thumb_face_count null: {face_null:.1f}%')
print(f'  thumb_has_text null  : {ocr_null:.1f}%')

if face_null > 90:
    print(f'\n  To populate face_count and OCR features for Phase 2:')
    print(f'    pip3 install mtcnn tensorflow easyocr')
    print(f'    python3 extract_thumb_features.py')
    print(f'    (processes all {len(df):,} saved thumbnails in data/thumbnails/)')
    print(f'    Estimated time: ~2 hours on CPU, ~20 min on Apple Silicon')

# Check if thumbnails are actually on disk
thumb_dir = DATA_DIR / 'thumbnails'
if thumb_dir.exists():
    n_thumbs = len(list(thumb_dir.glob('*.jpg')))
    print(f'\n  Thumbnails on disk: {n_thumbs:,} JPEGs in data/thumbnails/')
    coverage = n_thumbs / len(df) * 100
    print(f'  Coverage: {coverage:.1f}% of dataset rows have a saved thumbnail')
else:
    print('\n  data/thumbnails/ not found')

# ══════════════════════════════════════════════════════════════════════════
# 5. FINAL FEATURE LIST FOR MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════
print('\n\n' + '═'*60)
print('  FINAL FEATURE SETS FOR EACH MODEL')
print('═'*60)

FEATURES_XGBOOST = [
    # Group B — Channel context
    'channel_subscribers', 'channel_age_days', 'channel_total_videos',
    # Group C — Text (engineered)
    'title_length_chars', 'title_word_count', 'title_sentiment_polarity',
    'title_subjectivity', 'title_has_number', 'title_has_question',
    'title_caps_ratio', 'title_emoji_count',
    'description_length_chars', 'description_word_count',
    'description_url_count', 'combined_sentiment',
    'flesch_readability', 'cta_word_count', 'cta_presence',
    # Group C — Hashtag (OPTIMIZED)
    'hashtag_count', 'hashtag_optimal', 'hashtag_zero',
    'hashtag_spam', 'hashtag_quality_score',
    # Group D — Thumbnail (available)
    'thumb_brightness', 'thumb_contrast', 'thumb_saturation',
    # Group F — Temporal (UTC + LOCAL)
    'upload_hour_utc', 'upload_hour_local', 'upload_dow',
    'upload_month', 'upload_week_of_year',
    'is_peak_local', 'upload_time_bucket_enc',
    # Duration
    'duration_seconds',
]

FEATURES_ROBERTA = ['title', 'description']   # raw text

FEATURES_CATBOOST_SECTOR = [
    'title_length_chars', 'title_word_count', 'title_sentiment_polarity',
    'title_emoji_count', 'flesch_readability', 'hashtag_count',
    'hashtag_quality_score', 'description_url_count',
    'youtube_category_id', 'duration_seconds',
    'upload_hour_local', 'channel_age_days',
]

FEATURES_PHASE2_ADDITIONAL = [
    # Group E — add once transcript bug is fixed
    'speech_rate_wpm', 'transcript_word_count', 'transcript_qm_count',
    # Group D — add once face detection / OCR runs
    'thumb_face_count', 'thumb_has_text', 'thumb_text_word_count',
]

# Validate all features exist
available_xgb = [f for f in FEATURES_XGBOOST if f in df.columns]
missing_xgb   = [f for f in FEATURES_XGBOOST if f not in df.columns]

print(f'\n  XGBoost engagement classifier')
print(f'    Features available : {len(available_xgb)}')
print(f'    Features missing   : {len(missing_xgb)}  {missing_xgb}')

available_cat = [f for f in FEATURES_CATBOOST_SECTOR if f in df.columns]
print(f'\n  CatBoost sector classifier')
print(f'    Features available : {len(available_cat)}')

print(f'\n  RoBERTa text classifier')
print(f'    Input              : title + description (raw text)')

print(f'\n  Phase 2 additional features (after fixes)')
print(f'    Group E (transcript) : {len([f for f in FEATURES_PHASE2_ADDITIONAL if "transcript" in f or "speech" in f])} features')
print(f'    Group D (CV)         : {len([f for f in FEATURES_PHASE2_ADDITIONAL if "thumb" in f])} features')

# Save feature config to JSON for use by training scripts
feature_config = {
    'xgboost_features'   : available_xgb,
    'catboost_features'  : available_cat,
    'roberta_features'   : FEATURES_ROBERTA,
    'phase2_features'    : FEATURES_PHASE2_ADDITIONAL,
    'target_engagement'  : 'engagement_tier',
    'target_sector'      : 'sector',
    'optimal_hashtag_min': int(opt_min),
    'optimal_hashtag_max': int(opt_max),
}
config_path = DATA_DIR / 'feature_config.json'
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f'\n  Feature config saved → {config_path}')

# ══════════════════════════════════════════════════════════════════════════
# 6. SAVE ENHANCED DATASET
# ══════════════════════════════════════════════════════════════════════════
print(f'\n[5] Saving enhanced dataset...')

new_cols = ['channel_utc_offset', 'upload_hour_local', 'is_peak_local',
            'upload_time_bucket', 'upload_time_bucket_enc',
            'hashtag_optimal', 'hashtag_zero', 'hashtag_spam',
            'hashtag_quality_score', 'group_e_available']
new_cols_present = [c for c in new_cols if c in df.columns]

df.to_parquet(OUT_FILE, engine='pyarrow', compression='snappy', index=False)
print(f'  Saved → {OUT_FILE}')
print(f'  New columns added: {len(new_cols_present)}')
for c in new_cols_present:
    print(f'    + {c}')
print(f'  Total columns: {df.shape[1]}')

# Also save updated train/val/test splits
labeled = df[df['engagement_tier'].notna()].copy()
from sklearn.model_selection import train_test_split
labeled['strat_key'] = labeled['sector'].astype(str)+'_'+labeled['engagement_tier'].astype(str)
train, temp = train_test_split(labeled, test_size=0.30, random_state=42, stratify=labeled['strat_key'])
val, test   = train_test_split(temp,    test_size=0.50, random_state=42, stratify=temp['strat_key'])
for s in (train, val, test): s.drop(columns='strat_key', inplace=True)

train.to_parquet(DATA_DIR / 'btp2_train_v2.parquet', engine='pyarrow', compression='snappy', index=False)
val.to_parquet(DATA_DIR / 'btp2_val_v2.parquet',   engine='pyarrow', compression='snappy', index=False)
test.to_parquet(DATA_DIR / 'btp2_test_v2.parquet',  engine='pyarrow', compression='snappy', index=False)
print(f'\n  Updated splits saved (v2):')
print(f'    btp2_train_v2.parquet  {len(train):,} rows  ({len(train)/len(labeled)*100:.0f}%)')
print(f'    btp2_val_v2.parquet    {len(val):,} rows  ({len(val)/len(labeled)*100:.0f}%)')
print(f'    btp2_test_v2.parquet   {len(test):,} rows  ({len(test)/len(labeled)*100:.0f}%)')

print(f'\n{"═"*60}')
print(f'  Feature engineering v2 complete.')
print(f'  Next: python3 train_xgboost.py')
print(f'{"═"*60}')
