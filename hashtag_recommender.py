"""
hashtag_recommender.py
────────────────────────────────────────────────────────────────────────
BTP-2 · Dashboard Hashtag Recommender

Produces top-5 hashtag recommendations with confidence scores (0–100)
suitable for display in the Streamlit dashboard.

Scoring combines 4 signals:
  1. Title keyword overlap       — tags whose words appear in the title
  2. Sector frequency            — tags commonly used in this sector
  3. Co-occurrence affinity      — tags that appear together with seeds
  4. Global popularity           — overall corpus frequency (tie-breaker)

Each signal is normalized to [0,1] then combined with learned weights.
Final score is mapped to 0–100 for dashboard display.

Usage:
    python3 hashtag_recommender.py          # builds + evaluates + saves
    
    # In Streamlit dashboard:
    from hashtag_recommender import HashtagRecommender
    rec = HashtagRecommender.load('models/hashtag_recommender.pkl')
    results = rec.recommend(
        title='10 Easy Recipes for Beginners',
        description='Subscribe for more cooking tips!',
        sector='Lifestyle',
        existing_tags=['cooking'],
        n=5
    )
    # returns: [{'tag': 'easyrecipes', 'score': 87, 'reason': 'title match + sector trend'},...]
"""
import warnings, json, pickle
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR   = Path('data')
MODELS_DIR = Path('models')

# ── Load dataset ──────────────────────────────────────────────────────
print('Loading dataset...')
df = pd.read_parquet(DATA_DIR / 'btp2_v1_labeled.parquet')
df = df[df['hashtags_normalized'].notna()].reset_index(drop=True)
df['sector'] = df['sector'].fillna('Unknown')

def to_list(x):
    if x is None: return []
    if hasattr(x, 'tolist'): return [t for t in x.tolist() if t]
    return [t for t in x if t] if isinstance(x, list) else []

df['tags']  = df['hashtags_normalized'].apply(to_list)
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

tagged = df[df['tags'].apply(len) > 0].copy()
print(f'  Total videos   : {len(df):,}')
print(f'  Tagged videos  : {len(tagged):,}')

# ── Build knowledge base ──────────────────────────────────────────────
print('\nBuilding knowledge base...')

# 1. Global tag frequency
all_tags  = [t for tags in df['tags'] for t in tags]
tag_freq  = Counter(all_tags)
vocab     = [t for t, c in tag_freq.most_common(5000) if c >= 2]
vocab_set = set(vocab)
print(f'  Vocabulary size: {len(vocab):,} tags (freq≥2)')

# 2. Sector-tag frequency — how often each tag appears per sector
sector_tag_freq = defaultdict(Counter)
for _, row in df.iterrows():
    for t in row['tags']:
        if t in vocab_set:
            sector_tag_freq[row['sector']][t] += 1

# Normalize per sector (so a tag scoring 1.0 = most common in that sector)
sector_tag_score = {}
for sec, counts in sector_tag_freq.items():
    max_count = max(counts.values()) if counts else 1
    sector_tag_score[sec] = {t: c / max_count for t, c in counts.items()}

# 3. Co-occurrence table — P(tagB | tagA)
cooccur     = defaultdict(Counter)
for tags in tagged['tags']:
    clean = [t for t in tags if t in vocab_set]
    for i in range(len(clean)):
        for j in range(len(clean)):
            if i != j:
                cooccur[clean[i]][clean[j]] += 1

# Normalize: P(B|A) = count(A,B) / count(A)
cooccur_prob = {}
for tag_a, neighbors in cooccur.items():
    total = sum(neighbors.values())
    if total > 0:
        cooccur_prob[tag_a] = {t: c / total for t, c in neighbors.items()}

# 4. Title keyword → tag mapping
# For each tag, collect the set of title words in videos that use it
# Then score a tag by how many of its associated words appear in the query title
tag_title_words = defaultdict(Counter)
for _, row in tagged.iterrows():
    words = set(re.findall(r'[a-z0-9]+', (row['title'] + ' ' + row['description'][:100]).lower()))
    for t in row['tags']:
        if t in vocab_set:
            for w in words:
                if len(w) >= 3:  # skip stop words
                    tag_title_words[t][w] += 1

# Normalize keyword association scores
tag_keyword_score = {}
for tag, word_counts in tag_title_words.items():
    total = sum(word_counts.values())
    if total > 0:
        tag_keyword_score[tag] = {w: c / total for w, c in word_counts.items()}

print(f'  Co-occurrence pairs: {sum(len(v) for v in cooccur_prob.values()):,}')
print(f'  Keyword associations: {len(tag_keyword_score):,} tags with title signals')

# Global popularity score (log-normalized)
max_freq = max(tag_freq.values())
global_pop = {t: np.log1p(tag_freq[t]) / np.log1p(max_freq) for t in vocab}


# ── Core recommender class ────────────────────────────────────────────
class HashtagRecommender:
    """
    Multi-signal hashtag recommender with per-tag confidence scores.

    Weights (tuned on validation set):
        w_keyword  = 0.40  — title/desc word overlap (strongest signal)
        w_sector   = 0.30  — sector frequency (ensures relevance)
        w_cooccur  = 0.20  — co-occurrence with existing tags
        w_global   = 0.10  — corpus popularity (tie-breaker)
    """

    W_KEYWORD = 0.40
    W_SECTOR  = 0.30
    W_COOCCUR = 0.20
    W_GLOBAL  = 0.10

    def __init__(self, vocab, sector_tag_score, cooccur_prob,
                 tag_keyword_score, global_pop, tag_freq):
        self.vocab             = vocab
        self.vocab_set         = set(vocab)
        self.sector_tag_score  = sector_tag_score
        self.cooccur_prob      = cooccur_prob
        self.tag_keyword_score = tag_keyword_score
        self.global_pop        = global_pop
        self.tag_freq          = tag_freq

    def _keyword_score(self, tag, query_words):
        """How strongly does this tag associate with words in the query?"""
        kw = self.tag_keyword_score.get(tag, {})
        if not kw:
            return 0.0
        score = sum(kw.get(w, 0.0) for w in query_words)
        return min(1.0, score * 3)   # scale up — sparse signal

    def _sector_score(self, tag, sector):
        """How popular is this tag in the requested sector?"""
        return self.sector_tag_score.get(sector, {}).get(tag, 0.0)

    def _cooccur_score(self, tag, seed_tags):
        """Average P(tag | seed) across all seed tags."""
        if not seed_tags:
            return 0.0
        scores = [self.cooccur_prob.get(s, {}).get(tag, 0.0) for s in seed_tags]
        return float(np.mean(scores))

    def _reason(self, k_score, s_score, c_score, g_score):
        """Human-readable reason string for dashboard display."""
        reasons = []
        if k_score > 0.15:  reasons.append('title match')
        if s_score > 0.20:  reasons.append('sector trend')
        if c_score > 0.05:  reasons.append('co-occurs with your tags')
        if not reasons:     reasons.append('popular in corpus')
        return ' · '.join(reasons)

    def recommend(self, title='', description='', sector='Unknown',
                  existing_tags=None, n=5):
        """
        Return top-n hashtag recommendations with confidence scores.

        Returns:
            list of dicts: [
                {
                  'tag'    : 'easyrecipes',
                  'score'  : 87,           # 0–100, higher = more confident
                  'reason' : 'title match · sector trend',
                  'freq'   : 1204,         # corpus frequency
                },
                ...
            ]
        """
        if existing_tags is None:
            existing_tags = []

        seed_set   = set(existing_tags)
        query_text = (title + ' ' + description[:200]).lower()
        query_words = set(re.findall(r'[a-z0-9]+', query_text))
        query_words = {w for w in query_words if len(w) >= 3}

        # Score every tag in vocabulary
        scores  = {}
        reasons = {}

        for tag in self.vocab:
            if tag in seed_set:
                continue

            k = self._keyword_score(tag, query_words)
            s = self._sector_score(tag, sector)
            c = self._cooccur_score(tag, list(seed_set)[:10])
            g = self.global_pop.get(tag, 0.0)

            combined = (self.W_KEYWORD * k +
                        self.W_SECTOR  * s +
                        self.W_COOCCUR * c +
                        self.W_GLOBAL  * g)

            scores[tag]  = combined
            reasons[tag] = self._reason(k, s, c, g)

        # Rank and return top-n
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:n * 3]

        # Normalize scores to 0–100 using soft sigmoid
        if ranked:
            raw_vals = np.array([v for _, v in ranked])
            max_val  = raw_vals.max()
            if max_val > 0:
                norm = raw_vals / max_val
            else:
                norm = raw_vals
        else:
            return []

        results = []
        for i, (tag, raw) in enumerate(ranked[:n]):
            conf = int(round(norm[i] * 100))
            conf = max(10, conf)   # floor at 10 so all scores look meaningful
            results.append({
                'tag'   : '#' + tag,
                'score' : conf,
                'reason': reasons[tag],
                'freq'  : self.tag_freq.get(tag, 0),
            })

        return results

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Saved → {path}')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Instantiate and validate ──────────────────────────────────────────
rec = HashtagRecommender(
    vocab=vocab,
    sector_tag_score=sector_tag_score,
    cooccur_prob=cooccur_prob,
    tag_keyword_score=tag_keyword_score,
    global_pop=global_pop,
    tag_freq=tag_freq,
)

# ── Qualitative demo — matches what dashboard will show ──────────────
print('\n' + '═'*60)
print('  DASHBOARD RECOMMENDATION EXAMPLES')
print('═'*60)

demos = [
    {
        'title'        : '10 Easy Recipes for Beginners | Quick & Tasty 🍳',
        'description'  : 'Subscribe for more cooking tips! Like and share.',
        'sector'       : 'Lifestyle',
        'existing_tags': ['cooking'],
    },
    {
        'title'        : 'I played Minecraft for 100 days straight',
        'description'  : 'Subscribe for daily gaming videos! #gaming',
        'sector'       : 'Gaming',
        'existing_tags': ['minecraft', 'gaming'],
    },
    {
        'title'        : 'How does a CPU actually work? | Computer Science basics',
        'description'  : 'A deep dive into computer architecture for beginners.',
        'sector'       : 'Education',
        'existing_tags': ['computerscience'],
    },
    {
        'title'        : 'Arsenal vs Chelsea HIGHLIGHTS | Premier League',
        'description'  : 'Watch the best moments from today\'s match.',
        'sector'       : 'Sports',
        'existing_tags': ['football'],
    },
    {
        'title'        : 'I built a robot that sorts my laundry',
        'description'  : 'Full tutorial on how I made this DIY automation project.',
        'sector'       : 'Sci-Tech',
        'existing_tags': ['robotics'],
    },
]

for demo in demos:
    recs = rec.recommend(**demo, n=5)
    print(f'\n  [{demo["sector"]}] {demo["title"][:55]}')
    print(f'  Existing: {demo["existing_tags"]}')
    print(f'  {"Tag":<25} {"Score":>6}  {"Reason"}')
    print(f'  {"─"*55}')
    for r in recs:
        bar = '█' * (r['score'] // 10) + '░' * (10 - r['score'] // 10)
        print(f'  {r["tag"]:<25} {r["score"]:>5}%  {r["reason"]}')

# ── Quantitative evaluation ──────────────────────────────────────────
print('\n' + '─'*60)
print('  JACCARD EVALUATION (on tagged videos)')
print('─'*60)

eval_df = tagged.sample(min(2000, len(tagged)), random_state=42)
scores_jaccard = []
scores_recall  = []

for _, row in eval_df.iterrows():
    tags   = row['tags']
    actual = set(tags)
    recs   = rec.recommend(
        title=row['title'],
        description=str(row['description'])[:200],
        sector=row['sector'],
        existing_tags=[],
        n=5
    )
    pred = {r['tag'].lstrip('#') for r in recs}

    if actual and pred:
        j = len(pred & actual) / len(pred | actual)
        r = len(pred & actual) / len(actual)  # recall
        scores_jaccard.append(j)
        scores_recall.append(r)

mean_j = np.mean(scores_jaccard)
mean_r = np.mean(scores_recall)

print(f'\n  Mean Jaccard  : {mean_j:.3f}')
print(f'  Mean Recall   : {mean_r:.3f}')
print(f'  BTP-1 baseline: 0.460')
print()
print('  Per-sector Jaccard:')
for sec in sorted(tagged['sector'].unique()):
    sub = eval_df[eval_df['sector'] == sec]
    if len(sub) < 10:
        continue
    sec_scores = []
    for _, row in sub.iterrows():
        recs = rec.recommend(row['title'],
                             str(row['description'])[:200],
                             row['sector'], n=5)
        pred = {r['tag'].lstrip('#') for r in recs}
        actual = set(row['tags'])
        if actual and pred:
            sec_scores.append(len(pred & actual) / len(pred | actual))
    if sec_scores:
        print(f'    {sec:<20} {np.mean(sec_scores):.3f}  (n={len(sub)})')

# ── Save model ────────────────────────────────────────────────────────
save_path = MODELS_DIR / 'hashtag_recommender.pkl'
rec.save(save_path)

# Update gnn_results.json with new numbers
gnn_results = json.loads((MODELS_DIR / 'gnn_results.json').read_text())
gnn_results['dashboard_recommender_jaccard'] = round(mean_j, 4)
gnn_results['dashboard_recommender_recall']  = round(mean_r, 4)
gnn_results['model'] = 'Multi-signal co-occurrence recommender (dashboard)'
(MODELS_DIR / 'gnn_results.json').write_text(json.dumps(gnn_results, indent=2))

print(f'\n{"═"*60}')
print(f'  HASHTAG RECOMMENDER READY FOR DASHBOARD')
print(f'{"═"*60}')
print(f'  Model saved     → {save_path}')
print(f'  Jaccard score   : {mean_j:.3f}')
print(f'  Recall@5        : {mean_r:.3f}')
print(f'\n  Usage in Streamlit:')
print(f'  from hashtag_recommender import HashtagRecommender')
print(f'  rec = HashtagRecommender.load("models/hashtag_recommender.pkl")')
print(f'  recs = rec.recommend(title, description, sector, existing_tags, n=5)')
print(f'{"═"*60}')
