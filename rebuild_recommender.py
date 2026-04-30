"""
Rebuild hashtag recommender with proper word-boundary matching.
Run: python3 rebuild_recommender.py
"""
import sys, re, pickle, numpy as np, types
sys.path.insert(0, '.')
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from hashtag_recommender import HashtagRecommender

MODELS_DIR = Path('models')
DATA_DIR   = Path('data')

with open(MODELS_DIR / 'hashtag_recommender.pkl', 'rb') as f:
    rec = pickle.load(f)

# ── Blacklist ──────────────────────────────────────────────────────────
BLACKLIST = {
    'setindia','newshowalert','sonyliv','colorstv','vijaytelevision',
    'zeetv','starzplay','hotstar','sonyentertainment','sonylivindia',
    'fallontonight','jimmyfallon','jimmykimmel','nbcfallontonight',
    'thecodereport','chromefordevelopers','100secondsofcode','digdeeper',
    'iliketomakestuff','peopleareawesome','callmekevin','pewdiepie',
    'juventus','juve','yildiz','brighterdaysahead','theodorekarlovec',
    'sonyset','vishaldadlani','singingshow','realityshow','newseason',
    'aphmau','philza','smallant','ssundee','mrbean','kurzgesagt',
    'nationalgeographic','chrome','pydantic','webdevelopment',
    'fifawwc','fifaseries','splatoon3nal2026','nfldraft',
    'mentalhealthawareness','sharktankindia','sharktankindiaseason5',
    'sharktankindiasonyliv','indiasbestdancer3','vegetta777',
    'learnwithshorts','gameshow','newshow','scienceshorts',
    'shorts','subscribe','venv','flask','matplotlib','django',
}

# ── Rebuild vocab ──────────────────────────────────────────────────────
rec.vocab = [t for t in rec.vocab
             if t not in BLACKLIST
             and len(t) >= 4
             and rec.tag_freq.get(t, 0) >= 8]
rec.vocab_set = set(rec.vocab)
print(f'Vocab: {len(rec.vocab):,} tags')

# ── Tag word splitter ──────────────────────────────────────────────────
# Split compound tags into component words
# e.g. 'easyrecipes' -> {'easy', 'recipes', 'recipe'}
# e.g. 'premierleague' -> {'premier', 'league'}

# Build a simple word list for splitting (sorted by length desc)
WORD_LIST = sorted([
    'easy','recipes','recipe','cooking','food','gaming','game','games',
    'minecraft','football','soccer','sports','league','premier','highlights',
    'science','education','learn','tutorial','review','funny','comedy',
    'music','dance','travel','fitness','health','beauty','fashion',
    'tech','technology','phone','iphone','android','computer','code',
    'programming','robot','automation','artificial','intelligence',
    'morning','routine','productivity','tips','advice','guide','help',
    'india','america','world','global','best','top','most','viral',
    'live','show','episode','season','series','movie','film','video',
    'among','player','players','challenge','days','hours','minutes',
    'stand','jokes','humor','prank','reaction','short','long','full',
    'news','politics','president','government','economy','crypto',
    'sport','basketball','cricket','tennis','golf','baseball','hockey',
    'workout','exercise','yoga','meditation','mental','physical',
    'channel','subscribe','like','comment','share','follow',
    'english','hindi','spanish','french','german','japanese',
    'review','unboxing','haul','vlog','podcast','interview',
    'draw','paint','art','design','creative','photography',
    'diet','weight','lose','gain','muscle','cardio','running',
    'makeup','skincare','hair','nails','outfit','style',
    'build','make','craft','diy','project','home','house',
    'cat','dog','animal','nature','garden','plant','flower',
], key=len, reverse=True)

def split_tag_words(tag):
    """Split a compound hashtag into component words."""
    words = set()
    remaining = tag.lower()
    for word in WORD_LIST:
        if word in remaining and len(word) >= 4:
            words.add(word)
            remaining = remaining.replace(word, ' ')
    # Also add the full tag and any remaining chunks
    chunks = [c for c in re.split(r'\s+', remaining) if len(c) >= 4]
    words.update(chunks)
    words.add(tag)  # always include the full tag
    return words

# Precompute tag word sets for all vocab tags
tag_word_sets = {t: split_tag_words(t) for t in rec.vocab}


def smart_recommend(self, title='', description='', sector='Unknown',
                    existing_tags=None, n=5):
    if existing_tags is None:
        existing_tags = []
    seed_set = set(t.lstrip('#').lower() for t in existing_tags)

    # Extract meaningful words from title (primary) and description (secondary)
    title_words = set(re.findall(r'\b[a-z]{4,}\b', title.lower()))
    desc_words  = set(re.findall(r'\b[a-z]{4,}\b', description.lower()))

    # Remove common stopwords that cause false matches
    STOP = {'this','that','with','from','have','what','will','your','they',
            'been','when','were','then','than','here','there','most','more',
            'some','into','over','also','just','like','very','well','back',
            'much','many','only','even','been','their','would','could',
            'about','after','before','other','which','these','those'}
    title_words -= STOP
    desc_words  -= STOP

    scores  = {}
    reasons = {}

    for tag in self.vocab:
        if tag in seed_set:
            continue

        tag_words = tag_word_sets.get(tag, {tag})

        # ── Signal 1: Title keyword match (WORD BOUNDARY only) ─────────
        # Score = fraction of tag's component words found in title
        title_overlap = len(tag_words & title_words) / max(len(tag_words), 1)
        # Boost if FULL tag text appears as a word in title
        full_tag_in_title = tag in title_words
        k_title = min(1.0, title_overlap * 2 + (0.5 if full_tag_in_title else 0))

        # Description adds a smaller signal
        desc_overlap = len(tag_words & desc_words) / max(len(tag_words), 1)
        k_desc = min(0.4, desc_overlap)

        k = min(1.0, k_title * 0.75 + k_desc * 0.25)

        # ── Signal 2: Sector frequency ─────────────────────────────────
        sec_dict = self.sector_tag_score.get(sector, {})
        s = sec_dict.get(tag, 0.0)
        # Penalize single-sector tags (channel-specific)
        n_sectors = sum(1 for sd in self.sector_tag_score.values() if tag in sd)
        if n_sectors <= 1:
            s *= 0.2

        # ── Signal 3: Co-occurrence with seeds ─────────────────────────
        c = 0.0
        for seed in list(seed_set)[:5]:
            c += self.cooccur_prob.get(seed, {}).get(tag, 0.0)
        c = min(1.0, c * 2)

        # ── Signal 4: Global popularity ────────────────────────────────
        g = self.global_pop.get(tag, 0.0)

        # ── Combine ────────────────────────────────────────────────────
        # Title match and co-occurrence are king
        combined = 0.45 * k + 0.20 * s + 0.25 * c + 0.10 * g

        # HARD PENALTY: if zero title signal AND zero co-occurrence
        # only show if sector score is very high
        if k == 0 and c == 0 and s < 0.5:
            combined *= 0.1

        scores[tag] = combined

        # Reason string
        parts = []
        if k_title > 0.15:  parts.append('title match')
        if k_desc > 0.10:   parts.append('description match')
        if s > 0.15 and n_sectors > 1: parts.append('sector trend')
        if c > 0.08:        parts.append('co-occurs with your tags')
        if not parts:       parts.append('popular in corpus')
        reasons[tag] = ' · '.join(parts)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:n * 3]
    if not ranked:
        return []

    raw  = np.array([v for _, v in ranked])
    mx   = raw.max()
    norm = raw / mx if mx > 0 else raw

    results = []
    for i, (tag, _) in enumerate(ranked[:n]):
        results.append({
            'tag'   : '#' + tag,
            'score' : max(10, int(round(norm[i] * 100))),
            'reason': reasons[tag],
            'freq'  : self.tag_freq.get(tag, 0),
        })
    return results


rec.recommend = types.MethodType(smart_recommend, rec)

# ── Test suite ─────────────────────────────────────────────────────────
print('\n' + '='*60)
tests = [
    ('This is how I pranked the President of INDIA', 'Epic prank gone viral funny reaction', 'Entertainment', ['prank']),
    ('10 Easy Recipes for Beginners Quick and Tasty', 'Subscribe for weekly cooking tips', 'Lifestyle', ['cooking']),
    ('I survived 100 days in Minecraft Hardcore mode', 'The hardest gaming challenge I have ever done', 'Gaming', ['minecraft']),
    ('Arsenal vs Chelsea HIGHLIGHTS Premier League 2026', 'Watch the best match moments', 'Sports', ['football']),
    ('I built a robot that sorts my laundry automation DIY', 'Full build tutorial step by step', 'Sci-Tech', ['robotics']),
    ('How does a CPU work Computer Science basics explained', 'Deep dive into hardware for beginners', 'Education', ['computerscience']),
    ('My Morning Routine 2026 Productivity and Focus Tips', 'How I stay productive and focused every day', 'Lifestyle', []),
    ('AMONG US but with 100 players insane game', 'The most crazy round ever with friends', 'Gaming', ['gaming']),
    ('Stand Up Comedy Special LIVE show funny jokes', 'Best stand up moments of 2026', 'Comedy', []),
    ('iPhone 16 Pro honest Review camera test', 'Best smartphone camera comparison 2026', 'Sci-Tech', ['apple']),
    ('Cricket India vs Australia Test Match highlights', 'Day 3 full highlights and analysis', 'Sports', ['cricket']),
    ('Learn Python in 1 hour for absolute beginners', 'Complete tutorial with projects', 'Education', []),
]

for title, desc, sector, seeds in tests:
    recs = rec.recommend(title, desc, sector, seeds, n=5)
    print(f'\n[{sector}] {title[:58]}')
    for r in recs:
        print(f'  {r["tag"]:<28} {r["score"]:>3}%  {r["reason"]}')

# Save
with open(MODELS_DIR / 'hashtag_recommender.pkl', 'wb') as f:
    pickle.dump(rec, f)
print('\n✅ Saved → models/hashtag_recommender.pkl')
print('Restart dashboard: python3 -m streamlit run dashboard.py')
