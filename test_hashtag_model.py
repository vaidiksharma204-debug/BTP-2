"""
test_hashtag_model.py
─────────────────────────────────────────────────────────────────
Run this on your Mac inside btp2_scraper/:
    python3 test_hashtag_model.py

Tests the saved recommender model across 20 diverse cases
and prints a quality score for each recommendation.
"""
import sys, pickle, re
sys.path.insert(0, '.')
from pathlib import Path
from hashtag_recommender import HashtagRecommender

MODELS_DIR = Path('models')

print('Loading model...')
with open(MODELS_DIR / 'hashtag_recommender.pkl', 'rb') as f:
    rec = pickle.load(f)
print(f'Vocab size : {len(rec.vocab):,} tags')
print(f'Sectors    : {list(rec.sector_tag_score.keys())}')

# ── 20 test cases covering all sectors + edge cases ──────────────────
TESTS = [
    # (title, description, sector, seed_tags, expected_good_tags)
    # Lifestyle
    ('10 Easy Recipes for Beginners Quick Tasty Cooking',
     'Simple meals you can make at home subscribe for more',
     'Lifestyle', ['cooking'],
     {'easyrecipes','recipe','food','quickrecipes','tasty','cookingtips'}),

    ('My Morning Routine Productivity Tips 2026',
     'How I stay focused and productive every single day',
     'Lifestyle', [],
     {'productivity','routine','morningroutine','motivation','lifestyle'}),

    ('Best Street Food Tour INDIA Mumbai Delhi',
     'Eating the most amazing street food in India',
     'Lifestyle', ['streetfood'],
     {'india','food','streetfood','mumbai','delhi','travel','foodie'}),

    ('Home Workout No Equipment Full Body 30 Minutes',
     'Complete fitness routine you can do anywhere',
     'Lifestyle', ['workout'],
     {'fitness','exercise','workout','homeworkout','bodyweight'}),

    # Gaming
    ('I survived 100 days in Minecraft Hardcore mode',
     'The hardest gaming challenge I have ever attempted',
     'Gaming', ['minecraft'],
     {'hardcore','minecraft','gaming','challenge','survival'}),

    ('AMONG US but with 100 players insane lobbies',
     'The most chaotic and funny round ever played',
     'Gaming', ['gaming'],
     {'amongus','gaming','funny','multiplayer'}),

    ('GTA 6 First Impressions Review Gameplay 2026',
     'Everything you need to know about the new game',
     'Gaming', [],
     {'gaming','gameplay','review','gta'}),

    ('Roblox but I can only use FREE items challenge',
     'Playing Roblox with zero robux is impossible',
     'Gaming', ['roblox'],
     {'roblox','gaming','challenge','funny'}),

    # Sports
    ('Arsenal vs Chelsea HIGHLIGHTS Premier League 2026',
     'All the goals and best moments from todays match',
     'Sports', ['football'],
     {'premierleague','highlights','arsenal','chelsea','soccer','football'}),

    ('Cricket India vs Australia Test Match Day 3',
     'Full highlights analysis and wickets today',
     'Sports', ['cricket'],
     {'cricket','india','australia','test','highlights'}),

    ('NBA Finals Game 7 HIGHLIGHTS Best Moments',
     'The most intense basketball game of the year',
     'Sports', ['basketball'],
     {'nba','basketball','highlights','finals'}),

    ('Cristiano Ronaldo Top 50 Goals of His Career',
     'The greatest goals ever scored by CR7',
     'Sports', ['football'],
     {'football','soccer','goals','ronaldo','highlights'}),

    # Education
    ('Learn Python in 1 Hour Complete Beginners Tutorial',
     'Code along step by step projects for absolute beginners',
     'Education', [],
     {'python','tutorial','programming','learntocode','coding','beginners'}),

    ('How does a CPU actually work Computer Science explained',
     'Deep dive into computer hardware and architecture basics',
     'Education', ['computerscience'],
     {'science','hardware','computer','tutorial','technology','coding'}),

    ('The French Revolution explained in 10 minutes History',
     'Everything you need to know about the revolution',
     'Education', [],
     {'history','education','france','revolution','learn'}),

    # Comedy
    ('Stand Up Comedy Special LIVE show funny jokes 2026',
     'The best stand up moments from our live show',
     'Comedy', [],
     {'comedy','funny','standup','standupcomedy','live','jokes','humor'}),

    ('Try Not to Laugh Challenge IMPOSSIBLE funny clips',
     'These videos will make you laugh every time',
     'Comedy', [],
     {'funny','comedy','laugh','challenge','funnyvideos'}),

    # Sci-Tech
    ('iPhone 16 Pro Full Review Honest camera test opinion',
     'Is it worth buying in 2026 camera comparison',
     'Sci-Tech', ['apple'],
     {'iphone','review','camera','apple','smartphone','tech'}),

    ('I built a robot that sorts my laundry full automation DIY',
     'Complete build tutorial with code and assembly',
     'Sci-Tech', ['robotics'],
     {'robot','tech','tutorial','automation','programming','coding'}),

    # Entertainment edge case
    ('Funniest Prank Videos of 2026 compilation viral reaction',
     'Epic prank gone viral funny moments reaction compilation',
     'Entertainment', ['prank'],
     {'viral','funny','prank','comedy','reaction','viralvideo'}),
]

# ── Run tests ─────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  HASHTAG RECOMMENDER — 20-CASE TEST')
print('='*65)

KNOWN_BAD = {
    'setindia','newshowalert','shorts','subscribe','peopleareawesome',
    'goodmythicalmorning','indianidol','aphmau','pewdiepie','callmekevin',
    'digdeeper','chromefordevelopers','brighterdaysahead','yildiz',
    'juve','juventus','mentalhealthawareness','sonyset','vishaldadlani',
}

total_good = 0
total_bad  = 0
total_hit  = 0
total_tags = 0
results_by_sector = {}

for title, desc, sector, seeds, expected in TESTS:
    recs = rec.recommend(title, desc, sector, seeds, n=5)

    good_tags = [r for r in recs if r['tag'].lstrip('#') not in KNOWN_BAD]
    bad_tags  = [r for r in recs if r['tag'].lstrip('#') in KNOWN_BAD]
    hits      = [r for r in recs if r['tag'].lstrip('#') in expected]

    total_good += len(good_tags)
    total_bad  += len(bad_tags)
    total_hit  += len(hits)
    total_tags += len(recs)

    if sector not in results_by_sector:
        results_by_sector[sector] = {'good': 0, 'bad': 0, 'hit': 0, 'total': 0, 'cases': 0}
    results_by_sector[sector]['good']  += len(good_tags)
    results_by_sector[sector]['bad']   += len(bad_tags)
    results_by_sector[sector]['hit']   += len(hits)
    results_by_sector[sector]['total'] += len(recs)
    results_by_sector[sector]['cases'] += 1

    quality = '✅' if len(bad_tags) == 0 else '⚠️' if len(bad_tags) <= 1 else '❌'
    print(f'\n{quality} [{sector}] {title[:55]}')
    for r in recs:
        tag_clean = r['tag'].lstrip('#')
        marker = '💚' if tag_clean in expected else ('🔴' if tag_clean in KNOWN_BAD else '  ')
        print(f'  {marker} {r["tag"]:<28} {r["score"]:>3}%  {r["reason"]}')

# ── Summary ───────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  QUALITY SUMMARY')
print('='*65)

quality_pct = total_good / total_tags * 100
hit_rate    = total_hit  / total_tags * 100

print(f'\n  Overall quality (no bad tags) : {quality_pct:.1f}%  ({total_good}/{total_tags})')
print(f'  Expected tag hit rate         : {hit_rate:.1f}%  ({total_hit}/{total_tags})')
print(f'  Bad tag appearances           : {total_bad}  (target: 0)')

print(f'\n  Per-sector:')
print(f'  {"Sector":<18} {"Quality":>8} {"Hits":>6} {"Bad":>5}')
print(f'  {"─"*42}')
for sector, s in sorted(results_by_sector.items()):
    q = s['good'] / s['total'] * 100
    h = s['hit']  / s['total'] * 100
    b = s['bad']
    status = '✅' if b == 0 else '⚠️'
    print(f'  {status} {sector:<16} {q:>7.1f}%  {h:>5.1f}%  {b:>4}')

print(f'\n  Legend: 💚 = expected tag   🔴 = known bad tag   blank = neutral')
print()

if quality_pct >= 90:
    print('  ✅ Model is DASHBOARD-READY (≥90% quality)')
elif quality_pct >= 75:
    print('  ⚠️  Model is acceptable (≥75% quality) — demo with confidence')
else:
    print('  ❌ Quality too low — needs more blacklisting')
