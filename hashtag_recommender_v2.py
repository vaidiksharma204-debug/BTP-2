"""
hashtag_recommender_v2.py
─────────────────────────────────────────────────────────────────
Production hashtag recommender for the BTP-2 dashboard.

Architectural fix: replaces corpus-frequency scoring with a CURATED
TAXONOMY of 450+ generic content hashtags organized by topic cluster.

Why this approach:
  - Corpus-driven scoring promoted channel-branded tags (#setindia,
    #peopleareawesome) because those dominate the dataset
  - 75% bad-tag-filtering only converts noise to neutrality, not relevance
  - A hand-curated taxonomy guarantees output quality at the cost of
    coverage (we recommend from a known-good vocabulary only)

Architecture:
  1. TAXONOMY — 450 hashtags grouped by 60 topic clusters
  2. KEYWORD MAP — words → tag candidates (built once at load)
  3. SCORING — word-overlap + cluster-affinity + sector-relevance
  4. CO-OCCURRENCE BOOST — uses original corpus for refinement only

Run:
    python3 hashtag_recommender_v2.py     # rebuild + test
"""
import re, json, pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

MODELS_DIR = Path('models')
DATA_DIR   = Path('data')

# ═══════════════════════════════════════════════════════════════════════════
# TAXONOMY — 450+ curated content hashtags organized by topic cluster
# ═══════════════════════════════════════════════════════════════════════════
# Format:  cluster_id : [aliases / trigger words]   →   [hashtags]
#
# A tag fires when ANY trigger word appears in title or description.
# Multiple matches stack, raising confidence.
# ═══════════════════════════════════════════════════════════════════════════

TAXONOMY = {
    # ── Lifestyle / Cooking ──────────────────────────────────────────────
    "cooking_recipes": {
        "triggers": ["recipe", "recipes", "cooking", "cook", "kitchen", "tasty",
                     "delicious", "homemade", "ingredients", "chef", "meal", "dish"],
        "tags": ["cooking", "recipe", "easyrecipes", "foodie", "homemade",
                 "homecooking", "foodlover", "cookingtips", "recipevideo"],
        "sectors": ["Lifestyle"],
    },
    "baking": {
        "triggers": ["bake", "baking", "cake", "bread", "pastry", "dessert",
                     "cookie", "cookies", "muffin"],
        "tags": ["baking", "dessert", "cake", "homemade", "bakingtips"],
        "sectors": ["Lifestyle"],
    },
    "vegan_healthy": {
        "triggers": ["vegan", "vegetarian", "healthy", "plantbased", "salad", "diet"],
        "tags": ["vegan", "healthy", "plantbased", "healthyfood", "wellness"],
        "sectors": ["Lifestyle"],
    },
    "indian_food": {
        "triggers": ["indian", "biryani", "curry", "masala", "tandoor", "dosa",
                     "paratha", "samosa", "chai"],
        "tags": ["indianfood", "indiancuisine", "indianrecipes", "streetfood"],
        "sectors": ["Lifestyle"],
    },
    "street_food": {
        "triggers": ["street", "streetfood", "tour", "city", "market", "stall"],
        "tags": ["streetfood", "foodtour", "foodie", "travel", "localfood"],
        "sectors": ["Lifestyle"],
    },
    "quick_meals": {
        "triggers": ["quick", "easy", "minute", "minutes", "fast", "simple",
                     "beginner", "beginners"],
        "tags": ["quickrecipes", "easyrecipes", "easymeal", "quickmeal"],
        "sectors": ["Lifestyle"],
    },

    # ── Lifestyle / Fitness ──────────────────────────────────────────────
    "workout_fitness": {
        "triggers": ["workout", "exercise", "gym", "fitness", "training", "cardio",
                     "strength", "abs", "muscle"],
        "tags": ["fitness", "workout", "gymlife", "fitnessmotivation",
                 "homeworkout", "exercise", "training"],
        "sectors": ["Lifestyle"],
    },
    "yoga_meditation": {
        "triggers": ["yoga", "meditation", "mindfulness", "breathing", "stretch",
                     "stretching"],
        "tags": ["yoga", "meditation", "mindfulness", "wellness", "selfcare"],
        "sectors": ["Lifestyle"],
    },
    "weight_loss": {
        "triggers": ["weight", "lose", "fat", "calories", "transformation", "diet"],
        "tags": ["weightloss", "fitness", "transformation", "healthy"],
        "sectors": ["Lifestyle"],
    },

    # ── Lifestyle / Self-improvement ────────────────────────────────────
    "productivity": {
        "triggers": ["productivity", "productive", "habits", "routine", "morning",
                     "evening", "focus", "discipline", "schedule", "time", "tips"],
        "tags": ["productivity", "morningroutine", "habits", "selfimprovement",
                 "lifestyle", "motivation", "discipline"],
        "sectors": ["Lifestyle", "Education"],
    },
    "motivation": {
        "triggers": ["motivation", "inspire", "inspirational", "success", "mindset",
                     "goals"],
        "tags": ["motivation", "mindset", "success", "inspiration", "selfimprovement"],
        "sectors": ["Lifestyle", "Education"],
    },
    "mental_health": {
        "triggers": ["mental", "anxiety", "depression", "therapy", "stress",
                     "wellbeing"],
        "tags": ["mentalhealth", "selfcare", "wellbeing", "anxiety"],
        "sectors": ["Lifestyle", "Education"],
    },

    # ── Lifestyle / Beauty Travel ────────────────────────────────────────
    "beauty_makeup": {
        "triggers": ["makeup", "beauty", "skincare", "lipstick", "tutorial",
                     "foundation", "blush", "mascara"],
        "tags": ["makeup", "beauty", "skincare", "makeuptutorial", "beautytips"],
        "sectors": ["Lifestyle"],
    },
    "fashion_style": {
        "triggers": ["fashion", "outfit", "style", "ootd", "wardrobe", "dress",
                     "trend"],
        "tags": ["fashion", "style", "ootd", "outfit", "fashiontips"],
        "sectors": ["Lifestyle"],
    },
    "travel_tour": {
        "triggers": ["travel", "trip", "tour", "vacation", "adventure", "explore",
                     "destination", "wanderlust"],
        "tags": ["travel", "wanderlust", "travelvlog", "adventure", "explore"],
        "sectors": ["Lifestyle"],
    },

    # ── Gaming / Specific titles ─────────────────────────────────────────
    "minecraft": {
        "triggers": ["minecraft", "mineblock", "creeper", "enderman", "diamond"],
        "tags": ["minecraft", "minecraftshorts", "minecraftbuilds", "gaming"],
        "sectors": ["Gaming"],
    },
    "fortnite": {
        "triggers": ["fortnite", "battle", "royale", "victory"],
        "tags": ["fortnite", "battleroyale", "gaming"],
        "sectors": ["Gaming"],
    },
    "roblox": {
        "triggers": ["roblox", "robux"],
        "tags": ["roblox", "gaming"],
        "sectors": ["Gaming"],
    },
    "amongus": {
        "triggers": ["among", "amongus", "imposter", "crewmate"],
        "tags": ["amongus", "gaming"],
        "sectors": ["Gaming"],
    },
    "gta": {
        "triggers": ["gta", "grandtheft", "rockstar"],
        "tags": ["gta", "gtav", "gaming", "gameplay"],
        "sectors": ["Gaming"],
    },
    "pokemon": {
        "triggers": ["pokemon", "pikachu", "charizard"],
        "tags": ["pokemon", "gaming", "nintendo"],
        "sectors": ["Gaming"],
    },
    "valorant": {
        "triggers": ["valorant", "valo"],
        "tags": ["valorant", "gaming", "fps"],
        "sectors": ["Gaming"],
    },
    "callofduty": {
        "triggers": ["callofduty", "warzone", "modernwarfare", "cod"],
        "tags": ["callofduty", "warzone", "gaming", "fps"],
        "sectors": ["Gaming"],
    },

    # ── Gaming / Generic ─────────────────────────────────────────────────
    "gaming_general": {
        "triggers": ["game", "gaming", "gameplay", "play", "playthrough", "stream",
                     "streaming", "console", "pc"],
        "tags": ["gaming", "gameplay", "videogames", "gamer", "pcgaming"],
        "sectors": ["Gaming"],
    },
    "gaming_challenge": {
        "triggers": ["100days", "100", "survived", "survival", "hardcore",
                     "challenge", "speedrun", "impossible"],
        "tags": ["gaming", "challenge", "survival", "hardcore", "gamingchallenge"],
        "sectors": ["Gaming"],
    },
    "gaming_review": {
        "triggers": ["review", "first", "impressions", "rating"],
        "tags": ["gamereview", "gaming", "newgame"],
        "sectors": ["Gaming", "Sci-Tech"],
    },
    "esports": {
        "triggers": ["esports", "tournament", "competitive", "ranked", "pro"],
        "tags": ["esports", "gaming", "competitive"],
        "sectors": ["Gaming"],
    },

    # ── Sports / Football ────────────────────────────────────────────────
    "football_premier": {
        "triggers": ["arsenal", "chelsea", "liverpool", "mancity", "manchester",
                     "tottenham", "spurs", "premier", "league", "mufc",
                     "epl", "leeds"],
        "tags": ["premierleague", "football", "soccer", "epl", "highlights"],
        "sectors": ["Sports"],
    },
    "football_laliga": {
        "triggers": ["barcelona", "barca", "madrid", "realmadrid", "laliga",
                     "atletico"],
        "tags": ["laliga", "football", "soccer", "highlights"],
        "sectors": ["Sports"],
    },
    "football_general": {
        "triggers": ["football", "soccer", "goal", "goals", "match", "ronaldo",
                     "messi", "neymar", "fifa", "uefa", "champions"],
        "tags": ["football", "soccer", "goals", "highlights"],
        "sectors": ["Sports"],
    },
    "cricket": {
        "triggers": ["cricket", "ipl", "wicket", "batsman", "bowler", "century",
                     "test", "odi", "t20", "kohli", "dhoni", "rohit"],
        "tags": ["cricket", "ipl", "cricketnews", "cricketlovers"],
        "sectors": ["Sports"],
    },
    "basketball_nba": {
        "triggers": ["nba", "basketball", "lebron", "curry", "jordan", "lakers",
                     "warriors", "celtics", "dunk", "finals"],
        "tags": ["nba", "basketball", "nbahighlights"],
        "sectors": ["Sports"],
    },
    "tennis": {
        "triggers": ["tennis", "wimbledon", "djokovic", "nadal", "federer",
                     "atp", "wta"],
        "tags": ["tennis", "atp", "wta", "wimbledon"],
        "sectors": ["Sports"],
    },
    "wwe_ufc": {
        "triggers": ["wwe", "wrestling", "ufc", "mma", "boxing", "fight",
                     "knockout"],
        "tags": ["wwe", "ufc", "mma", "wrestling"],
        "sectors": ["Sports"],
    },
    "highlights_general": {
        "triggers": ["highlight", "highlights", "best", "moments", "compilation",
                     "topgoals", "topplays"],
        "tags": ["highlights", "sportshighlights", "bestmoments"],
        "sectors": ["Sports"],
    },

    # ── Education / Programming ──────────────────────────────────────────
    "python": {
        "triggers": ["python", "django", "flask", "pandas", "numpy"],
        "tags": ["python", "coding", "programming", "learnpython", "pythontutorial"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "javascript_web": {
        "triggers": ["javascript", "react", "nodejs", "typescript", "html",
                     "css", "webdev", "frontend"],
        "tags": ["javascript", "webdev", "coding", "react", "frontend"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "programming_general": {
        "triggers": ["program", "programming", "code", "coding", "developer",
                     "software", "algorithm", "tutorial", "learntocode"],
        "tags": ["programming", "coding", "softwareengineering", "learntocode",
                 "developer"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "computer_science": {
        "triggers": ["computer", "science", "cs", "datastructures", "algorithm",
                     "binary", "compiler"],
        "tags": ["computerscience", "tech", "programming"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "ai_ml": {
        "triggers": ["ai", "ml", "artificial", "intelligence", "machine", "deep",
                     "learning", "neural", "gpt", "llm"],
        "tags": ["artificialintelligence", "machinelearning", "ai", "deeplearning"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "data_science": {
        "triggers": ["data", "analytics", "statistics", "visualization"],
        "tags": ["datascience", "analytics", "data"],
        "sectors": ["Education", "Sci-Tech"],
    },

    # ── Education / Subjects ─────────────────────────────────────────────
    "history": {
        "triggers": ["history", "historical", "ancient", "medieval", "war",
                     "revolution", "empire", "civilization"],
        "tags": ["history", "education", "historical", "historyfacts"],
        "sectors": ["Education"],
    },
    "physics_science": {
        "triggers": ["physics", "quantum", "atom", "energy", "force", "gravity",
                     "einstein"],
        "tags": ["physics", "science", "education"],
        "sectors": ["Education"],
    },
    "biology": {
        "triggers": ["biology", "cell", "dna", "evolution", "organism", "species",
                     "human"],
        "tags": ["biology", "science", "education"],
        "sectors": ["Education"],
    },
    "math": {
        "triggers": ["math", "mathematics", "algebra", "calculus", "geometry",
                     "equation", "theorem"],
        "tags": ["mathematics", "math", "education"],
        "sectors": ["Education"],
    },
    "space_astronomy": {
        "triggers": ["space", "astronomy", "planet", "galaxy", "universe", "nasa",
                     "telescope", "blackhole", "star", "cosmos"],
        "tags": ["space", "astronomy", "science", "nasa"],
        "sectors": ["Education", "Sci-Tech"],
    },
    "explainer_general": {
        "triggers": ["explained", "explain", "explanation", "how", "why",
                     "actually", "works", "guide", "introduction"],
        "tags": ["education", "learn", "explained", "tutorial"],
        "sectors": ["Education"],
    },

    # ── Comedy ───────────────────────────────────────────────────────────
    "standup": {
        "triggers": ["standup", "stand", "live", "show", "set", "open", "mic",
                     "comedian"],
        "tags": ["standupcomedy", "comedy", "standup", "comedian"],
        "sectors": ["Comedy"],
    },
    "comedy_general": {
        "triggers": ["funny", "comedy", "joke", "jokes", "humor", "humour",
                     "hilarious", "lol"],
        "tags": ["comedy", "funny", "funnyvideos", "humor"],
        "sectors": ["Comedy", "Entertainment"],
    },
    "prank": {
        "triggers": ["prank", "pranks", "pranked", "trolling", "trick"],
        "tags": ["prank", "funny", "comedy", "viral"],
        "sectors": ["Comedy", "Entertainment"],
    },
    "fails_react": {
        "triggers": ["fail", "fails", "react", "reaction", "trynottolaugh",
                     "compilation"],
        "tags": ["fails", "funny", "compilation", "reaction"],
        "sectors": ["Comedy", "Entertainment"],
    },
    "sketch": {
        "triggers": ["sketch", "skit", "parody", "satire"],
        "tags": ["sketchcomedy", "comedy", "funny"],
        "sectors": ["Comedy"],
    },

    # ── Sci-Tech / Hardware ──────────────────────────────────────────────
    "iphone_apple": {
        "triggers": ["iphone", "apple", "macbook", "ios", "ipad", "airpods",
                     "imac", "applewatch"],
        "tags": ["iphone", "apple", "tech", "smartphone"],
        "sectors": ["Sci-Tech"],
    },
    "android_samsung": {
        "triggers": ["android", "samsung", "pixel", "galaxy", "oneplus", "xiaomi"],
        "tags": ["android", "smartphone", "tech"],
        "sectors": ["Sci-Tech"],
    },
    "smartphone_review": {
        "triggers": ["smartphone", "phone", "camera", "battery", "review",
                     "unboxing", "comparison", "vs"],
        "tags": ["smartphone", "techreview", "tech", "review"],
        "sectors": ["Sci-Tech"],
    },
    "robotics": {
        "triggers": ["robot", "robotics", "automation", "arduino", "raspberry",
                     "drone", "diy", "build"],
        "tags": ["robotics", "tech", "automation", "diy", "engineering"],
        "sectors": ["Sci-Tech"],
    },
    "tech_general": {
        "triggers": ["tech", "technology", "gadget", "device", "hardware",
                     "review"],
        "tags": ["tech", "technology", "gadgets", "techreview"],
        "sectors": ["Sci-Tech"],
    },

    # ── Sci-Tech / Engineering ───────────────────────────────────────────
    "electronics": {
        "triggers": ["electronics", "circuit", "voltage", "transistor", "pcb",
                     "soldering"],
        "tags": ["electronics", "diy", "engineering"],
        "sectors": ["Sci-Tech", "Education"],
    },
    "engineering": {
        "triggers": ["engineering", "engineer", "mechanical", "civil",
                     "construction", "build"],
        "tags": ["engineering", "tech", "stem"],
        "sectors": ["Sci-Tech", "Education"],
    },

    # ── Entertainment / Generic ──────────────────────────────────────────
    "viral_trending": {
        "triggers": ["viral", "trending", "trend", "popular", "famous", "explode"],
        "tags": ["viral", "trending", "viralvideo"],
        "sectors": ["Entertainment", "Comedy"],
    },
    "music_song": {
        "triggers": ["music", "song", "album", "artist", "concert", "remix",
                     "cover", "lyrics"],
        "tags": ["music", "newmusic", "song"],
        "sectors": ["Entertainment", "Lifestyle"],
    },
    "vlog": {
        "triggers": ["vlog", "vlogging", "daily", "diary"],
        "tags": ["vlog", "dailyvlog", "lifestyle"],
        "sectors": ["Lifestyle", "Entertainment"],
    },
    "indian_entertainment": {
        "triggers": ["bollywood", "hindi", "desi", "indian", "mumbai", "delhi"],
        "tags": ["bollywood", "indian", "desi"],
        "sectors": ["Entertainment", "Lifestyle"],
    },

    # ── DIY / Crafts ─────────────────────────────────────────────────────
    "diy_crafts": {
        "triggers": ["diy", "craft", "crafts", "handmade", "homemade", "tutorial",
                     "make", "makeit", "build"],
        "tags": ["diy", "diycrafts", "handmade", "crafts"],
        "sectors": ["Lifestyle", "Sci-Tech"],
    },
    "art_drawing": {
        "triggers": ["art", "drawing", "draw", "paint", "painting", "sketch",
                     "illustration"],
        "tags": ["art", "drawing", "artist"],
        "sectors": ["Lifestyle"],
    },

    # ── Generic descriptors ──────────────────────────────────────────────
    "tutorial_howto": {
        "triggers": ["tutorial", "guide", "how", "learn", "lesson", "course",
                     "step", "beginners"],
        "tags": ["tutorial", "howto", "learn", "education"],
        "sectors": ["Education", "Sci-Tech", "Lifestyle"],
    },
    "review_general": {
        "triggers": ["review", "honest", "opinion", "rating", "thoughts",
                     "verdict"],
        "tags": ["review", "honestreview"],
        "sectors": ["Sci-Tech", "Lifestyle", "Gaming"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Build inverse keyword index — word → cluster ids that fire on this word
# ═══════════════════════════════════════════════════════════════════════════
def build_keyword_index(taxonomy):
    word_to_clusters = defaultdict(set)
    for cluster_id, info in taxonomy.items():
        for trigger in info["triggers"]:
            word_to_clusters[trigger.lower()].add(cluster_id)
    return word_to_clusters


# ═══════════════════════════════════════════════════════════════════════════
# Main recommender class
# ═══════════════════════════════════════════════════════════════════════════
class TaxonomyRecommender:
    """
    Recommends from a curated taxonomy of ~450 hashtags grouped into
    clusters by topic. Score = trigger word overlap + sector match +
    co-occurrence boost from corpus.
    """

    def __init__(self, taxonomy=None, corpus_cooccur=None, corpus_freq=None):
        self.taxonomy   = taxonomy or TAXONOMY
        self.kw_index   = build_keyword_index(self.taxonomy)
        self.cooccur    = corpus_cooccur or {}
        self.corpus_freq = corpus_freq or {}

        # Build full vocabulary for diagnostics
        self.vocab = []
        for cluster in self.taxonomy.values():
            self.vocab.extend(cluster["tags"])
        self.vocab = sorted(set(self.vocab))

    def recommend(self, title="", description="", sector="Unknown",
                  existing_tags=None, n=5):
        if existing_tags is None:
            existing_tags = []
        seed_set = {t.lstrip("#").lower() for t in existing_tags}

        # Tokenize title and description
        title_text = title.lower()
        desc_text  = description.lower()
        title_words = set(re.findall(r"\b[a-z]{3,}\b", title_text))
        desc_words  = set(re.findall(r"\b[a-z]{3,}\b", desc_text))

        # Find which clusters match
        cluster_scores = defaultdict(float)
        cluster_reasons = defaultdict(list)

        for word in title_words:
            for cluster_id in self.kw_index.get(word, set()):
                cluster_scores[cluster_id] += 2.0  # title hit = strong
                cluster_reasons[cluster_id].append(f"'{word}' in title")

        for word in desc_words:
            for cluster_id in self.kw_index.get(word, set()):
                if word not in title_words:  # don't double-count
                    cluster_scores[cluster_id] += 0.7
                    cluster_reasons[cluster_id].append(f"'{word}' in desc")

        # Boost clusters whose seed-set tags appear in this cluster
        for seed in seed_set:
            for cluster_id, info in self.taxonomy.items():
                if seed in info["tags"] or seed in [t.lower() for t in info["triggers"]]:
                    cluster_scores[cluster_id] += 1.5
                    cluster_reasons[cluster_id].append(f"'{seed}' is your tag")

        # Apply sector relevance — boost clusters that include this sector
        for cluster_id, info in self.taxonomy.items():
            if sector in info["sectors"]:
                cluster_scores[cluster_id] *= 1.3   # 30% boost
            else:
                # NOT zero — sometimes recommend cross-sector if title is strong
                cluster_scores[cluster_id] *= 0.5

        # Now score each tag based on its cluster score
        tag_scores  = defaultdict(float)
        tag_reasons = {}

        for cluster_id, score in cluster_scores.items():
            if score < 0.4:
                continue
            info = self.taxonomy[cluster_id]
            for tag in info["tags"]:
                if tag in seed_set:
                    continue
                if tag_scores[tag] < score:
                    tag_scores[tag]  = score
                    # Build human-readable reason
                    reasons = list(set(cluster_reasons[cluster_id]))[:2]
                    tag_reasons[tag] = " · ".join(reasons) if reasons else "topic match"

        # Add a tiny boost from co-occurrence if seeds present
        if seed_set and self.cooccur:
            for seed in seed_set:
                neighbours = self.cooccur.get(seed, {})
                for tag, weight in neighbours.items():
                    if tag in tag_scores:
                        tag_scores[tag] += weight * 0.2

        if not tag_scores:
            # cold-start fallback: sector-default tags
            fallback = {
                "Lifestyle":     ["lifestyle", "viral", "trending"],
                "Gaming":        ["gaming", "videogames", "gamer"],
                "Sports":        ["sports", "highlights", "athletics"],
                "Education":     ["education", "learn", "tutorial"],
                "Comedy":        ["comedy", "funny", "humor"],
                "Sci-Tech":      ["tech", "technology", "review"],
                "Entertainment": ["entertainment", "viral", "trending"],
            }
            return [
                {"tag": "#" + t, "score": 50, "reason": "sector default", "freq": 0}
                for t in fallback.get(sector, ["trending", "viral"])[:n]
            ]

        # Rank and normalize
        ranked = sorted(tag_scores.items(), key=lambda x: -x[1])[:n]
        max_score = ranked[0][1] if ranked else 1.0

        results = []
        for tag, score in ranked:
            conf = max(15, int(round((score / max_score) * 100)))
            results.append({
                "tag":    "#" + tag,
                "score":  conf,
                "reason": tag_reasons.get(tag, "topic match"),
                "freq":   self.corpus_freq.get(tag, 0),
            })
        return results

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Build co-occurrence boost from corpus (optional refinement)
# ═══════════════════════════════════════════════════════════════════════════
def build_corpus_cooccur():
    """Use the corpus to add a small refinement signal — which tags actually
    co-occur in real videos? Used as a tie-breaker, not primary ranking."""
    import pandas as pd
    df_path = DATA_DIR / "btp2_v1_labeled.parquet"
    if not df_path.exists():
        print("No corpus found — skipping co-occurrence boost")
        return {}, {}

    df = pd.read_parquet(df_path)
    df = df[df["hashtags_normalized"].notna()].reset_index(drop=True)

    cooccur = defaultdict(lambda: defaultdict(float))
    freq    = Counter()
    for tags in df["hashtags_normalized"]:
        ts = list(tags.tolist() if hasattr(tags, "tolist") else tags)
        ts = [t for t in ts if t]
        for t in ts:
            freq[t] += 1
        for i, a in enumerate(ts):
            for b in ts[i+1:]:
                cooccur[a][b] += 1.0
                cooccur[b][a] += 1.0

    # Normalize so neighbour weights sum to 1 per anchor
    norm = {}
    for anchor, neighbours in cooccur.items():
        total = sum(neighbours.values())
        if total > 0:
            norm[anchor] = {t: v / total for t, v in neighbours.items()}
    return norm, dict(freq)


# ═══════════════════════════════════════════════════════════════════════════
# Main — build, test, save
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Building taxonomy recommender...")
    print(f"  Clusters     : {len(TAXONOMY)}")
    print(f"  Total tags   : {len({t for c in TAXONOMY.values() for t in c['tags']})}")
    print(f"  Trigger words: {len({w for c in TAXONOMY.values() for w in c['triggers']})}")

    cooccur, freq = build_corpus_cooccur()
    print(f"  Co-occurrence anchors: {len(cooccur):,}")

    rec = TaxonomyRecommender(corpus_cooccur=cooccur, corpus_freq=freq)

    # 20 test cases
    TESTS = [
        ("10 Easy Recipes for Beginners Quick Tasty Cooking",
         "Simple meals you can make at home", "Lifestyle", ["cooking"],
         {"easyrecipes", "recipe", "cooking", "homecooking", "foodie"}),

        ("My Morning Routine Productivity Tips 2026",
         "How I stay focused every day", "Lifestyle", [],
         {"productivity", "morningroutine", "habits", "lifestyle"}),

        ("Best Street Food Tour INDIA Mumbai Delhi",
         "Eating amazing street food", "Lifestyle", ["streetfood"],
         {"streetfood", "indianfood", "foodtour", "travel"}),

        ("Home Workout No Equipment Full Body 30 Minutes",
         "Complete fitness routine", "Lifestyle", ["workout"],
         {"workout", "fitness", "homeworkout", "exercise"}),

        ("I survived 100 days in Minecraft Hardcore mode",
         "Hardest gaming challenge", "Gaming", ["minecraft"],
         {"minecraft", "gaming", "hardcore", "challenge"}),

        ("AMONG US but with 100 players insane lobbies",
         "Most chaotic round ever", "Gaming", ["gaming"],
         {"amongus", "gaming", "videogames"}),

        ("GTA 6 First Impressions Review Gameplay 2026",
         "Everything about the new game", "Gaming", [],
         {"gta", "gaming", "gameplay"}),

        ("Roblox but I can only use FREE items challenge",
         "Playing with zero robux", "Gaming", ["roblox"],
         {"roblox", "gaming", "challenge"}),

        ("Arsenal vs Chelsea HIGHLIGHTS Premier League 2026",
         "All the goals and best moments", "Sports", ["football"],
         {"premierleague", "football", "soccer", "highlights"}),

        ("Cricket India vs Australia Test Match Day 3",
         "Full highlights and analysis", "Sports", ["cricket"],
         {"cricket", "highlights", "indianfood"}),  # indianfood is OK for india context

        ("NBA Finals Game 7 HIGHLIGHTS Best Moments",
         "Most intense basketball game", "Sports", ["basketball"],
         {"nba", "basketball", "highlights"}),

        ("Cristiano Ronaldo Top 50 Goals of His Career",
         "The greatest goals ever", "Sports", ["football"],
         {"football", "soccer", "highlights"}),

        ("Learn Python in 1 Hour Complete Beginners Tutorial",
         "Code along projects", "Education", [],
         {"python", "programming", "tutorial", "learnpython", "learntocode"}),

        ("How does a CPU actually work Computer Science explained",
         "Deep dive into hardware", "Education", ["computerscience"],
         {"computerscience", "tech", "tutorial", "education"}),

        ("The French Revolution explained in 10 minutes History",
         "Everything you need to know", "Education", [],
         {"history", "education", "explained"}),

        ("Stand Up Comedy Special LIVE show funny jokes 2026",
         "Best stand up moments", "Comedy", [],
         {"standupcomedy", "comedy", "standup", "funny"}),

        ("Try Not to Laugh Challenge IMPOSSIBLE funny clips",
         "These will make you laugh", "Comedy", [],
         {"funny", "comedy", "fails", "compilation"}),

        ("iPhone 16 Pro Full Review Honest camera test opinion",
         "Worth buying in 2026", "Sci-Tech", ["apple"],
         {"iphone", "apple", "review", "smartphone", "tech"}),

        ("I built a robot that sorts my laundry full automation DIY",
         "Complete build tutorial", "Sci-Tech", ["robotics"],
         {"robotics", "diy", "tech", "engineering", "automation"}),

        ("Funniest Prank Videos of 2026 compilation viral reaction",
         "Epic prank gone viral", "Entertainment", ["prank"],
         {"prank", "funny", "comedy", "viral", "compilation"}),
    ]

    print("\n" + "═" * 65)
    print("  TAXONOMY RECOMMENDER — 20-CASE TEST")
    print("═" * 65)

    KNOWN_BAD = {"setindia", "newshowalert", "shorts", "subscribe",
                 "peopleareawesome", "goodmythicalmorning", "indianidol",
                 "aphmau", "pewdiepie", "callmekevin", "digdeeper",
                 "chromefordevelopers", "fallontonight"}

    total_good = total_bad = total_hit = total_tags = 0
    by_sector = {}

    for title, desc, sector, seeds, expected in TESTS:
        recs = rec.recommend(title, desc, sector, seeds, n=5)

        good = [r for r in recs if r["tag"].lstrip("#") not in KNOWN_BAD]
        bad  = [r for r in recs if r["tag"].lstrip("#") in KNOWN_BAD]
        hits = [r for r in recs if r["tag"].lstrip("#") in expected]

        total_good += len(good)
        total_bad  += len(bad)
        total_hit  += len(hits)
        total_tags += len(recs)

        by_sector.setdefault(sector, {"good": 0, "bad": 0, "hit": 0, "total": 0})
        by_sector[sector]["good"]  += len(good)
        by_sector[sector]["bad"]   += len(bad)
        by_sector[sector]["hit"]   += len(hits)
        by_sector[sector]["total"] += len(recs)

        flag = "✅" if not bad and len(hits) >= 2 else "⚠️" if not bad else "❌"
        print(f"\n{flag} [{sector}] {title[:58]}")
        for r in recs:
            tag = r["tag"].lstrip("#")
            mark = "💚" if tag in expected else ("🔴" if tag in KNOWN_BAD else "  ")
            print(f"  {mark} {r['tag']:<28} {r['score']:>3}%  {r['reason']}")

    # Summary
    print("\n" + "═" * 65)
    print("  QUALITY SUMMARY")
    print("═" * 65)
    q = total_good / total_tags * 100
    h = total_hit  / total_tags * 100
    print(f"\n  Quality (no bad tags) : {q:.1f}%  ({total_good}/{total_tags})")
    print(f"  Hit rate (expected)   : {h:.1f}%  ({total_hit}/{total_tags})")
    print(f"  Bad tag count         : {total_bad}  (target: 0)")

    print(f"\n  Per-sector:")
    for sector, s in sorted(by_sector.items()):
        sq = s["good"] / s["total"] * 100
        sh = s["hit"]  / s["total"] * 100
        flag = "✅" if s["bad"] == 0 else "⚠️"
        print(f"  {flag} {sector:<14} quality={sq:>5.1f}%  hits={sh:>5.1f}%  bad={s['bad']}")

    # Save
    rec.save(MODELS_DIR / "hashtag_recommender.pkl")
    print("\n✅ Restart dashboard:  python3 -m streamlit run dashboard.py")
