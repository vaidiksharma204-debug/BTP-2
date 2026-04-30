"""
dashboard.py — BTP-2 YouTube Performance Predictor
Vaidik Sharma · 22MT10063 · IIT Kharagpur · Prof. Pabita Mitra
Run: python3 -m streamlit run dashboard.py
"""
import re, json, pickle, warnings, sys
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════
# TAXONOMY RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════
class TaxonomyRecommender:
    TAXONOMY = {
        "prank":        {"t":["prank","pranks","pranked","trolling","trick","dare","spank"],"tags":["prank","prankvideo","funny","comedy","viral","reaction"],"s":["Entertainment","Comedy"]},
        "reaction":     {"t":["reaction","react","shocked","surprised","responds","omg"],"tags":["reaction","viral","funny","shocked","reactionvideo"],"s":["Entertainment","Comedy"]},
        "comedy":       {"t":["funny","comedy","joke","humor","hilarious","lol","laugh","jokes","meme","roast"],"tags":["comedy","funny","funnyvideos","humor","hilarious","comedyvideo"],"s":["Entertainment","Comedy"]},
        "standup":      {"t":["standup","comedian","live","show","mic","special"],"tags":["standupcomedy","comedy","standup","comedian","live"],"s":["Comedy"]},
        "fails":        {"t":["fail","fails","compilation","funniest","trynottolaugh","bloopers"],"tags":["fails","funny","compilation","reaction","trynottolaugh"],"s":["Comedy","Entertainment"]},
        "viral":        {"t":["viral","trending","epic","insane","crazy","unbelievable","exposed","shocking"],"tags":["viral","trending","viralvideo","epic","mustsee"],"s":["Entertainment","Comedy","Gaming"]},
        "recipes":      {"t":["recipe","recipes","cooking","cook","kitchen","tasty","meal","food","chef","homemade","ingredients","dish","eat"],"tags":["cooking","recipe","easyrecipes","foodie","homecooking","cookingtips","mealprep"],"s":["Lifestyle"]},
        "quick_food":   {"t":["quick","easy","fast","simple","beginner","beginners","minute","minutes"],"tags":["easyrecipes","quickrecipes","easymeal","quickmeal","5minuterecipes"],"s":["Lifestyle"]},
        "indian_food":  {"t":["biryani","curry","masala","dosa","paratha","chai","paneer","tikka","naan","samosa"],"tags":["indianfood","indianrecipes","streetfood","indiancooking","desi"],"s":["Lifestyle"]},
        "street_food":  {"t":["streetfood","stall","vendor","roadside","localfood","hawker","chaat"],"tags":["streetfood","foodtour","foodie","localfood","foodstreet"],"s":["Lifestyle"]},
        "fitness":      {"t":["workout","exercise","gym","fitness","training","cardio","muscle","body","weight","reps","squat"],"tags":["fitness","workout","gymlife","homeworkout","exercise","fitnessmotivation"],"s":["Lifestyle"]},
        "yoga":         {"t":["yoga","meditation","mindfulness","stretch","breathing","wellness","calm","asana"],"tags":["yoga","meditation","wellness","selfcare","mindfulness"],"s":["Lifestyle"]},
        "productivity": {"t":["productivity","productive","habits","routine","morning","focus","discipline","tips","schedule"],"tags":["productivity","morningroutine","habits","selfimprovement","motivation","focus"],"s":["Lifestyle","Education"]},
        "motivation":   {"t":["motivation","inspire","success","mindset","goals","achieve","dream","hustle","confidence"],"tags":["motivation","mindset","success","inspiration","selfimprovement"],"s":["Lifestyle","Education"]},
        "beauty":       {"t":["makeup","beauty","skincare","lipstick","foundation","glow","routine","serum"],"tags":["makeup","beauty","skincare","makeuptutorial","beautytips","glowup"],"s":["Lifestyle"]},
        "fashion":      {"t":["fashion","outfit","style","ootd","wardrobe","dress","trend","clothes","aesthetic"],"tags":["fashion","style","ootd","outfit","fashiontips","aesthetic"],"s":["Lifestyle"]},
        "travel":       {"t":["travel","trip","tour","vacation","adventure","explore","destination","backpack","solo","abroad","country","countries","journey","passport","wanderlust"],"tags":["travel","wanderlust","travelvlog","adventure","explore","solotravel","travelguide"],"s":["Lifestyle"]},
        "budget_travel":{"t":["budget","cheap","affordable","save","money","cost","free","broke","backpacking","hostel","shoestring"],"tags":["budgettravel","solotravel","travel","backpacking","travelhacks","cheaptravel"],"s":["Lifestyle"]},
        "quit_job":     {"t":["quit","quitting","job","work","career","boss","office","corporate","freedom","resign","fired","salary"],"tags":["quitmyjob","solotravel","motivation","selfimprovement","lifestyle","digitalnomad"],"s":["Lifestyle","Entertainment"]},
        "india_travel": {"t":["india","mumbai","delhi","bangalore","kerala","goa","rajasthan","himalaya","varanasi"],"tags":["india","incredibleindia","indiatravel","desi","travelblog"],"s":["Lifestyle","Entertainment"]},
        "vlog":         {"t":["vlog","daily","diary","dayinmylife","behind","filming"],"tags":["vlog","dailyvlog","lifestyle","travelvlog","dayinmylife"],"s":["Lifestyle","Entertainment"]},
        "minecraft":    {"t":["minecraft","creeper","diamond","enderman","crafting","bedrock","nether","redstone"],"tags":["minecraft","minecraftbuilds","gaming","minecraftsurvival","minecraftpe"],"s":["Gaming"]},
        "fortnite":     {"t":["fortnite","battleroyale","victory","royale"],"tags":["fortnite","battleroyale","gaming"],"s":["Gaming"]},
        "roblox":       {"t":["roblox","robux"],"tags":["roblox","gaming"],"s":["Gaming"]},
        "amongus":      {"t":["among","imposter","crewmate","sus","emergency"],"tags":["amongus","gaming","multiplayer","imposter"],"s":["Gaming"]},
        "gta":          {"t":["gta","grandtheft","gta6","gta5","heist"],"tags":["gta","gtav","gaming","gameplay","openworld"],"s":["Gaming"]},
        "valorant":     {"t":["valorant","valo","sage","reyna","radiant","rank"],"tags":["valorant","gaming","fps","valorantclips"],"s":["Gaming"]},
        "gaming_gen":   {"t":["game","gaming","gameplay","play","stream","console","gamer","videogame"],"tags":["gaming","gameplay","videogames","gamer","pcgaming","consolegaming"],"s":["Gaming"]},
        "challenge_game":{"t":["100days","survived","survival","hardcore","challenge","speedrun","impossible","noob","only"],"tags":["gaming","challenge","survival","hardcore","gamingchallenge"],"s":["Gaming"]},
        "epl":          {"t":["arsenal","chelsea","liverpool","manchester","tottenham","spurs","premier","epl","united","everton"],"tags":["premierleague","football","soccer","epl","highlights"],"s":["Sports"]},
        "laliga":       {"t":["barcelona","barca","madrid","realmadrid","laliga","atletico"],"tags":["laliga","football","soccer","highlights"],"s":["Sports"]},
        "football":     {"t":["football","soccer","goal","goals","match","ronaldo","messi","champions","fifa","penalty"],"tags":["football","soccer","goals","highlights","footballskills"],"s":["Sports"]},
        "cricket":      {"t":["cricket","ipl","wicket","batsman","bowler","test","t20","kohli","dhoni","rohit","odi","bcci"],"tags":["cricket","ipl","cricketnews","cricketlovers","indiacricket"],"s":["Sports"]},
        "nba":          {"t":["nba","basketball","lebron","curry","lakers","warriors","dunk","finals","buzzer"],"tags":["nba","basketball","nbahighlights"],"s":["Sports"]},
        "tennis":       {"t":["tennis","wimbledon","djokovic","nadal","federer","atp","wta","slam"],"tags":["tennis","atp","wimbledon","grandslam"],"s":["Sports"]},
        "highlights":   {"t":["highlight","highlights","moments","goals","plays","clips","extended","greatest"],"tags":["highlights","sportshighlights","bestmoments","fullhighlights"],"s":["Sports"]},
        "python":       {"t":["python","pandas","numpy","django","flask","pytorch","tensorflow"],"tags":["python","coding","programming","learnpython","pythontutorial"],"s":["Education","Sci-Tech"]},
        "webdev":       {"t":["javascript","react","nodejs","html","css","webdev","frontend","typescript"],"tags":["javascript","webdev","coding","react","frontend","webdevelopment"],"s":["Education","Sci-Tech"]},
        "programming":  {"t":["programming","code","coding","developer","software","algorithm","learntocode","debug"],"tags":["programming","coding","learntocode","developer","softwareengineering"],"s":["Education","Sci-Tech"]},
        "ai_ml":        {"t":["artificial","intelligence","machine","deep","learning","neural","gpt","llm","chatgpt","gemini","openai"],"tags":["artificialintelligence","machinelearning","ai","deeplearning","chatgpt"],"s":["Education","Sci-Tech"]},
        "history":      {"t":["history","historical","ancient","medieval","war","revolution","empire","dynasty","battle","ww2","ww1"],"tags":["history","education","historyfacts","worldhistory"],"s":["Education"]},
        "science":      {"t":["physics","quantum","biology","chemistry","science","experiment","laboratory","evolution","dna"],"tags":["science","physics","biology","education","sciencefacts"],"s":["Education"]},
        "space":        {"t":["space","astronomy","planet","galaxy","universe","nasa","blackhole","star","cosmos","mars","moon"],"tags":["space","astronomy","science","nasa","spacefacts","universe"],"s":["Education","Sci-Tech"]},
        "math":         {"t":["math","mathematics","algebra","calculus","geometry","equation","theorem","statistics"],"tags":["mathematics","math","education","mathproblems","calculus"],"s":["Education"]},
        "explainer":    {"t":["explained","explain","works","guide","introduction","basics","beginners","understand","overview","simply"],"tags":["education","learn","explained","tutorial","beginner"],"s":["Education"]},
        "medical":      {"t":["doctor","medical","health","hospital","patient","medicine","diagnosis","surgery","nurse","exam","examination","disease","symptom"],"tags":["medical","health","healthcare","doctor","medicine","healthtips"],"s":["Education","Lifestyle"]},
        "iphone":       {"t":["iphone","apple","macbook","ios","ipad","airpods","applewatch","siri","macos"],"tags":["iphone","apple","tech","smartphone","ios","applefan"],"s":["Sci-Tech"]},
        "android":      {"t":["android","samsung","pixel","galaxy","oneplus","xiaomi"],"tags":["android","smartphone","tech","samsung","androidphone"],"s":["Sci-Tech"]},
        "phone_review": {"t":["smartphone","phone","camera","battery","review","unboxing","comparison","megapixel"],"tags":["smartphone","techreview","tech","review","unboxing","phonecomparison"],"s":["Sci-Tech"]},
        "robotics":     {"t":["robot","robotics","automation","arduino","raspberry","drone","diy","build","maker","servo"],"tags":["robotics","tech","automation","diy","engineering","maker","arduino"],"s":["Sci-Tech"]},
        "tech_gen":     {"t":["tech","technology","gadget","device","hardware","software","future","innovation","startup"],"tags":["tech","technology","gadgets","techreview","technews","innovation"],"s":["Sci-Tech"]},
        "diy":          {"t":["diy","craft","crafts","handmade","build","make","create","project","woodwork"],"tags":["diy","diycrafts","handmade","crafts","creative","makeit"],"s":["Lifestyle","Sci-Tech"]},
        "tutorial":     {"t":["tutorial","guide","learn","lesson","course","step","howto","walkthrough"],"tags":["tutorial","howto","learn","guide","stepbystep"],"s":["Education","Sci-Tech","Lifestyle"]},
        "review":       {"t":["review","honest","opinion","rating","thoughts","verdict","worth","recommend","pros","cons"],"tags":["review","honestreview","productreview","comparison"],"s":["Sci-Tech","Lifestyle","Gaming"]},
        "music":        {"t":["music","song","album","concert","remix","lyrics","artist","singer","band","rap","hiphop","pop"],"tags":["music","newmusic","song","artist","musicvideo","hiphop"],"s":["Entertainment"]},
        "bollywood":    {"t":["bollywood","hindi","desi","film","movie","actor","actress","ott","netflix","amazon"],"tags":["bollywood","hindi","desi","filmreview","movietrailer"],"s":["Entertainment"]},
    }
    FALLBACKS = {
        "Entertainment":["viralvideo","funny","trending","entertainment","reaction"],
        "Comedy":["comedy","funny","humor","viral","funnyvideos"],
        "Gaming":["gaming","videogames","gamer","gameplay"],
        "Sports":["sports","highlights","athletics"],
        "Education":["education","learn","tutorial","knowledge"],
        "Lifestyle":["lifestyle","viral","motivation","inspiration"],
        "Sci-Tech":["tech","technology","review","innovation"],
    }
    STOP = {"this","that","with","from","have","what","will","your","they","been","when","were",
            "than","here","most","more","some","into","just","very","well","much","many","only",
            "their","about","after","made","make","please","save","video","watch","also","even",
            "then","them","over","subscribe","like","comment","share","below","click","link",
            "check","down","every","week","weekly","month","year","time","next","while","going"}

    def __getattr__(self, name):
        defaults = {"_idx":None,"corpus_freq":{},"cooccur":{}}
        if name in defaults:
            object.__setattr__(self, name, defaults[name])
            return defaults[name]
        raise AttributeError(f"'{name}'")

    def _build_idx(self):
        idx = defaultdict(set)
        for cid, info in self.TAXONOMY.items():
            for trig in info["t"]: idx[trig.lower()].add(cid)
        return idx

    def _stem(self, words):
        stems = set()
        for w in words:
            for sfx in ("ing","tion","ked","ned","red","led","ed","er","es","s"):
                if w.endswith(sfx) and len(w)-len(sfx) >= 3: stems.add(w[:-len(sfx)])
        return words | stems

    def _idx_safe(self):
        idx = self.__dict__.get("_idx")
        if idx is None:
            idx = self._build_idx()
            object.__setattr__(self, "_idx", idx)
        return idx

    def recommend(self, title="", description="", sector="Unknown", existing_tags=None, n=5):
        idx = self._idx_safe()
        if existing_tags is None: existing_tags = []
        seed_set = {t.lstrip("#").lower() for t in existing_tags}
        t_raw = set(re.findall(r"[a-z]{3,}", title.lower())) - self.STOP
        d_raw = set(re.findall(r"[a-z]{3,}", description.lower())) - self.STOP
        t_words = self._stem(t_raw); d_words = self._stem(d_raw)
        cluster_data = {}
        for cid, info in self.TAXONOMY.items():
            triggers = set(info["t"])
            s1=len(triggers&t_raw); s2=len(triggers&t_words)-s1
            s3=len(triggers&d_raw)
            s4=sum(1 for sd in seed_set if sd in info["tags"] or sd in triggers)
            s5=1.3 if sector in info["s"] else 0.25
            raw=(s1*3.0+s2*1.2+s3*0.7+s4*1.5)*s5
            if raw>0.01: cluster_data[cid]={"raw":raw,"s1":s1,"s2":s2,"s3":s3,"s4":s4}
        if not cluster_data:
            tags_=self.FALLBACKS.get(sector,["viral","trending"])[:n]
            return [{"tag":"#"+t,"score":30+i*3,"reason":"sector default","freq":0} for i,t in enumerate(tags_)]
        tag_data={}
        for cid,ci in cluster_data.items():
            for tag in self.TAXONOMY[cid]["tags"]:
                if tag in seed_set: continue
                tw=set(re.findall(r"[a-z]+",tag))
                title_ov=len(tw&t_words)/max(len(tw),1); desc_ov=len(tw&d_words)/max(len(tw),1)
                tag_len_pen=max(0,(len(tag)-14)*0.015)
                final=ci["raw"]*(1.0+title_ov*1.3+desc_ov*0.4-tag_len_pen)
                if tag not in tag_data or final>tag_data[tag]["score"]:
                    parts=[]
                    if ci["s1"]>0 or title_ov>0.3: parts.append("title match")
                    if ci["s3"]>0 or desc_ov>0.3: parts.append("desc match")
                    if sector in self.TAXONOMY[cid]["s"]: parts.append("sector trend")
                    if ci["s4"]>0: parts.append("seed co-occurs")
                    tag_data[tag]={"score":final,"reason":" · ".join(parts) or "topic cluster","title_ov":title_ov}
        if not tag_data:
            tags_=self.FALLBACKS.get(sector,["viral"])[:n]
            return [{"tag":"#"+t,"score":30,"reason":"sector default","freq":0} for t in tags_]
        ranked=sorted(tag_data.items(),key=lambda x:-x[1]["score"])[:n]
        raw_vals=np.array([v["score"] for _,v in ranked],dtype=float)
        std_=raw_vals.std()
        if std_>0: exp_v=np.exp((raw_vals-raw_vals.max())/(std_*0.9+1e-6))
        else: exp_v=np.linspace(1.0,0.4,len(raw_vals))
        mn,mx_=exp_v.min(),exp_v.max()
        norm=(exp_v-mn)/(mx_-mn) if mx_>mn else np.linspace(1,0,len(exp_v))
        scores_pct=(norm*54+36).astype(int)
        for i,(tag,d) in enumerate(ranked):
            if d["title_ov"]>0.5: scores_pct[i]=min(91,scores_pct[i]+9)
            elif d["title_ov"]==0 and "title" not in d["reason"]: scores_pct[i]=max(28,scores_pct[i]-10)
        return [{"tag":"#"+tag,"score":int(scores_pct[i]),"reason":d["reason"],"freq":getattr(self,"corpus_freq",{}).get(tag,0)}
                for i,(tag,d) in enumerate(ranked)]

    def sector_scores(self, title="", description="", hashtags=None):
        idx=self._idx_safe()
        if hashtags is None: hashtags=[]
        seed_set={t.lstrip("#").lower() for t in hashtags}
        t_words=self._stem(set(re.findall(r"[a-z]{3,}",title.lower()))-self.STOP)
        d_words=self._stem(set(re.findall(r"[a-z]{3,}",description.lower()))-self.STOP)
        h_words=self._stem(set(re.findall(r"[a-z]{3,}"," ".join(seed_set))))
        sector_raw=defaultdict(float); total_sig=0.0
        for cid,info in self.TAXONOMY.items():
            triggers=set(info["t"])
            score=len(triggers&t_words)*3.0+len(triggers&d_words)*1.0+len(triggers&h_words)*1.5
            if score<0.5: continue
            for s in info["s"]: sector_raw[s]+=score
            total_sig+=score
        ALL=["Comedy","Education","Entertainment","Gaming","Lifestyle","Sci-Tech","Sports"]
        if total_sig<0.5: return {s:round(100.0/7,1) for s in ALL}
        total=sum(sector_raw.values()) or 1e-9
        return {s:round(sector_raw.get(s,0)/total*100,1) for s in ALL}

# ══════════════════════════════════════════════════════════════════════════
# TITLE / DESCRIPTION EXAMPLES (reference only)
# ══════════════════════════════════════════════════════════════════════════
TITLE_DESC_EXAMPLES = [
    {
        "sector": "🎮 Gaming",
        "title": "I Survived 100 Days in Minecraft Hardcore Mode",
        "desc": "The ultimate Minecraft hardcore challenge — 100 days of survival, building, and exploring. Will I make it to day 100? Subscribe for daily gaming content!",
        "why": "Challenge format + specific game + number in title = strong Gaming HIGH signals"
    },
    {
        "sector": "✈️ Lifestyle / Travel",
        "title": "I Quit My Job and Traveled 15 Countries in 6 Months on a Budget",
        "desc": "Hidden gems nobody talks about + honest solo travel tips for beginners. Save this before your next trip! Subscribe for weekly travel vlogs and budget hacks.",
        "why": "Emotional hook + number + budget keyword = good Lifestyle MID-HIGH signals"
    },
    {
        "sector": "📚 Education",
        "title": "Learn Python in 1 Hour — Complete Beginners Tutorial 2026",
        "desc": "Full Python course for absolute beginners. Code along with real projects — variables, functions, loops, and a final project. Subscribe for weekly programming tutorials!",
        "why": "High-demand keyword + specific duration + beginner = Education HIGH signals"
    },
    {
        "sector": "😂 Comedy",
        "title": "Stand Up Comedy Special LIVE — Funny Jokes Crowd Reactions 2026",
        "desc": "My first ever live comedy special! Full show with crowd reactions and unscripted moments. Like and subscribe if you laughed — more specials coming soon!",
        "why": "Live + crowd reaction + CTA = Comedy engagement signals"
    },
    {
        "sector": "🔧 Sci-Tech",
        "title": "iPhone 16 Pro Honest Review — Camera Test vs Samsung Galaxy S25",
        "desc": "In-depth iPhone 16 Pro camera comparison. Real-world tests, battery life, and is it worth upgrading from the 15 Pro? Honest pros and cons.",
        "why": "Comparison format + specific models + 'honest' = Sci-Tech review HIGH signals"
    },
    {
        "sector": "⚽ Sports",
        "title": "Arsenal vs Chelsea HIGHLIGHTS | Premier League 2026 | All Goals",
        "desc": "Extended highlights and all goals from today's Arsenal vs Chelsea Premier League match. Best moments, analysis, and player ratings.",
        "why": "Match-specific + highlights keyword + clubs = Sports viral format"
    },
]

# ══════════════════════════════════════════════════════════════════════════
# SECTOR PEAK REASONS
# ══════════════════════════════════════════════════════════════════════════
SECTOR_PEAK_REASONS = {
    "Gaming":      "Gaming audiences (18–35) are most active post-school/work. Streaming culture drives evening peaks (8–11 PM local). Weekday evenings > weekends for engagement rate.",
    "Education":   "Education viewers study in bursts: mornings (8–11 AM) and afternoon sessions (3–6 PM). Mon–Wed uploads maximize week-start motivation.",
    "Sports":      "Sports videos spike after matches end. Immediate post-match uploads get highest engagement. Weekends have volume but weekdays have better engagement ratios.",
    "Comedy":      "Comedy peaks during lunchtime (12–2 PM) and late-night wind-down (9 PM–midnight). People browse humor when relaxing.",
    "Lifestyle":   "Lifestyle/travel: morning inspiration (7–9 AM) and evening planning (8–10 PM). Thursday captures pre-weekend mindset shift.",
    "Sci-Tech":    "Tech audiences watch during work hours (10 AM–6 PM) as part of learning routines. Monday–Wednesday uploads get more algorithm push.",
    "Entertainment":"Entertainment peaks Friday evenings and weekends — leisure time. Weekday uploads still get algorithmic discovery before weekend rush.",
    "All":         "Overall YouTube peak: 8 PM local on weekdays. Thursday uploads get the longest discovery window (Fri–Sun organic spread).",
}

COUNTRY_UTC = {
    "India (+5:30)":5.5,"United States (-5)":-5.0,"United Kingdom (0)":0.0,
    "Australia (+10)":10.0,"Japan (+9)":9.0,"Germany (+1)":1.0,
    "Brazil (-3)":-3.0,"UAE (+4)":4.0,"Singapore (+8)":8.0,"Other (UTC)":0.0,
}
COUNTRY_ABBR = {
    "India (+5:30)":"IST","United States (-5)":"EST","United Kingdom (0)":"GMT",
    "Australia (+10)":"AEST","Japan (+9)":"JST","Germany (+1)":"CET",
    "Brazil (-3)":"BRT","UAE (+4)":"GST","Singapore (+8)":"SGT","Other (UTC)":"UTC",
}

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="BTP-2 · YouTube Predictor", page_icon="▶",
                   layout="wide", initial_sidebar_state="collapsed")

NAVY="#0F1B3C"; TEAL="#0D9488"; AMBER="#F59E0B"; RED="#E11D48"; PALE="#CCFBF1"; LIGHT="#E8F4F2"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Plus Jakarta Sans',sans-serif!important;background:#F8FAFB!important;}}
.stApp{{background:#F8FAFB!important;}}
.dash-header{{background:linear-gradient(135deg,{NAVY} 0%,#162d5e 60%,#1a3a72 100%);border-radius:16px;padding:1.4rem 2rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1.2rem;box-shadow:0 4px 24px rgba(15,27,60,.15);}}
.dash-header h1{{color:white;font-size:1.4rem;font-weight:700;margin:0;}}
.dash-header p{{color:{PALE};font-size:.78rem;margin:.2rem 0 0;font-family:'JetBrains Mono',monospace;}}
.model-badge{{display:inline-block;background:{NAVY};color:{PALE};border-radius:6px;padding:.18rem .65rem;font-size:.7rem;font-weight:600;font-family:'JetBrains Mono',monospace;margin-bottom:.8rem;}}
.kpi-card{{background:white;border-radius:12px;padding:1rem 1.2rem;border:1px solid #E2E8F0;box-shadow:0 1px 6px rgba(0,0,0,.04);}}
.kpi-label{{font-size:.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.09em;margin-bottom:.3rem;}}
.kpi-value{{font-size:1.65rem;font-weight:700;color:{NAVY};line-height:1;}}
.kpi-sub{{font-size:.76rem;color:#94A3B8;margin-top:.2rem;}}
.tier-box{{border-radius:14px;padding:1.4rem 1.6rem;color:white;margin-bottom:1rem;}}
.tier-name{{font-size:2.4rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;margin:.2rem 0;}}
.hash-pill{{display:inline-block;background:{LIGHT};color:{TEAL};border:1.5px solid {TEAL}55;border-radius:999px;padding:.22rem .8rem;font-size:.81rem;font-weight:600;font-family:'JetBrains Mono',monospace;margin:.15rem;}}
.sec-label{{font-size:.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem;}}
.insight-bar{{background:{LIGHT};border-left:3px solid {TEAL};border-radius:0 10px 10px 0;padding:.65rem 1rem;font-size:.82rem;color:#0f4a40;margin:.4rem 0;}}
.warn-bar{{background:#FFF7ED;border-left:3px solid {AMBER};border-radius:0 10px 10px 0;padding:.65rem 1rem;font-size:.82rem;color:#7c3a00;margin:.4rem 0;}}
.danger-bar{{background:#FFF1F2;border-left:3px solid {RED};border-radius:0 10px 10px 0;padding:.65rem 1rem;font-size:.82rem;color:#7f0020;margin:.4rem 0;}}
.example-card{{background:white;border-radius:10px;border:1px solid #E2E8F0;padding:.85rem 1rem;margin:.4rem 0;}}
.example-sector{{font-size:.72rem;font-weight:700;color:{TEAL};text-transform:uppercase;letter-spacing:.07em;}}
.example-title{{font-size:.88rem;font-weight:600;color:{NAVY};margin:.2rem 0;}}
.example-desc{{font-size:.78rem;color:#64748B;margin:.15rem 0;}}
.example-why{{font-size:.74rem;color:#94A3B8;font-style:italic;margin-top:.25rem;}}
.stButton>button{{background:{TEAL}!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:600!important;padding:.6rem 1.4rem!important;width:100%!important;}}
.stButton>button:hover{{background:#0a7a6e!important;}}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{{border-radius:9px!important;border-color:#E2E8F0!important;font-family:'Plus Jakarta Sans',sans-serif!important;}}
div.stProgress>div>div>div{{background:{TEAL}!important;border-radius:999px!important;}}
.dash-footer{{text-align:center;color:#CBD5E1;font-size:.74rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #E2E8F0;font-family:'JetBrains Mono',monospace;}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS + LOADERS
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR=Path("data"); MODELS_DIR=Path("models")
SECTORS=["Comedy","Education","Entertainment","Gaming","Lifestyle","Sci-Tech","Sports"]
DOW_NAMES=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
PEAK_WINDOW=(14,22)
TIER_COLORS={"HIGH":TEAL,"MID":AMBER,"LOW":RED}
TIER_EMOJI={"HIGH":"🚀","MID":"📈","LOW":"📉"}
TIER_VIEWS={"HIGH":"90k–300k","MID":"20k–80k","LOW":"500–15k"}
TIER_ER={"HIGH":"4.5–9%","MID":"2–4.5%","LOW":"0.3–2%"}
INV_LABEL={0:"HIGH",1:"LOW",2:"MID"}

@st.cache_resource
def load_xgboost():
    m=xgb.XGBClassifier(); m.load_model(MODELS_DIR/"xgboost_engagement.json"); return m

@st.cache_resource
def load_recommender():
    p=MODELS_DIR/"hashtag_recommender.pkl"
    if not p.exists(): return TaxonomyRecommender()
    try:
        with open(p,"rb") as f: obj=pickle.load(f)
        return obj if isinstance(obj,TaxonomyRecommender) else TaxonomyRecommender()
    except Exception: return TaxonomyRecommender()

@st.cache_data
def load_meta():
    p=DATA_DIR/"dataset_meta.json"; return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_config():
    p=DATA_DIR/"feature_config.json"
    return json.loads(p.read_text()) if p.exists() else {"xgboost_features":[]}

@st.cache_data
def load_df():
    p=DATA_DIR/"btp2_v1_labeled.parquet"; return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data
def get_sector_best_utc(df, sector):
    if df.empty or "upload_hour_utc" not in df.columns or "engagement_rate" not in df.columns:
        return 14
    sub=df if sector=="All" else (df[df["sector"]==sector] if "sector" in df.columns else df)
    if sub.empty: return 14
    return int(sub.groupby("upload_hour_utc")["engagement_rate"].mean().idxmax())

# ══════════════════════════════════════════════════════════════════════════
# FEATURE PREP
# ══════════════════════════════════════════════════════════════════════════
def prep_features(inp, features):
    row={f:0 for f in features}
    t=inp.get("title",""); d=inp.get("desc","")
    row["channel_subscribers"]=inp.get("subs",100000)
    row["channel_age_days"]=int(inp.get("age_yr",4)*365)
    row["channel_total_videos"]=200
    row["title_length_chars"]=len(t); row["title_word_count"]=len(t.split())
    row["description_length_chars"]=len(d); row["description_word_count"]=len(d.split())
    row["title_has_number"]=int(bool(re.search(r"\d",t)))
    row["title_has_question"]=int("?" in t)
    row["title_emoji_count"]=len(re.findall(r"[\U0001F300-\U0001F9FF]",t))
    letters=[c for c in t if c.isalpha()]
    row["title_caps_ratio"]=sum(c.isupper() for c in letters)/max(len(letters),1)
    row["description_url_count"]=len(re.findall(r"https?://",d))
    cta=re.findall(r"\b(subscribe|like|comment|share|follow|click|watch)\b",(t+" "+d).lower())
    row["cta_word_count"]=len(cta); row["cta_presence"]=int(bool(cta))
    try:
        from textblob import TextBlob
        pt=TextBlob(t).sentiment; pd_=TextBlob(d).sentiment
        row["title_sentiment_polarity"]=pt.polarity; row["title_subjectivity"]=pt.subjectivity
        row["description_sentiment_polarity"]=pd_.polarity; row["description_subjectivity"]=pd_.subjectivity
        row["combined_sentiment"]=(pt.polarity+pd_.polarity)/2
    except Exception: pass
    etags=inp.get("etags",[]); n=len(etags)
    row["hashtag_count"]=n; row["hashtag_zero"]=int(n==0)
    row["hashtag_optimal"]=int(0<=n<=4); row["hashtag_spam"]=int(n>10)
    row["hashtag_quality_score"]=0.0
    row["thumb_brightness"]=0.37; row["thumb_contrast"]=0.23; row["thumb_saturation"]=0.38
    row["upload_hour_utc"]=inp.get("hour_utc",14); row["upload_hour_local"]=inp.get("hour_local",19)
    row["upload_dow"]=inp.get("dow",3); row["upload_month"]=inp.get("month",4)
    row["upload_week_of_year"]=inp.get("week",17)
    row["is_peak_local"]=int(PEAK_WINDOW[0]<=inp.get("hour_local",19)<=PEAK_WINDOW[1])
    row["upload_time_bucket_enc"]=3
    try:
        words=re.findall(r"[A-Za-z]+",d); sents=[s for s in re.split(r"[.!?]+",d) if s.strip()]
        if words and sents:
            def sv(w):
                w=w.lower()
                if w.endswith("e") and len(w)>2: w=w[:-1]
                return max(1,len(re.findall(r"[aeiouy]+",w)))
            syls=sum(sv(w) for w in words)
            row["flesch_readability"]=max(0,min(121,206.835-1.015*(len(words)/len(sents))-84.6*(syls/len(words))))
    except Exception: row["flesch_readability"]=0
    row["duration_seconds"]=inp.get("dur_sec",600)
    X=pd.DataFrame([row])[features]
    for c in ["title_has_number","title_has_question","cta_presence","hashtag_optimal","hashtag_zero","hashtag_spam","is_peak_local"]:
        if c in X.columns: X[c]=X[c].fillna(0).astype(int)
    X["flesch_readability"]=X["flesch_readability"].fillna(0).clip(0,121)
    X["duration_seconds"]=X["duration_seconds"].fillna(600)
    X["title_caps_ratio"]=X["title_caps_ratio"].fillna(0)
    return X

# ══════════════════════════════════════════════════════════════════════════
# REASONING ENGINE
# ══════════════════════════════════════════════════════════════════════════
def generate_reasoning(inp, tier, probs, features, importances, best_utc_sector, df_full):
    t=inp.get("title",""); d=inp.get("desc","")
    subs=inp.get("subs",0); age_yr=inp.get("age_yr",0)
    hloc=inp.get("hour_local",14); dow=inp.get("dow",3)
    etags=inp.get("etags",[]); dur_sec=inp.get("dur_sec",600)
    hour_utc=inp.get("hour_utc",14); offset=inp.get("offset",0)
    country=inp.get("country","India (+5:30)")
    dur_min=dur_sec//60; title_len=len(t); title_words=len(t.split())
    has_number=bool(re.search(r"\d",t))
    caps_ratio=sum(c.isupper() for c in t if c.isalpha())/max(sum(c.isalpha() for c in t),1)
    n_tags=len(etags); peak=PEAK_WINDOW[0]<=hloc<=PEAK_WINDOW[1]
    desc_words=len(d.split()) if d else 0
    max_p=probs.max(); second_p=sorted(probs)[-2]; gap=(max_p-second_p)*100
    abbr=COUNTRY_ABBR.get(country,"local")
    best_local=int((best_utc_sector+offset)%24)
    best_period="PM" if best_local>=12 else "AM"
    best_disp=best_local%12 or 12

    positives=[]; warnings_=[]; improvements=[]

    # Borderline
    if gap<8:
        warnings_.append(f"⚡ <b>Borderline prediction</b> — only {gap:.0f}% confidence gap to next tier "
                        f"({max_p*100:.0f}% {tier} vs {second_p*100:.0f}% next). A single change could flip this.")

    # Channel authority
    ch_total_imp=(importances.get("channel_total_videos",0.089)+importances.get("channel_subscribers",0.067)+importances.get("channel_age_days",0.062))
    ch_score=min(1.0,(subs/500000)*0.4+min(age_yr/8,1)*0.35+0.25)
    if ch_score>=0.7:
        positives.append(f"✅ <b>Strong channel authority</b> ({subs/1000:.0f}K subs, {age_yr:.0f}yr) — top predictor group ({ch_total_imp*100:.0f}% weight) scores HIGH")
    elif ch_score>=0.4:
        warnings_.append(f"⚠️ <b>Medium channel authority</b> ({subs/1000:.0f}K subs, {age_yr:.0f}yr) — the top 3 XGBoost features account for {ch_total_imp*100:.0f}% of model weight, scoring MEDIUM range, biasing toward MID/LOW.")
        improvements.append(f"📈 <b>Grow to 500K+ subs</b> to cross the MID→HIGH threshold. Consistency rewards channel_total_videos (importance: 0.089).")
    else:
        warnings_.append(f"⚠️ <b>Low channel authority = primary {tier} driver</b> ({subs/1000:.0f}K subs, {age_yr:.1f}yr) — even with perfect content, small channels score LOW on {ch_total_imp*100:.0f}%-weighted authority group.")
        improvements.append(f"📈 <b>Channel growth is the #1 lever</b>: reach 100K subs for consistent MID predictions. Content quality alone cannot overcome channel size limitations in this model.")

    # Title
    if 40<=title_len<=70: positives.append(f"✅ <b>Good title length</b> ({title_len} chars) — optimal 40–70 char range")
    elif title_len>80:
        warnings_.append(f"⚠️ <b>Title too long</b> ({title_len} chars) — r=−0.28 with engagement")
        improvements.append("✂️ <b>Shorten title to 40–70 chars</b> — strongest negative text signal")
    if title_words>12:
        warnings_.append(f"⚠️ <b>Too many title words</b> ({title_words}) — r=−0.30, strongest text predictor")
        improvements.append("✂️ <b>Keep title under 10 words</b> — #1 text feature by importance")
    elif title_words<=10: positives.append(f"✅ <b>Concise title</b> ({title_words} words) — optimal range")
    if has_number: positives.append("✅ <b>Title has a number</b> — list titles get ~20% higher CTR")
    if caps_ratio>0.5:
        warnings_.append(f"⚠️ <b>Too many capitals</b> ({caps_ratio*100:.0f}%) — spam signal")
        improvements.append("🔡 <b>Use title case</b> — ALL CAPS reduces engagement")

    # Timing — sector-specific best UTC
    if peak:
        positives.append(f"✅ <b>Peak upload window</b> — {hour_utc:02d}:00 UTC = {hloc:02d}:00 {abbr} "
                        f"is within 14:00–22:00 {abbr} audience peak. "
                        f"Sector-optimal slot: {best_utc_sector:02d}:00 UTC = {best_disp:02d}:00 {best_period} {abbr}.")
    else:
        improvements.append(f"🕐 <b>Change upload to {best_utc_sector:02d}:00 UTC = {best_disp:02d}:00 {best_period} {abbr}</b> — "
                           f"dataset-optimal for your sector. Current {hour_utc:02d}:00 UTC = {hloc:02d}:00 {abbr} "
                           f"is outside peak window. Off-peak = fewer first-24h views = algorithm deprioritization.")

    dow_names_=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if dow==3: positives.append("✅ <b>Thursday upload</b> — highest avg engagement day (3.58% vs 3.2% dataset avg). Gets Fri–Sun discovery window.")
    elif dow in [1,2]: positives.append(f"✅ <b>{dow_names_[dow]}</b> — 2nd/3rd best day in dataset")
    elif dow in [5,6]:
        warnings_.append(f"⚠️ <b>{dow_names_[dow]} upload</b> — weekend shows slightly lower engagement rate")
        improvements.append("📅 <b>Shift to Thursday</b> for best measured engagement. Thu gets full Fri–Mon discovery window.")

    # Hashtags
    if n_tags==0: positives.append("✅ <b>Zero hashtags</b> — highest mean engagement (3.636%) in dataset. BTP-2 key finding.")
    elif 1<=n_tags<=4: positives.append(f"✅ <b>Optimal hashtag count</b> ({n_tags}) — sweet spot (mean ER: 3.2–3.4%)")
    elif n_tags>10:
        warnings_.append(f"❌ <b>Hashtag spam</b> ({n_tags}) — mean ER drops to 1.35% with >10 tags")
        improvements.append("🏷️ <b>Reduce to max 4 hashtags</b>")

    # Duration
    if 5<=dur_min<=15: positives.append(f"✅ <b>Optimal video length</b> ({dur_min} min) — 5–15 min sweet spot")
    elif dur_min>30: warnings_.append(f"⚠️ <b>Long video</b> ({dur_min} min) — r=−0.07, lower % completion reduces engagement rate")

    # All-good but LOW
    all_ok=(40<=title_len<=70 and n_tags<=4 and 5<=dur_min<=20)
    if tier=="LOW" and all_ok and ch_score<0.4:
        positives.append(f"ℹ️ <b>Why LOW despite good content signals?</b> Channel authority "
                        f"({ch_total_imp*100:.0f}% weight) is the bottleneck — {subs/1000:.0f}K subs × {age_yr:.0f}yr "
                        f"scores LOW on the authority group. At 500K+ subs, same content → MID–HIGH prediction.")

    return positives, warnings_, improvements

# ══════════════════════════════════════════════════════════════════════════
# LOAD ALL
# ══════════════════════════════════════════════════════════════════════════
xgb_model   = load_xgboost()
recommender = load_recommender()
meta        = load_meta()
config      = load_config()
FEATURES    = config.get("xgboost_features",[])
df_full     = load_df()
IMP_DICT    = dict(zip(FEATURES, xgb_model.feature_importances_)) if FEATURES else {}

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <div style="font-size:2.2rem">▶</div>
  <div><h1>YouTube Performance Predictor</h1>
  <p>BTP-2 · Vaidik Sharma · 22MT10063 · IIT Kharagpur · Supervisor: Prof. Pabita Mitra</p></div>
</div>""", unsafe_allow_html=True)

t1,t2,t3,t4,t5 = st.tabs([
    "🔮 Predict & Analyze",
    "🏷️ Hashtag & Sector",
    "⏰ Timing",
    "🔬 Model Insights",
    "📊 Dataset EDA",
])

# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# TAB 1 · PREDICT & ANALYZE
# ══════════════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="model-badge">🤖 XGBoost Engagement Classifier · 34 features · 69.12% accuracy · 2.3s training time</div>', unsafe_allow_html=True)

    with st.expander("📋 Title & Description Examples — click to see high-performing formats per sector", expanded=False):
        cols = st.columns(2)
        for i, ex in enumerate(TITLE_DESC_EXAMPLES):
            with cols[i%2]:
                st.markdown(f"""
                <div class="example-card">
                  <div class="example-sector">{ex['sector']}</div>
                  <div class="example-title">📌 {ex['title']}</div>
                  <div class="example-desc">"{ex['desc'][:120]}..."</div>
                  <div class="example-why">💡 {ex['why']}</div>
                </div>""", unsafe_allow_html=True)

    L,R = st.columns([1,1], gap="large")
    with L:
        st.markdown('<div class="sec-label">Video Details</div>', unsafe_allow_html=True)
        title = st.text_input("Title", placeholder="I Survived 100 Days in Minecraft Hardcore Mode", key="pt")
        desc  = st.text_area("Description", placeholder="The ultimate challenge — 100 days of survival, building, exploring...", height=90, key="pd")
        # Save to session_state → flows to all other tabs
        if title: st.session_state["shared_title"] = title
        if desc:  st.session_state["shared_desc"]  = desc
        # Auto-detect sector
        _sc = recommender.sector_scores(title, desc) if title else {}
        _auto_idx = SECTORS.index(max(_sc, key=_sc.get)) if _sc and max(_sc.values()) > 15 else 4
        c1,c2 = st.columns(2)
        sector  = c1.selectbox("Sector (auto-detected ✨)", SECTORS, index=_auto_idx, key="ps",
                               help="Auto-detected from your title and description. Change if incorrect.")
        dur_min = c2.number_input("Duration (min)", 1, 180, 12, key="pdu")
        st.markdown('<div class="sec-label" style="margin-top:.8rem">Channel</div>', unsafe_allow_html=True)
        c3,c4 = st.columns(2)
        subs   = c3.number_input("Subscribers", 1000, 50000000, 250000, step=1000, format="%d", key="psu")
        age_yr = c4.number_input("Age (years)", 0.5, 20.0, 4.0, step=0.5, key="pa")
        st.markdown('<div class="sec-label" style="margin-top:.8rem">Upload Timing</div>', unsafe_allow_html=True)
        c5,c6 = st.columns(2)
        uday   = c5.selectbox("Day", DOW_NAMES, index=3, key="pday")
        uhour  = c6.slider("Hour (UTC)", 0, 23, 14, key="ph")
        country = st.selectbox("Country", list(COUNTRY_UTC.keys()), key="pc")
        etags_r = st.text_input("Existing hashtags (comma-separated)", placeholder="travel, solotravel", key="pet")
        if etags_r: st.session_state["shared_tags"] = etags_r
        predict = st.button("🔮 Predict & Analyze", key="pbtn")

    with R:
        if predict and title.strip():
            etags  = [x.strip().lstrip("#").lower() for x in etags_r.split(",") if x.strip()]
            offset = COUNTRY_UTC.get(country,0)
            hloc   = int((uhour+offset)%24)
            dow    = DOW_NAMES.index(uday)
            abbr   = COUNTRY_ABBR.get(country,"local")
            best_utc = get_sector_best_utc(df_full, sector)
            best_local = int((best_utc+offset)%24)
            best_period = "PM" if best_local>=12 else "AM"
            best_disp = best_local%12 or 12

            inp={"title":title,"desc":desc,"subs":subs,"age_yr":age_yr,
                 "hour_utc":uhour,"hour_local":hloc,"dow":dow,
                 "month":4,"week":17,"dur_sec":dur_min*60,"etags":etags,
                 "offset":offset,"country":country}
            X     = prep_features(inp, FEATURES)
            probs = xgb_model.predict_proba(X)[0]
            tier  = INV_LABEL[int(probs.argmax())]
            conf  = int(probs.max()*100)
            col   = TIER_COLORS[tier]
            gap   = (probs.max()-sorted(probs)[-2])*100

            border_note=""
            if gap<8: border_note=f'<div style="font-size:.76rem;opacity:.85;margin-top:.3rem">⚡ Borderline — only {gap:.0f}% gap to next tier</div>'
            is_optimal=(uhour==best_utc)
            time_note=f"✅ Optimal for {sector}!" if is_optimal else f"💡 Best for {sector}: {best_utc:02d}:00 UTC = {best_disp:02d}:00 {best_period} {abbr}"

            st.markdown(f"""
            <div class="tier-box" style="background:linear-gradient(135deg,{col},{col}bb)">
              <div style="font-size:.68rem;font-weight:700;opacity:.8;text-transform:uppercase;letter-spacing:.1em">Predicted Engagement Tier</div>
              <div class="tier-name">{TIER_EMOJI[tier]}  {tier}</div>
              <div style="font-size:.83rem;opacity:.85">Confidence {conf}% &nbsp;·&nbsp; {TIER_VIEWS[tier]} views &nbsp;·&nbsp; {TIER_ER[tier]} engagement</div>
              {border_note}
              <div style="background:rgba(255,255,255,.25);border-radius:999px;height:5px;margin-top:.8rem">
                <div style="width:{conf}%;background:white;border-radius:999px;height:5px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-label">Tier probability distribution</div>', unsafe_allow_html=True)
            for tname,p in zip(["HIGH","LOW","MID"],probs):
                pct=int(p*100); c_=TIER_COLORS[tname]; star="★ " if tname==tier else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.48rem">
                  <span style="width:58px;font-size:.82rem;font-weight:{'700' if tname==tier else '400'};color:{c_}">{star}{tname}</span>
                  <div style="flex:1;background:#E2E8F0;border-radius:999px;height:9px">
                    <div style="width:{pct}%;background:{c_};border-radius:999px;height:9px"></div>
                  </div>
                  <span style="width:38px;text-align:right;font-size:.84rem;font-weight:700;color:{c_}">{pct}%</span>
                </div>""", unsafe_allow_html=True)

            peak_icon="✅" if PEAK_WINDOW[0]<=hloc<=PEAK_WINDOW[1] else "⚠️"
            st.markdown(f"""<div class="insight-bar">
              📅 <b>{uday}</b> &nbsp;·&nbsp; 🕐 Your slot: {uhour:02d}:00 UTC = {hloc:02d}:00 {abbr} {peak_icon} &nbsp;|&nbsp; {time_note}
            </div>""", unsafe_allow_html=True)

            recs = recommender.recommend(title=title, description=desc, sector=sector, existing_tags=etags, n=5)
            if recs:
                st.markdown('<div class="sec-label" style="margin-top:.8rem">Recommended hashtags</div>', unsafe_allow_html=True)
                for r in recs:
                    sc=r["score"]; c_=TEAL if sc>=65 else AMBER if sc>=48 else "#94A3B8"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.33rem">
                      <span class="hash-pill">{r['tag']}</span>
                      <div style="flex:1;background:#E2E8F0;border-radius:999px;height:5px">
                        <div style="width:{sc}%;background:{c_};border-radius:999px;height:5px"></div>
                      </div>
                      <span style="font-size:.78rem;font-weight:700;color:{c_};width:32px">{sc}%</span>
                      <span style="font-size:.71rem;color:#94A3B8;flex:2">{r['reason']}</span>
                    </div>""", unsafe_allow_html=True)

            positives,warnings_,improvements = generate_reasoning(inp,tier,probs,FEATURES,IMP_DICT,best_utc,df_full)
            st.markdown('<div class="sec-label" style="margin-top:1rem">Why this prediction?</div>', unsafe_allow_html=True)
            for p in positives: st.markdown(f'<div class="insight-bar">{p}</div>', unsafe_allow_html=True)
            for w in warnings_: st.markdown(f'<div class="warn-bar">{w}</div>', unsafe_allow_html=True)
            if improvements:
                st.markdown('<div class="sec-label" style="margin-top:.8rem">How to reach HIGH tier</div>', unsafe_allow_html=True)
                for imp in improvements: st.markdown(f'<div class="danger-bar">{imp}</div>', unsafe_allow_html=True)
        elif predict:
            st.warning("Please enter a video title.")
        else:
            st.markdown(f"""<div style="background:white;border:2px dashed #E2E8F0;border-radius:14px;padding:2.5rem;text-align:center;color:#94A3B8;margin-top:.5rem">
              <div style="font-size:2.5rem;margin-bottom:.5rem">🎬</div>
              <div style="font-size:1rem;font-weight:600;color:{NAVY}">Fill in the form and click Predict & Analyze</div>
              <div style="font-size:.82rem;margin-top:.4rem">
                XGBoost · 34 features · 20,308 training videos<br>
                Title entered here auto-fills Hashtag & Sector tab ✨
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 · HASHTAG & SECTOR (combined)
# ══════════════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="model-badge">🏷️ Taxonomy Recommender · Jaccard 0.276 · Recall@5 56.4% &nbsp;+&nbsp; 🎯 CatBoost Sector Classifier · 96.5% accuracy</div>', unsafe_allow_html=True)

    # Pre-fill from session_state if available
    _shared_title = st.session_state.get("shared_title", "")
    _shared_desc  = st.session_state.get("shared_desc",  "")
    _shared_tags  = st.session_state.get("shared_tags",  "")

    # ── INPUT PANEL ───────────────────────────────────────────────────
    if _shared_title:
        st.markdown('<div class="insight-bar" style="padding:.45rem 1rem;font-size:.78rem;margin-bottom:.6rem">✨ Auto-filled from Predict tab — edit below if needed</div>', unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns([3,3,2])
    with ic1:
        hs_title = st.text_input("Video title", value=_shared_title,
                                  placeholder="iPhone 16 Pro Honest Review — Camera Test vs Samsung Galaxy S25", key="htt")
    with ic2:
        hs_desc = st.text_area("Description", value=_shared_desc,
                                placeholder="In-depth camera comparison. Real-world tests, battery life...", height=68, key="hdt")
    with ic3:
        hs_seeds_r = st.text_input("Seed hashtags (optional)", value=_shared_tags,
                                    placeholder="tech, review", key="hse")
        hs_n  = st.slider("# recommendations", 3, 10, 6, key="hnn")

    hs_btn = st.button("🔍 Analyze Sector & Get Hashtags", key="hbtn")
    st.markdown("---")

    if hs_btn and hs_title.strip():
        # Persist updated values
        st.session_state["shared_title"] = hs_title
        st.session_state["shared_desc"]  = hs_desc
        st.session_state["shared_tags"]  = hs_seeds_r

        hs_seeds = [x.strip().lstrip("#").lower() for x in hs_seeds_r.split(",") if x.strip()]

        # ── COMPUTE ───────────────────────────────────────────────────
        sec_scores  = recommender.sector_scores(hs_title, hs_desc, hs_seeds)
        sorted_sec  = sorted(sec_scores.items(), key=lambda x: -x[1])
        top_sec, top_val = sorted_sec[0]
        sec2, val2       = sorted_sec[1] if len(sorted_sec)>1 else ("—",0)
        is_uniform       = max(sec_scores.values()) - min(sec_scores.values()) < 5

        # Auto-use top sector for hashtag recommendation
        hs_sector = top_sec if not is_uniform else "Gaming"
        recs = recommender.recommend(hs_title, hs_desc, hs_sector, hs_seeds, n=hs_n)

        # ── LAYOUT: two sections side by side ─────────────────────────
        sec_col, hash_col = st.columns([1,1], gap="large")

        # ── SECTION 1: SECTOR ANALYSIS ────────────────────────────────
        with sec_col:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem">
              <div style="font-size:1.5rem">🎯</div>
              <div>
                <div style="font-size:.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.09em">Content Sector</div>
                <div style="font-size:1.4rem;font-weight:700;color:{TEAL}">{top_sec if not is_uniform else "Low signal"}</div>
              </div>
              <div style="margin-left:auto;text-align:right">
                <div style="font-size:.68rem;color:#94A3B8">Affinity</div>
                <div style="font-size:1.2rem;font-weight:700;color:{TEAL}">{top_val:.0f}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if is_uniform:
                st.markdown(f"""<div class="warn-bar">
                  ⚠️ <b>Low sector signal</b> — no strong keywords detected.
                  Add specific terms like "minecraft", "cooking", "python tutorial", "premier league" for better detection.
                </div>""", unsafe_allow_html=True)
            else:
                # Sector bars
                for sec, score in sorted_sec:
                    if score < 1.0: continue
                    is_top = sec == top_sec
                    bar_c  = TEAL if is_top else AMBER if score > 15 else "#CBD5E1"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.42rem">
                      <span style="width:96px;font-size:.82rem;font-weight:{'700' if is_top else '400'};color:{NAVY if is_top else '#64748B'}">{sec}</span>
                      <div style="flex:1;background:#E2E8F0;border-radius:999px;height:8px">
                        <div style="width:{min(score,100):.0f}%;background:{bar_c};border-radius:999px;height:8px"></div>
                      </div>
                      <span style="width:36px;text-align:right;font-size:.82rem;font-weight:600;color:{bar_c}">{score:.0f}%</span>
                    </div>""", unsafe_allow_html=True)

                # Donut
                labels_ = [s for s,v in sorted_sec if v > 1.0]
                values_ = [v for s,v in sorted_sec if v > 1.0]
                if labels_:
                    fig_s = go.Figure(go.Pie(
                        labels=labels_, values=values_, hole=0.5,
                        marker_colors=[TEAL,AMBER,NAVY,"#7C3AED","#F97316","#06B6D4","#84CC16"][:len(labels_)],
                        textinfo="label+percent", textfont_size=11))
                    fig_s.update_layout(height=220, paper_bgcolor="white",
                        margin=dict(l=0,r=0,t=8,b=0),
                        font=dict(family="Plus Jakarta Sans",size=10), showlegend=False)
                    st.plotly_chart(fig_s, use_container_width=True)

                cross = f"  Secondary: <b>{sec2}</b> ({val2:.0f}%)" if val2 > 15 else ""
                st.markdown(f"""<div class="insight-bar">
                  <b>Primary sector: {top_sec}</b> ({top_val:.0f}% affinity).{cross}<br>
                  <span style="font-size:.78rem">CatBoost achieves 96.5% sector accuracy from content alone.
                  {'Cross-sector overlap may affect algorithmic recommendations.' if val2>20 else ''}</span>
                </div>""", unsafe_allow_html=True)

        # ── SECTION 2: HASHTAG RECOMMENDATIONS ───────────────────────
        with hash_col:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem">
              <div style="font-size:1.5rem">🏷️</div>
              <div>
                <div style="font-size:.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.09em">Hashtag Recommendations</div>
                <div style="font-size:1.4rem;font-weight:700;color:{TEAL}">Top {len(recs)} tags</div>
              </div>
              <div style="margin-left:auto;text-align:right">
                <div style="font-size:.68rem;color:#94A3B8">For sector</div>
                <div style="font-size:.9rem;font-weight:700;color:{TEAL}">{hs_sector}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Pill row
            pills = "".join(f'<span class="hash-pill">{r["tag"]}</span>' for r in recs)
            st.markdown(f'<div style="margin-bottom:.6rem">{pills}</div>', unsafe_allow_html=True)

            # Detailed bars
            for r in recs:
                sc = r["score"]
                lbl = "Strong" if sc>=70 else "Moderate" if sc>=50 else "Weak"
                c_  = TEAL if sc>=70 else AMBER if sc>=50 else "#94A3B8"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.38rem">
                  <span class="hash-pill" style="min-width:100px;text-align:center">{r['tag']}</span>
                  <div style="flex:1;background:#E2E8F0;border-radius:999px;height:7px">
                    <div style="width:{sc}%;background:{c_};border-radius:999px;height:7px"></div>
                  </div>
                  <span style="font-size:.78rem;font-weight:700;color:{c_};width:32px">{sc}%</span>
                  <span style="font-size:.7rem;color:#94A3B8;width:52px">{lbl}</span>
                  <span style="font-size:.7rem;color:#94A3B8;flex:2">{r['reason']}</span>
                </div>""", unsafe_allow_html=True)

            # Radar chart
            if recs:
                tags_r   = [r["tag"] for r in recs]
                scores_r = [r["score"] for r in recs]
                fig_r = go.Figure(go.Scatterpolar(
                    r=scores_r+[scores_r[0]], theta=tags_r+[tags_r[0]],
                    fill="toself", fillcolor="rgba(13,148,136,.15)",
                    line=dict(color=TEAL,width=2), marker=dict(size=6,color=TEAL)))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True,range=[0,100],
                               tickfont=dict(size=9),tickvals=[25,50,75,100])),
                    title="Confidence radar", height=250, paper_bgcolor="white",
                    font=dict(family="Plus Jakarta Sans",size=11),
                    margin=dict(l=20,r=20,t=44,b=10))
                st.plotly_chart(fig_r, use_container_width=True)

            st.markdown(f"""<div class="insight-bar">
              <b>Score logic:</b> Title word match = 65–91% · Description match = 38–60% ·
              Sector trend only = 28–45%. Tags are ranked by how specifically they match
              your content, not just how popular they are in the sector.
            </div>""", unsafe_allow_html=True)

    elif hs_btn:
        st.info("Please enter a video title.")
    else:
        # Empty state — two columns with placeholder
        ec1, ec2 = st.columns(2, gap="large")
        with ec1:
            st.markdown(f"""<div style="background:white;border:2px dashed #E2E8F0;border-radius:12px;padding:2rem;text-align:center;color:#94A3B8">
              <div style="font-size:2rem;margin-bottom:.4rem">🎯</div>
              <div style="font-size:.9rem;font-weight:600;color:{NAVY}">Sector Analysis</div>
              <div style="font-size:.78rem;margin-top:.3rem">Primary sector + affinity breakdown<br>+ donut chart</div>
            </div>""", unsafe_allow_html=True)
        with ec2:
            st.markdown(f"""<div style="background:white;border:2px dashed #E2E8F0;border-radius:12px;padding:2rem;text-align:center;color:#94A3B8">
              <div style="font-size:2rem;margin-bottom:.4rem">🏷️</div>
              <div style="font-size:.9rem;font-weight:600;color:{NAVY}">Hashtag Recommendations</div>
              <div style="font-size:.78rem;margin-top:.3rem">Top {6} tags with confidence scores<br>+ radar chart</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="text-align:center;color:#94A3B8;font-size:.82rem;margin-top:.5rem">
          Enter title above and click <b>Analyze Sector & Get Hashtags</b>
          {' &nbsp;·&nbsp; Or come from <b>Predict tab</b> — fields auto-fill ✨' if _shared_title else ''}
        </div>""", unsafe_allow_html=True)

with t3:
    st.markdown('<div class="model-badge">⏰ Temporal Analysis · Best UTC hour is sector-specific · Computed from 20,308 video dataset</div>', unsafe_allow_html=True)
    if not df_full.empty and "upload_dow" in df_full.columns and "engagement_rate" in df_full.columns:
        # Sector selector
        tr_sec = st.selectbox("Select sector", ["All"]+SECTORS, key="trs")
        sub = df_full if tr_sec=="All" else (df_full[df_full["sector"]==tr_sec] if "sector" in df_full.columns else df_full)
        best_utc_t = get_sector_best_utc(df_full, tr_sec)

        col_l, col_r = st.columns([2,1], gap="large")

        with col_l:
            # Day chart
            de = sub.groupby("upload_dow")["engagement_rate"].mean()
            de.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(de)]
            best_d = de.idxmax()
            c_days = [TEAL if d==best_d else "#CBD5E1" for d in de.index]
            fig=go.Figure(go.Bar(x=de.index,y=de.values,marker_color=c_days,
                text=[f"{v:.2f}%" for v in de.values],textposition="outside",textfont=dict(size=11)))
            fig.add_annotation(x=best_d,y=de[best_d],text="★ Best",showarrow=True,arrowhead=2,ay=-35,
                font=dict(size=10,color=TEAL),bgcolor="white",bordercolor=TEAL,borderwidth=1)
            fig.update_layout(title=f"Avg engagement by upload day — {tr_sec}",height=260,
                paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=0,t=48,b=0),
                font=dict(family="Plus Jakarta Sans",size=11),
                yaxis=dict(range=[de.min()*0.96,de.max()*1.12],showgrid=True,gridcolor="#F1F5F9"))
            st.plotly_chart(fig,use_container_width=True)

            # Hour chart
            if "upload_hour_utc" in df_full.columns:
                he = sub.groupby("upload_hour_utc")["engagement_rate"].mean()
                c_h=[TEAL if h==best_utc_t else AMBER if PEAK_WINDOW[0]<=h<=PEAK_WINDOW[1] else "#E2E8F0" for h in he.index]
                fig2=go.Figure(go.Bar(x=he.index,y=he.values,marker_color=c_h,
                    hovertemplate="<b>%{x}:00 UTC</b><br>Avg ER: %{y:.2f}%<extra></extra>"))
                fig2.add_vrect(x0=PEAK_WINDOW[0]-0.5,x1=PEAK_WINDOW[1]+0.5,fillcolor=AMBER,opacity=0.07,line_width=0,
                    annotation_text="Peak window (14–22 local)",annotation_position="top left",annotation_font=dict(size=9,color=AMBER))
                fig2.add_annotation(x=best_utc_t,y=he[best_utc_t],text=f"★ Best={best_utc_t:02d}:00",showarrow=True,arrowhead=2,ay=-35,
                    font=dict(size=10,color=TEAL),bgcolor="white",bordercolor=TEAL,borderwidth=1)
                fig2.update_layout(title=f"Avg engagement by UTC hour — {tr_sec}",height=220,
                    paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=0,t=48,b=0),
                    font=dict(family="Plus Jakarta Sans",size=11),
                    yaxis=dict(showgrid=True,gridcolor="#F1F5F9"),xaxis=dict(title="Hour (UTC)",dtick=2))
                st.plotly_chart(fig2,use_container_width=True)

            # SECTOR-SPECIFIC TIMING HEATMAP
            if "upload_hour_utc" in df_full.columns and "sector" in df_full.columns:
                st.markdown(f'<div class="sec-label" style="margin-top:.5rem">Timing heatmap — {tr_sec} sector only</div>', unsafe_allow_html=True)
                sub_h = sub.groupby(["upload_dow","upload_hour_utc"])["engagement_rate"].mean().unstack(fill_value=np.nan)
                if not sub_h.empty:
                    sub_h.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(sub_h)]
                    fig3=go.Figure(go.Heatmap(z=sub_h.values,x=[f"{h:02d}:00" for h in sub_h.columns],y=sub_h.index,
                        colorscale=[[0,"#F0FDF4"],[0.4,"#86EFAC"],[0.7,TEAL],[1,NAVY]],hoverongaps=False,
                        hovertemplate=f"<b>%{{y}}</b> at <b>%{{x}}</b><br>{tr_sec} avg ER: <b>%{{z:.2f}}%</b><extra></extra>",
                        colorbar=dict(title="ER %",thickness=12,len=0.8)))
                    # Mark best cell
                    valid=sub_h.values.copy(); valid[np.isnan(valid)]=0
                    if valid.max()>0:
                        bp=np.unravel_index(valid.argmax(),valid.shape)
                        fig3.add_annotation(x=f"{sub_h.columns[bp[1]]:02d}:00",y=sub_h.index[bp[0]],
                            text="★",showarrow=False,font=dict(size=14,color="white"))
                    fig3.update_layout(title=f"Sector heatmap: {tr_sec} — Day × Hour (UTC) vs avg engagement",height=260,
                        paper_bgcolor="white",margin=dict(l=0,r=0,t=48,b=0),
                        font=dict(family="Plus Jakarta Sans",size=11),
                        xaxis=dict(tickfont=dict(size=9)),yaxis=dict(tickfont=dict(size=10)))
                    st.plotly_chart(fig3,use_container_width=True)

            reason_text = SECTOR_PEAK_REASONS.get(tr_sec, SECTOR_PEAK_REASONS["All"])
            st.markdown(f"""<div class="insight-bar">
              📅 <b>Best day: {best_d}</b> &nbsp;·&nbsp; <b>Best UTC hour: {best_utc_t:02d}:00</b> (dataset-derived for {tr_sec})<br>
              <b>Why this timing?</b> {reason_text}
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown(f'<div class="sec-label">UTC → Local · optimal = {best_utc_t:02d}:00 UTC for {tr_sec}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="warn-bar" style="margin-bottom:.5rem"><b>Note:</b> Best UTC hour is <b>sector-specific</b> — computed from {tr_sec} data above, not a global "18:00" estimate.</div>', unsafe_allow_html=True)
            for cname,offset in COUNTRY_UTC.items():
                lh=int((best_utc_t+offset)%24); dp=lh%12 or 12; per="PM" if lh>=12 else "AM"
                pk=PEAK_WINDOW[0]<=lh<=PEAK_WINDOW[1]; pk_icon="✅" if pk else "⚠️"
                pk_label=f'<span style="color:{TEAL};font-size:.7rem">Peak</span>' if pk else f'<span style="color:{AMBER};font-size:.7rem">Off-peak</span>'
                abbr_=COUNTRY_ABBR.get(cname,"local")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:.42rem 0;border-bottom:1px solid #F1F5F9;font-size:.83rem">
                  <span style="color:#475569;flex:2">{cname}</span>
                  <span style="font-weight:700;color:{TEAL if pk else AMBER};font-family:'JetBrains Mono',monospace;flex:1;text-align:center">{dp:02d}:00 {per} {abbr_}</span>
                  <span style="flex:1;text-align:right">{pk_label} {pk_icon}</span>
                </div>""", unsafe_allow_html=True)

            # India-specific note
            india_local=int((best_utc_t+5.5)%24); india_min=int((0.5*60))
            india_peak=PEAK_WINDOW[0]<=india_local<=PEAK_WINDOW[1]
            st.markdown(f"""<div class="insight-bar" style="margin-top:.8rem">
              <b>India specifically ({tr_sec}):</b> Upload at {best_utc_t:02d}:00 UTC
              = {india_local:02d}:{india_min:02d} IST.
              {'✅ Falls within 14:00–22:00 IST peak window.' if india_peak else f'⚠️ Outside IST peak window. For India, try 10:30–16:30 UTC (= 16:00–22:00 IST).'}
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="warn-bar" style="margin-top:.5rem">
              <b>Why off-peak hurts:</b> YouTube boosts videos with fast early engagement.
              Off-peak uploads get fewer views in the first 6 hours → algorithm signals low interest
              → reduced organic reach. This effect compounds over 24–48 hours.
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 · MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="model-badge">🔬 All 5 Models · XGBoost 69.1% · RoBERTa 64.1% · CatBoost 99.1% · Ensemble 69.0% · Recommender Jaccard 0.276</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        "Model":["BTP-1 Random Forest","BTP-2 XGBoost ✅","BTP-2 RoBERTa","BTP-2 Ensemble","Oracle Ceiling"],
        "Task":["Engagement","Engagement","Engagement","Engagement","Theoretical"],
        "Accuracy":["61.2%","69.1%","64.1%","69.0%","78.2%"],
        "vs BTP-1":["—","+7.9 pp","+2.9 pp","+7.8 pp","+17.0 pp"],
        "Notes":["Baseline · 9,282 videos","34 features · 2.3s · BEST","RoBERTa-base · 124M params · 40 min MPS","XGB 70%+RoBERTa 30%","Max if routing perfectly"],
    }),use_container_width=True,hide_index=True)

    col1,col2=st.columns(2)
    with col1:
        fig=go.Figure(go.Bar(x=["BTP-1","RoBERTa","Ensemble","XGBoost","Oracle"],
            y=[61.2,64.1,69.0,69.12,78.2],marker_color=[RED,AMBER,AMBER,TEAL,PALE],
            text=["61.2%","64.1%","69.0%","69.1%","78.2%"],textposition="outside",textfont=dict(size=11)))
        fig.add_hline(y=61.2,line_dash="dash",line_color=RED,annotation_text="BTP-1",annotation_position="right")
        fig.add_hline(y=78.2,line_dash="dot",line_color="#94A3B8",annotation_text="Oracle ceiling",annotation_position="right")
        fig.update_layout(title="Engagement classification accuracy",yaxis=dict(range=[55,85]),height=290,
            paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=80,t=36,b=0),
            font=dict(family="Plus Jakarta Sans",size=11),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        fig=go.Figure()
        classes=["LOW","MID","HIGH"]
        xgb_f1=[0.75,0.57,0.75]; rob_f1=[0.72,0.52,0.67]
        fig.add_trace(go.Bar(name="XGBoost",x=classes,y=xgb_f1,marker_color=TEAL,
            text=[f"{v:.2f}" for v in xgb_f1],textposition="outside",textfont=dict(size=10)))
        fig.add_trace(go.Bar(name="RoBERTa",x=classes,y=rob_f1,marker_color=AMBER,
            text=[f"{v:.2f}" for v in rob_f1],textposition="outside",textfont=dict(size=10)))
        fig.update_layout(barmode="group",title="Per-class F1: MID hardest for both models",
            yaxis=dict(range=[0,0.9],showgrid=True,gridcolor="#F1F5F9",title="F1"),height=290,
            paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=0,t=48,b=0),
            font=dict(family="Plus Jakarta Sans",size=11),legend=dict(x=0.7,y=1))
        st.plotly_chart(fig,use_container_width=True)

    col3,col4=st.columns(2)
    with col3:
        fig=go.Figure(go.Pie(values=[1665,440,279,663],
            labels=["Both correct 54.6%","XGBoost only 14.4%","RoBERTa only 9.2%","Both wrong 21.8%"],
            hole=0.42,marker_colors=[TEAL,AMBER,PALE,RED],textinfo="label",textfont_size=10))
        fig.update_layout(title="Prediction overlap — oracle ceiling analysis",height=270,
            paper_bgcolor="white",margin=dict(l=0,r=0,t=36,b=0),
            font=dict(family="Plus Jakarta Sans",size=11),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with col4:
        if FEATURES:
            imp=pd.Series(xgb_model.feature_importances_,index=FEATURES).sort_values(ascending=False).head(12)
            def fc(f):
                if "channel" in f: return NAVY
                if "upload" in f or "peak" in f: return TEAL
                if "hashtag" in f: return "#7C3AED"
                return AMBER
            fig=go.Figure(go.Bar(x=imp.values,y=imp.index,orientation="h",
                marker_color=[fc(f) for f in imp.index],
                text=[f"{v:.4f}" for v in imp.values],textposition="outside",textfont=dict(size=9)))
            fig.update_layout(title="Feature importance · 🟦 Channel · 🟩 Timing · 🟣 Hashtag · 🟡 Text",
                height=270,paper_bgcolor="white",plot_bgcolor="white",
                margin=dict(l=0,r=60,t=48,b=0),font=dict(family="Plus Jakarta Sans",size=11),
                yaxis=dict(autorange="reversed"),xaxis=dict(showgrid=True,gridcolor="#F1F5F9"))
            st.plotly_chart(fig,use_container_width=True)

    col5,col6=st.columns(2)
    with col5:
        sectors_e=["Gaming","Comedy","Education","Lifestyle","Sports","Entertainment","Sci-Tech"]
        xgb_acc=[75.1,75.3,71.4,71.2,68.6,66.5,51.0]; rob_acc=[68.9,67.9,66.1,64.7,64.3,62.9,49.3]
        fig=go.Figure()
        fig.add_trace(go.Bar(name="XGBoost",y=sectors_e,x=xgb_acc,orientation="h",marker_color=TEAL,
            text=[f"{v:.0f}%" for v in xgb_acc],textposition="outside",textfont=dict(size=10)))
        fig.add_trace(go.Bar(name="RoBERTa",y=sectors_e,x=rob_acc,orientation="h",marker_color=AMBER,
            text=[f"{v:.0f}%" for v in rob_acc],textposition="outside",textfont=dict(size=10)))
        fig.add_vline(x=61.2,line_dash="dash",line_color=RED,annotation_text="BTP-1",annotation_position="top right")
        fig.update_layout(barmode="group",title="Per-sector accuracy: XGBoost vs RoBERTa",height=290,
            paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=60,t=48,b=0),
            font=dict(family="Plus Jakarta Sans",size=11),yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[40,85],showgrid=True,gridcolor="#F1F5F9"),legend=dict(x=0.55,y=0.05))
        st.plotly_chart(fig,use_container_width=True)

    with col6:
        fig=go.Figure(go.Bar(
            x=["Channel\nAuthority","Title\nOptimization","Timing\nOptimization","Hashtag\nOptimization","Thumbnails\n+ Transcripts"],
            y=[22,8,4,3,9],marker_color=[NAVY,TEAL,AMBER,"#7C3AED",PALE],
            text=["21.8% weight","~2.5% ER boost","~0.5% ER boost","~0.3% ER boost","Phase 3 target"],
            textposition="outside",textfont=dict(size=9)))
        fig.update_layout(title="Real-world impact: where to focus your energy",height=290,
            paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=0,r=0,t=48,b=0),
            font=dict(family="Plus Jakarta Sans",size=11),
            yaxis=dict(title="Relative impact",showgrid=True,gridcolor="#F1F5F9"))
        st.plotly_chart(fig,use_container_width=True)

    # Key findings
    st.markdown('<div class="sec-label" style="margin-top:.5rem">Key BTP-2 Findings — Real-World Implications</div>', unsafe_allow_html=True)
    findings=[
        (TEAL,"📊","Channel authority dominates (21.8% combined weight)","Top 3 features are all channel-level metrics. For creators: subscriber count and channel age predict engagement better than any content decision. New creators face structural disadvantage regardless of content quality."),
        (AMBER,"📝","Short titles win (r=−0.30 word count, r=−0.28 length)","The strongest text signals are negative — more title words = lower engagement. Optimal: 6–10 words, 40–70 chars. YouTube rewards concise, scannable titles."),
        ("#7C3AED","🏷️","Zero hashtags achieves highest engagement (3.636%)","Counter-intuitive: NO hashtags outperforms all counts. Likely mechanism: established channels don't need hashtags; heavy hashtag use signals low-quality spam content."),
        (TEAL,"🕐","Timing is significant but secondary (+0.5% ER in peak window)","Thursday + 14–22 local = measurable improvement, but dwarfed by channel authority. Timing optimization alone cannot overcome a weak channel."),
        (RED,"🎯","MID class is fundamentally ambiguous (F1=0.57 for both models)","Videos on the HIGH/LOW boundary can't be reliably classified. This is inherent to ordinal classification, not a model flaw. Use A/B testing for borderline content rather than relying on prediction alone."),
    ]
    for color,icon,title_,body in findings:
        st.markdown(f"""<div style="background:white;border-left:4px solid {color};border-radius:0 10px 10px 0;
            padding:.9rem 1.2rem;margin:.5rem 0;border:1px solid #E2E8F0;border-left:4px solid {color}">
          <div style="font-weight:700;color:{NAVY};font-size:.88rem">{icon} {title_}</div>
          <div style="color:#475569;font-size:.82rem;margin-top:.3rem">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:1rem">Sector classifier ablation</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Variant":["BTP-1 CatBoost","BTP-2 content only (no category_id)","BTP-2 all features"],
        "Accuracy":["66.8%","96.5%","99.1%"],
        "vs BTP-1":["—","+29.7 pp ✅","+32.3 pp ✅"],
        "Insight":["Limited features","Sector predictable from content alone","category_id adds only 2.6pp"],
    }),use_container_width=True,hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────
nvid=meta.get("total_videos",20308)
st.markdown(f"""
<div class="dash-footer">
  BTP-2 · YouTube Performance Predictor · Vaidik Sharma (22MT10063) · IIT Kharagpur ·
  Prof. Pabita Mitra &nbsp;|&nbsp; 5 models · {nvid:,} videos · 75 features · XGBoost 69.12%
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 · DATASET EDA (last tab — deepest info)
# ══════════════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="model-badge">📊 BTP-2 Dataset · 20,308 YouTube videos · 157 channels · 75 features · 8 sectors · YouTube Data API v3</div>', unsafe_allow_html=True)

    if df_full.empty:
        st.warning("Dataset not found at data/btp2_v1_labeled.parquet — run eda.py first.")
    else:
        # ── KPI row ──────────────────────────────────────────────────
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        total_vids = meta.get("total_videos", len(df_full))
        med_er  = df_full["engagement_rate"].median() if "engagement_rate" in df_full.columns else 0
        mean_er = df_full["engagement_rate"].mean()   if "engagement_rate" in df_full.columns else 0
        pct_high = (df_full["engagement_tier"]=="HIGH").mean()*100 if "engagement_tier" in df_full.columns else 33.3
        k1.metric("Total Videos",  f"{total_vids:,}")
        k2.metric("Channels",      f"{meta.get('unique_channels',157):,}")
        k3.metric("Features",      meta.get("schema_cols", 75))
        k4.metric("Median ER",     f"{med_er:.2f}%")
        k5.metric("Mean ER",       f"{mean_er:.2f}%")
        k6.metric("HIGH tier %",   f"{pct_high:.1f}%")

        st.markdown("---")

        # ── Row 1: Distribution charts ─────────────────────────────
        st.markdown('<div class="sec-label">Distribution</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)

        with r1c1:
            sc = df_full["sector"].value_counts()
            fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation="h",
                marker=dict(color=sc.values, colorscale=[[0,PALE],[1,NAVY]], showscale=False),
                text=[f"{v:,}" for v in sc.values], textposition="outside", textfont=dict(size=10)))
            fig.update_layout(title="Videos per sector", height=270,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=0,r=50,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", range=[0,sc.max()*1.18]))
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            if "engagement_tier" in df_full.columns:
                tc = df_full["engagement_tier"].value_counts()
                fig = go.Figure(go.Pie(values=tc.values, names=tc.index, hole=0.45,
                    marker_colors=[TEAL if n=="HIGH" else AMBER if n=="MID" else RED for n in tc.index],
                    textinfo="label+percent", textfont_size=12))
                fig.update_layout(title="Engagement tier distribution (33/33/34% split)",
                    height=270, paper_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with r1c3:
            if "engagement_rate" in df_full.columns:
                fig = go.Figure()
                for tier_, c_ in [("HIGH",TEAL),("MID",AMBER),("LOW",RED)]:
                    if "engagement_tier" in df_full.columns:
                        sub_ = df_full[df_full["engagement_tier"]==tier_]["engagement_rate"]
                    else:
                        sub_ = df_full["engagement_rate"]
                    fig.add_trace(go.Box(y=sub_, name=tier_, marker_color=c_,
                        boxpoints=False, line=dict(width=2)))
                fig.update_layout(title="Engagement rate distribution by tier",
                    height=270, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    yaxis=dict(title="ER %", showgrid=True, gridcolor="#F1F5F9", range=[0,15]),
                    showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Engagement by sector + title length ──────────────
        st.markdown('<div class="sec-label">Engagement patterns</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            if "engagement_rate" in df_full.columns:
                med = df_full.groupby("sector")["engagement_rate"].median().sort_values(ascending=False)
                fig = go.Figure(go.Bar(x=med.index, y=med.values,
                    marker=dict(color=med.values, colorscale=[[0,PALE],[1,TEAL]], showscale=False),
                    text=[f"{v:.2f}%" for v in med.values], textposition="outside", textfont=dict(size=11)))
                fig.update_layout(title="Median engagement rate by sector",
                    height=280, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=40), font=dict(family="Plus Jakarta Sans",size=11),
                    xaxis=dict(tickangle=-30),
                    yaxis=dict(range=[0,med.max()*1.22], showgrid=True, gridcolor="#F1F5F9"))
                st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            if "title_length_chars" in df_full.columns and "engagement_rate" in df_full.columns:
                # Bin title lengths
                bins = [0,20,40,60,80,100,200]
                labels_b = ["0–20","21–40","41–60","61–80","81–100","100+"]
                df_tl = df_full.copy()
                df_tl["title_bin"] = pd.cut(df_tl["title_length_chars"], bins=bins, labels=labels_b)
                tl_agg = df_tl.groupby("title_bin",observed=True)["engagement_rate"].agg(["mean","count"]).reset_index()
                tl_agg = tl_agg[tl_agg["count"]>=30]
                bar_c_ = [TEAL if lb in ["41–60","61–80"] else AMBER if lb in ["21–40"] else RED for lb in tl_agg["title_bin"].astype(str)]
                fig = go.Figure(go.Bar(x=tl_agg["title_bin"].astype(str), y=tl_agg["mean"],
                    marker_color=bar_c_,
                    text=[f"{v:.2f}%" for v in tl_agg["mean"]],
                    textposition="outside", textfont=dict(size=10)))
                fig.add_vrect(x0=1.5,x1=3.5, fillcolor=TEAL, opacity=0.05, line_width=0,
                    annotation_text="✅ Optimal (40–80 chars)", annotation_position="top left",
                    annotation_font=dict(size=9,color=TEAL))
                fig.update_layout(title="Title length vs engagement rate  (optimal: 40–80 chars)",
                    height=280, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    xaxis=dict(title="Title length (chars)"),
                    yaxis=dict(title="Avg ER %", showgrid=True, gridcolor="#F1F5F9",
                               range=[0, tl_agg["mean"].max()*1.22 if len(tl_agg)>0 else 5]))
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Timing heatmap + hashtag finding ─────────────────
        st.markdown('<div class="sec-label">Timing & hashtag patterns</div>', unsafe_allow_html=True)

        if "upload_dow" in df_full.columns and "upload_hour_utc" in df_full.columns and "engagement_rate" in df_full.columns:
            pivot = (df_full.groupby(["upload_dow","upload_hour_utc"])["engagement_rate"]
                     .mean().unstack(fill_value=np.nan))
            pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot)]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns], y=pivot.index,
                colorscale=[[0,"#FFFBEB"],[0.4,"#FCD34D"],[0.7,AMBER],[1,RED]],
                hoverongaps=False,
                hovertemplate="<b>%{y}</b> at <b>%{x}</b><br>Avg ER: <b>%{z:.2f}%</b><extra></extra>",
                colorbar=dict(title="ER %", thickness=12, len=0.8)))
            best_pos = np.unravel_index(np.nanargmax(pivot.values), pivot.shape)
            fig.add_annotation(x=f"{pivot.columns[best_pos[1]]:02d}:00", y=pivot.index[best_pos[0]],
                text="★ Best", showarrow=True, arrowhead=2,
                font=dict(size=10,color="black"), bgcolor="white", bordercolor="black", borderwidth=1)
            fig.update_layout(title="Upload timing heatmap — hover for values · ★ = best slot",
                height=280, paper_bgcolor="white",
                margin=dict(l=0,r=0,t=48,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                xaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            if "hashtag_count" in df_full.columns and "engagement_rate" in df_full.columns:
                hce = (df_full[df_full["hashtag_count"]<=12]
                       .groupby("hashtag_count")["engagement_rate"]
                       .agg(["mean","count"]).reset_index())
                hce = hce[hce["count"]>=30]
                c_h = [TEAL if r["hashtag_count"]<=4 else RED for _,r in hce.iterrows()]
                fig = go.Figure(go.Bar(x=hce["hashtag_count"], y=hce["mean"],
                    marker_color=c_h,
                    text=[f"{v:.2f}%" for v in hce["mean"]],
                    textposition="outside", textfont=dict(size=10)))
                fig.add_vrect(x0=-0.5,x1=4.5, fillcolor=TEAL, opacity=0.06, line_width=0,
                    annotation_text="✅ Optimal (0–4)", annotation_position="top left",
                    annotation_font=dict(size=9,color=TEAL))
                fig.update_layout(title="🔑 Key finding: 0 hashtags wins (3.64%)",
                    height=270, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=52,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    xaxis=dict(title="Hashtag count", dtick=1),
                    yaxis=dict(title="Avg ER %", range=[0,hce["mean"].max()*1.22 if len(hce)>0 else 5],
                               showgrid=True, gridcolor="#F1F5F9"))
                st.plotly_chart(fig, use_container_width=True)

        with r3c2:
            if "upload_dow" in df_full.columns and "engagement_rate" in df_full.columns:
                dow_er = df_full.groupby("upload_dow")["engagement_rate"].mean()
                dow_er.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(dow_er)]
                best_d = dow_er.idxmax()
                c_days = [TEAL if d==best_d else "#CBD5E1" for d in dow_er.index]
                fig = go.Figure(go.Bar(x=dow_er.index, y=dow_er.values, marker_color=c_days,
                    text=[f"{v:.2f}%" for v in dow_er.values],
                    textposition="outside", textfont=dict(size=11)))
                fig.add_annotation(x=best_d, y=dow_er[best_d], text="★ Best day",
                    showarrow=True, arrowhead=2, ay=-35,
                    font=dict(size=10,color=TEAL), bgcolor="white", bordercolor=TEAL, borderwidth=1)
                fig.update_layout(title=f"Best upload day: {best_d} (3.58% avg engagement)",
                    height=270, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=48,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    yaxis=dict(range=[dow_er.min()*0.97, dow_er.max()*1.12],
                               showgrid=True, gridcolor="#F1F5F9"))
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 4: Channel authority + duration ─────────────────────
        st.markdown('<div class="sec-label">Channel authority & content signals</div>', unsafe_allow_html=True)
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            if "channel_subscribers" in df_full.columns and "engagement_rate" in df_full.columns:
                # Bin subscribers into log-scale buckets
                df_cs = df_full.copy()
                bins_s  = [0, 10000, 50000, 100000, 500000, 1000000, 50000000]
                labs_s  = ["<10K","10K–50K","50K–100K","100K–500K","500K–1M","1M+"]
                df_cs["sub_bin"] = pd.cut(df_cs["channel_subscribers"], bins=bins_s, labels=labs_s)
                sub_agg = df_cs.groupby("sub_bin", observed=True)["engagement_rate"].median().reset_index()
                fig = go.Figure(go.Bar(x=sub_agg["sub_bin"].astype(str), y=sub_agg["engagement_rate"],
                    marker=dict(color=sub_agg["engagement_rate"],
                                colorscale=[[0,PALE],[1,NAVY]], showscale=False),
                    text=[f"{v:.2f}%" for v in sub_agg["engagement_rate"]],
                    textposition="outside", textfont=dict(size=10)))
                fig.update_layout(title="Median ER by subscriber count (channel authority)",
                    height=270, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    xaxis=dict(title="Subscribers"),
                    yaxis=dict(title="Median ER %", showgrid=True, gridcolor="#F1F5F9"))
                st.plotly_chart(fig, use_container_width=True)

        with r4c2:
            if "duration_seconds" in df_full.columns and "engagement_rate" in df_full.columns:
                df_dur = df_full[df_full["duration_seconds"]<=3600].copy()
                df_dur["dur_min_bin"] = pd.cut(df_dur["duration_seconds"]/60,
                    bins=[0,5,10,15,20,30,60], labels=["0–5","5–10","10–15","15–20","20–30","30–60"])
                dur_agg = df_dur.groupby("dur_min_bin", observed=True)["engagement_rate"].agg(["mean","count"]).reset_index()
                dur_agg = dur_agg[dur_agg["count"]>=30]
                c_dur = [TEAL if lb in ["5–10","10–15"] else AMBER if lb in ["0–5","15–20"] else RED
                         for lb in dur_agg["dur_min_bin"].astype(str)]
                fig = go.Figure(go.Bar(x=dur_agg["dur_min_bin"].astype(str), y=dur_agg["mean"],
                    marker_color=c_dur,
                    text=[f"{v:.2f}%" for v in dur_agg["mean"]],
                    textposition="outside", textfont=dict(size=10)))
                fig.update_layout(title="Video duration vs engagement (5–15 min sweet spot)",
                    height=270, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0,r=0,t=36,b=0), font=dict(family="Plus Jakarta Sans",size=11),
                    xaxis=dict(title="Duration (minutes)"),
                    yaxis=dict(title="Avg ER %", showgrid=True, gridcolor="#F1F5F9",
                               range=[0, dur_agg["mean"].max()*1.22 if len(dur_agg)>0 else 5]))
                st.plotly_chart(fig, use_container_width=True)

        # ── Key data insights ────────────────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:.5rem">Key dataset findings</div>', unsafe_allow_html=True)
        data_findings = [
            (TEAL,  "3.01% median engagement rate across 20,308 videos",
             "Range: 0.1%–45%. Distribution is heavily right-skewed — a small number of viral videos pull the mean up. The median (3.01%) is a more reliable baseline for realistic expectation-setting."),
            (AMBER, "Thursday is the highest engagement upload day (3.58% avg)",
             "Thursday consistently outperforms other days because the video has Friday–Sunday to organically spread. Weekend uploads face more competition and the algorithm doesn't have a full weekday window to push them."),
            (NAVY,  "Channel age and subscriber count are the top 2 predictors",
             "channel_total_videos (0.089), channel_subscribers (0.067), channel_age_days (0.062) — these three features alone account for 21.8% of the XGBoost model's predictive weight. Content quality is secondary."),
            ("#7C3AED","Only 2% caption coverage due to YouTube IP restrictions",
             "youtube-transcript-api was blocked by YouTube for most channels in the dataset. This is why transcript/caption features were excluded from the model — not enough coverage to be reliable."),
            (TEAL,  "Comedy sector has highest per-video engagement (3.77% median)",
             "Despite having fewer videos than Education or Gaming, Comedy content achieves the highest median engagement. Shorter, highly shareable content drives repeat engagement from existing subscribers."),
            (RED,   "MID tier has lowest F1 score (0.57) for both XGBoost and RoBERTa",
             "Videos in the MID tier are inherently ambiguous — they sit exactly at the boundary between HIGH and LOW. Neither structured features nor text embeddings can reliably distinguish them. A/B testing is recommended over model prediction for borderline content."),
        ]
        for color, title_, body in data_findings:
            st.markdown(f"""<div style="background:white;border-left:4px solid {color};border-radius:0 10px 10px 0;
                padding:.9rem 1.2rem;margin:.5rem 0;border:1px solid #E2E8F0;border-left:4px solid {color}">
              <div style="font-weight:700;color:{NAVY};font-size:.88rem">📌 {title_}</div>
              <div style="color:#475569;font-size:.82rem;margin-top:.3rem">{body}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
nvid = meta.get("total_videos", 20308)
st.markdown(f"""
<div class="dash-footer">
  BTP-2 · YouTube Performance Predictor · Vaidik Sharma (22MT10063) · IIT Kharagpur ·
  Prof. Pabita Mitra &nbsp;|&nbsp; 5 models · {nvid:,} videos · 75 features · XGBoost 69.12%
</div>""", unsafe_allow_html=True)
