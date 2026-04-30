"""
find_channel_ids_v3.py
──────────────────────────────────────────────────────────────────────
Resolves 150 YouTube @handles (30 per sector) → verified channel IDs
using the YouTube Data API v3 forHandle endpoint.

Sectors: Gaming · Education · Sports · Comedy · Lifestyle

Usage:
    python3 find_channel_ids_v3.py

Writes: seed_channels_v3.csv  (channel_id, channel_name, sector)
Cost  : ~150 quota units  (negligible)
"""
from __future__ import annotations
import os, re, time, csv, random
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set YOUTUBE_API_KEY in your .env file")

# ─────────────────────────────────────────────────────────────────────────────
# 30 NEW channels per sector (none appeared in seed_channels v1 or v2)
# Format: ("@handle", "display_name", "sector")
# ─────────────────────────────────────────────────────────────────────────────
HANDLES = [

    # ═══ GAMING (30) ═════════════════════════════════════════════════════════
    ("@penguinz0",          "MoistCr1TiKaL",         "Gaming"),
    ("@TheActMan",          "The Act Man",            "Gaming"),
    ("@SmallAnt",           "SmallAnt",               "Gaming"),
    ("@Northernlion",       "Northernlion",           "Gaming"),
    ("@RTGame",             "RTGame",                 "Gaming"),
    ("@Alpharad",           "Alpharad",               "Gaming"),
    ("@DougDoug",           "DougDoug",               "Gaming"),
    ("@CallMeKevin",        "CallMeKevin",            "Gaming"),
    ("@LazarBeam",          "LazarBeam",              "Gaming"),
    ("@TommyInnit",         "TommyInnit",             "Gaming"),
    ("@GeorgeNotFound",     "GeorgeNotFound",         "Gaming"),
    ("@Quackity",           "Quackity",               "Gaming"),
    ("@Ph1LzA",             "Ph1LzA",                 "Gaming"),
    ("@BadBoyHalo",         "BadBoyHalo",             "Gaming"),
    ("@Aphmau",             "Aphmau",                 "Gaming"),
    ("@SSundee",            "SSundee",                "Gaming"),
    ("@UnspeakableGaming",  "UnspeakableGaming",      "Gaming"),
    ("@Thinknoodles",       "Thinknoodles",           "Gaming"),
    ("@CaptainSparklez",    "CaptainSparklez",        "Gaming"),
    ("@MatthewPatrick13",   "Game Theory",            "Gaming"),
    ("@Ali-A",              "Ali-A",                  "Gaming"),
    ("@MrBossFTW",          "MrBossFTW",              "Gaming"),
    ("@OrkaBros",           "Orka Bros",              "Gaming"),
    ("@IGN",                "IGN",                    "Gaming"),
    ("@GameSpot",           "GameSpot",               "Gaming"),
    ("@GINXTV",             "GINX TV",                "Gaming"),
    ("@PlayStation",        "PlayStation",            "Gaming"),
    ("@Xbox",               "Xbox",                   "Gaming"),
    ("@NintendoAmerica",    "Nintendo America",       "Gaming"),
    ("@Blizzard",           "Blizzard Entertainment", "Gaming"),

    # ═══ EDUCATION (30) ══════════════════════════════════════════════════════
    ("@AsapSCIENCE",        "AsapSCIENCE",            "Education"),
    ("@TodayIFoundOut",     "Today I Found Out",      "Education"),
    ("@RealLifeLore",       "Real Life Lore",         "Education"),
    ("@GeographyNow",       "Geography Now",          "Education"),
    ("@OverSimplified",     "OverSimplified",         "Education"),
    ("@HistoryMatters",     "History Matters",        "Education"),
    ("@TierZoo",            "TierZoo",                "Education"),
    ("@NileRed",            "NileRed",                "Education"),
    ("@LockPickingLawyer",  "LockPickingLawyer",      "Education"),
    ("@MarkRober",          "Mark Rober",             "Education"),
    ("@StuffMadeHere",      "Stuff Made Here",        "Education"),
    ("@ActionLab",          "The Action Lab",         "Education"),
    ("@MinuteEarth",        "MinuteEarth",            "Education"),
    ("@HalfAsInteresting",  "Half as Interesting",    "Education"),
    ("@BranchEducation",    "Branch Education",       "Education"),
    ("@Fireship",           "Fireship",               "Education"),
    ("@Freecodecamp",       "freeCodeCamp",           "Education"),
    ("@TraversyMedia",      "Traversy Media",         "Education"),
    ("@BigThink",           "Big Think",              "Education"),
    ("@TED",                "TED",                    "Education"),
    ("@CleoCodes",          "Cleo Abram",             "Education"),
    ("@SimonOxfPhys",       "Simon Clark",            "Education"),
    ("@DrBeckyChambers",    "Becky Smethurst",        "Education"),
    ("@SabineHossenfelder", "Sabine Hossenfelder",    "Education"),
    ("@AstrumSpace",        "Astrum",                 "Education"),
    ("@SpaceRip",           "SpaceRip",               "Education"),
    ("@HowToAdultChannel",  "How To Adult",           "Education"),
    ("@medschoolinsiders",  "Med School Insiders",    "Education"),
    ("@KhanAcademy",        "Khan Academy",           "Education"),
    ("@mitocw",             "MIT OpenCourseWare",     "Education"),

    # ═══ SPORTS (30) ══════════════════════════════════════════════════════════
    ("@Olympics",           "Olympics",               "Sports"),
    ("@LaLiga",             "LaLiga",                 "Sports"),
    ("@Bundesliga",         "Bundesliga",             "Sports"),
    ("@SerieA",             "Serie A",                "Sports"),
    ("@ChampionsLeague",    "UEFA Champions League",  "Sports"),
    ("@TottenhamHotspur",   "Tottenham Hotspur",      "Sports"),
    ("@acmilan",            "AC Milan",               "Sports"),
    ("@juventus",           "Juventus",               "Sports"),
    ("@PSG",                "Paris Saint-Germain",    "Sports"),
    ("@MLS",                "Major League Soccer",    "Sports"),
    ("@ATP",                "ATP Tour",               "Sports"),
    ("@WTA",                "WTA Tennis",             "Sports"),
    ("@Wimbledon",          "Wimbledon",              "Sports"),
    ("@rolandgarros",       "Roland-Garros",          "Sports"),
    ("@WorldAthletics",     "World Athletics",        "Sports"),
    ("@FoxSports",          "Fox Sports",             "Sports"),
    ("@NBCSports",          "NBC Sports",             "Sports"),
    ("@CBSSports",          "CBS Sports",             "Sports"),
    ("@DAZN",               "DAZN",                   "Sports"),
    ("@CricketAustralia",   "Cricket Australia",      "Sports"),
    ("@ECBCricket",         "England Cricket",        "Sports"),
    ("@TheRealPCB",         "Pakistan Cricket",       "Sports"),
    ("@BWFbadminton",       "BWF Badminton",          "Sports"),
    ("@UFC",                "UFC",                    "Sports"),
    ("@PremierLeague",      "Premier League",         "Sports"),
    ("@NFLFilms",           "NFL Films",              "Sports"),
    ("@nbagleague",         "NBA G League",           "Sports"),
    ("@IPL",                "IPL",                    "Sports"),
    ("@PKL",                "Pro Kabaddi League",     "Sports"),
    ("@HockeyIndia",        "Hockey India",           "Sports"),

    # ═══ COMEDY (30) ══════════════════════════════════════════════════════════
    ("@ComedyCentral",      "Comedy Central",         "Comedy"),
    ("@SNL",                "Saturday Night Live",    "Comedy"),
    ("@StephenColbert",     "Stephen Colbert",        "Comedy"),
    ("@TrevorNoah",         "Trevor Noah",            "Comedy"),
    ("@HasanMinhaj",        "Hasan Minhaj",           "Comedy"),
    ("@MichaelReeves",      "Michael Reeves",         "Comedy"),
    ("@TheOdd1sOut",        "TheOdd1sOut",            "Comedy"),
    ("@JaidenAnimations",   "Jaiden Animations",      "Comedy"),
    ("@Domics",             "Domics",                 "Comedy"),
    ("@AlexClarkArt",       "Alex Clark",             "Comedy"),
    ("@GradeAUnderA",       "GradeAUnderA",           "Comedy"),
    ("@WillNE",             "Will NE",                "Comedy"),
    ("@TomScottGo",         "Tom Scott",              "Comedy"),
    ("@Nicholas DeOrio",    "Nicholas DeOrio",        "Comedy"),
    ("@QuentinReviews",     "Quinton Reviews",        "Comedy"),
    ("@SomeOrdinaryGamers", "Some Ordinary Gamers",   "Comedy"),
    ("@PatrickCC9005",      "Patrick CC",             "Comedy"),
    ("@Caddicarus",         "Caddicarus",             "Comedy"),
    ("@GordonRamsay",       "Gordon Ramsay",          "Comedy"),
    ("@TwoSetViolin",       "TwoSet Violin",          "Comedy"),
    ("@FailArmy",           "FailArmy",               "Comedy"),
    ("@JukinMedia",         "Jukin Media",            "Comedy"),
    ("@AFV",                "America's Funniest Home Videos", "Comedy"),
    ("@StandUpNBC",         "NBC Stand Up",           "Comedy"),
    ("@JimmyO'Brien",       "Jimmy O'Brien",          "Comedy"),
    ("@TommyTuckerr",       "Tommy Tucker",           "Comedy"),
    ("@ComingSoon",         "ComingSoon",             "Comedy"),
    ("@ScottishTwins",      "Scottish Twins",         "Comedy"),
    ("@LolOverlord",        "LOL Overlord",           "Comedy"),
    ("@KSI",                "KSI",                    "Comedy"),

    # ═══ LIFESTYLE (30) ═══════════════════════════════════════════════════════
    ("@CaseyNeistat",       "Casey Neistat",          "Lifestyle"),
    ("@PeterMcKinnon",      "Peter McKinnon",         "Lifestyle"),
    ("@UnboxTherapy",       "Unbox Therapy",          "Lifestyle"),
    ("@iJustine",           "iJustine",               "Lifestyle"),
    ("@SafiyaNygaard",      "Safiya Nygaard",         "Lifestyle"),
    ("@SimoneGiertz",       "Simone Giertz",          "Lifestyle"),
    ("@NikkieTutorials",    "NikkieTutorials",        "Lifestyle"),
    ("@JamesCharles",       "James Charles",          "Lifestyle"),
    ("@NYTCooking",         "NYT Cooking",            "Lifestyle"),
    ("@Maangchi",           "Maangchi",               "Lifestyle"),
    ("@JerryRigEverything", "JerryRigEverything",     "Lifestyle"),
    ("@MrMobile",           "Mr. Mobile",             "Lifestyle"),
    ("@mkbhd",              "MKBHD",                  "Lifestyle"),
    ("@LinusTechTips",      "Linus Tech Tips",        "Lifestyle"),
    ("@Dave2D",             "Dave2D",                 "Lifestyle"),
    ("@BestDressed",        "Best Dressed",           "Lifestyle"),
    ("@ZoeSugg",            "Zoella",                 "Lifestyle"),
    ("@LauraDIY",           "Laura DIY",              "Lifestyle"),
    ("@NathalieOutlet",     "Nathalie Outlet",        "Lifestyle"),
    ("@GabiDemartino",      "Gabi DeMartino",         "Lifestyle"),
    ("@ABCEntertainment",   "ABC Entertainment",      "Lifestyle"),
    ("@VisitBritain",       "Visit Britain",          "Lifestyle"),
    ("@HealthyGamerGG",     "Healthy Gamer GG",       "Lifestyle"),
    ("@struthless",         "struthless",             "Lifestyle"),
    ("@KenjiLopezAlt",      "J. Kenji Lopez-Alt",     "Lifestyle"),
    ("@ChefSteps",          "ChefSteps",              "Lifestyle"),
    ("@BrianLagerstrom",    "Brian Lagerstrom",       "Lifestyle"),
    ("@ThatDudeCanCook",    "That Dude Can Cook",     "Lifestyle"),
    ("@SortedFood",         "Sorted Food",            "Lifestyle"),
    ("@JacquesPerpin",      "Jacques Pépin",          "Lifestyle"),
]

# ─────────────────────────────────────────────────────────────────────────────

def _clean_handle(raw: str) -> str:
    m = re.search(r"@([A-Za-z0-9_.\-']+)", raw)
    return f"@{m.group(1)}" if m else raw.strip()


def lookup_one(yt, handle: str, name: str, sector: str) -> dict | None:
    bare = _clean_handle(handle).lstrip("@")
    for attempt in range(3):
        try:
            resp = yt.channels().list(
                part="snippet,statistics",
                forHandle=bare,
            ).execute()
            items = resp.get("items", [])
            if items:
                item = items[0]
                subs = int(item["statistics"].get("subscriberCount", 0))
                return {
                    "channel_id":   item["id"],
                    "channel_name": item["snippet"]["title"],
                    "sector":       sector,
                    "subscribers":  subs,
                }
            return None
        except HttpError as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"    API error for {handle}: {e}")
                return None


def main():
    yt = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

    sectors = ["Gaming", "Education", "Sports", "Comedy", "Lifestyle"]
    by_sector = {s: [h for h in HANDLES if h[2] == s] for s in sectors}
    for s, hs in by_sector.items():
        print(f"  {s}: {len(hs)} handles queued")

    print(f"\nLooking up {len(HANDLES)} handles…\n")

    found, missed = [], []

    for sector in sectors:
        print(f"{'─'*50}")
        print(f"  {sector.upper()}")
        print(f"{'─'*50}")
        for handle, name, sec in by_sector[sector]:
            result = lookup_one(yt, handle, name, sec)
            if result:
                found.append(result)
                print(f"  ✓ {handle:<32} → {result['channel_id']}  "
                      f"subs={result['subscribers']:>12,}  [{result['channel_name']}]")
            else:
                missed.append((handle, name, sector))
                print(f"  ✗ {handle:<32} → NOT FOUND")
            time.sleep(0.15 + random.uniform(0, 0.05))
        print()

    # Write output
    out = Path("seed_channels_v3.csv")
    found.sort(key=lambda r: (r["sector"], -r["subscribers"]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["channel_id", "channel_name", "sector"])
        w.writeheader()
        for r in found:
            w.writerow({k: r[k] for k in ["channel_id", "channel_name", "sector"]})

    # Summary
    print(f"{'═'*55}")
    print(f"Found   : {len(found)}/{len(HANDLES)} channels")
    for s in sectors:
        n = sum(1 for r in found if r["sector"] == s)
        print(f"  {s:<15}: {n}/30")
    print(f"\nNot found ({len(missed)}):")
    for h, n, s in missed:
        print(f"  {h}  [{n}]  ({s})")
    print(f"\nWritten → {out}")
    print(f"Quota used ≈ {len(HANDLES)} units")
    print(f"\nNext step:")
    print(f"  python3 run.py --seed {out} --max-per-channel 200")


if __name__ == "__main__":
    main()
