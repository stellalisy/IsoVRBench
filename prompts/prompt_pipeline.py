#!/usr/bin/env python3
"""Prompt-pool construction for label-free video reward-model training.

    Step 0  source        VidProM semantic-unique prompts
    Step 1  clean         strip tool/UI artifacts that are not part of the scene
    Step 2  validity      safety, length and script gates
    Step 3  blacklist     reject the small set of prompts that are unusable as scenes

Design rule (the whole reason this is a blacklist)
--------------------------------------------------
The first version of Step 3 was a WHITELIST: keep a prompt only if it contains one of 176
hand-listed scene words. It rejected 38.8% of otherwise-valid prompts, and reading a
sample of those showed ~40% were good scenes lost to vocabulary gaps ("samurai katana
cutting through a pig", "pilots in the aircraft"). A whitelist over an open class fails
CLOSED, which is the wrong failure direction: junk that slips through costs one weak
training pair, but a good prompt discarded is gone. Every rule below is therefore
deliberately narrow, and every rule was written against prompts that were read by hand.

Provenance of each rule
-----------------------
No rule here was written from imagination. Each was added after reading a sample of the
prompts the pipeline was getting wrong, in three rounds (see AUDIT below), and the
comment on each rule names the concrete prompt that motivated it. Rules that were tried
and removed are recorded too, so the record shows what did not work:

  * "non_english" by English-function-word ratio -- REMOVED. Rejected 7.08%, and reading
    the rejects showed most were keyword-style ENGLISH ("fire effect, fast movement,
    editing"), a normal prompt style here with no function words. A ratio test cannot
    separate that from foreign text. Replaced by non_latin_script + foreign_latin.
  * "empty_after_clean" at <4 alphabetic words -- RELAXED to <2. It was discarding good
    short prompts ("humans doing exercises", "leprechaun giving instructions").
  * SCENE whitelist -- REMOVED, see above.

Usage
-----
    python prompt_pipeline.py --src VidProM_semantic_unique.csv --measure
    python prompt_pipeline.py --src ... --measure --n 1700000
    python prompt_pipeline.py --src ... --sample-kept 120 --seed 777   print kept prompts to read
    python prompt_pipeline.py --src ... --sample-rejected 40           print rejects to read
    python prompt_pipeline.py --src ... --build --out pool/            write the frozen pool

--build writes pool_v2.0.jsonl plus pool_v2.0.manifest.json, which records the sha256 of
this file, the sha256 of the source csv, per-rule counts and the config. Rebuilding with
an unchanged manifest hash reproduces the pool exactly.
"""
import argparse, csv, hashlib, html, json, os, random, re, sys
from collections import Counter

VERSION = "v2.0"
# set from --src and --out in main()
SRC = None
OUT_DIR = None

# Step 2 validity gates
SAFETY_COLS = ["toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]
SAFETY_MAX = 0.5      # VidProM ships Detoxify scores; drop anything at or above this
LEN_MIN, LEN_MAX = 20, 400
ASCII_MIN = 0.85


# =====================================================================================
# Step 1 -- cleaning
# =====================================================================================
# The corpus is user input pasted from Discord bots, web UIs and phone keyboards, so a
# large share of every prompt is tool syntax rather than scene content. Cleaning removes
# that syntax; it never removes scene words. Ordering below is load-bearing and the
# comments say why wherever it is.

# Prompts are stored HTML-escaped, sometimes doubly ("&amp;quot;"). Unescape twice, then
# strip the wrapping quote. This MUST happen before any ^-anchored rule: while the quote
# was stripped at the end of cleaning instead, every ^-anchored rule silently failed on
# the (large) quoted subset.
QUOTES = re.compile(r"^[\s:;,.\-\u2013\u2014*#\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\[]+"
                    r"|[\s\"'\u201c\u201d\u2018\u2019\u00bb\]]+$")

# CJK-locale users write English prompts with full-width delimiters
# ("dancing girl\uff0cat the beach\uff0c-camera pan left"). Every regex below expects ASCII
# punctuation, so normalise first or none of them match.
FULLWIDTH = str.maketrans("\uff0c\u3001\u3002\uff1b\uff1a\uff01\uff1f\uff08\uff09\u3000", ",,.;:!?() ")

# Discord slash commands and "prompt:" prefixes, including truncated ones ("rompt: A
# vision in the heaven...") -- match the word tail so truncation does not matter.
SLASHCMD = re.compile(r"^\s*(?:/\w+\s*)?\w{0,3}rompt\s*:\s*|^\s*/\w+\s*:?\s*", re.I)
# a leading index from a numbered storyboard ("15. Generate an AI picture of ...")
LISTNUM = re.compile(r"^\s*\d{1,3}\s*[.)\]]\s+")
# social-media clipboard residue ("Liked by AiByDesign and others a sexy woman")
SOCIAL = re.compile(r"\bLiked by\b.*?\band others\b", re.I)
# delivery metadata ("video format vertical, ... duration 10 s")
DELIVERY = re.compile(r"[,;\s]*\b(?:video\s+format|duration|lenght|length)\s*[:=]?\s*"
                      r"(?:vertical|horizontal|portrait|landscape|square|\d+\s*(?:s|sec\w*|min\w*))\b", re.I)
# trailing quality assertions ("The black dragon was slayed by the knight. GOOD QUALITY IMAGE")
QUALBOILER = re.compile(r"[,.\s]*\b(?:good|best|high|top|great)\s+quality"
                        r"(?:\s+(?:image|video|render|photo))?\s*$", re.I)

HTMLTAG = re.compile(r"<[^>]{0,200}>")
UI = re.compile(r"\s*(?:Message\s*:.*?(?:\(Font\s*:[^)]*\))?"
                r"|\(\s*Font\s*:[^)]*\)"                     # (Font: MODERN) with no Message: prefix
                r"|\d+\s+Attachment\b"
                r"|image\s*:\s*\S+\.(?:png|jpg|jpeg|gif|heic))", re.I)
# Midjourney emphasis brackets wrapping the whole prompt: "((A fat cat stealing fish)) :1"
EMPHASIS = re.compile(r"[\(\[]{2,}|[\)\]]{2,}")
RATIO = re.compile(r"\b\d{1,2}\s*:\s*\d{1,2}\b|\b\d{2,}k\b|\buhd\b", re.I)

# Generation flags. Four details matter, each found by reading survivors:
#   - the delimiter before the dash may be a comma, not a space ("sunset,-camera pan left")
#   - a flag may take a SPACE-SEPARATED numeric argument ("-w 2", "-motion4 24"). Only
#     numbers are consumed, so descriptive words after a flag survive ("-camera pan left")
#   - a flag key may be glued to its value ("-motion4", "-gs24"), so a digit run is
#     allowed before the word boundary -- \b does not fire between a letter and a digit
#   - bare "w" is only a flag when it carries such an argument
FLAG = re.compile(r"(?:^|[\s,;])(?:[-\u2010-\u2015/]{1,2}\s?"
                  r"(?:motion|ar|gs|fps|seed|q|s|c|v|iw|no|style|uplight|test|neg|"
                  r"stylize|chaos|weird|camera|w)\d*\b[^\s]*"
                  r"(?:\s+\d+(?:\.\d+)?)?"
                  r"|--\w+(?:\s+[\w:.\/]+)?)", re.I)
# the same keys also appear WITHOUT a dash, straight out of a tool UI
# ("motion 4, a jim henson horror visual", "big door that opens w2 size 80", "n8000 cinematic")
BAREFLAG = re.compile(r"(?:^|[\s,;\-])(?:motion\s*\d+|gs\s*-?\d+|iw\s*[\d.]+|"
                      r"stylize\s*\d+|chaos\s*\d+|w\d+(?:\s+size\s+\d+)?|"
                      r"n\d{4,}|seed\s*\d+|lens\s+zoom|aspects?\s+ratio)\b", re.I)
WEIGHT = re.compile(r"\s*::\s*-?\d+(?:\.\d+)?")          # Midjourney ::10 multi-prompt weights
ORPHRATIO = re.compile(r"\s:\s?\d{1,2}\b")               # ":9" left when "16" was consumed
NUMRUN = re.compile(r"\s+\d{1,4}(?:\s+\d{1,4}){2,}\b")   # orphaned flag values mid-string
TRAILNUM = re.compile(r"(?:\s+\d+){2,}\s*\]?\s*$")       # ... and at the end

# A verbatim negative-prompt template recurring across the corpus, usually appended to an
# otherwise good prompt. Found only by reading long prompts IN FULL -- earlier samples
# were truncated for display and stopped before reaching it.
NEGTPL = re.compile(r"[,\s]*\b(?:morphing|erratic fluctuation(?: in motion)?|noisy|"
                    r"bad quality|distorted|poorly drawn|low quality|blurry|deformed|"
                    r"ugly|imperfections|grain|body warp|watermark|jpeg artifacts|"
                    r"oversaturated|static|poor quality|worst quality)\b", re.I)

# "Create a video of a bear eating honey" is a GOOD SCENE behind a request wrapper.
# Stripping the wrapper keeps the scene; rejecting the prompt would lose it. This is why
# the tool_request rule below sees only what survives this.
WRAPPER = re.compile(
    r"^\s*(?:please\s+)?(?:craft|create|generate|produce|design|make|show|draw)\s+"
    r"(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:(?:\d+\s*(?:s|sec\w*|min\w*)|short|quick|nice|cool|dynamic|stylish|realistic|"
    r"amazing|stunning|beautiful|epic|simple|timeline|animated|ai)\s+){0,3}"
    r"(?:video|animation|clip|scene|image|picture|photo)\s*"
    r"(?:of|about|showing|showcasing|depicting|featuring|with|for|in\s+which|that|which|where)?\s*",
    re.I)


def clean(p):
    """Step 1. Remove tool syntax, keep scene content. Order matters where noted."""
    p = html.unescape(html.unescape(p))
    p = QUOTES.sub("", p)               # unquote BEFORE any ^-anchored rule
    p = p.translate(FULLWIDTH)          # normalise BEFORE any rule expecting ASCII space
    p = SLASHCMD.sub("", p)
    p = LISTNUM.sub("", p)
    p = SOCIAL.sub(" ", p)
    p = DELIVERY.sub(" ", p)
    p = QUALBOILER.sub("", p)
    # RATIO must precede FLAG: FLAG's numeric-argument group would otherwise eat "16" out
    # of "-ar 16:9" and leave an orphan ":9" behind.
    for r in (HTMLTAG, UI, EMPHASIS, RATIO, FLAG, NEGTPL, WEIGHT, ORPHRATIO, BAREFLAG, NUMRUN):
        p = r.sub(" ", p)
    p = WRAPPER.sub("", p)
    p = TRAILNUM.sub("", p)
    # removing a clause mid-sentence leaves ", ," runs and stranded punctuation; tidy so
    # the length and word counts the gates below use are honest
    p = re.sub(r"\s{2,}", " ", p)
    p = re.sub(r"\s*([,;])(?:\s*[,;])+", r"\1", p)
    p = re.sub(r"\s+([,;.])", r"\1", p)
    return p.strip().strip('",;.- ').strip()


# =====================================================================================
# Step 3 -- the blacklist
# =====================================================================================
EN = set("the a an and or of in on at to is are was were with for from by this that it "
         "he she they his her their we you i as be been has have had not no but if then "
         "there here when where while into over under out up down".split())

RULES = {}

# --- 1. non-Latin script -------------------------------------------------------------
# Only actual non-Latin characters. The ASCII>=85% gate in Step 2 already removes most
# foreign prompts; this catches the mixed remainder.
NONLATIN = re.compile(r"[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
                      r"\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")


def r_nonlatin(p):
    hits = len(NONLATIN.findall(p))
    return hits >= 3 and hits / max(len(p), 1) > 0.10


RULES["non_latin_script"] = r_nonlatin

# --- 1b. foreign language in LATIN script --------------------------------------------
# Spanish, Italian, French, Serbian, Indonesian and Tagalog prompts all pass the script
# test. A low English-function-word ratio ALONE flags keyword-style English, so require
# it in CONJUNCTION with actual foreign function words; both together are high precision.
FOREIGN = set("el la los las un una del al que de y con por para en es son fue como pero "
              "su sus lo se no mas muy este esta eso aquel donde cuando "
              "le les des du dans sur avec pour est sont ils elle nous vous "
              "der die das ein eine und ist sind mit auf fur nicht sich auch "
              "il lo gli che di da per con non sono anche "
              "e o os as um uma nao mais muito seu sua pelo pela "
              "ang mga sa ng na ay ako ikaw siya kami kayo sila "
              "yang dan di ke dari untuk dengan adalah ini itu tidak "
              "je da se na su sam bio nije samo ovo".split())
# these are ordinary English words too; leaving them in inflates the foreign ratio on
# English text and produced the false positive "carrot vitamins, 3d CARTOON ANIMATION Y"
FOREIGN -= {"no", "as", "son", "con", "die", "per", "sam", "bio", "dan", "la", "al", "am",
            "us", "is", "in", "so", "one", "at", "an", "it", "on", "or", "ill", "sue"}


def r_foreign_latin(p):
    w = re.findall(r"[a-zA-Z']+", p.lower())
    if len(w) < 5:
        return False
    en = sum(1 for x in w if x in EN) / len(w)
    # single letters (y, e, o, a) are function words in several of these languages and
    # match inside English text, so they do not count
    fo = sum(1 for x in w if len(x) >= 2 and x in FOREIGN) / len(w)
    # the second clause catches short prompts where one English-looking token ("a") lifts
    # the English ratio just past the first cut -- e.g. "una flor de 2 hojas que cambie"
    return (en < 0.10 and fo >= 0.15) or (en < 0.20 and fo >= 0.30)


RULES["foreign_latin"] = r_foreign_latin


# --- 2. nothing left after cleaning ---------------------------------------------------
# The prompt WAS the metadata. Two words is the floor; below that there is no scene.
def r_empty(p):
    return len(re.findall(r"[a-zA-Z]{2,}", p)) < 2


RULES["empty_after_clean"] = r_empty

# --- 3. a request ABOUT the tool rather than a description of a scene -----------------
TOOL = re.compile(r"^\s*(?:best|good|nice|cool)\s+(?:video|animation|clip)\b"
                  r"|\b(?:welcome|intro|promo|advertis\w*|commercial)\s+video\s+for\b"
                  r"|\bvideo\s+that\s+you\s+can\s+make\b"
                  r"|^\s*(?:craft|create|generate|produce|design)\s+(?:a|an|the)?\s*"
                  r"(?:dynamic|short|quick|nice|cool|stylish)?\s*(?:video|animation|clip)\b"
                  r"|^\s*video\s+for\b", re.I)


def r_tool(p):
    """Sees only post-WRAPPER text, so "create a video of a bear eating honey" arrives as
    "a bear eating honey" and is kept. A wrapper phrasing in front of a long, contentful
    description is still a usable scene ("Create a video that tells the story of Noah's
    Ark, highlighting key moments..."), so gate on how much content remains."""
    if not TOOL.search(p):
        return False
    return len(re.findall(r"[a-zA-Z]{2,}", p)) < 12


RULES["tool_request"] = r_tool

# --- 3b. logo / typography ------------------------------------------------------------
# Graphic-design jobs, not scenes, and the generator cannot render legible text anyway.
# "jesus walking across a lake wearing a t-shirt with the Sveltejs logo" is a SCENE that
# happens to contain a logo, so require the prompt to be ABOUT the logo.
LOGO = re.compile(r"^\s*(?:a\s+|an\s+|the\s+)?logo\b"
                  r"|\blogo\s+(?:for|of|design|idea|concept)\b"
                  r"|\b(?:design|create|make|draw)\s+(?:a\s+|an\s+|the\s+)?logo\b"
                  r"|\btypograph|\bwordmark\b|\beach\s+letter\b"
                  r"|\buse\s+a\s+[\w\s]{0,20}font\b", re.I)
RULES["logo_typography"] = lambda p: bool(LOGO.search(p))

# --- 4. an edit instruction pointing at an image we do not have -----------------------
DANGLING = re.compile(r"^\s*(?:make|change|edit|add|remove|fix|turn|redo|replace|swap)\b[^.]{0,40}?"
                      r"\b(?:this|it|the (?:image|picture|photo|video|attachment))\b", re.I)
RULES["dangling_edit"] = lambda p: bool(DANGLING.search(p))


# =====================================================================================
# driver
# =====================================================================================
def iter_prompts(n):
    """Steps 0-2. Yields (raw, cleaned) for every prompt passing safety and validity."""
    csv.field_size_limit(10 ** 7)
    with open(SRC) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if n and i >= n:
                break
            raw = (row.get("prompt") or "").strip()
            if not raw:
                continue
            try:
                if any(float(row.get(c) or 0) >= SAFETY_MAX for c in SAFETY_COLS):
                    continue
            except ValueError:
                continue
            p = clean(raw)
            if not (LEN_MIN <= len(p) <= LEN_MAX):
                continue
            if sum(1 for ch in p if ord(ch) < 128) / len(p) < ASCII_MIN:
                continue
            yield raw, p


def classify(p):
    """Returns the name of the first rule that rejects p, or None."""
    for k, fn in RULES.items():
        if fn(p):
            return k
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global SRC, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="VidProM_semantic_unique.csv from the VidProM dataset")
    ap.add_argument("--out", default="pool", help="directory --build writes to")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sample-kept", type=int, default=0)
    ap.add_argument("--sample-rejected", type=int, default=0)
    ap.add_argument("--rule", default=None, help="restrict --sample-rejected to one rule")
    ap.add_argument("--n", type=int, default=0, help="rows to scan, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    SRC, OUT_DIR = a.src, a.out

    if a.sample_kept or a.sample_rejected:
        bucket = []
        for _, p in iter_prompts(a.n):
            r = classify(p)
            if a.sample_kept and r is None:
                bucket.append(p)
            elif a.sample_rejected and r is not None and (a.rule is None or r == a.rule):
                bucket.append("[%s] %s" % (r, p))
        random.Random(a.seed).shuffle(bucket)
        for i, p in enumerate(bucket[:max(a.sample_kept, a.sample_rejected)], 1):
            print("%4d| %s" % (i, p[:150]))
        return

    kept, rejected, examples, out = 0, Counter(), {k: [] for k in RULES}, []
    for raw, p in iter_prompts(a.n):
        r = classify(p)
        if r:
            rejected[r] += 1
            if len(examples[r]) < a.show:
                examples[r].append(p)
        else:
            kept += 1
            if a.build:
                out.append(p)

    tot = kept + sum(rejected.values())
    print("prompt_pipeline %s" % VERSION)
    print("scanned         : %s rows" % (a.n or "all"))
    print("passed steps 0-2: %d" % tot)
    print("KEPT            : %d (%.2f%%)" % (kept, 100 * kept / tot))
    print("REJECTED        : %d (%.2f%%)\n" % (tot - kept, 100 * (tot - kept) / tot))
    print("%-20s %8s %9s" % ("rule", "n", "% of pool"))
    for k in RULES:
        print("%-20s %8d %8.2f%%" % (k, rejected[k], 100 * rejected[k] / tot))
    if a.show:
        print("\n--- what each rule removes (read these for false positives) ---")
        for k in RULES:
            if examples[k]:
                print("\n[%s]" % k)
                for e in examples[k]:
                    print("   %s" % e[:130])

    if a.build:
        os.makedirs(OUT_DIR, exist_ok=True)
        pool = f"{OUT_DIR}/pool_{VERSION}.jsonl"
        with open(pool, "w") as f:
            for i, p in enumerate(out):
                f.write(json.dumps({"id": "vp%07d" % i, "prompt": p}) + "\n")
        manifest = {
            "version": VERSION,
            "source": os.path.basename(SRC),
            "source_sha256": sha256(SRC),
            "script_sha256": sha256(os.path.abspath(__file__)),
            "rows_scanned": a.n or "all",
            "passed_validity": tot,
            "kept": kept,
            "rejected_total": tot - kept,
            "rejected_by_rule": dict(rejected),
            "config": {"safety_max": SAFETY_MAX, "len": [LEN_MIN, LEN_MAX],
                       "ascii_min": ASCII_MIN, "safety_cols": SAFETY_COLS},
        }
        with open(f"{OUT_DIR}/pool_{VERSION}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nwrote %s  (%d prompts)" % (pool, len(out)))
        print("wrote %s/pool_%s.manifest.json" % (OUT_DIR, VERSION))


if __name__ == "__main__":
    main()
