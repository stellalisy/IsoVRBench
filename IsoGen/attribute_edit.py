#!/usr/bin/env python3
"""Attribute Edit: rewrite a prompt by swapping trigger phrases for targets.

The procedure that built the released ta_attrib pairs. A swap is list-legal:
its source is a trigger in triggers.json, its target comes from that
trigger's own targets and never from its forbidden list. Every occurrence is
replaced, members of one equivalence class (`swiming`, `swimming`) share a
target, and case, number and determiner are fitted to the prompt.

    edits = TriggerList("triggers.json", "trigger_classes.json")
    edited, swaps = edits.edit(prompt, k=1)
"""
import collections
import json
import random
import re

PLURAL = re.compile(r"(s|es|ies)$", re.I)


DETERMINERS = ("a", "an", "the", "this", "that", "these", "those",
               "his", "her", "their", "its", "my", "your", "our",
               # numerals and quantifiers are determiners too. Leaving them out produced
               # "Five a boisterous adolescent boy twins" in the v4 validation set.
               "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
               "ten", "eleven", "twelve", "several", "many", "few", "some", "both", "all")
_DET_RX = "|".join(DETERMINERS)


def fit_determiner(det, target):
    """Merge a determiner already in the prompt with one carried by the target.

    `a girl` + `a small child` -> `a small child`, not `a a small child`. Then fix a/an
    against whatever word now follows, so `a` + `elderly man` -> `an elderly man`. Case of
    the prompt's determiner is preserved, since it may start the sentence.
    """
    if not det:
        return target
    m = re.match(r"(?:a|an|the)\s+(.*)", target, re.I)
    if m:
        target = m.group(1)
    if det.lower() in ("a", "an"):
        fixed = "an" if re.match(r"[aeiou]", target, re.I) else "a"
        det = fixed.capitalize() if det[0].isupper() else fixed
    return det + " " + target


def match_case(src, tgt):
    """Give the target the casing the source had.

    Triggers and targets are all stored lowercase, but ~12% of occurrences in real prompts
    are not: 6.1% Capitalised, 5.4% mixed, 0.4% UPPERCASE, and 390 of the capitalised ones
    are sentence-initial. Substituting the literal lowercase target lowercases P' but not
    P, so the pair differs in casing as well as in the intended swap.

    build_edit_specs.py does `tgt.capitalize() if orig[:1].isupper() else tgt`, which
    handles the common case but turns "CAT" into "Dog" and "storm trooper" into
    "Storm trooper" -- capitalize() lowercases everything after the first character.
    """
    if src.isupper() and len(src) > 1:
        return tgt.upper()
    words = src.split(" ")
    if len(words) > 1 and all(w[:1].isupper() for w in words if w):
        # a genuinely title-cased source ("Darth Vader") -> title-case the whole target
        return " ".join(w[:1].upper() + w[1:] for w in tgt.split(" "))
    if src[:1].isupper():
        # sentence-initial or a single capitalised word: capitalise only the first
        # character, so "Realistic" -> "2d cartoon animation", not "2d Cartoon Animation"
        return tgt[:1].upper() + tgt[1:]
    return tgt


def is_plural(w):
    return bool(PLURAL.search(w)) and not w.lower().endswith("ss")


def match_number(src, tgt):
    """A plural source needs a plural target: 'mountains -> hill' reads wrong."""
    if " " in tgt:
        return tgt
    if is_plural(src) and not is_plural(tgt):
        return tgt + ("es" if re.search(r"(s|x|z|ch|sh)$", tgt) else "s")
    if not is_plural(src) and is_plural(tgt):
        return tgt                      # leave alone; de-pluralising is error-prone
    return tgt


# Second halves of compounds that bind tightly to the preceding noun. Replacing the first
# half leaves a phrase that is not English: "city scape" -> "snowbound mountain village
# scape". Found by auditing the first 120 generated pairs.
BOUND_NEXT = {
    "scape", "scapes", "side", "sides", "line", "lines", "house", "houses", "yard",
    "yards", "light", "lights", "fall", "falls", "storm", "storms", "bow", "way",
    "ways", "board", "room", "rooms", "top", "front", "back", "ground", "work",
    "works", "shop", "shops", "field", "fields", "walk", "print", "prints",
}

# Words that bind to a FOLLOWING trigger, so replacing the trigger breaks the phrase:
# "light flooded" -> "light dry" (annotation #58, 2026-08-21).
BOUND_PREV = {
    "light", "sun", "moon", "star", "wind", "rain", "snow", "storm", "fire", "water",
    "blood", "ice", "wave", "flood", "candle", "torch", "lamp", "neon", "laser",
}

# Some dimensions only apply in a specific context, but their triggers are bare words that
# occur everywhere. hair-color lists `black`, `brown`, `green`, `pink` and `dark` alongside
# the correctly-scoped `black hair`. In the annotated batch, 7 of 8 `black` swaps were NOT
# hair -- "black ninzi armor", "empty black space", "black night", "fluffy black cat",
# "black to white" -- and one was "handsome black rapper", where a race term was replaced
# with a hair colour. So a bare colour trigger from this dimension only fires when a head
# noun for that dimension is nearby.
DIM_CONTEXT = {
    "hair-color": {"hair", "hairs", "haired", "locks", "mane", "ponytail", "ponytails",
                   "braid", "braids", "bun", "curls", "curl", "beard", "moustache",
                   "mustache", "fringe", "bangs", "dreadlocks", "afro", "wig"},
    "eye-colour": {"eye", "eyes", "eyed", "iris", "irises", "gaze", "pupil", "pupils"},
}

# `blond` and `blonde` are the only hair-color triggers that are themselves HAIR words
# rather than plain colours, and that makes the target's form depend on the slot. Targets
# are stored as bare colours, which is correct when a hair noun follows the trigger
# ("blonde hair" -> "jet black hair") and wrong when one does not ("a blonde woman" ->
# "a jet black woman"). Storing the `-haired` form instead just moves the breakage to the
# other slot ("blonde hair" -> "black-haired hair"), so neither form can be stored -- it
# has to be picked per occurrence. Every other hair-color trigger is a bare colour
# (`black`, `brown`, `pink`, `dark`, `green`) where the bare target reads correctly in
# both slots, so this affects exactly two entries.
HAIR_ADJ_TRIGGERS = {"blond", "blonde"}
ATTRIBUTIVE = {
    "jet black": "jet-black-haired", "bright pink": "pink-haired",
    "bright green": "green-haired", "fiery red": "red-haired",
    "bright red": "red-haired", "platinum blonde": "platinum-blonde-haired",
    "snow white": "white-haired", "snow-white": "white-haired",
    "emerald green": "green-haired", "white": "white-haired",
}


# How far to look for a hair noun before giving up and using the attributive form. The
# scan stops at a preposition, which is what separates "blonde wavy shoulder-length hair"
# (hair noun, bare colour) from "blonde woman with long hair" (the hair belongs to a later
# phrase, so the trigger modifies `woman` and needs the attributive). A plain token budget
# cannot tell those apart; the preposition can, and it is a closed class.
HAIR_LOOKAHEAD = 4
SCAN_STOP = {"with", "in", "on", "at", "of", "and", "or", "from", "to", "by", "near",
             "beside", "under", "over", "against", "while", "as", "for", "into"}


def _hair_noun_follows(mo):
    for w in re.findall(r"[A-Za-z']+", mo.string[mo.end():])[:HAIR_LOOKAHEAD]:
        lw = w.lower()
        if lw in SCAN_STOP:
            return False
        if lw in DIM_CONTEXT["hair-color"]:
            return True
    return False


def hair_target_form(target, mo, dim, trig):
    """Bare colour before a hair noun, attributive `-haired` form otherwise.

    The third slot -- the bare nominal "a blonde in a red dress" -- gets the attributive
    too, giving "a pink-haired in a red dress". That is not idiomatic, because `blonde`
    is a lexicalised noun in English and `pink-haired` is only ever an adjective, but it
    is unambiguous to a video model and it is 5.2% of occurrences. An earlier version
    skipped it via a person-noun allowlist and lost four times as many good occurrences
    to gaps in that list.
    """
    if dim != "hair-color" or trig.lower() not in HAIR_ADJ_TRIGGERS:
        return target
    if _hair_noun_follows(mo):
        return target
    return ATTRIBUTIVE.get(target.lower(),
                           re.sub(r"\s+", "-", target.lower()) + "-haired")


def substitute(prompt, trigger, target, dim=None):
    """Replace every `trigger` in `prompt` with `target`, fitting case, number, determiner.

    The ONE place a swap is performed. The breakage checker calls this too -- an earlier
    version of it re-implemented the substitution and went on reporting 66 failures that
    the builder had already stopped producing.
    """
    pat = re.compile(r"\b(?:(" + _DET_RX + r")\s+)?(" + re.escape(trigger) + r")\b", re.I)

    def repl(mo):
        f = hair_target_form(target, mo, dim, trigger)
        core = match_case(mo.group(2), match_number(mo.group(2), f))
        return fit_determiner(mo.group(1), core)

    return pat.subn(repl, prompt)


CONTEXT_WINDOW = 3


def context_ok(prompt, mo, trig, dim):
    """For a scope-sensitive dimension, require its head noun near the occurrence."""
    need = DIM_CONTEXT.get(dim)
    if not need:
        return True
    if trig.lower() in HAIR_ADJ_TRIGGERS:
        # `blond`/`blonde` ARE hair words, so the generic "needs a hair noun nearby" rule
        # is wrong for them -- it rejected "a blonde woman walks", the commonest slot of
        # all. Every occurrence is in scope; hair_target_form picks the form.
        return True
    if any(w in trig.lower().split() for w in need):
        return True                       # trigger already names the context
    words = re.findall(r"[A-Za-z']+", prompt)
    low = [w.lower() for w in words]
    tw = trig.lower().split()[0]
    for idx, w in enumerate(low):
        if w == tw:
            lo = max(0, idx - CONTEXT_WINDOW)
            if need & set(low[lo:idx + CONTEXT_WINDOW + 1]):
                return True
    return False


def occurrence_ok(prompt, mo):
    """Reject an individual occurrence that whole-word matching gets wrong.

    Two failure modes, both found by auditing the first generated batch:

    proper nouns -- "van" inside the film title "Van Helsing" produced "the movie Ice
    cream truck Helsing". The signal is a capitalised token mid-sentence sitting next to
    another capital ("Van Helsing", "Ball Z", "Black Adam"). This over-fires on capitalised
    style tags such as "Realistic Photography", which would have been fine to swap; that
    costs a few candidate prompts out of 80k and is the right trade against corrupting a
    title.

    split compounds -- "city scape", where the following token only makes sense bound to
    the original word.
    """
    tok = mo.group(0)
    before = prompt[:mo.start()].rstrip()
    after = prompt[mo.end():].lstrip()
    nxt = after.split(" ")[0].strip(",.;:!?\"'“”") if after else ""
    prv = before.split(" ")[-1].strip(",.;:!?\"'“”") if before else ""

    if (tok[:1].isupper() and before and before[-1:] not in ".!?"
            and nxt[:1].isupper()):
        return False
    if nxt.lower() in BOUND_NEXT:
        return False
    if prv.lower() in BOUND_PREV:
        return False
    return True



class TriggerList:
    """triggers.json plus its equivalence classes, indexed for matching."""

    def __init__(self, list_path, class_path):
        E = json.load(open(list_path))["entries"]
        # Classes must come from the SAME list as the entries. They carry merged targets
        # and take priority over per-entry targets, so pairing v2 entries with v1-derived
        # classes silently reinstates unreviewed targets for all 173 class members.
        raw = json.load(open(class_path))
        version = json.load(open(list_path))["manifest"]["version"]
        if raw.get("_source_list") and not raw["_source_list"].startswith(version):
            raise SystemExit("class file was built from %s, but the entries are %s"
                             % (raw["_source_list"], version))
        CLASSES = {k: v for k, v in raw.items() if not k.startswith("_")}
        self.classes = CLASSES
        # review-flagged entries are known-bad, mostly a wrong dimension label
        self.flagged = {e["trigger"].lower() for e in E if e.get("review_flag")}

        # member -> its class, so every variant in a prompt receives the same target
        member_of = {}
        for canon, d in CLASSES.items():
            for m in d["members"]:
                member_of[m.lower()] = canon

        # trigger -> LIST of senses, one per dimension the trigger appears under.
        #
        # This used to be `INFO[t] = (...)` inside the loop, which silently kept only the LAST
        # entry for a trigger. 41 triggers have more than one entry and 38 of them lost targets:
        # `black` is listed under `colour` with 14 targets AND under `hair-color` with 3, so the
        # builder saw only the hair-color sense -- 2 of whose 3 targets are platinum blond(e).
        # That is why round 1 produced "platinum blonde" on 4 of 8 `black` swaps, and why `black
        # ninzi armor` and `empty black space` were treated as hair at all. `red` was seeing 2
        # targets of 17.
        INFO = collections.defaultdict(list)
        _seen = {}
        for e in E:
            t = e["trigger"].lower()
            c = member_of.get(t)
            # A class exists to keep SUBSTITUTION CONSISTENT across variants -- if a prompt says
            # both `swiming` and `swimming`, both must take the same target. It is not a target
            # pool. Taking CLASSES[c]["targets"] merged targets across every dimension the word
            # appears under, so `girl` under human-age-band inherited `boy` and `grown man` from
            # its species entry, undoing the sense separation above. Use the entry's own targets;
            # fall back to the class only for a member with no entry of its own, which is how an
            # added correct spelling (`swimming`) gets any targets at all.
            tg = e["targets"] or (CLASSES[c]["targets"] if c else [])
            # banned is unioned, because a word forbidden for any variant should stay forbidden
            fb = sorted(set(e.get("forbidden_targets", []))
                        | set(CLASSES[c]["banned"] if c else []))
            key = (t, e["dimension"])
            if key in _seen:                       # same trigger AND dimension twice -> union
                prev = _seen[key]
                prev["targets"] = sorted(set(prev["targets"]) | set(tg))
                prev["banned"] = sorted(set(prev["banned"]) | set(fb))
                continue
            sense = {"targets": list(tg), "banned": list(fb), "dimension": e["dimension"],
                     "class": c}
            _seen[key] = sense
            INFO[t].append(sense)
        self.info = INFO

        # Compiled once, not once per prompt. Also index each trigger by its first
        # alphanumeric token: `\btrigger\b` cannot match a prompt that lacks that token,
        # so the index is a sound prefilter and turns a 1,066-regex scan into a handful.
        self._pat = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in INFO}
        self._first = collections.defaultdict(list)
        self._order = {}
        for i, t in enumerate(INFO):
            self._order[t] = i
            tok = re.findall(r"[a-z0-9]+", t.lower())
            self._first[tok[0] if tok else t].append(t)

    def swaps_for(self, prompt):
        """Every list-legal swap available in this prompt, at most one per equivalence class."""
        low = prompt.lower()
        out, used_class = [], set()
        _present = set(re.findall(r"[a-z0-9]+", low))
        # sorted by the original INFO order, NOT by set-iteration order. Only one member of
        # an equivalence class is emitted, so whichever member is reached first decides the
        # targets used -- and set order depends on PYTHONHASHSEED, which would make the whole
        # builder non-deterministic between runs.
        _cands = sorted({t for tok in _present for t in self._first.get(tok, ())},
                        key=self._order.__getitem__)
        for trig in _cands:
            senses = self.info[trig]
            pat = self._pat[trig]
            if not pat.search(low):
                continue

            # Choose the SENSE by context. `black` is both a colour and a hair colour; the
            # scope-sensitive reading wins only when its head noun is present ("black hair"),
            # otherwise the general reading applies ("black armor" is a colour). Ordering
            # scope-sensitive senses first makes the specific one win when it fits.
            mo0 = pat.search(prompt)
            ranked = sorted(senses, key=lambda s: s["dimension"] not in DIM_CONTEXT)
            sense = next((s for s in ranked
                          if context_ok(prompt, mo0, trig, s["dimension"])), None)
            if sense is None:
                continue
            targets, banned = sense["targets"], sense["banned"]
            dim, canon = sense["dimension"], sense["class"]

            key = canon or trig
            if key in used_class:
                continue
            bl = {b.lower() for b in banned}
            cands = [t for t in targets
                     if t.lower() not in bl and not re.search(
                         r"\b" + re.escape(t.lower()) + r"\b", low)]
            if not cands:
                continue
            # every member of the class present in the prompt, so all get the same target
            members = self.classes[canon]["members"] if canon else [trig]
            present, spans, skip = [], [], False
            for m in members:
                ms = list(re.finditer(r"\b" + re.escape(m) + r"\b", prompt, re.I))
                if not ms:
                    continue
                # ALL occurrences get replaced, so if any one of them is a proper noun or a
                # split compound the whole swap is unusable -- we cannot replace some and not
                # others without leaving the source visible in P'.
                if not all(occurrence_ok(prompt, mo) for mo in ms):
                    skip = True
                    break
                if not context_ok(prompt, ms[0], m, dim):
                    skip = True
                    break
                present.append(m)
                spans += [(mo.start(), mo.end()) for mo in ms]
            if skip or not present:
                continue
            used_class.add(key)
            out.append({"trigger": trig, "class": canon, "dimension": dim,
                        "candidates": cands, "members_present": present, "spans": spans})
        return out

    def choose_swaps(self, av, max_k, rng):
        """Pick up to max_k swaps that can actually be applied together.

        Availability is NOT achievability: a prompt with four swaps available may yield
        only two that survive the one-per-dimension and span-overlap filters.
        """
        av = list(av)
        rng.shuffle(av)
        chosen, spans, used_dims = [], [], set()
        for s in av:
            if len(chosen) >= max_k:
                break
            # one swap per dimension: `a young Asian boy` offers `young` and `boy`, both
            # human-age-band at disjoint spans, and applying both gives "an elderly Asian
            # elderly man"
            if s["dimension"] in used_dims:
                continue
            # review-flagged entries are known-bad, mostly a wrong dimension label
            if s["trigger"].lower() in self.flagged:
                continue
            if any(not (e <= s0 or s1 <= b) for b, e in spans for s0, s1 in s["spans"]):
                continue
            chosen.append(s)
            used_dims.add(s["dimension"])
            spans += list(s["spans"])
        return chosen

    def edit(self, prompt, k, seed=0):
        """Apply up to k swaps. Returns the edited prompt and one record per swap,
        or (None, []) when no swap applies. The swaps chosen depend only on the
        prompt, and the targets on the prompt and seed."""
        chosen = self.choose_swaps(self.swaps_for(prompt), k, random.Random(prompt))
        rng = random.Random("%s:%s" % (seed, prompt))
        edited, applied = prompt, []
        for s in chosen:
            tgt = rng.choice(s["candidates"])
            n = 0
            for m in s["members_present"]:
                edited, c = substitute(edited, m, tgt, s["dimension"])
                n += c
            if n:
                applied.append({"trigger": s["trigger"], "target": tgt,
                                "dimension": s["dimension"], "occurrences": n})
        if not applied or edited == prompt:
            return None, []
        return edited, applied
