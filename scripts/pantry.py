#!/usr/bin/env python3
"""
Pantry -- the whole ingestion pipeline in one file.

Reads a URL or pasted text from a GitHub issue, extracts a structured recipe
with Claude, validates it, and writes it to recipes/pending/.

Single file on purpose: it makes setup from a phone two files instead of eight.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import pathlib
import re
import time
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from datetime import date
from urllib.parse import urlparse

import anthropic
import requests



# ==========================================================================
# VALIDATION
# ==========================================================================

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- vocab

VEG_STATUSES = {
    "inherently-veg",
    "easily-adaptable",
    "needs-parallel-cook",
    "not-adaptable",
}
RECIPE_TYPES = {"meal", "component"}
STATUSES = {"pending", "approved", "rejected"}
SOURCE_TYPES = {"personal", "instagram", "tiktok", "web", "text"}
SECTIONS = {
    "produce", "pantry", "spices", "canned",
    "dairy", "meat", "frozen", "bakery", "other",
}
UNITS = {
    "g", "kg", "oz", "lb",
    "ml", "l", "tsp", "tbsp", "cup", "fl_oz",
    "each", "clove", "can", "packet", "bunch", "slice", "pinch", "block",
}

# Ingredients that are commonly assumed vegetarian but are not.
# Anything here that appears without an accompanying swap gets flagged.
HIDDEN_ANIMAL = {
    "worcestershire": "usually contains anchovies",
    "fish sauce": "anchovy based",
    "oyster sauce": "oyster extract",
    "anchov": "fish",
    "parmesan": "traditionally animal rennet",
    "parmigiano": "traditionally animal rennet",
    "pecorino": "traditionally animal rennet",
    "gelatin": "animal collagen",
    "lard": "pork fat",
    "tallow": "beef fat",
    "bone broth": "animal stock",
    "chicken broth": "animal stock",
    "chicken stock": "animal stock",
    "beef broth": "animal stock",
    "beef stock": "animal stock",
    "duck fat": "animal fat",
    "bacon": "pork",
    "pancetta": "pork",
    "caesar": "dressing usually contains anchovy",
}

# Obvious meat terms -- used to sanity-check an "inherently-veg" claim.
MEAT_TERMS = {
    "chicken", "beef", "pork", "lamb", "veal", "turkey", "duck", "bacon",
    "pancetta", "prosciutto", "sausage", "chorizo", "ham", "brisket",
    "shrimp", "prawn", "salmon", "tuna", "cod", "crab", "lobster",
    "clam", "mussel", "scallop", "oyster", "fish", "anchov",
}

# Household standing constraints.
BANNED_PATTERNS = [
    (r"\bskin[- ]on\b", "household avoids skin-on cuts"),
    (r"\blamb\b", "household does not eat lamb"),
]

# Never scale linearly with servings. Matched as whole phrases against the
# item name only -- not prep or notes, or "garlic cloves" trips the "clove"
# spice entry and "green chiles" trips "chile".
NONLINEAR = {
    "salt", "baking soda", "baking powder", "yeast", "cayenne",
    "chili powder", "chile powder", "red pepper flakes", "pepper flakes",
    "black pepper", "white pepper", "nutmeg", "ground cloves",
    "cinnamon", "saffron", "vanilla extract", "msg",
}

REF_RE = re.compile(r"\{(i\d+)\}")


# ---------------------------------------------------------------- result

@dataclass
class Result:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        out = []
        for e in self.errors:
            out.append(f"  ERROR    {e}")
        for w in self.warnings:
            out.append(f"  WARNING  {w}")
        return "\n".join(out) or "  clean"


# ---------------------------------------------------------------- helpers

def _text_of(ing: dict) -> str:
    return " ".join(
        str(ing.get(k) or "") for k in ("item", "prep", "note")
    ).lower()


def _all_text(r: dict) -> str:
    parts = [r.get("title", ""), r.get("description", "")]
    parts += [_text_of(i) for i in r.get("ingredients", [])]
    parts += [s.get("text", "") for s in r.get("steps", [])]
    return " ".join(parts).lower()


# ---------------------------------------------------------------- checks

def _check_required(r: dict, res: Result) -> None:
    for f in ("id", "title", "ingredients", "steps", "vegetarian", "rawText"):
        if f not in r:
            res.err(f"missing required field: {f}")

    if "id" in r and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", r["id"]):
        res.err(f"id is not a clean slug: {r['id']!r}")

    if r.get("status") not in STATUSES:
        res.err(f"status must be one of {sorted(STATUSES)}, got {r.get('status')!r}")

    if r.get("recipeType", "meal") not in RECIPE_TYPES:
        res.err(f"recipeType invalid: {r.get('recipeType')!r}")

    st = (r.get("source") or {}).get("type")
    if st not in SOURCE_TYPES:
        res.err(f"source.type invalid: {st!r}")

    if not str(r.get("rawText", "")).strip():
        res.err("rawText is empty -- the original must always be preserved")


def _check_ingredients(r: dict, res: Result) -> set[str]:
    ids: set[str] = set()
    for n, ing in enumerate(r.get("ingredients", [])):
        iid = ing.get("id", f"<index {n}>")
        if iid in ids:
            res.err(f"duplicate ingredient id: {iid}")
        ids.add(iid)

        if not ing.get("item"):
            res.err(f"{iid}: no item name")

        if ing.get("section") not in SECTIONS:
            res.warn(f"{iid}: unknown section {ing.get('section')!r} -> shopping list will bucket as 'other'")

        unit = ing.get("unit")
        if unit is not None and unit not in UNITS:
            res.warn(f"{iid}: unrecognised unit {unit!r}")

        qty, vague = ing.get("qty"), ing.get("vague", False)

        # The fabrication check: a quantity that appears nowhere in the source.
        # A non-empty note is treated as documenting the derivation -- "2 32 oz
        # boxes" -> 64, or "2-3 tablespoons" -> 2.5. That's why the prompt
        # requires a note whenever the number isn't lifted verbatim.
        if qty is not None and not vague and not (ing.get("note") or "").strip():
            if not _qty_in_source(qty, r.get("rawText", "")):
                res.warn(
                    f"{iid} ({ing.get('item')}): qty {qty} not found in source and no note "
                    f"explaining it -- verify it wasn't invented"
                )

        if qty is None and not vague:
            res.err(f"{iid}: qty is null but vague is false -- set vague true and put the phrasing in note")

        if vague and not (ing.get("note") or "").strip():
            res.err(f"{iid}: marked vague but note is empty -- the original phrasing is the value")

        # Scaling sanity. A component recipe *is* the seasoning -- doubling a
        # spice blend genuinely doubles every line -- so skip it there.
        if r.get("recipeType", "meal") != "component":
            name = str(ing.get("item") or "").lower()
            if ing.get("scalable", True) and any(t in name for t in NONLINEAR):
                res.warn(f"{iid} ({ing.get('item')}): marked scalable -- seasonings usually shouldn't scale linearly")

    return ids


def _qty_in_source(qty, raw: str) -> bool:
    """Loose check that a number actually appears in the source."""
    raw = raw.lower()
    cands = set()
    if float(qty) == int(qty):
        cands.add(str(int(qty)))
    cands.add(str(qty).rstrip("0").rstrip("."))
    # common fraction spellings
    frac = {0.25: ["1/4", ".25"], 0.5: ["1/2", ".5"], 0.75: ["3/4", ".75"],
            0.33: ["1/3"], 0.67: ["2/3"], 1.5: ["1 1/2", "1.5"],
            2.5: ["2 1/2", "2.5"], 4.5: ["4-5", "4 1/2"], 7: ["7-8", "6-8"],
            2.0: ["2-3"]}
    cands.update(frac.get(round(float(qty), 2), []))
    return any(c in raw for c in cands if c)


def _check_steps(r: dict, ids: set[str], res: Result) -> None:
    steps = r.get("steps", [])
    if not steps:
        res.err("no steps -- even a spice blend needs one 'combine' step")

    used: set[str] = set()
    for s in steps:
        sid = s.get("id", "<no id>")
        if not (s.get("text") or "").strip():
            res.err(f"{sid}: empty step text")
        if not (s.get("title") or "").strip():
            res.warn(f"{sid}: no title -- cooking mode shows this as the header")

        arr = set(s.get("ingredientIds", []))
        inline = set(REF_RE.findall(s.get("text", "")))

        for ref in arr - ids:
            res.err(f"{sid}: ingredientIds references unknown {ref}")
        for ref in inline - ids:
            res.err(f"{sid}: step text references unknown {ref}")
        for ref in inline - arr:
            res.err(f"{sid}: {ref} used inline but missing from ingredientIds")

        used |= arr

        secs = s.get("timerSeconds")
        if secs is not None:
            if not isinstance(secs, (int, float)) or secs <= 0:
                res.err(f"{sid}: timerSeconds must be a positive number, got {secs!r}")
            elif secs > 60 * 60 * 24:
                res.warn(f"{sid}: timer over 24h -- check the units")

    for orphan in sorted(ids - used):
        res.warn(f"{orphan} appears in ingredients but is never used in a step")


def _check_vegetarian(r: dict, ids: set[str], res: Result) -> None:
    veg = r.get("vegetarian") or {}
    status = veg.get("status")

    if status not in VEG_STATUSES:
        res.err(f"vegetarian.status invalid: {status!r}")
        return

    if not (veg.get("notes") or "").strip() and status != "inherently-veg":
        res.err(f"vegetarian.notes required when status is {status} -- this is read at 5pm on a weeknight")

    for sw in veg.get("swaps", []):
        tgt = sw.get("ingredientId")
        if tgt not in ids:
            res.err(f"vegetarian swap points at unknown ingredient {tgt!r}")
        if not (sw.get("with") or "").strip():
            res.err(f"vegetarian swap for {tgt} has no replacement")

    # Does the claim survive contact with the ingredient list?
    swapped = {sw.get("ingredientId") for sw in veg.get("swaps", [])}
    flagged: set[str] = set()

    for ing in r.get("ingredients", []):
        txt = _text_of(ing)
        iid = ing.get("id")

        for term, why in HIDDEN_ANIMAL.items():
            if term in txt:
                flagged.add(iid)
                if iid in swapped:
                    break
                # "inherently-veg" and "easily-adaptable" both assert the dish
                # is servable to the vegetarian. An unswapped animal product
                # contradicts that outright.
                if status in ("inherently-veg", "easily-adaptable"):
                    res.err(
                        f"{iid} ({ing.get('item')}) is {why}, status is {status}, "
                        f"and no swap targets it -- add a swap or change the status"
                    )
                else:
                    res.warn(f"{iid} ({ing.get('item')}): {why} -- no swap recorded")
                break

        if status == "inherently-veg" and iid not in swapped:
            for m in MEAT_TERMS:
                if m in txt:
                    flagged.add(iid)
                    res.err(f"{iid} ({ing.get('item')}) looks like meat but status is inherently-veg")
                    break

    # A swap aimed at something with no animal-product signal is the failure
    # mode that reads as correct: well-formed, plausible, and pointing at the
    # wrong line.
    for sw in veg.get("swaps", []):
        tgt = sw.get("ingredientId")
        if tgt in ids and tgt not in flagged:
            name = next((i.get("item") for i in r["ingredients"] if i.get("id") == tgt), tgt)
            res.err(
                f"vegetarian swap targets {tgt} ({name}), which has no animal product in it "
                f"-- the swap is probably pointing at the wrong ingredient"
            )


def _check_household(r: dict, res: Result) -> None:
    """Standing constraints: no skin-on, no lamb."""
    blob = _all_text(r)
    for pat, why in BANNED_PATTERNS:
        if re.search(pat, blob):
            res.warn(f"{why} -- found in this recipe, check sourceNotes for a substitution")


def _check_times(r: dict, res: Result) -> None:
    a, t = r.get("activeTimeMin"), r.get("totalTimeMin")
    if a is not None and t is not None and a > t:
        res.err(f"activeTimeMin ({a}) exceeds totalTimeMin ({t})")

    if r.get("recipeType", "meal") == "meal" and not r.get("serves"):
        res.warn("no serves value -- meal planner can't scale this")

    total_timer = sum(
        s.get("timerSeconds") or 0 for s in r.get("steps", [])
    ) / 60
    if t and total_timer > t * 1.25:
        res.warn(f"step timers sum to {total_timer:.0f}m but totalTimeMin is {t}")


# ---------------------------------------------------------------- entry

def validate(recipe: dict) -> Result:
    res = Result()
    if not isinstance(recipe, dict):
        res.err("not a JSON object")
        return res

    if "error" in recipe:
        res.err(f"extractor returned an error: {recipe.get('reason', 'no reason given')}")
        return res

    _check_required(recipe, res)
    ids = _check_ingredients(recipe, res)
    _check_steps(recipe, ids, res)
    _check_vegetarian(recipe, ids, res)
    _check_household(recipe, res)
    _check_times(recipe, res)
    return res


# ==========================================================================
# FETCHING
# ==========================================================================

import re
import subprocess
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

# Below this, a caption almost certainly isn't a full recipe.
CAPTION_MIN_CHARS = 220


@dataclass
class Fetched:
    text: str
    source_type: str          # instagram | tiktok | web | text
    url: str | None
    author: str | None
    used_transcript: bool = False
    note: str | None = None   # anything the reviewer should know
    images: list[tuple[str, str]] = field(default_factory=list)  # (media_type, base64)


IMAGE_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".heic": "image/jpeg",
}


# GitHub rewrites a pasted photo into one of these markdown links.
ATTACHMENT_RE = re.compile(
    r"!\[[^\]]*\]\((https://(?:user-images\.githubusercontent\.com|github\.com/user-attachments)/[^\s)]+)\)"
    r"|<img[^>]+src=\"(https://(?:user-images\.githubusercontent\.com|github\.com/user-attachments)/[^\"]+)\"",
    re.I,
)


def attachment_urls(body: str) -> list[str]:
    return [u for m in ATTACHMENT_RE.finditer(body) for u in m.groups() if u]


def fetch_attachments(urls: list[str]) -> Fetched:
    """Photos pasted straight into the issue -- no Shortcut needed."""
    images = []
    for url in urls[:3]:                       # a recipe spread is rarely more
        r = requests.get(url, timeout=30, headers={"User-Agent": UA})
        r.raise_for_status()
        blob = r.content
        if len(blob) > 5_000_000:
            raise ValueError("that image is over 5MB")
        media = r.headers.get("content-type", "image/jpeg").split(";")[0]
        if media not in IMAGE_TYPES.values():
            media = "image/jpeg"
        images.append((media, base64.b64encode(blob).decode()))

    if not images:
        raise ValueError("no readable image in the issue")

    return Fetched(
        text="",
        source_type="image",
        url=None,
        author=None,
        note=f"read from {len(images)} attached image(s)",
        images=images,
    )


def looks_like_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_TYPES


def fetch_image(path: str) -> Fetched:
    """A screenshot or photo committed to the repo, referenced by path."""
    root = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    full = (root / path.lstrip("/")).resolve()
    if not str(full).startswith(str(root.resolve())):
        raise ValueError("image path escapes the repository")
    if not full.exists():
        raise FileNotFoundError(f"no file at {path}")

    data = full.read_bytes()
    if len(data) > 5_000_000:
        raise ValueError("image is over 5MB — take a smaller screenshot")

    media = IMAGE_TYPES[full.suffix.lower()]
    return Fetched(
        text="",
        source_type="image",
        url=None,
        author=None,
        note=f"read from the image at {path}",
        images=[(media, base64.b64encode(data).decode())],
    )


def classify(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if "instagram.com" in host:
        return "instagram"
    if "tiktok.com" in host:
        return "tiktok"
    return "web"


def fetch(source: str, allow_transcription: bool = True) -> Fetched:
    """`source` is a URL or a blob of pasted text."""
    if not re.match(r"https?://", source.strip()):
        return Fetched(text=source.strip(), source_type="text", url=None, author=None)

    url = source.strip()
    kind = classify(url)
    return _fetch_social(url, kind, allow_transcription) if kind in ("instagram", "tiktok") \
        else _fetch_web(url)


# ------------------------------------------------------------------ web

def _fetch_web(url: str) -> Fetched:
    html = requests.get(url, headers={"User-Agent": UA}, timeout=25).text

    # Recipe sites almost all publish JSON-LD. When present it's far cleaner
    # than anything we'd get by stripping the page, and it skips the 2,000
    # words about the author's trip to Tuscany.
    ld = _jsonld_recipe(html)
    if ld:
        return Fetched(text=ld, source_type="web", url=url,
                       author=urlparse(url).hostname,
                       note="Read from the page's structured recipe data.")

    return Fetched(text=_strip_html(html), source_type="web", url=url,
                   author=urlparse(url).hostname)


def _jsonld_recipe(html: str) -> str | None:
    import json
    blocks = re.findall(
        r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>',
        html, re.S | re.I)
    for b in blocks:
        try:
            data = json.loads(b.strip())
        except Exception:
            continue
        for node in _walk(data):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if "Recipe" in types:
                return json.dumps(node, indent=1)[:12000]
    return None


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _strip_html(html: str) -> str:
    html = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>|</p>|</li>|</h[1-6]>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#39;", "'").replace("&quot;", '"'))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()[:14000]


# --------------------------------------------------------------- social

def _fetch_social(url: str, kind: str, allow_transcription: bool) -> Fetched:
    """Caption AND audio, always -- they carry different halves of the recipe.

    This used to skip transcription whenever the caption looked long enough,
    on the assumption that a long caption was a complete one. For a reel that
    is exactly backwards: the caption is usually the ingredient list, and the
    method lives in the voiceover. Steps like "boil the limes first" appear
    only in the audio, and a caption-length check will never catch that.
    """
    if "/stories/" in url:
        raise ValueError(
            "Instagram stories need a logged-in session and disappear after 24 "
            "hours, so nothing here can fetch one. Screenshot it and attach the "
            "image to an issue instead -- that path reads recipes fine.")

    meta = _ytdlp_meta(url)
    caption = (meta.get("description") or "").strip()
    author = meta.get("uploader") or meta.get("channel")
    trace(f"caption: {len(caption)} chars")

    if not allow_transcription:
        if len(caption) < CAPTION_MIN_CHARS:
            note = "Transcription is off and the caption is thin -- steps may be missing."
        else:
            note = ("Transcription is off, so anything said only in the video "
                    "was not captured.")
        return Fetched(text=caption, source_type=kind, url=url, author=author, note=note)

    try:
        transcript = _transcribe(url)
    except Exception as e:
        trace(f"transcription failed: {e}")
        return Fetched(
            text=caption, source_type=kind, url=url, author=author,
            note=(f"Could not read the audio ({e}), so this is the caption only. "
                  "Any step that was only spoken is missing."))

    if not transcript:
        return Fetched(text=caption, source_type=kind, url=url, author=author,
                       note="No speech found in the video; caption only.")

    combined = (
        "CAPTION (as written by the author -- trust these amounts):\n"
        f"{caption}\n\n"
        "SPOKEN TRANSCRIPT (what they say in the video -- the method is usually "
        "here, and often includes steps the caption omits):\n"
        f"{transcript}"
    )
    return Fetched(text=combined, source_type=kind, url=url, author=author,
                   used_transcript=True,
                   note="Built from the caption and the voiceover together -- "
                        "check amounts before cooking.")


def _ytdlp_meta(url: str) -> dict:
    import json
    out = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-warnings", "--skip-download", url],
        capture_output=True, text=True, timeout=90)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def _load_whisper():
    """Installed on demand. It drags in torch (a few hundred MB), so paying
    that cost on every run to serve the minority of posts with a thin caption
    would be silly -- we only reach here when there's nothing else to read."""
    try:
        import whisper
        return whisper
    except ImportError:
        pass

    trace("caption too short -- installing whisper to read the audio (slow, one-off per run)")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "openai-whisper"],
                   check=True, timeout=900)
    import whisper
    return whisper


def _transcribe(url: str) -> str:
    """Audio only -- much smaller and faster than pulling the video."""
    with tempfile.TemporaryDirectory() as tmp:
        trace("downloading audio")
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3",
             "-o", f"{tmp}/a.%(ext)s", "--no-warnings", url],
            capture_output=True, timeout=240, check=True)

        whisper = _load_whisper()
        trace("transcribing")
        model = whisper.load_model("base")
        text = model.transcribe(f"{tmp}/a.mp3")["text"].strip()
        trace(f"transcript: {len(text)} chars")
        return text


# ==========================================================================
# EXTRACTION
# ==========================================================================

import json
import os
import re
from datetime import date

import anthropic


MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3

SYSTEM = """You convert unstructured recipe content into a strict JSON schema.

Output ONLY valid JSON. No markdown fences, no preamble, no commentary.

## Household context

A family of six in Austin: two adults, four kids (6, 9, 12, 14). One adult is
vegetarian and the household is practiced at splitting meals. The cook is
skilled and equipped: gas range, well-seasoned cast iron, carbon steel wok,
Ooni, Big Green Egg.

Standing constraints. Apply them silently -- only mention one in sourceNotes
if you ACTUALLY changed something because of it. Never state that a constraint
did not apply; a recipe with no meat in it should say nothing about meat.
- No skin-on cuts. Substitute the skinless version.
- No lamb. Extract faithfully but set household.inRotation to false.
- The household stocks vegetarian Worcestershire. Never flag it.

## Rules

1. NEVER invent a quantity. If the source doesn't give one, use qty: null,
   vague: true, and put the source's own phrasing in `note`. A missing amount
   is fine. A fabricated one is not.

2. If you derive a number rather than lifting it verbatim -- "2 32oz boxes"
   becoming 64, or "2-3 tablespoons" becoming 2.5 -- you MUST record the
   original phrasing in `note`. A derived number with an empty note gets
   flagged as invented.

3. Preserve vague measurements. "A couple glugs", "big handful", "to taste"
   are valid values, not problems to solve.

4. rawText is mandatory and verbatim. Copy the source exactly. Never summarise.

5. scalable: false for salt, leavening, chilies, and strong spices -- they
   don't multiply linearly with servings.

6. vegetarian.status is the most-used field in the whole app. Choose carefully:
   - inherently-veg      no meat, no changes needed
   - easily-adaptable    one swap, no extra pan (chicken stock -> vegetable)
   - needs-parallel-cook protein must be cooked separately
   - not-adaptable       meat is structural (brisket, birria)

   Every non-vegetarian ingredient must have a matching entry in
   vegetarian.swaps pointing at that ingredient's exact id. Watch for hidden
   ones: fish sauce, oyster sauce, anchovy, gelatin, animal-rennet cheeses,
   chicken or beef stock.

   vegetarian.notes gets read at 5pm on a weeknight. Concrete, one or two
   sentences, tells the cook what to actually do.

7. activeTimeMin is hands-on only. totalTimeMin includes unattended time.
   Estimate from the steps if the source gives none.

8. Steps reference ingredients inline as {i01}. Every id used inline must also
   appear in that step's ingredientIds. Set unattended: true for anything not
   needing the cook present. Give every step a short imperative title.

9. timerSeconds whenever a step has a duration. Midpoint for a range.

10. status is always "pending". recipeType is "component" for something that
    is an input to other recipes (spice blend, sauce, dough) rather than a
    meal.

11. If the content is not a recipe, output exactly:
    {"error": "not_a_recipe", "reason": "<one line>"}

12. If it is a recipe but critically incomplete, extract what exists and add
    "extractionWarnings": ["..."] describing what's missing.

Also set:
- sourceNotes: ONLY things that change what the cook does -- a substitution you
  applied, a step the source skipped, a quantity that looks wrong, a time that
  omits unattended work. Leave it null if there is nothing of that kind. Never
  use it to narrate your own process, restate the schema, or report that a
  household rule was not triggered.
- confidence: "high" | "medium" | "low". Use "low" for anything reconstructed
  from spoken audio alone.

## Schema

{SCHEMA}
"""

SCHEMA = """
{
  "id": "kebab-case-slug",
  "title": str,
  "description": str,
  "source": {"type": "instagram|tiktok|web|text|personal", "url": str|null,
             "author": str|null, "addedBy": str},
  "dateAdded": "YYYY-MM-DD",
  "status": "pending",
  "recipeType": "meal|component",
  "componentOf": [str],
  "yield": str|null,
  "cuisine": str,
  "tags": [str],
  "mainProtein": str,
  "equipment": [str],
  "activeTimeMin": int,
  "totalTimeMin": int,
  "serves": int|null,
  "vegetarian": {
    "status": "inherently-veg|easily-adaptable|needs-parallel-cook|not-adaptable",
    "notes": str,
    "swaps": [{"ingredientId": "iNN", "with": str}]
  },
  "ingredients": [{
    "id": "iNN", "qty": number|null, "unit": str|null, "item": str,
    "prep": str|null, "note": str|null,
    "section": "produce|pantry|spices|canned|dairy|meat|frozen|bakery|other",
    "vague": bool, "optional": bool, "scalable": bool
  }],
  "steps": [{
    "id": "sNN", "title": str, "text": str,
    "timerSeconds": int|null, "unattended": bool, "ingredientIds": ["iNN"]
  }],
  "household": {"lastCooked": null, "timesCooked": null, "ratings": {},
                "kidNotes": null, "inRotation": true},
  "rawText": str,
  "sourceNotes": str|null,
  "confidence": "high|medium|low"
}
"""


class ExtractionFailed(Exception):
    pass


def extract(fetched: Fetched, added_by: str = "jess",
            client: anthropic.Anthropic | None = None) -> tuple[dict, Result]:
    client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user = _build_user_message(fetched, added_by)
    messages = [{"role": "user", "content": user}]
    last: Result | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _call(client, messages)
        recipe = _parse(raw)

        if "error" in recipe:
            raise ExtractionFailed(recipe.get("reason", "not a recipe"))

        _stamp(recipe, fetched, added_by)
        result = validate(recipe)
        last = result

        if result.ok:
            return recipe, result

        if attempt == MAX_ATTEMPTS:
            break

        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
                "The extraction failed validation. Fix these and return the "
                "complete corrected JSON — same rules, no commentary:\n\n"
                + "\n".join(f"- {e}" for e in result.errors)},
        ]

    # Three attempts and still failing. Hand it over anyway: it goes to the
    # review queue, not the collection, and a human looking at a flagged
    # recipe beats silently binning two minutes of work.
    recipe["confidence"] = "low"
    recipe["extractionErrors"] = [str(e) for e in (last.errors if last else [])]
    return recipe, last


def _build_user_message(f: Fetched, added_by: str) -> list[dict]:
    head = [f"Source type: {f.source_type}"]
    if f.url:
        head.append(f"URL: {f.url}")
    if f.author:
        head.append(f"Author: {f.author}")
    if f.used_transcript:
        head.append(
            "NOTE: this has TWO sources -- a written caption and a transcript of "
            "the voiceover. Use both. They cover different things:\n"
            "- Amounts and ingredient names: trust the CAPTION. Spoken numbers "
            "are easy to mishear.\n"
            "- Method and technique: mostly in the TRANSCRIPT. Cooks routinely "
            "describe steps out loud that never make it into the caption -- "
            "soaking, boiling, resting, straining, blooming spices. Every such "
            "step must appear in your steps array.\n"
            "- If the transcript mentions an ingredient the caption omits, include "
            "it with qty null and vague true rather than guessing an amount.\n"
            "- Where the two genuinely conflict, follow the caption and say so in "
            "sourceNotes.\n"
            "Set confidence to medium (not high) since speech recognition is "
            "imperfect.")
    if f.images:
        head.append(
            "NOTE: the recipe is in the attached image. Transcribe it exactly as "
            "written into rawText -- do not tidy it up. If any quantity is cut "
            "off, blurred, or you are guessing at a digit, leave qty null, set "
            "vague true, and say so in sourceNotes. Never fill in a number you "
            "cannot actually read. Set confidence to medium or low."
        )
    head.append(f"addedBy: {added_by}")

    text = "\n".join(head)
    if f.text.strip():
        text += "\n\n---\n\n" + f.text

    blocks: list[dict] = []
    for media, b64 in f.images:
        blocks.append({"type": "image",
                       "source": {"type": "base64", "media_type": media, "data": b64}})
    blocks.append({"type": "text", "text": text})
    return blocks


def _call(client, messages) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM.replace("{SCHEMA}", SCHEMA),
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _parse(raw: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip(), flags=re.M)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            raise ExtractionFailed("model did not return JSON")
        return json.loads(m.group(0))


def _stamp(recipe: dict, f: Fetched, added_by: str) -> None:
    """Fields we own, not the model."""
    recipe["status"] = "pending"
    recipe["dateAdded"] = date.today().isoformat()
    recipe.setdefault("source", {})
    recipe["source"].update({
        "type": f.source_type, "url": f.url,
        "author": f.author, "addedBy": added_by,
    })
    if f.note:
        existing = recipe.get("sourceNotes") or ""
        recipe["sourceNotes"] = (existing + " " + f.note).strip()
    # rawText must be the source, not the model's echo of it.
    recipe["rawText"] = f.text


# ==========================================================================
# GITHUB ISSUE RUNNER
# ==========================================================================

#!/usr/bin/env python3
"""
Bridge between a GitHub issue and the extraction pipeline.

The Shortcut opens an issue whose body is a URL (or pasted recipe text).
This reads it, runs the pipeline, writes the recipe to recipes/pending/,
and leaves a readable comment on the issue either way.

Nothing raises. A failure still needs to produce a comment the person can
act on from their phone.
"""




# In Actions this is the checkout root. Falls back to the parent of scripts/
# so the file still works if you run it locally.
ROOT = pathlib.Path(
    os.environ.get("GITHUB_WORKSPACE")
    or pathlib.Path(__file__).resolve().parents[1]
)
PENDING = ROOT / "recipes" / "pending"
COMMENT = ROOT / "comment.md"

# GitHub username -> who it shows up as in the app.
PEOPLE = {
    # "jdw818": "jess",
}


def out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"::notice::{key}={value}", flush=True)


def say(md: str) -> None:
    """The reason goes on the issue -- and into the job log, so a run that
    saves nothing explains itself without anyone opening the issue."""
    COMMENT.write_text(md)
    print("\n----- what went on the issue -----", flush=True)
    print(md, flush=True)
    print("----------------------------------\n", flush=True)


def trace(msg: str) -> None:
    print(f"[pantry] {msg}", flush=True)


def existing_ids() -> dict[str, str]:
    found = {}
    for folder in ("pending", "approved", "rejected"):
        d = ROOT / "recipes" / folder
        if d.exists():
            for f in d.glob("*.json"):
                found[f.stem] = folder
    return found


def existing_urls() -> dict[str, str]:
    """Source URLs already ingested -- the only reliable 'same recipe' signal."""
    found = {}
    for folder in ("pending", "approved", "rejected"):
        d = ROOT / "recipes" / folder
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                url = (json.loads(f.read_text()).get("source") or {}).get("url")
            except Exception:
                continue
            if url:
                found[url.split("?")[0].rstrip("/")] = f.stem
    return found


def free_id(base: str, taken: dict[str, str]) -> str:
    """Two different recipes can slug the same. Don't throw the second away."""
    if base not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base}-{n}"
        if candidate not in taken:
            return candidate
    return f"{base}-{int(time.time())}"


def main() -> int:
    body = (os.environ.get("ISSUE_BODY") or "").strip()
    author = os.environ.get("ISSUE_AUTHOR", "")
    added_by = PEOPLE.get(author, author or "unknown")

    if not body:
        say("Nothing in the issue body. Send a link, paste the recipe text, "
            "or attach a photo.")
        out("saved", "false")
        return 0

    # The Shortcut sends a bare URL, but a person typing it might add words.
    m = re.search(r"https?://\S+", body)
    source = m.group(0) if m else body

    trace(f"issue body: {len(body)} chars, author={author!r} -> addedBy={added_by!r}")
    trace(f"first line: {body.splitlines()[0][:120]!r}" if body.splitlines() else "empty body")

    # A photo pasted into the issue, or a path committed by a Shortcut.
    attached = attachment_urls(body)
    img = re.search(r"(?:^|\s)([\w./-]+\.(?:jpe?g|png|gif|webp|heic))\s*$", body, re.I)

    trace(f"route: {'attachment' if attached else 'repo-image' if (img and not m) else 'url' if m else 'text'}")

    try:
        if attached:
            fetched = fetch_attachments(attached)
        elif img and not m:
            fetched = fetch_image(img.group(1))
        else:
            fetched = fetch(
                source,
                allow_transcription=os.environ.get("PANTRY_TRANSCRIBE") == "1",
            )
    except Exception as e:
        say(
            "**Couldn't read that link.**\n\n"
            f"```\n{e}\n```\n\n"
            "Open the page, copy the recipe text, and paste it into a new issue "
            "instead -- or attach a photo of the recipe to an issue."
        )
        out("saved", "false")
        return 0

    if not fetched.images and len(fetched.text.strip()) < 60:
        say(
            "**Not enough text there to work with.**\n\n"
            "The page or caption came back nearly empty. Paste the recipe text directly."
        )
        out("saved", "false")
        return 0

    try:
        recipe, result = extract(fetched, added_by=added_by)
    except ExtractionFailed as e:
        # Only raised when the source genuinely isn't a recipe.
        say(f"**That doesn't look like a recipe.**\n\n```\n{e}\n```")
        out("saved", "false")
        return 0
    except Exception as e:
        say(f"**Extraction failed.**\n\n```\n{type(e).__name__}: {e}\n```\n\nTry again.")
        out("saved", "false")
        return 0

    trace(f"extracted {recipe.get('id')!r} — {len(recipe.get('ingredients', []))} ingredients, "
          f"confidence={recipe.get('confidence')}, validation={'ok' if result.ok else 'FAILED'}")
    for e in recipe.get("extractionErrors", []):
        trace(f"validation error (saving anyway, flagged): {e}")
    if result.warnings:
        for w in result.warnings:
            trace(f"warning: {w}")

    seen = existing_ids()
    trace(f"{len(seen)} recipes already on disk")

    # Same source URL means we really have already ingested this one.
    src_url = (recipe.get("source") or {}).get("url")
    if src_url:
        prior = existing_urls().get(src_url.split("?")[0].rstrip("/"))
        if prior:
            say(f"**{recipe['title']}** came from a link already saved as "
                f"`{prior}`. Nothing to do.")
            out("saved", "false")
            return 0

    # A slug collision is just two recipes with similar names -- keep both.
    original = recipe["id"]
    recipe["id"] = free_id(original, seen)
    renamed = recipe["id"] != original
    if renamed:
        trace(f"slug {original!r} was taken; saving as {recipe['id']!r}")

    PENDING.mkdir(parents=True, exist_ok=True)
    path = PENDING / f"{recipe['id']}.json"
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n")
    trace(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    report = _report(recipe, result, fetched)
    if renamed:
        report += (f"\n\nSaved as `{recipe['id']}` — `{original}` was taken by a "
                   f"different recipe.")
    say(report)
    out("saved", "true")
    out("title", recipe["title"].replace("\n", " "))
    return 0


VEG_LABEL = {
    "inherently-veg": "Vegetarian as written",
    "easily-adaptable": "Easily adapted",
    "needs-parallel-cook": "Needs a parallel cook",
    "not-adaptable": "Not adaptable",
}


def _report(recipe: dict, result, fetched) -> str:
    veg = recipe["vegetarian"]
    lines = [
        f"### {recipe['title']}",
        "",
        recipe.get("description", ""),
        "",
        f"**{VEG_LABEL.get(veg['status'], veg['status'])}** — {veg.get('notes') or 'no notes'}",
        "",
        f"{recipe.get('activeTimeMin', '?')} min active · "
        f"{recipe.get('totalTimeMin', '?')} min total · "
        f"serves {recipe.get('serves') or '?'}",
        "",
        f"{len(recipe['ingredients'])} ingredients · {len(recipe['steps'])} steps · "
        f"confidence: {recipe.get('confidence', 'unknown')}",
    ]

    if recipe.get("sourceNotes"):
        lines += ["", f"> {recipe['sourceNotes']}"]

    if fetched.used_transcript:
        lines += ["", "Built from the caption and the video's voiceover together. "
                      "Spoken amounts are easy to mishear -- check them."]

    if recipe.get("extractionErrors"):
        lines += ["", "**This one didn't pass validation.** It's in the queue so you "
                      "can look at it, but check these before you approve:"]
        lines += [f"- {e}" for e in recipe["extractionErrors"]]

    if recipe.get("extractionWarnings"):
        lines += ["", "**The source was incomplete:**"]
        lines += [f"- {w}" for w in recipe["extractionWarnings"]]

    if result.warnings:
        lines += ["", "**Worth a look before cooking:**"]
        lines += [f"- {w}" for w in result.warnings]

    lines += ["", f"Saved to `recipes/pending/{recipe['id']}.json` — waiting for review."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
