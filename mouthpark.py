#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "Pillow>=10.0.0",
# ]
# # The phoneme recognizer (allosaurus + torch) is installed by
# # `mouthpark.py --install-deps` or the app's Install button, which skips
# # allosaurus's compiler-only extras (editdistance, resampy/numba).
# ///
"""MouthPark — South Park-style lip-synced mouth animation from an audio file.

One self-contained file. Everything the old ``mouthpark/`` package did lives
here, plus the default mouth artwork (embedded at the bottom), so this script
runs from anywhere with no asset folder.

Pipeline::

    audio ─► ffmpeg (16 kHz mono PCM) ─► allosaurus phonemes (IPA)
          ─► phoneme→mouth mapping ─► fixed-FPS quantization + min-hold
          ─► frames piped straight into ffmpeg ─► WebM/VP9α | MOV/ProRes 4444 | MP4

Quick start::

    uv run mouthpark.py voice.mp3              # PEP 723: uv installs deps for you
    python mouthpark.py voice.mp3 --preview    # opaque background + audio
    python mouthpark.py                        # no arguments: opens the app in your browser
    python mouthpark.py --doctor               # check your environment
    python mouthpark.py --self-test            # run the built-in tests

Exit codes: 0 ok · 1 bad input/usage · 2 mouth asset problem ·
3 phoneme recognition failure · 4 ffmpeg/encoding failure · 130 interrupted.

MIT licensed.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Final

__version__: Final = "0.6.1"

log = logging.getLogger("mouthpark")

# ════════════════════════════════════════════════════════════════════════════
# Errors & exit codes
# ════════════════════════════════════════════════════════════════════════════

EXIT_OK: Final = 0
EXIT_BAD_INPUT: Final = 1
EXIT_ASSET: Final = 2
EXIT_RECOGNITION: Final = 3
EXIT_ENCODE: Final = 4
EXIT_INTERRUPTED: Final = 130


class MouthParkError(Exception):
    """A user-facing failure with an exit code. Never shows a traceback."""

    def __init__(self, message: str, code: int = EXIT_BAD_INPUT) -> None:
        super().__init__(message)
        self.code = code


# ════════════════════════════════════════════════════════════════════════════
# Limits (defensive caps — everything user-controlled is bounded)
# ════════════════════════════════════════════════════════════════════════════

FPS_RANGE: Final = (1, 60)
MIN_HOLD_RANGE: Final = (1, 30)
MAX_CANVAS_SIDE: Final = 8192          # px, any rendered canvas
MAX_MOUTH_SIDE: Final = 4096           # px, a single mouth PNG
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
MAX_EVENTS: Final = 2_000_000
MAX_PHONEME_LEN: Final = 16
DEFAULT_MAX_DURATION: Final = 1800.0   # seconds (30 min); 0 disables
FFMPEG_PROBE_TIMEOUT: Final = 30       # seconds
FFMPEG_DECODE_TIMEOUT: Final = 900     # seconds

# Pillow's decompression-bomb guard, tightened to what we could ever render.
_MAX_IMAGE_PIXELS: Final = MAX_CANVAS_SIDE * MAX_CANVAS_SIDE


# ════════════════════════════════════════════════════════════════════════════
# Phoneme → mouth mapping
# ════════════════════════════════════════════════════════════════════════════

REST: Final = None  # "no phoneme" — rendered according to --rest

MOUTH_NAMES: Final = (
    "closed", "clenched", "ah", "ee", "oh", "woo", "bite", "tongue", "uh", "rr",
)

DEFAULT_IPA_TO_MOUTH: Final[Mapping[str, str | None]] = MappingProxyType({
    # closed: m, b, p (+ glottal stop as a brief closure)
    "m": "closed", "b": "closed", "p": "closed", "ʔ": "closed",
    # clenched: d, t, s, z, k, g, n, y, sh, ch, j, zh
    "d": "clenched", "t": "clenched", "s": "clenched", "z": "clenched",
    "k": "clenched", "ɡ": "clenched", "g": "clenched", "n": "clenched",
    "ŋ": "clenched", "j": "clenched",  # IPA j = English "y"
    "ʃ": "clenched", "tʃ": "clenched", "dʒ": "clenched", "ʒ": "clenched",
    # ah: A-vowels, I-diphthong, h, British "lot"
    "a": "ah", "ɑ": "ah", "æ": "ah", "aɪ": "ah", "aʊ": "ah", "h": "ah", "ɒ": "ah",
    # ee: E-vowels, long A, high-front I
    "i": "ee", "iː": "ee", "ɪ": "ee", "ɛ": "ee", "eɪ": "ee",
    # bare "e" — allosaurus emits this for the letter-name "A"
    "e": "tongue",
    # oh: O-vowels, aw
    "o": "oh", "oʊ": "oh", "ɔ": "oh", "ɔː": "oh", "ɔɪ": "oh",
    # woo: U-vowels, w
    "u": "woo", "uː": "woo", "ʊ": "woo", "w": "woo",
    # bite: f, v
    "f": "bite", "v": "bite",
    # tongue: l, th
    "l": "tongue", "θ": "tongue", "ð": "tongue",
    # uh: the stressed "cup" vowel only
    "ʌ": "uh", "ɜ": "uh",
    # rr: r family
    "r": "rr", "ɹ": "rr", "ɻ": "rr", "ɝ": "rr",
    # reduction vowels → rest (mapping them to "uh" made it appear everywhere)
    "ə": REST, "ɚ": REST,
})

DEFAULT_FALLBACK: Final = "clenched"

# IPA phonemes that may hold across silence (elongation), regardless of the
# mouth they render as — so the letter-name "A" (e → tongue) holds like a vowel.
VOWEL_PHONEMES: Final = frozenset({
    "a", "ɑ", "æ", "aɪ", "aʊ", "ɒ",
    "e", "ɛ", "eɪ", "i", "iː", "ɪ",
    "o", "oʊ", "ɔ", "ɔː", "ɔɪ",
    "u", "uː", "ʊ",
    "ʌ", "ɜ", "ɝ",
})

_STRIP_MARKS = str.maketrans("", "", "ːˈˌ")


def clean_ipa(ipa: str) -> str:
    """Remove length and stress marks."""
    return ipa.translate(_STRIP_MARKS).strip()


def is_vowel(ipa: str) -> bool:
    if not ipa:
        return False
    cleaned = clean_ipa(ipa)
    return cleaned in VOWEL_PHONEMES or (len(cleaned) > 1 and cleaned[0] in VOWEL_PHONEMES)


@dataclass(frozen=True, slots=True)
class Mapper:
    """IPA → mouth lookup. Immutable; build a custom one with :meth:`with_overrides`."""

    table: Mapping[str, str | None] = DEFAULT_IPA_TO_MOUTH
    fallback: str = DEFAULT_FALLBACK

    def __call__(self, ipa: str) -> str | None:
        if not ipa:
            return self.fallback
        cleaned = clean_ipa(ipa)
        if cleaned in self.table:
            return self.table[cleaned]
        # Multi-char symbol we don't know: try its first character, but never
        # let a "rest" entry swallow an unknown cluster.
        if len(cleaned) > 1 and self.table.get(cleaned[0]) is not None:
            return self.table[cleaned[0]]
        return self.fallback

    def with_overrides(self, overrides: Mapping[str, Any]) -> Mapper:
        table = dict(self.table)
        fallback = self.fallback
        for raw_key, value in overrides.items():
            if not isinstance(raw_key, str) or not raw_key or len(raw_key) > MAX_PHONEME_LEN:
                raise MouthParkError(f"mapping: bad phoneme key {raw_key!r}")
            if raw_key == "_fallback":
                if value not in MOUTH_NAMES:
                    raise MouthParkError(f"mapping: _fallback must be one of {', '.join(MOUTH_NAMES)}")
                fallback = value
                continue
            if raw_key.startswith("_"):
                continue  # reserved for comments / metadata
            if value is not None and value not in MOUTH_NAMES:
                raise MouthParkError(
                    f"mapping: {raw_key!r} → {value!r} is not a mouth "
                    f"(use one of {', '.join(MOUTH_NAMES)} or null for rest)"
                )
            table[clean_ipa(raw_key)] = value
        return Mapper(MappingProxyType(table), fallback)

    def to_json_obj(self) -> dict[str, Any]:
        return {"_fallback": self.fallback, **dict(self.table)}


def load_mapping(path: Path) -> Mapper:
    data = read_json_file(path, what="mapping file")
    if not isinstance(data, dict):
        raise MouthParkError(f"{path}: mapping must be a JSON object of phoneme → mouth")
    if len(data) > 10_000:
        raise MouthParkError(f"{path}: too many mapping entries")
    return Mapper().with_overrides(data)


# ════════════════════════════════════════════════════════════════════════════
# Quantization (timed phonemes → one mouth per frame)
# ════════════════════════════════════════════════════════════════════════════

# Allosaurus emits each phoneme as a ~45 ms pulse. Each pulse is extended to
# the next pulse's start, capped by these (seconds). Beyond the cap → REST.
VOWEL_HOLD: Final = 2.0
SILENCE_GAP: Final = 0.25


@dataclass(frozen=True, slots=True)
class PhonemeEvent:
    phoneme: str
    start: float  # seconds
    end: float    # seconds


def extend_events(events: Sequence[PhonemeEvent]) -> list[PhonemeEvent]:
    """Hold each phoneme until the next one; vowels may hold longer than consonants."""
    out: list[PhonemeEvent] = []
    for i, ev in enumerate(events):
        next_start = events[i + 1].start if i + 1 < len(events) else math.inf
        cap = VOWEL_HOLD if is_vowel(ev.phoneme) else SILENCE_GAP
        new_end = max(ev.end, min(next_start, ev.end + cap))
        out.append(PhonemeEvent(ev.phoneme, ev.start, new_end))
    return out


def quantize(
    events: Iterable[PhonemeEvent],
    duration: float,
    fps: int = 18,
    min_hold: int = 2,
    mapper: Callable[[str], str | None] | None = None,
) -> list[str | None]:
    """Return one mouth name (or REST) per frame.

    Each frame takes the mouth with the most time in its window; uncovered time
    counts toward REST. Then runs shorter than ``min_hold`` merge into a neighbor.
    """
    mapper = mapper or Mapper()
    evs = extend_events(sorted(events, key=lambda e: e.start))
    mouths = [mapper(e.phoneme) for e in evs]
    n_frames = max(1, int(round(duration * fps)))
    frame_len = 1.0 / fps

    # prefix_max_end[k] = max(end of evs[0..k]); lets us skip everything that
    # can no longer overlap without changing iteration order (O(n) instead of O(n·m)).
    prefix_max_end: list[float] = []
    running = -math.inf
    for ev in evs:
        running = max(running, ev.end)
        prefix_max_end.append(running)

    frames: list[str | None] = []
    lo = 0
    for i in range(n_frames):
        f_start, f_end = i * frame_len, (i + 1) * frame_len
        while lo < len(evs) and prefix_max_end[lo] <= f_start:
            lo += 1
        totals: dict[str | None, float] = defaultdict(float)
        covered = 0.0
        for j in range(lo, len(evs)):
            ev = evs[j]
            if ev.end <= f_start:
                continue
            if ev.start >= f_end:
                break
            overlap = min(ev.end, f_end) - max(ev.start, f_start)
            if overlap <= 0:
                continue
            totals[mouths[j]] += overlap
            covered += overlap
        totals[REST] += max(0.0, (f_end - f_start) - covered)
        frames.append(max(totals.items(), key=lambda kv: kv[1])[0])

    return enforce_min_hold(frames, min_hold)


def enforce_min_hold(frames: Sequence[str | None], min_hold: int) -> list[str | None]:
    """Merge runs shorter than ``min_hold`` into the longer neighbor (prev wins ties).

    Single forward pass; same result as the old repeat-until-stable loop.
    """
    if min_hold <= 1 or not frames:
        return list(frames)

    runs: list[list[Any]] = []  # [mouth, length]
    for f in frames:
        if runs and runs[-1][0] == f:
            runs[-1][1] += 1
        else:
            runs.append([f, 1])

    i = 0
    while i < len(runs):
        if runs[i][1] >= min_hold or len(runs) == 1:
            i += 1
            continue
        prev_run = runs[i - 1] if i > 0 else None
        next_run = runs[i + 1] if i + 1 < len(runs) else None
        if prev_run is not None and (next_run is None or prev_run[1] >= next_run[1]):
            prev_run[1] += runs.pop(i)[1]
            if i < len(runs) and runs[i][0] == prev_run[0]:
                prev_run[1] += runs.pop(i)[1]
        else:
            assert next_run is not None
            next_run[1] += runs.pop(i)[1]
            if i > 0 and runs[i - 1][0] == next_run[0]:
                runs[i - 1][1] += runs.pop(i)[1]
                i -= 1
        # re-examine position i (it may still be short)

    out: list[str | None] = []
    for mouth, length in runs:
        out.extend([mouth] * length)
    return out


# ════════════════════════════════════════════════════════════════════════════
# JSON helpers (timeline in/out, mapping)
# ════════════════════════════════════════════════════════════════════════════


def read_json_file(path: Path, *, what: str) -> Any:
    if not path.is_file():
        raise MouthParkError(f"{what} not found: {path}")
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise MouthParkError(f"{what} is too large ({size} bytes): {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MouthParkError(f"{what} is not valid UTF-8 JSON: {path}: {e}") from None


def load_events(path: Path) -> list[PhonemeEvent]:
    """Read events from a timeline JSON written by --timeline-out (or a bare list)."""
    data = read_json_file(path, what="events file")
    items = data.get("events") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise MouthParkError(f"{path}: expected an 'events' list")
    if len(items) > MAX_EVENTS:
        raise MouthParkError(f"{path}: too many events ({len(items)})")
    events: list[PhonemeEvent] = []
    for n, item in enumerate(items):
        try:
            ph, start, end = item["phoneme"], float(item["start"]), float(item["end"])
        except (TypeError, KeyError, ValueError):
            raise MouthParkError(f"{path}: event #{n} needs phoneme/start/end") from None
        if not isinstance(ph, str) or len(ph) > MAX_PHONEME_LEN:
            raise MouthParkError(f"{path}: event #{n} has a bad phoneme")
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            raise MouthParkError(f"{path}: event #{n} has bad timing ({start}, {end})")
        events.append(PhonemeEvent(ph, start, end))
    return events


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ════════════════════════════════════════════════════════════════════════════
# ffmpeg helpers
# ════════════════════════════════════════════════════════════════════════════


# ── Which ffmpeg? ────────────────────────────────────────────────────────────
# Order: --ffmpeg (this run) → $MOUTHPARK_FFMPEG → saved choice (--set-ffmpeg or
# the app's Settings) → ffmpeg on PATH. A chosen file must be named ffmpeg[.exe]
# and must answer `-version` like ffmpeg before it is ever used.

_FFMPEG_OVERRIDE: str | None = None   # set by --ffmpeg
_FFMPEG_NAMES: Final = ("ffmpeg.exe", "ffmpeg") if sys.platform == "win32" else ("ffmpeg",)
MAX_CONFIG_BYTES: Final = 64 * 1024


def config_path() -> Path:
    if env := os.environ.get("MOUTHPARK_CONFIG_DIR"):
        return Path(env).expanduser() / "config.json"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mouthpark" / "config.json"


def load_config() -> dict[str, Any]:
    p = config_path()
    try:
        if not p.is_file() or p.stat().st_size > MAX_CONFIG_BYTES:
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def save_config(**changes: Any) -> None:
    cfg = load_config()
    for k, v in changes.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    write_json_atomic(config_path(), cfg)


def validate_ffmpeg(spec: str | Path) -> tuple[str, str]:
    """Check a user-chosen ffmpeg. Accepts the exe itself or a folder containing it
    (or its bin/). Returns (absolute path, version line)."""
    raw = str(spec).strip().strip('"').strip("'")
    if not raw:
        raise MouthParkError("no ffmpeg path given")
    p = Path(raw).expanduser()
    if p.is_dir():
        found = next((c for d in (p, p / "bin") for n in _FFMPEG_NAMES if (c := d / n).is_file()), None)
        if found is None:
            raise MouthParkError(f"no ffmpeg in {p} (looked for {' / '.join(_FFMPEG_NAMES)}, also in bin\\)"
                                 if sys.platform == "win32" else f"no ffmpeg in {p} or {p / 'bin'}")
        p = found
    if not p.is_file():
        raise MouthParkError(f"not a file: {p}")
    if p.name.lower() not in _FFMPEG_NAMES:
        raise MouthParkError(f"that's {p.name!r}, not ffmpeg — pick the file named {' or '.join(_FFMPEG_NAMES)}")
    p = p.resolve()
    if sys.platform != "win32" and not os.access(p, os.X_OK):
        raise MouthParkError(f"{p} isn't executable")
    try:
        out = subprocess.run([str(p), "-hide_banner", "-version"], capture_output=True, text=True,
                             timeout=FFMPEG_PROBE_TIMEOUT, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        raise MouthParkError(f"couldn't run {p}: {e}") from None
    first = (out.stdout or "").strip().splitlines()[:1]
    if out.returncode != 0 or not first or not first[0].startswith("ffmpeg version"):
        raise MouthParkError(f"{p} didn't respond like ffmpeg")
    return str(p), first[0].split(" Copyright")[0][:120]


def find_ffmpeg() -> tuple[str | None, str]:
    """(path or None, where it came from). Never raises."""
    choices = [("--ffmpeg", _FFMPEG_OVERRIDE), ("MOUTHPARK_FFMPEG", os.environ.get("MOUTHPARK_FFMPEG")),
               ("saved setting", load_config().get("ffmpeg"))]
    for source, value in choices:
        if isinstance(value, str) and value.strip():
            p = Path(value.strip().strip('"')).expanduser()
            if p.is_dir():
                p = next((d / n for d in (p, p / "bin") for n in _FFMPEG_NAMES if (d / n).is_file()), p)
            if p.is_file() and p.name.lower() in _FFMPEG_NAMES:
                return str(p.resolve()), source
            log.warning("ffmpeg from %s not found (%s); falling back", source, value)
    exe = shutil.which("ffmpeg")
    return exe, ("PATH" if exe else "not found")


def ffmpeg_path() -> str:
    exe, _ = find_ffmpeg()
    if not exe:
        raise MouthParkError(
            "ffmpeg not found. Install it (Windows: winget install ffmpeg · macOS: brew install ffmpeg · "
            "Linux: sudo apt install ffmpeg), or point MouthPark at it: in the app, Settings → ffmpeg → "
            "Browse…; on the command line, --set-ffmpeg PATH.",
            EXIT_ENCODE,
        )
    return exe


_PICKER_CODE: Final = r"""
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    sys.exit(3)
root = tk.Tk(); root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
kw = {"title": "Select ffmpeg"}
if sys.platform == "win32":
    kw["filetypes"] = [("ffmpeg", "ffmpeg.exe"), ("Programs", "*.exe")]
path = filedialog.askopenfilename(**kw)
root.destroy()
sys.stdout.write(path or "")
"""
_PICKER_LOCK = threading.Lock()


def pick_ffmpeg_dialog() -> str | None:
    """Show the OS file picker (in a separate Python process, so it works from any thread
    and on macOS). Returns the chosen path, or None if cancelled."""
    if not _PICKER_LOCK.acquire(blocking=False):
        raise MouthParkError("a file picker is already open — check your taskbar")
    try:
        try:
            proc = subprocess.run([sys.executable, "-c", _PICKER_CODE], capture_output=True, text=True,
                                  timeout=600, stdin=subprocess.DEVNULL, check=False)
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is None or proc.returncode != 0:
            raise MouthParkError("the file picker isn't available here — paste the path to ffmpeg instead")
        return proc.stdout.strip() or None
    finally:
        _PICKER_LOCK.release()


def ff_file(path: Path) -> str:
    """Force ffmpeg's plain-file protocol.

    Without the ``file:`` prefix a name like ``http://…``, ``concat:…`` or
    ``-evil.mp3`` would be treated as a URL, protocol or option.
    """
    return "file:" + str(path.resolve())


def available_encoders(ffmpeg: str) -> set[str]:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-encoders"],
            capture_output=True, text=True, timeout=FFMPEG_PROBE_TIMEOUT, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def decode_to_wav(src: Path, dst: Path, ffmpeg: str) -> None:
    """Any input ffmpeg understands → 16 kHz mono 16-bit PCM WAV (what allosaurus wants)."""
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", ff_file(src), "-vn", "-map", "0:a:0",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", ff_file(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=FFMPEG_DECODE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        raise MouthParkError(f"ffmpeg timed out decoding {src}", EXIT_BAD_INPUT) from None
    if proc.returncode != 0 or not dst.is_file():
        raise MouthParkError(
            f"could not decode audio from {src} (is it an audio file?)\n{_tail(proc.stderr)}",
            EXIT_BAD_INPUT,
        )


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / float(rate) if rate else 0.0
    except (wave.Error, EOFError, OSError) as e:
        raise MouthParkError(f"could not read decoded audio: {e}", EXIT_BAD_INPUT) from None


# ════════════════════════════════════════════════════════════════════════════
# Phoneme recognition (allosaurus) with a verified model download
# ════════════════════════════════════════════════════════════════════════════

MODEL_NAME: Final = "uni2005"  # what allosaurus calls "latest"
MODEL_URL: Final = "https://github.com/xinjli/allosaurus/releases/download/v1.0/latest.tar.gz"
MODEL_TARBALL_SHA256: Final = "7ccf374aba1dde19b527a988709c59ed9697ab1c1df231f2b2428dd81784a316"
MODEL_PT_SHA256: Final = "552751ac877abcba22481a9d022c82b22f97b0254b72bd4e312cd744ed291cbe"
MODEL_MAX_BYTES: Final = 200 * 1024 * 1024


def cache_dir() -> Path:
    if env := os.environ.get("MOUTHPARK_CACHE"):
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "mouthpark"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_ok(model_dir: Path) -> bool:
    pt = model_dir / "model.pt"
    return pt.is_file() and sha256_file(pt) == MODEL_PT_SHA256


def _download_model(models_root: Path) -> Path:
    """Download the allosaurus model, verify its SHA-256, extract safely."""
    import tarfile
    import urllib.request

    models_root.mkdir(parents=True, exist_ok=True)
    log.warning("Downloading phoneme model (~40 MB, one time) from %s", MODEL_URL)
    with tempfile.TemporaryDirectory(dir=models_root, prefix=".dl-") as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        h = hashlib.sha256()
        total = 0
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": f"mouthpark/{__version__}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, tar_path.open("wb") as fh:
                if not resp.geturl().startswith("https://"):
                    raise MouthParkError("model download was redirected off HTTPS", EXIT_RECOGNITION)
                while chunk := resp.read(1 << 20):
                    total += len(chunk)
                    if total > MODEL_MAX_BYTES:
                        raise MouthParkError("model download is unexpectedly large", EXIT_RECOGNITION)
                    h.update(chunk)
                    fh.write(chunk)
        except OSError as e:
            raise MouthParkError(f"could not download the phoneme model: {e}", EXIT_RECOGNITION) from None
        if h.hexdigest() != MODEL_TARBALL_SHA256:
            raise MouthParkError(
                "phoneme model checksum mismatch — refusing to use it "
                f"(got {h.hexdigest()}, expected {MODEL_TARBALL_SHA256})",
                EXIT_RECOGNITION,
            )
        extract_to = Path(tmp) / "x"
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_to, filter="data")  # blocks ../, absolute paths, links out
        staged = extract_to / MODEL_NAME
        if not _model_ok(staged):
            raise MouthParkError("downloaded model failed verification", EXIT_RECOGNITION)
        final = models_root / MODEL_NAME
        if final.exists():
            shutil.rmtree(final)
        os.replace(staged, final)
    return final


def find_model() -> Path | None:
    """A verified local model, if there is one: MouthPark cache → allosaurus's own dir."""
    candidates = [cache_dir() / "models" / MODEL_NAME]
    with contextlib.suppress(Exception):
        import importlib.util

        spec = importlib.util.find_spec("allosaurus")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).parent / "pretrained" / MODEL_NAME)
    for c in candidates:
        if c.is_dir():
            if _model_ok(c):
                return c
            log.warning("Ignoring model at %s (checksum mismatch)", c)
    return None


def ensure_model() -> Path:
    """Locate a verified model, downloading (and verifying) it if needed."""
    found = find_model()
    if found is not None:
        log.info("Using phoneme model at %s", found)
        return found
    return _download_model(cache_dir() / "models")


# ── Dependency stand-ins & installer ────────────────────────────────────────
# allosaurus declares `editdistance` (only used for training) and `resampy`
# (→ numba; only used when audio isn't 16 kHz, and MouthPark always feeds it
# 16 kHz). Both are compiled packages that often have no wheel for brand-new
# Pythons, which makes `pip install allosaurus` demand a C++ compiler. So:
# if either is missing or broken, MouthPark supplies a tiny stand-in, and
# --install-deps installs allosaurus without them.

def _standin_editdistance() -> Any:
    import types

    mod = types.ModuleType("editdistance")
    mod.__doc__ = "MouthPark stand-in for editdistance (pure Python Levenshtein)."

    def eval(a: Sequence[Any], b: Sequence[Any]) -> int:  # noqa: A001 — mirrors editdistance.eval
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]

    mod.eval = eval
    mod.distance = eval
    return mod


def _standin_resampy() -> Any:
    import types

    mod = types.ModuleType("resampy")
    mod.__doc__ = "MouthPark stand-in for resampy (scipy.signal.resample_poly)."

    def resample(x: Any, sr_orig: float, sr_new: float, axis: int = -1, **_: Any) -> Any:
        from scipy.signal import resample_poly

        a, b = int(round(sr_orig)), int(round(sr_new))
        if a == b:
            return x
        g = math.gcd(a, b)
        return resample_poly(x, b // g, a // g, axis=axis)

    mod.resample = resample
    return mod


_STANDINS: Final = {"editdistance": _standin_editdistance, "resampy": _standin_resampy}


def install_standins() -> list[str]:
    """Provide stand-ins for missing/broken optional deps. Returns the names replaced."""
    import importlib

    replaced = []
    for name, make in _STANDINS.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:  # missing, or installed but broken (e.g. numba for a newer Python)
            sys.modules[name] = make()
            replaced.append(name)
    if replaced:
        log.info("Using MouthPark's built-in stand-in for: %s", ", ".join(replaced))
    return replaced


# (import name, pip name, compiled?) — what recognition really needs.
_RECOGNIZER_DEPS: Final = (
    ("numpy", "numpy", True), ("scipy", "scipy", True), ("torch", "torch", True),
    ("pandas", "pandas", True), ("regex", "regex", True), ("yaml", "PyYAML", True),
    ("unicodecsv", "unicodecsv", False), ("munkres", "munkres", False),
)
# Installed without their declared dependencies (that's the whole point).
_RECOGNIZER_NODEPS: Final = (("allosaurus", "allosaurus==1.0.2"), ("panphon", "panphon>=0.20"))


def _have(module: str) -> bool:
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return False
    # A bare leftover folder (e.g. allosaurus's model dir after uninstalling) shows up as a
    # namespace package with no origin — that isn't an installed package.
    return spec is not None and spec.origin not in (None, "namespace")


def recognizer_missing() -> list[str]:
    """pip names of what's still needed for phoneme recognition."""
    return [pip for mod, pip, _ in _RECOGNIZER_DEPS if not _have(mod)] + \
           [mod for mod, _ in _RECOGNIZER_NODEPS if not _have(mod)]


def _pip_base() -> list[str]:
    if _have("pip"):
        return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    uv = shutil.which("uv")
    if uv:  # uv-created environments have no pip
        return [uv, "pip", "install", "--python", sys.executable]
    raise MouthParkError("this Python has no pip. Run: python -m ensurepip --upgrade", EXIT_RECOGNITION)


def install_recognizer(echo: Callable[[str], None] = print) -> list[str]:
    """Install what phoneme recognition needs, without ever touching packages you already
    have (no upgrades, no downgrades) and without anything that needs a C++ compiler."""
    import importlib

    base = _pip_base()
    wanted = [pip for mod, pip, _ in _RECOGNIZER_DEPS if not _have(mod)]
    compiled = [pip for mod, pip, c in _RECOGNIZER_DEPS if c and not _have(mod)]
    nodeps = [spec for mod, spec in _RECOGNIZER_NODEPS if not _have(mod)]
    steps: list[list[str]] = []
    if wanted:
        # Prebuilt wheels only for the compiled ones: never fall back to compiling.
        steps.append([*base, *(["--only-binary", ",".join(compiled)] if compiled else []), *wanted])
    if nodeps:
        steps.append([*base, "--no-deps", *nodeps])
    if not steps:
        echo("Everything the phoneme recognizer needs is already installed.")
        return []
    for cmd in steps:
        echo("$ " + " ".join(Path(cmd[0]).name if i == 0 else a for i, a in enumerate(cmd)))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    stdin=subprocess.DEVNULL, errors="replace")
        except OSError as e:
            raise MouthParkError(f"couldn't run pip: {e}", EXIT_RECOGNITION) from None
        assert proc.stdout is not None
        seen = []
        for line in proc.stdout:
            if line.strip():
                echo(line.rstrip())
                seen.append(line)
        if proc.wait() != 0:
            hint = ""
            if any("externally-managed-environment" in ln for ln in seen):
                hint = (" This Python is managed by your operating system. Make a virtual environment "
                        "(python -m venv .venv), then run MouthPark with that Python.")
            elif "torch" in " ".join(cmd):
                hint = (" PyTorch has no prebuilt package for this Python yet — check "
                        "https://pytorch.org/get-started/locally/ for one that matches.")
            raise MouthParkError(f"pip failed (exit {proc.returncode}).{hint}", EXIT_RECOGNITION)
    importlib.invalidate_caches()
    still = recognizer_missing()
    if still:
        raise MouthParkError(f"still missing after install: {', '.join(still)}", EXIT_RECOGNITION)
    echo("Phoneme recognizer installed.")
    return wanted + nodeps


_RECOGNIZER_CACHE: dict[str, Any] = {}
_RECOGNIZER_LOCK = threading.Lock()


def recognize(wav_path: Path) -> list[PhonemeEvent]:
    """Run allosaurus on a 16 kHz mono WAV; return timed IPA events."""
    install_standins()
    try:
        from argparse import Namespace

        import torch
        from allosaurus.am.factory import read_am
        from allosaurus.app import Recognizer
        from allosaurus.lm.factory import read_lm
        from allosaurus.pm.factory import read_pm
    except ImportError as e:
        raise MouthParkError(
            f"the phoneme recognizer isn't installed for this Python (missing {e.name}). "
            "Click 'Install' next to the recognizer in the app, or run: python mouthpark.py --install-deps",
            EXIT_RECOGNITION,
        ) from None

    try:
        with _RECOGNIZER_LOCK, contextlib.redirect_stdout(sys.stderr), torch.no_grad():
            model = _RECOGNIZER_CACHE.get("model")
            if model is None:  # load once per process (the GUI renders many times)
                model_dir = ensure_model()
                # Build the recognizer ourselves: allosaurus.read_recognizer() may
                # silently download an unverified model if its own folder is empty.
                config = Namespace(model=MODEL_NAME, device_id=-1, lang="ipa", approximate=False, prior=None)
                model = Recognizer(read_pm(model_dir, config), read_am(model_dir, config),
                                   read_lm(model_dir, config), config)
                _RECOGNIZER_CACHE["model"] = model
            raw = model.recognize(str(wav_path), "eng", timestamp=True)
    except MouthParkError:
        raise
    except Exception as e:  # allosaurus/torch internals: report, don't traceback
        raise MouthParkError(f"phoneme recognition failed: {type(e).__name__}: {e}",
                             EXIT_RECOGNITION) from None
    return parse_allosaurus(raw)


def parse_allosaurus(raw: str) -> list[PhonemeEvent]:
    """Parse ``start duration phone`` lines, skipping anything malformed."""
    events: list[PhonemeEvent] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start, dur = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(start) and math.isfinite(dur)) or start < 0 or dur < 0:
            continue
        events.append(PhonemeEvent(parts[2][:MAX_PHONEME_LEN], start, start + dur))
    return events


# ════════════════════════════════════════════════════════════════════════════
# Mouth assets & frame composition
# ════════════════════════════════════════════════════════════════════════════

FLESH_RGB: Final = (241, 194, 165)


def _pil():
    try:
        from PIL import Image
    except ImportError:
        raise MouthParkError("Pillow is not installed. Try: python -m pip install Pillow") from None
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    return Image


def _open_png(data: bytes | Path, label: str, max_side: int):
    Image = _pil()
    try:
        src = io.BytesIO(data) if isinstance(data, bytes) else data
        with Image.open(src) as im:
            if im.format != "PNG":
                raise MouthParkError(f"{label} is not a PNG", EXIT_ASSET)
            if max(im.size) > max_side:
                raise MouthParkError(f"{label} is {im.size}, larger than {max_side}px", EXIT_ASSET)
            return im.convert("RGBA")
    except MouthParkError:
        raise
    except Exception as e:  # PIL raises many types for corrupt files
        raise MouthParkError(f"could not read {label}: {e}", EXIT_ASSET) from None


def load_mouths(mouths_dir: Path | None) -> dict[str, Any]:
    """Load the 10 mouth PNGs (from a folder, or the embedded defaults)."""
    images: dict[str, Any] = {}
    for name in MOUTH_NAMES:
        if mouths_dir is None:
            img = _open_png(base64.b64decode(_EMBEDDED_MOUTHS[name]), f"embedded {name}.png", MAX_MOUTH_SIDE)
        else:
            p = mouths_dir / f"{name}.png"
            if not p.is_file():
                raise MouthParkError(f"missing mouth asset: {p}", EXIT_ASSET)
            img = _open_png(p, str(p), MAX_MOUTH_SIDE)
        images[name] = img
    sizes = {im.size for im in images.values()}
    if len(sizes) != 1:
        detail = ", ".join(f"{n}={im.size}" for n, im in images.items())
        raise MouthParkError(f"all mouth PNGs must share one size; got {detail}", EXIT_ASSET)
    return images


def export_mouths(dest: Path, force: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in MOUTH_NAMES:
        p = dest / f"{name}.png"
        if p.exists() and not force:
            raise MouthParkError(f"{p} exists (use --force to overwrite)")
        p.write_bytes(base64.b64decode(_EMBEDDED_MOUTHS[name]))
    print(f"Wrote {len(MOUTH_NAMES)} mouth PNGs to {dest}")


@dataclass(slots=True)
class Layout:
    canvas: tuple[int, int]
    center: tuple[float, float]           # mouth centre on the canvas
    mouth_size: tuple[int, int]           # after scaling, before rotation
    rotation: float                       # degrees, positive = counter-clockwise
    base: Any                             # RGBA canvas under every frame
    alpha: bool                           # does the output keep transparency?


@dataclass(frozen=True, slots=True)
class Cover:
    """An ellipse painted over the character's original mouth."""

    color: tuple[int, int, int]
    center: tuple[float, float]
    size: tuple[float, float]


@dataclass(slots=True)
class Pack:
    """A character pack: image + where the mouth goes (made by mapper.html)."""

    name: str
    image: Any                            # RGBA PIL image
    position: tuple[float, float] | None
    mouth_scale: float | None
    rotation: float | None
    cover: Cover | None
    rest: str | None


def parse_color(spec: str) -> tuple[int, int, int]:
    s = spec.strip().removeprefix("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise MouthParkError(f"expected a #RRGGBB colour, got {spec!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def parse_pair(spec: str, sep: str, what: str, *, allow_negative: bool = False) -> tuple[int, int]:
    m = re.fullmatch(rf"\s*(-?\d+)\s*{re.escape(sep)}\s*(-?\d+)\s*", spec)
    if not m:
        raise MouthParkError(f"{what}: expected A{sep}B, got {spec!r}")
    a, b = int(m.group(1)), int(m.group(2))
    if not allow_negative and (a <= 0 or b <= 0):
        raise MouthParkError(f"{what}: values must be positive")
    return a, b


def build_layout(
    mouth_size: tuple[int, int],
    *,
    scale: float,
    character: Path | Any | None,
    canvas: tuple[int, int] | None,
    position: tuple[float, float] | None,
    bg: tuple[int, int, int] | None,
    want_alpha: bool,
    rotation: float = 0.0,
    cover: Cover | None = None,
) -> Layout:
    Image = _pil()
    mw, mh = max(1, round(mouth_size[0] * scale)), max(1, round(mouth_size[1] * scale))

    if isinstance(character, Path):
        base = open_image(character, str(character))
    elif character is not None:
        base = character.copy()
    elif canvas is not None:
        base = Image.new("RGBA", canvas, (0, 0, 0, 0))
    else:
        base = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
    cw, ch = base.size

    # yuv420 codecs need even dimensions; pad right/bottom transparently.
    ew, eh = cw + (cw % 2), ch + (ch % 2)
    if max(ew, eh) > MAX_CANVAS_SIDE:
        raise MouthParkError(f"canvas {ew}x{eh} exceeds {MAX_CANVAS_SIDE}px")
    if (ew, eh) != (cw, ch):
        padded = Image.new("RGBA", (ew, eh), (0, 0, 0, 0))
        padded.paste(base, (0, 0))
        base = padded

    if cover is not None:
        base.alpha_composite(_ellipse_layer(base.size, cover, rotation))

    if bg is not None:
        under = Image.new("RGBA", base.size, (*bg, 255))
        under.alpha_composite(base)
        base = under

    cx, cy = position if position is not None else (cw / 2, ch / 2)
    half = max(mw, mh) / 2
    if cx + half <= 0 or cy + half <= 0 or cx - half >= ew or cy - half >= eh:
        raise MouthParkError(f"position {cx:g},{cy:g} puts the mouth entirely off the {cw}x{ch} canvas")

    opaque = base.getextrema()[3][0] == 255  # min alpha
    return Layout((ew, eh), (cx, cy), (mw, mh), rotation, base, alpha=want_alpha and not opaque)


def _ellipse_layer(size: tuple[int, int], cover: Cover, rotation: float):
    """Anti-aliased (4× supersampled), optionally rotated ellipse on a clear layer."""
    Image = _pil()
    from PIL import ImageDraw

    ss = 4
    w, h = max(1.0, cover.size[0]), max(1.0, cover.size[1])
    pw, ph = math.ceil(w) + 4, math.ceil(h) + 4
    patch = Image.new("RGBA", (pw * ss, ph * ss), (0, 0, 0, 0))
    ox, oy = (pw - w) / 2 * ss, (ph - h) / 2 * ss
    ImageDraw.Draw(patch).ellipse((ox, oy, ox + w * ss, oy + h * ss), fill=(*cover.color, 255))
    patch = patch.resize((pw, ph), Image.Resampling.LANCZOS)
    if rotation:
        patch = patch.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.paste(patch, (round(cover.center[0] - patch.width / 2), round(cover.center[1] - patch.height / 2)))
    return layer


def open_image(src: Path | bytes, label: str):
    """Open a character/background image (PNG/JPEG/WebP) safely as RGBA."""
    Image = _pil()
    if isinstance(src, Path) and not src.is_file():
        raise MouthParkError(f"character image not found: {src}")
    try:
        with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
            if im.format not in {"PNG", "JPEG", "WEBP"}:
                raise MouthParkError(f"{label}: use a PNG, JPEG or WebP image")
            if max(im.size) > MAX_CANVAS_SIDE:
                raise MouthParkError(f"{label} is {im.size}, larger than {MAX_CANVAS_SIDE}px")
            return im.convert("RGBA")
    except MouthParkError:
        raise
    except Exception as e:
        raise MouthParkError(f"could not read {label}: {e}") from None


# ── Character packs ─────────────────────────────────────────────────────────
# A pack is a folder or .zip holding pack.json plus the character image:
#   {"mouthpark_pack": 1, "name": "…", "image": "character.png",
#    "position": [x, y], "mouth_scale": 0.9, "rotation": 0,
#    "cover": {"color": "#f1c2a5", "center": [x, y], "size": [w, h]} | null,
#    "rest": "closed" | "blank"}

PACK_MAX_ENTRY_BYTES: Final = 64 * 1024 * 1024
PACK_MAX_ENTRIES: Final = 64
_PACK_IMAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,120}\.(png|jpe?g|webp)", re.IGNORECASE)


def _num(v: Any, what: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise MouthParkError(f"pack: {what} must be a number between {lo:g} and {hi:g}")
    return float(v)


def _num_pair(v: Any, what: str, lo: float, hi: float) -> tuple[float, float]:
    if not isinstance(v, list) or len(v) != 2:
        raise MouthParkError(f"pack: {what} must be [a, b]")
    return _num(v[0], what, lo, hi), _num(v[1], what, lo, hi)


def load_pack(path: Path) -> Pack:
    """Read a pack from a folder, a pack.json, or a .zip — without extracting anything."""
    import zipfile

    if path.is_dir():
        path = path / "pack.json"
    if not path.is_file():
        raise MouthParkError(f"pack not found: {path}")

    if path.suffix.lower() == ".zip":
        try:
            zf = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as e:
            raise MouthParkError(f"{path}: not a valid zip ({e})") from None
        with zf:
            infos = zf.infolist()
            if len(infos) > PACK_MAX_ENTRIES:
                raise MouthParkError(f"{path}: too many files for a pack")
            # pack.json may sit at the root or inside one top-level folder.
            by_name = {i.filename: i for i in infos if not i.is_dir()}
            meta_name = next((n for n in by_name if n == "pack.json" or
                              (n.count("/") == 1 and n.endswith("/pack.json"))), None)
            if meta_name is None:
                raise MouthParkError(f"{path}: no pack.json inside")
            prefix = meta_name[: -len("pack.json")]

            def read_member(name: str) -> bytes:
                info = by_name.get(prefix + name)
                if info is None:
                    raise MouthParkError(f"{path}: missing {name}")
                if info.flag_bits & 0x1:
                    raise MouthParkError(f"{path}: encrypted packs are not supported")
                if info.file_size > PACK_MAX_ENTRY_BYTES:
                    raise MouthParkError(f"{path}: {name} is too large")
                with zf.open(info) as fh:  # read with a hard cap, whatever the header claims
                    data = fh.read(PACK_MAX_ENTRY_BYTES + 1)
                if len(data) > PACK_MAX_ENTRY_BYTES:
                    raise MouthParkError(f"{path}: {name} is too large")
                return data

            meta = _parse_pack_json(read_member("pack.json"), str(path))
            image_bytes = read_member(meta["image"])
    else:
        root = path.parent.resolve()
        if path.stat().st_size > MAX_JSON_BYTES:
            raise MouthParkError(f"{path}: pack.json is too large")
        meta = _parse_pack_json(path.read_bytes(), str(path))
        img_path = (root / meta["image"]).resolve()
        if not img_path.is_relative_to(root) or not img_path.is_file():
            raise MouthParkError(f"{path}: image {meta['image']!r} must be a file next to pack.json")
        if img_path.stat().st_size > PACK_MAX_ENTRY_BYTES:
            raise MouthParkError(f"{img_path} is too large")
        image_bytes = img_path.read_bytes()

    image = open_image(image_bytes, f"pack image {meta['image']}")
    W, H = image.size
    reach = MAX_CANVAS_SIDE * 2
    cover = None
    if meta.get("cover"):
        c = meta["cover"]
        if not isinstance(c, dict):
            raise MouthParkError("pack: cover must be an object or null")
        cover = Cover(parse_color(str(c.get("color", ""))),
                      _num_pair(c.get("center"), "cover.center", -reach, reach),
                      _num_pair(c.get("size"), "cover.size", 1, MAX_CANVAS_SIDE))
    rest = meta.get("rest")
    if rest not in (None, "closed", "blank"):
        raise MouthParkError("pack: rest must be 'closed' or 'blank'")
    name = meta.get("name")
    return Pack(
        name=name[:80] if isinstance(name, str) and name.strip() else Path(meta["image"]).stem,
        image=image,
        position=_num_pair(meta["position"], "position", -reach, reach) if "position" in meta else (W / 2, H / 2),
        mouth_scale=_num(meta["mouth_scale"], "mouth_scale", 0.05, 20) if "mouth_scale" in meta else None,
        rotation=_num(meta["rotation"], "rotation", -180, 180) if "rotation" in meta else None,
        cover=cover,
        rest=rest,
    )


def _parse_pack_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MouthParkError(f"{label}: pack.json is not valid JSON: {e}") from None
    if not isinstance(meta, dict) or meta.get("mouthpark_pack") != 1:
        raise MouthParkError(f"{label}: not a MouthPark pack (expected \"mouthpark_pack\": 1)")
    image = meta.get("image")
    if not isinstance(image, str) or not _PACK_IMAGE_NAME.fullmatch(image):
        raise MouthParkError(f"{label}: 'image' must be a plain PNG/JPEG/WebP file name")
    return meta


def compose_frames(mouths: Mapping[str, Any], layout: Layout, rest: str) -> dict[str | None, Any]:
    """Pre-render every distinct frame once (≤ 11 images)."""
    Image = _pil()
    mw, mh = layout.mouth_size
    cx, cy = layout.center
    out: dict[str | None, Any] = {}
    for name, src in mouths.items():
        img = src if src.size == (mw, mh) else src.resize((mw, mh), Image.Resampling.LANCZOS)
        if layout.rotation:
            img = img.rotate(layout.rotation, resample=Image.Resampling.BICUBIC, expand=True)
        frame = layout.base.copy()
        layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        layer.paste(img, (round(cx - img.width / 2), round(cy - img.height / 2)))
        frame.alpha_composite(layer)
        out[name] = frame
    out[REST] = out["closed"] if rest == "closed" else layout.base.copy()
    return out


def write_png_frames(frames: Sequence[str | None], rendered: Mapping[str | None, Any],
                     out_dir: Path, alpha: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = [p for p in out_dir.glob("frame_*.png") if re.fullmatch(r"frame_\d{6}\.png", p.name)]
    if stale:
        log.warning("Removing %d old frame_*.png files from %s", len(stale), out_dir)
        for p in stale:
            p.unlink()
    encoded: dict[str | None, bytes] = {}
    for key, img in rendered.items():
        buf = io.BytesIO()
        (img if alpha else img.convert("RGB")).save(buf, "PNG")
        encoded[key] = buf.getvalue()
    for i, mouth in enumerate(frames):
        (out_dir / f"frame_{i:06d}.png").write_bytes(encoded[mouth])


# ════════════════════════════════════════════════════════════════════════════
# Encoding
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Container:
    muxer: str
    supports_alpha: bool
    video_alpha: tuple[str, ...]
    video_opaque: tuple[str, ...]
    audio: tuple[str, ...]
    encoders: tuple[str, ...]


CONTAINERS: Final[Mapping[str, Container]] = MappingProxyType({
    ".webm": Container(
        "webm", True,
        ("-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30",
         "-auto-alt-ref", "0", "-row-mt", "1"),
        ("-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "30", "-row-mt", "1"),
        ("-c:a", "libopus", "-b:a", "128k"),
        ("libvpx-vp9", "libopus"),
    ),
    ".mov": Container(
        "mov", True,
        ("-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"),
        ("-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"),
        ("-c:a", "pcm_s16le"),
        ("prores_ks", "pcm_s16le"),
    ),
    ".mp4": Container(
        "mp4", False,
        (),
        ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
         "-movflags", "+faststart"),
        ("-c:a", "aac", "-b:a", "192k"),
        ("libx264", "aac"),
    ),
})


def container_for(output: Path) -> Container:
    try:
        return CONTAINERS[output.suffix.lower()]
    except KeyError:
        raise MouthParkError(
            f"unsupported output type {output.suffix or '(none)'}; use {', '.join(CONTAINERS)}"
        ) from None


def encode_video(
    frames: Sequence[str | None],
    rendered: Mapping[str | None, Any],
    *,
    size: tuple[int, int],
    fps: int,
    alpha: bool,
    container: Container,
    output: Path,
    audio: Path | None,
    ffmpeg: str,
    progress: Callable[[str, float | None], None] | None = None,
    cancel: threading.Event | None = None,
) -> None:
    """Pipe raw RGBA frames into ffmpeg; write to a temp file, then atomically rename."""
    raw = {k: v.tobytes() for k, v in rendered.items()}
    w, h = size
    video_args = container.video_alpha if alpha else container.video_opaque

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.stem}.", suffix=output.suffix)
    os.close(fd)
    tmp = Path(tmp_name)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-framerate", str(fps),
        "-i", "pipe:0",
    ]
    if audio is not None:
        cmd += ["-i", ff_file(audio)]
    cmd += ["-map", "0:v:0", *video_args]
    if audio is not None:
        cmd += ["-map", "1:a:0", *container.audio, "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-f", container.muxer, ff_file(tmp)]
    log.debug("ffmpeg: %s", " ".join(cmd))

    ok = False
    try:
        with tempfile.TemporaryFile("w+b") as errlog:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errlog)
            assert proc.stdin is not None
            try:
                total = len(frames)
                for i, mouth in enumerate(frames):
                    if cancel is not None and cancel.is_set():
                        raise MouthParkError("cancelled", EXIT_INTERRUPTED)
                    proc.stdin.write(raw[mouth])
                    if progress is not None and (i % 12 == 0 or i == total - 1):
                        progress("Encoding video", (i + 1) / total)
                proc.stdin.close()
            except BrokenPipeError:
                pass  # ffmpeg died; its stderr explains why
            except BaseException:
                proc.kill()
                raise
            finally:
                timeout = 120 + len(frames) / max(fps, 1) * 10
                try:
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise MouthParkError("ffmpeg timed out while encoding", EXIT_ENCODE) from None
            errlog.seek(0)
            err = errlog.read().decode("utf-8", "replace")
        if code != 0 or tmp.stat().st_size == 0:
            raise MouthParkError(f"ffmpeg failed (exit {code}):\n{_tail(err)}", EXIT_ENCODE)
        os.replace(tmp, output)
        ok = True
    finally:
        if not ok:
            with contextlib.suppress(OSError):
                tmp.unlink()


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class Options:
    input: Path
    output: Path
    fps: int = 18
    min_hold: int = 2
    mouths_dir: Path | None = None
    mapping: Path | None = None
    rest: str | None = None               # None → pack's choice, else "closed"
    background: tuple[int, int, int] | None = None
    character: Path | None = None
    canvas: tuple[int, int] | None = None
    position: tuple[int, int] | None = None
    mouth_scale: float | None = None
    rotation: float | None = None
    pack: Path | None = None
    with_audio: bool = False
    keep_frames: Path | None = None
    events_in: Path | None = None
    timeline_out: Path | None = None
    max_duration: float = DEFAULT_MAX_DURATION
    force: bool = False
    progress: Callable[[str, float | None], None] | None = None  # GUI hook: (stage, 0–1 or None)
    cancel: threading.Event | None = None                         # GUI hook: set to stop


def _report(opts: Options, stage: str, frac: float | None = None) -> None:
    if opts.cancel is not None and opts.cancel.is_set():
        raise MouthParkError("cancelled", EXIT_INTERRUPTED)
    if opts.progress is not None:
        opts.progress(stage, frac)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return a.resolve() == b.resolve()


def validate_paths(opts: Options) -> None:
    if not opts.input.is_file():
        raise MouthParkError(f"input is not a file: {opts.input}")
    if _same_file(opts.input, opts.output) or opts.input.resolve() == opts.output.resolve():
        raise MouthParkError("output would overwrite the input audio")
    if opts.output.exists() and not opts.force:
        raise MouthParkError(f"{opts.output} already exists (use --force to overwrite)")
    if opts.output.exists() and not opts.output.is_file():
        raise MouthParkError(f"output path is not a regular file: {opts.output}")
    if opts.timeline_out is not None:
        if opts.timeline_out.resolve() in {opts.input.resolve(), opts.output.resolve()}:
            raise MouthParkError("--timeline-out must be a different file")
        if opts.timeline_out.exists() and not opts.force:
            raise MouthParkError(f"{opts.timeline_out} already exists (use --force to overwrite)")


def run(opts: Options) -> Path:
    validate_paths(opts)
    container = container_for(opts.output)

    # Fail fast on tooling before any slow work.
    ffmpeg = ffmpeg_path()
    have = available_encoders(ffmpeg)
    needed = container.encoders if opts.with_audio else container.encoders[:1]
    missing = [e for e in needed if e not in have]
    if missing:
        raise MouthParkError(f"your ffmpeg lacks encoder(s): {', '.join(missing)}", EXIT_ENCODE)

    mapper = load_mapping(opts.mapping) if opts.mapping else Mapper()
    mouths = load_mouths(opts.mouths_dir)
    log.info("Mouths: %s (%dx%d)", opts.mouths_dir or "built-in", *next(iter(mouths.values())).size)

    # Command-line flags win over the pack; the pack wins over defaults.
    pack = load_pack(opts.pack) if opts.pack else None
    if pack is not None:
        if opts.character or opts.canvas:
            raise MouthParkError("--pack already includes the character; drop --character/--canvas")
        log.info("Pack: %s (%dx%d)", pack.name, *pack.image.size)
    character = pack.image if pack else opts.character
    position = opts.position or (pack.position if pack else None)
    scale = opts.mouth_scale or (pack.mouth_scale if pack else None) or 1.0
    rotation = opts.rotation if opts.rotation is not None else ((pack.rotation if pack else None) or 0.0)
    rest = opts.rest or (pack.rest if pack else None) or "closed"

    bg = opts.background
    want_alpha = bg is None
    if want_alpha and not container.supports_alpha and character is None:
        log.warning("%s has no alpha channel — using a flesh-tone background", opts.output.suffix)
        bg, want_alpha = FLESH_RGB, False
    layout = build_layout(
        next(iter(mouths.values())).size, scale=scale, character=character,
        canvas=opts.canvas, position=position, bg=bg,
        want_alpha=want_alpha and container.supports_alpha,
        rotation=rotation, cover=pack.cover if pack else None,
    )

    with tempfile.TemporaryDirectory(prefix="mouthpark_") as work:
        wav = Path(work) / "input.wav"
        log.info("Decoding %s", opts.input)
        _report(opts, "Decoding audio")
        decode_to_wav(opts.input, wav, ffmpeg)
        duration = wav_duration(wav)
        if duration <= 0:
            raise MouthParkError("input audio is empty")
        if opts.max_duration and duration > opts.max_duration:
            raise MouthParkError(
                f"audio is {duration:.0f}s, over the --max-duration limit of {opts.max_duration:.0f}s"
            )
        if opts.events_in:
            events = load_events(opts.events_in)
            log.info("Loaded %d events from %s", len(events), opts.events_in)
        else:
            log.info("Recognizing phonemes (%.2fs of audio)…", duration)
            _report(opts, "Listening for phonemes")
            events = recognize(wav)
    log.info("%d phoneme events, %.2fs", len(events), duration)

    frames = quantize(events, duration, fps=opts.fps, min_hold=opts.min_hold, mapper=mapper)
    log.info("%d frames @ %d fps, canvas %dx%d, alpha=%s", len(frames), opts.fps, *layout.canvas, layout.alpha)

    _report(opts, "Drawing frames")
    rendered = compose_frames(mouths, layout, rest)
    if opts.keep_frames is not None:
        write_png_frames(frames, rendered, opts.keep_frames, layout.alpha)
        log.info("Wrote PNG frames to %s", opts.keep_frames)

    encode_video(
        frames, rendered, size=layout.canvas, fps=opts.fps, alpha=layout.alpha,
        container=container, output=opts.output,
        audio=opts.input if opts.with_audio else None, ffmpeg=ffmpeg,
        progress=opts.progress, cancel=opts.cancel,
    )

    if opts.timeline_out is not None:
        write_json_atomic(opts.timeline_out, {
            "mouthpark": __version__,
            "source": opts.input.name,
            "duration": round(duration, 6),
            "fps": opts.fps,
            "min_hold": opts.min_hold,
            "events": [{"phoneme": e.phoneme, "start": round(e.start, 6), "end": round(e.end, 6),
                        "mouth": mapper(e.phoneme)} for e in sorted(events, key=lambda e: e.start)],
            "frames": frames,
        })
        log.info("Wrote timeline %s", opts.timeline_out)
    return opts.output


# ════════════════════════════════════════════════════════════════════════════
# Doctor & self-test
# ════════════════════════════════════════════════════════════════════════════


def doctor() -> int:
    ok = True

    def line(good: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✔' if good else '✘'} {label}{(' — ' + detail) if detail else ''}")

    print(f"MouthPark {__version__} · Python {sys.version.split()[0]} · {sys.platform}")
    line(sys.version_info >= (3, 11), "Python ≥ 3.11")
    try:
        import PIL

        line(True, "Pillow", PIL.__version__)
    except ImportError:
        line(False, "Pillow", "pip install Pillow")
    ff, source = find_ffmpeg()
    line(bool(ff), "ffmpeg", f"{ff} (from {source})" if ff else "not found — use --set-ffmpeg PATH")
    if ff:
        enc = available_encoders(ff)
        for suffix, c in CONTAINERS.items():
            have = [e for e in c.encoders if e in enc]
            print(f"      {suffix:<5} {'ready' if len(have) == len(c.encoders) else 'missing ' + ', '.join(set(c.encoders) - set(have))}")
    missing = recognizer_missing()
    line(not missing, "phoneme recognizer", "ready" if not missing else
         f"missing {', '.join(missing)} — run: python mouthpark.py --install-deps")
    standins = [n for n in _STANDINS if not _have(n)]
    if standins and not missing:
        print(f"  · using built-in stand-ins for: {', '.join(standins)} (fine — only needed for training)")
    found = find_model()
    print(f"  · phoneme model: {found or 'downloads (and is verified) on first run'}")
    try:
        load_mouths(None)
        line(True, "built-in mouths")
    except MouthParkError as e:
        line(False, "built-in mouths", str(e))
    return EXIT_OK if ok else EXIT_BAD_INPUT


def self_test() -> int:
    import unittest

    class T(unittest.TestCase):
        def test_mapping(self) -> None:
            m = Mapper()
            self.assertEqual(m("m"), "closed")
            self.assertEqual(m("iː"), "ee")
            self.assertEqual(m("ˈɔː"), "oh")
            self.assertIsNone(m("ə"))
            self.assertEqual(m("əx"), "clenched")      # unknown cluster → fallback
            self.assertEqual(m("tʃʰ"), "clenched")     # prefix match
            self.assertEqual(m("q"), "clenched")
            self.assertEqual(m(""), "clenched")

        def test_overrides(self) -> None:
            m = Mapper().with_overrides({"θ": "bite", "ə": "uh", "_fallback": "closed", "_note": 1})
            self.assertEqual((m("θ"), m("ə"), m("q")), ("bite", "uh", "closed"))
            with self.assertRaises(MouthParkError):
                Mapper().with_overrides({"a": "nope"})

        def test_vowels(self) -> None:
            self.assertTrue(is_vowel("e"))
            self.assertTrue(is_vowel("aɪ"))
            self.assertFalse(is_vowel("s"))

        def test_min_hold_matches_reference(self) -> None:
            import random

            rng = random.Random(1234)
            for _ in range(3000):
                seq = [rng.choice(["a", "b", "c", None]) for _ in range(rng.randint(0, 40))]
                for mh in (1, 2, 3, 5):
                    self.assertEqual(enforce_min_hold(seq, mh), _reference_min_hold(seq, mh), (seq, mh))

        def test_quantize(self) -> None:
            evs = [PhonemeEvent("m", 0.0, 0.045), PhonemeEvent("a", 0.3, 0.345)]
            frames = quantize(evs, 1.0, fps=10, min_hold=1)
            self.assertEqual(len(frames), 10)
            self.assertEqual(frames[0], "closed")
            self.assertEqual(frames[3:6], ["ah", "ah", "ah"])  # vowel holds
            self.assertEqual(quantize([], 0.0, fps=12), [None])

        def test_quantize_matches_reference(self) -> None:
            import random

            rng = random.Random(99)
            phones = ["m", "a", "s", "ə", "oʊ", "l", "x", "e"]
            for _ in range(300):
                t, evs = 0.0, []
                for _ in range(rng.randint(0, 30)):
                    t += rng.uniform(0.0, 0.4)
                    evs.append(PhonemeEvent(rng.choice(phones), t, t + rng.choice([0.045, 0.3])))
                dur = t + rng.uniform(0, 1)
                for fps in (12, 18, 24):
                    self.assertEqual(quantize(evs, dur, fps=fps, min_hold=2),
                                     _reference_quantize(evs, dur, fps, 2))

        def test_parse_allosaurus(self) -> None:
            evs = parse_allosaurus("0.1 0.045 p\n0.2 0.045\nnan 1 x\n0.3 0.045 ɑ\n")
            self.assertEqual([e.phoneme for e in evs], ["p", "ɑ"])

        def test_parsers(self) -> None:
            self.assertEqual(parse_color("#F1c2A5"), (241, 194, 165))
            self.assertEqual(parse_pair("640x480", "x", "c"), (640, 480))
            for bad in ("red", "#12345", "#GGGGGG"):
                with self.assertRaises(MouthParkError):
                    parse_color(bad)

        def test_ff_file(self) -> None:
            for tricky in ("-y.mp3", "http://x", "concat:a|b"):
                s = ff_file(Path(tricky))
                self.assertTrue(s.startswith("file:") and not s.startswith("file:-"), s)

        def test_embedded_mouths(self) -> None:
            m = load_mouths(None)
            self.assertEqual(set(m), set(MOUTH_NAMES))

        def test_packs(self) -> None:
            import zipfile

            Image = _pil()
            with tempfile.TemporaryDirectory() as tmpdir:
                d = Path(tmpdir)
                Image.new("RGB", (301, 200), (10, 120, 200)).save(d / "guy.png")
                meta = {"mouthpark_pack": 1, "name": "Guy", "image": "guy.png", "position": [150, 120],
                        "mouth_scale": 0.3, "rotation": 10,
                        "cover": {"color": "#ff0000", "center": [150, 120], "size": [40, 20]}}
                (d / "pack.json").write_text(json.dumps(meta))
                pk = load_pack(d)
                self.assertEqual((pk.name, pk.image.size, pk.rotation), ("Guy", (301, 200), 10.0))
                with zipfile.ZipFile(d / "p.zip", "w") as zf:  # pack.json inside one folder is fine
                    zf.write(d / "guy.png", "Guy/guy.png")
                    zf.write(d / "pack.json", "Guy/pack.json")
                self.assertEqual(load_pack(d / "p.zip").position, (150.0, 120.0))
                lay = build_layout((200, 200), scale=pk.mouth_scale, character=pk.image, canvas=None,
                                   position=pk.position, bg=None, want_alpha=True,
                                   rotation=pk.rotation, cover=pk.cover)
                self.assertEqual(lay.canvas, (302, 200))  # padded to even
                self.assertEqual(lay.base.getpixel((150, 120))[:3], (255, 0, 0))  # cover painted
                self.assertEqual(lay.base.getpixel((5, 5))[:3], (10, 120, 200))
                rendered = compose_frames(load_mouths(None), lay, "blank")
                self.assertEqual(rendered[REST].getpixel((150, 120))[:3], (255, 0, 0))
                for evil in ("../guy.png", "/etc/passwd", "sub/guy.png", "guy.exe"):
                    (d / "pack.json").write_text(json.dumps({**meta, "image": evil}))
                    with self.assertRaises(MouthParkError, msg=evil):
                        load_pack(d)
                (d / "pack.json").write_text(json.dumps({**meta, "mouth_scale": "big"}))
                with self.assertRaises(MouthParkError):
                    load_pack(d)

        def test_ffmpeg_choice(self) -> None:
            with tempfile.TemporaryDirectory() as tmpdir:
                d = Path(tmpdir)
                with self.assertRaises(MouthParkError):       # wrong name
                    (d / "evil.exe").write_text("x")
                    validate_ffmpeg(d / "evil.exe")
                with self.assertRaises(MouthParkError):       # folder without ffmpeg
                    validate_ffmpeg(d)
                with self.assertRaises(MouthParkError):       # empty
                    validate_ffmpeg("  ")
                fake = d / "fake" / _FFMPEG_NAMES[-1]
                fake.parent.mkdir()
                if sys.platform != "win32":                  # right name, wrong program
                    fake.write_text("#!/bin/sh\necho hello\n")
                    fake.chmod(0o755)
                    with self.assertRaisesRegex(MouthParkError, "respond like ffmpeg"):
                        validate_ffmpeg(fake)
                real = shutil.which("ffmpeg")
                if real:
                    path, version = validate_ffmpeg(f'"{real}"')  # quotes from "Copy as path" are OK
                    self.assertTrue(version.startswith("ffmpeg version"))
                    save_config(ffmpeg=path)
                    self.assertEqual(find_ffmpeg(), (path, "saved setting"))
                    save_config(ffmpeg=str(d / "gone" / "ffmpeg"))  # moved/deleted → falls back
                    self.assertEqual(find_ffmpeg()[1], "PATH")
                    save_config(ffmpeg=None)
                    self.assertNotIn("ffmpeg", load_config())

        def test_standins(self) -> None:
            ed = _standin_editdistance()
            self.assertEqual((ed.eval("kitten", "sitting"), ed.eval([], [1, 2]), ed.eval("ab", "ab")), (3, 2, 0))
            rs = _standin_resampy()
            import numpy as np

            x = np.sin(np.linspace(0, 20, 48000)).astype(np.float32)
            self.assertEqual(rs.resample(x, 48000, 16000).shape, (16000,))
            self.assertIs(rs.resample(x, 16000, 16000), x)

        def test_render_roundtrip(self) -> None:
            if not find_ffmpeg()[0]:
                self.skipTest("ffmpeg not installed")
            with tempfile.TemporaryDirectory() as tmpdir:
                d = Path(tmpdir)
                wav = d / "tone.wav"
                with wave.open(str(wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\0\0" * 16000)
                ev = d / "ev.json"
                ev.write_text(json.dumps({"events": [{"phoneme": "a", "start": 0.1, "end": 0.2}]}))
                out = run(Options(input=wav, output=d / "o.webm", events_in=ev,
                                  timeline_out=d / "t.json", with_audio=True))
                self.assertGreater(out.stat().st_size, 0)
                tl = json.loads((d / "t.json").read_text())
                self.assertEqual(len(tl["frames"]), 18)
                with self.assertRaises(MouthParkError):  # no silent overwrite
                    run(Options(input=wav, output=d / "o.webm", events_in=ev))
                with self.assertRaises(MouthParkError):  # never clobber the input
                    run(Options(input=wav, output=wav, events_in=ev, force=True))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)
    old = os.environ.get("MOUTHPARK_CONFIG_DIR")
    with tempfile.TemporaryDirectory() as cfg:  # never touch the user's real settings
        os.environ["MOUTHPARK_CONFIG_DIR"] = cfg
        try:
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            if old is None:
                os.environ.pop("MOUTHPARK_CONFIG_DIR", None)
            else:
                os.environ["MOUTHPARK_CONFIG_DIR"] = old
    return EXIT_OK if result.wasSuccessful() else EXIT_BAD_INPUT


def _reference_min_hold(frames: Sequence[str | None], min_hold: int) -> list[str | None]:
    """The original v0.2.0 algorithm, kept verbatim for the self-test."""
    if min_hold <= 1 or not frames:
        return list(frames)
    runs: list[list[Any]] = []
    for f in frames:
        if runs and runs[-1][0] == f:
            runs[-1][1] += 1
        else:
            runs.append([f, 1])
    changed = True
    while changed:
        changed = False
        for i, run_ in enumerate(runs):
            if run_[1] >= min_hold:
                continue
            prev_run = runs[i - 1] if i > 0 else None
            next_run = runs[i + 1] if i + 1 < len(runs) else None
            target = None
            if prev_run and next_run:
                target = prev_run if prev_run[1] >= next_run[1] else next_run
            elif prev_run:
                target = prev_run
            elif next_run:
                target = next_run
            if target is None:
                break
            target[1] += run_[1]
            runs.pop(i)
            changed = True
            break
        if changed:
            co: list[list[Any]] = []
            for r in runs:
                if co and co[-1][0] == r[0]:
                    co[-1][1] += r[1]
                else:
                    co.append(r)
            runs = co
    out: list[str | None] = []
    for mouth, length in runs:
        out.extend([mouth] * length)
    return out


def _reference_quantize(events, duration, fps, min_hold):
    """The original v0.2.0 per-frame scan, kept for the self-test."""
    m = Mapper()
    evs = extend_events(sorted(events, key=lambda e: e.start))
    n = max(1, int(round(duration * fps)))
    fl = 1.0 / fps
    frames = []
    for i in range(n):
        fs, fe = i * fl, (i + 1) * fl
        totals: dict[str | None, float] = defaultdict(float)
        covered = 0.0
        for ev in evs:
            if ev.end <= fs:
                continue
            if ev.start >= fe:
                break
            ov = min(ev.end, fe) - max(ev.start, fs)
            if ov <= 0:
                continue
            totals[m(ev.phoneme)] += ov
            covered += ov
        totals[REST] += max(0.0, fl - covered)
        frames.append(max(totals.items(), key=lambda kv: kv[1])[0])
    return _reference_min_hold(frames, min_hold)


# ════════════════════════════════════════════════════════════════════════════
# GUI — a private local web app (python mouthpark.py --gui)
# ════════════════════════════════════════════════════════════════════════════
#
# Security model: the server listens on 127.0.0.1 only, on a random port.
# Every API call must carry a random per-launch token (sent as a custom
# header, which also forces a CORS preflight that other sites can't pass),
# the Host header must be exactly 127.0.0.1/localhost:<port> (blocks DNS
# rebinding), and the page ships a strict Content-Security-Policy with no
# third-party origins. The browser never names file paths: uploads land in a
# private temp folder under server-chosen names, and only files the server
# itself produced can be downloaded. The folder is deleted on exit.

GUI_LIMITS: Final[Mapping[str, int]] = MappingProxyType({
    "audio": 512 * 1024 * 1024,
    "pack": 128 * 1024 * 1024,
    "mouth": 16 * 1024 * 1024,
})
GUI_AUDIO_EXTS: Final = frozenset({
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm",
    ".mp4", ".mov", ".mkv", ".aif", ".aiff", ".wma", ".caf",
})
_GUI_MIME: Final = MappingProxyType({
    ".webm": "video/webm", ".mov": "video/quicktime", ".mp4": "video/mp4", ".json": "application/json",
})


def _safe_stem(name: str, default: str = "mouthpark") -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip(" .-_")[:60]
    return stem or default


@dataclass(slots=True)
class _Job:
    id: str
    audio_id: str
    download_name: str
    output: Path
    timeline: Path
    status: str = "queued"          # queued → running → done | error | cancelled
    stage: str = "Queued"
    progress: float | None = None
    error: str = ""
    log: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    frames: int = 0
    duration: float = 0.0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "stage": self.stage, "progress": self.progress,
                "error": self.error, "log": self.log[-200:], "frames": self.frames,
                "duration": self.duration, "format": self.output.suffix.lstrip("."),
                "download_name": self.download_name}


class _JobLog(logging.Handler):
    def __init__(self, job: _Job) -> None:
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            self.job.log.append(record.getMessage())


class GuiApp:
    def __init__(self, workdir: Path) -> None:
        import secrets

        self.dir = workdir
        self.token = secrets.token_urlsafe(32)
        self.uploads: dict[str, tuple[str, Path, str]] = {}   # id → (kind, path, original name)
        self.mouth_dir = workdir / "mouths"
        self.events: dict[str, Path] = {}                     # audio id → timeline with its phonemes
        self.jobs: dict[str, _Job] = {}
        self.current: _Job | None = None
        self.lock = threading.Lock()
        self.port = 0
        self.shutdown: Callable[[], None] = lambda: None
        self.install_log: list[str] = []
        self.install_status = "idle"      # idle → running → done | error
        self.install_error = ""

    # ── uploads ───────────────────────────────────────────────────────
    def save_upload(self, kind: str, name: str, stream: IO[bytes], length: int) -> dict[str, Any]:
        import secrets

        if kind not in GUI_LIMITS:
            raise MouthParkError(f"unknown upload kind {kind!r}")
        if length > GUI_LIMITS[kind]:
            raise MouthParkError(f"{kind} file is too large")
        if kind == "audio":
            ext = Path(name).suffix.lower()
            if ext not in GUI_AUDIO_EXTS:
                raise MouthParkError(f"{ext or 'that file'} isn't a supported audio type")
            dest = self.dir / f"audio-{secrets.token_hex(8)}{ext}"
        elif kind == "pack":
            dest = self.dir / f"pack-{secrets.token_hex(8)}.zip"
        else:  # a single mouth PNG; `name` must be one of the 10 shapes
            if name not in MOUTH_NAMES:
                raise MouthParkError(f"mouth files must be named one of {', '.join(MOUTH_NAMES)}")
            self.mouth_dir.mkdir(exist_ok=True)
            dest = self.mouth_dir / f"{name}.png"
        remaining = length
        with dest.open("wb") as fh:
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    raise MouthParkError("upload was cut short")
                fh.write(chunk)
                remaining -= len(chunk)
        if kind == "mouth":
            _open_png(dest, f"{name}.png", MAX_MOUTH_SIDE)  # validate now, not at render time
            return {"ok": True, "name": name}
        uid = secrets.token_hex(8)
        self.uploads[uid] = (kind, dest, name[:200])
        return {"id": uid}

    def _upload(self, uid: Any, kind: str) -> tuple[Path, str]:
        entry = self.uploads.get(uid) if isinstance(uid, str) else None
        if entry is None or entry[0] != kind:
            raise MouthParkError(f"that {kind} upload is gone — pick the file again")
        return entry[1], entry[2]

    # ── rendering ────────────────────────────────────────────────────
    def start_render(self, req: dict[str, Any]) -> dict[str, Any]:
        import secrets

        with self.lock:
            if self.current is not None and self.current.status in {"queued", "running"}:
                raise MouthParkError("a render is already running")
            opts, job = self._build(req, secrets.token_hex(6))
            self.jobs[job.id] = job
            self.current = job
        threading.Thread(target=self._work, args=(job, opts), name=f"render-{job.id}", daemon=True).start()
        return {"id": job.id}

    def _build(self, req: dict[str, Any], job_id: str) -> tuple[Options, _Job]:
        def pick(key: str, choices: Iterable[Any], default: Any) -> Any:
            v = req.get(key, default)
            if v not in choices:
                raise MouthParkError(f"bad value for {key}: {v!r}")
            return v

        def number(key: str, lo: float, hi: float, default: float | None) -> float | None:
            v = req.get(key, default)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise MouthParkError(f"{key} must be between {lo:g} and {hi:g}")
            return float(v)

        audio_path, audio_name = self._upload(req.get("audio"), "audio")
        fmt = pick("format", ("webm", "mov", "mp4"), "webm")
        pack = self._upload(req.get("pack"), "pack")[0] if req.get("pack") else None
        bg = req.get("background")
        canvas = req.get("canvas")
        position = req.get("position")
        for key, val in (("canvas", canvas), ("position", position)):
            if val is not None and not (isinstance(val, list) and len(val) == 2 and
                                        all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                            for x in val)):
                raise MouthParkError(f"{key} must be [a, b]")
        if canvas is not None and not all(1 <= c <= MAX_CANVAS_SIDE for c in canvas):
            raise MouthParkError("canvas size is out of range")

        job_dir = self.dir / f"job-{job_id}"
        job_dir.mkdir()
        mapping_path = None
        overrides = req.get("mapping")
        if overrides:
            if not isinstance(overrides, dict) or len(overrides) > 10_000:
                raise MouthParkError("mapping must be an object")
            Mapper().with_overrides(overrides)  # validate before starting
            mapping_path = job_dir / "mapping.json"
            mapping_path.write_text(json.dumps(overrides, ensure_ascii=False), encoding="utf-8")

        if req.get("custom_mouths"):
            load_mouths(self.mouth_dir)  # raises a clear error if any of the 10 is missing

        stem = _safe_stem(audio_name)
        job = _Job(job_id, req["audio"], f"{stem}.{fmt}", job_dir / f"out.{fmt}", job_dir / "timeline.json")
        opts = Options(
            input=audio_path, output=job.output, timeline_out=job.timeline, force=True,
            fps=int(number("fps", *FPS_RANGE, 18) or 18),
            min_hold=int(number("min_hold", *MIN_HOLD_RANGE, 2) or 2),
            rest=pick("rest", ("closed", "blank"), "closed"),
            background=parse_color(bg) if isinstance(bg, str) and bg else None,
            pack=pack,
            canvas=(int(canvas[0]), int(canvas[1])) if canvas and not pack else None,
            position=(float(position[0]), float(position[1])) if position and not pack else None,
            mouth_scale=number("mouth_scale", 0.05, 20, None) if not pack else None,
            rotation=number("rotation", -180, 180, None) if not pack else None,
            mouths_dir=self.mouth_dir if req.get("custom_mouths") else None,
            mapping=mapping_path,
            with_audio=bool(req.get("with_audio", False)),
            max_duration=number("max_duration", 0, 86400, DEFAULT_MAX_DURATION) or 0.0,
            events_in=self.events.get(job.audio_id) if req.get("reuse_phonemes", True) else None,
        )

        def progress(stage: str, frac: float | None) -> None:
            job.stage, job.progress = stage, frac

        opts.progress, opts.cancel = progress, job.cancel
        return opts, job

    def _work(self, job: _Job, opts: Options) -> None:
        handler = _JobLog(job)
        log.addHandler(handler)
        previous = log.level
        if previous == logging.NOTSET or previous > logging.INFO:
            log.setLevel(logging.INFO)
        job.status, job.stage = "running", "Starting"
        try:
            if opts.events_in is not None:
                job.log.append("Reusing the phonemes from the last render of this audio.")
            run(opts)
            tl = json.loads(job.timeline.read_text(encoding="utf-8"))
            job.frames, job.duration = len(tl.get("frames", [])), float(tl.get("duration", 0))
            self.events[job.audio_id] = job.timeline
            job.status, job.stage, job.progress = "done", "Done", 1.0
        except MouthParkError as e:
            cancelled = job.cancel.is_set()
            job.status = "cancelled" if cancelled else "error"
            job.error = "Cancelled." if cancelled else str(e)
            job.stage = "Cancelled" if cancelled else "Failed"
        except Exception as e:  # never kill the server on an unexpected bug
            log.debug("render crashed", exc_info=True)
            job.status, job.stage, job.error = "error", "Failed", f"{type(e).__name__}: {e}"
        finally:
            log.removeHandler(handler)
            log.setLevel(previous)

    def ffmpeg_info(self) -> dict[str, Any]:
        ff, source = find_ffmpeg()
        enc = available_encoders(ff) if ff else set()
        version = ""
        if ff:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                out = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True,
                                     timeout=FFMPEG_PROBE_TIMEOUT, check=False, stdin=subprocess.DEVNULL)
                version = (out.stdout.strip().splitlines() or [""])[0].split(" Copyright")[0][:120]
        return {"path": ff, "source": source, "version": version,
                "formats": {s.lstrip("."): bool(enc) and c.encoders[0] in enc for s, c in CONTAINERS.items()}}

    def set_ffmpeg(self, spec: str | None) -> dict[str, Any]:
        if spec is None:
            save_config(ffmpeg=None)  # back to automatic (PATH)
        else:
            path, _ = validate_ffmpeg(spec)
            save_config(ffmpeg=path)
        return self.ffmpeg_info()

    def install_state(self) -> dict[str, Any]:
        return {"status": self.install_status, "log": self.install_log[-400:], "error": self.install_error,
                "missing": recognizer_missing() if self.install_status != "running" else []}

    def start_install(self) -> dict[str, Any]:
        with self.lock:
            if self.install_status == "running":
                raise MouthParkError("already installing")
            if self.current is not None and self.current.status in {"queued", "running"}:
                raise MouthParkError("wait for the render to finish first")
            self.install_status, self.install_error, self.install_log = "running", "", []

        def work() -> None:
            try:
                install_recognizer(echo=self.install_log.append)
                self.install_status = "done"
            except MouthParkError as e:
                self.install_status, self.install_error = "error", str(e)
            except Exception as e:
                self.install_status, self.install_error = "error", f"{type(e).__name__}: {e}"

        threading.Thread(target=work, name="install-deps", daemon=True).start()
        return self.install_state()

    def info(self) -> dict[str, Any]:
        ffi = self.ffmpeg_info()
        missing = recognizer_missing()
        has_allo = not missing
        return {
            "version": __version__,
            "ffmpeg": ffi,
            "formats": ffi["formats"],
            "allosaurus": has_allo,
            "recognizer_missing": missing,
            "install": self.install_state(),
            "model_cached": find_model() is not None if has_allo else False,
            "mouth_names": list(MOUTH_NAMES),
            "mouths": dict(_EMBEDDED_MOUTHS),
            "mapping": Mapper().to_json_obj(),
            "limits": {"fps": FPS_RANGE, "min_hold": MIN_HOLD_RANGE, "canvas": MAX_CANVAS_SIDE,
                       "max_duration": DEFAULT_MAX_DURATION},
        }


def _make_gui_handler(app: GuiApp) -> type:
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlsplit

    csp = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
           "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
           "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

    class Handler(BaseHTTPRequestHandler):
        server_version = "MouthPark"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
            log.debug("gui: " + fmt, *args)

        # ── plumbing ──────────────────────────────────────────────
        def _headers(self, code: int, ctype: str, length: int, extra: Mapping[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self._headers(code, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _fail(self, code: int, msg: str) -> None:
            self.close_connection = True
            self._json({"error": msg}, code)

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            if host in {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}:
                return True
            self._fail(403, "bad host")
            return False

        def _authed(self, query: dict[str, list[str]], allow_query: bool = False) -> bool:
            import hmac

            given = self.headers.get("X-MouthPark-Token", "")
            if not given and allow_query:
                given = (query.get("t") or [""])[0]
            if given and hmac.compare_digest(given, app.token):
                return True
            self._fail(403, "missing or wrong token — reopen the link MouthPark printed")
            return False

        def _body_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 4 * 1024 * 1024:
                raise MouthParkError("request body missing or too large")
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise MouthParkError("request body isn't JSON") from None

        # ── routes ────────────────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if url.path == "/":
                body = _GUI_HTML.encode("utf-8")
                self._headers(200, "text/html; charset=utf-8", len(body),
                              {"Content-Security-Policy": csp, "X-Frame-Options": "DENY"})
                self.wfile.write(body)
                return
            if url.path == "/favicon.ico":
                self._headers(204, "text/plain", 0)
                return
            if url.path == "/api/info":
                if self._authed(q):
                    self._json(app.info())
                return
            if url.path == "/api/install":
                if self._authed(q):
                    self._json(app.install_state())
                return
            m = re.fullmatch(r"/api/job/([0-9a-f]{12})", url.path)
            if m:
                if self._authed(q):
                    job = app.jobs.get(m.group(1))
                    self._json(job.public()) if job else self._fail(404, "no such job")
                return
            m = re.fullmatch(r"/api/result/([0-9a-f]{12})/(video|timeline)", url.path)
            if m:
                if self._authed(q, allow_query=True):
                    self._send_result(m.group(1), m.group(2), bool(q.get("download")))
                return
            self._fail(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return
            try:
                if url.path == "/api/upload":
                    length = int(self.headers.get("Content-Length") or -1)
                    if length <= 0:
                        raise MouthParkError("empty upload")
                    kind, name = (q.get("kind") or [""])[0], (q.get("name") or [""])[0]
                    limit = GUI_LIMITS.get(kind, 0)
                    if length > limit:
                        self._fail(413, f"{kind or 'file'} is too large")
                        return
                    self._json(app.save_upload(kind, name, self.rfile, length))
                elif url.path == "/api/ffmpeg":
                    req = self._body_json()
                    spec = req.get("path") if isinstance(req, dict) else None
                    if spec is not None and (not isinstance(spec, str) or len(spec) > 4096):
                        raise MouthParkError("path must be text")
                    self._json(app.set_ffmpeg(spec))
                elif url.path == "/api/ffmpeg/browse":
                    chosen = pick_ffmpeg_dialog()
                    self._json(app.set_ffmpeg(chosen) if chosen else {"cancelled": True})
                elif url.path == "/api/install":
                    self._json(app.start_install())
                elif url.path == "/api/mouths/reset":
                    shutil.rmtree(app.mouth_dir, ignore_errors=True)
                    self._json({"ok": True})
                elif url.path == "/api/render":
                    req = self._body_json()
                    if not isinstance(req, dict):
                        raise MouthParkError("render request must be an object")
                    self._json(app.start_render(req))
                elif url.path == "/api/cancel":
                    if app.current is not None:
                        app.current.cancel.set()
                    self._json({"ok": True})
                elif url.path == "/api/quit":
                    self._json({"ok": True})
                    threading.Thread(target=app.shutdown, daemon=True).start()
                else:
                    self._fail(404, "not found")
            except MouthParkError as e:
                self._fail(400, str(e))
            except (ValueError, OSError) as e:
                self._fail(400, f"{type(e).__name__}: {e}")

        def _send_result(self, job_id: str, what: str, download: bool) -> None:
            job = app.jobs.get(job_id)
            if job is None or job.status != "done":
                self._fail(404, "that render isn't available")
                return
            path = job.output if what == "video" else job.timeline
            name = job.download_name if what == "video" else Path(job.download_name).stem + ".timeline.json"
            size = path.stat().st_size
            start, end = 0, size - 1
            code = 200
            extra = {"Accept-Ranges": "bytes"}
            if download:
                extra["Content-Disposition"] = f'attachment; filename="{name}"'
            rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
            if rng and (rng.group(1) or rng.group(2)):
                if rng.group(1):
                    start = int(rng.group(1))
                    end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
                else:
                    start = max(0, size - int(rng.group(2)))
                if start > end or start >= size:
                    self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                    return
                code = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._headers(code, _GUI_MIME[path.suffix], end - start + 1, extra)
            with path.open("rb") as fh:
                fh.seek(start)
                left = end - start + 1
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)

    return Handler


def run_gui(port: int = 0, open_browser: bool = True) -> int:
    from http.server import ThreadingHTTPServer

    workdir = Path(tempfile.mkdtemp(prefix="mouthpark_gui_"))
    with contextlib.suppress(OSError):
        os.chmod(workdir, 0o700)  # mkdtemp already does this; be explicit
    app = GuiApp(workdir)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _make_gui_handler(app))
    except OSError as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise MouthParkError(f"couldn't start the GUI on port {port}: {e}") from None
    server.daemon_threads = True
    app.port = server.server_address[1]
    app.shutdown = server.shutdown
    url = f"http://127.0.0.1:{app.port}/#token={app.token}"
    print(f"MouthPark {__version__} GUI running at:\n  {url}\nKeep this window open. Press Ctrl-C (or Quit in the app) to stop.",
          flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        if app.current is not None:
            app.current.cancel.set()
        server.server_close()
        shutil.rmtree(workdir, ignore_errors=True)
        print("MouthPark GUI stopped.", flush=True)
    return EXIT_OK


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def _int_in(lo: int, hi: int) -> Callable[[str], int]:
    def conv(s: str) -> int:
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {s!r}") from None
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"must be between {lo} and {hi}")
        return v
    return conv


def _float_in(lo: float, hi: float) -> Callable[[str], float]:
    def conv(s: str) -> float:
        try:
            v = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {s!r}") from None
        if not (math.isfinite(v) and lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"must be between {lo} and {hi}")
        return v
    return conv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mouthpark",
        description="South Park-style lip-synced mouth animation from an audio file.",
        epilog="Output type follows the extension: .webm (VP9 + alpha, default), "
               ".mov (ProRes 4444 + alpha, for Premiere/Final Cut/Resolve), .mp4 (H.264, opaque).",
    )
    p.add_argument("input", nargs="?", type=Path,
                   help="voice audio (anything ffmpeg can read); leave out to open the app")
    p.add_argument("output", nargs="?", type=Path, help="output video (default: INPUT with .webm)")

    g = p.add_argument_group("timing")
    g.add_argument("--fps", type=_int_in(*FPS_RANGE), default=18, help="frame rate (default: 18)")
    g.add_argument("--min-hold", type=_int_in(*MIN_HOLD_RANGE), default=2,
                   help="minimum frames a mouth must hold (default: 2)")
    g.add_argument("--rest", choices=("closed", "blank"), default=None,
                   help="what silence looks like: the closed mouth (default) or nothing")

    g = p.add_argument_group("look")
    g.add_argument("--pack", type=Path, metavar="PACK",
                   help="character pack (.zip, folder, or pack.json) made with mapper.html")
    g.add_argument("--mouths-dir", type=Path, help="folder with your own 10 mouth PNGs (default: built-in)")
    g.add_argument("--mapping", type=Path, help="JSON of phoneme → mouth overrides (see --print-mapping)")
    g.add_argument("--background", action="store_true", help="flesh-tone opaque background (no alpha)")
    g.add_argument("--bg-color", metavar="#RRGGBB", help="custom opaque background (implies --background)")
    g.add_argument("--character", type=Path, help="composite onto this image (PNG/JPEG/WebP)")
    g.add_argument("--canvas", metavar="WxH", help="transparent canvas size, e.g. 1920x1080")
    g.add_argument("--position", metavar="X,Y", help="mouth centre on the canvas (default: centre)")
    g.add_argument("--mouth-scale", type=_float_in(0.05, 20), default=None, help="resize the mouths (default: 1.0)")
    g.add_argument("--rotation", type=_float_in(-180, 180), default=None, metavar="DEG",
                   help="tilt the mouth, degrees counter-clockwise (default: 0)")

    g = p.add_argument_group("output")
    g.add_argument("--with-audio", action="store_true", help="mux the input audio into the video")
    g.add_argument("--preview", action="store_true", help="shortcut for --background --with-audio")
    g.add_argument("--keep-frames", type=Path, metavar="DIR", help="also write the PNG sequence here")
    g.add_argument("--timeline-out", type=Path, metavar="FILE.json", help="write phonemes + per-frame mouths as JSON")
    g.add_argument("--events-in", type=Path, metavar="FILE.json",
                   help="reuse phonemes from a --timeline-out file (skips recognition)")
    g.add_argument("-f", "--force", action="store_true", help="overwrite existing output files")
    g.add_argument("--max-duration", type=_float_in(0, 86400), default=DEFAULT_MAX_DURATION,
                   metavar="SEC", help=f"refuse longer audio; 0 = no limit (default: {DEFAULT_MAX_DURATION:.0f})")

    g = p.add_argument_group("utilities")
    g.add_argument("--ffmpeg", metavar="PATH", help="use this ffmpeg (exe or its folder) for this run")
    g.add_argument("--set-ffmpeg", metavar="PATH", help="remember this ffmpeg for every run ('auto' = use PATH) and exit")
    g.add_argument("--gui", action="store_true", help="open the MouthPark app in your browser (local only)")
    g.add_argument("--port", type=_int_in(0, 65535), default=0, help="port for --gui (default: any free port)")
    g.add_argument("--no-browser", action="store_true", help="with --gui: print the link instead of opening it")
    g.add_argument("--install-deps", action="store_true",
                   help="install the phoneme recognizer (no compiler needed; never changes what you have) and exit")
    g.add_argument("--doctor", action="store_true", help="check ffmpeg/allosaurus/encoders and exit")
    g.add_argument("--self-test", action="store_true", help="run the built-in tests and exit")
    g.add_argument("--print-mapping", action="store_true", help="print the phoneme → mouth table as JSON and exit")
    g.add_argument("--export-mouths", type=Path, metavar="DIR", help="write the built-in mouth PNGs to DIR and exit")

    p.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for ffmpeg commands)")
    p.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _setup_console(verbose: int, quiet: bool) -> None:
    for stream in (sys.stdout, sys.stderr):  # IPA on Windows consoles (cp1252)
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    level = logging.ERROR if quiet else (logging.DEBUG if verbose > 1 else
                                         logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_console(args.verbose, args.quiet)

    try:
        if args.set_ffmpeg is not None:
            if args.set_ffmpeg.strip().lower() == "auto":
                save_config(ffmpeg=None)
                print(f"ffmpeg: automatic (from PATH). Saved in {config_path()}")
            else:
                path, version = validate_ffmpeg(args.set_ffmpeg)
                save_config(ffmpeg=path)
                print(f"ffmpeg: {path}\n  {version}\nSaved in {config_path()}")
            return EXIT_OK
        if args.ffmpeg:
            global _FFMPEG_OVERRIDE
            _FFMPEG_OVERRIDE = validate_ffmpeg(args.ffmpeg)[0]
        if args.install_deps:
            install_recognizer()
            return EXIT_OK
        if args.self_test:
            return self_test()
        if args.gui:
            return run_gui(args.port, open_browser=not args.no_browser)
        if args.doctor:
            return doctor()
        if args.print_mapping:
            m = load_mapping(args.mapping) if args.mapping else Mapper()
            print(json.dumps(m.to_json_obj(), ensure_ascii=False, indent=1))
            return EXIT_OK
        if args.export_mouths:
            export_mouths(args.export_mouths, args.force)
            return EXIT_OK
        if args.input is None:
            # No audio given (double-click, "Run" in an IDE, bare `python mouthpark.py`) → open the app.
            return run_gui(args.port, open_browser=not args.no_browser)

        bg: tuple[int, int, int] | None = None
        if args.bg_color:
            bg = parse_color(args.bg_color)
        elif args.background or args.preview:
            bg = FLESH_RGB
        if args.character and args.canvas:
            raise MouthParkError("use either --character or --canvas, not both")
        canvas = parse_pair(args.canvas, "x", "--canvas") if args.canvas else None
        position = parse_pair(args.position, ",", "--position", allow_negative=True) if args.position else None

        output = args.output or args.input.with_suffix(".webm")
        opts = Options(
            input=args.input, output=output, fps=args.fps, min_hold=args.min_hold,
            mouths_dir=args.mouths_dir, mapping=args.mapping, rest=args.rest, background=bg,
            character=args.character, canvas=canvas, position=position, mouth_scale=args.mouth_scale,
            rotation=args.rotation, pack=args.pack,
            with_audio=args.with_audio or args.preview, keep_frames=args.keep_frames,
            events_in=args.events_in, timeline_out=args.timeline_out,
            max_duration=args.max_duration, force=args.force,
        )
        written = run(opts)
        if not args.quiet:
            print(f"Wrote {written}")
        return EXIT_OK
    except MouthParkError as e:
        log.error("ERROR: %s", e)
        return e.code
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return EXIT_INTERRUPTED
    except BrokenPipeError:  # e.g. `mouthpark --print-mapping | head`
        with contextlib.suppress(OSError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK


# The GUI page (served by --gui). Self-contained: no external requests.
_GUI_HTML: Final = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>MouthPark</title>
<style>
:root {
  --mat: #1f5d4c; --mat-line: rgba(255,255,255,.08); --mat-line-major: rgba(255,255,255,.17);
  --panel: #ffffff; --panel-2: #f1f4f2; --ink: #16201c; --ink-soft: #55635d; --rule: #d7dfdb;
  --marker: #ffd23f; --pin: #e5484d; --ok: #1e8a5a; --focus: #2b6cff;
  --font: ui-rounded, "SF Pro Rounded", "Segoe UI Variable Display", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  color-scheme: light; box-sizing: border-box;
  padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --mat: #163f35; --panel: #171d1b; --panel-2: #222a27; --ink: #eef3f0; --ink-soft: #a3b1ab; --rule: #33403b; --ok: #4cc38a; color-scheme: dark; }
}
:root[data-theme="dark"] { --mat: #163f35; --panel: #171d1b; --panel-2: #222a27; --ink: #eef3f0; --ink-soft: #a3b1ab; --rule: #33403b; --ok: #4cc38a; color-scheme: dark; }
html { height: 100%; scroll-padding-top: env(safe-area-inset-top, 0px); }
*, *::before, *::after { box-sizing: inherit; }
body { margin: 0; height: 100%; background: var(--panel); color: var(--ink); font: 15px/1.45 var(--font); font-variant-numeric: tabular-nums; }
button, input, select { font: inherit; color: inherit; }
.hidden { display: none !important; }

.app { display: grid; grid-template-rows: auto minmax(0,1fr); height: 100%; }
header { display: flex; align-items: center; gap: 14px; padding: 10px 16px; border-bottom: 1px solid var(--rule); }
header b { font-size: 20px; font-weight: 800; letter-spacing: -.02em; }
header .ver { color: var(--ink-soft); font-size: 13px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
.chip { font-size: 12.5px; font-weight: 600; padding: 4px 9px; border-radius: 99px; background: var(--panel-2); border: 1px solid var(--rule); }
.chip.ok::before { content: "● "; color: var(--ok); }
.chip.bad::before { content: "● "; color: var(--pin); }
.body { display: grid; grid-template-columns: minmax(0,1fr) 380px; min-height: 0; }
@media (max-width: 880px) { .body { grid-template-columns: 1fr; grid-template-rows: 56vh auto; } .app { height: auto; } }

/* stage */
.stage { position: relative; display: grid; grid-template-rows: auto minmax(0,1fr) auto; min-height: 320px; background: var(--mat); }
.tabs { display: flex; gap: 4px; padding: 8px 10px 0; }
.tabs button { border: 0; background: rgba(0,0,0,.25); color: #fff; padding: 7px 14px; border-radius: 9px 9px 0 0; font-weight: 700; cursor: pointer; }
.tabs button[aria-selected="true"] { background: rgba(0,0,0,.5); }
.pane { position: relative; overflow: hidden; }
.mat { background-color: var(--mat);
  background-image: linear-gradient(var(--mat-line-major) 1px, transparent 1px), linear-gradient(90deg, var(--mat-line-major) 1px, transparent 1px),
    linear-gradient(var(--mat-line) 1px, transparent 1px), linear-gradient(90deg, var(--mat-line) 1px, transparent 1px);
  background-size: 120px 120px, 120px 120px, 24px 24px, 24px 24px; }
canvas#view { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: grab; }
canvas#view.over-thing { cursor: move; }
canvas#view.picking { cursor: crosshair; }
canvas#view:focus-visible { outline: 3px solid var(--marker); outline-offset: -3px; }
.hud { position: absolute; right: 10px; top: 10px; display: flex; gap: 5px; }
.hud button, .hud span { background: rgba(0,0,0,.45); color: #fff; border: 0; border-radius: 8px; padding: 6px 10px; font-weight: 700; font-size: 13px; }
.hud button { cursor: pointer; }
.tester { display: flex; align-items: center; gap: 6px; padding: 8px 10px; background: rgba(0,0,0,.3); overflow-x: auto; }
.tester .shape { flex: none; width: 44px; height: 44px; border-radius: 9px; border: 2px solid transparent; background: rgba(255,255,255,.9); padding: 1px; cursor: pointer; }
.tester .shape[aria-pressed="true"] { border-color: var(--marker); background: var(--marker); }
.tester .shape img { width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
.tester .t-btn { flex: none; border: 1.5px solid #fff; background: transparent; color: #fff; border-radius: 9px; padding: 8px 12px; font-weight: 700; cursor: pointer; }
.tester .t-btn[aria-pressed="true"] { background: var(--marker); border-color: var(--marker); color: #16201c; }
.empty { position: absolute; inset: 20px; display: grid; place-items: center; text-align: center; color: #fff; border: 2px dashed rgba(255,255,255,.4); border-radius: 18px; padding: 20px; pointer-events: none; }
.empty h1 { font-size: clamp(26px, 3.6vw, 40px); line-height: 1.05; margin: 0 0 8px; font-weight: 800; letter-spacing: -.02em; }
.empty p { margin: 0 auto; max-width: 38ch; opacity: .85; }
body.dragging .empty, body.dragging .dropveil { border-color: var(--marker); }
.dropveil { position: fixed; inset: 0; z-index: 10; background: rgba(22,32,28,.55); display: grid; place-items: center; color: #fff; font-size: 26px; font-weight: 800; pointer-events: none; }

.result { display: grid; place-items: center; padding: 16px; height: 100%; }
.checker { background-color: #fff; background-image: linear-gradient(45deg, #d9d9d9 25%, transparent 25%), linear-gradient(-45deg, #d9d9d9 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d9d9d9 75%), linear-gradient(-45deg, transparent 75%, #d9d9d9 75%);
  background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0; border-radius: 10px; overflow: hidden; }
.result video { display: block; max-width: 100%; max-height: calc(100% - 70px); }
.result .card { background: var(--panel); color: var(--ink); border-radius: 14px; padding: 14px 16px; margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }
.resultwrap { display: flex; flex-direction: column; align-items: center; max-width: 100%; max-height: 100%; }

/* panel */
.panel { display: grid; grid-template-rows: minmax(0,1fr) auto; border-left: 1px solid var(--rule); min-height: 0; }
@media (max-width: 880px) { .panel { border-left: 0; border-top: 1px solid var(--rule); } }
.scroll { overflow-y: auto; }
.step { padding: 14px 18px 16px; border-bottom: 1px solid var(--rule); }
.step h2 { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; font-size: 16px; font-weight: 800; }
.step h2 .n { display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; background: var(--ink); color: var(--panel); font-size: 13px; flex: none; }
.step.done h2 .n { background: var(--marker); color: #16201c; }
.hint { color: var(--ink-soft); font-size: 13.5px; margin: 0 0 10px; }
.hint.warn { color: var(--pin); font-weight: 600; }
.row { display: grid; grid-template-columns: 84px minmax(0,1fr) 68px; align-items: center; gap: 10px; margin: 7px 0; }
.row > label, .row > span.lbl { font-size: 13.5px; color: var(--ink-soft); }
.row input[type=range] { width: 100%; accent-color: var(--ink); }
input[type=number], input[type=text], select { width: 100%; min-width: 0; padding: 6px 8px; border: 1px solid var(--rule); border-radius: 8px; background: var(--panel-2); font-weight: 600; font-size: 14px; }
input[type=number] { text-align: right; }
.pair { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 8px; margin: 7px 0; }
.pair label { display: flex; align-items: center; gap: 8px; font-size: 13.5px; color: var(--ink-soft); min-width: 0; }
.field { display: grid; gap: 4px; margin: 8px 0; }
.field > label { font-size: 13.5px; color: var(--ink-soft); }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 8px 13px; border-radius: 10px; border: 1.5px solid var(--ink); background: transparent; font-weight: 700; font-size: 14px; cursor: pointer; text-decoration: none; color: var(--ink); }
.btn.primary { background: var(--ink); color: var(--panel); }
.btn[aria-pressed="true"] { background: var(--marker); border-color: var(--marker); color: #16201c; }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btns { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.seg { display: inline-flex; border: 1.5px solid var(--ink); border-radius: 10px; overflow: hidden; }
.seg button { border: 0; background: transparent; padding: 7px 12px; font-weight: 700; font-size: 13.5px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--panel); }
.seg.wide { display: flex; } .seg.wide button { flex: 1; }
.check { display: flex; align-items: center; gap: 9px; font-weight: 600; cursor: pointer; margin: 8px 0; }
.check input { width: 18px; height: 18px; accent-color: var(--ink); flex: none; }
.check small { display: block; font-weight: 400; color: var(--ink-soft); font-size: 12.5px; }
.swatch { width: 30px; height: 30px; border-radius: 8px; border: 1.5px solid var(--rule); flex: none; }
.line { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
.filebox { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border: 1.5px dashed var(--rule); border-radius: 12px; }
.filebox .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.filebox .name small { display: block; color: var(--ink-soft); font-weight: 400; }
audio { width: 100%; margin-top: 8px; height: 36px; }
details > summary { cursor: pointer; font-weight: 700; padding: 12px 18px; border-bottom: 1px solid var(--rule); list-style-position: inside; }
details[open] > summary { border-bottom: 0; }
.mapgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(98px, 1fr)); gap: 6px; padding: 0 18px 14px; }
.mapgrid label { display: grid; grid-template-columns: 34px 1fr; align-items: center; gap: 4px; font-size: 15px; font-weight: 700; }
.mapgrid select { padding: 4px 4px; font-size: 12.5px; }
.mapgrid select.changed { border-color: var(--ink); background: var(--marker); color: #16201c; }
.mouthlist { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 8px; }
.mouthlist div { aspect-ratio: 1; border-radius: 9px; background: var(--panel-2); border: 1.5px solid var(--rule); display: grid; place-items: center; font-size: 11px; color: var(--ink-soft); overflow: hidden; }
.mouthlist div.have { border-color: var(--ok); }
.mouthlist img { width: 100%; height: 100%; object-fit: contain; }

.renderbar { padding: 12px 18px 14px; border-top: 1px solid var(--rule); background: var(--panel); }
.renderbar .btn.primary { width: 100%; padding: 12px; font-size: 16px; }
.bar { height: 8px; border-radius: 99px; background: var(--panel-2); overflow: hidden; margin-top: 10px; }
.bar i { display: block; height: 100%; width: 0; background: var(--ink); border-radius: 99px; transition: width .2s; }
.bar.indet i { width: 30%; animation: slide 1.1s ease-in-out infinite; }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(340%); } }
.stagetxt { display: flex; justify-content: space-between; gap: 8px; font-size: 13.5px; margin-top: 6px; color: var(--ink-soft); }
.err { color: var(--pin); font-weight: 600; font-size: 13.5px; margin-top: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }
pre.log { max-height: 160px; overflow: auto; background: var(--panel-2); border-radius: 8px; padding: 8px 10px; font: 12px/1.45 var(--mono); margin: 6px 0 0; white-space: pre-wrap; }
kbd { font: 700 11.5px var(--font); border: 1px solid var(--rule); border-bottom-width: 2px; border-radius: 5px; padding: 0 5px; }
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
.chip.btnchip { cursor: pointer; font: inherit; font-size: 12.5px; font-weight: 600; color: var(--ink); }
.ffpath { font: 600 12.5px/1.4 var(--mono); overflow-wrap: anywhere; background: var(--panel-2); border-radius: 8px; padding: 8px 10px; margin: 0 0 6px; }
.ffpath.none { color: var(--pin); font-family: var(--font); }
.okmsg { color: var(--ok); font-weight: 600; font-size: 13.5px; }
a.link { color: var(--ink); font-weight: 700; }
.stopped { position: fixed; inset: 0; z-index: 20; display: grid; place-items: center; background: var(--panel); text-align: center; padding: 20px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } .bar.indet i { width: 100%; opacity: .5; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <b>MouthPark</b><span class="ver" id="ver"></span>
    <div class="chips" id="chips"></div>
    <button type="button" class="btn" id="quit" title="Stop the MouthPark app">Quit</button>
  </header>
  <div class="body">
    <section class="stage">
      <div class="tabs" role="tablist">
        <button type="button" role="tab" id="tabChar" aria-selected="true" aria-controls="paneChar">Character</button>
        <button type="button" role="tab" id="tabResult" aria-selected="false" aria-controls="paneResult" disabled>Result</button>
      </div>
      <div class="pane mat" id="paneChar" role="tabpanel">
        <canvas id="view" tabindex="0" aria-label="Preview. Drag the mouth to move it; arrow keys nudge."></canvas>
        <div class="empty" id="empty"><div><h1>Drop a voice clip and a character</h1><p>Audio goes to step 1, images and packs to step 2 — or just drop them anywhere on this window.</p></div></div>
        <div class="hud" id="hud">
          <button type="button" id="zoomOut" aria-label="Zoom out">−</button><span id="zoomLabel">100%</span>
          <button type="button" id="zoomIn" aria-label="Zoom in">+</button><button type="button" id="zoomFit">Fit</button>
          <button type="button" id="zoomMouth">Mouth</button>
        </div>
      </div>
      <div class="pane hidden" id="paneResult" role="tabpanel">
        <div class="result"><div class="resultwrap">
          <div class="checker" id="videoBox"><video id="video" controls playsinline></video></div>
          <p class="hint hidden" id="movNote" style="color:#fff;margin-top:12px">Browsers can't play ProRes .mov files — download it to watch.</p>
          <div class="card">
            <span id="resultInfo"></span>
            <a class="btn primary" id="dlVideo" href="#" download>Download video</a>
            <a class="btn" id="dlTimeline" href="#" download>Download timeline</a>
          </div>
        </div></div>
      </div>
      <div class="tester" id="tester" role="group" aria-label="Test mouth shapes">
        <button type="button" class="t-btn" id="talk" aria-pressed="false">Talk</button>
        <button type="button" class="t-btn" id="silence" aria-pressed="false">Silence</button>
      </div>
    </section>

    <aside class="panel">
      <div class="scroll">
        <section class="step" id="s1">
          <h2><span class="n">1</span> Voice</h2>
          <div class="filebox">
            <div class="name" id="audioName">No audio yet<small>MP3, WAV, M4A, FLAC, OGG…</small></div>
            <label class="btn" for="audioFile">Choose…</label>
            <input type="file" id="audioFile" class="hidden" accept="audio/*,video/mp4,video/webm,video/quicktime,.mp3,.wav,.m4a,.flac,.ogg,.opus,.aac,.aif,.aiff">
          </div>
          <audio id="audio" controls class="hidden"></audio>
        </section>

        <section class="step" id="s2">
          <h2><span class="n">2</span> Character</h2>
          <div class="seg wide" role="group" aria-label="What to render">
            <button type="button" id="modeMouth" aria-pressed="true">Mouth only</button>
            <button type="button" id="modeChar" aria-pressed="false">On a character</button>
          </div>

          <div id="mouthOpts">
            <p class="hint" style="margin-top:10px">Renders just the mouth, ready to put on a character in your editor.</p>
            <div class="row"><label for="scale">Size</label><input type="range" id="scale" min="0.1" max="4" step="0.01"><input type="number" id="scaleN" min="0.05" max="20" step="0.01"></div>
            <label class="check"><input type="checkbox" id="useCanvas"> Place it on a bigger canvas</label>
            <div id="canvasOpts" class="hidden">
              <div class="pair"><label>W <input type="number" id="cvW" min="2" max="8192" value="1920"></label><label>H <input type="number" id="cvH" min="2" max="8192" value="1080"></label></div>
              <p class="hint">Drag the mouth to where your character's face will be.</p>
            </div>
          </div>

          <div id="charOpts" class="hidden">
            <div class="btns" style="margin-top:10px">
              <label class="btn" for="imgFile">Choose image…</label>
              <label class="btn" for="packFile">Open pack…</label>
              <input type="file" id="imgFile" class="hidden" accept="image/png,image/jpeg,image/webp">
              <input type="file" id="packFile" class="hidden" accept=".zip,application/zip">
            </div>
            <p class="hint" id="imgInfo">PNG, JPEG or WebP — or a pack from the mapper.</p>
            <div id="placeOpts" class="hidden">
              <p class="hint">Drag the mouth onto the face. <kbd>←</kbd><kbd>↑</kbd><kbd>→</kbd><kbd>↓</kbd> nudge, <kbd>Shift</kbd> ×10, scroll to zoom.</p>
              <div class="pair"><label>X <input type="number" id="mx" step="1"></label><label>Y <input type="number" id="my" step="1"></label></div>
              <div class="row"><label for="scale2">Size</label><input type="range" id="scale2" min="0.05" max="4" step="0.01"><input type="number" id="scale2N" min="0.05" max="20" step="0.01"></div>
              <div class="row"><label for="rot">Tilt</label><input type="range" id="rot" min="-45" max="45" step="0.5"><input type="number" id="rotN" min="-180" max="180" step="0.5"></div>
              <div class="row"><label for="ghost">See-through</label><input type="range" id="ghost" min="0" max="0.8" step="0.05" value="0"><span></span></div>
              <label class="check"><input type="checkbox" id="coverOn"><span>Cover the original mouth<small>Paints a skin-coloured patch over the drawn-on mouth.</small></span></label>
              <div id="coverOpts" class="hidden">
                <div class="line">
                  <span class="swatch" id="swatch"></span>
                  <button type="button" class="btn" id="pick" aria-pressed="false">Pick skin colour</button>
                  <input type="color" id="coverColor" aria-label="Cover colour" style="width:40px;height:34px;border:0;background:none;padding:0">
                </div>
                <div class="line"><span class="hint" style="margin:0">Dragging moves</span>
                  <div class="seg" role="group" aria-label="What dragging moves"><button type="button" id="dragMouth" aria-pressed="true">Mouth</button><button type="button" id="dragCover" aria-pressed="false">Cover</button></div>
                </div>
                <div class="row"><label for="cw">Width</label><input type="range" id="cw" min="2" max="600" step="1"><input type="number" id="cwN" min="1" max="8192"></div>
                <div class="row"><label for="chh">Height</label><input type="range" id="chh" min="2" max="600" step="1"><input type="number" id="chhN" min="1" max="8192"></div>
              </div>
              <div class="field"><label for="pname">Pack name</label><input type="text" id="pname" maxlength="60"></div>
              <button type="button" class="btn" id="savePack">Save as pack…</button>
            </div>
          </div>
        </section>

        <section class="step" id="s3">
          <h2><span class="n">3</span> Mouths &amp; timing</h2>
          <div class="seg wide" role="group" aria-label="Mouth artwork">
            <button type="button" id="mBuiltin" aria-pressed="true">Built-in mouths</button>
            <button type="button" id="mCustom" aria-pressed="false">My own mouths</button>
          </div>
          <div id="customOpts" class="hidden">
            <p class="hint" style="margin-top:10px">Pick all 10 PNGs at once. Name them after the shapes (ah.png, ee.png…), same size, transparent background.</p>
            <label class="btn" for="mouthFiles">Choose mouth PNGs…</label>
            <input type="file" id="mouthFiles" class="hidden" accept="image/png" multiple>
            <div class="mouthlist" id="mouthList"></div>
            <p class="hint warn hidden" id="mouthErr"></p>
          </div>
          <div class="row" style="margin-top:12px"><label for="fps">Frame rate</label><input type="range" id="fps" min="6" max="30" step="1" value="18"><input type="number" id="fpsN" min="1" max="60" value="18"></div>
          <div class="row"><label for="hold">Min hold</label><input type="range" id="hold" min="1" max="6" step="1" value="2"><input type="number" id="holdN" min="1" max="30" value="2"></div>
          <p class="hint">Lower frame rate and longer hold = choppier, more cut-out look.</p>
          <div class="field"><label for="rest">During silence show</label>
            <select id="rest"><option value="closed">The closed mouth</option><option value="blank">Nothing</option></select></div>
        </section>

        <section class="step" id="s4">
          <h2><span class="n">4</span> Output</h2>
          <div class="field"><label for="format">Format</label>
            <select id="format">
              <option value="webm">WebM — transparent (web, OBS, most editors)</option>
              <option value="mov">MOV ProRes 4444 — transparent (Premiere, Final Cut, Resolve)</option>
              <option value="mp4">MP4 — no transparency (sharing, previews)</option>
            </select></div>
          <div class="field"><label for="bg">Background</label>
            <div class="line" style="margin:0">
              <select id="bg" style="flex:1"><option value="none">Transparent</option><option value="flesh">Flesh tone</option><option value="custom">Custom colour</option></select>
              <input type="color" id="bgColor" value="#80c080" class="hidden" aria-label="Background colour" style="width:40px;height:34px;border:0;background:none;padding:0">
            </div></div>
          <p class="hint hidden" id="mp4Note">MP4 has no transparency, so a flesh-tone background is used when nothing else fills the frame.</p>
          <label class="check"><input type="checkbox" id="withAudio" checked> Include the voice audio</label>
          <label class="check"><input type="checkbox" id="reuse" checked><span>Reuse phonemes between renders<small>Only the first render of a clip listens to it; after that, look changes are instant.</small></span></label>
        </section>

        <details id="ffBox">
          <summary>Settings: ffmpeg</summary>
          <div style="padding:0 18px 14px">
            <p class="hint">MouthPark uses ffmpeg to read audio and write video. Pick the one you want, or leave it on automatic.</p>
            <div class="ffpath" id="ffPath"></div>
            <p class="hint" id="ffMeta" style="margin:0 0 8px"></p>
            <div class="btns">
              <button type="button" class="btn primary" id="ffBrowse">Browse…</button>
              <button type="button" class="btn" id="ffAuto">Use automatic</button>
            </div>
            <div class="field"><label for="ffInput">Or paste the path to ffmpeg (the exe or its folder)</label>
              <div class="line" style="margin:0"><input type="text" id="ffInput" spellcheck="false" autocomplete="off" placeholder="C:\ffmpeg\bin\ffmpeg.exe" style="flex:1">
              <button type="button" class="btn" id="ffUse">Use</button></div></div>
            <div id="ffStatus" role="status" class="hint" style="margin:6px 0"></div>
            <p class="hint" style="margin:0">Don't have it? <a class="link" href="https://ffmpeg.org/download.html" target="_blank" rel="noopener noreferrer">Get ffmpeg</a> — on Windows, <code>winget install ffmpeg</code> works too.</p>
          </div>
        </details>
        <details id="recBox">
          <summary>Settings: phoneme recognizer</summary>
          <div style="padding:0 18px 14px">
            <p class="hint">The part that listens to the voice (allosaurus + PyTorch). Installing skips the add-ons that need a C++ compiler — MouthPark has built-in stand-ins for them — and never upgrades or downgrades anything you already have.</p>
            <div class="ffpath" id="recState"></div>
            <div class="btns"><button type="button" class="btn primary" id="recInstall">Install</button></div>
            <div id="recStatus" role="status" class="hint" style="margin:6px 0"></div>
            <pre class="log hidden" id="recLog"></pre>
          </div>
        </details>
        <details id="adv">
          <summary>Phoneme → mouth mapping</summary>
          <p class="hint" style="padding:0 18px">Which mouth each sound uses. Changed entries are highlighted.</p>
          <div class="line" style="padding:0 18px"><label class="hint" style="margin:0" for="fallback">Unknown sounds</label><select id="fallback" style="width:auto"></select>
            <button type="button" class="btn" id="mapReset">Reset</button></div>
          <div class="mapgrid" id="mapGrid"></div>
        </details>
        <details>
          <summary>Limits</summary>
          <div style="padding:0 18px 14px"><div class="row"><label for="maxDur">Max length</label><input type="number" id="maxDur" min="0" max="86400" value="1800"><span class="hint" style="margin:0">sec</span></div>
          <p class="hint">Refuses longer audio. 0 = no limit.</p></div>
        </details>
      </div>

      <div class="renderbar">
        <button type="button" class="btn primary" id="render" disabled>Render</button>
        <div id="progress" class="hidden">
          <div class="bar" id="bar"><i></i></div>
          <div class="stagetxt"><span id="stage"></span><button type="button" class="btn" id="cancel" style="padding:3px 10px">Cancel</button></div>
        </div>
        <div class="err hidden" id="renderErr" role="alert"></div>
        <details id="logBox" class="hidden"><summary style="padding:8px 0;border:0">Log</summary><pre class="log" id="log"></pre></details>
      </div>
    </aside>
  </div>
</div>
<div class="dropveil hidden" id="dropveil">Drop to add</div>
<div class="stopped hidden" id="stopped"><div><h1 style="margin:0 0 8px">MouthPark has stopped</h1><p class="hint">You can close this tab. Run <code>python mouthpark.py --gui</code> to start it again.</p></div></div>

<script>
"use strict";
// ── Token & API ───────────────────────────────────────────────────
let TOKEN = "";
try {
  const m = location.hash.match(/token=([\w-]+)/);
  if (m) { TOKEN = m[1]; sessionStorage.setItem("mp-token", TOKEN); history.replaceState(null, "", location.pathname); }
  else TOKEN = sessionStorage.getItem("mp-token") || "";
} catch { /* storage blocked */ }
async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), "X-MouthPark-Token": TOKEN } });
  let data = null;
  try { data = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}
const $ = (id) => document.getElementById(id);
const MOUTH_PX = 200;
let INFO = null, NAMES = [];
const builtin = {}, custom = {};

// ── State ─────────────────────────────────────────────────────────
const S = {
  audioId: null, mode: "mouth",
  img: null, imgBytes: null, imgExt: "png",
  mouth: { x: 100, y: 100, scale: 1, rot: 0 },
  cover: { on: false, color: "#f1c2a5", x: 0, y: 0, w: 60, h: 30 }, coverMoved: false, coverSized: false,
  canvasOn: false, cv: [1920, 1080], cvPos: [960, 700],
  customMouths: false, shape: "ah", ghost: 0, talking: false, silence: false, drag: "mouth", picking: false,
  view: { zoom: 1, fit: 1, ox: 0, oy: 0, touched: false },
  jobId: null, packKey: null, packId: null, mapping: {},
};
let rendering = false, pollTimer = null;
const mouthImg = (n) => (S.customMouths && custom[n] ? custom[n] : builtin[n]);

// ── Boot ──────────────────────────────────────────────────────────
(async function boot() {
  try { INFO = await api("/api/info"); }
  catch (e) { $("empty").innerHTML = `<div><h1>Can't reach MouthPark</h1><p>${escapeHtml(e.message)}. Reopen the link printed in your terminal.</p></div>`; return; }
  NAMES = INFO.mouth_names;
  $("ver").textContent = "v" + INFO.version;
  const ffChip = document.createElement("button"); ffChip.type = "button"; ffChip.id = "ffChip"; ffChip.className = "chip btnchip";
  ffChip.title = "Choose which ffmpeg to use"; ffChip.onclick = openFfmpeg; $("chips").appendChild(ffChip);
  const ac = document.createElement("button"); ac.type = "button"; ac.id = "recChip"; ac.className = "chip btnchip";
  ac.title = "Phoneme recognizer"; ac.onclick = openRec; $("chips").appendChild(ac);
  applyRec();
  if (!INFO.allosaurus) openRec();
  for (const opt of $("format").options) opt.dataset.label = opt.textContent;
  applyFfmpeg(INFO.ffmpeg);
  if (!INFO.ffmpeg.path) openFfmpeg();
  const tester = $("tester"), talkBtn = $("talk");
  for (const n of NAMES) {
    const im = new Image(); im.src = "data:image/png;base64," + INFO.mouths[n]; im.onload = draw; builtin[n] = im;
    const b = document.createElement("button"); b.type = "button"; b.className = "shape"; b.dataset.shape = n; b.title = n; b.setAttribute("aria-label", "Show " + n);
    const bi = document.createElement("img"); bi.src = im.src; bi.alt = ""; b.appendChild(bi);
    b.onclick = () => { stopTalk(); S.silence = false; syncSilence(); setShape(n); };
    tester.insertBefore(b, talkBtn);
  }
  buildMapping(); setShape("ah"); syncMouthList(); syncAll(); updateRenderable();
})();
// ── phoneme recognizer ────────────────────────────────────────────
function applyRec() {
  const ok = INFO.allosaurus, chip = $("recChip");
  chip.className = "chip btnchip " + (ok ? "ok" : "bad");
  chip.textContent = ok ? (INFO.model_cached ? "phoneme model ready" : "phoneme model downloads on first render") : "recognizer missing — install";
  $("recState").textContent = ok ? "Installed and ready." : "Missing: " + INFO.recognizer_missing.join(", ");
  $("recState").classList.toggle("none", !ok);
  $("recInstall").classList.toggle("hidden", ok);
  updateRenderable();
}
function openRec() { const d = $("recBox"); d.open = true; d.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
let recTimer = null;
async function pollInstall() {
  clearTimeout(recTimer);
  let st; try { st = await api("/api/install"); } catch (e) { $("recStatus").textContent = e.message; return; }
  const log = $("recLog"); log.classList.remove("hidden"); log.textContent = st.log.join("\n"); log.scrollTop = 1e9;
  if (st.status === "running") { $("recStatus").textContent = "Installing… PyTorch is big, this can take a few minutes."; recTimer = setTimeout(pollInstall, 700); return; }
  $("recInstall").disabled = false;
  try { INFO = { ...INFO, ...(await api("/api/info")) }; } catch { /* keep old */ }
  applyRec(); applyFfmpeg(INFO.ffmpeg);
  const s = $("recStatus");
  if (st.status === "done") { s.textContent = "Installed. You're ready to render."; s.className = "okmsg"; }
  else { s.textContent = st.error || "Install failed."; s.className = "hint warn"; }
}
$("recInstall").onclick = async () => {
  $("recInstall").disabled = true; $("recStatus").className = "hint"; $("recStatus").textContent = "Starting…";
  try { await api("/api/install", { method: "POST" }); pollInstall(); }
  catch (e) { $("recInstall").disabled = false; $("recStatus").textContent = e.message; $("recStatus").className = "hint warn"; }
};

// ── ffmpeg settings ───────────────────────────────────────────────
function applyFfmpeg(ff) {
  INFO.ffmpeg = ff; INFO.formats = ff.formats;
  const chip = $("ffChip");
  chip.className = "chip btnchip " + (ff.path ? "ok" : "bad");
  chip.textContent = ff.path ? "ffmpeg" : "ffmpeg missing — set it up";
  $("ffPath").textContent = ff.path || "No ffmpeg found. Browse to it, paste its path, or install it.";
  $("ffPath").classList.toggle("none", !ff.path);
  const src = { "PATH": "Automatic (found on PATH)", "saved setting": "Chosen by you", "--ffmpeg": "From --ffmpeg", "MOUTHPARK_FFMPEG": "From MOUTHPARK_FFMPEG" }[ff.source] || "";
  $("ffMeta").textContent = ff.path ? [src, ff.version].filter(Boolean).join(" · ") : "";
  for (const opt of $("format").options) {
    const ok = !!ff.path && ff.formats[opt.value] !== false;
    opt.disabled = !ok; opt.textContent = opt.dataset.label + (ff.path && !ok ? " (not in this ffmpeg)" : "");
  }
  if ($("format").selectedOptions[0]?.disabled) { const first = [...$("format").options].find((o) => !o.disabled); if (first) $("format").value = first.value; }
  updateRenderable();
}
function openFfmpeg() { const d = $("ffBox"); d.open = true; d.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
function ffStatus(msg, kind) { const s = $("ffStatus"); s.textContent = msg; s.className = kind === "err" ? "hint warn" : kind === "ok" ? "okmsg" : "hint"; }
async function ffCall(path, body, busyMsg) {
  for (const id of ["ffBrowse", "ffAuto", "ffUse"]) $(id).disabled = true;
  ffStatus(busyMsg);
  try {
    const r = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (r.cancelled) ffStatus("");
    else if (body.path === null) { applyFfmpeg(r); ffStatus(r.path ? "Automatic — using the ffmpeg on your PATH." : "Automatic, but there's no ffmpeg on your PATH. Browse to it or install it.", r.path ? "ok" : "err"); }
    else { applyFfmpeg(r); ffStatus("Saved — MouthPark will use this ffmpeg from now on.", "ok"); }
  } catch (e) { ffStatus(e.message, "err"); }
  finally { for (const id of ["ffBrowse", "ffAuto", "ffUse"]) $(id).disabled = false; }
}
$("ffBrowse").onclick = () => ffCall("/api/ffmpeg/browse", {}, "A file picker opened — it may be behind this window.");
$("ffAuto").onclick = () => ffCall("/api/ffmpeg", { path: null }, "Checking…");
$("ffUse").onclick = () => { const v = $("ffInput").value.trim(); if (v) ffCall("/api/ffmpeg", { path: v }, "Checking…"); };
$("ffInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("ffUse").click(); });

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ── Step 1: audio ─────────────────────────────────────────────────
let audioUrl = null;
async function setAudio(file) {
  if (!file) return;
  $("audioName").innerHTML = `${escapeHtml(file.name)}<small>Uploading…</small>`;
  S.audioId = null; updateRenderable();
  try {
    const r = await api(`/api/upload?kind=audio&name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    S.audioId = r.id;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(file); $("audio").src = audioUrl; $("audio").classList.remove("hidden");
    $("audioName").innerHTML = `${escapeHtml(file.name)}<small>${(file.size / 1048576).toFixed(1)} MB</small>`;
    $("s1").classList.add("done");
  } catch (e) {
    $("audioName").innerHTML = `${escapeHtml(file.name)}<small style="color:var(--pin)">${escapeHtml(e.message)}</small>`;
  }
  $("audio").onloadedmetadata = () => {
    const d = $("audio").duration;
    if (Number.isFinite(d)) $("audioName").querySelector("small").textContent += ` · ${d.toFixed(1)} s`;
  };
  updateRenderable();
}
$("audioFile").onchange = (e) => setAudio(e.target.files[0]);

// ── Step 2: mode & character ──────────────────────────────────────
function setMode(m) {
  S.mode = m;
  $("modeMouth").setAttribute("aria-pressed", String(m === "mouth"));
  $("modeChar").setAttribute("aria-pressed", String(m === "char"));
  $("mouthOpts").classList.toggle("hidden", m !== "mouth");
  $("charOpts").classList.toggle("hidden", m !== "char");
  S.view.touched = false; fitView(); updateRenderable();
}
$("modeMouth").onclick = () => setMode("mouth");
$("modeChar").onclick = () => setMode("char");
$("useCanvas").onchange = (e) => { S.canvasOn = e.target.checked; $("canvasOpts").classList.toggle("hidden", !S.canvasOn); S.view.touched = false; fitView(); };
for (const [id, i] of [["cvW", 0], ["cvH", 1]]) $(id).onchange = (e) => {
  const v = Math.round(+e.target.value); if (!(v >= 2 && v <= 8192)) { e.target.value = S.cv[i]; return; }
  S.cv[i] = v; S.cvPos[i] = Math.min(S.cvPos[i], v); S.view.touched = false; fitView();
};

const IMG_TYPES = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
async function setImage(blob, name, fromPack) {
  const ext = IMG_TYPES[blob.type];
  if (!ext) { $("imgInfo").textContent = "Use a PNG, JPEG or WebP image."; return false; }
  if (blob.size > 64 * 1048576) { $("imgInfo").textContent = "That image is over 64 MB."; return false; }
  let bmp;
  try { bmp = await createImageBitmap(blob); } catch { $("imgInfo").textContent = "Couldn't read that image."; return false; }
  if (Math.max(bmp.width, bmp.height) > INFO.limits.canvas) { $("imgInfo").textContent = `Images can be at most ${INFO.limits.canvas}px on a side.`; return false; }
  S.img = bmp; S.imgBytes = new Uint8Array(await blob.arrayBuffer()); S.imgExt = ext; pickCanvas = null;
  $("imgInfo").textContent = `${name} · ${bmp.width} × ${bmp.height}`;
  if (!fromPack) {
    S.mouth = { x: Math.round(bmp.width / 2), y: Math.round(bmp.height * 0.6), scale: round2(Math.max(0.05, bmp.width * 0.18 / MOUTH_PX)), rot: 0 };
    S.coverMoved = S.coverSized = false; S.cover.on = false; $("coverOn").checked = false; $("coverOpts").classList.add("hidden");
    $("pname").value = name.replace(/\.[^.]+$/, "").slice(0, 60);
  }
  $("scale2").max = Math.max(4, round2(bmp.width / MOUTH_PX));
  $("cw").max = $("chh").max = Math.max(600, bmp.width);
  $("placeOpts").classList.remove("hidden"); $("s2").classList.add("done");
  setMode("char"); syncAll(); canvas.focus({ preventScroll: true });
  return true;
}
$("imgFile").onchange = (e) => e.target.files[0] && setImage(e.target.files[0], e.target.files[0].name, false);
$("packFile").onchange = (e) => e.target.files[0] && openPack(e.target.files[0]);

// Read a pack zip (stored or deflated) — the same format the mapper and --pack use.
async function readZip(file) {
  if (file.size > 128 * 1048576) throw new Error("pack is too large");
  const buf = await file.arrayBuffer(), dv = new DataView(buf), u8 = new Uint8Array(buf);
  let eocd = -1;
  for (let i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("not a zip file");
  const count = dv.getUint16(eocd + 10, true); let p = dv.getUint32(eocd + 16, true);
  if (count > 64) throw new Error("too many files for a pack");
  const dec = new TextDecoder(), entries = {};
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("damaged zip");
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true);
    const nlen = dv.getUint16(p + 28, true), elen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true), off = dv.getUint32(p + 42, true);
    const name = dec.decode(u8.subarray(p + 46, p + 46 + nlen));
    entries[name] = { method, csize, usize, off, flags: dv.getUint16(p + 8, true) };
    p += 46 + nlen + elen + clen;
  }
  entries.read = async (name) => {
    const e = entries[name]; if (!e) throw new Error("missing " + name);
    if (e.flags & 1) throw new Error("encrypted packs aren't supported");
    if (e.usize > 64 * 1048576) throw new Error(name + " is too large");
    const start = e.off + 30 + dv.getUint16(e.off + 26, true) + dv.getUint16(e.off + 28, true);
    const raw = u8.slice(start, start + e.csize);
    if (e.method === 0) return raw;
    if (e.method !== 8) throw new Error("unsupported compression");
    const out = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
    if (out.length > 64 * 1048576) throw new Error(name + " is too large");
    return out;
  };
  return entries;
}
async function openPack(file) {
  try {
    const z = await readZip(file);
    const metaName = Object.keys(z).find((n) => n === "pack.json" || /^[^/]+\/pack\.json$/.test(n));
    if (!metaName) throw new Error("no pack.json inside");
    const prefix = metaName.slice(0, -"pack.json".length);
    const meta = JSON.parse(new TextDecoder().decode(await z.read(metaName)));
    if (meta.mouthpark_pack !== 1) throw new Error("not a MouthPark pack");
    if (typeof meta.image !== "string" || !/^[A-Za-z0-9][A-Za-z0-9 ._-]{0,120}\.(png|jpe?g|webp)$/i.test(meta.image)) throw new Error("bad image name in pack.json");
    const ext = meta.image.split(".").pop().toLowerCase();
    const type = ext === "png" ? "image/png" : ext === "webp" ? "image/webp" : "image/jpeg";
    const bytes = await z.read(prefix + meta.image);
    const num = (v, d) => (typeof v === "number" && Number.isFinite(v) ? v : d);
    const pos = Array.isArray(meta.position) ? meta.position : null;
    if (!(await setImage(new Blob([bytes], { type }), meta.name || file.name, true))) return;
    S.mouth = { x: num(pos && pos[0], S.img.width / 2), y: num(pos && pos[1], S.img.height / 2), scale: num(meta.mouth_scale, 1), rot: num(meta.rotation, 0) };
    const c = meta.cover;
    S.cover.on = !!c; $("coverOn").checked = !!c; $("coverOpts").classList.toggle("hidden", !c);
    if (c) { S.cover.color = /^#[0-9a-f]{6}$/i.test(c.color) ? c.color : "#f1c2a5"; S.cover.x = num(c.center?.[0], S.mouth.x); S.cover.y = num(c.center?.[1], S.mouth.y);
             S.cover.w = num(c.size?.[0], 60); S.cover.h = num(c.size?.[1], 30); S.coverMoved = S.coverSized = true; }
    if (meta.rest === "blank" || meta.rest === "closed") $("rest").value = meta.rest;
    $("pname").value = String(meta.name || "").slice(0, 60);
    $("imgInfo").textContent = `Pack “${meta.name || file.name}” · ${S.img.width} × ${S.img.height}`;
    syncAll();
  } catch (e) { setMode("char"); $("imgInfo").textContent = "Couldn't open that pack: " + e.message; }
}

// ── Mouth artwork ─────────────────────────────────────────────────
function setMouthSet(customOn) {
  S.customMouths = customOn;
  $("mBuiltin").setAttribute("aria-pressed", String(!customOn)); $("mCustom").setAttribute("aria-pressed", String(customOn));
  $("customOpts").classList.toggle("hidden", !customOn); syncMouthList(); draw(); updateRenderable();
}
$("mBuiltin").onclick = () => setMouthSet(false);
$("mCustom").onclick = () => setMouthSet(true);
$("mouthFiles").onchange = async (e) => {
  const err = $("mouthErr"); err.classList.add("hidden");
  const files = [...e.target.files], problems = [];
  for (const f of files) {
    const n = f.name.replace(/\.png$/i, "").toLowerCase();
    if (!NAMES.includes(n)) { problems.push(`${f.name} isn't one of the 10 shape names`); continue; }
    try {
      await api(`/api/upload?kind=mouth&name=${n}`, { method: "POST", body: f });
      const im = new Image(); im.src = URL.createObjectURL(f); await im.decode(); custom[n] = im;
    } catch (x) { problems.push(`${f.name}: ${x.message}`); }
  }
  const sizes = new Set(Object.values(custom).map((im) => im.naturalWidth + "x" + im.naturalHeight));
  if (sizes.size > 1) problems.push("the mouth PNGs aren't all the same size");
  if (problems.length) { err.textContent = problems.join("\n"); err.classList.remove("hidden"); }
  e.target.value = ""; syncMouthList(); draw(); updateRenderable();
};
function syncMouthList() {
  const box = $("mouthList"); box.textContent = "";
  for (const n of NAMES) {
    const d = document.createElement("div");
    if (custom[n]) { d.className = "have"; const i = document.createElement("img"); i.src = custom[n].src; i.alt = n; i.title = n; d.appendChild(i); }
    else d.textContent = n;
    box.appendChild(d);
  }
  for (const b of document.querySelectorAll(".tester .shape")) b.firstChild.src = mouthImg(b.dataset.shape).src;
}
const customReady = () => NAMES.every((n) => custom[n]);

// ── Timing & output ───────────────────────────────────────────────
function pairInputs(rangeId, numId, lo, hi, onSet) {
  const apply = (v) => { if (!Number.isFinite(v)) return; v = Math.min(hi, Math.max(lo, v)); $(rangeId).value = v; $(numId).value = v; onSet && onSet(v); };
  $(rangeId).oninput = (e) => apply(Math.round(+e.target.value));
  $(numId).onchange = (e) => apply(Math.round(+e.target.value));
}
pairInputs("fps", "fpsN", 1, 60, (v) => { if (S.talking) { stopTalk(); startTalk(); } });
pairInputs("hold", "holdN", 1, 30);
$("rest").onchange = draw;
function syncOutput() {
  const fmt = $("format").value, bg = $("bg").value;
  $("bgColor").classList.toggle("hidden", bg !== "custom");
  $("mp4Note").classList.toggle("hidden", !(fmt === "mp4" && bg === "none"));
  draw();
}
$("format").onchange = syncOutput; $("bg").onchange = syncOutput; $("bgColor").oninput = draw;
function bgHex() {
  const v = $("bg").value;
  if (v === "flesh") return "#f1c2a5";
  if (v === "custom") return $("bgColor").value;
  return null;
}

// ── Mapping editor ────────────────────────────────────────────────
function buildMapping() {
  const opts = (sel) => ["closed", "clenched", "ah", "ee", "oh", "woo", "bite", "tongue", "uh", "rr", ""].map((m) =>
    `<option value="${m}"${m === sel ? " selected" : ""}>${m || "rest"}</option>`).join("");
  const def = INFO.mapping, grid = $("mapGrid"); grid.textContent = "";
  $("fallback").innerHTML = opts(def._fallback).replace('<option value="">rest</option>', "");
  $("fallback").onchange = () => { S.mapping._fallback = $("fallback").value; if (S.mapping._fallback === def._fallback) delete S.mapping._fallback; };
  for (const [ph, m] of Object.entries(def)) {
    if (ph.startsWith("_")) continue;
    const lab = document.createElement("label"); lab.innerHTML = `<span>${escapeHtml(ph)}</span><select aria-label="Mouth for ${escapeHtml(ph)}">${opts(m ?? "")}</select>`;
    const sel = lab.querySelector("select");
    sel.onchange = () => {
      const v = sel.value === "" ? null : sel.value;
      if (v === (m ?? null)) delete S.mapping[ph]; else S.mapping[ph] = v;
      sel.classList.toggle("changed", ph in S.mapping);
    };
    grid.appendChild(lab);
  }
}
$("mapReset").onclick = () => { S.mapping = {}; buildMapping(); };

// ── Canvas preview (shared with the mapper) ───────────────────────
const canvas = $("view"), ctx = canvas.getContext("2d");
function base() {
  if (S.mode === "char" && S.img) return { w: S.img.width, h: S.img.height, img: S.img };
  if (S.mode === "char") return null;
  if (S.canvasOn) return { w: S.cv[0], h: S.cv[1], img: null };
  const d = Math.max(1, Math.round(MOUTH_PX * S.mouth.scale)); return { w: d, h: d, img: null };
}
function mouthPos() {
  if (S.mode === "char") return [S.mouth.x, S.mouth.y];
  if (S.canvasOn) return S.cvPos;
  const b = base(); return [b.w / 2, b.h / 2];
}
function resize() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
  S.view.touched ? draw() : fitView();
}
new ResizeObserver(resize).observe(canvas);
function fitView() {
  const b = base(); if (!b) { draw(); return; }
  const pad = 48 * (window.devicePixelRatio || 1);
  const f = Math.min((canvas.width - pad) / b.w, (canvas.height - pad) / b.h, S.mode === "mouth" && !S.canvasOn ? 2.5 : Infinity);
  S.view.fit = f; S.view.zoom = 1; S.view.touched = false;
  S.view.ox = (canvas.width - b.w * f) / 2; S.view.oy = (canvas.height - b.h * f) / 2; draw();
}
const SC = () => S.view.fit * S.view.zoom;
const toScreen = (x, y) => [S.view.ox + x * SC(), S.view.oy + y * SC()];
const toImage = (sx, sy) => [(sx - S.view.ox) / SC(), (sy - S.view.oy) / SC()];
function zoomAt(f, sx, sy) {
  if (!base()) return;
  const [ix, iy] = toImage(sx, sy);
  S.view.zoom = Math.min(40, Math.max(0.2, S.view.zoom * f));
  S.view.ox = sx - ix * SC(); S.view.oy = sy - iy * SC(); S.view.touched = true; draw();
}
$("zoomIn").onclick = () => zoomAt(1.25, canvas.width / 2, canvas.height / 2);
$("zoomOut").onclick = () => zoomAt(0.8, canvas.width / 2, canvas.height / 2);
$("zoomFit").onclick = fitView;
$("zoomMouth").onclick = () => {
  if (!base()) return;
  const [mx, my] = mouthPos();
  S.view.zoom = Math.max(0.2, Math.min(40, Math.min(canvas.width, canvas.height) * 0.45 / (MOUTH_PX * S.mouth.scale) / S.view.fit));
  S.view.ox = canvas.width / 2 - mx * SC(); S.view.oy = canvas.height / 2 - my * SC(); S.view.touched = true; draw();
};
let checker = null;
function checkerPattern() {
  if (checker) return checker;
  const c = document.createElement("canvas"); c.width = c.height = 16; const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, 16, 16); g.fillStyle = "#d9d9d9"; g.fillRect(0, 0, 8, 8); g.fillRect(8, 8, 8, 8);
  return (checker = ctx.createPattern(c, "repeat"));
}
function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
  const b = base();
  $("empty").classList.toggle("hidden", !!b && (S.mode === "mouth" || !!S.img));
  $("hud").classList.toggle("hidden", !b);
  if (!b || !INFO) return;
  const s = SC(), [ix, iy] = toScreen(0, 0), w = b.w * s, h = b.h * s;
  const bg = bgHex() || ($("format").value === "mp4" && !(S.mode === "char") ? "#f1c2a5" : null);
  ctx.fillStyle = bg || checkerPattern(); ctx.fillRect(ix, iy, w, h);
  if (b.img) { ctx.imageSmoothingQuality = "high"; ctx.drawImage(b.img, ix, iy, w, h); }
  ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.strokeRect(ix - .5, iy - .5, w + 1, h + 1);
  const rot = S.mode === "char" ? S.mouth.rot : 0, rad = -rot * Math.PI / 180;
  if (S.mode === "char" && S.cover.on) {
    const c = S.cover, [cx, cy] = toScreen(c.x, c.y);
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rad);
    ctx.beginPath(); ctx.ellipse(0, 0, c.w * s / 2, c.h * s / 2, 0, 0, Math.PI * 2); ctx.fillStyle = c.color; ctx.fill();
    if (S.drag === "cover" && !S.talking) { ctx.setLineDash([6, 5]); ctx.strokeStyle = "#ffd23f"; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.restore();
  }
  const shape = S.silence ? ($("rest").value === "blank" ? null : "closed") : S.shape;
  const [mxI, myI] = mouthPos(), [mx, my] = toScreen(mxI, myI), size = MOUTH_PX * S.mouth.scale * s;
  const im = shape && mouthImg(shape);
  if (im && im.complete && im.naturalWidth) {
    const ar = im.naturalHeight / im.naturalWidth;
    ctx.save(); ctx.translate(mx, my); ctx.rotate(rad); ctx.globalAlpha = S.mode === "char" ? 1 - S.ghost : 1;
    ctx.drawImage(im, -size / 2, -size * ar / 2, size, size * ar); ctx.restore();
  }
  if (!S.talking && (S.mode === "char" || S.canvasOn)) {
    ctx.save(); ctx.translate(mx, my); ctx.strokeStyle = "#e5484d"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(-9, 0); ctx.lineTo(9, 0); ctx.moveTo(0, -9); ctx.lineTo(0, 9); ctx.stroke();
    if (S.drag === "mouth") { ctx.rotate(rad); ctx.setLineDash([6, 5]); ctx.strokeStyle = "rgba(255,210,63,.9)"; ctx.strokeRect(-size / 2, -size / 2, size, size); }
    ctx.restore();
  }
  $("zoomLabel").textContent = Math.round(S.view.zoom * 100) + "%";
}

// pointer & keys
let drag = null;
const draggable = () => S.mode === "char" ? !!S.img : S.canvasOn;
function hit(ix, iy) {
  if (!draggable()) return null;
  if (S.mode === "char" && S.drag === "cover" && S.cover.on) {
    const c = S.cover, a = S.mouth.rot * Math.PI / 180, dx = ix - c.x, dy = iy - c.y;
    const rx = dx * Math.cos(a) - dy * Math.sin(a), ry = dx * Math.sin(a) + dy * Math.cos(a);
    return (rx / (c.w / 2)) ** 2 + (ry / (c.h / 2)) ** 2 <= 1.3 ? "cover" : null;
  }
  const [mx, my] = mouthPos(), half = MOUTH_PX * S.mouth.scale / 2;
  return Math.abs(ix - mx) <= half && Math.abs(iy - my) <= half ? "mouth" : null;
}
function evPos(e) { const r = canvas.getBoundingClientRect(), d = canvas.width / r.width; return [(e.clientX - r.left) * d, (e.clientY - r.top) * d]; }
function setMouthPos(x, y) { if (S.mode === "char") { S.mouth.x = x; S.mouth.y = y; } else { S.cvPos = [x, y]; } }
canvas.addEventListener("pointerdown", (e) => {
  if (!base()) return;
  canvas.setPointerCapture(e.pointerId);
  const [sx, sy] = evPos(e), [ix, iy] = toImage(sx, sy);
  if (S.picking) { pickColor(ix, iy); return; }
  const t = hit(ix, iy), [mx, my] = mouthPos();
  if (t === "mouth") drag = { kind: "mouth", dx: ix - mx, dy: iy - my };
  else if (t === "cover") drag = { kind: "cover", dx: ix - S.cover.x, dy: iy - S.cover.y };
  else drag = { kind: "pan", sx, sy, ox: S.view.ox, oy: S.view.oy };
});
canvas.addEventListener("pointermove", (e) => {
  if (!base()) return;
  const [sx, sy] = evPos(e), [ix, iy] = toImage(sx, sy);
  if (!drag) { canvas.classList.toggle("over-thing", !!hit(ix, iy)); return; }
  if (drag.kind === "mouth") setMouthPos(Math.round(ix - drag.dx), Math.round(iy - drag.dy));
  else if (drag.kind === "cover") { S.cover.x = Math.round(ix - drag.dx); S.cover.y = Math.round(iy - drag.dy); S.coverMoved = true; }
  else { S.view.ox = drag.ox + sx - drag.sx; S.view.oy = drag.oy + sy - drag.sy; S.view.touched = true; }
  syncAll();
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("pointercancel", () => { drag = null; });
canvas.addEventListener("wheel", (e) => { if (!base()) return; e.preventDefault(); const [sx, sy] = evPos(e); zoomAt(Math.exp(-e.deltaY * 0.0015), sx, sy); }, { passive: false });
canvas.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 10 : 1;
  const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
  if (d && draggable()) {
    e.preventDefault();
    if (S.mode === "char" && S.drag === "cover" && S.cover.on) { S.cover.x += d[0]; S.cover.y += d[1]; S.coverMoved = true; }
    else { const [x, y] = mouthPos(); setMouthPos(x + d[0], y + d[1]); }
    syncAll();
  } else if (e.key === "+" || e.key === "=") zoomAt(1.25, canvas.width / 2, canvas.height / 2);
  else if (e.key === "-") zoomAt(0.8, canvas.width / 2, canvas.height / 2);
});
window.addEventListener("keydown", (e) => {
  if (e.key === " " && !/INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY|A|VIDEO|AUDIO/.test(document.activeElement.tagName)) { e.preventDefault(); toggleTalk(); }
  if (e.key === "Escape" && S.picking) setPicking(false);
});

// colour picking
let pickCanvas = null;
function pickColor(ix, iy) {
  if (!S.img) return;
  if (!pickCanvas) { pickCanvas = document.createElement("canvas"); pickCanvas.width = S.img.width; pickCanvas.height = S.img.height; pickCanvas.getContext("2d", { willReadFrequently: true }).drawImage(S.img, 0, 0); }
  const x = Math.max(1, Math.min(S.img.width - 2, Math.round(ix))), y = Math.max(1, Math.min(S.img.height - 2, Math.round(iy)));
  const d = pickCanvas.getContext("2d").getImageData(x - 1, y - 1, 3, 3).data; let r = 0, g = 0, b = 0;
  for (let i = 0; i < 36; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
  S.cover.color = "#" + [r, g, b].map((v) => Math.round(v / 9).toString(16).padStart(2, "0")).join("");
  setPicking(false); syncAll();
}
function setPicking(on) { S.picking = on; $("pick").setAttribute("aria-pressed", String(on)); $("pick").textContent = on ? "Click the skin…" : "Pick skin colour"; canvas.classList.toggle("picking", on); if (on) canvas.focus({ preventScroll: true }); }
$("pick").onclick = () => setPicking(!S.picking);

// fake talking
let talkTimer = null;
function setShape(n) { S.shape = n; for (const b of document.querySelectorAll(".tester .shape")) b.setAttribute("aria-pressed", String(b.dataset.shape === n)); draw(); }
function talkSeq() { const q = []; for (let i = 0; i < 60; i++) { const n = Math.random() < 0.18 ? "closed" : NAMES[1 + Math.floor(Math.random() * (NAMES.length - 1))]; const h = +$("holdN").value + Math.floor(Math.random() * 3); for (let k = 0; k < h; k++) q.push(n); } return q; }
function toggleTalk() { S.talking ? stopTalk() : startTalk(); }
function startTalk() {
  S.talking = true; S.silence = false; syncSilence(); $("talk").setAttribute("aria-pressed", "true"); $("talk").textContent = "Stop";
  let q = talkSeq(), i = 0; talkTimer = setInterval(() => { if (i >= q.length) { q = talkSeq(); i = 0; } setShape(q[i++]); }, 1000 / (+$("fpsN").value || 18));
}
function stopTalk() { if (!S.talking) return; clearInterval(talkTimer); S.talking = false; $("talk").setAttribute("aria-pressed", "false"); $("talk").textContent = "Talk"; draw(); }
$("talk").onclick = toggleTalk;
function syncSilence() { $("silence").setAttribute("aria-pressed", String(S.silence)); draw(); }
$("silence").onclick = () => { stopTalk(); S.silence = !S.silence; syncSilence(); };

// controls <-> state
const round2 = (v) => Math.round(v * 100) / 100, clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
function bind(rangeId, numId, set) {
  const apply = (v) => { if (Number.isFinite(v)) { set(v); syncAll(); } };
  $(rangeId).addEventListener("input", (e) => apply(+e.target.value));
  $(numId).addEventListener("change", (e) => apply(+e.target.value));
}
const setScale = (v) => { S.mouth.scale = round2(clamp(v, 0.05, 20)); if (S.mode === "mouth" && !S.canvasOn) S.view.touched = false; };
bind("scale", "scaleN", (v) => { setScale(v); if (S.mode === "mouth" && !S.canvasOn) requestAnimationFrame(fitView); });
bind("scale2", "scale2N", setScale);
bind("rot", "rotN", (v) => { S.mouth.rot = clamp(Math.round(v * 2) / 2, -180, 180); });
bind("cw", "cwN", (v) => { S.cover.w = Math.round(clamp(v, 1, 8192)); S.coverSized = true; });
bind("chh", "chhN", (v) => { S.cover.h = Math.round(clamp(v, 1, 8192)); S.coverSized = true; });
$("ghost").oninput = (e) => { S.ghost = +e.target.value; draw(); };
for (const [id, k] of [["mx", "x"], ["my", "y"]]) $(id).onchange = (e) => { const v = Math.round(+e.target.value); if (Number.isFinite(v)) { S.mouth[k] = v; syncAll(); } };
$("coverOn").onchange = (e) => { S.cover.on = e.target.checked; $("coverOpts").classList.toggle("hidden", !S.cover.on); if (!S.cover.on) setDrag("mouth"); syncAll(); };
$("coverColor").oninput = (e) => { S.cover.color = e.target.value; syncAll(); };
function setDrag(w) { S.drag = w; $("dragMouth").setAttribute("aria-pressed", String(w === "mouth")); $("dragCover").setAttribute("aria-pressed", String(w === "cover")); draw(); }
$("dragMouth").onclick = () => setDrag("mouth"); $("dragCover").onclick = () => setDrag("cover");
function syncAll() {
  const m = S.mouth, c = S.cover;
  if (!S.coverMoved) { c.x = m.x; c.y = m.y; }
  if (!S.coverSized) { c.w = Math.round(MOUTH_PX * m.scale * 0.6); c.h = Math.round(MOUTH_PX * m.scale * 0.3); }
  $("mx").value = m.x; $("my").value = m.y;
  for (const id of ["scale", "scaleN", "scale2", "scale2N"]) $(id).value = m.scale;
  $("rot").value = clamp(m.rot, -45, 45); $("rotN").value = m.rot;
  $("cw").value = $("cwN").value = c.w; $("chh").value = $("chhN").value = c.h;
  $("coverColor").value = c.color; $("swatch").style.background = c.color;
  draw();
}

// ── Packs out ─────────────────────────────────────────────────────
function packJson() {
  return JSON.stringify({ mouthpark_pack: 1, name: ($("pname").value.trim() || "character").slice(0, 60), image: "character." + S.imgExt,
    position: [S.mouth.x, S.mouth.y], mouth_scale: S.mouth.scale, rotation: S.mouth.rot,
    cover: S.cover.on ? { color: S.cover.color, center: [S.cover.x, S.cover.y], size: [S.cover.w, S.cover.h] } : null,
    rest: $("rest").value }, null, 1) + "\n";
}
function slug(s) { const t = s.normalize("NFKD").replace(/[^\w\s-]/g, "").trim().replace(/[\s_]+/g, "-").replace(/-+/g, "-").slice(0, 60); return /^[A-Za-z0-9]/.test(t) ? t : "character" + (t ? "-" + t : ""); }
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function makeZip(files) {
  const enc = new TextEncoder(), parts = [], central = []; let off = 0;
  const now = new Date(), t = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1), d = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const f of files) {
    const name = enc.encode(f.name), data = f.data, crc = crc32(data), h = new DataView(new ArrayBuffer(30)), c = new DataView(new ArrayBuffer(46));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(10, t, true); h.setUint16(12, d, true);
    h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, name.length, true);
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint16(12, t, true); c.setUint16(14, d, true);
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, off, true);
    parts.push(new Uint8Array(h.buffer), name, data); central.push(new Uint8Array(c.buffer), name); off += 30 + name.length + data.length;
  }
  const size = central.reduce((a, b) => a + b.length, 0), e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, size, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...central, new Uint8Array(e.buffer)], { type: "application/zip" });
}
function packBlob() {
  const s = slug($("pname").value || "character");
  return { name: s, blob: makeZip([{ name: `${s}/pack.json`, data: new TextEncoder().encode(packJson()) }, { name: `${s}/character.${S.imgExt}`, data: S.imgBytes }]) };
}
$("savePack").onclick = () => {
  if (!S.img) return;
  const { name, blob } = packBlob(), a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name + ".zip"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// ── Drag & drop anywhere ──────────────────────────────────────────
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if ([...(e.dataTransfer?.types || [])].includes("Files")) { dragDepth++; $("dropveil").classList.remove("hidden"); } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("dropveil").classList.add("hidden"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; $("dropveil").classList.add("hidden");
  for (const f of e.dataTransfer.files) {
    const n = f.name.toLowerCase();
    if (n.endsWith(".zip")) openPack(f);
    else if (IMG_TYPES[f.type]) setImage(f, f.name, false);
    else if (f.type.startsWith("audio/") || /\.(mp3|wav|m4a|aac|flac|ogg|oga|opus|aif|aiff|wma|caf)$/.test(n) || f.type.startsWith("video/")) setAudio(f);
  }
});
window.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) { const f = it.getAsFile(); setImage(f, "Pasted image", false); } });

// ── Render ────────────────────────────────────────────────────────
function updateRenderable() {
  const why = INFO && !INFO.ffmpeg.path ? "Set up ffmpeg first" : INFO && !INFO.allosaurus ? "Install the phoneme recognizer first" : !S.audioId ? "Add a voice clip first" : S.mode === "char" && !S.img ? "Pick a character image" : S.customMouths && !customReady() ? "Add all 10 mouth PNGs" : "";
  $("render").disabled = !!why || rendering; $("render").textContent = rendering ? "Rendering…" : why || "Render";
}
$("render").onclick = async () => {
  $("renderErr").classList.add("hidden"); rendering = true; updateRenderable();
  try {
    const req = { audio: S.audioId, format: $("format").value, fps: +$("fpsN").value, min_hold: +$("holdN").value, rest: $("rest").value,
      background: bgHex(), with_audio: $("withAudio").checked, reuse_phonemes: $("reuse").checked, custom_mouths: S.customMouths,
      mapping: Object.keys(S.mapping).length ? S.mapping : null, max_duration: +$("maxDur").value || 0 };
    if (S.mode === "char") {
      const key = packJson() + S.img.width + "x" + S.img.height + S.imgBytes.length;
      if (key !== S.packKey || !S.packId) {
        setProgress("Sending character", null);
        const r = await api("/api/upload?kind=pack&name=pack.zip", { method: "POST", body: packBlob().blob });
        S.packId = r.id; S.packKey = key;
      }
      req.pack = S.packId;
    } else {
      req.mouth_scale = S.mouth.scale;
      if (S.canvasOn) { req.canvas = S.cv; req.position = S.cvPos; }
    }
    const { id } = await api("/api/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) });
    S.jobId = id; $("logBox").classList.remove("hidden"); poll();
  } catch (e) { fail(e.message); }
};
function setProgress(stage, frac) {
  $("progress").classList.remove("hidden"); $("stage").textContent = stage;
  $("bar").classList.toggle("indet", frac == null); $("bar").firstChild.style.width = frac == null ? "" : Math.round(frac * 100) + "%";
}
function fail(msg) { rendering = false; $("progress").classList.add("hidden"); $("renderErr").textContent = msg; $("renderErr").classList.remove("hidden"); updateRenderable(); }
async function poll() {
  clearTimeout(pollTimer);
  let j;
  try { j = await api(`/api/job/${S.jobId}`); } catch (e) { fail(e.message); return; }
  $("log").textContent = j.log.join("\n"); $("log").scrollTop = 1e9;
  if (j.status === "queued" || j.status === "running") { setProgress(j.stage, j.progress); pollTimer = setTimeout(poll, 250); return; }
  rendering = false; updateRenderable(); $("progress").classList.add("hidden");
  if (j.status === "done") showResult(j); else fail(j.error || "Render failed.");
}
$("cancel").onclick = () => api("/api/cancel", { method: "POST" }).catch(() => {});
function showResult(j) {
  const base = `/api/result/${j.id}`, t = `t=${encodeURIComponent(TOKEN)}`;
  const v = $("video"); v.src = `${base}/video?${t}`;
  $("movNote").classList.toggle("hidden", j.format !== "mov"); $("videoBox").classList.toggle("hidden", j.format === "mov");
  $("dlVideo").href = `${base}/video?${t}&download=1`; $("dlVideo").setAttribute("download", j.download_name);
  $("dlTimeline").href = `${base}/timeline?${t}&download=1`;
  $("resultInfo").textContent = `${j.frames} frames · ${j.duration.toFixed(1)} s · ${j.format.toUpperCase()}`;
  $("tabResult").disabled = false; showTab("result"); v.play().catch(() => {});
}
function showTab(which) {
  const r = which === "result";
  $("tabChar").setAttribute("aria-selected", String(!r)); $("tabResult").setAttribute("aria-selected", String(r));
  $("paneChar").classList.toggle("hidden", r); $("paneResult").classList.toggle("hidden", !r); $("tester").classList.toggle("hidden", r);
  if (r) stopTalk(); else { $("video").pause(); requestAnimationFrame(resize); }
}
$("tabChar").onclick = () => showTab("char"); $("tabResult").onclick = () => showTab("result");
$("quit").onclick = async () => { if (!confirm("Stop MouthPark? Renders you haven't downloaded will be deleted.")) return; try { await api("/api/quit", { method: "POST" }); } catch {} $("stopped").classList.remove("hidden"); };
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════════
# Embedded default mouth artwork (10 × 200×200 RGBA PNG, base64).
# Replace with --mouths-dir, or dump them with --export-mouths DIR.
# ════════════════════════════════════════════════════════════════════════════

_EMBEDDED_MOUTHS: Final[Mapping[str, str]] = MappingProxyType({
    "closed": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAEqklEQVR42u3XPWs1WxXA8f/ae85JniRPfCluqX4HS8HGzsbi"
        "2gki2In1BbEVrf0AgmhrIbZ6FdsLgvgttPMheZIzM3sti5mT5BF7ueH/gxDm5ezZ+8xae60DkiRJkiRJkiRJkiRJkiRJkiRJ"
        "kiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
        "kiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiT9f8RrWswnP/64/vjnz6AFZHA4NpbTQmuNkYPeOzmK1oP7u5n3jyfu7h8ZI1nW"
        "QUSQVfQWVEHrjRwFQFH03iCLw7GzrrkdV3E4dE6nlaqiCm5ujoylaL3x+DDTWrCsSVGMkft8krc3l4x1cJpXlpEcpk4mvL29"
        "YH5cySxO8wIBVfDlL11ze3XJMgb//Nc7RhURcDx2ljmJBjm2c5kFEVTV/pJje9tVRAQji2Kbb1VRQOzPOWsR7Fe2EWK7F6C3"
        "4Ec/+A6//NXvwwT5HPj5T75fP/3Fb2htf8kF9XKV9T8W34IWbIG+H2duAZV7IFVu5+vlgAE16ilwz19i9PYUYZVF2z/XWqNq"
        "+w9sx5kQsC7JdOx74DaWZX0avx/O9wc5kjGSsRbRgt6CPrXn+WbRpwCC1raJPa0nAgpaDzK3xD9nRGXSpwmqqErWNZnO4+4J"
        "0VojM4neyJFUwHxaaVkso151gkyvZSF//ewfRA9ub99QOZ53/j2oR+ZToPap7zt5UJXEOcQDxnh5HE8BUueEqeeEi/36047b"
        "Yvt8BBF7Yu0BWi8CPSLox8N2/li0KchR9N63ZD0HZgSZW6WqaXtOJkRjC+w9cLeSUbSpMUbSe98rR5BtGwe2hBqjmPb7CmhT"
        "3xJpDKhOO3b6FKzz2Cpobt9HtP2voE+dqmB5OL36FuvVJMi3v/l1/vSXv3N/9wC1VYDM5+v1olxGLFsL1bZIb+eWqsXeiQRj"
        "1FPLFHthOLcgWwvzYfktts7uXIkq94TYW5zad/QXvcr23NzOj3UL2Oedvp6fx9OGvyfm/vxzS7QnVPDcEr1M3PME47/m93IB"
        "L+/fO7EPxv+g8u4Xvvfxt/jt7z71N8jnxQ+/+436w6d/o7XOcepcXx/pEVxeHCCL29s3dDpMcPfukcd54f3DicxiXhaiNe7f"
        "b8dfePuG66sLqKBPwXxaaL2TI5kOHRIOh4lgu55rsVZymleyinUdRNtalS0REnoj1wERtNa4uJgY88rx0Pnooy9yc/2G0+OJ"
        "eV1Y5uR4PGyB34NlGRCQe+UjY2uVCFqH+WFhVDKyuLw40KNzeXWgRnJ9dcHl8YLrm0seH07My8zd/czF5ZFO4+3tJb11rq/f"
        "MLXG1c0Fj/cz708n/v3u/VZ5luRwdWDMg6urI1/76lf45Ge/DiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
        "kiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ"
        "kiRJ0mvzH33dJb7FgOxQAAAAAElFTkSuQmCC"
    ),
    "clenched": (
        "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"
        "/et45/nD913L2krfb+/ORdkg/xee9bRL/Wvf+B5GowHDfh/TGOazmtmsRgnB5Zc/lZ958Qu43z3vw3kX3jO4C9YwnezhnePs"
        "eIy1Fu8sUmmEFPGfkK+QUgZ/XgU1azybY6MKM5lMwwnvU2QQTv8s0bq4kGOQnDaCigtNyvCzlHHjxKShiK+Zg+u44VLW3Eer"
        "SEfSHY/H0VUMCxvf2WQiPJZouYKL5/Jrh+cI7yNda3ovwQUMMY01Nnw2QuJ9lK49Wd1zzlM3TXiMFAgPVVUhPAgdLJE1BmMa"
        "rHHM6xnNvGa0vMTKygoHDh3monvdlyuefiV7O9uMJ3vccOONXHPN+3jPu97L9tYuQggOHFhj/8Ya27tjBPgf+oGH8+cfvmtI"
        "x/Iu5VJ9/8P9n/z1dSwvLYXgsq4Zj8c084YrrryST376Oq7502t44g88iX37N9jdOs3pE8fYOn2a2WRK0zQ459CVQlcVUsl4"
        "8svg79sYIygZ1ooQGNuktHcb4MbHpPNcKZUVIyklQspsIci5CIlSKi/ysPDIwayUAqVVDKBFXswix+mL6+H06U201tkSiOhC"
        "yajO5S8wXpeQ4f/nDShS3EO0LmEjBdcuXE+Kh6QQ8fpliGHwrX7TPlUr6ciYc4mvpbUOcYyH/mCAdY7JeMLO1lk2T5xkZ+ss"
        "3ntWV9a57EmX8ba3vo0v3PRZXvmKX2BpNGRzc4tZ3bCyPGJ1ZYm/+PCN3O8eB33ZIB1+7GmP9X/zkRtZXl3GxW9mPJ7yPY9+"
        "DB+89lqufvcf8/CHPpitzU1OHT/ObDbHNhYhJUrLvAlymUf2v30u1UCEz9zl0zIkxHxwxHEx6ee9y3FEiCV8TsA553N8kfx7"
        "GTeMzVaCaD2SC9bJeotWlk2X5RcUrcD29k64Vpeupc1+t9l3F38fVm96lpDDCfkd70LQLoSM8Q3ZShItkbU2JAOj+4jvCA7x"
        "4OhaQO/Sr8PucdaBAF0FkcEaS7/fR0lFr98DCJ7AfMbpzU02jx3jnEPn8Cu/8qt88eYbeebTL2c8ngThQEvW15b5ytdPcb8L"
        "7/xNcpfYIK95+U/5q//6kyyvLmHjFz6dTXnRz/xnPvLhv+P7Ln0MmyeOcWbzdPDLtc4neD4tk+RZacgJwHCqu6hqyXh6ihhP"
        "AKwsLyfHBhFji5S9Tid9Ou2lkOG0jf9OFiKs7+SGBVkzlaMIIdvni5nrsNhktEDBlQuCgM/75PyjF+C9ay1Ect18uykWrk8k"
        "Q9TWYYVSlZAETHkh71y2oFLI/PgsNqTar4jKVjjVhLUHkFT5k2stZ0yc2qioCRm+F6UUSqqQY+pXTKYzTtz+Tfatb/Cuq6/m"
        "13/1V9ndHWOtRyjF/o1VvnLrKZ706Hv6u/0GedXr/oh+vxdNv2QynvDSl76c17/+N6knU05861tZOk0Lu9XiJaYJ0qfSGmsW"
        "F3n60oJsGRQjLwTOh9+vrq5lv991yjlCHk90TtYYEItwAruoinkffk7PkSxCG6OAVCLWQIXNExaNa61bdGtcfAweloajmDvx"
        "ObufbIzqulIQpF/ZWkwpZSwPsXnjeu/w8TptPPGD5XD5OVKs47zPG0FK1Xlf5ODeufgZxRgMwBrXKYsJh0GQl1OMY/IhJaVk"
        "sDRkvDvh7MlNfuHlV/FfX/e77O2G2MsBa2vL/O1nvsYvPf/f+rvtBnniJQ/089oyHA7QSjHeHfPUp17Gq171y+xubzGbz+j1"
        "e/nUT66LjqpNCjp9/GKSXy1kx4eWYuGkTI8BWF1fj3FEOkJlfgy0z+W8i745WbpNizQE5W02X3ZqnBAC05h42oock7T7t70u"
        "H9UBIaDq9/AxbkguHJ4YC4lsERLGtCd2a1nD5k25kyBDh+e31oaNEK8lLXxiPJJWpFQqW4xkBVPI1rXgKU6iE3eJZLXjZ5by"
        "Lil28hDiLK3YOr3J837mBfzaL/8yW1u74QBSkuHygNe/7W/uvhbk45/5MsPREGMNe+MJ62urvPHNb6KZzZhMZigtc2VGcI8E"
        "KYxMp21aPOlkDIu1tTS+U2IeZM0o1zqHTmUoXesRT8tQGStbdyXFNfEPUnGiS5aAdKKHjWqMyTVe3sVFkeKTfH0OXH4jUUUC"
        "Z/xCDJUVr2idgvrlo6zq2jJ3H/+Wzt/GWqr0eYWkZrgCa4OFDNcSF0WnDD9tDBdl6fQ8Qsj83z593p24z6e8UMrtRDfUu7a8"
        "30WLI4QEB1unTvOLr7yKH73yR9je3sM5z6DXZ9o4nvfvnuzvdhvklS95lm+8o6o0SoZs7y+98r9wwXnnsbOzTVWpsBBjYk1K"
        "Ea64k5cQMtYnRRXHeZ/diu4XnM9rKfCiPTmrXpX7KrTSMdnWnqo+unJKRfUqWoUkCIjs/Lc+vLzDKS5kUMK6p+sd/5u4AXMp"
        "CyFWQIqc9AvxQY5YorVIylr4767SlB8TXVOpkpVIKpbMChbZVepk+xc+N9kp+2/fu0+1ADEv1G0NkEosWLT2etrrz5+blHgc"
        "9WTK773hdzh8zkEm0xleCHqjHlf/9cfufhbkXdd8iKrSeA/jyYwLzjuXZz/32ezsbIOIilTHVLiOIpVO8nzyR/85lZkYY9tT"
        "K5Z8h9Mx+MoiPkaFHYcnBJL5OX2wU/nUTad5DGKTX57cnWxx4s+uU/7uHbleKwf2zufYpQ18ZTZhTV1HxcjlJqtWBfMhARmf"
        "W0oZrWIblzjTBvLW2HxNIWAOeZtuuYw1NtdlJbUqqFed2CQmGoOKRs7lJKvVuojhWpKlFFlZI7+P5OqFLH5rcXZ29zh0zhFe"
        "+PwXYOpggStVsbM7vfttkFtvP8NotITSEmMsV/7wj7Cxsc58NktpiXwi+7hRWo0/uQ/pJCMGzqLjLvlOPaDIgXjKJqfevJRD"
        "S8WIyVcPzU8dK9EhPbYjokUPqVNIGJU0IVuBihyU88+eUHZO2lSTlU1hXHzJuoVy/XZBpifJry/FghLVrXgOmXSRN2qqE0uu"
        "6oJr5d1iDqRTUeBidYJUIkvPbc5EZPHBp2w+rcuXDrk2XvT5vc0nY370WVewurFCY0P8JqXkUQ+80N9tNsgznnxJbh51xiKA"
        "xz/+8TT1PJaDiFx0F07a2KttbC7TSDFFUmvAYxsTNxLtY70PNsJ7cOm/yQV+uSU1Fv35JNFE/d87l6XgtDDSGnCuVYdEzGAn"
        "qTefls4jRGhsctbhrcuybHiP4e91dH9SzBNiC4fH5XyMszbna0T0DZ2zWRwIfv3i+/d5MYach2naz8ul9+rTUvYL5TWptKZ9"
        "fZebwES0MCk+CZYofV7t36fPzzuXa82yFep4B0lk2dne5sJ73otHXvwoZuMZ3kOv1+OLt9x+9yk1+Ydbv4HuhQ66+bxheXmJ"
        "e150byazUO7RNKE8O2wej/Opocll314pnSXDlDNw3uOMyb6ydR7Z+fCttagoU4Jg3jTZVdK6j3WepmkQvV5YnNYho3TbdcGc"
        "S8G363T2yWjF6LgTbUNWShymGq7w92DiYlGVjr65pNI9kBJjLXiZr985j3DhvFUxcZnKzvPvfdsAlnvYU9VxPLXrmWlzOCIo"
        "Si5Z4ZjzcVisDdXGTWNCx2F0c0M1cBwyIVh4/+l15cL1+fz9YUW2/tY6tNatikiQg50XXHHFFVz74WsRUlD1KsZ1fffZILfd"
        "dhKtNMY6ZrOG+9//XjzoQQ8CPGsra60UmFyLdAqRfFwbsrjG5EDRGYdUCqUk9awOfRLOoXWFmdeoSsaFqXAepK5Y31in0hVN"
        "Y1hbX2EwGHDOkfNyjiV/kbF61hobStNjki2c3iJv3NTv4eP1ZVfF2gXJNFTgyuwaSqXRqspSbn/Yo98fcvDg4WgZm5z51r1+"
        "tqKhJmuxxqt7kPiYr3GxYjnnNlz4rJw1WeZO17G8vIIUgsFgwL3vexG9/oDD55yLkKoty8mzIXyu/coea8zzdE8KZ9tWZykU"
        "yDZx2V5v+5kqVfGTz3k2r37Na9k6c4rl5RWc81z1H5/hX/uWb99wiTttg8zmhrX1EY7YpzCd8Nrf/m180+C9oZ416Cou5Epg"
        "TTDFpjY0pmZ7a4ft7S02T57GekNd19jGcvTCC9i/fx/HvnmC2XyOVILeoEczNcgqFNfpKlSrKlWxu7NLv99HSMlNN93E857/"
        "PIinmnGW2awG6dg8dYa93V1M3XD4vMOM+iN0r2I+nWO9YTqeYoyljn0k9cywurFCT1dYZ5nPG9bWV9FKh/dlPboKyl1dN1S6"
        "x5du/nt0v48zDc95znM4evQeKK1xWE4cO0FTz+lVfdYPrOMNqEqBA6FCUKwqjfBBzTJNmzxUWjKNhZimsWzsX2djbZ2VtRUm"
        "exN0pbDOUU+DMHDdJz6FlKF+6+WvfCVrq6vM51P6wwFSKvr9XitGOJ/LYJIEr7UCL0JxpQ1DJMbjCfN6xt7eHpXuce6RI5xz"
        "7iGaWR2saGPx3jKfzcMhphVb2zvU8xohVD4or/vMzd/WdXqnVEy+5qrn+lf8xh+yurpKYxuEVMzG01xr1K16bYO6NnhOF63i"
        "iRg692h7xhHZ1KdkmLdBvXExh5ESi0II+sN+cDvqOnwhOcOYTjwR5c82Z5JP6M715DE+MfhVsh3O4JxfdBE7B7AjWKH+oJ+f"
        "ranrHN9IqbJLldz25KK1X6MPrtKC5WjFgZw76ShRuX9EBv8qKUlVr0JIFVzduo6Bu1+wqHm6Ct0hQ+T4ZeH764wx6taA+VYL"
        "y55Beq30vnqx+1N4wd7emPvf4wBfuvWU+K7eIEnqX9tYZzafhy/ex5gilZOn8TnpE42l3cnXznm1O26kjpmXSgZJV7Q5iZQo"
        "TO88BI3hZNVao5REa50l2BCAdhaG8wuvlXzs7sZOl6xip6HMrbLtQkkuRRIg8KlPvkEqGS2NzrKsizVU3bIToquWNmqSpttf"
        "t65fim0WXLxOmbyLkq81YVCFMU2MharcqNYtcWkTkC7P/6K73D0LGzUdMDZtzLhhuz327QESPlfrUpdkOBz29sZceGSNr952"
        "9tu2bu8UFevVv/Bc732o8EzqUsi42tBf0JjYTxGC5BQs59qlmJ11eSSNDepW7NATArx1QdFJlbrWhtxAOlHj6exjZWvKS3jX"
        "Kk8h15DUl/D4XF0b1SFBp1o4ybVhvk/MdAflLOczvA/980m29m3+Jq9tR+42bCto46LzreUk16O1C00S/onBDalU0rvwuj62"
        "DufUea7RSnkllzPe6R/bmPDdWNvWh1kXcyRBRUzfhbM2v9/0OXofSlusCc+TatJSBXX6f+n3Nv6c6tVc/C6980wmzXd/kF71"
        "RK7T8YRMb13XoY20MyEwmeSuKsSCS3NHU07upsvqSefv/iWUVPSHfaSWmKZhvLe3UFvVTjLsnCrxidMguW41bFKJutfVLdS9"
        "4/voKnJah75z7xyTOE6nTVp2XEulYrXsYsdjx8PJrk2ewLiYwMnuVHqASJab4GLJaLX2xnt5A3e/D586HKMk7Vn8Hlzn9WW0"
        "cDmf1cot0YW94/cZv0elqKoqWM4migpCfPdvEO9iYs9ahJDU85rV1VXOPe9IdCVULgfv5kKkFCip0JUOJecq6P5SCypVsba2"
        "yurqGitry/jo2ghkLh4UQmCdQ2mFloqNjf2c2jzNW3//95FKsb62yk885wU4Z2jqBucddV1DzEb3+hX9qs++/Rssx+EG81mD"
        "Ew5vPVpriAtMENppm7qhMQ3zed2esHjqxuQGK60Ua2sb/NZrf4MzW2ew1vHc5zyb+zzgvpw8eYoqunxLoxGDfp/BcMBsNsOY"
        "hslkHspRXCsDJ9UvVelaa4N0TStXh+pnhTUeVYXPsYliwbvf/R7OnD2Ds5bLnnoZD3n4wzh7+gxShgoFIcJgOyFgMp5inaWZ"
        "h372g4cPorViMp5gjGEymeIFTPYmTGcTts/uYG0Yt9Qf9FHpe4wTIeu6wTQ1xjqmkzHf+MbtuZrBO4dt3Hd/kB5LPvzK2irO"
        "WfZ2J/zQDz2da665JuQzhOCOuWafMrpdeTH5wPF/qxjMhmCyE4SKxfLwdNIqpfnyP/w9D3vww3Decb/735+bbropDIxztq1c"
        "janwthpWdFLiLpeqpNdr5161cufi64uF50uL9nse+UhuuukLWOe44YbP8ohHXIwxdZ6UuPD6qWkLz+IHc4egmUVLyIIlEJ1C"
        "ThElcc0TnvAEPvHxT+CB66+/gYsvfjimmcfZweTyedGpVUuTKmWshs61JHcwba387PLnSvw5ub7GGKpej49+9GM87bKnUlWK"
        "Xk+zN56xPupzcnv83S/zJqVGxvGet976T8znc8Z7ewjRZn7pdPulCR6524+UiOsEmx2VSnXyKPi2dDuM2nTs29jP6VObuKgA"
        "TSZj9nZ3GY93cx8DKd8Qg9iuSqNidW764tN1LGzcVNPkugtq0TMy1nH+BReEa5dh0sLmqU3q+Yzjx47F0n4f4yefy/6l6iYC"
        "2yklSaWSUuYRqSnvQnRdQlAeixPjYAhjHYcPn4N1Nm/ura0tptMpJ44fo9ersqqXui6tiaNP3R1cNQ+6V2Gb2A+S8jOd+q9U"
        "6tLWd0Up3zgOHNjP3s421lkGuo+PFdijqgeMv/sThYOhDv5pVJtOndpkMpvQ62msNdF3lblgTyqFEGHRCyniFw+6kgt9Im2y"
        "TuSJ6EqHzHkqMZdK0MRTSqQuQyWY1XUcrtanETVaqVaedK0FW0jELciTQCValw4WNmhWd+LjZRyBqgmztES31zz2j1c9Ta+q"
        "gmsW30vecB3ZFR8y8UG1I0+Kl9Lmkpjg9qnYHmxDwjROXAGBcGn4XdtzonuaqlIMBn1U6tCMAbrSCistulKYxuSarqy2eYHu"
        "RRfXuHhIyajuBcsrK9F+n97HshqD0poTZ8/k7y9VK19+2SW86Z3fvv6QO60Wa315GMoXjKOqKk6cOME/3fJVBoM+tmmQSWmy"
        "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"
    ),
    "ah": (
        "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"
        "6LBhrdaRSa5fsxlfrfWtz8I8BU3s12wJRISdnYL/9pt/GB7kVmLTGlplZ4Sch4exJVOftbarVm1LBeQWNkkKbKlN7c509aZJ"
        "c2veyFSrUJwiXgSy0q7mRaxH2x+fx5xdqV3I8yQS5qf1vjU6VLuS1LDh40ktdzGdU/e1C+z0LNhU+klDNv0MwKh1FW/n+0ls"
        "tNCQt+ScUUrxw16W4gbn0yJh/TW8NbXRtGuSC+bt8abMYFf8JqybFYSEm0cnYSC3lMHKWW9/yRlSST7aRpSnyW806bXow0NM"
        "t7jVGkyMaPnHVo5A1u+Qt2oAVY0nqdI2TcOnc8nOmtnEE5AYafNYPXuxJGuybD0ko9Yit/Baq3s6hnb6Wbej0tb+WswD0vh8"
        "jNomSlOONN3uynJ5w5aGlyXnofnyPpeMaVC9EwlF+3BYZf322Vq4KK9BDHj1+V5hIP/f8fVveDUzM5bd4oWpbL0ck7jQGRmb"
        "kateYjCb2pmneUkudgtqv4fG8j7TqrWtmsOWnD2NCSDmecx7WY5jXxt9Gm3yfuTFv1JUct67vw7JT7rXd0x9K55NWSoItVtV"
        "5+U9Ih1OUTOP3Ml6N8yLVJ2uMqQGmLReQ3HQp7qM6740h0k5CbVrvwetG5mRrhvJi67dOA4DuVV4x8+8Gx2EzWbjodS6VmWc"
        "khfcRjV6VM3nXgbZ3ASXjPstq6N2PBdQmOfoXRff6Oq1bVqTtpbZZNUo+dJN9xwjZje6F9NkRWlwosfM0rUtJPL3nMjn8fbW"
        "3OBomu4uNGwe01Amta8v5Jk6FYGhBJ4XAA12j3xaS87Z23UtN+larNUWta2+eUz6t9YZ/+hb/jKHgdwCfOTaoU9FzD4xPetB"
        "7H7rAfBZt5jbVmH9DzyGDbDs8UtJYvf5sGOavO61BmXE7FDYn6H5z8yEGYVrsvum0xrdsCYmCtylsxCjBddCHpOsr9rpaDGO"
        "j0lNynyBVUTIrjiudXg3C4NsNrH3qFuFXT/H3sb57R0+J8ze+8jxxro56+AkwH/eJj+aZzPv9/4PPxQe5Fag9o6yLK4hmqcW"
        "5pyx7CzuMUzcN7RYejMm8hGaop4dFWMfFK0HQm7T5DUVTGpfa4y12zXbmgDzXBoOdR2fY17Kcp45RLMJItKg1bxF2Oo75lqy"
        "CgaNaiVPmuCVf1+vpskUTVPjaZqnNXs8THOyxgKSUQuydXFjHzVGTqL5jYVSPt/XC/88dp9obnV0choGcivQO1wXNQ9lrt5P"
        "PnRQllyafkqYHet3aH7WfToiT9uXPJcZVXcf9gy4OtcGU49JJxK72wT2pqyXhWEm+DNdV9aVCN3qEvr+/Husr0XrNE1XL1gX"
        "n+vKtGJuOYZQu83ZJsmRxphRex2J0lD/muYK04TH2l2+M/599NNUNfZ1Xf1zZ+tstN4X61lp7FX305NNGMitQM7kMbPro5JN"
        "+2ja6jr6tclzENL6CI8VBlMhyxJo+/5kf9+6ZQdDZfkHTd8r4RV7dZ+ZsbO7MxgflYzLDQtneITByh6vp5y34n7WxHmetbU9"
        "Mws+A8XCINDIi/zzmKjwlMn77WWdQR8zd03Iqa+zc1dpfh9LfxIp3dw95Jr3MwI+cHH6/qFTW3ZzGMitwI6OyHSmKhGYyeNq"
        "C5EsGe2tuzo12QbYqV+8rm2iapPruLqHOKNbzgSFNPVaOGOFMWXdqGOrt+SU0FYdnUO6yxDJcyNgzP2V188+SML2JFrNBpiU"
        "x+Yp1WKsq3GWw9PsbaauSp4YORtex8A0zWRbLlW1Mi7GPybI89Sbb5V+e18+ScYfl5xlXFIKA7kVeMLlA9jsQdMLjfyCfOYU"
        "TR7A6FubMpjK6DMvZTQh1VZRW3MWax7qbJSmJZqzzqprRd8S5UQJO0tx6tWYHs+NytBBuUfSBTpiRNjaiz6vP7CwyDyPMVzz"
        "zZ58Kn2aOhfJ369UvLPXKToGE+WiRc3hrHg4b8H1/YhaJ1qWMlX80/BqKW1t0PI8iICn3nUlDORW4IlXLqCuqy+AsamFwjYN"
        "5sbrCF36ojebjR8Qq7zbEGcLE+wgeNeeThbx4QlWN9CfkVpJn0IojHADQFbNk3k0Y7WM6TE6tfeOVisI7BVw5u40s9UpUspY"
        "lmWwYDx2Dco8LpWaqNcSD8pTeEXbntH6OWwjLkbPjH0+3VsAxgAJa6QSJrE460aTGsAKj74NKyfUtXl95s7LYSC3BE9/2p0o"
        "JbkK1/o5SPMTY46MFbKpg17swujyy3l7g2zOZWtelQ2ABrsSxEkAm1Yi9jh6w+3GNWo3l1Hh9tFaPLyIGaD9XJ7oX7vVvUsS"
        "Qx6yWTeD/tWOwTHhZKrJTMOpbZVBa81pb6nGj8Ie+/xhbDFWOY1NUzL7S9e3+dfyWIuQxtwu769XxpESYX+/4C3f/cMUBnIL"
        "8NxnfTzWzTo1E9UxbADkwwqkogufTm5u35t4lH2x1QI5F58oOE89macIjr2Cwu7Ms3Obb5zt0xifwV75KCHNE9r0/HO//Fj2"
        "k3ReFsMVwZA1abKieaJVtbLdtdFpfB5wpW+fJ0tqOGR0dO9mPKxqZZ3DNX0OnadLQef45iLLSJsZ+lSB39qjqIacU8Lh9SMc"
        "7O2e2Ur6mbDqvSXz5TueiKOjw2nrU9Lp4tnXj3XtnDPjAaRBqfmynBHb236P3iWpn4clyEjSUXWvSlXa16qGeTmNYqWHNZZL"
        "PKYvXpq1htGYx5KV0dln7jIGbTr33hvblIiw6g7D5tR18oYp6zXHKN8IvTwbWOetnhpr6BpGO4ZYeGekiiRNlyZU7rTEVBk6"
        "q9uYmnldVzz5iRfwRw88HB7kVkFyj+p7vX2ioMbqmBZWtqpdf9oHYVqibD3VxttP44BqrXLINWG1UEdaWXnUCfrwMl4is8FQ"
        "kC7CrqxO783zHAt3rN/c+kR6Y6RSvP5C0yRGLzrq7CzWz0HmU8E9TFkWHz3KUBWyyj7G3hLy98s8jMMmlgxDxNYg7NE/IxX9"
        "tVahtrkjJfM02Fris64VlAltrZ4rUTu7belnwkD2l4Lem1TNVUMkO/5GPWRWsoIgM3iNrk1js6xQsmMVQNJdIJRGMW0edWOe"
        "wxgZ1k20ROJZutcr4IfUlbdewZf6wrzrMOvggzlWb9p0NSbSW/+KyVNsUafmU7ptiiGSdVtoA1f3YtREMGT17pWUJWu16Xue"
        "3jdNHjePbVxepU/bHjJPdSbujLwUX3T6wuc9MwzkVuKz/+zzcfPmofySt8KgMRndp5gnHajQpZPPmRqdwMFWmW59q/HKaFQX"
        "Aaahe/IdghPFauzUqEiPvmw5ZNKhZ6/TfE6dVb6dtyQotgatz9Xx3kY7byI3fNvXYfUYo4zLIsMT5lxH9qSQV+TNE5vzm8kB"
        "k+DYbGDPv8yzKYnRZjZMmjq9aNpb3yIKPu+zPzVykFteMCyJL16+grWuEjpZQ1IiP5y9yuR1ScxHQm23JoGnCvCYdTXH5KMW"
        "MXb5uTjRWCce+/+GTJxVkavPp7Nw5xFSpNqppjJ26+1Y1+p/9tpOTr6g05qxLK5vKv9wSYqFUlbP0XCt1jpNgKStHSr2+dga"
        "Nq+k2654fT67DCz59lNjewsTaYI/9pPY50VEIN7g2vXTGNpwq7HkhLpZvVaRi8rRlZId7NFgb2wIdJpZqjTWH/jhnozDYn+e"
        "eq3tBu+taxNU0zUGbZrDpRVvPFbxC6dSzbBIJ8B3nR5fcpLqvw+EHqGTvReb7dv69jJAk5ZnHShnXY8jhMxblXRfCa05zMzY"
        "CY086F5f/smsS3yGRssX9XRW2pimlW3yPMdHJ3jChQOcZZwZA3nuM56Edd3IL113jttGp3WtXrBLj5n8MXb4qTZLm5Ro2r/h"
        "44OK9oTrMGi5yacD5gUDqxWQ376SB0mhcrTWtpF8642dtTuy6BwsW7wjJEQfbcHa22H5lU81MRWuv24a+9nVIzalfW35Zpp2"
        "wPuFoBeF5WWYNlL5OmoaXYmmKkg5SS6oSgLPwTDyK5uv1VrD577ixWfaQM6U60sEvnL5ClbttbZ1aV5h7qMwaDc1ayuphRY+"
        "stQrv6Ozrmjiy1rbsJvSqOJ5+6yFHVaIM4OQw0aeH8yyEA9/0phXBbLpiTJHt7amI1F18B2TLtsck+udeTKlMrNrvXzHqJIR"
        "5jVG4VAN3ya8T95x9MpO69Z0IzBzx6ozyUoRyre16nlgmqT/tpLuxvUb6Mwxm/fxwt1PvYzN5vQx2qlR9bb+gzlBzWnUQfxK"
        "0Bbc1psn3i6P18IaANdTja7F7R5vPW1eoTctkxhDGbStvrYxLqfpTpAxGWRZhKHb2dnxHEn6Wxg5F/F6kP70lJWlmiaye5tI"
        "H+ugc0qyaGgSXw5NFbm2y1S8oDFlhZ0NzEKD1+bD4jabjU9/TJNq13KSvCScnh7jrjsv4azjTBnIm7/6dTg6PpH5tlrNlblY"
        "I5GdF3Zbg5QXBbn7/CZnWoyVYvYEc11XZb/GuFCT2vtyHTadUvKE3aapbC3DmVpaLYeR3ER6K+qqOw4nitkmM9oERLZqto4l"
        "hc7rknxsTGex5N7meCXd5ZFoMGa+paqN9mLutqRU5xz7REidNUzkxdSS82hR1pzFBJSyAZcBTjg9XfFX3vC6M28gZ879XdjN"
        "jLI3dpqTJb7Jq82AsTg6nR1je+y828+UspakG5tl9K+FUmOn+iwn795vMr7enMECDRmLTw+Z1jL4ILlppGht2z3xtqrZK9h1"
        "9QR8Z2dHpr8/ZrWBKwJUhJnnddAmqefBjs0e0SrwYJ5CuTTtKhQDFr0ZObVsdRDzKienGyyJcfPwNDZMPd74old9Oo6ODpEz"
        "gbl53D/WEdBQuGq/td34Mo1Qb1oddtC9mw9TMq2ddVpdaz41EWObla9dm5mdkQfYjkCbldWmUZ0y6Fq7AmW5lPeg20R4mcul"
        "W5u4g2EV+u7GbwPyRMGr8hnfNcLeeTgPozCFsTN9mqd4CNVHnz9pH7sxXWutPmRbaPbsBmmdiATC5uQEb/jiV+M84Exa+FIS"
        "L8uuFLhsbCeNgc0mSOx629lNPu/0kwFsQ7Y7d+uJXEN6N6S1FNjd3XOZvD2uDVWDCxyb5C80GrosKR593YPyNdaozpNFePxW"
        "Wm0yVNpCQjCWZRl0tMvjx8hS34A1TTEZ7cijhiEepKPkjFUHYtj4UrtQzHtiUgmYtzMmzYZWmGG31nFyfIx6hgdWn2kPAgBv"
        "/qrX4Pj4GDllkHP8Y3QoacXc9peTtrhK0a+PopdpiKxOYRVpZYuahmilLIP9IZGm9CZ0se3508t8bH/VA85aUOPePVSx2VsW"
        "IsnEkT5tzx35Qp+mQlqB1GsUHdOOQR6r5Zi3CojemagkhIVcUNYsaVEx0bx7Eb6ay2sokBzL1j6sm9VD0a7at6PDI7zyMz4Z"
        "5wVn1so/7kkX+U+un+Lg4ILcbETTUpmxyNLi+GUpUneoFYy+pQa2Q2wDCmxFm9GhXRN7UfNmsO4PEZXrMh3IsYDG2Cpg5Em+"
        "L53naScqdydyitor3UpHs41LdZUwe+KebG3DlrFr0tx5qzfdPCt4qH6TbtUyStdk9zatEnpRSP9I36J1bfaXrMQjbE5X9HqK"
        "0/V8eI8z60EA4EMfuUl1XTUph8978hZZvanLIpMLV9UnJdu34T3b3acY2rwnY5WcFd6SrovuaLPKXvF1Xb3eYKGQhV+lFEmS"
        "G/uExFb7aNWdNmTZng/zIkapyvysUYgEdHI6P2ZqinooM1TLabiPqfU+GX5aWz12qFsS3r3wyBPb5Z7V9pAoK9i1tWCzdpyc"
        "HOOt3/JXcZ5wprc3fMVrX4bDG9flFzy9E2N6ilKhdsAsvC952t9BYy8gpbH7w1p87YfSY6rFc31kniAP74mn4XV0qrysLpNh"
        "buY1RLJefD+is2fa/AWVsFhVHLodd3RGkktf5qksvrk3jYPuVXAnNdh72ud6kPycEBJVV1enTD4NPyV4vcemwtTjQ3ztV7wK"
        "f+cf/DM6TwZy5t/M85/zFL7v/odw+coV7RlXGUZj7OwszvbY12zhi4URptj1hiWGrivrQ5RICbmMqY5ErLWGvtWwZXqk1nhr"
        "5ZnvF5wq8fZ6RJ1rm6dGb8kIpTCYOAzDhI3+zPq+dOvu+Bkb45O8Ycrp62m6irXK+iauTMP78PS55ZHfWPgJMHJZcO3qVXzm"
        "C56BX733A+fKOM6FgQDAU++8yP/jkUNcvnxJzmqCSC/0lynFu3EwrRjn83FZhzto3zVIG6CqjPiXmz354CeToht7ZMyUkT6+"
        "j1xZKBk6UbyYN+o3o3DnAkaVn5ScvU5iDVSt6v4T7S9pU8hEU3hE0kwJ2G6QlLXTkr1yvhVK9e6H3hqqjO7aVhfzmL4oii1c"
        "v34Vdz/lCv7owavnzjjOjYEAwF137PNHHj3G5SuXhZmx4XG2Gkz/XNuI5eXAJY/7yxS6zHvR58482RNSVU0MnyJIBFfSQsV+"
        "JhwcuxHJ5wwDDMoJrEtpWHdvzAU/o9Me28JrwsN1XQfDNHkp2XNSxuPYNl3fkSjaryFfH9IZy71yzj7UwRZ4GrnADJSy4Pq1"
        "q/i4uy7ggYdunkvjOPM5yIw/efSYnvOMO3H92vXtNWemo8p2m2rOAdLdF1NO0cba6Hlr1JhHlXzu7qztknVsY+Ijd+k03Jxu"
        "tMjYfACCEQhZ2bGUt9cG2OG3B5/1XEmnuLTWsdms3gs+NmhJb3vyCe/JPV/b2sM+dqfYTC0LJ9kqqiQJuXUKWodiUhn99WtX"
        "8QlPf+K5No5zZSAA8AcffJg+66XPwtHhkUw1122wrTWdcj5qEq72NfnHtA9Dph6OKeuA7Rup3qHo9YnH3PB1bVobsU1O2edH"
        "eVw/U8G1+U51Zrg8HpP4kN2TDAYpTQs7jalrWtvIOmih9a5hW/Xqu1HB62YF1MPJTK7m3o6t/wOjrtO5ucrgxrVreNEnPQP3"
        "f/Dhc20c585AAOAX33M//bWv/nycHJ/g+OhYd5Jj1p17a2zrfYu+9V7sNDE6NhxirV69Tjrr1mhdm3AyTypkxjQtHn67LzvF"
        "WSYL29wL2OvUm9oGsuWUZUaxJdolY2d3xzdN1akfZuz/kDlc9rptt4hv2spJdhPyttJ3sFOj+CpF1oJaG27cuIEvfs3L8Vv3"
        "feDcG8e5ykH+NNxxaZevHW6wt7eLZVl0EuMwCmN1bBcGphUDPusK7Emqzeby+VJNkmbGCEFsSNvYOw4X9vkSHUqorXoCzVqx"
        "Lhpu1Sq3tU2C9/qGhVQYOQ1sW5Svfxu9K1aTabWPUKrrMlGwh04+EmlaDiS5GTt5cXh4A3u7C771G78a3/G2H7otjOPcGwgA"
        "fPnnvZjf/rP3ohNg+i2RVUjPhvWc27QUK6jNfel+w2ryKuuPm+/QMDZKqs6y/ti3MfmUFJvBtWi1vPuMX1MG29o0WxpqPei1"
        "SQiUU0JZFp/bZcmDDLXGFqngdLHmF5ST5DxaEZ8nu7jq2Zg4wBUCRzdvAr3hns9+Id7xc++9bQzjtjEQw4ue/TT+7T98UEcH"
        "ZezuLDoQjlwa7h/K1IFXlCaWqnxWFqj7tts0DZv2zbTaCSi1A62CJ2OWLUxK3m8ydgLCZRulFFcIWxsvY6ySk46+5uraNG3u"
        "HQPjmosPvT9Ee9sH60WefPfO2Nld0Bk4unkE7hUv+9Tn4Bd//Q9uO8O47QzE8CnPfir/3v1/jESEnb09HOzvyVIYMJiH3H3u"
        "VffxOq376gRLYC3XgO0zLHn0p6teyqrcY+IhubbKDFGm0q/Y3dvzqYqjH0U3UdWxg71PY4hswPT8+NApjbZXHoBT15b7VJ2g"
        "WHT3Smsdx0dH6L3hJS/4BLz7vffftoZx2xqI4TWf9Xz++V+7D8gZKe9g72AXtYrHWDd1a8mmGwmPOH2WdZhCmFUjbjlMVRm5"
        "jxodo0XGVimtPZCyUjJUIrs3Mk81D6g2epdVc2XUr5ZgVO81RggZRVx1vrAPtNATcHKyQV1PwQy85PnPxG/8zh/d9oZx2xuI"
        "4fu+4038PT/w7/DAo4eonbEsO9jb28Pu3qISDcJmbTAFrS3L8coyRmJvNQwLyWxu1xhQTU63+h4PK0pq3JS2+lPYDcLqM9aA"
        "RdOcKiES4AMb5j2Nti4h+S4T4Pj4FKebU6mgJ8JnvOjZ+IJ7Pg1v/Sc/FoYRBvK/x9//G1/Ov/P7H8TP/MJv4uEbp55sl5Kx"
        "t783DhrG0GaaPI3d+mxTGh8zhqi17UWd5mHsV1FrQ1ky2LRck8Mxz2Br43wLVqJJXj+NDYV+f5bBdCcnp9IolglPe9JlvPzl"
        "z8Oz7roT3/3P/0OcgTCQ/zt869e/nh946GG8+zd+D/d/+BGcrmN4mrXpZq3G7+wUlJSRSvJ1Aa0ziuYn0P4MExS655mmvRsj"
        "ZfIVY8e2pz7Cx5luNqu3w0prLXvdRpsJseSEKxd28LmvfBFe/pLn45ve+i/jdx4Gcos9zVu+jH/rvX+IDz90E8ebFQ9fvYHj"
        "0xM8eu1YE/ehbbKCnYU6VncZgkfeSp5dy+Xfn1Sqb3s6CESsxUnCxQu7uLC3gwsHO8iJ8ISLB7j7KXfhh3/yV+J3GwbysYfv"
        "/86/xH/80FWcbipu3jxEJuD0eIPTXvHo1UMcn1ScrhW7uwt2lwVXLu1j2dnB/v4uuHVs1oqbN4+xWRtqqyglo+SMu+68jKc/"
        "7S58+/f++/idBQKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAI"
        "BAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQC"
        "gUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAI3Db4nxbwd13tgxVMAAAAAElFTkSuQmCC"
    ),
    "ee": (
        "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"
        "AAAASUVORK5CYII="
    ),
    "oh": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAYEElEQVR42u2dXaymV1XH/2vv533PfJzzTqfyJe1ooQXaBKcY"
        "aBtBPqpVQ1OCmCZAUESDiSYEo9GgiXBjvFFCVLwm8UovvFATuTCKCkRokOErVKCD0BYoBTrtzDlnzjnvs/deXqy19vMcCO2U"
        "flH9/5J2Zs7He07eZ6+9vtcCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQ"
        "QgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEII"
        "IYQQQgghhBBCCCGEEEIIIWTO29/8M3rqWSs9cWyhl20udZmTLnLSIYkuhqRDtj+Xi6RHj2Y9sbXQl994jf7pH/2K8t17ahC+"
        "BU8cv/qGV+nHz/w37rnvHMaqaAosN5ZIOdsbL8AwJLSqkCRotSHlhNYUqopaq/03FkgWLLLgZ254Pj74kbv43CggT19+4eXX"
        "6r98/EuorWG5sYFhMWDIGdoUkoCmCjRTCiEYIoD9D/519vecM1pTpCGjjAUH+/topeCGn7gCd3z263x+FJCnD6956Qv0o586"
        "CxXB1tYmWhMoFNoack4oY4EKAAVSToAqWlPknCAiqLVCFchDgjZFqRXLxQKtKpCA1poJGoCDMmLc28d11zwHd971TT7HJ4jE"
        "t+Cx85Y3vFJzSvrhM1/G1moLm1ubUAhaq2i1QlVRSgWSAApgpkFSEqgqVBtEBCkJarHvGYYBzbUOVKHNzK7WKpZDxvHNTdx1"
        "7zmIQG979Uvop1CD/PDx3Gds6TfP7eD45iYEAtWK2hqSZHuDXTOEOQVViNlTUG3IQ0arDQqg1Yo8ZGgDUkoQMXOslgoRey1J"
        "glpa/3dKApWE/Yu7WB1Z4tzOPp8pNchTzzt//TZd5qTnLhxgc2sFCFDKaDc+XCv4jS/imqIpUpqccEDMfIJAAOQ8uOAoWm12"
        "f6kLQk5orZkjn8RepzbU2gAojm5uYXcNDEn0bbe/ktqEGuQp9DVueIF+5L/OYnNrCzJkjOsRKQu0NrTW/PYXtNamN9q1hgmI"
        "fU3TZoKiwGIxuKkFqKppjzad85RmPko2YTPnPaEWe73FcsBYGvZ2tnHDi6/EHZ/7Gp8vBeTJ5QVXXa7/c/eD2DqxQqnV/AmZ"
        "fIuUxW916e+wusbIHsnKg5lfqs18E5iAiAjGsSIldE0j4lGuZN59aJ8kYtEw2NeaL2NmnKSE3QsXcP3Vz8KZu+7nM6aJ9eRw"
        "6tkr/fI9D2LThaPWim4gKSy/IambRACQRPq/VRtSFhR33E2zmHYQcc2STdOYkNnnJYn9HBF49BcpJyyXCywWQ9dQqgoBoK1i"
        "tTqBT5+9Hze/5BTNLQrIE89Vp07q1761jdWJFUoplqsAoK2h1momVmtm9uTkvsKkBcwvsYNvvrr5J9mFqpRiouamWc7Jv8Z8"
        "DHHtgW66mbnmcoZaqoeNG6BAqQWrrS38x2fuxVtuOU0hoYA8cbz6phfq3V97CKvVqptEIuEr2M0NMdNGIs+RZBIOscOcc0LO"
        "2c0vRfKcBqAYhuwaQMwU89cwzWNmFNxHWSwGQBXr9YhSxh4Nk5QORc0aFKvVFv7mXz+L33vzqygk9EEef97927frn/zl3+HY"
        "1qY7xg3it3cZRyiAPGQkSRA3fWqtdtNX80kiEiXJolWllB6FkmSawhKHOORfiAgg8IiWOfOSErQ1DIsB4zgipWRaKUmPgIX2"
        "idcdUsLuhW2sm/J5U0AeXxY56bBcIg0LaKtm1rigKBTZzZw85J4hr7X2kG5Eo0KTtFqhUCQx5S3ZDrw57dpfP+XcNVTtPot9"
        "vNWGFKFfbf1n1VIhKc18EqvtEghEFVL28eDFkc+cJtbj5HdccZlWVeRhsIPd1CNSU3g26qcimYemgEesWm2ISK9CXbNY8m8s"
        "xQ6z50NqbWjuQ9ifdRbatdosVaCW4o67CYQ2zELDnkNpzTP4QBktcYmUcHHdcOOLr6KpRQF57LznHa/Xe77xEI4f34S2hpQi"
        "l6HdJ0gpIclUVFiraZXwF1JOgEehst/sKWckEQyLAaUWiKSpzMS1jAfFUGpk0D0aFglC1yaSkvseIWCtR8dSzp61t0DCWEYc"
        "W23hzJ13471/8MsUEppYj43Lji10vwiGjaVrh3keAhg8l2ERJz+k/nUWujWTS7LlMcyPV/9TUFvtvgm8QDFCu+7Du8kmPXol"
        "7tJHLmRws66pomn4Rv719uJuvgnyYMK03h+xWgL3n9/js6cG+cH4zTfdrOcvjtg4erRHqMxHmOqfJDSHR5+g8OiTTj6JO9iR"
        "GYdYhKppQxLTQHnI5id4pMu0D7omCX8iiSCljJwH5JwwLLyQ0R1yez3pvo540CDnPGX2S8ORo0fwne19/NW730otQg3yg3F0"
        "mRV52SNGEdJt7gwnibscXVDC/o+MeTjJtXgIVmB1V/6Ot2ofz9k0R3fQI5fh7k1rOFQOn3PGYrlAqxXjevRol0e3/PX9V/Lo"
        "VurZfBPIhP29PZzc3MA3H9jh86cGeXS87fWv0P2xYrFcQNWjS5g0R2iJlJKVf8gUWrWMee7NTqrqwiFTmNdNteXG0m92EzjT"
        "LNr9GUkW6k1ZejRMVTGOI9YHa5RSzPme/W6Qw1deyqmHluP3a02xcfQI7j+3iz/73dupRahBHh0njy91rwjSMPTwa2tWC5Uk"
        "TTVVHjVq2ro5E7f8sBgsd+ECUqv2W11EZiFd6be8CaP0QywJKGPxBGP2aJb7OxCUVg/Vd80DCCHAcKEy88/yLk0VeZGwt7eP"
        "q599El+899s8AxSQR6FSBbq1OoGxjBAv/iul9kSeQLqGaH7IxQ92JADjQIuIHcw2HX4ReDbe8xQpTLSo1cqeGOy6C+pJQoVi"
        "MQwoHgxozfpHamne524KoRTL1wi8pTdaeF1wmkfNxv2L2DuoPAM0sS6NX7z5eg+gtqlfo02ZaYsmmW1vGXX7OwSev2h90EL2"
        "YsXipehR5l5rm0W6JrMnhKiMBSl7+FbMtKql9sjWOBZ3xuF5Dtcq/u8QhtYaSq3uN4ln6Se506Y4GBve9/tvopn1MAx8CyY+"
        "duYubGwcgVVETRW5rTXkJFA/1EDrTvjgtVWKqQ9EVbsTHzVWJnRmjkW0yW76jJzhwmiHvYyla6HoHckpoTTTHOqqf272JRHr"
        "ZEyWOW8yZedrnQcOTMhztt/j3z/xeT54mliXRhbR45ubWJeChfsRKVmlrUjyvIMCEpEha5XV2oCE3uthN745znGAWzOhGtwk"
        "UkyZ7+yjfiJ6ZfVVtXcOhkkV9VbVxwOhN1bBy1cwab0hoZV6yPFPPkGloSGnhP39A5w8NuBb59mmSxPrEfit239KFUAaPIyr"
        "rWfJLSOuqK1CRVBLQfPbXZuZWDlN+YaUpwamCNfG5+A3f4R+85B7hXDOybWWYrFYdNPMolpp9lqzfEdK3UfBPFrmGjAy7Zh1"
        "KIYDv1wu8NDFNR8+BeSR+dwX7wUAG4iQrIYpDHutUURogxUkmzmkfQCDV+hmb3/1pinrD2mH/YLS3LyxIsdxPbp74b0fan5N"
        "qcX+LDWcBvd93NF2Aa0eLUsJHqVqXjFcPbrm/kdrfd5W89cREZRKF4QCcgnc951t5GHoznDOsxopzyGEJoiwqn0cfURP9egR"
        "4OZWzhgWw3fd+GYiRaNVZMyTh2n7kIfwFzw0HF83b6gSkW5mhVbo5mI207DW2qNucijTbvVhIsCf/+GbKSUUkIfnW+d2ewWu"
        "etl68whV6tMPp2akmFIS5SelFP/+6mXn05TElN0vsQ/0gQuDZ9K1TTd++CO1uhB4mHn+c0JrhQOfUp6y5q4tQsNFuYv4/K3w"
        "X6b6MMGHPvpJHgBGsR6eg1KxPLLhxYHT+M+53zAl5MLen93uQL+tVee2voVuBeZvNFW00pAgaJHAS6aBomMwZ+297XA/xSQi"
        "hsm1XgJfSjGfxhOGkTw04fHEoxdPNo38y6QBIcC5nQMeAGqQh6e2qU5KZw1OtUwTSprXZZmvUg8VFIYpFCZWKdYY1Z15aB+q"
        "EOXsrTXrC5lV3Y7j2KeTRONVxIyjjz2iYVHPpU1RxmI9JG5ahY8TQ+aqh44lNIrncxSCbQoINcgjofASdbRDmiNqreLwi1ge"
        "JKXc8xZNp/Bsr8qNF43Ik+dLwnQKMy0an6xsZXTfIvW23u53uGD0ymAAi8WiR6xCgJIIZBhg3YcJKtZ8NVUae8FlFpvgKIKi"
        "jQeAGuT78/732P4NiT5yn0YSzCcaRsY7zKhSyiyX0Q5pGdNEpjLGden1V2Ws3kg49ZCkJID7HvFzYhqK3fgWIYvq36gA/u4h"
        "Ej2zX9UreCN3Ul04vY+lTpNTxpEC8v1ggiiiPiJ64uRl/cDbbT0l7axZqvl4UO0lHBEu7QWKM6GSnugrgALL5RIiVi4yb8CK"
        "mq2xFI9KWVIyDnSfxNharzCGa5W5sIYwRr+JRbOyDdF27RKaJDTY3v4err7yctx5lgPmqEEu4Y2Ihqf5Qe8fd0c8chHmZ/ig"
        "ht51OEWXtOkhX0VnAhF+ivkfVg5iWfdJEzRtODg46P6L5U/M0ddeADm1/4qkXn4SvSUm8NOkxqgFg2qfC9wiwkYoIN+PoxvJ"
        "b/I8y3qbiZRzOtQBGFGtYbDFODFhpNZqo3/SVLmbs7feAlgfrG1Uj2e4o6fE+kzE+9ml95BE9W0tBarNHfhJkOb1VT16plNJ"
        "fUrJ/BTXLn2MkH99HvznZx4DCsgjsMwZ47r04W9zcylyB0lSz4eEfR9Vvuqh1+pmT/gQpVh0KQ7tcrmc/INYqhMmGbziVtFb"
        "bK3pybQVPA/T2rQ2IVoIw0eK3631XpbqX6vdpo7JKVE9vHn8GA8ABeQRNMixDWsGcVMqppNEVCpa0y0ClLov0Ye++cCEeZg1"
        "XLyIeA2eqY95uxEYqK32gdcRvm0+WdE0VPI8ydQZCDftogU4fCYTntTzJRE6jukq5vi7qIiF2m78yet4AOikPzxXnDyqD+yp"
        "3/CtTxkJWz6mHsbB7o56TkCbTVSMNzZKQnw9wjAMfQBEnvWepyze8JR6xlvi8LovlLwgsZYG8XE/IXDqE01iH0kEC6IsvnlV"
        "cO9UFOmT4hMEO7u7qLXxHFCDPDxXn3oGSrGDPNVeTXN1+3yryGr7DW4HUbsvcXgOr729wzD0zsHkXz99f+qaato8pX2OVowm"
        "jahT74H3KNd000nvIuzf2yc4yqF6Lrh2VACbqw0+fArII/Pam1/mE9KtlsmamaLOdjJPqlfXdlvfb/3YKxjlH+YITzNye60U"
        "pu+x6mD1m943R/ldPo4R4YqyevNzervvbEq81lnmvzbvRUnWC+85leY5GUEUS1oO5ciw5MOniXVpDFl0c3NlIhFrBlRRasNy"
        "sTCH3Q/sMAyzkhSfhOhDFiI6VcbSfYGIMiWJhiavnYIt0gmnPYRwPt19GkgnvURkmnSCPg7ou6t/D4eo1cvwPeM+JIwHa1x3"
        "9Y/ik5/7Cs8BNcgl+CHPPIGdnV1zqkvzknP4obby9Ihu2eGeesnj45EFD+GxvR/ofeDViyDLWNwsm2q/WmmoYzRPeT9JncaJ"
        "6nxtdMzqrbUnM2PAdSml+yHhdySZ5m8BgKhgvX+AW3/uJj54CsilcdstN/rQaJ95pdLXm0WuYH6TR85B+/A28T0hEaHCoXqq"
        "iCBF/4g51uhOf8rTsk5xH6GUimHIfdRpON/apjnAPTHpEbiUzM+pXryIbiKi+0qtVZw8eQx//L6/pfagifUobgwR3VqtgDCZ"
        "cjp0g0fZexxEmzqiGIaE6prDFnIC4zj2Xefhz0TOwtp0Ux/RE9GsuRmlaq23EQiI79PZFiuLhKVZYaS4r5K6aQeV3qobwrd/"
        "8SJOv/AKfPLOe3gGqEEuneue9xzs7+/3KBNEexFiaAJEpCiSer5jMM36xKMkPQ8+r8rT2b2/xJ3lYci9p3zaJ2Lh4eVy0TsD"
        "o/AxzUadhkkXScvUl/Hk75nkGEPkUhKb0KINv3TrK/jAqUF+gDdFoKvVCdRaevQHMjVMSZ8zpV0TaNMuQNO862nVWi0xwif7"
        "5xRltP526fO3pikmw2JwIao9GRldiSnbHF5rwGo2Jsid/tBwyUO61XcpdsHzsULLoeH8zprPnxrk0XPquSdx8eJuL/6DTD0X"
        "fRK7O8o55R6VsjXOpec31us1opzD9hAO8CLbWb969s8lj5xNJSjh4Oc8TUmMsG3/PcTqrTDbWdIdIKCPRJ1qvhIODvbx1jfe"
        "wgdNDfLYtMjx41s28K3prORkaqe1TVOYTTfRPg0xdnNEo6KN9ykY8tDnbG0cWaCUatlu1zJzp39ebt8LClUP9XXELyv+s6v1"
        "C/eK5D5pJdnfD/YPsNpM+PYDnIVFDfIYOP2iK7B3cafnDiLpFyN7oj/EChGjtkqsU09hGew+7A09wx3mkmmJ2svra6nTXCzP"
        "q5j20r7pNug7SfzvsQy0jyTFNHnRuhknf0TrSOGggDx2PvOFr0tOgvVotz30e/MRvZx8NmChjJ6XaM26E6P3Y7QFm+G/NPct"
        "tMVyzuRVweqVvL4stE4+RCzpbDFZEfaxNNti1TVaTkCzMLK4xtvd2cVLT/8YHy4F5PHhHW99HdYHe57ci9qqaZoJMJWgQLX3"
        "kzS1Wb4pC6IOKpxot5L6DpAw12S2+EZmFbrdpwD6ONRhMfRekD5xMfkqBt+8myzk1X0Y1YZFFnzszN3UHhSQx4f3feAf5Ibr"
        "n4f9vd1pIJxOdU2xRbYLjEe1oOj9IDXm9/puj9h53lo9pAmiu28++bBn5XtHoPseqv17o0xFdVbjhennm5eZcHF7F+98++v5"
        "UOmkP/78yMnj+tCFPay2tqzdNZbnRMbbx/lEPVSYX7FMZxwLckporXYNYULRMCysmSlqtURhN7+i95+32RKcEBRryqqTkx5N"
        "U1n66gZow3JjiYcePI9Xv+wafOiOL/F5U0CeGI4us1YFtlYrrNdj90VSztAY/9MX1aALR9zsfQqih3AlJVuN4xtzYx1C350+"
        "m4jYmg1hmBccds0SEyCB3uEY69uHnLG9vYNrTp3EF77yHT5rCsgTy5FF0qLA5taWm0TaI1VWNtL6TsB+iStQa+lbqOLAi9/8"
        "vXEq1itg2hDVZvVWU5l86j0l6r5RCFRPFGrDsFhg58I2nnH5Udx3P5d10gd5EtgfmxxZJGxfuGCOr9/U2uCzdpObUtora5tP"
        "R0w+Eb6WagMgZv6IjRxtWI+2tXZYDIjG3ySpl7/Pp6FE7VUrrfexo+8CGXDh/DaeefkxCgcF5MllZ6/Ilc9Z4fxDD7njnZBk"
        "2gXY51ZB+4STyGMkz2aXWlDG0Vtlp13mqg212kT2mJQSewzD77GcC6Y11D4NRcR2m0gSbF+4gOdfeQLfuH+bwkEBefK5++vn"
        "5TU3XIu93YtYr8feXRjzrYBpgqJZUnKorwPdrJK+51BVMbgfM65tzXN8fzjprenUp9JiumKYawkHB2vsnt/Ga1/1Etz11XMU"
        "DvogTy3v+o036F/89T+iVMWJy1Zu4UgfPxrNSrFVVn1wG6DfMzExBr7FSJ4YzzPvgY96sPA3TLtYbmZ3ZweXnzyObz9Ak4oC"
        "8kPG6Rc+Vz9/9j4MOWNza7MLSUxSjB72EJjapm5F9TE9rU3r1WIYw2I5oFWvvRqSt/DOhma3ht3di4AqXnfLy/D3//wJPk8K"
        "yA8vN734x/XMnfeiAbjssq1p6Y2PA6qlHeouNH/Fixw9yhXjg6IpqveaW1lWn8m7s7MLqOKl11+NOz51ls+RAvL04edvvFY/"
        "fOZLWNeG45vHIQoMi2wlKyI9ymV5FPHejoh+xVoEezTDMKAcjFAo1mPB+uAAQxL87CtP44P/9mk+PwrI05dfu+0m/c9Pn8VX"
        "73sQRbVvlEp5wJEjGz3K1Z/EoZUI6rN8bXrjxiLh2Sc3cdutP433f+Cf+NzI/y3e+6436a03XasvP/08fdFVz9SNRdIk0JTs"
        "v5xFcxJdDEmHIenm0UFfdOpyfc/vvJELNgkhhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQggh"
        "hBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQ"
        "Qv7f8r/HxsqveGsRsAAAAABJRU5ErkJggg=="
    ),
    "woo": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAALH0lEQVR42u3ceWxc13XH8d+9b+bNDDdxEylZEiVrcSRZqiVL"
        "diRblZfYqKI2DhwEaKE6TtPARYG0RYACbgLU/sNugcI1AhdFgcJtUhQJgjZp7AaFI9SykUJxLCuprM3aaNKyKJGSuJOzkvPm"
        "3f7x3iwM+k8ByyDd7wcgn0TO/HPeHJ5z7r0zEgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPD/nSEEH5/77lonP+lpvlyR"
        "n/RkPF8/PX5BLz37hL7+3PcWPNY5J2uM9ty1WsXCnFIZX5J0/MwwgfwYWUJw6/3u47+uF575sm5fv0YHP7vvf/s7ZSV5v/JV"
        "T5b4mko2a/+uOwgoCfLJ8bUvH5QkXRkaq7/gjaf/+u9BJTz7uc3rVzz78vffPOP7yQuZTOp8OpM676eTFx/dv/34jq2r/jjj"
        "256ZQqBlLR2SpCCsaM+O9QT2Y+IRgltn07pObVi3SkffOakN61apuSmtv/ibH2rDmp57k1bfGRmb/fPRqeJDk7OlnmQy0eV5"
        "Xpe1tsuzXuf7QxOr8oX5z07n5g55np178J47T88UCpViaV75XEFtvtVkbo4gkyBL077dG5UvFJXLF5UvFLW8q1N/+KXfTJw4"
        "fenrw6NTP7wxVdwwVw5lJFkjmXgcdE4yxijpGVUqTqVy2BZU3MEPhm7ct3bN8p+NjU5NF7J53bG2V3dtXKnzV8YINgmy9PTd"
        "1innnFJ+Us45lStJ/eDHb/7T6FT+6ZvTRSU9U5swjLFRlsjIejZOlfr3SuhUKFfWzxVLh9Je+NptyzvGWptSyhZKmpzOqVQO"
        "CTgzyNLxJ08+IucCPXDPVjkXaHK6ov7BwRf7h8afHJ/OK+VZWWtljJGMkZOLJ/Ho6lzj+G7keVZW0uUb2eWl+fA/7ry9d00Y"
        "hvrFuSHt3L5RL37jCYJOBVkannhsrzJpX2k/pUzal5NV3+qurw6NTP5Vca4iI8nEyeGcZE2cKHKSMTIyceJUf2TkXCgjI2uM"
        "5oKwozQ3393R3fXq7X0r9NDe7ZKk1986Q/CpIEvDbK5Yu27s6+k9cWbgzyZmS1HTZOJK0cC5sNZONf67dpOsrT2jOB/q9OWJ"
        "L5Ur4VcGr1xXLptVLpsl6FSQpWFlZ5s+HB6TKk4fDo9pNld6euj69Bfm5iu1F3utetjo75OJ26yokhg556J5xNYrSfQt+l0p"
        "CLWyq2XHM1/7wsuTM/lyMpnQwfu36z/ffo8bQIIsbju39GlTX68qodOmvt72M5eu/sPV8VxbwkatU/VLZuEhBmutbPwzJydr"
        "ouSxxigMw1rbZYyRNVZBeb69kC8MDl+fOLWmt1OSSBBarMXt2Hf/VOcHrun3Pr9b5weu6cS5D3Zcn8iu9r16paj1TvEkXk0Y"
        "55xC5yQTt1TOyTkpDKNVLmNsNJPIKOlZjUzkNTA0+uj3D/9SKT+plJ/kBtwCCULw0bo8MiPnouvaFW0HK06ynie5agtV7Zic"
        "jDW1ZDEmyhnnXLwnErVUxtQnltrAbjxVnFQOKg9+4/cPtPzsxIUckafFWvzD+WxWhw7s0I+OvKtDB3b4/VduvjgyWexJWBu1"
        "R/EMYYyJB2/XUFmMXBinQtxKxb+Ml37dgmqT8IzGp7OFNSs7/n5ytlCcKwd6/8ooN4EKsni9dWJQJ88OKD9vdPLsQGKioGW+"
        "50XjhqsnR1Q0XG1Ij6pGlZMLJdl4cHeKB/jocdVhfj4ItWFlW2/SMw93L2v6N6JPgix67R2tam9NK5ktqb01bW4MjLpo1jaK"
        "X+nxXkf8YndO1Z30xuSp74O4hrSJfx5Gj/MTVqNTOf383Ut+JXQEnwRZ/H7r4V1qbUkrmyuptSXdN5M/3j44Mi0/odoM4sJQ"
        "xtoFg3h9QcspDJ08L1rulUw9MapLvfEcUm3R5ivOhZw0IUGWgr/+9uEF83prc/NUyk+0uoa/8CZe7q2N3nEuRPNJ9M6QanWx"
        "1sqFYVxJTG1orz7RRjOJNYYKQoIsAU2e1VwlVCq6GsmZ6pxtZBfsn7vQ1QdzG73wo/0ORUu6xtX3P2rDfb3rCsJQnc1N2nrH"
        "2ulyOZAkDV6d5CaQIIu4xXro17RpbY/evzKqTWt7dOSdizp9eVKpRHUVy9V2yRWvSEWtVhhXh6iy1OaRX3lDdDSrRI9NeEZj"
        "U7kblz64ejRfLBP8W4SNwo/Qfbs+pb/89uvVa9FLeEeSnqm1TGFYPy4SrU01tF61hDANA3x9ubc62Ff/PzcXaMvG2+YfO3B/"
        "uGfXFu3ZtYUbQIIsbsXSvB66d3Pt2tLS/HrGj7aaqku0UVLEVcTVZnOFLmw4c1VNmiiNQhdGz3RhlDjGKO1bNaeSr7/wd6/k"
        "9+/drv3xqV7QYi1ambSvi/3X9NQX9+ti/xFdn8692t7WfD502lo9ul6fzKXGxSk1rGgZGx17D+Nj7qZxO11OQRBqRXezbl/T"
        "+5PuzjaNjk8TfCrI0qgg3V1tOvT0y+rualNvZ8u8n9TLQVCpH1KMq4mqA3ecFNZ68endeosVTySyxtY2FavHez1Ph5/66sEf"
        "dy9fpptjE+rpbucG3AIcNfkIvXn8otJNCe3etlEjE5PasqFPd27sG3Bh+JXR6XxzwrPRMG7rG4LRIpWpvplw4RumZGrvBam2"
        "W05SV3tGHa3+k28dO391RU+nrGflnHT0GKd5qSCL3LJ0Rm+8fVZDw1N64+2z+tefHJtIpRJ/lPRcQ2u1MAkUzyd2wWnf+HHV"
        "OhK6eLnYaf3q5T86dX7knQf27lRXe7uCoKLnv/UvBJ8KsvilmxK6du6YXvzmU3rpWy/owP7dOn3xg3MTEzPjMuagsVaeZ2UU"
        "Lf26cOEpX2usGg7zqqG0KHROGV8/z49cf/zTe7YHST+pfKGkRNLql6cGCP4twEeP3gLOOTWlEircOKqmFftljdPKni6Vg/Jz"
        "obxnZotB7dhJdd5wDWdOrDEKGw4wRlsnTj0dmbPloPiIZxKjqWRCzkm//fijev4lqgcVZAnJXj2lR+6/W6/8+xvaf+92/eLM"
        "JW3b1KeTF6/+dN/dmwvXb47vlNRUqeZE9dRJw0FG61mFYbT825T21NvZ8o93b1n3xRtjs9Ol+bJK5UAHHt6jfKGkE2cGCToV"
        "ZGnJ+J6scVp7W49GJ6e0a9smNWfSms4WNTU9tbJQLH8zqIR/MJEtpYKwPrBX260wdOpsS6sp47/jWfPse/3DRx64d7PGJ2cl"
        "SUEQfaripQ8nCDYVZOkJKk4p39OWDav14ci4VvV06rWjp9W1LC0/kchduHLz8D2fWvXKlk2rRlOe3PD4TH9YqQzOl4PB3s7m"
        "obu3rfvnTDr53O88/uDzl6/e7J+ZyWl8alaeZzSbz2uuHOjKyAyBpoIsbb9x/zY1Z9LKF0uNFUTNvq/ujla9dbJfk4VALc1N"
        "8uL3rgdBoGw2r4fvu1P7Pr1NUzM5edbq5Jn3axXk3MAIwSVBPjl2b12jhLUKwlAJa/Vu/zV9ZucGtbS3amh4XKcGrtc+ZnR1"
        "zzJt3rhauXxRjx3Yq1cPv633LlyRJM3m+cBqEuQTbN/2deruaNX4VPb/VEEk6W+/8xoBBAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "+IT5H4AHh28fJpBhAAAAAElFTkSuQmCC"
    ),
    "bite": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAARqklEQVR42u3daYxlR3UH8P+pqvv2XjybZ/NuwQyxrRm8Jl7w"
        "oEmCkow9JE5Cgu3YbAEJKcYhBBTIB4wSy0ERkaIoH+KAg+wsFsGGsZWAiZfIJjYC4WUygBm8zBi7PbSn3f26+717b9XJh6p7"
        "3xsCJCgGz6T/P2k07p7u1/3eq3OrzqlT1wARERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "ERERERERERERER1jhC/B0eHGP3yrLiwOkZcFBstDGGvgfYA1grIoIc7B+4Buu4PjJifwRx//O753DJD/H95+xaW6rCUe+co3"
        "8PzMPLwPaDYdlgcFVADvA0QEUI3viAJiBFDAWoMQAiBA8AoAUChEBAaCVsNBFZicbGHN9AR+49d34sN/cgvfVwbI0el979qt"
        "9z/wNTz97AwazmJ+UGA5L2DEQIxBq92CBoVzDgIga1oEr3DOIngPaw18GeAyi+ADrLMIZUDWylDkJUQEZVlCAWhQ2MzAlx7G"
        "CBaXBijzAlCFEUEjM3jNaRtx+smb8Q97HuR7zQB5dVy+81z98lf34ntzSwAERgTNVhONrAHXcCjLEnleIi9yBO8RQjVjAMYA"
        "ISiMMekNibNDt9eGMRaDpWWU3kMEKMs4w6gCJs0wxkr62MAYQbPZgKrAujjzLPWXUZQxsDIxOHv7adi183x88MZP871ngPzk"
        "vPfqnfqZux7CocMDeFV0uk2IWBgxGCwPACjK0kNEEFQhEExNTWHzCRuhQTE52UO33cPPnPVaiDp0em2EMsAYgXUWN//tLSjy"
        "Id53/e+j9Dnm5+YBUQwHBRYX++j3FzD74iyWB4uYn5/HwsISBvkAs7OHUZahfmeNCNrtJpyzKEuPvMhRFh6TnQbeeP7r8M//"
        "9nWOAQbIK+fKXRfp7Xc/iCIonDXo9HooS4/BYADvPTJj0et2cfbPbsfGdZuw7bztmO4dh80bN+PMM7Zi9brjEXwJ67K4pHJZ"
        "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"
    ),
    "tongue": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAi+ElEQVR42u2debDtWVXfP3v4/c45d3j33dfzCHTTQCuNCNjM"
        "AgkgKhjBASU0iKIlVQEDCmoSSpNgjKTikEgojTEOKbTUSCQosRVNK1PLqMwzPY+v+7137z3D77f3Xvlj7d/v3IdVaFnR0vfW"
        "p+rWnc4595x7fmuvvdb6rrXBMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzD"
        "MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzD"
        "MAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMAzDMIwvhbN/wd8u/+413yLv//PPMpu1FCfcduf9nNibc+99+3RdousSUgqr"
        "LtU3wxGiw3vPdNpy5OgRdo9uQUo8+PLz+fW3vMfeMzOQf5j85GtfIG/+vfdw2133cc/xfRZdJmX5S7fz3hGCwzmH1O9dfTdE"
        "BOc8IkIphdQXvZOs3y0HSP3+ggu2uPZRV/OIqx7I637mN+39NAP5+8NLvvmJ8tbrP8DefEURIRVwztFOW8jCZBrJuRBioOSi"
        "BlAEnNTPjiKCA3LOeO8QEbz3iIDzTj2KD4DgQkAy+Ojo+0yWwmq+1McCnIPgHRecu8OjH3Elb/mD99v7awbyd8t1z3mc3Pjh"
        "T/GFW05QgBgjzaQBAe89pRScc+ScidFTiuAclAI+OHLKhBAoRcBVw3BqGDiHwxGCH38vRXDeI0Xw9efeq6fxwasBej96oq5L"
        "iBS6ZQcizNrA0570CN769g/ae238bcURz5crLz1HonPicbK5PZWdo9tyzrm7snP0iOzsbMt0OpHpbCIxBnHeCSDOIc4jTRMk"
        "BC+xCeIc4p3Tz96Jc05/F4M0jdefeQSn9+eLP+rjDp+9dzLbmEhsvLSTKJvbG3LBRefIeecdlfPO35XtIxvSRC/eIZdecERe"
        "9wPPF3tHzYP8f+E/vvbF8h/e+Fvcde8Bznl2drdxeHLJ7O8fUHJBRAgh4H0ghMDmxgYPuOIBbG1tsbWxyUMf9jCe/eyvpYkN"
        "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"
        "AElFTkSuQmCC"
    ),
    "uh": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAASXElEQVR42u3da5BcR3UH8P/pe++8dmdXb62ethDSWrYljL1Y"
        "GMcIGQVXME6AUogDJPAhwRDsEIoqqpLgJDySFMSGlE3x+JKicBEeZaASx5CKjXFix7YkwLJkPSz5tZZkrWRpHzOzM7M7033y"
        "ofveO+ukEkNWNlT9f1XSSLs7q63pe6b7nD59BRARERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "ERERERG9VMKX4OX1B797NQAgigz2H3oWex59EkkSY7bjAFE/KCKAAJ2Owwfe+2ZcPHw+2jOzKBULaM/M4uOfuYMv5Msk5ktw"
        "7r3vHW8EACRxjGeeGcNUs4m3bHstkiiCOgDqoM75tysBoAoY/9xiMcHmTefhli99F8a2YZ3DdduGYZ3DDx44yheXAfKr7fKL"
        "1+LQ0WcxtHQhLty4Ln3F5exEXTvWQgwAMeFx7gwCOMzMdLD/0Ci2vWGLLFu6SN/34Z2Q+Cps3bIKALB1yyrs2neCLzQD5FfL"
        "my7b4GeAQoyhoaUAIGfGa/rMsTFMNZv6lm2v3ZhE0ZvUYfb/mEGSzZvO+94tX/ruWWPb+M6dP5C3bduoqsCFG4bQmlVcuGEI"
        "B4+O8UVngPzyW7mkCgBYtXYItakGTp0+i4n6tAwtXagXbly3/tjJF274yb5np8ql4rXjE/UromIBEgkKMQCF/00EqgonXTw9"
        "egq3fuX7b923/+CBK0cu+OdjJyd2/+fe49i6ZRWGFg+i3Z7B4SfHsG1kGPsPj2K80eYgMEn/5XTlyDAEwMb1q/H0k89lM8ip"
        "s1NYuqh68/Nn6u9U1UuePjGOdtsiKsRaLsUCpzCRCbOIHxJVv9xq1KcViGTl8j6UExwdrJbv/tmhUx/dcsFKdK2TOI50fKqO"
        "T/7J9Xhg9yFcdfkmPLD7EL525wMcEAbIL9GS6nV+SfXEM2MQANuvugS1qQbQ7Q6PPn/mjvFaY+SFWlc6DtpXikUAWOeyQVAF"
        "jPEzh6pCjMA5RRxFUACdrlPnnCzojxAJ9i5fuuCjpaRwfzFJsP/wKP70pnfjE3/7D+iEiUg5JPMi4kvw/7fh/EWoN1uoVPqx"
        "aEEVWy/bhFZ7BuVy8at79j1125nJ5rr6LKSQxIiNiDqFQqFOIeqTcpFs8vDSj2uYTVQljgxaHUXX6tD4ZP23N6xdccV0q/3N"
        "izatwSUXrcbAYAUf/cPrMFgtwXRmcHJ8moPDGeSVtWb5AABg/drl4RUVfPmWj8VXv/1jH+qq3tbqpksmBwmZtwsJuYSKlSpg"
        "IoE6hXMOYgwEAmME1jr/XJGQoqgPHGMQowvtus995e9u+MR7bry9owCKYfYYuXgVCrHBj/ce4yAxQF45QwCq5y8CACys9mPt"
        "0OKBcnXg4Pd+8MgqSQoQAWy360u38EsoP1NICI6e0q6PL79Esj4vcc5BBOFRICHAJI4gEqFRm8bbrr64nUQyfPr0xHOtZhvl"
        "SgmtZhuLBoq496fPcZC4xHrlLFw+gG7XIUki7Dsyhr5KctdDe564BEnRX+w9AdC7hPJ/N2E2EDjnYIzAGAOkM0o6W/Q8UXqL"
        "XQBK5SIOPvl8nEh3R+TMHYWCmS1EPp8xIti5/RI89PgoB+oXZPgS/GJu/shO3PyRneicqqFzqoZWp4PzVgzcfmT09K93TKLO"
        "2ZA7AFEUpWlFtqRSVV+1gt8fVKchQfczhHM+AvzsEZ7nNHwPkwWcOodSpazHx+qbrXa+Xkr6EEWCSrEfUSSIIw4xZ5CX2fmr"
        "FmD1iiX4yWNPYMnqxVJa2IfWzOwNU/X2pzqawADiL34/I0i2ASjZ0irPxX3NyRjJV7yShVOYbTTkMQoTRUjiOE3cfQAakbaF"
        "JtLdtH7VomTZosH7zl+5UIpJgmIhwUOPj+Lyi8/HidOTHDwGyLn3x++7BuVKBeetXo5mawbtdnv7dGv22+MNK0ks2cwAKKLI"
        "ZBWpsNqas2wSSDazZLlJCKI0jtLnigAGvvzrnPXLMfHLsyQ2MlbrYNlA8Y2bXrViVIC9A/1lCIBCuYQPvPsafP+ePRw8Bsi5"
        "N3riFFavWIL+SgmbN61b02zN3nf/I4crxVIBan3e4KwFTLjY0/2NnkTCiAkfd3mlRPzXz6mhiF9+paFjjIFTN6fGkgZQHBvU"
        "p2dQqzWOP3X8hR9+60f7MeuABx99EidOvsAZhFWsc++D12/HsiW+tPuT/U/j+dOTm6fqrX3Hz0yjEIcyrnUAFCIGURT5QHAu"
        "zAI+WIwx4WPO5xTZJiHCnkieO1hrQ6z48BEjcOogYTbJZiPjh7QzM6uDJey46IJ1971/55vxtTt/hPfvfDNqtTpu/PTXOYhM"
        "0s+dUjFBta+Eal8J73rr6/unavV/efbkJAqx8e/0YSccoSPXOZc26WaJdvpxhcK6dGbxs4BT5/uwXlTe9Um8/7NzCmi+ZHPO"
        "P8d2nV/OGSPLli2+a8eVr1m1a+8R7LjyNdi19wj6ykUOIJdY5877f+tKQAQTU9NotWdxZqJ++/OnJq8+PdlGEuU9VMYYv4Tq"
        "2eswJiyeTMgvevIMDcsnP+tIVvHSfOXVUyIOpWBJZ6rwTmcMoAqnDiaOUC7EycZ1KzYP9JfvWFgtizEGcWSwfWQD7nn4IAeT"
        "M8j8i+MIpWKCUjHB+rVDC//1/r0b9x4eQ7kY5aVb1bn7FmlnbrpDnkZE7yo3zBxp+VZDdKS5h9M8sXdhWaYhOIzp+XgItcgA"
        "R49P4N4HHl25dcv6NX3lom7dsh595SIKCRu4f64x50vw0nW7FpOT06g1Wzg+dvYN5XLpTYgiVVVR+OAwc3YD843CNOF2XZcF"
        "TjrjiEjo5vV5ixENARPexcIf4jj2y7KuzSpf6QyT/xsCZxVwotX+ykX3PXLgHZHgtjNnJjBeb2FRtcyB5Axy7maQOI7wrmuu"
        "SJrTM3+05/Fn0VeOxb/Ta1gK9QSIpqcDfe+Usw4IS7B0l92I5EuytN8qfB8NV7/vybK+h0s1KwL43fa567C0eDw4WJR7HzmE"
        "06fPvmf7yPDiQhJj+8gwCkmMm2+4joPJGWT+HTjqG/+OnTxTHXth6q1RoZjtgPt3er8vkfVTaW+pUPPq1Yv2Q9JkPEwpPiMJ"
        "OYaJ4JNy+J13m5427NlXyapjoul5K3S7FjCxHjs5fvm3f/jIUKM5c3bfE8+h0ZzB6za/moPJAJlfH9p5Bb5858PZagswnWq1"
        "kjjre6jmnr/wCXdkTJ64w0Dhz3nkQeC7ddPZJ9vXMIA6zH1+FPkcxbk5G4omSr+H30TUMIFFxmCgWpYf/fQpYM+RDgDcfMN1"
        "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"
        "+vPf4qARERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "EREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREdEv5r8AU9wXd/BNb2wAAAAASUVO"
        "RK5CYII="
    ),
    "rr": (
        "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAA"
        "DsIAAA7CARUoSoAAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCA1LjEuMTITAUd0AAAAuGVYSWZJSSoACAAAAAUAGgEFAAEA"
        "AABKAAAAGwEFAAEAAABSAAAAKAEDAAEAAAACAAAAMQECABEAAABaAAAAaYcEAAEAAABsAAAAAAAAAPJ2AQDoAwAA8nYBAOgD"
        "AABQYWludC5ORVQgNS4xLjEyAAADAACQBwAEAAAAMDIzMAGgAwABAAAAAQAAAAWgBAABAAAAlgAAAAAAAAACAAEAAgAEAAAA"
        "Ujk4AAIABwAEAAAAMDEwMAAAAACDfy8cctDT3wAAEPFJREFUeF7tnXfMFUXbh+exd7FXiiX2QuwFfTF2icIfYE0UiVhBea3R"
        "RCkaxYKI8Q9RMIqoaFBfI8ZuxMRorNi7gsbee3e/uYYdvyfmVV95OGXPua5ksjvbzu7M/Zv7npk95wQRERERERERERERERER"
        "ERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERER"
        "ERERERERERERERERERERERERERERkXmlo1zKfKQoim5x0XtuLvyrXHbr6Oj4d7neZeJnjI+LL+bmwsxyOSt+Rt4m8wEF0gWi"
        "kfaNi14x9YyGOTptbCLi/Y2MizkxzY7392DaKFILorH1jmlEma00PAfPU2ZF/jmlIGiBWx6eU8HIXxINpFtMg8tsW0M5UB5l"
        "VtoVRfH3KJY2JFY4nWv5h1huLQytYEwt0cluNJQj5VlmpcrEiuwVU1t0tusN5Ur5llmpErZw9cXyrghUFC1bmZU6Qrm3mlBa"
        "aiY9Vs6Ajo6O/5TZlmD69OnFm2++GV566aXw3nvvheWWWy5tX2CBBVKKzxx+/vnntIzPHhZaaKGwxBJLhFVWWSWsueaaYYUV"
        "Vgj7779/Xeu5leqhJQQSK6RXrJDZZbaSzJgxo3j++efDO++8Ez755JPw7rvvhi+++CIZPkL45Zdfwg8//BB++umn34VBWnDB"
        "BcPCCy+crsEx5Nm/2GKLhUUXXTRtZ7n66quH7t27h169eoUTTjihLvXeCvVSeYHEShgcK+HqMls5Lr744uLVV18NL7zwQsBT"
        "4AVIeAKMnfUskpxnmQUDiIHjf/3115SA43/77bfw3XffhR9//DEsssgiYemllw6LL754WGONNcImm2wStthiizBo0KCa2kDV"
        "66eyAokFz9uxlXxz9aqrrioefvjhJIoPP/wwfP/998mACY1YYtCAZ8DIv/3227Qkz34S4E0IrzIIiQT5WASFaDg/r3/11VdJ"
        "QD179kxC2XLLLcPRRx9da6FUsr4qKZAqFvYNN9yQQqgHH3wwhVCIgpYfI86hEQIhrEIQeAf25VYfw+c49gH7CJ0weoSC4WP0"
        "5DmOPOtcF8iznc9ELJzPNb/88su0bbfddgv77rtv6N+/f81soor1VjmBxELuHQt5Vpltes4///xi5syZ4Y033kjhDiLIrTmt"
        "fDZoPAFGzLJ3795ho402Cssvv3wSCEKgg04oRhj20UcfJYEtueSSaR8C4ZqsY/Sscy3WEQjhF5+T89wHeYTHsYiOPNc7++yz"
        "w4ABA2opkkrVX6UEUqXCPemkk4rHH388eQuMMBstYMh/NExaf7wGo09DhgwJ++23XxITBk0Lj3G//PLL4dprrw3PPPNMeO21"
        "19K5GD/nsZ/rIJwsCD6Pz2Gd63MM23KeBGzPadVVVw1HHHFEGDx4sCKJzC2hClCVQj3kkEOKGNMXDzzwQHj77bfDN998k1pq"
        "IKwhYaCMUm211VZh9913T0aNF2E7oll//fWTODD4HCKxb6211gqnnnpqGDhwYFh55ZXTtTFyBEG/BePneAw9C4N8FgJ5jmUb"
        "61kUeR/Hffzxx+Hcc88NkydPnrujBsTPmkV9ltmmphICiYVJ7NrU4jjqqKOK7t27pxGpHAJlw6SFJ2HAGPKOO+4YLrroonDB"
        "BReEgw8+OKy99tqp45z333LLLSkMAgwYowU8Sbdu3cJ2220Xdt555yQSjkN0zI907rDPK4R9n332Wbj99tu5j1qLpOknFec2"
        "T03O6NGj545nNhlTpkwpYkWPimIY9frrr6dJuffffz8Jg/4DrTUtOXn2bbjhhuHYY48Ne++9dxIJcxO03K+88kp466230jkY"
        "OZOCeJd11lkntfYYLUv6GiTCIMTxwQcfpJEwxEfYhlg4titwrwgVUTJgEK9fs68SN2u9dqbpPUhsZZrutZErr7yy6NevX3He"
        "eecl41522WVTGEXLi/EutdRSqe/x6aefpn2I4Zhjjkkd4IMOOihsttlmyRMAhs58BF6Ac/AYhE73339/MngExr7cz8CLIKoe"
        "PXqEDTbYIKy44opJiGwnJOsqfDb3xPM89dRT4bTTTquZF4FmrN/ONHUnPRZeU/U7rrnmmuLmm29OIRStK8aL0dLq08LTb2DY"
        "FOPHYDHgPfbYI2y99dZpzoFWHjB29nM+MPQ7ZsyY5EXwAAhspZVWCqecckrYYYcdUj6DAT/55JNh1qxZ4emnn06JfgufjXfh"
        "froCoSHhGs+DYFdbbbXc72nLTnulRrEaBbH4HXfckQwTAyJh3Bg8rXfuYGPcGP+BBx4Ytt122+Qp6FgDhotHyCNNCIRj8Q5z"
        "5sxhRj3cfffdvwuNY5mXuPDCC9P5CPLZZ58N9913X7qP2bPnvsGBKHJoxbW5l67A/SByrocXQfyEhBMmTGhLW2nah44G1Tca"
        "XcN/qobh2nvvvTe1pnk0CjFgSCTWMVKMnuUBBxzAu07pvafsXWjhCbVypx2DRgQZjrv00kuTF6HF5hzEs9566xHipPMIdx59"
        "9NHkMTgeuB8Mmc9mGy0/QuoKCJf7Y7CAPhH3wbNEYdbUVpqlvv+IHuRPuPrqq4sbb7wxzT1gNBgi4Q2iwBjpyOJF8CDsYxut"
        "99ChQ3ltI4VFOZTKcE7uBHMegiJxTTxD7LQmA0d0XI9j2YeBPvHEE2mdz0IIjHp9/fXXKZzjWtwb5+Qwbl7hOnze559/np4P"
        "oTBZOWLEiHDkkUe2nb00ZSc9VnRDh//OOOOMYuzYsSmkoZXGYPEChD+EMRgxHmHjjTcOJ598cjj99NOTEDAmJvHoowDnZQhb"
        "OJ/rYNAZhIQh4gV4PR0wdGA7ImNSMA8EcC7C4NrLLLNMOiaLrKviAJ4P8QGC5zMAgdaaRtf7f6NZR7EGlMu6wvtSffr0KaZN"
        "m5aMhNYZo8NIMRaMkFc+Nt988zBs2LBwzjnnhMMPPzzstNNOaSSKYxke5RiOpzUmXAIMj3UEwT76Hcy033bbbWHixIkBb1UP"
        "I/w78G7Ze+GtWPL8hJh1oCH1/lcYYpWceeaZxU033ZRaaFpzDIN+AsZMq4/xM6RKh5WJOoZuc58EUUydOpXh35SnL8Fsd2cY"
        "Aua7HoiHsA1Pg0gYCsYjAJ+RvUejyGEj94FY8Ix4TEbkat0PaUbaXiCTJk1KfQ3CGFp44m1CIUAYtJ58f2KvvfZKb7ziLfJ+"
        "DIiwipaWYdcosjTJx5wHnXVCH0aceOWECUQEwTqC4LPwSIRKHMc18C58ZiPJ4uCeaBxoJPCETHLec889CqSdGDlyZDF9+vTk"
        "AfAOtJaEVogCD0KfgOHaXXfdNXkNZsPpS2DIGA3Gg2ED71Ydd9xx4bHHHkujT7yNy3XwFHznA1FhcByPEbIk8TkYJIIhkW8k"
        "+fO5FxoAQsMskLvuukuBtAO33nprMWXKlPDcc88lI6CvQUtOuIPxMrxJSNG3b9+w5557Jg8CvNrBTDkQinAO4QchGR5o3Lhx"
        "yYPgBTB+QiY8DCCqHEKRMDrEhkHSEcabIKBGgzC4d56N+8whFqI3xGoDLr/88uLOO+8ML774Yqp8jABDZUmfAkMeNGhQmuzj"
        "+9sYPwaPIDiGLz3Rn2Ckio4rs994HeYO2IeAOIcQBeMnYfjkuQYgFASBV0GQLDHM3PdpJNwn94dAaDxYp3z45mEsNwXSylxy"
        "ySXF5MmTU4UDAsEQMFYMEyNnnRcFeeWcEInQiePxBOQ5nv4EhoQBIR7OJRTB2PPoF9uy9wDEkfscCAax5e3k2c75jYb74j5y"
        "/wqB8NzbbLMNv7CiQFoVfhxhwoQJqRNOS40B5P4ABkoeo250C46IMFIEiFcB7hEBISbW8Tjk2c9xHM95JESO8HOfh+fiWJ6L"
        "47Iw/4zs6XJIyIQh64ceeijvZLWdQJp1HmS+g7FgXMT+GE8OrzAytmdv0mi4P4yUe2IQgLAPw+/cp+EYRsIwZDwXk4js51l4"
        "Nlp8+g0ckxsCBEIZ/B1cK//2FmEkn8UABi9btiNN2SJEw63JT8WMHTu24CurGArGhSAwIFIWCcbZSLJgMWrg3rjf7AXyhCOJ"
        "Z8hej3UScC7HZw8CrPNsee7mz+CY3FDgbZkU7devH9+UrLmt1Kreu0KzCqRmv34xZsyY9CMKjGDRIjNihVHQ6mKY2TgaBfeQ"
        "BQx4ORL3xT7CQ+4br4Lx41E4nn1so/9Da88AQ+4/cC3Ejzg4/6/IL1FyPQYcDjvssLoVSC3rfV5pSoHUmssuuyx5EibtCCdo"
        "KQklcrzdSHKIhFFj4HgDBEIeT4HnyJ6BfXxvZN11103zFMzb1NOg24GmLcxoFDV9/Xnq1KnF9ddfn2bAaS0RBkaHMTYSPAVe"
        "gSWtOcvOw8YICGHwrT9eltx+++1bQhS1ru95pa1bG74Idd111/3+HQs8Ca13I0EAeAmEgRBYRyCET3lCEW+xyy671PSneWQu"
        "TV3A0Wjr8lVMfqrnkUceSaELsXojwUtwH7nzzTqiYTafbydOnjy55URRr3qeF5p6mJdCi4VX8y/1Ry/S0b9//+RFaLlZAgYK"
        "ecQo9wdI5AnJWObjORdjJo9hk/I+ljmEo2+BEDkX78BxXJsRKuAa7MsTdUxcDh8+vFXFMbJZxQG66E5cccUVBe9TZSFgoBgq"
        "YQ3Gj4FnOht+ZxAV55CyYP64jf4E5zGqRD+DdWbgSXnug+MYQOBdsFGjRllPDaISBR8NqG7Df/xhDf2Shx56KA2Z0hlmdItJ"
        "t9xpJmUwbhLGzxIxsATWSRyfj8FLIDzExj6WJK7NJB0vRHIMr3ZMmjSppYURy6Tpf8y6MhUQC7OuceqJJ55YzJgxI7XqzAdg"
        "zLwWn/so5DMYf06QRZRT3odA8EaIje35uyfkEQ37mJjbZ599+KXGVhdH0/Y7OlOpSqh3ofL+Fl+m4oVFwh4S7zplwycByyyO"
        "TN6XPUven7cTxgHzL+xjLmPTTTfl101aWhhQ73rsCpWrjEYU7sCBAwt+cgdDZr4kGztLDL5zyvuA9c6hFwlPkfOIhH4GXmPi"
        "xIktLwyIz14ZcUAlKyUWct1jV759SN8EL4LRM/KUR6UQRhYA+zKdxYEYSLnPwSsu9DPGjx/fFsKARtRbV6ls5TSisKdNm1bw"
        "S4eIg5Eohn8xdqAvgVAQBUvIYmGmnr4LQ7b5vwGHDx/eNsKAKooDKl9JseDr/gbo0KFDC37oDe+A8SMIBIJoEAEeA+GQZ3SK"
        "vgX/AzJkyJC2EgU0on7mJy1RYbES6v53w7HzXvA/Hvy2FYIg9GLEi4k/+il0uvv06RPOOuusthNFphH1In9BrJC6//DYsGHD"
        "ihg2FT169Chin6Igz08JlbvbkkbUg/yPxMrpFlNd/3Ni3LhxxfHHH1/w987lpraEcqf8y6w0M1ZUfbG8K0qsuF60bGVW5iOU"
        "K+VbZqXK0MLFNKLMShegHCnPMiutRqzcvuWq/AMstzaDVjCmwWVW/guUD+VUZqVdUSz/j6KQvyUaSO+Y2qJzz3PyvGVW5J9T"
        "CqYlOvk8h4L432jb1yDmB9HI6LQyzNmzo6NjdNrYRMT7wwPOiWl2vL+m+0mdKqBAakA0TOL33EL/q1zyNuu/y/UuEz9jfFzk"
        "t2NnlstZ8TMq98asiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiI"
        "iIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiEhTE8L/Ad67ZXkPjoL5AAAAAElFTkSuQmCC"
    ),
})


if __name__ == "__main__":
    _code = main()
    if _code:  # only raise SystemExit on failure, so IDE debuggers don't stop on a clean exit
        sys.exit(_code)
