# MouthPark

South Park-style lip-synced mouth animation from an audio file — in **one Python file**.

**Input:** an English voice recording (`.wav`, `.mp3`, anything ffmpeg can read).
**Output:** a video of just the mouth on a transparent background, ready to drop onto your character — or, new in 0.3, composited straight onto the character for you.

```
audio → ffmpeg (16 kHz mono) → allosaurus phonemes (IPA)
      → phoneme → mouth (10 shapes) → 18 fps + min-hold
      → frames piped into ffmpeg → .webm (VP9 α) | .mov (ProRes 4444 α) | .mp4 (H.264)
```

`mouthpark.py` contains everything, including the default mouth artwork and a full point-and-click app (`--gui`), so you can copy that single file anywhere and run it. `mapper.html` is a standalone copy of the character mapper. See [CHANGELOG.md](CHANGELOG.md) for what's changed.

---

## Character packs (the easy way to use your own character)

1. Open **`mapper.html`** in any browser (double-click it; works offline).
2. Drop your character image on the green mat.
3. Drag the mouth onto the face; set size and tilt. Use **See-through** to line it up with the drawn-on mouth, and **Zoom to mouth** for precision.
4. If the picture already has a mouth, tick **Cover the original mouth**, click **Pick skin colour**, click the face, and size the patch.
5. Click through the shapes (or press **Space** to fake-talk) to check every one fits.
6. **Download pack (.zip)**, then:

```bash
python mouthpark.py voice.mp3 talking.mp4 --pack my-character.zip --with-audio
```

A pack is just a zip (or folder) with `pack.json` and the image; MouthPark reads it without extracting anything. Command-line flags such as `--position`, `--mouth-scale`, `--rotation` and `--rest` override what's in the pack, so you can fine-tune without re-exporting.

```json
{ "mouthpark_pack": 1, "name": "My character", "image": "character.png",
  "position": [660, 845], "mouth_scale": 0.9, "rotation": 6,
  "cover": { "color": "#b89f82", "center": [660, 845], "size": [130, 60] },
  "rest": "blank" }
```

`rotation` is in degrees counter-clockwise. `cover` is an ellipse painted over the original mouth (it tilts with the mouth). `rest: "blank"` shows just the cover during silence.

---

## The app (no command line needed)

```bash
python mouthpark.py
```

Run it with no arguments and the app opens in your browser. The first time, click **recognizer missing — install** in the top-right corner to set up the phoneme recognizer. It's one click, needs no compiler, and never changes packages you already have. That also works when you double-click the file (if `.py` opens with Python) or press **Start/Run** in Visual Studio, VS Code or PyCharm, so there are no launcher scripts. `--gui` does the same thing explicitly. The app:

1. **Voice**: drop in an audio clip and play it back right there.
2. **Character**: choose *Mouth only*, optionally on a bigger canvas you drag the mouth around on, or *On a character*: pick an image, drag the mouth onto the face, set size and tilt, and cover the drawn-on mouth with a colour-picked patch. You can open and save packs here too.
3. **Mouths & timing**: built-in or your own 10 PNGs, frame rate, min hold, what silence looks like.
4. **Output**: WebM, MOV or MP4; background; include audio.
5. **Render**: watch progress, then play the result (transparency shows on a checkerboard) and download the video or the timeline.

The bar under the preview flips through every mouth shape, or fakes talking (Space), so you can check the fit before rendering. The first render of a clip listens for phonemes. After that, changing the look re-renders in a second or two. The phoneme → mouth table is editable under *Phoneme → mouth mapping*.

**Private by design:** the app only listens on `127.0.0.1` (your own computer), on a random port, behind a secret per-launch token in the link it opens. Other websites and other devices can't talk to it. Files you add go into a private temp folder that's deleted when you click **Quit** or press Ctrl-C in the terminal window. The page loads nothing from the internet. `--port N` picks a fixed port; `--no-browser` just prints the link.

## Quick start (command line)

You need **Python 3.11+** and **ffmpeg** on your PATH.

**With [uv](https://docs.astral.sh/uv/)** (recommended — the file declares its own dependencies, PEP 723):

```bash
uv run mouthpark.py voice.mp3          # → voice.webm
```

**With pip:**

```bash
python -m pip install Pillow
python mouthpark.py --install-deps           # the phoneme recognizer, no compiler needed
python mouthpark.py voice.mp3
```

Don't use `pip install allosaurus` directly. It pulls in `editdistance` and `numba`, which on new Pythons (e.g. 3.14 on Windows) have no prebuilt packages and demand Microsoft C++ Build Tools. MouthPark doesn't need either for recognition: `--install-deps` installs allosaurus without them, and MouthPark uses small built-in stand-ins when they're missing. If you already have them, they're used as normal.

**As an installed command:** `pip install .` then `mouthpark voice.mp3`.

The first run downloads the allosaurus phoneme model (~40 MB) into your user cache, checks its SHA-256, and reuses it afterwards. Run `python mouthpark.py --doctor` any time to check your setup.

### Choosing which ffmpeg

MouthPark uses the ffmpeg on your PATH by default. To use a specific one, for example a portable ffmpeg folder that isn't on PATH:

- **In the app:** click the **ffmpeg** chip (top right) or open **Settings: ffmpeg**, then **Browse…** to the exe, or paste its path (the exe or its folder; `bin\` is checked too). **Use automatic** goes back to PATH. If ffmpeg can't be found at all, this panel opens by itself.
- **Command line:** `--set-ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"` remembers it for every run (`--set-ffmpeg auto` to undo). `--ffmpeg PATH` uses one just for this run. The `MOUTHPARK_FFMPEG` environment variable also works.

Order of preference: `--ffmpeg` → `MOUTHPARK_FFMPEG` → your saved choice → PATH. The saved choice lives in `%APPDATA%\mouthpark\config.json` (Windows), `~/Library/Application Support/mouthpark/config.json` (macOS), or `~/.config/mouthpark/config.json` (Linux).

For safety, a chosen file must be named `ffmpeg` / `ffmpeg.exe` **and** answer `ffmpeg -version` like the real thing before MouthPark will use it.

### Installing ffmpeg

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| Windows | `winget install ffmpeg` (or `choco install ffmpeg`) |

---

## Usage

```
python mouthpark.py INPUT [OUTPUT] [options]
```

If `OUTPUT` is omitted it writes `INPUT` with a `.webm` extension. The **output extension picks the format**:

| Extension | Codec | Transparency | Good for |
|---|---|---|---|
| `.webm` | VP9 | yes | web, OBS, most editors (default) |
| `.mov` | ProRes 4444 | yes | Premiere, Final Cut, DaVinci Resolve, After Effects |
| `.mp4` | H.264 | no (flesh background added) | quick previews, sharing |

### Options

| Flag | Default | What it does |
|---|---|---|
| `--fps N` | `18` | Frame rate (1–60). |
| `--min-hold N` | `2` | Minimum frames a mouth shape must hold; shorter runs merge into a neighbour. |
| `--rest closed\|blank` | `closed` | During silence show the closed mouth, or nothing at all. |
| `--pack PACK` | — | Character pack (.zip / folder / pack.json) from `mapper.html`. |
| `--mouths-dir DIR` | built-in | Use your own 10 mouth PNGs. |
| `--mapping FILE.json` | — | Override which phoneme uses which mouth (see below). |
| `--background` | off | Opaque flesh-tone background (no alpha). |
| `--bg-color #RRGGBB` | — | Custom opaque background; implies `--background`. |
| `--character IMG` | — | Composite the mouth onto a PNG/JPEG/WebP character image. |
| `--canvas WxH` | mouth size | Place the mouth on a larger transparent canvas. |
| `--position X,Y` | centre | Where the mouth's centre goes on the canvas/character. |
| `--mouth-scale F` | `1.0` | Resize the mouths. |
| `--rotation DEG` | `0` | Tilt the mouth (counter-clockwise). |
| `--with-audio` | off | Mux the input audio into the video. |
| `--preview` | off | Shortcut for `--background --with-audio`. |
| `--keep-frames DIR` | — | Also write the PNG sequence. |
| `--timeline-out FILE.json` | — | Save phonemes + the per-frame mouth list. |
| `--events-in FILE.json` | — | Reuse a saved timeline — skips recognition, so re-renders are instant. |
| `--ffmpeg PATH` | — | Use this ffmpeg for this run. |
| `--set-ffmpeg PATH\|auto` | — | Remember an ffmpeg for every run (then exit). |
| `-f, --force` | off | Allow overwriting existing output files. |
| `--install-deps` | — | Install the phoneme recognizer (no compiler, never changes what you have), then exit. |
| `--max-duration SEC` | `1800` | Refuse longer audio (0 = no limit). |
| `-v` / `-vv` / `-q` | — | More logging / show ffmpeg commands / errors only. |

App: run with no INPUT, or `--gui` (with `--port N`, `--no-browser`). Utilities (run and exit): `--doctor`, `--self-test`, `--print-mapping`, `--export-mouths DIR`, `--version`.

### Examples

```bash
# Transparent mouth for compositing (VP9 + alpha)
python mouthpark.py voice.mp3

# For Premiere / Final Cut / Resolve
python mouthpark.py voice.mp3 voice.mov

# Quick preview with sound on a flesh background
python mouthpark.py voice.mp3 --preview

# Straight onto the sample character, with audio, as an MP4
python mouthpark.py test02.mp3 j01_talking.mp4 \
    --character characters/j01.png --position 660,845 --mouth-scale 0.9 --with-audio

# 1080p transparent canvas, mouth placed for your rig, nothing shown in silence
python mouthpark.py voice.mp3 --canvas 1920x1080 --position 960,700 --rest blank

# Recognise once, then iterate on the look instantly
python mouthpark.py voice.mp3 --timeline-out voice.timeline.json
python mouthpark.py voice.mp3 v2.webm --events-in voice.timeline.json --fps 12 --min-hold 3
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Bad input or usage (missing file, output exists, bad option, audio too long…) |
| 2 | Missing / mismatched / unreadable mouth asset |
| 3 | Phoneme recognition failure (allosaurus missing, model download/verification failed…) |
| 4 | ffmpeg missing or encoding failed |
| 130 | Interrupted (Ctrl-C) |

---

## Mouth assets

Ten PNGs, all the **same size** with **transparent backgrounds**. The defaults are embedded in `mouthpark.py`; the same files are in `mouths/` for editing.

| File | Phoneme bucket |
|---|---|
| `closed.png` | m, b, p, glottal stop, **silence** |
| `clenched.png` | d, t, s, z, k, g, n, y, sh, ch, j, zh (+ anything unknown) |
| `ah.png` | a, æ, ɑ, aɪ, aʊ, ɒ, h |
| `ee.png` | ɛ, eɪ, i, iː, ɪ |
| `oh.png` | o, oʊ, ɔ, ɔː, ɔɪ |
| `woo.png` | u, uː, ʊ, w |
| `bite.png` | f, v |
| `tongue.png` | l, θ, ð, plus bare `e` (allosaurus emits it for letter-name "A") |
| `uh.png` | ʌ, ɜ (stressed "cup" vowel only) |
| `rr.png` | r, ɹ, ɻ, ɝ |

Reduction vowels `ə` and `ɚ` count as rest.

**Your own art:** make 10 PNGs with the same file names and pass `--mouths-dir path/to/folder`. `--export-mouths DIR` writes the built-in set out as a starting point.

**Your own mapping:** no code editing needed any more. Write a JSON file of overrides and pass `--mapping`:

```json
{ "θ": "bite", "ð": "bite", "ə": "uh", "_fallback": "clenched" }
```

Values are a mouth name or `null` (rest). Keys starting with `_` are ignored except `_fallback`. `--print-mapping` prints the full table; `examples/mapping.example.json` is a template.

---

## How the timing works

Allosaurus emits each phoneme as a ~45 ms pulse. Each pulse is held until the next one starts, capped at **2.0 s for vowels** (so "ohhhh" holds `oh`) and **0.25 s for consonants**. Past the cap the frame is rest. Each frame shows whichever mouth covers most of its time window, then any run shorter than `--min-hold` frames merges into its longer neighbour so nothing flickers. The knobs are `VOWEL_HOLD` and `SILENCE_GAP` near the top of `mouthpark.py`.

`--timeline-out` saves this so you can inspect it:

```json
{ "fps": 18, "duration": 6.0, "events": [{"phoneme": "h", "start": 0.12, "end": 0.165, "mouth": "ah"}, …],
  "frames": [null, null, "ah", "ah", "ee", …] }
```

---

## Safety notes

MouthPark is designed to be safe to point at files you didn't make:

- The phoneme model download is pinned to a SHA-256 checksum, size-capped, HTTPS-only, and unpacked with Python's safe tar filter. A modified or corrupted model is rejected.
- Every path handed to ffmpeg uses the `file:` protocol, so file names can't be misread as URLs, protocols, or options.
- Nothing is overwritten without `--force`; the input can never be overwritten; videos are written to a temp file and renamed into place, so a failure never leaves a half-written file.
- Images, JSON files, canvas size, audio length, FPS and all numeric options are range-checked; Pillow's decompression-bomb limit is tightened.
- No shell is ever invoked; every subprocess has a timeout; errors print a message, not a traceback.

---

## Troubleshooting

**`Microsoft Visual C++ 14.0 or greater is required` / `Failed building wheel for editdistance`**: this comes from `pip install allosaurus`. Use `python mouthpark.py --install-deps` (or the app's Install button) instead; it skips the packages that need compiling.

**`the phoneme recognizer isn't installed`**: run `python mouthpark.py --install-deps` with the same Python you run MouthPark with (in Visual Studio, that's the project's selected environment), or click Install in the app.

**`torch` fails to install**: PyTorch may not have a package for a brand-new Python yet. Check https://pytorch.org/get-started/locally/, or install a slightly older Python alongside.

**`ffmpeg not found`**: install it (table above) and open a new terminal, or point MouthPark at it (app: *Settings: ffmpeg → Browse…*; CLI: `--set-ffmpeg PATH`).

**Mouth looks wrong on a vowel** — run with `--timeline-out t.json`, look at which phonemes allosaurus heard, and add overrides with `--mapping`. Re-render instantly with `--events-in t.json`.

**Something else** — `python mouthpark.py --doctor` checks Python, Pillow, ffmpeg and its encoders, allosaurus, and the model cache. `--self-test` runs the built-in tests.

---

## Project layout

```
MouthPark-0.6.1/
├── mouthpark.py          # the whole program (+ embedded default mouths)
├── mapper.html           # standalone copy of the character mapper
├── mouths/               # the 10 default mouth PNGs, for editing
├── characters/           # sample characters to composite onto
├── examples/mapping.example.json
├── test.mp3, test02.mp3, test03.mp3   # regression samples
├── CHANGELOG.md          # what changed from 0.2.0
├── PRD.md                # original design doc
├── pyproject.toml, requirements.txt
```

## AI WARNING

Forked entirely by Claude

## License

MIT.
