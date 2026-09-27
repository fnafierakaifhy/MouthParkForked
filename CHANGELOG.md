# Changelog

## 0.6.1 — no compiler needed

### Fixed

- **`pip install allosaurus` failed on Python 3.14 / Windows** with "Microsoft Visual C++ 14.0 or greater is required" while building `editdistance`. Recognition never uses `editdistance`: it's only for allosaurus's training code. It doesn't need `resampy`→`numba` either, because MouthPark always converts audio to 16 kHz first.
  - When either package is missing or broken, MouthPark now supplies a small built-in stand-in: a pure-Python Levenshtein for editdistance, and SciPy's `resample_poly` for resampy. If you have the real ones, those are used.
  - Recognition with the stand-ins gives identical phonemes and frames on the sample clips.
- A leftover allosaurus folder (its model survives an uninstall) is no longer mistaken for an installed package.

### New

- **`python mouthpark.py --install-deps`** and an **Install** button in the app (Settings: phoneme recognizer; the header chip opens it):
  - It installs only what's missing.
  - It uses prebuilt packages only for compiled dependencies, so it never falls back to compiling.
  - It adds allosaurus and panphon without their compiler-only extras.
  - It never upgrades or downgrades anything you already have.
  - It works with pip, or with uv in environments that have no pip.
  - It gives clear hints when PyTorch has no package for your Python, or when the OS locks the system Python.
- `--doctor` shows exactly what the recognizer is missing and which stand-ins are in use.
- Render stays disabled, with a clear "Install the phoneme recognizer first" message, until the recognizer is ready.

### Changed

- `requirements.txt`, `pyproject.toml` and the script's inline (PEP 723) dependencies no longer list `allosaurus` directly, since that is what dragged in the compiler-only packages. Use `--install-deps` or the app's Install button.


## 0.6.0 — choose your ffmpeg

### New

- **Settings: ffmpeg** in the app:
  - **Browse…** opens your system's file picker.
  - You can paste a path to the exe or its folder (`bin\` is checked too); quotes from Explorer's "Copy as path" are fine.
  - **Use automatic** goes back to PATH.
  - The panel shows which ffmpeg is in use, where it came from, and its version.
  - The **ffmpeg** chip in the header opens the panel, and it opens by itself if no ffmpeg is found.
  - The format list updates to match what the chosen ffmpeg can encode.
- **Command line:** `--set-ffmpeg PATH` (or `auto`) saves your choice; `--ffmpeg PATH` applies to one run. The `MOUTHPARK_FFMPEG` environment variable also works. `--doctor` shows which ffmpeg is used and why.
- Your choice is saved in a small per-user `config.json` and applies to both the app and the command line.

### Safer

- A chosen ffmpeg must be named `ffmpeg` / `ffmpeg.exe` **and** answer `-version` like ffmpeg before it's saved or run, so a random program can't be dropped in.
- MouthPark runs the file picker as a separate, fixed Python snippet, never built from input, and only one picker can be open at a time.
- If a saved path later disappears, MouthPark warns and falls back to PATH.
- Self-tests use a throwaway settings folder, so they never change your real settings.


## 0.5.1

- **Running `mouthpark.py` with no arguments opens the app**, so double-clicking it or pressing Run in an IDE just works. Before, it stopped with "the INPUT audio file is required" (`SystemExit: 2`).
- **No more `SystemExit(0)` on a clean exit.** The script only raises `SystemExit` for a non-zero exit code, so Visual Studio's and VS Code's debuggers don't break when you click Quit.
- **Removed the `.bat` / `.command` / `.sh` launchers.** The `.py` is all you need.


## 0.5.0 — the app

### New

- **`python mouthpark.py --gui`** opens a point-and-click app in your browser that does everything the command line does:
  - voice upload with playback;
  - mouth-only rendering, on its own or on a bigger canvas you drag it around;
  - the full character mapper (drag, size, tilt, see-through, cover patch with eyedropper), plus opening and saving packs;
  - built-in or custom mouth PNGs;
  - frame rate, min hold and silence;
  - WebM/MOV/MP4 output, background and audio;
  - an editable phoneme → mouth table and a max-length limit;
  - rendering with a live progress bar, log and Cancel;
  - playback of the result over a checkerboard, and downloads of the video and timeline.
- Drag files anywhere onto the window: audio goes to Voice, images and `.zip` packs go to Character. Paste an image to use it.
- The shape strip previews every mouth or fakes talking at your chosen frame rate and hold.
- **Faster repeat renders:** the phoneme model stays loaded, and phonemes are reused for the same clip, so changing the look re-renders in about a second.
- Double-click launchers for Windows and macOS (removed again in 0.5.1).

### Safer

- The app server binds to 127.0.0.1 only, on a random port. Every API call needs a 256-bit per-launch token, sent in a custom header (which other sites can't send without failing CORS) and compared in constant time. The token is removed from the address bar after load.
- It rejects any Host header other than `127.0.0.1:<port>` / `localhost:<port>`, which blocks DNS-rebinding attacks.
- The page ships a strict Content-Security-Policy (`default-src 'none'`, no third-party origins, no eval, no framing) plus `nosniff`, `no-referrer` and `no-store`.
- The browser never supplies file paths. Uploads get server-chosen names in a private (0700) temp folder, and only outputs the server made can be downloaded. Upload sizes are capped per kind, audio extensions and mouth names are allow-listed, mouth PNGs are validated on upload, and every render option is type- and range-checked. The folder is deleted on Quit or Ctrl-C.
- Only one render runs at a time. Cancel stops encoding cleanly, and a crash in a render never takes the server down.

### Fixed

- `--doctor` now finds a verified model installed in allosaurus's own folder, not just in MouthPark's cache.


## 0.4.0 — character packs + mapper

### New

- **`mapper.html`**: a drag-and-drop page for fitting the mouth onto any character image. Drag to place; size and tilt sliders; exact X/Y fields; arrow-key nudging; zoom and pan; a see-through slider to line up with the drawn-on mouth; live preview of all 10 shapes plus fake talking; one-click **Download pack (.zip)**. It runs entirely in the browser (works offline), and the image never leaves your machine.
- **Cover patch**: paint a skin-coloured ellipse over the character's original mouth, with an eyedropper that averages a 3×3 area. It tilts with the mouth, and with `rest: "blank"` it's what shows during silence.
- **`--pack PACK`** loads a pack from a `.zip`, a folder or a `pack.json`. CLI flags still override the pack's values.
- **`--rotation DEG`** tilts the mouth.

### Safer

- Packs are read straight out of the zip without extracting anything. There's a 64-file cap and a 64 MB per-file cap enforced on the bytes actually read (not the size the header claims), and encrypted entries are refused.
- The pack's image must be a plain PNG/JPEG/WebP file name: no folders, no `..`, no absolute paths. For folder packs the resolved path must stay inside the pack folder, so symlinks can't escape it.
- Every number in `pack.json` is type- and range-checked.
- New self-test covers packs, including rejecting traversal names.


## 0.3.0 — single-file rebuild

The five-module `mouthpark/` package is now one file, `mouthpark.py`, with the default mouth art embedded. Given the same audio and settings it recognises the **same phonemes and picks the same mouth for every frame** as 0.2.0 — this was checked on all three sample clips, and the built-in `--self-test` compares the new timing code against the original algorithm on thousands of random inputs.

### What it still does (unchanged from 0.2.0)

- Recognises phonemes with allosaurus; no transcript needed.
- Maps IPA to the same 10 mouth shapes with the same table, stress/length stripping, prefix fallback, and `clenched` for unknowns; schwa counts as rest.
- Holds vowels up to 2.0 s and consonants up to 0.25 s, picks the dominant mouth per frame, and merges runs shorter than `--min-hold`.
- Defaults: 18 fps, min-hold 2, silence shows `closed.png`.
- Writes WebM/VP9 with alpha; `--background`, `--bg-color`, `--with-audio`, `--keep-frames`, `--mouths-dir` and `-v` all work as before.
- Exit codes 0–3 mean the same things.

### New

- **One file, no assets needed.** Copy `mouthpark.py` anywhere and run it. It declares its dependencies inline (PEP 723), so `uv run mouthpark.py voice.mp3` installs them automatically.
- **More output formats**, chosen by extension: `.mov` (ProRes 4444 with alpha, for Premiere/Final Cut/Resolve) and `.mp4` (H.264).
- **Compositing** (from the PRD's v2 list): `--character IMG` puts the mouth on a character image; `--canvas WxH`, `--position X,Y` and `--mouth-scale` place and size it.
- **`--rest blank`** shows nothing during silence (what the PRD originally specified), instead of the closed mouth.
- **`--mapping FILE.json`** changes the phoneme→mouth table without editing code. `--print-mapping` shows the current table.
- **`--timeline-out` / `--events-in`** save the recognised phonemes and per-frame mouths as JSON, then reuse them to re-render instantly without running recognition again.
- **`--preview`**: shortcut for `--background --with-audio`.
- **`--doctor`** checks your environment, including whether your ffmpeg has the needed encoders.
- **`--self-test`** runs the built-in tests.
- **`--export-mouths DIR`** writes the built-in mouths out so you can customise them.
- **`-q` / `-vv`** for quiet or very verbose (shows the ffmpeg command) logging.

### Safer

- **Verified model download.** 0.2.0 let allosaurus fetch its model with no integrity check and unpack it with an unfiltered `tarfile.extractall` (a path-traversal risk). MouthPark now downloads the model itself over HTTPS with a size cap, checks a pinned SHA-256, unpacks it with Python's `data` tar filter, and re-verifies `model.pt` before every load. The model file is a PyTorch pickle, so this check is what makes loading it trustworthy. The model is cached per user (`~/.cache/mouthpark`, `%LOCALAPPDATA%\mouthpark`, or `~/Library/Caches/mouthpark`; override with `MOUTHPARK_CACHE`) instead of inside site-packages.
- **ffmpeg inputs can't be hijacked.** Every path is passed as `file:/absolute/path`, so a file named `-y.mp3`, `http://…` or `concat:…` is treated as a plain file.
- **No silent overwrites.** Existing outputs need `--force`, and the output can never be the input file. (In 0.2.0, `mouthpark voice.mp3 voice.mp3 --with-audio` would have destroyed the recording.)
- **Atomic writes.** Video and JSON are written to a temp file and renamed into place, so a crash or Ctrl-C never leaves a half-written file.
- **Everything is bounded:** fps 1–60 (the old "cap" was never enforced), min-hold 1–30, canvas ≤ 8192 px, mouth PNGs ≤ 4096 px, JSON ≤ 64 MB, audio ≤ 30 min by default (`--max-duration`), timeouts on every subprocess, and a tighter Pillow decompression-bomb limit. Mouth files must actually be PNGs.
- **Clean errors.** ffmpeg failures, corrupt audio and allosaurus errors print a short message with the relevant ffmpeg output, not a Python traceback. Ctrl-C exits with 130.
- **Fewer dependencies:** `click` and `numpy` are no longer direct dependencies (argparse is in the standard library; numpy still arrives via allosaurus).

### Fixed

- **Stale frames:** `--keep-frames` into a folder left over from a longer run used to encode the old extra frames onto the end of the video. Frames are now piped straight to ffmpeg, and old `frame_NNNNNN.png` files in the folder are cleared first.
- **Default mouths folder** was relative to the current directory, so running from anywhere else failed. The built-in mouths now always work.
- **Late failures:** a missing ffmpeg or encoder used to be discovered only after the slow recognition step. It's now checked first.
- **WAV edge cases:** 24-bit, float, stereo and 48 kHz WAVs are now always converted to 16 kHz mono PCM before recognition, rather than passed through as-is.
- **Odd-sized canvases** are padded to even dimensions, which yuv420 encoders require.
- **Windows consoles** no longer crash printing IPA symbols; no more `PYTHONIOENCODING` workaround.
- **Docs:** README and PRD disagreed on FPS (18 vs 12) and on what silence looks like; both are now documented as options.
- New exit code **4** for ffmpeg/encoding failures (0.2.0 used 3, the same as recognition failures).

### Faster

- No temporary PNG sequence: the ≤ 11 distinct frames are rendered once and raw pixels are streamed into ffmpeg.
- Frame quantization is O(frames + events) instead of O(frames × events), and min-hold is a single pass instead of a repeat-until-stable loop. Output is identical.

### Small behaviour differences

- Clip length is now measured from the decoded audio samples rather than ffprobe's container duration, which includes MP3 encoder padding. This can make the video **one frame shorter** than 0.2.0 on some MP3s (2 of the 3 samples). The mouths on every other frame are identical.
- Python's `import mouthpark.recognizer`-style module paths no longer exist; use `import mouthpark` (the functions `recognize`, `quantize`, `Mapper`, `enforce_min_hold` etc. live there).
