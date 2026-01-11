# small personal tool to generate vertical story videos
# ollama -> piper -> ffmpeg + word-by-word subtitles
# not meant to be a librar

import json
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
import wave
from datetime import datetime
from pathlib import Path

# --- paths / config ----------------------------------------------------------


def get_base_dir() -> Path:
    """Figure out where the app lives (dev vs frozen)."""
    if getattr(sys, "frozen", False):
        # pyinstaller etc.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = get_base_dir()

VIDEOS_DIR = APP_DIR / "videos"
VOICES_DIR = APP_DIR / "voices"
PIPER_DIR = APP_DIR / "piper"
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

OLLAMA_MODEL = "llama3.1:8b"  # change if you prefer a different one

CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")
CURRENCY_RE = re.compile(r"\$([0-9]+)")


def log(msg: str) -> None:
    print(msg, flush=True)


# --- story generation --------------------------------------------------------


def _strip_control_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def make_fallback_story(length: str, custom_topic: str | None = None) -> dict:
    # basic backup so the whole thing doesn't die if ollama is down
    base = "I found something in my partner's phone that changed everything. "
    if custom_topic:
        base = custom_topic.strip() + " "

    stories = {
        "Short": base + "What I saw shattered my trust in seconds. Now I'm questioning our entire relationship.",
        "Medium": base + (
            "I thought we were solid. Then last night I saw messages that made "
            "my blood run cold. Everyone says I should forgive them. I don't know if I can."
        ),
        "Long": base + (
            "For years I believed we were the perfect couple. Then one tiny mistake "
            "showed me a second life they were hiding from me. Every photo, every call, every lie "
            "was right there in front of me. Now I'm stuck between walking away or pretending "
            "I never opened that phone."
        ),
    }

    story_text = stories.get(length, stories["Medium"])
    return {
        "title": "You won't believe what I discovered...",
        "story": story_text,
        "caption": "This changed everything 💔",
        "hashtags": "storytime drama relationships fyp",
        "ui_type": "story",
    }


def ask_ollama_for_story(mode: str, length: str = "Medium", custom_topic: str | None = None) -> dict:
    """
    Ask Ollama to spit out a story in JSON format.
    If anything goes sideways, returns a simple fallback story.
    """
    # rough word targets; not super important
    word_targets = {"Short": 180, "Medium": 360, "Long": 700}
    target_words = word_targets.get(length, 360)

    if mode == "aita":
        brief = f"""
Write a first-person AITA-style story.

Rough guidelines:
- about {target_words} words
- start straight with the situation (no greetings / intros)
- include ages, time frames, a concrete setting, and some dialogue
- make it morally grey (both sides arguable)
- build up to a decisive moment
- end with one short line that invites debate
"""
    elif mode == "relationships":
        brief = f"""
Write a first-person relationship drama story.

Rough guidelines:
- about {target_words} words
- open with a strong emotional line
- include modern stuff (texts, screenshots, calls, money, family pressure, etc.)
- 2–3 lines of dialogue
- keep it realistic and specific (not soap opera)
- finish on a line that makes people argue
"""
    elif mode == "mystery":
        brief = f"""
Write a first-person grounded mystery / thriller.

Rough guidelines:
- present-day, realistic (no ghosts, no magic)
- about {target_words} words
- start with a “something is wrong” vibe
- reveal clues gradually (no giant info-dumps)
- use a few concrete locations and some dialogue
- end with a twist that changes how earlier details look
"""
    else:
        brief = f"""
Write a high-engagement first-person story.

Rough guidelines:
- about {target_words} words
- hook early, build tension, have at least one twist
- keep it grounded and plausible
- end with a line that invites opinions
"""

    if custom_topic:
        brief += f"\nUse this situation as the core:\n{custom_topic.strip()}\n"

    prompt = f"""{brief}

Reply with ONLY valid JSON in this exact structure:
{{
  "title": "...",
  "story": "...",
  "caption": "...",
  "hashtags": "..."
}}"""

    try:
        import ollama  # type: ignore

        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp["message"]["content"]

        # models sometimes wrap JSON with junk before/after; try to slice it out
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("Didn't find a JSON-looking block in Ollama output")

        raw_json = _strip_control_chars(content[start:end])
        data = json.loads(raw_json)

        for key in ("title", "story", "caption", "hashtags"):
            if key not in data:
                raise ValueError(f"Missing JSON key: {key}")

        data["ui_type"] = "story"
        return data

    except Exception as e:
        log(f"[warn] story generation failed, falling back. reason: {e}")
        return make_fallback_story(length, custom_topic)


def clean_for_tts(raw: str) -> str:
    if not raw:
        return ""

    text = str(raw)
    # turn "$100" into "100 dollars"
    text = CURRENCY_RE.sub(r"\1 dollars", text)
    text = text.replace("$", " dollars ")
    # squish "!!!???" etc
    text = re.sub(r"[?.!]{2,}", " ", text)
    # collapse whitespace
    return " ".join(text.split())


# --- Piper / TTS stuff -------------------------------------------------------


def find_voice_models() -> dict:
    """
    Look around a few dirs for .onnx voice models.
    Returns {pretty_name: Path}.
    """
    voices: dict[str, Path] = {}

    for folder in (PIPER_DIR, VOICES_DIR, APP_DIR):
        if not folder.exists():
            continue
        for onnx_file in folder.rglob("*.onnx"):
            # make a slightly nicer name out of the filename
            name = onnx_file.stem.replace("-", " ").replace("_", " ").title()
            if name not in voices:
                voices[name] = onnx_file

    return voices


def locate_piper() -> list[str] | None:
    """
    Try a bunch of places to find the Piper executable or module.
    This is a bit hacky but OK for personal use.
    """
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    candidates: list[list[str]] = []

    exe = shutil.which("piper")
    if exe:
        candidates.append([exe])

    # a couple of obvious dirs and app-local places
    for loc in (PIPER_DIR, VOICES_DIR, APP_DIR, APP_DIR / "bin", Path.home() / ".local" / "bin"):
        if not loc.exists():
            continue
        for n in ("piper.exe", "piper", "piper.bin"):
            p = loc / n
            if p.is_file():
                candidates.append([str(p)])

    # fallback: python -m
    candidates.append([sys.executable, "-m", "piper"])
    candidates.append([sys.executable, "-m", "piper_tts"])

    # de-dup
    seen: set[tuple[str, ...]] = set()
    uniq: list[list[str]] = []
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            uniq.append(c)
            seen.add(t)

    for cmd in uniq:
        try:
            r = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                timeout=5,
                creationflags=creation_flags,
            )
            if r.returncode == 0:
                return cmd
        except Exception:
            # silently ignore; we'll just say "not found" later
            pass

    return None


def generate_speech(text: str, voice_path: Path, output_path: Path, speed: float = 1.0) -> None:
    """
    Run Piper on the given text and write a wav file.
    """
    # keep piper happy: strip weird unicode stuff
    safe_text = " ".join(text.split()).encode("ascii", errors="ignore").decode("ascii")
    safe_text = " ".join(safe_text.split())
    if not safe_text.strip():
        raise RuntimeError("Nothing left to synthesize after cleaning text.")

    piper_cmd = locate_piper()
    if not piper_cmd:
        raise RuntimeError("Piper CLI not found. Try: pip install piper-tts (or put it on PATH).")

    if not voice_path.exists():
        raise RuntimeError(f"Voice model not found: {voice_path}")

    json_path = voice_path.with_suffix(".onnx.json")
    if not json_path.exists():
        # some models use .json instead
        json_path = Path(str(voice_path).replace(".onnx", ".json"))
    if not json_path.exists():
        raise RuntimeError(f"Voice config not found: {json_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # in piper, "length_scale" is sort of inverse of speed
    length_scale = 1.0 / max(0.5, min(2.0, speed))

    cmd = piper_cmd + [
        "--model", str(voice_path),
        "--config", str(json_path),
        "--output_file", str(output_path),
        "--length_scale", str(length_scale),
        "--sentence_silence", "0.3",
    ]

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    r = subprocess.run(
        cmd,
        input=safe_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        creationflags=creation_flags,
    )

    if r.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {(r.stderr or '')[:300]}")

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Piper TTS produced an empty/tiny file.")


# --- subtitles / timings -----------------------------------------------------


def fake_word_timings(text: str, audio_duration: float):
    """
    Not actual alignment, just evenly spread words over audio length
    with a bit of extra time on punctuation.
    """
    words = text.split()
    if not words or audio_duration <= 0:
        return []

    wps = len(words) / audio_duration
    timings = []
    t = 0.1  # tiny offset so it doesn't start instantly

    for word in words:
        clean = word.strip(".,!?;:'\"")
        base = 1.0 / wps
        factor = max(0.8, min(1.5, 0.8 + (len(clean) / 15.0)))
        dur = base * factor

        if word.endswith((".", "!", "?")):
            dur *= 1.5
        elif word.endswith(","):
            dur *= 1.2

        dur = max(0.15, min(1.2, dur))
        timings.append((word, t, t + dur))
        t += dur

    scale = audio_duration / t if t > 0 else 1.0
    return [(w, s * scale, e * scale) for (w, s, e) in timings]


def seconds_to_ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass_subtitles(text: str, audio_duration: float, output_path: Path) -> None:
    """
    Build an .ass file with a word-by-word highlight effect.
    """
    word_timings = fake_word_timings(text, audio_duration)
    if not word_timings:
        return

    header = [
        "[Script Info]",
        "Title: Subtitles",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Montserrat,95,&H00FFFFFF,&H000000FF,&H00000000,&HB0000000,"
        "-1,0,0,0,100,100,1,0,1,6,3,5,40,40,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def esc(s: str) -> str:
        s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        return s

    chunks: list[list[tuple[str, float, float]]] = []
    cur: list[tuple[str, float, float]] = []

    for word, start, end in word_timings:
        cur.append((word, start, end))
        stripped = word.rstrip(".,!?;:")
        has_punct = word != stripped
        # arbitrary-ish rules so lines aren't too long or too short
        if len(cur) >= 2 and (has_punct or len(cur) >= 3):
            chunks.append(cur)
            cur = []

    if cur:
        chunks.append(cur)

    lines = header[:]
    for chunk in chunks:
        for idx, (word, start, end) in enumerate(chunk):
            parts = []
            for i, (w, _, _) in enumerate(chunk):
                w = esc(w)
                if i == idx:
                    # active word
                    parts.append("{\\c&H00E5FF&\\3c&H0099FF&\\fscx120\\fscy120\\b1\\blur2}" + w + "{\\r}")
                elif i < idx:
                    # already spoken
                    parts.append("{\\c&HDDDDDD&\\alpha&H60&}" + w + "{\\r}")
                else:
                    # upcoming
                    parts.append("{\\c&HFFFFFF&}" + w + "{\\r}")

            st = seconds_to_ass_time(start)
            et = seconds_to_ass_time(end)
            anim = "{\\t(0,80,\\fscx125\\fscy125\\blur3)\\t(80,150,\\fscx120\\fscy120\\blur2)}"
            lines.append(f"Dialogue: 0,{st},{et},Default,,0,0,180,,{anim}{' '.join(parts)}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# --- video composition -------------------------------------------------------


def render_video(story_text: str, audio_path: Path, video_bg: Path, output_path: Path) -> None:
    """
    Use ffmpeg (via imageio-ffmpeg to locate it) to glue everything together.
    """
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install imageio-ffmpeg") from e

    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate) if rate > 0 else 0.0
    except Exception as e:
        raise RuntimeError(f"Failed to read audio {audio_path}: {e}") from e

    if duration <= 0:
        raise RuntimeError(f"Invalid audio duration: {duration}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    subs_path = run_dir / "subs.ass"
    write_ass_subtitles(story_text, duration, subs_path)

    subs_escaped = subs_path.name.replace("\\", "\\\\").replace(":", "\\:")
    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p,ass={subs_escaped}"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-stream_loop", "-1", "-i", str(video_bg),
        "-i", str(audio_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "21",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-r", str(FPS),
        "-movflags", "+faststart",
        str(output_path),
    ]

    r = subprocess.run(cmd, capture_output=True, cwd=str(run_dir))
    if r.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"ffmpeg failed: {(r.stderr or '')[:300]}")


# --- CLI / main --------------------------------------------------------------


def pick_mode() -> str:
    log("")
    log("Choose story type:")
    log("  1) AITA")
    log("  2) Relationship drama")
    log("  3) Mystery / thriller")
    choice = input("Enter 1/2/3 (default 1): ").strip()
    if choice == "2":
        return "relationships"
    if choice == "3":
        return "mystery"
    return "aita"


def main() -> None:
    log(f"App directory : {APP_DIR}")
    log(f"Videos        : {VIDEOS_DIR}")
    log(f"Voices        : {VOICES_DIR}")

    bg_candidates = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov"))
    if not bg_candidates:
        raise SystemExit(f"No background videos found in: {VIDEOS_DIR}")

    bg_video = random.choice(bg_candidates)
    log(f"Background    : {bg_video.name}")

    voices = find_voice_models()
    if not voices:
        raise SystemExit(f"No Piper voices (.onnx) found in: {VOICES_DIR} or {PIPER_DIR}")

    # for now just pick the first one
    voice_name, voice_path = next(iter(voices.items()))
    log(f"Voice         : {voice_name} -> {voice_path.name}")

    mode = pick_mode()
    custom_topic = input("\nCustom topic (optional, press Enter to skip): ").strip() or None
    length = "Medium"  # could make this interactive too, but meh

    log("Generating story...")
    story = ask_ollama_for_story(mode=mode, length=length, custom_topic=custom_topic)

    raw_story = story.get("story", "")
    tts_text = clean_for_tts(raw_story)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{ts}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    audio_path = run_dir / "voice.wav"
    video_path = run_dir / "video.mp4"
    meta_path = run_dir / "meta.json"

    log("Generating TTS...")
    try:
        generate_speech(tts_text, voice_path, audio_path, speed=1.0)
    except Exception as e:
        log(f"TTS failed: {e}")
        raise SystemExit(1)

    log("Rendering video...")
    try:
        render_video(tts_text, audio_path, bg_video, video_path)
    except Exception as e:
        log(f"Video composition failed: {e}")
        raise SystemExit(1)

    meta = {
        "title": story.get("title", ""),
        "caption": story.get("caption", ""),
        "hashtags": story.get("hashtags", ""),
        "story": raw_story,
        "video_path": str(video_path),
        "folder": str(run_dir),
        "mode": mode,
        "length": length,
        "timestamp": ts,
        "word_count": len(raw_story.split()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"Done: {video_path}")
    log(f"Meta: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Cancelled.")
    except Exception as e:
        log(f"Error: {e}")
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        raise
