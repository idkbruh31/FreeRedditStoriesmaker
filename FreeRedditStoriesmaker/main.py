from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import traceback
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = get_app_dir()

VIDEOS_DIR = APP_DIR / "videos"
VOICES_DIR = APP_DIR / "voices"
PIPER_DIR = APP_DIR / "piper"
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

OLLAMA_MODEL = "llama3.1:8b"

CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")
CURRENCY_RE = re.compile(r"\$([0-9]+)")


def log(msg: str) -> None:
    print(msg, flush=True)


def sanitize_json_block(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def create_fallback_story(length: str, custom_topic: Optional[str] = None) -> dict:
    base = "I found something in my partner's phone that changed everything. "
    if custom_topic:
        base = f"{custom_topic.strip()} "

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


def generate_story(mode: str, length: str = "Medium", custom_topic: Optional[str] = None) -> dict:
    word_targets = {"Short": 180, "Medium": 360, "Long": 700}
    target_words = word_targets.get(length, 360)

    if mode == "aita":
        brief = f"""
Write a first-person AITA-style post.

Constraints:
- {target_words} words (roughly).
- Start immediately with the situation and the hook (no greetings).
- Include ages, time frames, a specific setting, and a few lines of dialogue.
- Make it morally grey; both sides should feel plausible.
- Escalate to one decisive moment.
- End on one short line that invites debate.
"""
    elif mode == "relationships":
        brief = f"""
Write a first-person relationship drama.

Constraints:
- {target_words} words (roughly).
- Open with a strong emotional first sentence.
- Modern details (texts, screenshots, calls, money, family pressure, etc.).
- 2–3 lines of dialogue.
- Keep it realistic and specific, not melodramatic.
- End on a sharp final line that makes people argue.
"""
    elif mode == "mystery":
        brief = f"""
Write a grounded first-person mystery/thriller.

Constraints:
- Present-day, realistic (no supernatural).
- {target_words} words (roughly).
- Open with a clear “something is wrong” hook.
- Reveal clues gradually; avoid info-dumps.
- Include a few concrete scenes/locations and some dialogue.
- End with a twist that re-frames earlier details.
"""
    else:
        brief = f"""
Write a high-engagement first-person story.

Constraints:
- {target_words} words (roughly).
- Hook early, build tension, include at least one twist.
- Keep it grounded and plausible.
- End with a line that invites opinions.
"""

    if custom_topic:
        brief += f"\nUse this situation as the core:\n{custom_topic.strip()}\n"

    prompt = f"""{brief}

Output ONLY valid JSON with exactly these keys:
{{
  "title": "...",
  "story": "...",
  "caption": "...",
  "hashtags": "..."
}}"""

    try:
        import ollama  # type: ignore

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON block found")

        data = json.loads(sanitize_json_block(content[start:end]))
        for k in ("title", "story", "caption", "hashtags"):
            if k not in data:
                raise ValueError(f"Missing key: {k}")

        data["ui_type"] = "story"
        return data

    except Exception as e:
        log(f"Story generation failed, using fallback: {e}")
        return create_fallback_story(length, custom_topic)


def prepare_tts_text(raw: str) -> str:
    if not raw:
        return ""
    text = str(raw)
    text = CURRENCY_RE.sub(r"\1 dollars", text)
    text = text.replace("$", " dollars ")
    text = re.sub(r"[?.!]{2,}", " ", text)
    return " ".join(text.split())


def detect_piper_voices() -> Dict[str, Path]:
    voices: Dict[str, Path] = {}
    for search_dir in (PIPER_DIR, VOICES_DIR, APP_DIR):
        if not search_dir.exists():
            continue
        for onnx_file in search_dir.rglob("*.onnx"):
            name = onnx_file.stem.replace("-", " ").replace("_", " ").title()
            voices.setdefault(name, onnx_file)
    return voices


def find_piper_exe() -> Optional[List[str]]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    candidates: List[List[str]] = []

    exe = shutil.which("piper")
    if exe:
        candidates.append([exe])

    for loc in (PIPER_DIR, VOICES_DIR, APP_DIR, APP_DIR / "bin", Path.home() / ".local" / "bin"):
        if not loc.exists():
            continue
        for name in ("piper.exe", "piper", "piper.bin"):
            p = loc / name
            if p.exists() and p.is_file():
                candidates.append([str(p)])

    candidates.append([sys.executable, "-m", "piper"])
    candidates.append([sys.executable, "-m", "piper_tts"])

    seen = set()
    uniq: List[List[str]] = []
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
            pass

    return None


def generate_speech(text: str, voice_path: Path, output_path: Path, speed: float = 1.0) -> bool:
    safe_text = " ".join(text.split()).encode("ascii", errors="ignore").decode("ascii")
    safe_text = " ".join(safe_text.split())
    if not safe_text.strip():
        raise RuntimeError("Nothing left to synthesize after cleaning text.")

    piper_cmd = find_piper_exe()
    if not piper_cmd:
        raise RuntimeError("Piper CLI not found. Install: pip install piper-tts")

    if not voice_path.exists():
        raise RuntimeError(f"Voice model not found: {voice_path}")

    json_path = voice_path.with_suffix(".onnx.json")
    if not json_path.exists():
        json_path = Path(str(voice_path).replace(".onnx", ".json"))
    if not json_path.exists():
        raise RuntimeError(f"Voice config not found: {json_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    if r.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        return True

    raise RuntimeError(f"Piper TTS failed: {(r.stderr or '')[:300]}")


def create_word_timings(text: str, audio_duration: float) -> List[Tuple[str, float, float]]:
    words = text.split()
    if not words or audio_duration <= 0:
        return []

    wps = len(words) / audio_duration
    timings: List[Tuple[str, float, float]] = []
    t = 0.1

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


def seconds_to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_subtitles(text: str, audio_duration: float, output_path: Path) -> None:
    word_timings = create_word_timings(text, audio_duration)
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

    chunks: List[List[Tuple[str, float, float]]] = []
    cur: List[Tuple[str, float, float]] = []

    for word, start, end in word_timings:
        cur.append((word, start, end))
        stripped = word.rstrip(".,!?;:")
        has_punct = word != stripped
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
                    parts.append("{\\c&H00E5FF&\\3c&H0099FF&\\fscx120\\fscy120\\b1\\blur2}" + w + "{\\r}")
                elif i < idx:
                    parts.append("{\\c&HDDDDDD&\\alpha&H60&}" + w + "{\\r}")
                else:
                    parts.append("{\\c&HFFFFFF&}" + w + "{\\r}")

            st = seconds_to_ass_time(start)
            et = seconds_to_ass_time(end)
            anim = "{\\t(0,80,\\fscx125\\fscy125\\blur3)\\t(80,150,\\fscx120\\fscy120\\blur2)}"
            lines.append(f"Dialogue: 0,{st},{et},Default,,0,0,180,,{anim}{' '.join(parts)}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def compose_final_video(story_text: str, audio_path: Path, video_bg: Path, output_path: Path) -> bool:
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
        raise RuntimeError(f"Failed to read audio: {audio_path}: {e}") from e

    if duration <= 0:
        raise RuntimeError(f"Invalid audio duration: {duration}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    subs_path = run_dir / "subs.ass"
    create_subtitles(story_text, duration, subs_path)

    subs_escaped = subs_path.name.replace("\\", "\\\\").replace(":", "\\:")
    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p,ass={subs_escaped}"
    )

    cmd = [
        ffmpeg, "-y",
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
        return False
    return True


def choose_mode() -> str:
    log("\nChoose story type:")
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

    voices = detect_piper_voices()
    if not voices:
        raise SystemExit(f"No Piper voices (.onnx) found in: {VOICES_DIR} or {PIPER_DIR}")

    voice_name, voice_path = next(iter(voices.items()))
    log(f"Voice         : {voice_name} -> {voice_path.name}")

    mode = choose_mode()
    custom_topic = input("\nCustom topic (optional): ").strip() or None
    length = "Medium"

    log("Generating story...")
    story = generate_story(mode=mode, length=length, custom_topic=custom_topic)

    raw_story = story.get("story", "")
    tts_text = prepare_tts_text(raw_story)

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
    ok = compose_final_video(tts_text, audio_path, bg_video, video_path)
    if not ok:
        raise SystemExit("Video composition failed.")

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
