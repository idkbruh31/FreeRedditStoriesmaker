import os
import sys
import json
import re
import random
import subprocess
import shutil
import traceback
import wave
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ===================== PATHS & CONSTANTS =====================

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

BASE_DIR = APP_DIR

VIDEOS_DIR = APP_DIR / "videos"   # put your gameplay .mp4/.mov here
VOICES_DIR = APP_DIR / "voices"   # put piper .onnx + .json here
PIPER_DIR = APP_DIR / "piper"     # optional extra search folder
OUTPUT_DIR = APP_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

# Your local Ollama model name
OLLAMA_MODEL = "llama3.1:8b"

# ===================== ASCII BANNER =====================

def print_banner():
    # enable ANSI colors on Windows 10+ (no-op elsewhere)
    if os.name == "nt":
        os.system("")

    RED = "\033[91m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    banner = r"""
--     _______   ________  _______   _______   ______  ________ 
--    /       \ /        |/       \ /       \ /      |/        |
--    $$$$$$$  |$$$$$$$$/ $$$$$$$  |$$$$$$$  |$$$$$$/ $$$$$$$$/ 
--    $$ |__$$ |$$ |__    $$ |  $$ |$$ |  $$ |  $$ |     $$ |   
--    $$    $$< $$    |   $$ |  $$ |$$ |  $$ |  $$ |     $$ |   
--    $$$$$$$  |$$$$$/    $$ |  $$ |$$ |  $$ |  $$ |     $$ |   
--    $$ |  $$ |$$ |_____ $$ |__$$ |$$ |__$$ | _$$ |_    $$ |   
--    $$ |  $$ |$$       |$$    $$/ $$    $$/ / $$   |   $$ |   
--    $$/   $$/ $$$$$$$$/ $$$$$$$/  $$$$$$$/  $$$$$$/    $$/    
--                                                              
--                                                              
--                                                                                                                                                                            

     R E D D I T   S T O R Y   M A K E R
             by  timi1q2w3e
"""

    top_border = CYAN + "#" * 60 + RESET
    bottom_border = CYAN + "#" * 60 + RESET

    print(top_border)
    print(RED + banner + RESET)
    print(bottom_border)

    # simple fake loading bar (cheap "boot" vibe)
    print()
    for i in range(0, 21):
        pct = i * 5
        bar = ("█" * i) + ("░" * (20 - i))
        color = MAGENTA if i < 20 else CYAN
        sys.stdout.write(color + f"\r[ {bar} ] {pct:3d}%  booting D E M O build..." + RESET)
        sys.stdout.flush()
        time.sleep(0.03)
    print("\n")


# ===================== SMALL HELPERS =====================

def sanitize_json_block(text: str) -> str:
    """Remove control chars so json.loads has a chance."""
    return re.sub(r"[\x00-\x1F\x7F]", "", text)


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
        "caption": "This changed everything 💔 #storytime",
        "hashtags": "storytime drama relationships viral fyp trending",
        "ui_type": "story",
    }


def generate_story(
    mode: str,
    length: str = "Medium",
    custom_topic: Optional[str] = None,
) -> dict:
    """Generate a Reddit-style story via Ollama or fallback."""
    word_targets = {"Short": 180, "Medium": 360, "Long": 700}
    target_words = word_targets.get(length, 360)

    if mode == "aita":
        base = f"""You are ghostwriting a viral Reddit "Am I The Asshole?" (AITA) post that will be narrated on TikTok.

Requirements:
- First-person POV, very conversational, like someone ranting to a friend.
- About {target_words} words.
- Start IMMEDIATELY with the hook (e.g. "AITA for ...?" or "I (26F) did..."). No greetings, no meta talk.
- Include concrete ages, time frames, locations and a few short bits of dialogue.
- Make the situation morally grey so the comments can argue both sides.
- Build tension, escalate the conflict, then reveal one decisive moment.
- End on a short, punchy last line that naturally makes people comment who was right.
- Never say you're an AI, never talk about Reddit, TikTok or storytelling structure. Just tell the story."""
    elif mode == "relationships":
        base = f"""You are writing a first-person relationship drama story that will be read aloud on TikTok.

Requirements:
- Modern, realistic situation: cheating, betrayal, money, family drama, or a shocking revelation.
- About {target_words} words.
- Start with a strong emotional hook in the first sentence.
- Include specific ages, time frames, details (screenshots, texts, calls, etc.) and 2–3 lines of dialogue.
- Show the MC's internal conflict so viewers can empathise even if they disagree.
- End on a sharp line that makes people want to type paragraphs in the comments.
- Do NOT mention Reddit, TikTok, or that this is a story. Just speak like a real person."""
    elif mode == "mystery":
        base = f"""You are writing a grounded mystery/thriller story in first person for TikTok narration.

Requirements:
- Set in the present day, no fantasy or supernatural powers.
- About {target_words} words.
- Start with a hook that implies something is seriously wrong.
- Drip-feed clues and red flags, keep tension rising, and avoid info-dumps.
- Include specific scenes, locations, and at least a little dialogue.
- End with a sharp twist that re-contextualises earlier details.
- No meta commentary, no "dear reader" stuff – just pure story."""
    else:
        base = f"""You are writing a high-engagement first-person short story for TikTok narration.

Requirements:
- About {target_words} words.
- Start with a hook, build tension, add at least one twist.
- Make it realistic and grounded.
- End with a line that makes people comment their opinion."""

    if custom_topic:
        base += (
            "\n\nUse this as the core situation, but still follow the rules above:\n"
            f"\"{custom_topic.strip()}\""
        )

    user_prompt = f"""{base}

Return ONLY valid JSON in this exact format, with no explanation and no extra text:

{{
  "title": "Engaging title for TikTok",
  "story": "The full story text meant to be narrated out loud, as one continuous story.",
  "caption": "Short TikTok caption with some emojis",
  "hashtags": "storytime drama fyp viral ..."
}}"""

    try:
        import ollama  # type: ignore

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": user_prompt}]
        )
        content = response["message"]["content"]

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON block found in Ollama response")

        json_block = sanitize_json_block(content[start:end])
        data = json.loads(json_block)

        for key in ["title", "story", "caption", "hashtags"]:
            if key not in data:
                raise ValueError(f"Missing key: {key}")

        data["ui_type"] = "story"
        return data

    except Exception as e:
        print("Ollama story generation failed, using fallback:", e)
        traceback.print_exc()
        return create_fallback_story(length, custom_topic)


# ===================== TTS (PIPER) =====================

CURRENCY_RE = re.compile(r"\$([0-9]+)")


def prepare_tts_text(raw: str) -> str:
    if not raw:
        return ""

    text = str(raw)

    # Replace money like $8000 with "8000 dollars"
    text = CURRENCY_RE.sub(r"\1 dollars", text)
    text = text.replace("$", " dollars ")

    # Collapse multi punctuation
    text = re.sub(r"[?.!]{2,}", " ", text)
    text = " ".join(text.split())
    return text


def detect_piper_voices() -> Dict[str, Path]:
    voices: Dict[str, Path] = {}
    search_dirs = [PIPER_DIR, VOICES_DIR, BASE_DIR]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for onnx_file in search_dir.rglob("*.onnx"):
            filename = onnx_file.stem
            display_name = filename.replace("-", " ").replace("_", " ").title()
            if display_name not in voices:
                voices[display_name] = onnx_file
    return voices


def find_piper_exe() -> Optional[List[str]]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

    candidates: List[List[str]] = []

    exe = shutil.which("piper")
    if exe:
        candidates.append([exe])

    search_locations = [
        PIPER_DIR,
        VOICES_DIR,
        BASE_DIR,
        BASE_DIR / "bin",
        Path.home() / ".local" / "bin",
    ]
    for loc in search_locations:
        if not loc.exists():
            continue
        for name in ["piper.exe", "piper", "piper.bin"]:
            exe_path = loc / name
            if exe_path.exists() and exe_path.is_file():
                candidates.append([str(exe_path)])

    candidates.append([sys.executable, "-m", "piper"])
    candidates.append([sys.executable, "-m", "piper_tts"])

    seen = set()
    uniq: List[List[str]] = []
    for cmd in candidates:
        key = tuple(cmd)
        if key not in seen:
            uniq.append(cmd)
            seen.add(key)

    for cmd in uniq:
        try:
            result = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                timeout=5,
                creationflags=creation_flags,
            )
            if result.returncode == 0:
                return cmd
        except Exception:
            continue

    return None


def generate_speech(text: str, voice_path: Path, output_path: Path, speed: float = 1.0) -> bool:
    text = " ".join(text.split())
    safe_text = text.encode("ascii", errors="ignore").decode("ascii")
    safe_text = " ".join(safe_text.split())
    if not safe_text.strip():
        raise RuntimeError("Nothing left to synthesize after cleaning text for TTS.")

    piper_cmd = find_piper_exe()
    if not piper_cmd:
        raise RuntimeError(
            "Piper TTS CLI not found.\n"
            "Fix:\n"
            "  1) Activate your venv.\n"
            "  2) pip install piper-tts\n"
            "  3) Make sure 'piper' is on PATH or usable via 'python -m piper'"
        )

    if not voice_path.exists():
        raise RuntimeError(f"Voice model file not found: {voice_path}")

    json_path = voice_path.with_suffix(".onnx.json")
    if not json_path.exists():
        json_path = Path(str(voice_path).replace(".onnx", ".json"))
    if not json_path.exists():
        raise RuntimeError(f"Voice config not found next to model: {json_path}")

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

    print(">> Running Piper:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        input=safe_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        creationflags=creation_flags,
    )

    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        print(f"✓ TTS generated: {output_path} ({output_path.stat().st_size} bytes)")
        return True

    print("Piper stderr:", (result.stderr or "")[:400])
    raise RuntimeError("Piper TTS failed")


# ===================== SUBTITLES – ORIGINAL PREMIUM VERSION =====================

def create_word_timings(text: str, audio_duration: float) -> List[tuple]:
    words = text.split()
    if not words or audio_duration <= 0:
        return []

    wps = len(words) / audio_duration
    timings = []
    current_time = 0.1

    for word in words:
        clean = word.strip(".,!?;:'\"")
        base_dur = 1.0 / wps
        length_factor = 0.8 + (len(clean) / 15.0)
        length_factor = max(0.8, min(1.5, length_factor))
        dur = base_dur * length_factor

        if word.endswith((".", "!", "?")):
            dur *= 1.5
        elif word.endswith(","):
            dur *= 1.2

        dur = max(0.15, min(1.2, dur))
        timings.append((word, current_time, current_time + dur))
        current_time += dur

    if current_time > 0:
        scale = audio_duration / current_time
        timings = [(w, s * scale, e * scale) for w, s, e in timings]

    return timings


def seconds_to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_subtitles(text: str, audio_duration: float, output_path: Path):
    """
    Original premium-style word-by-word highlight subtitles
    copied from the GUI script.
    """
    word_timings = create_word_timings(text, audio_duration)
    if not word_timings:
        return

    header = [
        "[Script Info]",
        "Title: Premium Subtitles",
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
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    lines = header.copy()
    chunks = []
    current_chunk = []

    for word, start, end in word_timings:
        current_chunk.append((word, start, end))
        stripped = word.rstrip(".,!?;:")
        has_punct = word != stripped

        if len(current_chunk) >= 2 and (has_punct or len(current_chunk) >= 3):
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    def _ass_escape_local(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N")
        return text

    for chunk_words in chunks:
        for word_idx, (word, start, end) in enumerate(chunk_words):
            parts = []
            for i, (w, _, _) in enumerate(chunk_words):
                w_esc = _ass_escape_local(w)
                if i == word_idx:
                    parts.append(
                        "{\\c&H00E5FF&\\3c&H0099FF&\\fscx120\\fscy120\\b1\\blur2}"
                        f"{w_esc}{{\\r}}"
                    )
                elif i < word_idx:
                    parts.append("{\\c&HDDDDDD&\\alpha&H60&}" + w_esc + "{\\r}")
                else:
                    parts.append("{\\c&HFFFFFF&}" + w_esc + "{\\r}")

            subtitle_text = " ".join(parts)
            st = seconds_to_ass_time(start)
            et = seconds_to_ass_time(end)
            anim = "{\\t(0,80,\\fscx125\\fscy125\\blur3)\\t(80,150,\\fscx120\\fscy120\\blur2)}"
            lines.append(
                f"Dialogue: 0,{st},{et},Default,,0,0,180,,{anim}{subtitle_text}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _ass_escape(text: str) -> str:
    # kept for compatibility (not used here, but harmless)
    if not text:
        return ""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\n", "\\N")
    return text


# ===================== VIDEO COMPOSITION =====================

def compose_final_video(
    story_text: str,
    audio_path: Path,
    video_bg: Path,
    output_path: Path,
) -> bool:
    """
    Very simple: loop gameplay, add ORIGINAL premium subtitles, mux with TTS audio.
    No overlays, no logos, no music. Free/crippled but still looks clean.
    Uses only stdlib + imageio-ffmpeg (no moviepy).
    """
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "imageio-ffmpeg is required.\n"
            "Install with:\n"
            "  pip install imageio-ffmpeg"
        ) from e

    # --- get audio duration using the wave module ---
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate) if rate > 0 else 0.0
    except Exception as e:
        raise RuntimeError(f"Failed to read audio duration from {audio_path}: {e}") from e

    if duration <= 0:
        raise RuntimeError(f"Invalid audio duration ({duration}s) for {audio_path}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # create original premium ASS subtitles
    subs_path = run_dir / "subs.ass"
    create_subtitles(story_text, duration, subs_path)

    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p,ass={subs_path.name}"
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

    print(">> Running ffmpeg to render video...")
    result = subprocess.run(cmd, capture_output=True, cwd=str(run_dir))
    if result.returncode != 0 or not output_path.exists():
        print("ffmpeg stderr:", (result.stderr or b"")[:400])
        return False

    print(f"✓ Final video: {output_path}")
    return True


# ===================== CLI =====================

def choose_mode() -> str:
    print("\nChoose story type (Reddit-style):")
    print("  1) AITA (Am I The Asshole?)  [default]")
    print("  2) Relationship drama")
    print("  3) Mystery / thriller")

    choice = input("Enter 1/2/3 (default 1): ").strip()
    if choice == "2":
        return "relationships"
    if choice == "3":
        return "mystery"
    return "aita"


def main():
    print_banner()

    CYAN = "\033[96m"
    RESET = "\033[0m"

    print(CYAN + "============================================================" + RESET)
    print(CYAN + "  REDDIT STORY MAKER – FREE VERSION (crippled demo)" + RESET)
    print(CYAN + "============================================================" + RESET)
    print(f"App directory   : {APP_DIR}")
    print(f"Videos folder   : {VIDEOS_DIR}")
    print(f"Voices folder   : {VOICES_DIR}")
    print()

    # Check background videos
    bg_candidates = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov"))
    if not bg_candidates:
        print("ERROR: No background videos found.")
        print(f"Put at least one .mp4 or .mov into: {VIDEOS_DIR}")
        sys.exit(1)

    bg_video = random.choice(bg_candidates)
    print(f"Using background: {bg_video.name}")

    # Check piper voices
    voices = detect_piper_voices()
    if not voices:
        print("ERROR: No Piper voices (.onnx) found.")
        print(f"Put at least one model into: {VOICES_DIR} or {PIPER_DIR}")
        sys.exit(1)

    # Pick first voice automatically (no real customization in free version)
    voice_name, voice_path = next(iter(voices.items()))
    print(f"Using voice model: {voice_name} -> {voice_path.name}")
    print()

    mode = choose_mode()
    custom_topic = input("\nCustom topic / prompt (optional, press Enter to skip): ").strip()
    if not custom_topic:
        custom_topic = None

    # For the free version, we just lock length to 'Medium'
    length = "Medium"

    print("\n>> Generating story with Ollama...")
    story = generate_story(mode=mode, length=length, custom_topic=custom_topic)
    raw_tts_text = story.get("story", "")
    tts_text = prepare_tts_text(raw_tts_text)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{ts}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    audio_path = run_dir / "voice.wav"
    video_path = run_dir / "video.mp4"
    meta_path = run_dir / "meta.json"

    print(">> Generating TTS with Piper...")
    try:
        generate_speech(tts_text, voice_path, audio_path, speed=1.0)
    except Exception as e:
        print("TTS failed:", e)
        sys.exit(1)

    print(">> Rendering video...")
    ok = compose_final_video(
        story_text=tts_text,
        audio_path=audio_path,
        video_bg=bg_video,
        output_path=video_path,
    )
    if not ok:
        print("ERROR: Video composition failed.")
        sys.exit(1)

    meta = {
        "title": story.get("title", ""),
        "caption": story.get("caption", ""),
        "hashtags": story.get("hashtags", ""),
        "story": story.get("story", ""),
        "video_path": str(video_path),
        "folder": str(run_dir),
        "mode": mode,
        "length": length,
        "timestamp": ts,
        "word_count": len(story.get("story", "").split()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Video saved to: {video_path}")
    print(f"Metadata saved to: {meta_path}")
    print("This is the free crippled version. The premium build is where the real control lives.")


if __name__ == "__main__":
    main()
