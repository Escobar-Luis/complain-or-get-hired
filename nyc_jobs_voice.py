"""Voice front end for the NYC jobs agent.

Speak a complaint about NYC, hear the answer.
  .venv/bin/python nyc_jobs_voice.py                       # loop: Enter, talk, listen
  .venv/bin/python nyc_jobs_voice.py --text "rats in Astoria" --once
Speech in: Mistral Voxtral. Speech out: the macOS `say` voice.
"""
import argparse
import os
import select
import signal
import shutil
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from mistralai.client import Mistral

from nyc_jobs_agent import ask

HERE = os.path.dirname(os.path.abspath(__file__))
STT_MODELS = ["voxtral-mini-latest", "voxtral-mini-2507"]
REWRITE_MODEL = "mistral-small-latest"
MIC_HELP = ("Mic not available: System Settings > Privacy & Security > Microphone, "
            "allow your terminal app, then restart it. Falling back to typing.")


def _client() -> Mistral:
    load_dotenv(os.path.join(HERE, ".env"))
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def record(seconds: int = 8) -> str | None:
    """Record 16 kHz mono wav until Enter is pressed again or `seconds` pass.
    Returns the path, "" if rec ran but caught no sound, None if rec failed.
    Enter-to-stop, not sox's silence trigger: on the MacBook mic the room hum
    (0.45% RMS) sits right at the speech threshold, so auto start/stop was flaky."""
    rec = shutil.which("rec") or "/opt/homebrew/bin/rec"
    if not os.path.exists(rec):
        return None
    path = os.path.join(tempfile.gettempdir(), "nyc_voice_clip.wav")
    if os.path.exists(path):
        os.remove(path)
    cmd = [rec, "-q", "-V1", "-r", "16000", "-c", "1", "-b", "16", path, "trim", "0", str(seconds)]
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    except OSError:
        return None
    while proc.poll() is None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()
            proc.send_signal(signal.SIGINT)  # sox closes the wav cleanly on SIGINT
            break
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if proc.returncode not in (0, -signal.SIGINT, 130) and not os.path.exists(path):
        return None
    # A bare wav header is 44 bytes; anything this small has no audio.
    if not os.path.exists(path) or os.path.getsize(path) <= 4000:
        return ""
    return path


def transcribe(client: Mistral, path: str) -> str:
    last = None
    for model in STT_MODELS:
        try:
            with open(path, "rb") as f:
                out = client.audio.transcriptions.complete(
                    model=model, file={"content": f, "file_name": "clip.wav"})
            return (out.text or "").strip()
        except Exception as err:  # model id rejected: try the next one
            last = err
    raise RuntimeError(f"Voxtral transcription failed: {last}")


def spoken_version(client: Mistral, answer: str) -> str:
    try:
        resp = client.chat.complete(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content":
                       "Rewrite this answer as 3 spoken sentences for a voice assistant, "
                       "keep every number and the job title, no bullets, no markdown.\n\n" + answer}],
        )
        text = resp.choices[0].message.content
    except Exception:
        text = answer
    text = " ".join(str(text).replace("*", "").replace("#", "").split())
    if len(text) > 600:
        cut = text[:600]
        text = cut[: cut.rfind(".") + 1] if "." in cut else cut
    return text


def speak(text: str) -> None:
    if text and shutil.which("say"):
        subprocess.run(["say", "-r", "190", text])


def turn(client: Mistral, question: str) -> None:
    print(f"\nYou: {question}")
    print("thinking...")
    answer = ask(question)
    print("\nspeaking...")
    speak(spoken_version(client, answer))


def get_question(client: Mistral, use_mic: bool) -> tuple[str | None, bool]:
    """Returns (question, use_mic). use_mic flips off if the mic fails."""
    if use_mic:
        input("\nPress Enter, then speak... ")
        print("listening... press Enter when done (stops by itself at 8 s)")
        path = record()
        if path:
            text = transcribe(client, path)
            print(f"[heard] {text or '(nothing)'}")
            return text, True
        if path == "":
            print("[heard] nothing. Talk a bit longer. If the mic is blocked: "
                  "System Settings > Privacy & Security > Microphone.")
            return "", True
        print(MIC_HELP)
        use_mic = False
    return input("\nType your complaint (or quit): ").strip(), use_mic


def main() -> None:
    p = argparse.ArgumentParser(description="Speak a complaint about NYC, hear the answer.")
    p.add_argument("--text", help="skip the mic and use this text")
    p.add_argument("--once", action="store_true", help="do one turn and exit")
    p.add_argument("--type", action="store_true", help="type instead of using the mic")
    args = p.parse_args()

    client = _client()
    if args.text:
        turn(client, args.text)
        if args.once:
            return
    use_mic = not args.type
    try:
        while True:
            q, use_mic = get_question(client, use_mic)
            if not q:
                print("Heard nothing. Try again.")
                continue
            if q.lower().strip(" .!?") in ("quit", "exit", "stop"):
                break
            turn(client, q)
            if args.once:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nbye")


if __name__ == "__main__":
    main()
