"""Small deterministic teaching datasets. No precomputed scores or model claims."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shutil
import struct
import subprocess
import tempfile
import wave


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    text: str = ""
    kind: str = "text"


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    description: str
    query: str
    expected_key: str
    files: tuple[Example, ...]
    lesson: str
    task: str = "SearchQuery"


EXPERIMENTS = {
    "text": Experiment(
        "text", "Why did the hydraulic pump fail?", "Paraphrase retrieval across four unrelated documents.",
        "Why did the hydraulic pump fail?", "pump", (
            Example("pump", "Equipment incident — hydraulic pump", "A hydraulic pump overheated to 120 degrees Celsius during an offshore maintenance shift. Inspection revealed inadequate lubrication and a clogged cooling line. Engineers attributed the breakdown to insufficient lubrication and heat buildup."),
            Example("coffee", "Coffee shop daily report", "The cafe sold 220 cappuccinos, ordered fresh beans, and repaired a leaking espresso machine. Staff will review milk stock tomorrow."),
            Example("rocket", "Rocket mission briefing", "A rocket mission launched a communications satellite into low Earth orbit. The ground team reviewed telemetry and fuel reserves."),
            Example("security", "Account security policy", "New user accounts require multifactor authentication, password rotation after compromise, and administrator review of unusual sign-ins."),
        ),
        "The pump report should rank ahead of unrelated coffee, rocket, and security records. No answer is generated."),
    "code": Experiment(
        "code", "Find login validation code", "Find code by what it does rather than its function name.",
        "Find the function that validates passwords during login", "authentication", (
            Example("authentication", "authentication.py", "def check_credentials(email, password, stored_hash):\n    if not email or not password:\n        return False\n    return password_hasher.verify(password, stored_hash)"),
            Example("csv", "csv_import.py", "import csv\n\ndef load_rows(path):\n    with open(path, newline='') as handle:\n        return list(csv.DictReader(handle))"),
            Example("math", "arithmetic.py", "def calculate_invoice_total(prices):\n    return round(sum(prices), 2)"),
        ),
        "The authentication example should rank above the CSV and arithmetic examples.", "CodeRetrieval"),
    "image": Experiment(
        "image", "Find a cat in pictures", "Match an English description to unlabeled synthetic illustrations.",
        "A picture of an orange cat", "cat", (
            Example("cat", "Synthetic illustration A", kind="image"),
            Example("coffee", "Synthetic illustration B", kind="image"),
            Example("rocket", "Synthetic illustration C", kind="image"),
        ),
        "Illustration A depicts a cat. These are simple drawings, not photo-quality benchmark data."),
    "audio": Experiment(
        "audio", "Find an alarm sound", "Search two generated non-speech waveforms.",
        "An electronic repeating alarm beep", "alarm", (
            Example("alarm", "Synthetic recording A", kind="audio"),
            Example("rain", "Synthetic recording B", kind="audio"),
        ),
        "Recording A contains repeated beeps; B contains generated noise. Performance is not guaranteed."),
    "pdf": Experiment(
        "pdf", "Find an inspection report", "Search passages extracted from selectable-text PDFs.",
        "Find an inspection explaining overheating caused by blocked cooling lines", "inspection", (
            Example("inspection", "Equipment inspection.pdf", "The inspection team observed a blocked coolant line and insufficient oil on the hydraulic pump. High friction caused the temperature to rise until the machine stopped."),
            Example("cafe", "Cafe operations.pdf", "The cafe stocked coffee beans, audited registers and replaced a milk refrigerator seal."),
        ),
        "The equipment inspection report should rank above the cafe report."),
    "voice": Experiment(
        "voice", "Find the spoken maintenance report", "Search synthetic speech; requires a local speech synthesizer.",
        "Someone talking about a failed pump and overheating", "pump_voice", (
            Example("pump_voice", "Spoken recording A", "The hydraulic pump stopped after overheating caused by poor lubrication.", "voice"),
            Example("coffee_voice", "Spoken recording B", "The coffee shop received a shipment of roasted beans.", "voice"),
        ),
        "Recording A contains speech about the pump. Requires 'espeak' or 'say' plus ffmpeg."),
    "video": Experiment(
        "video", "Find a rocket video", "Find a moving synthetic animation using a text description.",
        "Find a short video of a rocket launch", "rocket_clip", (
            Example("rocket_clip", "Synthetic clip A", kind="video"),
            Example("coffee_clip", "Synthetic clip B", kind="video"),
        ),
        "Clip A contains a static rocket drawing encoded as video, not a moving launch. Requires ffmpeg."),
}


def manifest() -> list[dict]:
    return [{"id": e.id, "title": e.title, "description": e.description, "query": e.query,
             "expected_key": e.expected_key, "lesson": e.lesson, "task": e.task,
             "sample_count": len(e.files), "kinds": sorted(set(f.kind for f in e.files))}
            for e in EXPERIMENTS.values()]


def _image(path: Path, symbol: str) -> None:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (320, 240), (241, 245, 249))
    d = ImageDraw.Draw(im)
    if symbol == "cat":
        d.ellipse((85, 57, 235, 202), fill=(226, 143, 55))
        d.polygon(((105, 80), (106, 28), (146, 61)), fill=(226, 143, 55))
        d.polygon(((190, 61), (214, 28), (220, 89)), fill=(226, 143, 55))
        d.ellipse((129, 117, 140, 134), fill=(38, 35, 35)); d.ellipse((180, 117, 191, 134), fill=(38, 35, 35))
        d.polygon(((156, 141), (164, 141), (160, 151)), fill=(244, 189, 191))
    elif symbol == "coffee":
        d.rounded_rectangle((95, 90, 215, 205), radius=12, fill=(116, 64, 41))
        d.ellipse((95, 75, 215, 115), fill=(63, 38, 29));d.arc((192, 100, 253, 185), 260, 100, fill=(63, 38, 29), width=13)
        for x in (126,160,194): d.arc((x-10,28,x+8,84),170,320, fill=(170,170,170), width=3)
    else:
        d.polygon(((158, 29),(193, 101),(185, 173),(135,173),(127,101)),fill=(190, 50, 57))
        d.ellipse((146,93,174,120),fill=(70,159,197));d.polygon(((140,166),(160,221),(180,166)),fill=(244, 181, 61))
        d.polygon(((134,143),(102,191),(134,173)),fill=(55,72,93));d.polygon(((186,143),(218,191),(186,173)),fill=(55,72,93))
    im.save(path)


def _audio(path: Path, mode: str):
    rate = 16000; dur = 2.5
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(rate)
        frames = bytearray()
        for n in range(int(rate*dur)):
            t=n/rate
            if mode == "alarm":
                v = .5*math.sin(2*math.pi*850*t) if int(t*3)%2==0 else 0
            else:
                # deterministic mixture, not a claim to be a real rain recording
                v = .22 * (math.sin(2*math.pi*1777*t) + math.sin(2*math.pi*3109*t))
            frames.extend(struct.pack('<h', int(max(-1,min(1,v))*25000)))
        wf.writeframes(frames)


def _pdf(path: Path, text: str):
    # Minimal valid single-page text PDF, fully offline and dependency-free.
    text = text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
    streams = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    data = f'BT /F1 11 Tf 45 700 Td ({text[:340]}) Tj ET'.encode('latin1', errors='replace')
    streams.append(f'<< /Length {len(data)} >>\nstream\n'.encode()+data+b'\nendstream')
    pdf = bytearray(b'%PDF-1.4\n'); offsets=[0]
    for idx, obj in enumerate(streams,1):
        offsets.append(len(pdf));pdf.extend(f'{idx} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(pdf);pdf.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for pos in offsets[1:]:pdf.extend(f'{pos:010d} 00000 n \n'.encode())
    pdf.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    path.write_bytes(pdf)


def create_media(exp_id: str, example: Example, target: Path) -> Path:
    """Create a media example in a controlled temp dir; no network or user input."""
    target.mkdir(parents=True,exist_ok=True)
    if example.kind == 'image':
        path=target/f'{exp_id}_{example.key}.png';_image(path, example.key);return path
    if example.kind == 'audio':
        path=target/f'{exp_id}_{example.key}.wav';_audio(path,example.key);return path
    if exp_id == 'pdf':
        path=target/f'{example.key}.pdf';_pdf(path,example.text);return path
    if example.kind == 'voice':
        espeak=shutil.which('espeak') or shutil.which('espeak-ng')
        ffmpeg=shutil.which('ffmpeg')
        if not espeak and not (shutil.which('say') and ffmpeg):
            raise RuntimeError('Voice experiment needs espeak/espeak-ng or macOS say + ffmpeg')
        path=target/f'{example.key}.wav'
        if espeak:
            subprocess.run([espeak,'-w',str(path),example.text],check=True,capture_output=True,timeout=20)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                aiff=Path(tmp)/'voice.aiff'
                subprocess.run(['say','-o',str(aiff),example.text],check=True,capture_output=True,timeout=20)
                subprocess.run([ffmpeg,'-nostdin','-y','-loglevel','error','-i',str(aiff),'-ac','1','-ar','16000',str(path)],check=True,capture_output=True,timeout=20)
        return path
    if example.kind == 'video':
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise RuntimeError('Video experiment needs ffmpeg on PATH')
        still=target/f'{example.key}.png';_image(still,'rocket' if example.key=='rocket_clip' else 'coffee')
        path=target/f'{example.key}.mp4'
        subprocess.run([ffmpeg,'-nostdin','-y','-loglevel','error','-loop','1','-framerate','1','-i',str(still),'-t','2','-vf','format=yuv420p','-c:v','mpeg4',str(path)],check=True,capture_output=True,timeout=30)
        return path
    raise ValueError('Unknown example type')