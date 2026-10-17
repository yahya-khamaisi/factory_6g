#!/usr/bin/env python3
"""Generate a listen-along narration (text + AAC audio) for one thesis chapter.

Unlike the PDF exports, this runs on the **host** (macOS): it uses the built-in
``say`` synthesiser. It does *not* go through ``pdftotext`` -- it reads the
chapter TeX source directly, so there are no hyphenation or page-break
artifacts.

    python3 scripts/tools/export_chapter_narration.py            # chapter 1
    python3 scripts/tools/export_chapter_narration.py --chapter 2
    python3 scripts/tools/export_chapter_narration.py --voice Serena --text-only

Outputs (in ``thesis/exports/``):
    NN_<name>_narration.txt   the spoken script
    NN_<name>.m4a             AAC mono 22.05 kHz, matching the previous exports
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS_DIR = ROOT / "thesis" / "chapters"
EXPORTS_DIR = ROOT / "thesis" / "exports"

# chNN stem -> (export prefix, chapter number)
CHAPTERS = {
    "ch01_introduction": ("01_introduction", 1),
    "ch02_literature_review": ("02_literature_review", 2),
    "ch03_methodology": ("03_methodology", 3),
    "ch04_results": ("04_results", 4),
    "ch05_discussion": ("05_discussion", 5),
    "ch06_conclusions": ("06_conclusions", 6),
}

# Cross-reference labels that point outside the chapter being narrated. The
# "Chapter~" / "Figure~" / "Appendix~" word already sits in the source, so these
# expand to the bare number/letter only.
STATIC_REFS = {
    "ch:introduction": "1",
    "ch:literature": "2",
    "ch:methodology": "3",
    "ch:results": "4",
    "ch:discussion": "5",
    "ch:conclusions": "6",
    "app:config": "A",
    "app:stopping": "A",
    "fig:lr-taxonomy": "2.1",
    "fig:fading-scenarios": "2.2",
    "fig:ce-tradeoff": "2.3",
    "fig:drl-rm-loop": "2.4",
    "fig:lr-eval-coupling": "2.5",
    "tab:lr-survey-matrix": "2.1",
}

# Spoken forms for acronyms and symbols. Applied as whole-word replacements,
# longest key first. Keep MAC / cobot as-is: ``say`` already pronounces them the
# way engineers do.
SPOKEN = {
    "E_b/N_0": "E b over N naught",
    "AI-native": "A-I-native",
    "AI": "A I",
    "PHY/MAC": "physical layer and MAC",
    "PHY-to-MAC": "physical layer to MAC",
    "PHY--MAC": "physical layer and MAC",
    "PHY-only": "physical-layer-only",
    "PHY": "physical layer",
    "URLLC": "U R L L C",
    "B5G": "beyond 5 G",
    "5G": "5 G",
    "6G": "6 G",
    "3GPP": "3 G P P",
    "TR~38.901": "T R 38.901",
    "TR 38.901": "T R 38.901",
    "UMi": "U M i",
    "OFDM": "O F D M",
    "QPSK": "Q P S K",
    "16-QAM": "16 Q A M",
    "64-QAM": "64 Q A M",
    "QAM": "Q A M",
    "AWGN": "A W G N",
    "CSI": "C S I",
    "BER": "bit error rate",
    "SNR": "signal-to-noise ratio",
    "LOS": "line of sight",
    "NLOS": "non line of sight",
    "LMMSE": "L M M S E",
    "WMMSE": "W M M S E",
    "MMSE": "M M S E",
    "DFT": "D F T",
    "DRL": "D R L",
    "AGVs": "A G Vs",
    "AGV": "A G V",
    "UTs": "user terminals",
    "UT": "user terminal",
    "PF": "P F",
    "LS": "L S",
    "RQ1": "R Q 1",
    "RQ2": "R Q 2",
    "RQ3": "R Q 3",
    "RQs": "research questions",
    "C1": "C 1",
    "C2": "C 2",
    "C3": "C 3",
    "C4": "C 4",
    "P1": "P 1",
    "P2": "P 2",
    "P3": "P 3",
    "P4": "P 4",
    "P5": "P 5",
    "IIoT": "industrial I O T",
    "MIMO-OFDM": "M I M O O F D M",
    "MIMO": "M I M O",
    "MSE": "mean-square error",
    "InF": "indoor factory",
    "RFID": "R F I D",
    "LTE": "L T E",
    "CNN": "C N N",
    "GNN": "G N N",
    "NOMA": "N O M A",
    "AMC": "adaptive modulation and coding",
    "UAV": "U A V",
    "eMBB": "E M B B",
    "mMTC": "M M T C",
    "CQI": "C Q I",
    "QoS": "quality of service",
    "LDAMP": "L D A M P",
    "NP-hard": "N P-hard",
    "TR~22.804": "T R 22.804",
    "TR 22.804": "T R 22.804",
    "GPU": "G P U",
    "RM": "resource management",
    "CE": "channel estimation",
    "RL": "reinforcement learning",
}


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        if re.match(r"\s*%", line):
            continue
        # inline comment: unescaped %
        line = re.sub(r"(?<!\\)%.*$", "", line)
        out.append(line)
    return "\n".join(out)


def brace_arg(s: str, start: int) -> tuple[str, int]:
    """Return (contents, index-after-closing-brace) for a {...} beginning at s[start]=='{'."""
    assert s[start] == "{"
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def extract_environment(tex: str, name: str, start: int) -> tuple[str, int]:
    """Return (body, index-after-\\end{name}) for the env opening at start."""
    open_tag = f"\\begin{{{name}}}"
    close_tag = f"\\end{{{name}}}"
    body_start = start + len(open_tag)
    end = tex.index(close_tag, body_start)
    return tex[body_start:end], end + len(close_tag)


def clean_inline(text: str, label_numbers: dict[str, str]) -> str:
    """Resolve refs, drop citations and formatting, keep readable prose."""
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:label|index)\{[^}]*\}", "", text)

    def ref_sub(m: re.Match) -> str:
        key = m.group(1)
        if key in label_numbers:
            return label_numbers[key]
        return STATIC_REFS.get(key, "")

    text = re.sub(r"\\(?:ref|cref|Cref|autoref)\{([^}]*)\}", ref_sub, text)

    text = text.replace("~", " ")
    text = text.replace(r"15\,\text{m}\times15\,\text{m}", "15 metres by 15 metres")
    text = text.replace(r"\,\textmu s", " microseconds")
    text = text.replace(r"\textmu s", " microseconds")
    text = re.sub(r"\\(?:emph|textbf|textit|textrm|texttt|mbox|text)\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\%", " percent")
    text = text.replace("$E_b/N_0$", "E_b/N_0")  # tokenised, expanded later
    text = re.sub(r"\$\\hat\{\\mathbf\{([A-Za-z])\}\}\$", r"\1-hat", text)
    text = re.sub(r"\$1-10\^\{-(\d+)\}\$", r"1 minus 10 to the minus \1", text)
    # inline maths that appears in captions: $K=1$, $M=10$, $|h|^2$, $10^{-M/10}$
    text = re.sub(r"\$([A-Za-z])\s*=\s*([\d.]+)\$", r"\1 equal to \2", text)
    text = re.sub(r"\|([A-Za-z])\|\^\{?2\}?", r"the squared magnitude of \1", text)
    text = re.sub(r"10\^\{-([A-Za-z])/(\d+)\}", r"10 to the minus \1 over \2", text)
    text = text.replace(" < ", " less than ").replace(" > ", " greater than ")
    text = re.sub(r"\$10\^\{-(\d+)\}\$", r"10 to the minus \1", text)
    text = re.sub(r"\$(\d+(?:\.\d+)?)\\times10\^\{(\d+)\}\$", r"\1 times 10 to the \2", text)
    text = re.sub(r"\$\\mathcal\{O\}\(([^)]+)\)\$", r"order \1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"10\^\{-(\d+)\}", r"10 to the minus \1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\\times10\^\{(\d+)\}", r"\1 times 10 to the \2", text)
    text = re.sub(r"10\^\{(\d+)\}", r"10 to the \1", text)
    text = re.sub(r"\\hat\{\\mathbf\{([A-Za-z])\}\}", r"\1-hat", text)
    text = re.sub(r"\\mathcal\{O\}\(([^)]+)\)", r"order \1", text)
    text = text.replace(r"M\log M", "M log M")
    text = re.sub(r"\\sqrt\{([^}]*)\}", r"square root of \1", text)
    # Greek letters keep their spoken name instead of vanishing with the generic \cmd strip
    text = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|kappa|lambda|mu|nu|xi|rho|sigma|tau|phi|chi|psi|omega)\b",
                  lambda m: m.group(1), text)
    text = text.replace(r"\,", " ")
    text = text.replace(r"\times", " by ")
    text = text.replace(r"\sim", " about ")
    text = re.sub(r"\\(?:cvHi|cvMed|cvLo|cvNone)\b", "", text)
    text = text.replace(r"\newline", " ")
    # remaining generic \cmd{arg} -> arg ; \cmd -> ""
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")

    # compound / range dashes before the generic em-dash conversion
    text = text.replace("PHY--MAC", "physical layer and MAC")
    text = text.replace("reliability--throughput--latency", "reliability, throughput, and latency")
    text = re.sub(r"reliability--(throughput|latency)", r"reliability-versus-\1", text)
    text = re.sub(r"\b([A-Za-z]*\d)--([A-Za-z]*\d)\b", r"\1 to \2", text)  # C1--C4, RQ1--RQ3, 2--4
    text = text.replace("---", " \u2014 ").replace("--", " \u2014 ")
    text = re.sub(r"\s+\u2014\s+", " \u2014 ", text)

    # roman-numeral list markers -> words
    romans = {"i": "first", "ii": "second", "iii": "third", "iv": "fourth", "v": "fifth"}
    text = re.sub(r"\((i|ii|iii|iv|v)\)~?\s*",
                  lambda m: romans[m.group(1)] + ", ", text)

    # tidy punctuation left by removed citations
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def apply_spoken(text: str) -> str:
    # ranges: C1--C4 already had -- turned into em dash; handle "X to Y" forms
    text = re.sub(r"\bRQ1\s*\u2014\s*RQ3\b", "R Q 1 to R Q 3", text)
    text = re.sub(r"\bC1\s*\u2014\s*C4\b", "C 1 to C 4", text)
    text = re.sub(r"\bC1\s*\u2014\s*C3\b", "C 1 to C 3", text)
    text = re.sub(r"\bP1\s*\u2014\s*P5\b", "P 1 to P 5", text)
    text = re.sub(r"\bP1--P5\b", "P 1 to P 5", text)
    for key in sorted(SPOKEN, key=len, reverse=True):
        pattern = re.escape(key)
        if key[0].isalnum():
            pattern = r"\b" + pattern
        if key[-1].isalnum():
            pattern = pattern + r"\b"
        text = re.sub(pattern, SPOKEN[key].replace("\\", r"\\"), text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _collapse_redundant_paren(m: re.Match) -> str:
    before, inner = m.group(1), m.group(2)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    nb, ni = norm(before), norm(inner)
    if len(ni) >= 3 and (nb.endswith(ni) or ni.endswith(nb)):
        return before
    return m.group(0)


def polish(para: str) -> str:
    # "least squares (L S)" stays; "bit error rate (bit error rate)" collapses
    para = re.sub(r"([A-Za-z][\w /-]{0,60}?) \(([A-Za-z0-9][\w /-]{0,40})\)",
                  _collapse_redundant_paren, para)
    # capitalise sentence starts, but keep "et al. frame" lowercase after the abbreviation
    para = para.replace("et al.", "et al\uE000")
    para = re.sub(r"(^|[.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), para)
    para = para.replace("et al\uE000", "et al.")
    para = re.sub(r"\s+([.,;:!?])", r"\1", para)
    para = re.sub(r"[ \t]+", " ", para)
    return para.strip()


def rewrap(text: str) -> str:
    """Collapse hard-wrapped lines inside a paragraph to one line, then polish."""
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(polish(" ".join(p.split())) for p in paras if p.strip())


def build_narration(stem: str) -> str:
    tex = strip_comments((CHAPTERS_DIR / f"{stem}.tex").read_text())
    _, chap_no = CHAPTERS[stem]

    # ---- pass 1: number sections / figures, record label -> number ----
    label_numbers: dict[str, str] = dict(STATIC_REFS)
    sec = subsec = fig = tab = 0
    # tokenise into (kind, payload) then re-emit
    pieces: list[tuple[str, str]] = []
    chapter_title = ""
    i = 0
    while i < len(tex):
        if tex.startswith("\\chapter", i):
            j = tex.index("{", i)
            chapter_title, i = brace_arg(tex, j)
            continue
        if tex.startswith("\\section", i) and tex[i + 8] == "{":
            title, i = brace_arg(tex, i + 8)
            sec += 1
            subsec = 0
            num = f"{chap_no}.{sec}"
            pieces.append(("heading", f"Section {num}. {title}"))
            m = re.match(r"\s*\\label\{([^}]*)\}", tex[i:])
            if m:
                label_numbers[m.group(1)] = num
                i += m.end()
            continue
        if tex.startswith("\\subsection", i) and tex[i + 11] == "{":
            title, i = brace_arg(tex, i + 11)
            subsec += 1
            num = f"{chap_no}.{sec}.{subsec}"
            pieces.append(("heading", f"Section {num}. {title}"))
            m = re.match(r"\s*\\label\{([^}]*)\}", tex[i:])
            if m:
                label_numbers[m.group(1)] = num
                i += m.end()
            continue
        if tex.startswith("\\begin{figure}", i):
            body, i = extract_environment(tex, "figure", i)
            fig += 1
            num = f"{chap_no}.{fig}"
            cap = ""
            cm = re.search(r"\\caption\{", body)
            if cm:
                cap, _ = brace_arg(body, cm.end() - 1)
            lm = re.search(r"\\label\{([^}]*)\}", body)
            if lm:
                label_numbers[lm.group(1)] = num
            pieces.append(("figure", f"Figure {num}. {cap}"))
            continue
        if tex.startswith("\\begin{table}", i):
            body, i = extract_environment(tex, "table", i)
            tab += 1
            num = f"{chap_no}.{tab}"
            cap = ""
            cm = re.search(r"\\caption\{", body)
            if cm:
                cap, _ = brace_arg(body, cm.end() - 1)
            lm = re.search(r"\\label\{([^}]*)\}", body)
            if lm:
                label_numbers[lm.group(1)] = num
            spoken = f"Table {num}. {cap}"
            # the note runs to the end of its own float/box, not to the end of the body
            gap = re.search(r"\\textit\{Gap:\}(.*?)(?:\\end\{|\Z)", body, re.S)
            if gap:
                # the caption spells out each rating after its marker; the note
                # uses the bare marker, so only the note needs the word supplied
                spoken += " Gap: " + gap.group(1).replace(r"\cvLo", "rare")
            pieces.append(("figure", spoken))
            continue
        if tex.startswith("\\begin{description}", i):
            body, i = extract_environment(tex, "description", i)
            for im in re.finditer(r"\\item\[([^\]]*)\]\s*(.*?)(?=\\item\[|\Z)", body, re.S):
                pieces.append(("para", f"{im.group(1)}. {im.group(2)}"))
            continue
        if tex.startswith("\\begin{enumerate}", i):
            body, i = extract_environment(tex, "enumerate", i)
            items = re.split(r"\\item\b", body)[1:]
            for n, it in enumerate(items, 1):
                pieces.append(("para", f"{n}. {it}"))
            continue
        if tex.startswith("\\begin{itemize}", i):
            body, i = extract_environment(tex, "itemize", i)
            for it in re.split(r"\\item\b", body)[1:]:
                pieces.append(("para", it))
            continue
        # plain text run up to the next control structure we care about
        nxt = len(tex)
        for marker in ("\\chapter", "\\section", "\\subsection",
                       "\\begin{figure}", "\\begin{table}",
                       "\\begin{description}",
                       "\\begin{enumerate}", "\\begin{itemize}"):
            k = tex.find(marker, i + 1)
            if k != -1:
                nxt = min(nxt, k)
        pieces.append(("text", tex[i:nxt]))
        i = nxt

    # ---- pass 2: clean + assemble ----
    # Figure captions are held and emitted after the next prose block, so the
    # listener hears the paragraph that introduces the figure first.
    out_blocks: list[str] = []
    pending_figs: list[str] = []

    def flush_figs() -> None:
        while pending_figs:
            out_blocks.append(pending_figs.pop(0))

    for kind, payload in pieces:
        if kind == "heading":
            flush_figs()
            out_blocks.append("\n" + apply_spoken(clean_inline(payload, label_numbers)))
        elif kind == "figure":
            cleaned = apply_spoken(clean_inline(payload, label_numbers))
            if cleaned:
                pending_figs.append(cleaned)
        elif kind == "para":
            cleaned = apply_spoken(clean_inline(payload, label_numbers))
            if cleaned:
                out_blocks.append(cleaned)
            flush_figs()
        else:  # text
            chunk = apply_spoken(clean_inline(payload, label_numbers))
            for para in re.split(r"\n\s*\n", chunk):
                para = " ".join(para.split())
                if para:
                    out_blocks.append(para)
            flush_figs()
    flush_figs()

    title = f"Chapter {chap_no}. {chapter_title}".strip().rstrip(".") + "."
    return rewrap(title + "\n\n" + "\n\n".join(out_blocks)) + "\n"


# Non-English narration text is translated and maintained by hand (this script
# only normalises English TeX). ``--lang ar`` skips the TeX build and just
# synthesises the existing ``NN_<name>_narration_<lang>.txt``.
LANG_DEFAULTS = {
    "en": {"voice": "Daniel", "rate": 170, "suffix": ""},
    "ar": {"voice": "Majed", "rate": 235, "suffix": "_ar"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chapter", type=int, default=1, help="chapter number (default 1)")
    ap.add_argument("--lang", default="en", choices=sorted(LANG_DEFAULTS),
                    help="en builds text from the TeX source; ar synthesises the "
                         "hand-maintained NN_<name>_narration_ar.txt (default en)")
    ap.add_argument("--voice", help="macOS 'say' voice (per-lang default)")
    ap.add_argument("--rate", type=int, help="words per minute (per-lang default)")
    ap.add_argument("--text-only", action="store_true", help="write the script but skip audio")
    ap.add_argument("--audio-only", action="store_true",
                    help="synthesise the existing narration text without rebuilding it")
    args = ap.parse_args()

    defaults = LANG_DEFAULTS[args.lang]
    voice = args.voice or defaults["voice"]
    rate = args.rate or defaults["rate"]
    suffix = defaults["suffix"]

    stem = next((s for s, (_, n) in CHAPTERS.items() if n == args.chapter), None)
    if stem is None:
        sys.exit(f"unknown chapter {args.chapter}")
    prefix = CHAPTERS[stem][0]

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = EXPORTS_DIR / f"{prefix}_narration{suffix}.txt"
    if args.audio_only:
        if not txt_path.exists():
            sys.exit(f"{txt_path} not found")
    elif args.lang == "en":
        txt_path.write_text(build_narration(stem))
    elif not txt_path.exists():
        sys.exit(f"{txt_path} not found -- translate it by hand first")
    script = txt_path.read_text()
    words = len(script.split())
    print(f"using {txt_path}  ({words} words, ~{words / rate:.0f} min at {rate} wpm)")

    if args.text_only:
        return
    if sys.platform != "darwin":
        sys.exit("audio needs macOS 'say'; re-run with --text-only elsewhere")
    m4a_path = EXPORTS_DIR / f"{prefix}{suffix}.m4a"
    synthesize_audio(script, voice=voice, rate=rate, m4a_path=m4a_path)
    print(f"wrote {m4a_path}")


# ``say -f`` silently under-writes very long scripts (chapter 2 is ~9k words).
# Synthesise paragraph packs, convert each to the same PCM, then concatenate.
CHUNK_WORDS = 1200


def split_chunks(script: str, max_words: int = CHUNK_WORDS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", script) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for para in paras:
        w = len(para.split())
        if cur and n + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def synthesize_audio(script: str, *, voice: str, rate: int, m4a_path: Path) -> None:
    chunks = split_chunks(script)
    print(f"synthesising {len(chunks)} chunks with {voice} at {rate} wpm")
    with tempfile.TemporaryDirectory(prefix="narration_") as td:
        tmp = Path(td)
        wavs: list[Path] = []
        for i, chunk in enumerate(chunks, 1):
            txt = tmp / f"c{i:02d}.txt"
            aiff = tmp / f"c{i:02d}.aiff"
            wav = tmp / f"c{i:02d}.wav"
            txt.write_text(chunk)
            print(f"  chunk {i}/{len(chunks)} ({len(chunk.split())} words)")
            subprocess.run(
                ["say", "-v", voice, "-r", str(rate), "-o", str(aiff), "-f", str(txt)],
                check=True,
            )
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1",
                 str(aiff), str(wav)],
                check=True,
            )
            wavs.append(wav)
        out_wav = tmp / "all.wav"
        with wave.open(str(wavs[0]), "rb") as first:
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]
        for wpath in wavs[1:]:
            with wave.open(str(wpath), "rb") as src:
                if src.getparams()[:3] != params[:3]:
                    sys.exit(f"wav param mismatch: {wpath}")
                frames.append(src.readframes(src.getnframes()))
        with wave.open(str(out_wav), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(b"".join(frames))
        # AAC mono 22.05 kHz, 64 kbps -- matches the earlier chapter exports
        subprocess.run(
            ["afconvert", "-f", "m4af", "-d", "aac@22050", "-b", "64000",
             "-c", "1", str(out_wav), str(m4a_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
