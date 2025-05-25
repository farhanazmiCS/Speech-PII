import pandas as pd
from vosk import Model, KaldiRecognizer
import soundfile as sf
import json
import re, string
import ast, re

# ────────────────────────────────────────────────────────────
# Helper functions for saving and exporting DataFrames to/from CSV
# ────────────────────────────────────────────────────────────
def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ────────────────────────────────────────────────────────────
# Helper functions for running alignment model
# ────────────────────────────────────────────────────────────
def run_vosk(audio_path: str, vosk_model: Model) -> list:
    # Load audio
    audio_data, sample_rate = sf.read(audio_path)
    
    # Prepare recognizer
    rec = KaldiRecognizer(vosk_model, sample_rate)
    rec.SetWords(True)
    
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # Stereo to mono

    pcm_data = (audio_data * 32767).astype("int16").tobytes()

    rec.AcceptWaveform(pcm_data)
    result = json.loads(rec.FinalResult())
    
    return result

# ────────────────────────────────────────────────────────────
# Helper functions for tokenization of transcripts (Hypotheses/References)
# ────────────────────────────────────────────────────────────
digit_map = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

# Match tags (case-insensitive, underscore tolerant)
TAG_RE   = re.compile(r'\[[A-Z0-9_]+?_(?:START|END)\]', re.I)
TOKEN_RE = re.compile(r'\[[^\]]+]|[\w]+|[@._-]')

# These PII types will have hyphens removed inside the block
PII_TAGS_REMOVE_HYPHEN = {'CREDIT_CARD', 'PHONE', 'BANK_ACCOUNT'}

def tokenize_reference(text: str):
    tokens = []
    inside_pii_block = None  # None or tag name (e.g., 'PHONE')

    for part in TOKEN_RE.findall(text):
        # ---------- 1. Exact tag: check START/END ----------
        if TAG_RE.fullmatch(part):
            tokens.append(part)
            if part.upper().endswith('_START]'):
                inside_pii_block = part[1:-1].replace('_START', '').upper()
            elif part.upper().endswith('_END]'):
                inside_pii_block = None
            continue

        # ---------- 2. Symbols ----------
        if part in {'@', '.', '_'}:
            tokens.append(part)
            continue

        # ---------- 3. Remove hyphens if inside selected PII ----------
        if inside_pii_block in PII_TAGS_REMOVE_HYPHEN:
            part = part.replace('-', '')

        # ---------- 4. Contains digits: explode ----------
        if re.search(r'\d', part):
            for ch in part:
                if ch.isdigit():
                    tokens.append(digit_map[ch])
                elif ch.isalpha():
                    tokens.append(ch.lower())
            continue

        # ---------- 5. Normal word ----------
        clean = part.lower().strip(string.punctuation)
        if clean:
            tokens.append(clean)

    return tokens

from difflib import SequenceMatcher
import string
import re

# ────────────────────────────────────────────────────────────
# Helper functions for alignment
# ────────────────────────────────────────────────────────────
def clean_token(token):
    """Lowercase and strip punctuation from token unless it's a tag."""
    if token.startswith("[") and token.endswith("]"):
        return token
    return token.lower().translate(str.maketrans('', '', string.punctuation))

def is_tag(token):
    return token.startswith("[") and token.endswith("]")

def is_start_tag(token):
    return token.endswith("_START]")

def is_end_tag(token):
    return token.endswith("_END]")

# ────────────────────────────────────────────────────────────
# Main Alignment Function
# ────────────────────────────────────────────────────────────
def align_transcript_with_vosk(vosk_words, transcript):
    # Tokenize reference into tokens including tags
    ref_tokens = tokenize_reference(transcript)

    # Save positions of tags
    tag_positions = [(i, t) for i, t in enumerate(ref_tokens) if is_tag(t)]

    # Remove tags from reference tokens
    ref_tokens_clean = [t for t in ref_tokens if not is_tag(t)]
    ref_tokens_clean_norm = [clean_token(t) for t in ref_tokens_clean]

    vosk_tokens = [w['word'] for w in vosk_words]
    vosk_tokens_clean = [clean_token(t) for t in vosk_tokens]

    # Align with SequenceMatcher
    matcher = SequenceMatcher(None, ref_tokens_clean_norm, vosk_tokens_clean, autojunk=False)

    aligned = []
    i_clean = 0  # index in cleaned ref tokens
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for i_r, i_v in zip(range(i1, i2), range(j1, j2)) if tag == "equal" else []:
            aligned.append({
                'word': ref_tokens_clean[i_r],
                'start': vosk_words[i_v]['start'],
                'end': vosk_words[i_v]['end']
            })
        if tag == "replace":
            for i_r in range(i1, i2):
                start = vosk_words[j1]['start'] if j1 < len(vosk_words) else None
                end = vosk_words[j2 - 1]['end'] if (j2 - 1) < len(vosk_words) else None
                aligned.append({
                    'word': ref_tokens_clean[i_r],
                    'start': start if i_r == i1 else None,
                    'end': end if i_r == i2 - 1 else None
                })
        if tag == "delete":
            for i_r in range(i1, i2):
                aligned.append({
                    'word': ref_tokens_clean[i_r],
                    'start': None,
                    'end': None
                })

    # Reinsert tags with null times
    for pos, tag in tag_positions:
        aligned.insert(pos, {'word': tag, 'start': None, 'end': None})

    return aligned

# ────────────────────────────────────────────────────────────
# Helper functions for extracting PII tuples from aligned transcripts
# ────────────────────────────────────────────────────────────
_START_RE = re.compile(r'\[([A-Z_]+?)_START\]', re.I)
_END_RE   = re.compile(r'\[([A-Z_]+?)_END\]'  , re.I)

# ---------- helper: load cell → list[dict] ---------------------
def _to_tokens(cell):
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return json.loads(cell)          # JSON
        except json.JSONDecodeError:
            return ast.literal_eval(cell)    # python literal
    raise ValueError("aligned_transcript must be list or JSON-string")

# ---------- helper: first non-null AFTER idx -------------------
def _fwd_time(tokens, idx, field):
    for t in tokens[idx + 1:]:
        if t[field] is not None:
            return t[field]
    return None

# ---------- helper: last non-null BEFORE idx -------------------
def _back_time(tokens, idx, field):
    for t in reversed(tokens[:idx]):
        if t[field] is not None:
            return t[field]
    return None

# ────────────────────────────────────────────────────────────
# main pii tuples extraction function
# ────────────────────────────────────────────────────────────
def extract_pii_tuples(df,
                       align_col="aligned_transcript",
                       out_col="pii_tuples"):
    """
    Populate df[out_col] with [(start_time, end_time, TAG), …]

    Fallback order
    --------------
    • [TAG_START] start-time
        1. next token .start
        2. previous token .start

    • [TAG_END] end-time
        1. previous token .end
        2. next token .end
    """
    out_rows = []

    for cell in df[align_col]:
        tokens      = _to_tokens(cell)
        tuples_row  = []
        open_tag    = None
        open_start  = None

        for idx, tok in enumerate(tokens):
            word = tok["word"]

            # ---------- opening tag ----------
            m_open = _START_RE.search(word)
            if m_open:
                open_tag   = m_open.group(1)

                # 1. next token .start
                if open_start is None:
                    open_start = _fwd_time(tokens, idx, "start")
                # 2. previous token .start
                if open_start is None and idx > 0:
                    open_start = tokens[idx - 1]["start"]
                continue

            # ---------- closing tag ----------
            m_close = _END_RE.search(word)
            if m_close and open_tag == m_close.group(1):
                # 1. previous token .end
                if end_t is None and idx > 0:
                    end_t = tokens[idx - 1]["end"]
                # 2. next token .end
                if end_t is None:
                    end_t = _fwd_time(tokens, idx, "end")

                tuples_row.append((open_start, end_t, open_tag))
                open_tag, open_start = None, None
                continue

        out_rows.append(tuples_row)

    df[out_col] = out_rows
    return df
