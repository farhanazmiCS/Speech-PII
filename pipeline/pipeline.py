import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from openai import OpenAI

from utils import save_df, load_df
from prompts import build_correction_prompt, build_tagging_prompt
from utils import tokenize_reference, align_transcript_with_vosk
from utils import extract_pii_tuples, retrieve_key
import typer

from vosk import Model, KaldiRecognizer
import soundfile as sf
import json, ast

VALID_LLMS = [
    "gpt-4o", "llama-3-70b-instruct"
]

class SpeechPIIPipeline:
    def __init__(self,
                 asr_model_name: str = "f-azm17/whisper-small_en_seed_gretel_similar0.3-default-tokenizer",
                 llm_model: str = "gpt-4o",
                 device: int = 0):
        # ASR: low-level for n-best
        if device == 0:
            self.device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
        elif device == 1:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.processor = WhisperProcessor.from_pretrained(asr_model_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_name).to(self.device)
        # LLM pipeline (uses OpenAI API or HF pipeline)
        if llm_model not in VALID_LLMS:
            raise ValueError(f"Invalid LLM model: {llm_model}. Choose from {VALID_LLMS}.")
        elif llm_model == "llama-3-70b-instruct":
            self.llm = pipeline("text2text-generation", model=llm_model, device=device)
        elif llm_model == "gpt-4o":
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self,
                   input_dirs: list[str],
                   n_best: int = 1,
                   temperature: float = 1.0) -> pd.DataFrame:
        """
        Produce greedy or n-best transcriptions, with progress bar.
        """
        rows: List[dict] = []
        # gather all file paths
        file_paths: List[str] = []
        for d in input_dirs:
            for fn in sorted(os.listdir(d)):
                file_paths.append(os.path.join(d, fn))

        with typer.progressbar(file_paths, label="🔊 Transcribing audio") as bar:
            for path in bar:
                fn = os.path.basename(path)
                waveform, sr = librosa.load(path, sr=16_000)
                inputs = self.processor(
                    waveform, sampling_rate=sr, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    generated_ids = self.asr_model.generate(
                        input_features=inputs["input_features"],
                        temperature=temperature,
                        num_beams=n_best,
                        num_return_sequences=n_best,
                        early_stopping=True
                    )
                transcripts = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                row = {"file": fn}
                if n_best == 1:
                    row["transcript"] = transcripts[0]
                else:
                    for i, t in enumerate(transcripts, start=1):
                        row[f"rank_{i}"] = t

                rows.append(row)

        return pd.DataFrame(rows)

    def correct(self,
            df: pd.DataFrame,
            out_csv: str | None = None) -> pd.DataFrame:
        """
        HyPoRadise‐style ASR correction.
        - If df has columns rank_1…rank_N, use those.
        - Otherwise, fall back to the 'transcript' column.
        - Wraps a progress bar and optionally writes CSV.
        """
        df_out: List[dict] = []

        # 1) find your N-best columns or default to greedy 'transcript'
        rank_cols = sorted(
            [c for c in df.columns if c.startswith("rank_")],
            key=lambda x: int(x.split("_")[1])
        )
        if not rank_cols and "transcript" in df.columns:
            rank_cols = ["transcript"]

        items = list(df.iterrows())
        with typer.progressbar(items, label="✏️ Correcting transcripts") as bar:
            for _, row in bar:
                # build list of hypotheses in order
                n_best_list = [row[c] for c in rank_cols]
                prompt = build_correction_prompt(n_best_list)

                response = self.llm.responses.create(
                    model="gpt-4o",
                    instructions=(
                        "You are an expert ASR‐correction assistant. "
                        "Given the N-best hypotheses, synthesize the true transcription"
                        "and improve the structure of any entities (e.g., emails, numbers, etc.) you can find."
                    ),
                    input=prompt
                )
                df_out.append({
                    "file":      row["file"],
                    "transcript": response.output_text
                })

        df_corr = pd.DataFrame(df_out)
        if out_csv:
            df_corr.to_csv(out_csv, index=False)
        return df_corr


    def tag(self,
            df: pd.DataFrame,
            text_col: str,
            method: str,
            out_csv: str | None = None) -> pd.DataFrame:
        """
        LLM-based PII tagging, with progress bar.
        """
        df_out = []
        items = list(df.iterrows())
        with typer.progressbar(items, label=f"🏷️ Tagging PII ({method})") as bar:
            for _, r in bar:
                prompt = build_tagging_prompt(r[text_col], method)
                response = self.llm.responses.create(
                    model="gpt-4o",
                    instructions=(
                        "You are an expert in tagging PII."
                    ),
                    input=prompt
                )
                raw = response.output_text

                # define markers
                reason_marker = "### REASONING ###"
                output_marker = "### OUTPUT ###"

                # extract reason (text between ### REASONING ### and ### OUTPUT ###)
                if reason_marker in raw:
                    after_reason = raw.split(reason_marker, 1)[1]
                    if output_marker in after_reason:
                        reason = after_reason.split(output_marker, 1)[0].strip()
                    else:
                        reason = after_reason.strip()
                else:
                    reason = ""

                # extract tagged transcript (text after ### OUTPUT ###)
                if output_marker in raw:
                    tagged = raw.split(output_marker, 1)[1].strip()
                else:
                    tagged = raw.strip()

                df_out.append({
                    "file":   r["file"],
                    "tagged": tagged,
                    "reason": reason
                })

            result = pd.DataFrame(df_out)
            if out_csv:
                result.to_csv(out_csv, index=False)
            return result
        
    def get_vosk_timestamps(self,
                             df: pd.DataFrame,
                             audio_col: str,
                             vosk_model: Model,
                             out_json: str | None = None) -> pd.DataFrame:
        """
        Iterate through df[audio_col], run Vosk on each path, and collect timestamps.
        Adds a 'vosk_words' column (list of word dicts). Optionally writes CSV.
        """
        df_out = []
        items = list(df.iterrows())

        with typer.progressbar(items, label="🎙️ Retrieving Vosk timestamps") as bar:
            for _, row in bar:
                rec = KaldiRecognizer(vosk_model, 16000)
                rec.SetWords(True)
                index = int(retrieve_key(row[audio_col]))
                if index <= 150:
                    path = '../../data/Audio_Files_for_testing/' + row[audio_col]
                else:
                    path = '../../data/newtest_151_500_updated_TTS/' + row[audio_col]
                try:
                    audio_data, _ = sf.read(path)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    pcm_data = (audio_data * 32767).astype("int16").tobytes()
                    rec.AcceptWaveform(pcm_data)
                    result = json.loads(rec.FinalResult())
                    df_out.append({
                        "file":       path,
                        "vosk_words": result['result']
                    })
                except Exception as e:
                    df_out.append({
                        "file":       path,
                        "vosk_words": []
                    })

        df_vosk = pd.DataFrame(df_out)
        if out_json:
            df_vosk.to_json(out_json, orient='records', lines=True)
        return df_vosk

    def align_and_extract(self,
                          df: pd.DataFrame,
                          tagged_col: str,
                          method: str,
                          out_csv: str) -> pd.DataFrame:
        """
        Align & extract PII tuples, with progress bar.
        """
        df_out = []
        items = list(df.iterrows())
        with typer.progressbar(items,
                               label=f"🔍 Aligning & extracting ({method})") as bar:
            for _, r in bar:
                aligned = align_transcript_with_vosk(r['vosk_words'], r[tagged_col])
                triplets = extract_pii_tuples(
                    pd.DataFrame({"aligned_transcript": [aligned]})
                )

                triplets = triplets.to_dict('records')
                
                df_out.append(triplets[0])

        result = pd.DataFrame(df_out)
        result.to_csv(out_csv, index=False)
        return result

    def evaluate(self,
             true_df: pd.DataFrame,
             pred_df: pd.DataFrame,
             classes: list[str],
             offset_tolerance: float = 0.5) -> None:
        """
        Compute entity‐level precision, recall, F1 and confusion matrix based on
        time‐boundary matching within a tolerance, then matching labels.

        Args:
            true_df: DataFrame with column 'pii_tuples' containing lists of
                    (start, end, label) ground-truth spans.
            pred_df: DataFrame with same structure for predictions.
            classes: List of allowed PII labels.
            offset_tolerance: Max seconds difference allowed on start/end to count as a boundary match.
        """
        # Initialize counters
        metrics = {lab: {'tp':0, 'fp':0, 'fn':0} for lab in classes}
        idx_map = {lab:i for i, lab in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)

        # Iterate rows
        for gt_row, pred_row in zip(true_df.itertuples(), pred_df.itertuples()):
            gt_entities   = gt_row.pii_tuples or []
            pred_entities = pred_row.pii_tuples or []
            gt_left       = gt_entities.copy()

            # Match each predicted span to a ground-truth span
            for p_start, p_end, p_lab in pred_entities:
                matched = False
                for i, (g_start, g_end, g_lab) in enumerate(gt_left):
                    if (abs(p_start - g_start) <= offset_tolerance and
                        abs(p_end   - g_end)   <= offset_tolerance):
                        matched = True
                        # update confusion matrix
                        gi, pi = idx_map[g_lab], idx_map[p_lab]
                        cm[gi, pi] += 1
                        # true positive or label-swap
                        if p_lab == g_lab:
                            metrics[p_lab]['tp'] += 1
                        else:
                            metrics[p_lab]['fp'] += 1
                            metrics[g_lab]['fn'] += 1
                        gt_left.pop(i)
                        break
                if not matched:
                    # false positive for unmatched prediction
                    if p_lab in metrics:
                        metrics[p_lab]['fp'] += 1

            # any gt left are false negatives
            for _, _, g_lab in gt_left:
                metrics[g_lab]['fn'] += 1

        # Compute precision, recall, F1
        p_list, r_list, f_list = [], [], []
        for lab in classes:
            tp = metrics[lab]['tp']
            fp = metrics[lab]['fp']
            fn = metrics[lab]['fn']
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            p_list.append(prec)
            r_list.append(rec)
            f_list.append(f1)

        overall_f1 = np.mean(f_list)

        # Print results
        print("Per-class F1:")
        for lab, f1 in zip(classes, f_list):
            print(f"  {lab}: {f1:.3f}")
        print(f"\nOverall (macro) F1: {overall_f1:.3f}\n")

        print("Confusion Matrix:")
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        print(df_cm)
