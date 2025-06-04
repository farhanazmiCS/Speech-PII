import typer
import pandas as pd
from pipeline import SpeechPIIPipeline
from utils import save_df, load_df
from vosk import Model
import json, ast
from typing import List

app = typer.Typer()

@app.command()
def transcribe(
    model_name: str = "f-azm17/whisper-small_en_seed_gretel_similar0.3-default-tokenizer",
    llm_model: str = "gpt-4o",
    audio_dirs: list[str] = ["../../data/Audio_Files_for_testing", "../../data/newtest_151_500_updated_TTS"],
    n_best: int = 1,
    out_csv: str = "data/transcriptions.csv"
):
    """
    Step 1: ASR transcription (greedy or n-best).
    """
    pipe = SpeechPIIPipeline(asr_model_name=model_name, llm_model=llm_model)
    df = pipe.transcribe(audio_dirs, n_best=n_best)
    save_df(df, out_csv)
    typer.echo(f"Transcriptions ({n_best}-best) saved to: {out_csv}")

@app.command()
def correct(
    in_csv: str = "data/transcriptions.csv",
    out_csv: str = "data/transcriptions_corrected.csv"
):
    """
    Step 2: LLM-based correction of transcripts.
    Automatically handles greedy vs. N-best.
    """
    pipe = SpeechPIIPipeline()
    df = load_df(in_csv)
    df_corr = pipe.correct(df, out_csv=None)
    save_df(df_corr, out_csv)
    typer.echo(f"Corrected transcripts saved to: {out_csv}")

@app.command()
def tag(
    in_csv: str = "data/transcriptions_corrected.csv",
    text_col: str = "transcript",
    method: str = "zero_shot_icl",
    out_csv: str = "data/transcriptions_tagged.csv"
):
    """
    Step 3: LLM-based PII tagging.
    """
    pipe = SpeechPIIPipeline()
    df = load_df(in_csv)
    df_tag = pipe.tag(df, text_col=text_col, method=method, out_csv=None)
    save_df(df_tag, out_csv)
    typer.echo(f"Tagged transcripts ({method}) saved to: {out_csv}")

@app.command()
def vosk(
    in_csv: str = "data/transcriptions.csv",
    audio_col: str = "file",
    vosk_model_dir: str = "models/vosk-model-en-us-0.42-gigaspeech",
    out_json: str = "../../data/vosk_output/vosk_output.json"
):
    """
    Run Vosk for word-level timestamps.
    """
    pipe = SpeechPIIPipeline()
    df = load_df(in_csv)
    vosk_model = Model(vosk_model_dir)
    df_vosk = pipe.get_vosk_timestamps(df, audio_col=audio_col, vosk_model=vosk_model, out_json=out_json)
    #save_df(df_vosk, out_json)
    typer.echo(f"Vosk timestamps saved to: {out_json}")

@app.command()
def extract(
    input_csv: str = typer.Argument(..., help="CSV with columns 'file' and 'tagged'"),
    output_csv: str = typer.Argument(..., help="Where to write clean_text + pii_tuples"),
    allowed_labels: List[str] = typer.Option(
        None,
        "--allowed_labels",
        "-l",
        help="One PII label per flag (e.g. --label EMAIL --label PHONE)."
    )
):
    """
    Strip out all [LABEL_START]…[LABEL_END] from each row of INPUT_CSV (which must
    have columns 'file' and 'tagged').  Only labels listed via (--label) are retained.
    Writes OUTPUT_CSV containing columns:
        file, clean_text, pii_tuples  (where pii_tuples is a list of (start,end,label)).
    """
    pipe = SpeechPIIPipeline()
    if not allowed_labels:
        allowed_labels = [
            "EMAIL", "NRIC", "CREDIT_CARD", "PHONE",
            "PASSPORT_NUM", "BANK_ACCOUNT", "CAR_PLATE", "PERSON"
        ]

    df_out = pipe.extract(input_csv, output_csv, allowed_labels)
    typer.echo(f"Done—wrote {len(df_out)} rows to {output_csv}")

@app.command()
def extract_using_timestamps(
    method: str = "",
    in_csv: str = "",
    out_csv: str = "",
    vosk_json: str = "../../data/vosk_output/vosk_output.json",
):
    """
    Step 4: Align & extract PII tuples.
    Requires `vosk_words` column (or load from vosk_json).
    """
    pipe = SpeechPIIPipeline()
    df = load_df(in_csv)
    vosk_df = pd.read_json(
        vosk_json,
        lines=True,
        orient="records"
        )
    df["vosk_words"] = vosk_df["vosk_words"]
    df_trip = pipe.align_and_extract(df, tagged_col="tagged", method=method, out_csv=out_csv)
    save_df(df_trip, out_csv)
    typer.echo(f"Extracted triplets saved to: {out_csv}")

@app.command()
def evaluate(
    gt_csv: str = "../../data/triplets/ref_triplets_500.csv",
    pred_csv: str = "../../data/triplets/triplets_no_correct_zero_shot_icl.csv",
    tolerance: float = 5
):
    """
    Step 5: Evaluation (precision, recall, F1, confusion matrix).
    """
    pipe = SpeechPIIPipeline()
    gt_df = load_df(gt_csv)
    pred_df = load_df(pred_csv)
    pipe.evaluate(
        true_df=gt_df,
        pred_df=pred_df,
        classes=[
            'EMAIL','NRIC','CREDIT_CARD','PHONE',
            'PASSPORT_NUM','BANK_ACCOUNT','CAR_PLATE',
            'PERSON'
        ],
        offset_tolerance=tolerance,
    )

@app.command()
def evaluate_index(
    true_csv: str = typer.Argument(..., help="Path to the CSV with ground‐truth pii_tuples"),
    pred_csv: str = typer.Argument(..., help="Path to the CSV with predicted pii_tuples"),
    allowed_labels: List[str] = typer.Option(
        None,
        "--allowed_labels",
        "-l",
        help="One PII label per flag (e.g. --label EMAIL --label PHONE)."
    ),
    tolerance: int = typer.Option(
        5,
        help="Character‐index tolerance for matching start/end (default = 5 → exact match)"
    )
):
    """
    Evaluate PII extraction by character‐index matching with optional tolerance.
    Only the labels in --labels will be counted.
    """
    pipeline = SpeechPIIPipeline()
    # Parse label list
    try:
        classes = [lab.strip() for lab in allowed_labels.split(",") if lab.strip()]
    except Exception as e:
        classes = [
            'EMAIL', 'NRIC', 'CREDIT_CARD', 'PHONE',
            'PASSPORT_NUM', 'BANK_ACCOUNT', 'CAR_PLATE'
        ]

    # Load both dataframes
    true_df = pd.read_csv(true_csv)
    pred_df = pd.read_csv(pred_csv)

    # Call our index‐based evaluator with tolerance
    pipeline.evaluate_by_index(true_df, pred_df, classes, tolerance)

if __name__ == "__main__":
    app()
