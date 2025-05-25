## Overview

This pipeline orchestrates ASR transcription, LLM-based correction & tagging, alignment, PII extraction, and evaluation in a modular, configurable way. You can run end-to-end or pick & choose steps.

---

## Dependencies

```bash
pip install transformers datasets pandas openai typer sklearn
```

---

## Usage

```bash
$ python cli.py --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  transcribe    Step 1: ASR transcription (greedy or n-best)
  correct       Step 2: LLM-based transcript correction
  tag           Step 3: LLM-based PII tagging
  extract       Step 4: Align & extract PII tuples
  evaluate      Step 5: Evaluation (precision/recall/F1/confusion)

───────────────────────────────────────────────────────────────────────────────
```

```bash
$ python cli.py transcribe --help
Usage: cli.py transcribe [OPTIONS]

  Step 1: ASR transcription (greedy or n-best).

Options:
  --model-name TEXT       Whisper model name (default: f-azm17/whisper-small_en_seed_gretel_similar0.3-default-tokenizer)
  --llm-model TEXT        LLM model (default: openai/gpt-4o)
  --audio-dirs TEXT...    List of audio directories
                          (default: ["data/Audio_Files_for_testing","data/newtest_151_500_updated_TTS"])
  --n-best INT            Number of beams / return sequences (default: 1)
  --out-csv TEXT          Output CSV path (default: data/transcriptions.csv)
  --help                  Show this message and exit.

```
Example:

```
python cli.py transcribe \
  --model-name f-azm17/whisper-small_en_... \
  --n-best 5 \
  --out-csv data/500_test_transcriptions_5-best.csv
```


───────────────────────────────────────────────────────────────────────────────

```bash
$ python cli.py correct --help
Usage: cli.py correct [OPTIONS]

  Step 2: LLM-based transcript correction.

Options:
  --in-csv TEXT     Input CSV of transcriptions
                    (default: data/transcriptions.csv)
  --out-csv TEXT    Output CSV for corrected transcripts
                    (default: data/transcriptions_corrected.csv)
  --help            Show this message and exit.
```

Example:

```
python cli.py correct \
  --in-csv data/500_test_transcriptions_greedy.csv \
  --out-csv data/500_test_transcriptions_corrected.csv
```

───────────────────────────────────────────────────────────────────────────────


```bash
$ python cli.py tag --help
Usage: cli.py tag [OPTIONS]

  Step 3: LLM-based PII tagging.

Options:
  --in-csv TEXT     Input CSV of corrected transcripts
                    (default: data/transcriptions_corrected.csv)
  --text-col TEXT   Column name for corrected text
                    (default: transcript)
  --method TEXT     Tagging method: zero_shot, few_shot,
                    zero_shot_cot, few_shot_cot
                    (default: zero_shot)
  --out-csv TEXT    Output CSV for tagged transcripts
                    (default: data/transcriptions_tagged.csv)
  --help            Show this message and exit.
```

Example:

```
python cli.py tag \
  --in-csv data/500_test_transcriptions_corrected.csv \
  --method few_shot_cot \
  --out-csv data/500_test_transcriptions_tagged_few_shot_cot.csv
```

───────────────────────────────────────────────────────────────────────────────

```bash
$ python cli.py extract --help
Usage: cli.py extract [OPTIONS]

  Step 4: Align & extract PII tuples.

Options:
  --in-csv TEXT       Input CSV of tagged transcripts
                      (default: data/transcriptions_tagged.csv)
  --tagged-col TEXT   Column name for tagged text
                      (default: tagged)
  --vosk-json TEXT    Optional JSON file with vosk_words column
  --out-csv TEXT      Output CSV for triplets
                      (default: data/triplets.csv)
  --help              Show this message and exit.
```

Example:

```
python cli.py extract \
  --in-csv data/500_test_transcriptions_tagged_few_shot.csv \
  --vosk-json data/500_vosk_words.json \
  --out-csv data/hypo_triplets_500_few_shot.csv
```

───────────────────────────────────────────────────────────────────────────────

```bash
$ python cli.py evaluate --help
Usage: cli.py evaluate [OPTIONS]

  Step 5: Evaluation (precision/recall/F1/confusion).

Options:
  --gt-csv TEXT       Ground-truth triplets CSV
                      (default: data/ground_truth.csv)
  --pred-csv TEXT     Predicted triplets CSV
                      (default: data/triplets.csv)
  --tolerance FLOAT   Time-offset tolerance in seconds
                      (default: 0.5)
  --help              Show this message and exit.
```

Example:

```
python cli.py evaluate \
  --gt-csv data/500_gt_triplets.csv \
  --pred-csv data/hypo_triplets_500_few_shot.csv \
  --tolerance 0.05
```