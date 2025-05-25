def build_correction_prompt(n_best_list: list[str]) -> str:
    """
    HyPoRadise‐style prompt: ask the LLM to synthesize the true transcription
    from a 5-best ASR hypotheses list.

    Args:
        n_best_list: list of hypotheses in rank order (rank_1 first).
    Returns:
        A single prompt string.
    """
    header = (
        "Report the true transcription from the N-best hypotheses:\n\n"
    )
    best = f"Best hypothesis (rank 1):\n{n_best_list[0]}\n\n"
    others = ""
    if len(n_best_list) > 1:
        others = "Other hypotheses:\n" + "\n".join(
            f"{i}) {hypo}" for i, hypo in enumerate(n_best_list[1:], start=2)
        ) + "\n\n"
    footer = "Report the true (corrected) transcription:\n\nDO NOT include any additional text or explanations, just the corrected transcription.\n"
    return header + best + others + footer

def build_tagging_prompt(transcript: str, method: str) -> str:
    """
    Build a prompt to ask the LLM to tag PII. Supported methods:
      - zero_shot:    no examples
      - few_shot:     include a couple of examples
      - zero_shot_cot: ask for step-by-step reasoning
      - few_shot_cot:  examples + step-by-step
    """
    templates = {
        "zero_shot": (
            "Tag all PII entities in the transcript below using [TAG_START] and [TAG_END].\n\n"
            f"{transcript}\n\n"
            "Return only the tagged transcript."
        ),
        "few_shot": (
            "Here are examples:\n"
            "Transcript: Alice's email is [EMAIL_START]alice@example.com[EMAIL_END].\n"
            "Tagged: Alice's email is [EMAIL_START]alice@example.com[EMAIL_END].\n\n"
            "Now tag the following:\n"
            f"{transcript}\n\n"
            "Return only the tagged transcript."
        ),
        "zero_shot_cot": (
            "Think step by step and tag all PII in the transcript below. Use [TAG_START] and [TAG_END].\n\n"
            f"{transcript}\n\n"
            "Return only the tagged transcript."
        ),
        "few_shot_cot": (
            "Example 1:\n"
            "Transcript: My phone is [PHONE_START]123-4567[PHONE_END].\n"
            "Step-by-step:\n"
            "1. Identify '123-4567' as a phone.\n"
            "2. Surround with tags.\n"
            "Result: My phone is [PHONE_START]123-4567[PHONE_END].\n\n"
            "Now do the same for:\n"
            f"{transcript}\n\n"
            "Return only the tagged transcript."
        )
    }
    return templates.get(method, templates["zero_shot"])
