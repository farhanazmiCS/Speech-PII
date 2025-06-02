# prompts.py
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
    Build a PII-tagging prompt according to the specified method, using
    a templates dict and fallback to zero_shot_cot.
    """
    boundary_tags = (
        "Use these PII tags *exactly*:\n"
        "- [EMAIL_START]…[EMAIL_END]\n"
        "- [NRIC_START]…[NRIC_END]\n"
        "- [CREDIT_CARD_START]…[CREDIT_CARD_END]\n"
        "- [PHONE_START]…[PHONE_END]\n"
        "- [PASSPORT_NUM_START]…[PASSPORT_NUM_END]\n"
        "- [BANK_ACCOUNT_START]…[BANK_ACCOUNT_END]\n"
        "- [CAR_PLATE_START]…[CAR_PLATE_END]\n"
        "- [PERSON_START]…[PERSON_END]\n"
        "- [DATE_START]…[DATE_END]\n"
    )

    common = (
        "Follow the template when outputting your response.\n"
        "Preserve the transcript exactly; do **not** correct typos or rephrase.\n"
        "Only add start/end tags around PII as described in the PII tags section.\n"
        "DO NOT COME UP WITH YOUR OWN PII TAGS; use the ones provided.\n"
        "Output the final tagged transcript below the ### OUTPUT ### section.\n"
    )

    cot_instruct = (
        "Think step-by-step, and output your reasoning thought process for each tag below the ### REASONING ### section.\n"
    )

    # all nine examples packed into one list
    examples = [
        {
            "in":  "okay uh my full name is janice Teh uh Wei Ling and I C number is G 0881816 P.",
            "out": "okay uh my full name is [PERSON_START] janice Teh uh Wei Ling [PERSON_END] and I C number is [NRIC_START] G 0881816 P [NRIC_END].",
            "reason": "“janice Teh uh Wei Ling” is a full name → PERSON; “G 0881816 P” matches NRIC format → NRIC."
        },
        {
            "in":  "d a n n y dot tan at protonmail dot com is my secure email.",
            "out": "[EMAIL_START] d a n n y dot tan at protonmail dot com [EMAIL_END] is my secure email.",
            "reason": "Obfuscated email format → EMAIL."
        },
        {
            "in":  "So her bank account number is 173-2845-10 and she banks with POSB.",
            "out": "So her bank account number is [BANK_ACCOUNT_START] 173-2845-10 [BANK_ACCOUNT_END] and she banks with POSB.",
            "reason": "Explicitly referred bank account number → BANK_ACCOUNT."
        },
        {
            "in":  "Can I make the payment using my credit card? It’s 5243 8821 9912 3109 and expires next year.",
            "out": "Can I make the payment using my credit card? It’s [CREDIT_CARD_START] 5243 8821 9912 3109 [CREDIT_CARD_END] and expires next year.",
            "reason": "16-digit sequence → CREDIT_CARD."
        },
        {
            "in":  "The license plate of the car parked outside is SKB 9012 C.",
            "out": "The license plate of the car parked outside is [CAR_PLATE_START] SKB 9012 C [CAR_PLATE_END].",
            "reason": "Valid Singapore car plate → CAR_PLATE."
        },
        {
            "in":  "You can verify my identity using passport number K 1065125 K.",
            "out": "You can verify my identity using passport number [PASSPORT_NUM_START] K 1065125 K [PASSPORT_NUM_END].",
            "reason": "Valid passport format → PASSPORT_NUM."
        },
        {
            "in":  "She was born on two september 1995.",
            "out": "She was born on [DATE_START] two september 1995 [DATE_END].",
            "reason": "Full date → DATE."
        },
        {
            "in":  "I spoke to jon this morning about the refund request.",
            "out": "I spoke to [PERSON_START] jon [PERSON_END] this morning about the refund request.",
            "reason": "Name mention → PERSON."
        },
        {
            "in":  "My NRIC is S1234567D and my email is felicia123 at gmail dot com.",
            "out": "My NRIC is [NRIC_START] S1234567D [NRIC_END] and my email is [EMAIL_START] felicia123 at gmail dot com [EMAIL_END].",
            "reason": "NRIC pattern → NRIC; obfuscated email → EMAIL."
        }
    ]

    # build each template string once
    few_shot_cot = f"""You are a PII-tagging assistant. {common} {cot_instruct}

    {boundary_tags}

    ### EXAMPLES ###
    """
    for i, ex in enumerate(examples, start=1):
        few_shot_cot += (
            f"EXAMPLE TRANSCRIPT {i}\n"
            f"{ex['in']}\n\n"
            f"EXAMPLE PROCESSED TRANSCRIPT {i}\n"
            f"{ex['out']}\n\n"
            f"REASON {i}\n"
            f"{ex['reason']}\n\n"
            "---\n\n"
        )
    few_shot_cot += f"""### TRANSCRIPT ###
        {transcript}

        ### REASONING ###
        1. ...
        2. ...
        N. ...

        ### OUTPUT ###
    """.strip()

    templates = {
        "zero_shot_cot": f"""You are a PII-tagging assistant. {common} {cot_instruct}
            
            {boundary_tags}

            ### TRANSCRIPT ###
            {transcript}

            ### REASONING ###
            1. ...
            2. ...
            N. ...

            ### OUTPUT ###
        """.strip(),

        "few_shot_cot": few_shot_cot,

        "zero_shot_icl": f"""You are a PII-tagging assistant. {common}

            {boundary_tags}

            ### TRANSCRIPT ###
            {transcript}

            ### OUTPUT ###
        """.strip(),

        "few_shot_icl": f"""You are a PII-tagging assistant. {common}

            {boundary_tags}

            ### EXAMPLE ###
            Transcript: {examples[0]['in']}
            Output:     {examples[0]['out']}

            ---
            ### TRANSCRIPT ###
            {transcript}

            ### OUTPUT ###
        """.strip(),
        }

    return templates.get(method, templates["zero_shot_icl"])