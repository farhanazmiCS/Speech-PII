#!/usr/bin/env bash
set -e



### EXECUTING EXTRACTION ###

# Allowed tags (comma-separated)
ALLOWED="EMAIL,NRIC,CREDIT_CARD,PHONE,PASSPORT_NUM,BANK_ACCOUNT,CAR_PLATE,PERSON"

# Methods to extract from
METHODS=(
  zero_shot_icl
  few_shot_icl
  zero_shot_cot
  few_shot_cot
)

# n-best types
TYPES=(
  no_correct
  greedy
  2_best
  3_best
  4_best
  5_best
)

for type in "${TYPES[@]}"; do
  for method in "${METHODS[@]}"; do
    IN="../../data/tagged_transcripts/500_${type}_${method}.csv"
    OUT="../../data/triplets_new/triplets_500_${type}_${method}.csv"

    python cli.py extract \
      "${IN}" \
      "${OUT}"
  done
done