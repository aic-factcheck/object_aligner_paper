#!/bin/bash

# Synthetic — generated, no download (Org2Graph; seed 20260520 reproduces the paper)
export B=data/synra_codex/splits/gepa_synra_codex
uv run python scripts/prepare_synra_codex.py                 --id-prefix synra_codex_obf    --out-dir $B/pilot   --seed 20260520
uv run python scripts/prepare_synra_codex.py --no-obfuscate  --id-prefix synra_codex_noobf  --out-dir $B/noobf   --seed 20260520
uv run python scripts/prepare_synra_codex.py --vocab small   --id-prefix synra_codex_obf6   --out-dir $B/obf6    --seed 20260520
uv run python scripts/prepare_synra_codex.py --no-obfuscate --vocab small --id-prefix synra_codex_noobf6 --out-dir $B/noobf6 --seed 20260520

# Synthetic — Facts2Order (the paper uses the two *_tight builds)
uv run python scripts/prepare_synra_sort.py --key-mode stated --closeness tight
uv run python scripts/prepare_synra_sort.py --key-mode hidden --closeness tight

# Real — auto-download
uv run python scripts/prepare_scierc.py            # downloads SciERC; builds pilot + native splits
uv run python scripts/prepare_biored.py            # downloads BioRED
uv run python scripts/prepare_natural_plan.py --task trip      # clones the Natural Plan repo
uv run python scripts/prepare_sentence_ordering.py --corpus rocstories   # HuggingFace
uv run python scripts/prepare_amr.py --variant bio   # downloads Bio AMR (archived ISI corpus)
