# 🧩 object-aligner — paper experiments

<p align="center">
  <a href="https://github.com/aic-factcheck">
    <img src="https://github.com/aic-factcheck.png" alt="AIC" width="120" />
  </a>
</p>

<p align="center">
  <em>Reference implementation of the experiments in the Object Aligner (OA) paper —
  intrinsic graph-metric validation and extrinsic GEPA prompt-optimization.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13%2B-blue" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  <img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange" />
  <a href="https://github.com/aic-factcheck/object_aligner"><img alt="Object Aligner" src="https://img.shields.io/badge/library-object__aligner-blueviolet" /></a>
</p>

It builds on the public OA library
([github.com/aic-factcheck/object_aligner](https://github.com/aic-factcheck/object_aligner));
the experiment code here is a thin layer of dataset loaders, GEPA adapters,
native-metric scorers, and a results aggregator.

---

## 📦 Install

Requires **Python 3.13+** and the [`uv`](https://docs.astral.sh/uv/) package manager.

```bash
uv sync                                # OA (from GitHub) + all deps + dev tools
# GEPA optimizer (not a uv source dep). Install from GitHub at the pinned
# commit the experiments were run against — the current PyPI release predates
# the `gepa.lm` module the reflection-LM wrapper needs, so `pip install gepa`
# alone fails with `ModuleNotFoundError: No module named 'gepa.lm'`.
uv pip install 'gepa[full] @ git+https://github.com/gepa-ai/gepa@df291bd8a3b99c70d2ff067ee2e89e0cdb81406a'
cp .env.example .env                   # then fill in your keys / endpoints
```

`uv sync` pulls Object Aligner straight from its public GitHub repo (see
`[tool.uv.sources]` in `pyproject.toml`). Everything runs inside the project
`.venv/` via `uv run …`.

> ℹ️ If a newer PyPI `gepa` release later includes the `gepa.lm` module you can
> switch to `uv pip install 'gepa[full]'`; until then use the pinned GitHub commit
> above (it is the exact GEPA the paper's runs used).

### 🤖 Models / endpoints

Both LMs are addressed over **OpenAI-compatible chat-completions** APIs, configured
in `config/experiment.yaml` and overridable by environment variables (see
`.env.example`):

- **Reflection LM** (the GEPA prompt proposer): the paper uses a frozen **GPT-5** via
  the OpenAI API — set `OPENAI_API_KEY`.
- **Task LM** (the model being optimized): the paper serves local **Gemma-4-26B-A4B-it**
  and **Gemma-4-E4B-it** (with **Qwen3.5-35B-A3B** / **Qwen3.5-9B** as a robustness
  check) behind a local vLLM server. Point `OAEXP_TASK_LM_BASE_URL` /
  `OAEXP_TASK_LM_MODEL` at your own endpoint.

The experiments only need an OpenAI-compatible endpoint for the task LM — serve the
model however you like. For reference, the exact recipe we used is recorded in
**`config/cllm/`**: one YAML per model giving the vLLM invocation and the SLURM job
(partition, GPU/CPU/memory, walltime, modules) it ran under on our cluster. These
files were consumed by an in-house launcher (a thin middle layer that submits the vLLM
job to SLURM and exposes its OpenAI-compatible endpoint); the launcher itself is not
part of this release, but the YAMLs document the precise model IDs, vLLM flags
(e.g. `--max-model-len`), and resources behind each reported run. Read them as the
authoritative source for *how the task models were run*, and adapt to your own
infrastructure.

---

## 🗂️ Repository layout

```
config/   LM config (experiment.yaml) + author cllm launch configs
data/     Schemas, seed prompts, and shipped result aggregates (bulk is regenerated)
scripts/  Experiment entry points — prepare / run / eval / report
src/      Shared library, importable as `object_aligner_exp.*`
```

Under `data/`, only **inputs** (OA `*.jsonc` schemas, `seed_prompt.txt`, dataset
READMEs) and the small **shipped aggregates** are version-controlled. The bulky
corpora, train/val/test splits, and per-run GEPA logs are **regenerated** by the
scripts below and are git-ignored.

---

## 📚 Datasets

The paper evaluates two axes. Each dataset ID maps to a paper-facing name:

| Axis | Paper name | Dataset ID | Prepare → split dir |
|------|------------|-----------------|---------------------|
| 🕸️ Referential | O2G PW | `synra_codex_noobf`  | `data/synra_codex/splits/gepa_synra_codex/noobf` |
| 🕸️ Referential | O2G PN | `synra_codex_noobf6` | `data/synra_codex/splits/gepa_synra_codex/noobf6` |
| 🕸️ Referential | O2G CW | `synra_codex_obf`    | `data/synra_codex/splits/gepa_synra_codex/pilot` |
| 🕸️ Referential | O2G CN | `synra_codex_obf6`   | `data/synra_codex/splits/gepa_synra_codex/obf6` |
| 🕸️ Referential | SciERC | `scierc_native_exact`| `data/scierc/splits/gepa_scierc/native` |
| 🕸️ Referential | BioRED | `biored`             | `data/biored/splits/gepa_biored/main` |
| 🕸️ Referential | Bio AMR| `amr_bio`            | `data/amr/splits/gepa_amr_bio/main` |
| 🔢 Order | F2O S | `synra_sort_stated_tight` | `data/synra_sort/splits/gepa_synra_sort/stated_tight` |
| 🔢 Order | F2O H | `synra_sort_hidden_tight` | `data/synra_sort/splits/gepa_synra_sort/hidden_tight` |
| 🔢 Order | NATURAL PLAN | `natural_plan_trip` | `data/natural_plan/splits/gepa_trip_planning/main` |
| 🔢 Order | ROCStories | `sentence_ordering_rocstories` | `data/sentence_ordering/splits/gepa_rocstories/main` |

(O2G = *Org2Graph*, the synthetic referential dataset; variant codes P/C =
plain/coded identifier values, W/N = wide/narrow vocabulary. F2O = *Facts2Order*,
the synthetic ordering dataset; S/H = stated/hidden sort key.)

### ⚙️ Preparing the data

All prepare scripts are seeded; re-running them reproduces the paper's splits exactly
under `data/<dataset>/{preprocessed,splits}/`.

The convenience wrapper **`prepare_all_datasets.sh`** (in the repo root) runs every
command below in order — build all splits in one go with:

```bash
bash prepare_all_datasets.sh
```

It downloads the real corpora (SciERC, BioRED, Natural Plan, ROCStories, Bio AMR) and
generates the synthetic ones, so a network connection is required on first run.
Equivalently, run the individual steps:

```bash
# Synthetic — generated, no download (Org2Graph; seed 20260520 reproduces the paper)
B=data/synra_codex/splits/gepa_synra_codex
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
```

---

## 🔬 Intrinsic experiments (no LLM)

These perturbation probes are fully deterministic. The shipped
`data/{synra_codex,synra_sort}/intrinsic/v2/` directories already contain the exact
outputs the paper figures use, so you can inspect them directly — or regenerate them:

```bash
# Referential alignment (Org2Graph): relabel-invariance + monotone damage
uv run python scripts/prepare_synra_codex_intrinsic.py --root data/synra_codex/intrinsic/v2
uv run python scripts/score_synra_codex_intrinsic.py   --root data/synra_codex/intrinsic/v2
uv run python scripts/per_op_synra_codex_intrinsic.py  --root data/synra_codex/intrinsic/v2
uv run python scripts/report_synra_codex_intrinsic.py  --root data/synra_codex/intrinsic/v2
uv run python scripts/figures_synra_codex_intrinsic.py --root data/synra_codex/intrinsic/v2

# Order regime (Facts2Order): smooth Kendall-distance grading, length-change edits
uv run python scripts/prepare_synra_sort_intrinsic.py  --root data/synra_sort/intrinsic/v2
uv run python scripts/score_synra_sort_intrinsic.py    --root data/synra_sort/intrinsic/v2
uv run python scripts/report_synra_sort_intrinsic.py   --root data/synra_sort/intrinsic/v2
uv run python scripts/figures_synra_sort_intrinsic.py  --root data/synra_sort/intrinsic/v2
```

Outputs (per `--root`): `test_scores.jsonl`, `test_summary.json`,
`per_op_scores.jsonl` (RA), `report.md`, `summary.md`, and figures under `imgs/`.

---

## 🚀 Extrinsic experiments (GEPA)

Each run optimizes a task prompt with GEPA, using OA as the reward. The contrast is
between an **OA schema with the mechanism on** and its ablation:

- 🕸️ **Referential axis** — referential-alignment schema (`*_ra` / `*_exact`) vs the
  literal-id `strict` ablation.
- 🔢 **Order axis** — order-sensitive `fixed` schema vs the order-blind `hungarian`
  ablation.

…crossed with two reward **arms**: `oa_score` (scalar OA score only) and `oa_feedback`
(scalar score **plus** ranked deterministic corrections, K=5). The feedback-breadth
figure additionally sweeps `oa_feedback{1,10,all}` and the `oa_gold` upper bound. Each
cell is run for **10 seeds** (`--seed 0…9`), with budget **800** for the synthetic
datasets and **600** for the real ones.

> 🐚 The `run_*.sh` / `evaluate_*.sh` shell wrappers are **not** part of this repo
> (they orchestrate the authors' local vLLM cluster). The exact underlying commands
> are below; they are the ground truth the paper was run with.

Set your task-LM endpoint first (`.env` or inline), e.g.
`export OAEXP_TASK_LM_BASE_URL=http://127.0.0.1:8000/v1` and
`export OAEXP_TASK_LM_MODEL=google/gemma-4-26B-A4B-it`.

**Referential datasets** (each run, plus its mirror with `--schema`/`--cross-schema`
swapped, is one leg):

```bash
# Org2Graph — generic NL→graph runner; --split-dir selects the variant (see table)
uv run python scripts/run_gepa_synra_nl.py --arm oa_feedback \
    --schema data/synra_codex/schemas/synra_codex_ra.jsonc \
    --cross-schema data/synra_codex/schemas/synra_codex_strict.jsonc \
    --split-dir data/synra_codex/splits/gepa_synra_codex/pilot \
    --seed-prompt data/synra_codex/seed_prompt.txt \
    --max-metric-calls 800 --task-temperature 0.0 --seed 0 \
    --run-name gepa_synra_codex_obf/oa_feedback/ra/gemma4-26b/s0

# SciERC (native split, exact schemas)
uv run python scripts/run_gepa_scierc.py --arm oa_feedback \
    --schema data/scierc/schemas/scierc_exact.jsonc \
    --cross-schema data/scierc/schemas/scierc_strict_exact.jsonc \
    --split-dir data/scierc/splits/gepa_scierc/native \
    --seed-prompt data/scierc/seed_prompt.txt \
    --max-metric-calls 600 --seed 0 \
    --run-name gepa_scierc_native_exact/oa_feedback/ra/gemma4-26b/s0

# BioRED (exact schemas)
uv run python scripts/run_gepa_biored.py --arm oa_feedback \
    --schema data/biored/schemas/biored_exact.jsonc \
    --cross-schema data/biored/schemas/biored_strict_exact.jsonc \
    --split-dir data/biored/splits/gepa_biored/main \
    --seed-prompt data/biored/seed_prompt.txt \
    --max-metric-calls 600 --seed 0 \
    --run-name gepa_biored/oa_feedback/ra/gemma4-26b/s0

# Bio AMR
uv run python scripts/run_gepa_amr.py --arm oa_feedback \
    --schema data/amr/schemas/amr_ra.jsonc \
    --cross-schema data/amr/schemas/amr_strict.jsonc \
    --split-dir data/amr/splits/gepa_amr_bio/main \
    --seed-prompt data/amr/bio/seed_prompt.txt \
    --max-metric-calls 600 --seed 0 \
    --run-name gepa_amr_bio/oa_feedback/ra/gemma4-26b/s0
```

**Order datasets** (`fixed` vs `hungarian`; the sentence-ordering runner drives both
ROCStories and Facts2Order):

```bash
# Facts2Order (stated_tight / hidden_tight) and ROCStories
uv run python scripts/run_gepa_sentence_ordering.py --arm oa_feedback \
    --schema data/synra_sort/schemas/synra_sort_fixed.jsonc \
    --cross-schema data/synra_sort/schemas/synra_sort_hungarian.jsonc \
    --split-dir data/synra_sort/splits/gepa_synra_sort/stated_tight \
    --seed-prompt data/synra_sort/seed_prompt.txt \
    --max-metric-calls 800 --seed 0 \
    --run-name gepa_synra_sort_stated_tight/oa_feedback/fixed/gemma4-26b/s0

uv run python scripts/run_gepa_sentence_ordering.py --arm oa_feedback \
    --schema data/sentence_ordering/schemas/sentence_ordering_fixed.jsonc \
    --cross-schema data/sentence_ordering/schemas/sentence_ordering_hungarian.jsonc \
    --split-dir data/sentence_ordering/splits/gepa_rocstories/main \
    --seed-prompt data/sentence_ordering/rocstories/seed_prompt.txt \
    --max-metric-calls 600 --seed 0 \
    --run-name gepa_sentence_ordering_rocstories/oa_feedback/fixed/gemma4-26b/s0

# Natural Plan (trip)
uv run python scripts/run_gepa_natural_plan.py --arm oa_feedback \
    --schema data/natural_plan/trip/schemas/trip_fixed.jsonc \
    --cross-schema data/natural_plan/trip/schemas/trip_hungarian.jsonc \
    --split-dir data/natural_plan/splits/gepa_trip_planning/main \
    --seed-prompt data/natural_plan/trip/seed_prompt.txt \
    --max-metric-calls 600 --seed 0 \
    --run-name gepa_natural_plan_trip/oa_feedback/fixed/gemma4-26b/s0
```

Each run writes to `data/runs/<run-name>/` (`config.json`, `summary.json`,
`holdout_scores.jsonl`, task/reflection LM logs, trajectory). Use `--help` on any
runner for the full flag set (concurrency, retries, etc.).

Run all 10 seeds and both arms (`oa_score`, `oa_feedback`), and both schema legs, to
reproduce a full table cell.

---

## 📊 Evaluation and native metrics

The held-out OA score is computed during the run on the default `test` split. To
re-score on the larger `test_full` split and compute **leaderboard-native** metrics:

> 📝 **Note.** `test_full` was *not* used for the numbers reported in the paper. The
> smaller `test` splits turned out to be sufficient — the larger split gave essentially
> the same variance at much higher compute cost — so all reported results are on
> `test`. `evaluate_full.py` and the `test_full.jsonl` splits are kept here only for
> completeness / optional larger-sample re-scoring.

```bash
# OA holdout on test_full (any run dir)
uv run python scripts/evaluate_full.py --run-dir data/runs/<run-name> \
    --split data/scierc/splits/gepa_scierc/native/test_full.jsonl

# Native metrics: Smatch F1 (AMR), relation F1 (SciERC/BioRED), PMR/Kendall-τ
# (sentence ordering / Facts2Order), exact-match solve rate (Natural Plan)
uv run python scripts/eval_amr.py             --run-dir data/runs/<amr-run>
uv run python scripts/eval_scierc.py          --run-dir data/runs/<scierc-run>
uv run python scripts/eval_biored.py          --run-dir data/runs/<biored-run>
uv run python scripts/eval_natural_plan.py    --run-dir data/runs/<np-run>
uv run python scripts/eval_sentence_ordering.py --run-dir data/runs/<so-run>   # also Facts2Order
uv run python scripts/score_scierc.py         --run-dir data/runs/<scierc-run> # PL-Marker-style
```

`--schema PATH` / `--cross-schema PATH` (accepted by every runner and the baselines)
set the primary OA schema and any extra schemas to score against; the primary schema's
filename stem becomes the run-name suffix and the key under
`summary["holdout"]["cross_scores"]`.

---

## 📈 Results

Aggregate every run under `data/runs/` into the paper's source-of-record tables:

```bash
uv run python scripts/build_results_report.py --all-datasets --all-models
```

This writes `data/results/`:

- **`results_tables.json`** — the machine-readable, table-keyed aggregate the paper's
  figures and tables are generated from. **This file is shipped**, so it already
  contains the published numbers; rebuilding it requires your own GEPA runs
  (statistically equivalent, not byte-identical).
- `results.json` / `results.md` — the same aggregates as a human-readable report.

`scripts/report_runs.py` prints a quick per-family status summary of `data/runs/`
(completed vs incomplete runs); it is read-only and safe to run anytime.

The paper's figure/table generators (which consume `results_tables.json` and the
intrinsic `v2/` dirs) live in the paper sources, not in this repository.

---

## 📜 Citation

If you use this code or Object Aligner in academic work, please cite the paper (in
preparation):

```bibtex
@misc{drchal2026objectaligner,
  title  = {Object Aligner: A Configurable JSON Schema Similarity Score for Graphs,
            Applied to LLM Prompt Optimization},
  author = {Drchal, Jan},
  year   = {2026},
  note   = {Reference implementation: https://github.com/aic-factcheck/object_aligner}
}
```

Object Aligner library: <https://github.com/aic-factcheck/object_aligner>.

---

## 📝 License

[MIT](LICENSE)
