"""Experiment-wide configuration.

LM settings come from a YAML config file (default ``config/experiment.yaml``).
Environment variables ``OAEXP_TASK_LM_*`` / ``OAEXP_REFLECTION_LM_*`` override
matching YAML keys; ``OAEXP_CONFIG`` selects a different YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()  # picks up .env if present; harmless if not

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"


@dataclass(frozen=True)
class LMConfig:
    """OpenAI-compatible endpoint description.

    ``max_tokens`` is optional and follows three-way semantics:

      * ``None``  — YAML didn't set anything; the script's CLI default applies.
      * ``0``     — explicit "no limit"; the API call omits ``max_tokens`` /
                    ``max_completion_tokens`` so the model uses its own default.
      * positive  — that exact cap.
    """

    model: str
    base_url: str
    api_key_env: str | None = None  # name of env var holding the key; None = no key
    max_tokens: int | None = None   # see class docstring; None=use CLI default, 0=unlimited

    def api_key(self) -> str:
        if self.api_key_env is None:
            # OpenAI SDK still wants a non-empty string for local servers.
            return "not-needed"
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set; "
                f"cannot reach LM {self.model!r} at {self.base_url!r}."
            )
        return key


def _coerce_api_key_env(raw: Any) -> str | None:
    """YAML `null` / `~` / empty-string all mean 'no key'."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip() == "":
        return None
    return str(raw)


def _coerce_max_tokens(raw: Any, *, where: str) -> int | None:
    """YAML ``max_tokens`` accepts: missing/null -> None, int >=0 -> int."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is a subclass of int — reject explicitly
        raise ValueError(f"{where}: max_tokens must be an int (or null), got bool")
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"{where}: max_tokens must be >= 0 (use 0 for unlimited)")
        return raw
    raise ValueError(f"{where}: max_tokens must be an int or null, got {type(raw).__name__}")


def _load_lm(node: dict[str, Any], env_prefix: str) -> LMConfig:
    """Build an LMConfig from a YAML node, with env-var overrides on top."""
    if not isinstance(node, dict):
        raise ValueError(f"{env_prefix}: expected a mapping, got {type(node).__name__}")

    try:
        model = node["model"]
        base_url = node["base_url"]
    except KeyError as exc:
        raise ValueError(f"{env_prefix}: missing required key {exc.args[0]!r}") from exc

    api_key_env = _coerce_api_key_env(node.get("api_key_env"))
    max_tokens = _coerce_max_tokens(node.get("max_tokens"), where=env_prefix)

    # Env-var overrides take precedence over YAML.
    model = os.environ.get(f"{env_prefix}_MODEL", model)
    base_url = os.environ.get(f"{env_prefix}_BASE_URL", base_url)
    env_override = os.environ.get(f"{env_prefix}_API_KEY_ENV")
    if env_override is not None:
        api_key_env = _coerce_api_key_env(env_override)
    mt_env = os.environ.get(f"{env_prefix}_MAX_TOKENS")
    if mt_env is not None:
        max_tokens = _coerce_max_tokens(
            int(mt_env) if mt_env.strip() else None, where=f"{env_prefix} (env)"
        )

    return LMConfig(
        model=str(model),
        base_url=str(base_url),
        api_key_env=api_key_env,
        max_tokens=max_tokens,
    )


def _resolve_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    override = os.environ.get("OAEXP_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _load_config_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"LM config not found at {path}. "
            f"Copy {DEFAULT_CONFIG_PATH.name!r} or set OAEXP_CONFIG=<your.yaml>."
        )
    with path.open("r") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return doc


def _build_task_lm() -> LMConfig:
    doc = _load_config_doc(_resolve_config_path(None))
    return _load_lm(doc.get("task_lm", {}), "OAEXP_TASK_LM")


def _build_reflection_lm() -> LMConfig:
    doc = _load_config_doc(_resolve_config_path(None))
    return _load_lm(doc.get("reflection_lm", {}), "OAEXP_REFLECTION_LM")


@dataclass(frozen=True)
class ExpConfig:
    task_lm: LMConfig = field(default_factory=_build_task_lm)
    reflection_lm: LMConfig = field(default_factory=_build_reflection_lm)

    data_root: Path = REPO_ROOT / "data"
    runs_root: Path = REPO_ROOT / "data" / "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExpConfig":
        """Load explicitly from a specific YAML file (bypasses OAEXP_CONFIG)."""
        doc = _load_config_doc(Path(path).expanduser().resolve())
        return cls(
            task_lm=_load_lm(doc.get("task_lm", {}), "OAEXP_TASK_LM"),
            reflection_lm=_load_lm(doc.get("reflection_lm", {}), "OAEXP_REFLECTION_LM"),
        )

    @property
    def rebel_raw(self) -> Path:
        return self.data_root / "rebel" / "raw"

    @property
    def rebel_preprocessed(self) -> Path:
        return self.data_root / "rebel" / "preprocessed"

    @property
    def rebel_splits(self) -> Path:
        return self.data_root / "rebel" / "splits"

    @property
    def rebel_schema_path(self) -> Path:
        return self.data_root / "rebel" / "schemas" / "rebel.jsonc"

    @property
    def rebel_seed_prompt_path(self) -> Path:
        return self.data_root / "rebel" / "seed_prompt.txt"

    @property
    def gepa_rebel_pilot(self) -> Path:
        return self.rebel_splits / "gepa_rebel" / "pilot"

    @property
    def scierc_raw(self) -> Path:
        return self.data_root / "scierc" / "raw"

    @property
    def scierc_preprocessed(self) -> Path:
        return self.data_root / "scierc" / "preprocessed"

    @property
    def scierc_splits(self) -> Path:
        return self.data_root / "scierc" / "splits"

    @property
    def scierc_schema_path(self) -> Path:
        return self.data_root / "scierc" / "schemas" / "scierc.jsonc"

    @property
    def scierc_strict_schema_path(self) -> Path:
        return self.data_root / "scierc" / "schemas" / "scierc_strict.jsonc"

    @property
    def scierc_seed_prompt_path(self) -> Path:
        return self.data_root / "scierc" / "seed_prompt.txt"

    @property
    def gepa_scierc_pilot(self) -> Path:
        return self.scierc_splits / "gepa_scierc" / "pilot"

    @property
    def gepa_scierc_native(self) -> Path:
        return self.scierc_splits / "gepa_scierc" / "native"

    # --- BioRED (document-level biomedical relation extraction) ------------
    # DocRED-shaped task (text -> typed coref-entities + relations). Single
    # variant: the gold entity `id` is the opaque normalized concept accession
    # (Entrez/MeSH/dbSNP/...), so a literal-id relation compare is starved while
    # referential alignment routes by name+type. Raw BioC-JSON under raw/,
    # preprocessed JSONL under preprocessed/ (both gitignored).

    @property
    def biored_root(self) -> Path:
        return self.data_root / "biored"

    @property
    def biored_raw(self) -> Path:
        return self.biored_root / "raw"

    @property
    def biored_preprocessed(self) -> Path:
        return self.biored_root / "preprocessed"

    @property
    def biored_schema_path(self) -> Path:
        return self.biored_root / "schemas" / "biored.jsonc"

    @property
    def biored_strict_schema_path(self) -> Path:
        return self.biored_root / "schemas" / "biored_strict.jsonc"

    @property
    def biored_exact_schema_path(self) -> Path:
        return self.biored_root / "schemas" / "biored_exact.jsonc"

    @property
    def biored_strict_exact_schema_path(self) -> Path:
        return self.biored_root / "schemas" / "biored_strict_exact.jsonc"

    @property
    def biored_seed_prompt_path(self) -> Path:
        return self.biored_root / "seed_prompt.txt"

    @property
    def gepa_biored_main(self) -> Path:
        return self.biored_root / "splits" / "gepa_biored" / "main"

    # --- WEC-Eng (cross-document event coreference) ------------------------
    # Clustering task (a list of tagged event mentions -> a partition into event
    # clusters). Mapped onto OA as a membership multigraph: mentions are the
    # idScope (aligned by the echoed unique marker), each event carries its
    # members as refs. The gold mention id is the opaque WEC mention_id, so a
    # literal-id member compare is starved (strict near-degenerate) while
    # referential alignment routes the partition. Raw gold-mention JSON under
    # raw/, synthesised instance pools under preprocessed/ (both gitignored).

    @property
    def wec_eng_root(self) -> Path:
        return self.data_root / "wec_eng"

    @property
    def wec_eng_raw(self) -> Path:
        return self.wec_eng_root / "raw"

    @property
    def wec_eng_preprocessed(self) -> Path:
        return self.wec_eng_root / "preprocessed"

    @property
    def wec_eng_schema_path(self) -> Path:
        return self.wec_eng_root / "schemas" / "wec_eng.jsonc"

    @property
    def wec_eng_strict_schema_path(self) -> Path:
        return self.wec_eng_root / "schemas" / "wec_eng_strict.jsonc"

    @property
    def wec_eng_seed_prompt_path(self) -> Path:
        return self.wec_eng_root / "seed_prompt.txt"

    @property
    def gepa_wec_eng_main(self) -> Path:
        return self.wec_eng_root / "splits" / "gepa_wec_eng" / "main"

    # --- DocRED / Re-DocRED (document-level relation extraction) -----------
    # SciERC-shaped task (text -> typed coref-entities + relations). Two label
    # variants share ONE schema family (idScope/ref): `pcode` uses the raw
    # Wikidata property codes, `obf` uses the persisted obfuscation bijection
    # (data/docred/obf_map.json). Raw *_revised.json under raw/, preprocessed
    # JSONL per variant under preprocessed/<variant>/ (both gitignored).

    @property
    def docred_root(self) -> Path:
        return self.data_root / "docred"

    @property
    def docred_raw(self) -> Path:
        return self.docred_root / "raw"

    @property
    def docred_preprocessed(self) -> Path:
        return self.docred_root / "preprocessed"

    @property
    def docred_obf_map_path(self) -> Path:
        return self.docred_root / "obf_map.json"

    @property
    def docred_schema_path(self) -> Path:
        return self.docred_root / "schemas" / "docred.jsonc"

    @property
    def docred_strict_schema_path(self) -> Path:
        return self.docred_root / "schemas" / "docred_strict.jsonc"

    @property
    def docred_exact_schema_path(self) -> Path:
        return self.docred_root / "schemas" / "docred_exact.jsonc"

    @property
    def docred_strict_exact_schema_path(self) -> Path:
        return self.docred_root / "schemas" / "docred_strict_exact.jsonc"

    @property
    def docred_seed_prompt_path(self) -> Path:
        return self.docred_root / "seed_prompt.txt"

    @property
    def gepa_docred_pcode_main(self) -> Path:
        return self.docred_root / "splits" / "gepa_docred_pcode" / "main"

    @property
    def gepa_docred_obf_main(self) -> Path:
        return self.docred_root / "splits" / "gepa_docred_obf" / "main"

    @property
    def recipeflow_raw(self) -> Path:
        return self.data_root / "recipeflow" / "raw"

    @property
    def recipeflow_preprocessed(self) -> Path:
        return self.data_root / "recipeflow" / "preprocessed"

    @property
    def recipeflow_splits(self) -> Path:
        return self.data_root / "recipeflow" / "splits"

    @property
    def recipeflow_schema_path(self) -> Path:
        return self.data_root / "recipeflow" / "schemas" / "recipeflow.jsonc"

    @property
    def recipeflow_strict_schema_path(self) -> Path:
        return self.data_root / "recipeflow" / "schemas" / "recipeflow_strict.jsonc"

    @property
    def recipeflow_seed_prompt_path(self) -> Path:
        return self.data_root / "recipeflow" / "seed_prompt.txt"

    @property
    def gepa_recipeflow_pilot(self) -> Path:
        return self.recipeflow_splits / "gepa_recipeflow" / "pilot"

    @property
    def cro_raw(self) -> Path:
        return self.data_root / "cro" / "raw"

    @property
    def cro_preprocessed(self) -> Path:
        return self.data_root / "cro" / "preprocessed"

    @property
    def cro_splits(self) -> Path:
        return self.data_root / "cro" / "splits"

    @property
    def cro_schema_path(self) -> Path:
        return self.data_root / "cro" / "schemas" / "cro.jsonc"

    @property
    def cro_strict_schema_path(self) -> Path:
        return self.data_root / "cro" / "schemas" / "cro_strict.jsonc"

    @property
    def cro_seed_prompt_path(self) -> Path:
        return self.data_root / "cro" / "seed_prompt.txt"

    @property
    def gepa_cro_pilot(self) -> Path:
        return self.cro_splits / "gepa_cro" / "pilot"

    @property
    def synra_basic_root(self) -> Path:
        return self.data_root / "synra_basic"

    @property
    def synra_basic_schema_path(self) -> Path:
        return self.synra_basic_root / "schemas" / "synra_basic.jsonc"

    @property
    def synra_basic_strict_schema_path(self) -> Path:
        return self.synra_basic_root / "schemas" / "synra_basic_strict.jsonc"

    @property
    def synra_basic_v1(self) -> Path:
        return self.synra_basic_root / "v1"

    @property
    def synra_codex_root(self) -> Path:
        return self.data_root / "synra_codex"

    @property
    def synra_codex_ra_schema_path(self) -> Path:
        return self.synra_codex_root / "schemas" / "synra_codex_ra.jsonc"

    @property
    def synra_codex_strict_schema_path(self) -> Path:
        return self.synra_codex_root / "schemas" / "synra_codex_strict.jsonc"

    @property
    def synra_codex_intrinsic_v1(self) -> Path:
        return self.synra_codex_root / "intrinsic" / "v1"

    @property
    def synra_codex_intrinsic_v2(self) -> Path:
        """Randomized-generation redesign of v1 (random sizes / twin density /
        edge densities instead of the inherited GEPA cell grid; no splits)."""
        return self.synra_codex_root / "intrinsic" / "v2"

    @property
    def synra_nl_root(self) -> Path:
        return self.data_root / "synra_nl"

    @property
    def synra_nl_schema_path(self) -> Path:
        return self.synra_nl_root / "schemas" / "synra_nl.jsonc"

    @property
    def synra_nl_strict_schema_path(self) -> Path:
        return self.synra_nl_root / "schemas" / "synra_nl_strict.jsonc"

    @property
    def synra_nl_seed_prompt_path(self) -> Path:
        return self.synra_nl_root / "seed_prompt.txt"

    @property
    def gepa_synra_nl_pilot(self) -> Path:
        return self.synra_nl_root / "splits" / "gepa_synra_nl" / "pilot"

    @property
    def gepa_synra_nl_hard(self) -> Path:
        return self.synra_nl_root / "splits" / "gepa_synra_nl" / "hard"

    # --- NATURAL PLAN (Trip Planning + Meeting Planning) ------------------
    # Both tasks live under one dataset root; each carries its own seed prompt,
    # fixed/hungarian schema pair, and splits. Raw JSON is shared in raw/.

    @property
    def natural_plan_root(self) -> Path:
        return self.data_root / "natural_plan"

    @property
    def natural_plan_raw(self) -> Path:
        return self.natural_plan_root / "raw"

    @property
    def natural_plan_preprocessed(self) -> Path:
        return self.natural_plan_root / "preprocessed"

    @property
    def trip_fixed_schema_path(self) -> Path:
        return self.natural_plan_root / "trip" / "schemas" / "trip_fixed.jsonc"

    @property
    def trip_hungarian_schema_path(self) -> Path:
        return self.natural_plan_root / "trip" / "schemas" / "trip_hungarian.jsonc"

    @property
    def trip_seed_prompt_path(self) -> Path:
        return self.natural_plan_root / "trip" / "seed_prompt.txt"

    @property
    def gepa_trip_planning_main(self) -> Path:
        return self.natural_plan_root / "splits" / "gepa_trip_planning" / "main"

    @property
    def meeting_fixed_schema_path(self) -> Path:
        return self.natural_plan_root / "meeting" / "schemas" / "meeting_fixed.jsonc"

    @property
    def meeting_hungarian_schema_path(self) -> Path:
        return self.natural_plan_root / "meeting" / "schemas" / "meeting_hungarian.jsonc"

    @property
    def meeting_seed_prompt_path(self) -> Path:
        return self.natural_plan_root / "meeting" / "seed_prompt.txt"

    @property
    def gepa_meeting_planning_main(self) -> Path:
        return self.natural_plan_root / "splits" / "gepa_meeting_planning" / "main"

    # --- Sentence ordering (ROCStories + arXiv abstracts) -----------------
    # Two corpora share ONE corpus-agnostic schema pair (wire shape
    # {"order":[int]}); each has its own seed prompt + split dir. Preprocessed
    # JSONL under preprocessed/ (gitignored).

    @property
    def sentence_ordering_root(self) -> Path:
        return self.data_root / "sentence_ordering"

    @property
    def sentence_ordering_preprocessed(self) -> Path:
        return self.sentence_ordering_root / "preprocessed"

    @property
    def sentence_ordering_fixed_schema_path(self) -> Path:
        return self.sentence_ordering_root / "schemas" / "sentence_ordering_fixed.jsonc"

    @property
    def sentence_ordering_hungarian_schema_path(self) -> Path:
        return self.sentence_ordering_root / "schemas" / "sentence_ordering_hungarian.jsonc"

    @property
    def rocstories_seed_prompt_path(self) -> Path:
        return self.sentence_ordering_root / "rocstories" / "seed_prompt.txt"

    @property
    def arxiv_seed_prompt_path(self) -> Path:
        return self.sentence_ordering_root / "arxiv" / "seed_prompt.txt"

    @property
    def gepa_rocstories_main(self) -> Path:
        return self.sentence_ordering_root / "splits" / "gepa_rocstories" / "main"

    @property
    def gepa_arxiv_main(self) -> Path:
        return self.sentence_ordering_root / "splits" / "gepa_arxiv" / "main"

    # --- synra_sort (synthetic sort-by-stated-key ordering) ----------------
    # A fully synthetic ordering task: each example lists N items each with an
    # explicitly stated sortable key; gold is the index permutation that sorts
    # them. Same `{"indices":[...]}` wire shape as sentence_ordering, so the
    # fixed/hungarian schemas and sentence_ordering_eval apply unchanged. Built
    # for the fixed-order alignment study (and usable as a GEPA task).

    @property
    def synra_sort_root(self) -> Path:
        return self.data_root / "synra_sort"

    @property
    def synra_sort_preprocessed(self) -> Path:
        return self.synra_sort_root / "preprocessed"

    @property
    def synra_sort_fixed_schema_path(self) -> Path:
        return self.synra_sort_root / "schemas" / "synra_sort_fixed.jsonc"

    @property
    def synra_sort_hungarian_schema_path(self) -> Path:
        return self.synra_sort_root / "schemas" / "synra_sort_hungarian.jsonc"

    @property
    def synra_sort_seed_prompt_path(self) -> Path:
        return self.synra_sort_root / "seed_prompt.txt"

    @property
    def gepa_synra_sort_main(self) -> Path:
        return self.synra_sort_root / "splits" / "gepa_synra_sort" / "main"

    @property
    def synra_sort_intrinsic_v1(self) -> Path:
        return self.synra_sort_root / "intrinsic" / "v1"

    @property
    def synra_sort_intrinsic_v2(self) -> Path:
        """Randomized-generation redesign of v1 (random N / key type /
        closeness / distractors instead of the cell product; no splits)."""
        return self.synra_sort_root / "intrinsic" / "v2"

    # --- PlanBench (Blocksworld plan generation) --------------------------
    # Single Blocksworld task: fixed/hungarian schema pair + seed prompt + one
    # split dir. Raw HF data is cached under raw/, preprocessed JSONL under
    # preprocessed/ (both gitignored like other datasets).

    @property
    def planbench_root(self) -> Path:
        return self.data_root / "planbench"

    @property
    def planbench_raw(self) -> Path:
        return self.planbench_root / "raw"

    @property
    def planbench_preprocessed(self) -> Path:
        return self.planbench_root / "preprocessed"

    @property
    def blocksworld_fixed_schema_path(self) -> Path:
        return self.planbench_root / "blocksworld" / "schemas" / "blocksworld_fixed.jsonc"

    @property
    def blocksworld_hungarian_schema_path(self) -> Path:
        return self.planbench_root / "blocksworld" / "schemas" / "blocksworld_hungarian.jsonc"

    @property
    def blocksworld_seed_prompt_path(self) -> Path:
        return self.planbench_root / "blocksworld" / "seed_prompt.txt"

    @property
    def gepa_blocksworld_main(self) -> Path:
        return self.planbench_root / "splits" / "gepa_blocksworld" / "main"

    @property
    def graphimg_root(self) -> Path:
        return self.data_root / "graphimg"

    @property
    def graphimg_strict_schema_path(self) -> Path:
        return self.graphimg_root / "schemas" / "graphimg_strict.jsonc"

    @property
    def graphimg_ra_schema_path(self) -> Path:
        return self.graphimg_root / "schemas" / "graphimg_ra.jsonc"

    @property
    def graphimg_v1_seed_prompt_path(self) -> Path:
        return self.graphimg_root / "v1" / "seed_prompt.txt"

    @property
    def graphimg_v2_seed_prompt_path(self) -> Path:
        return self.graphimg_root / "v2" / "seed_prompt.txt"

    @property
    def gepa_graphimg_v1(self) -> Path:
        # Local copy of graphimg's v1 release (mirrors dataset_graphimg/out/graphimg_v1/).
        # The data/ tree is gitignored; refresh from the source repo with:
        #   cp -R /Users/drchajan/devel/python/dataset_graphimg/out/graphimg_v1 \
        #         data/graphimg/v1
        return self.graphimg_root / "v1"

    # --- AMR (sentence → meaning graph) -----------------------------------
    # One corpus-agnostic ra/strict schema pair (wire shape root/nodes/relations);
    # each corpus variant (littleprince, …) has its own seed prompt + split dir.
    # Raw PENMAN under raw/, preprocessed JSONL under preprocessed/ (gitignored).

    @property
    def amr_root(self) -> Path:
        return self.data_root / "amr"

    @property
    def amr_raw(self) -> Path:
        return self.amr_root / "raw"

    @property
    def amr_preprocessed(self) -> Path:
        return self.amr_root / "preprocessed"

    @property
    def amr_ra_schema_path(self) -> Path:
        return self.amr_root / "schemas" / "amr_ra.jsonc"

    @property
    def amr_strict_schema_path(self) -> Path:
        return self.amr_root / "schemas" / "amr_strict.jsonc"

    @property
    def littleprince_seed_prompt_path(self) -> Path:
        return self.amr_root / "littleprince" / "seed_prompt.txt"

    @property
    def gepa_amr_littleprince_main(self) -> Path:
        return self.amr_root / "splits" / "gepa_amr_littleprince" / "main"

    @property
    def bio_seed_prompt_path(self) -> Path:
        return self.amr_root / "bio" / "seed_prompt.txt"

    @property
    def gepa_amr_bio_main(self) -> Path:
        return self.amr_root / "splits" / "gepa_amr_bio" / "main"

    # --- molecules (image → molecular graph; OCSR) ------------------------
    # Image-input dataset (graphimg-style split layout: train/validation/test
    # subdirs with labels.jsonl + images/). One ra/strict schema pair (wire
    # shape atoms/bonds); each variant (rendered, decimer) has its own seed
    # prompt + split dir. Raw/preprocessed under raw/ + preprocessed/ (gitignored).

    @property
    def molecules_root(self) -> Path:
        return self.data_root / "molecules"

    @property
    def molecules_raw(self) -> Path:
        return self.molecules_root / "raw"

    @property
    def molecules_preprocessed(self) -> Path:
        return self.molecules_root / "preprocessed"

    @property
    def molecules_ra_schema_path(self) -> Path:
        return self.molecules_root / "schemas" / "molecules_ra.jsonc"

    @property
    def molecules_strict_schema_path(self) -> Path:
        return self.molecules_root / "schemas" / "molecules_strict.jsonc"

    @property
    def molecules_rendered_seed_prompt_path(self) -> Path:
        return self.molecules_root / "rendered" / "seed_prompt.txt"

    @property
    def gepa_molecules_rendered_main(self) -> Path:
        return self.molecules_root / "splits" / "gepa_molecules_rendered" / "main"

    @property
    def molecules_decimer_seed_prompt_path(self) -> Path:
        return self.molecules_root / "decimer" / "seed_prompt.txt"

    @property
    def gepa_molecules_decimer_main(self) -> Path:
        return self.molecules_root / "splits" / "gepa_molecules_decimer" / "main"
