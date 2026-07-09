"""Load config.yaml into plain dataclasses. One config file, no OmegaConf trio."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class DataCfg:
    raw_path: str = "data/raw/crag_task_1_and_2_dev_v4.jsonl.bz2"
    splits_dir: str = "splits"
    test_size: float = 0.5
    calib_size: float = 0.3
    stratify_on: list[str] = field(default_factory=lambda: ["domain", "question_type"])


@dataclass
class RetrieveCfg:
    embedder_model: str = "BAAI/bge-base-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "BAAI/bge-reranker-base"
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    top_k: int = 20
    top_n: int = 5
    rerank_tau: float = 0.0


@dataclass
class GenerateCfg:
    base_url_env: str = "LLM_BASE_URL"
    model_env: str = "LLM_MODEL"
    max_tokens: int = 256
    temperature: float = 0.0
    idk_string: str = "I don't know."


@dataclass
class NliCfg:
    model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class GateCfg:
    artifact_path: str = "artifacts/gate.joblib"
    calibration: str = "temperature"
    c: float = 1.0


@dataclass
class JudgeCfg:
    base_url_env: str = "JUDGE_BASE_URL"
    model_env: str = "JUDGE_MODEL"
    cache_path: str = "artifacts/judge_cache.jsonl"
    finance_tolerance: float = 0.01


@dataclass
class ServiceCfg:
    gen_cache_dir: str = "artifacts/gen_cache"
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class Config:
    seed: int = 13
    data: DataCfg = field(default_factory=DataCfg)
    retrieve: RetrieveCfg = field(default_factory=RetrieveCfg)
    generate: GenerateCfg = field(default_factory=GenerateCfg)
    nli: NliCfg = field(default_factory=NliCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    judge: JudgeCfg = field(default_factory=JudgeCfg)
    service: ServiceCfg = field(default_factory=ServiceCfg)


_SECTIONS = {
    "data": DataCfg,
    "retrieve": RetrieveCfg,
    "generate": GenerateCfg,
    "nli": NliCfg,
    "gate": GateCfg,
    "judge": JudgeCfg,
    "service": ServiceCfg,
}


def _build_section(cls, raw: dict[str, Any] | None):
    if not raw:
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML; falls back to dataclass defaults for missing keys."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
    return Config(
        seed=raw.get("seed", 13),
        **{name: _build_section(cls, raw.get(name)) for name, cls in _SECTIONS.items()},
    )


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
