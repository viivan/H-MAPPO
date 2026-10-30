from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from model.config import DEFAULT_CONFIG, DatasetConfig
from model.task import DAGTask


def list_gv_files(subdirs: Iterable[str], cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    files: List[Path] = []
    for subdir in subdirs:
        folder = cfg.resolve_subdir(subdir)
        files.extend(sorted(folder.glob(f"*{cfg.file_suffix}")))
    return files


def list_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.train_subdirs, cfg)


def list_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.test_subdirs, cfg)


def list_public_v3_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:
    """List standardized DAGGen V3 training files."""
    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.public_v3_train_subdirs, cfg)


def list_public_v3_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:
    """List standardized DAGGen V3 test files."""
    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.public_v3_test_subdirs, cfg)


def list_elevator_v1_gv_files(
    split: str,
    cfg: Optional[DatasetConfig] = None,
    scenario: Optional[str] = None,
) -> List[Path]:
    """List the canonical Elevator-DAG-v1 files from its isolated workspace."""
    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    root = cfg.project_root / "elevator_experiments" / "data" / "elevator_dag_v1" / split
    if scenario:
        root = root / scenario
    return sorted(root.glob(f"**/*{cfg.file_suffix}"))


def list_elevator_v1_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:
    return list_elevator_v1_gv_files("train", cfg)


def list_elevator_v1_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:
    return list_elevator_v1_gv_files("test", cfg)


def list_elevator_v1_scenario_gv_files(
    split: str, scenario: str, cfg: Optional[DatasetConfig] = None
) -> List[Path]:
    return list_elevator_v1_gv_files(split, cfg, scenario=scenario)


def list_elevator_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.elevator_train_subdirs, cfg)


def list_elevator_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.elevator_test_subdirs, cfg)


def load_dag_tasks(file_paths: Iterable[Path], start_task_id: int = 0) -> List[DAGTask]:

    dag_tasks: List[DAGTask] = []
    for offset, file_path in enumerate(file_paths):
        dag_tasks.append(DAGTask.from_dot_file(str(file_path), task_id=start_task_id + offset))
    return dag_tasks




def list_safety_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.safety_train_subdirs, cfg)


def list_safety_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.safety_test_subdirs, cfg)


def list_dispatch_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.dispatch_train_subdirs, cfg)


def list_dispatch_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.dispatch_test_subdirs, cfg)


def list_predmaint_train_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.predmaint_train_subdirs, cfg)


def list_predmaint_test_gv_files(cfg: Optional[DatasetConfig] = None) -> List[Path]:

    cfg = cfg if cfg is not None else DEFAULT_CONFIG.dataset
    return list_gv_files(cfg.predmaint_test_subdirs, cfg)


def infer_num_subtasks_from_path(file_path: Path) -> int:

    candidates = [file_path.parent.name, file_path.name]
    for text in candidates:
        m = re.search(r"daggen_v3[_\.](\d+)", text)
        if m:
            return int(m.group(1))
        m = re.fullmatch(r"n(10|15|20|25)", text)
        if m:
            return int(m.group(1))

        m = re.search(r"random(\d+)", text)
        if m:
            return int(m.group(1))
        m = re.search(r"random\.(\d+)\.", text)
        if m:
            return int(m.group(1))

        m = re.search(r"elevator(\d+)", text)
        if m:
            return int(m.group(1))

        if "safety_monitor" in text:
            return 9

        m = re.search(r"smart_dispatch.*_N(\d+)", text)
        if m:
            return int(m.group(1))

        if "predictive_maintenance" in text:
            return 6
    raise ValueError(f"无法从路径推断 DAG 节点数: {file_path}")
