"""
资源模型
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from model.config import DEFAULT_CONFIG, ResourceConfig


# --------------------------------------------------------------------------- #
#  边缘服务器
# --------------------------------------------------------------------------- #

class EdgeServer:

    def __init__(self, server_id: int, F_max: float, rho_th: Optional[float] = None):
        if rho_th is None:
            rho_th = DEFAULT_CONFIG.resource.overload_threshold
        self.server_id = server_id
        self.F_max = F_max
        self.rho_th = rho_th
        self._allocated: Dict[Tuple[int, int], float] = {}
        self._peak_alloc_snapshot = 0.0
        self._load_ratio_snapshot = 0.0

    # ------------------------------------------------------------------ #
    #  资源状态查询
    # ------------------------------------------------------------------ #

    @property
    def allocated_total(self) -> float:
        return sum(self._allocated.values())

    @property
    def available_capacity(self) -> float:
        return max(0.0, self.F_max - self._peak_alloc_snapshot)

    @property
    def load_ratio(self) -> float:
        return self._load_ratio_snapshot

    @property
    def is_overloaded(self) -> bool:
        return self.load_ratio >= self.rho_th

    # ------------------------------------------------------------------ #
    #  资源分配/释放
    # ------------------------------------------------------------------ #

    def allocate(self,
                 task_id: int,
                 subtask_id: int,
                 f_alloc: float) -> bool:

        if f_alloc < 0:
            raise ValueError(f"f_alloc 不能为负数，收到 {f_alloc}")

        key = (task_id, subtask_id)
        prev = self._allocated.get(key, 0.0)
        if f_alloc > self.F_max + 1e-9:
            return False

        self._allocated[key] = f_alloc
        return True

    def release(self, task_id: int, subtask_id: int) -> None:
        self._allocated.pop((task_id, subtask_id), None)

    def get_allocation(self, task_id: int, subtask_id: int) -> float:
        return self._allocated.get((task_id, subtask_id), 0.0)

    def reset(self) -> None:

        self._allocated.clear()
        self._peak_alloc_snapshot = 0.0
        self._load_ratio_snapshot = 0.0

    def update_load_snapshot(self, peak_alloc: float) -> None:

        self._peak_alloc_snapshot = max(0.0, peak_alloc)
        self._load_ratio_snapshot = (
            self._peak_alloc_snapshot / self.F_max if self.F_max > 0 else float('inf')
        )

    # ------------------------------------------------------------------ #
    #  执行时延
    # ------------------------------------------------------------------ #

    def execution_time(self, c: float, f_alloc: float) -> float:

        if f_alloc <= 0:
            return float('inf')
        return c / f_alloc

    def __repr__(self) -> str:
        return (f"EdgeServer(m={self.server_id}, "
                f"F_max={self.F_max:.2e}, "
                f"rho={self.load_ratio:.3f}, "
                f"overloaded={self.is_overloaded})")


# --------------------------------------------------------------------------- #
#  云服务器
# --------------------------------------------------------------------------- #

class CloudServer:

    def __init__(self, F_cloud: float):
        self.F_cloud = F_cloud

    def execution_time(self, c: float) -> float:

        return c / self.F_cloud

    def __repr__(self) -> str:
        return f"CloudServer(F_cloud={self.F_cloud:.2e})"


# --------------------------------------------------------------------------- #
#  资源管理器
# --------------------------------------------------------------------------- #

class ResourceManager:

    def __init__(self,
                 num_base_stations: Optional[int] = None,
                 F_max_list: Optional[List[float]] = None,
                 F_cloud: Optional[float] = None,
                 bw_matrix: Optional[np.ndarray] = None,
                 rho_th: Optional[float] = None,
                 cfg: Optional[ResourceConfig] = None):

        cfg = cfg if cfg is not None else DEFAULT_CONFIG.resource
        num_base_stations = num_base_stations if num_base_stations is not None else cfg.num_base_stations
        F_max_list = F_max_list if F_max_list is not None else list(cfg.edge_cpu_hz)
        F_cloud = F_cloud if F_cloud is not None else cfg.cloud_cpu_hz
        rho_th = rho_th if rho_th is not None else cfg.overload_threshold

        if len(F_max_list) != num_base_stations:
            raise ValueError("F_max_list 长度必须等于 num_base_stations")

        self.M = num_base_stations
        self.cfg = cfg
        self.baseline_edge_alloc_hz = cfg.baseline_edge_alloc_hz

        self.edge_servers: List[EdgeServer] = [
            EdgeServer(m, F_max_list[m], rho_th)
            for m in range(num_base_stations)
        ]

        self.cloud = CloudServer(F_cloud)

        if bw_matrix is not None:
            if bw_matrix.shape != (num_base_stations, num_base_stations):
                raise ValueError("bw_matrix 形状必须为 (M, M)")
            self.bw_matrix = bw_matrix.astype(np.float64)
        else:
            default_bw = cfg.inter_edge_bandwidth_bps
            self.bw_matrix = np.full(
                (num_base_stations, num_base_stations),
                default_bw,
                dtype=np.float64,
            )
            np.fill_diagonal(self.bw_matrix, 0.0)

    # ------------------------------------------------------------------ #
    #  查询接口
    # ------------------------------------------------------------------ #

    def get_edge_server(self, m: int) -> EdgeServer:

        return self.edge_servers[m]

    def inter_edge_bandwidth(self, src_m: int, dst_m: int) -> float:

        return float(self.bw_matrix[src_m, dst_m])

    def find_least_loaded_edge(self, exclude: Optional[int] = None) -> int:

        min_load = float('inf')
        best_m = 0
        for m, es in enumerate(self.edge_servers):
            if m == exclude:
                continue
            if es.load_ratio < min_load:
                min_load = es.load_ratio
                best_m = m
        return best_m

    def load_state(self) -> np.ndarray:

        return np.array([es.load_ratio for es in self.edge_servers], dtype=np.float32)

    def available_capacity_state(self) -> np.ndarray:

        return np.array([es.available_capacity for es in self.edge_servers],
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  约束检查
    # ------------------------------------------------------------------ #

    def check_resource_constraint(self,
                                   all_decisions: List[Dict[int, int]],
                                   all_f_allocs: List[Dict[int, float]]) -> Dict[int, bool]:

        result = {}
        for m in range(self.M):
            valid = True
            for decisions, f_allocs in zip(all_decisions, all_f_allocs):
                for j, x in decisions.items():
                    if 1 <= x <= self.M and (x - 1) == m:
                        f_alloc = f_allocs.get(j, 0.0)
                        if f_alloc > self.edge_servers[m].F_max + 1e-9:
                            valid = False
                            break
                if not valid:
                    break
            result[m] = valid
        return result

    def check_load_constraint(self) -> Dict[int, bool]:

        return {m: (es.load_ratio <= 1.0 + 1e-9)
                for m, es in enumerate(self.edge_servers)}

    def update_dynamic_load_state(self, dynamic_load: Dict[int, Dict[str, object]]) -> None:

        for m, es in enumerate(self.edge_servers):
            peak_alloc = 0.0
            if m in dynamic_load:
                peak_alloc = float(dynamic_load[m].get('peak_alloc', 0.0))
            es.update_load_snapshot(peak_alloc)

    # ------------------------------------------------------------------ #
    #  重置
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        for es in self.edge_servers:
            es.reset()

    def __repr__(self) -> str:
        lines = [f"ResourceManager(M={self.M})"]
        for es in self.edge_servers:
            lines.append(f"  {es}")
        lines.append(f"  {self.cloud}")
        return "\n".join(lines)
