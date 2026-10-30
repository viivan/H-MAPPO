from __future__ import annotations

import copy
import heapq
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from model import DEFAULT_CONFIG, ChannelModel, OffloadingEnvironment, ResourceManager
from model.dataset import (
    infer_num_subtasks_from_path,
    list_train_gv_files,
    list_test_gv_files,
    list_public_v3_train_gv_files,
    list_public_v3_test_gv_files,
    list_elevator_v1_train_gv_files,
    list_elevator_v1_test_gv_files,
    list_elevator_v1_scenario_gv_files,
    list_elevator_train_gv_files,
    list_elevator_test_gv_files,
    list_safety_train_gv_files,
    list_safety_test_gv_files,
    list_dispatch_train_gv_files,
    list_dispatch_test_gv_files,
    list_predmaint_train_gv_files,
    list_predmaint_test_gv_files,
    load_dag_tasks,
)


class EventDrivenOffloadEnv:
    # 基础观测维度（不含 agent id / 资源状态）
    _OBS_TASK = 10

    def __init__(
        self,
        split: str = "train",
        num_agents: Optional[int] = None,
        seed: int = 42,
        sample_with_replacement: bool = False,
        min_subtasks: Optional[int] = None,
        max_subtasks: Optional[int] = None,
        curriculum_schedule: Optional[List[Dict]] = None,
        f_alloc_mode: str = "baseline",
        cloud_share_mode: str = "shared",
        cloud_share_factor: float = 0.3,
        cloud_share_floor: float = 0.4,
        heuristic_state_mode: str = "none",
        task_selection_mode: str = "urgency",
        urgency_progress_weight: float = 0.1,
        tmax_mode: str = "adaptive",
        tmax_slack: float = 1.5,
        tmax_min: float = 10.0,
        tmax_max: float = 120.0,
        deadline_penalty: float = 2.0,
        objective_alpha: Optional[float] = None,
        objective_beta: Optional[float] = None,
        soft_violation_coeff: float = 1.0,
        dataset_mode: str = "public",
        d_scale: float = 1e6,
        c_scale: float = 1e8,
        delta_scale: float = 1e6,
        system_profile: str = "default",
    ) -> None:
        self.cfg = self._build_system_config(system_profile)
        if objective_alpha is not None or objective_beta is not None:
            alpha = (
                float(objective_alpha)
                if objective_alpha is not None
                else 1.0 - float(objective_beta)
            )
            beta = (
                float(objective_beta)
                if objective_beta is not None
                else 1.0 - alpha
            )
            if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
                raise ValueError("objective_alpha and objective_beta must be in [0, 1]")
            if not np.isclose(alpha + beta, 1.0, atol=1e-9):
                raise ValueError("objective_alpha and objective_beta must sum to 1")
            self.cfg.objective.alpha = alpha
            self.cfg.objective.beta = beta
        self.split = split
        self.num_agents = num_agents if num_agents is not None else self.cfg.resource.total_elevators()
        self.seed = seed
        self.sample_with_replacement = sample_with_replacement
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.curriculum_schedule = curriculum_schedule or []
        self.f_alloc_mode = f_alloc_mode
        self.cloud_share_mode = cloud_share_mode
        self.cloud_share_factor = float(cloud_share_factor)
        self.cloud_share_floor = float(cloud_share_floor)
        self.heuristic_state_mode = heuristic_state_mode
        self.task_selection_mode = task_selection_mode
        self.urgency_progress_weight = float(urgency_progress_weight)
        self.tmax_mode = tmax_mode
        self.tmax_slack = tmax_slack
        self.tmax_min = tmax_min
        self.tmax_max = tmax_max
        self.deadline_penalty = deadline_penalty
        self.soft_violation_coeff = soft_violation_coeff
        self.dataset_mode = dataset_mode
        self.d_scale = float(d_scale)
        self.c_scale = float(c_scale)
        self.delta_scale = float(delta_scale)
        self.heuristic_action_masker = None
        self.rng = random.Random(seed)
        self.current_episode: int = 1

        # 数据集加载：根据 dataset_mode 选择对应的文件列表
        _train_loaders = {
            "public_v3": lambda: list_public_v3_train_gv_files(self.cfg.dataset),
            "elevator_v1": lambda: list_elevator_v1_train_gv_files(self.cfg.dataset),
            "elevator_v1_fault_diagnosis": lambda: list_elevator_v1_scenario_gv_files("train", "fault_diagnosis", self.cfg.dataset),
            "elevator_v1_dispatch": lambda: list_elevator_v1_scenario_gv_files("train", "dispatch", self.cfg.dataset),
            "elevator_v1_predictive_maintenance": lambda: list_elevator_v1_scenario_gv_files("train", "predictive_maintenance", self.cfg.dataset),
            "elevator": lambda: list_elevator_train_gv_files(self.cfg.dataset),
            "safety": lambda: list_safety_train_gv_files(self.cfg.dataset),
            "dispatch": lambda: list_dispatch_train_gv_files(self.cfg.dataset),
            "predmaint": lambda: list_predmaint_train_gv_files(self.cfg.dataset),
        }
        _test_loaders = {
            "public_v3": lambda: list_public_v3_test_gv_files(self.cfg.dataset),
            "elevator_v1": lambda: list_elevator_v1_test_gv_files(self.cfg.dataset),
            "elevator_v1_fault_diagnosis": lambda: list_elevator_v1_scenario_gv_files("test", "fault_diagnosis", self.cfg.dataset),
            "elevator_v1_dispatch": lambda: list_elevator_v1_scenario_gv_files("test", "dispatch", self.cfg.dataset),
            "elevator_v1_predictive_maintenance": lambda: list_elevator_v1_scenario_gv_files("test", "predictive_maintenance", self.cfg.dataset),
            "elevator": lambda: list_elevator_test_gv_files(self.cfg.dataset),
            "safety": lambda: list_safety_test_gv_files(self.cfg.dataset),
            "dispatch": lambda: list_dispatch_test_gv_files(self.cfg.dataset),
            "predmaint": lambda: list_predmaint_test_gv_files(self.cfg.dataset),
        }
        if split == "train":
            loader = _train_loaders.get(dataset_mode)
            self.dataset_files = loader() if loader else list_train_gv_files(self.cfg.dataset)
        else:
            loader = _test_loaders.get(dataset_mode)
            self.dataset_files = loader() if loader else list_test_gv_files(self.cfg.dataset)
        if not self.dataset_files:
            raise ValueError("未找到 DAG 数据集文件")

        self._file_sizes: Dict[Path, int] = {
            p: infer_num_subtasks_from_path(p) for p in self.dataset_files
        }
        # DAGTask 解析缓存：每个文件只 parse 一次，避免 pydotplus 反复 IO
        self._dag_task_cache: Dict[Path, object] = {}

        self.M: int = self.cfg.resource.num_base_stations
        self.action_dim: int = self.M + 2
        # obs = 10 task-feat + N agent-id + 2M server-state
        self.base_obs_dim: int = self._OBS_TASK + self.num_agents + 2 * self.M
        self.obs_dim: int = self.base_obs_dim + self._heuristic_feature_dim()
        # state = obs（单 agent，state = obs）
        self.state_dim: int = self.obs_dim

        # episode 状态（reset 初始化）
        self.env: Optional[OffloadingEnvironment] = None
        self.current_files: List[Path] = []
        self._topo_orders: List[List[int]] = []
        self._cursors: List[int] = []
        self._decisions: List[Dict[int, int]] = []
        self._f_allocs: List[Dict[int, float]] = []
        self._finish_times: List[Dict[int, float]] = []
        self._edge_intervals: List[List[Tuple[float, float, float]]] = []
        # 云端执行区间 [(start, end), ...]，用于按并发任务共享云端算力。
        self._cloud_intervals: List[Tuple[float, float]] = []
        self._energy: List[float] = []
        self._E_subtask_ref: List[List[float]] = []   # 每个子任务的本地能耗参考值

        # 当前待调度的 (agent_i, subtask_j)
        self._current_i: int = 0
        self._current_j: int = 0
        self.episode_step: int = 0

        # avail_actions 基础（全部开放），启发式 masker 可在每步覆盖
        self._avail_actions: np.ndarray = np.ones(self.action_dim, dtype=np.float32)

    @staticmethod
    def _build_system_config(system_profile: str):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        profile = (system_profile or "default").lower()
        if profile in {"default", "legacy", "base"}:
            return cfg
        if profile not in {"paper_public_v1", "paper_public_v2", "paper_public_v3"}:
            raise ValueError(f"未知 system_profile: {system_profile}")

        if profile == "paper_public_v3":
            cfg.channel.uplink_bandwidth_hz = 10.0e6
            cfg.channel.tx_power_w = 0.20
            cfg.channel.rx_power_w = 0.10
            cfg.channel.backhaul_delay_s = 0.15
            cfg.channel.default_inter_edge_bandwidth_bps = 100e6

            cfg.resource.edge_cpu_hz = [6.0e9, 6.0e9, 6.0e9]
            cfg.resource.cloud_cpu_hz = 20.0e9
            cfg.resource.local_cpu_hz = 1.0e9
            cfg.resource.inter_edge_bandwidth_bps = 100e6
            cfg.resource.baseline_edge_alloc_hz = 2.0e9

            cfg.energy.kappa = 1.0e-27
            cfg.energy.edge_wait_power_w = 0.10
            cfg.energy.cloud_wait_power_w = 0.50
            cfg.energy.cloud_backhaul_energy_per_bit = 2.0e-10

            cfg.objective.alpha = 0.60
            cfg.objective.beta = 0.40
            cfg.objective.default_t_max_s = 30.0
            cfg.objective.constraint_penalty_coeff = 1.0
            return cfg

        cfg.channel.uplink_bandwidth_hz = 60.0e6
        cfg.channel.backhaul_delay_s = 0.35
        cfg.channel.default_inter_edge_bandwidth_bps = 200e6

        cfg.resource.edge_cpu_hz = [3.0e9, 4.5e9, 6.0e9]
        cfg.resource.cloud_cpu_hz = 7.0e9
        cfg.resource.local_cpu_hz = 0.75e9
        cfg.resource.inter_edge_bandwidth_bps = 200e6
        cfg.resource.baseline_edge_alloc_hz = 1.25e9

        cfg.energy.kappa = 6.0e-27
        cfg.energy.edge_wait_power_w = 0.12
        cfg.energy.cloud_wait_power_w = 1.15
        cfg.energy.cloud_backhaul_energy_per_bit = 1.5e-9

        cfg.objective.alpha = 0.55
        cfg.objective.beta = 0.45
        cfg.objective.default_t_max_s = 30.0
        cfg.objective.constraint_penalty_coeff = 1.0

        if profile == "paper_public_v2":
            cfg.channel.backhaul_delay_s = 0.55
            cfg.resource.baseline_edge_alloc_hz = 1.50e9
            cfg.objective.alpha = 0.60
            cfg.objective.beta = 0.40
        return cfg

    # ------------------------------------------------------------------ #
    #  公开属性
    # ------------------------------------------------------------------ #

    @property
    def N(self) -> int:
        return self.env.N if self.env else self.num_agents

    @property
    def max_episode_steps(self) -> int:
        if not self._topo_orders:
            return 1
        return sum(len(t) for t in self._topo_orders)

    def set_episode_index(self, episode: int) -> None:
        self.current_episode = max(1, int(episode))

    # ------------------------------------------------------------------ #
    #  内部：DAG/环境构建
    # ------------------------------------------------------------------ #

    def _active_size_bounds(self) -> Tuple[Optional[int], Optional[int]]:
        min_k = self.min_subtasks
        max_k = self.max_subtasks
        for stage in self.curriculum_schedule:
            until_ep = int(stage.get("until_episode", 0))
            if until_ep <= 0 or self.current_episode <= until_ep:
                min_k = int(stage["min_subtasks"]) if "min_subtasks" in stage else min_k
                max_k = int(stage["max_subtasks"]) if "max_subtasks" in stage else max_k
                break
        return min_k, max_k

    def _eligible_files(self) -> List[Path]:
        min_k, max_k = self._active_size_bounds()
        result = []
        for p in self.dataset_files:
            k = self._file_sizes[p]
            if min_k is not None and k < min_k:
                continue
            if max_k is not None and k > max_k:
                continue
            result.append(p)
        return result

    def _sample_files(self) -> List[Path]:
        candidates = self._eligible_files()
        if not candidates:
            raise ValueError("课程过滤后无可用 DAG 文件")
        if self.sample_with_replacement or len(candidates) < self.num_agents:
            return [self.rng.choice(candidates) for _ in range(self.num_agents)]
        return self.rng.sample(candidates, self.num_agents)

    def _estimate_reference_finish_time(
        self, env: OffloadingEnvironment, i: int
    ) -> float:

        dag = env.dag_tasks[i]
        topo = dag.topological_sort()

        share = max(1.0, self.num_agents / max(1, self.M))
        F_edge_min = float(min(es.F_max for es in env.rm.edge_servers))
        f_edge_eff = F_edge_min / (share + 1.0)
        f_edge_eff = max(f_edge_eff, 0.3 * float(env.rm.baseline_edge_alloc_hz))

        F_cloud = float(env.rm.cloud.F_cloud)
        if self.cloud_share_mode == "shared":
            f_cloud_eff = F_cloud / (1.0 + self.cloud_share_factor * self.num_agents)
            f_cloud_eff = max(f_cloud_eff, self.cloud_share_floor * F_cloud)
        else:
            f_cloud_eff = F_cloud

        decisions: Dict[int, int] = {}
        finish_times: Dict[int, float] = {}
        for j in topo:
            best_x = 0
            best_finish = float("inf")
            for x in range(self.action_dim):
                if 1 <= x <= self.M:
                    f_alloc = f_edge_eff
                else:
                    f_alloc = 0.0
                preds = dag.subtasks[j].predecessors
                if preds:
                    est = max(
                        finish_times[p] + env.ch.intermediate_tran_delay(
                            delta_bits=dag.get_edge_data_bits(p, j),
                            x_src=decisions[p], x_dst=x,
                            distance=env.distances[i],
                            home_m=env.home_servers[i],
                            M=env.M,
                            wire_bw_matrix=env.rm.bw_matrix,
                        )
                        for p in preds
                    )
                else:
                    est = 0.0

                if x == self.M + 1:
                    cloud_time = float(env.dag_tasks[i].subtasks[j].c) / max(f_cloud_eff, 1e-9)
                    total = env._tran_delay(i, j, x) + cloud_time
                else:
                    total = env.subtask_total_time(i, j, x, f_alloc)
                finish = est + total

                if finish < best_finish:
                    best_finish = finish
                    best_x = x
            decisions[j] = best_x
            finish_times[j] = best_finish
        exit_nodes = dag.get_exit_nodes()
        return float(max(finish_times[j] for j in exit_nodes))

    def _load_dag_tasks_cached(self, dag_files: List[Path]) -> List:
        """带缓存的 DAGTask 加载，每个文件只解析一次。"""
        result = []
        uncached = [p for p in dag_files if p not in self._dag_task_cache]
        if uncached:
            fresh = load_dag_tasks(uncached)
            for p, t in zip(uncached, fresh):
                self._dag_task_cache[p] = t
        for p in dag_files:
            result.append(self._dag_task_cache[p])
        return result

    def _build_env(self, dag_files: List[Path]) -> OffloadingEnvironment:
        """构建 OffloadingEnvironment，含 adaptive T_max。"""
        dag_tasks = self._load_dag_tasks_cached(dag_files)
        n = len(dag_tasks)
        channel_model = ChannelModel(self.cfg.channel)
        resource_manager = ResourceManager(cfg=self.cfg.resource)
        home_servers = self.cfg.default_home_servers()[:n]
        f_local_list = self.cfg.default_local_cpu_list()[:n]
        distances = self.cfg.default_distance_list()[:n]

        # 先用 static T_max 建一个临时 env 用于估算
        static_tmax = self.cfg.default_tmax_list()[:n]
        if self.tmax_mode == "dataset":
            missing = [i for i, dag in enumerate(dag_tasks) if dag.deadline_s is None]
            if missing:
                raise ValueError(f"DAG files missing deadline_s for task indexes: {missing}")
            t_max_list = [float(dag.deadline_s) for dag in dag_tasks]
        elif self.tmax_mode == "static":
            t_max_list = static_tmax
        else:
            temp_env = OffloadingEnvironment(
                dag_tasks=list(dag_tasks),  # 传副本，不影响缓存对象
                channel_model=channel_model,
                resource_manager=resource_manager,
                home_servers=home_servers,
                f_local_list=f_local_list,
                distances=distances,
                T_max_list=static_tmax,
                energy_cfg=self.cfg.energy,
                objective_cfg=self.cfg.objective,
            )
            t_max_list = []
            for i in range(n):
                ref = self._estimate_reference_finish_time(temp_env, i)
                tmax = float(
                    max(self.tmax_min, min(self.tmax_max, self.tmax_slack * ref))
                )
                t_max_list.append(tmax)

        return OffloadingEnvironment(
            dag_tasks=list(dag_tasks),
            channel_model=channel_model,
            resource_manager=resource_manager,
            home_servers=home_servers,
            f_local_list=f_local_list,
            distances=distances,
            T_max_list=t_max_list,
            energy_cfg=self.cfg.energy,
            objective_cfg=self.cfg.objective,
        )

    # ------------------------------------------------------------------ #
    #  内部：资源状态
    # ------------------------------------------------------------------ #

    def _get_load_avail(self) -> Tuple[np.ndarray, np.ndarray]:
        load = np.zeros(self.M, dtype=np.float32)
        avail = np.ones(self.M, dtype=np.float32)
        for m in range(self.M):
            F_max = self.env.rm.edge_servers[m].F_max
            ivs = self._edge_intervals[m]
            if not ivs:
                continue
            evts: List[Tuple[float, int, float]] = []
            for s, f_, g in ivs:
                evts.append((s, 1, g))
                evts.append((f_, -1, g))
            evts.sort(key=lambda e: (e[0], e[1]))
            cur = 0.0
            peak = 0.0
            for _, etype, g in evts:
                cur += etype * g
                peak = max(peak, cur)
            total = sum(g for _, _, g in ivs)
            if F_max > 0:
                load[m] = float(peak / F_max)
                avail[m] = float(max(0.0, 1.0 - total / F_max))
        return load, avail

    def _count_concurrent_on_edge(self, m: int, at_time: Optional[float]) -> int:
        if at_time is None:
            return 0
        return sum(1 for start, end, _ in self._edge_intervals[m] if start <= at_time < end)

    def _count_concurrent_on_cloud(self, at_time: Optional[float]) -> int:
        if at_time is None:
            return 0
        return sum(1 for start, end in self._cloud_intervals if start <= at_time < end)

    def _cloud_exec_time_dyn(self, i: int, j: int, at_time: Optional[float] = None) -> float:

        env = self.env
        F_cloud = float(env.rm.cloud.F_cloud)
        if F_cloud <= 0.0:
            return float("inf")
        c = float(env.dag_tasks[i].subtasks[j].c)
        if self.cloud_share_mode == "shared":
            concurrent = self._count_concurrent_on_cloud(at_time)
            shared = F_cloud / (1.0 + self.cloud_share_factor * concurrent)
            floor = self.cloud_share_floor * F_cloud
            effective = max(floor, shared)
            return c / max(effective, 1e-9)
        return c / F_cloud

    def _compute_f_alloc(self, m: int, at_time: Optional[float] = None) -> float:
        server = self.env.rm.edge_servers[m]
        if self.f_alloc_mode == "full":
            return float(server.F_max)
        if self.f_alloc_mode == "load_aware":
            concurrent = self._count_concurrent_on_edge(m, at_time)
            shared = float(server.F_max / (concurrent + 1))
            floor = float(0.3 * self.env.rm.baseline_edge_alloc_hz)
            return max(floor, shared)
        return float(self.env.rm.baseline_edge_alloc_hz)

    def _placement_energy(
        self,
        i: int,
        j: int,
        x: int,
        f_alloc: float,
        cloud_at_time: Optional[float] = None,
    ) -> float:
        """计算单个子任务在动作 x 下的设备侧能耗。"""
        env = self.env
        st = env.dag_tasks[i].subtasks[j]
        if x == 0:
            return float(env.kappa * st.c * (env.f_local[i] ** (env.energy_cfg.zeta - 1)))

        tx_energy = env.ch.transmission_energy(st.d, env.distances[i])
        upload_time = env.ch.upload_delay(st.d, env.distances[i])
        if 1 <= x <= self.M:
            exec_time = env._edge_exec_time(i, j, f_alloc)
            wait_energy = env.energy_cfg.edge_wait_power_w * (upload_time + exec_time)
            return float(tx_energy + wait_energy)

        # 云端：使用按并发共享后的执行时延，等待时间也相应变长
        cloud_time = self._cloud_exec_time_dyn(i, j, at_time=cloud_at_time)
        wait_energy = env.energy_cfg.cloud_wait_power_w * (upload_time + cloud_time)
        backhaul_energy = env.energy_cfg.cloud_backhaul_energy_per_bit * st.d
        return float(tx_energy + wait_energy + backhaul_energy)

    # ------------------------------------------------------------------ #
    #  内部：ready 队列 & 紧迫度
    # ------------------------------------------------------------------ #

    def _urgency(self, i: int) -> float:
        """
        agent i 当前子任务的紧迫度（越大越需要先调度）。
        = max(predecessor finish) / T_max[i]
        如果没有前驱则 = 0（马上可调度，用步骤比例打破平局）
        """
        cursor = self._cursors[i]
        topo = self._topo_orders[i]
        if cursor >= len(topo):
            return -1.0  # 已结束

        j = topo[cursor]
        env = self.env
        st = env.dag_tasks[i].subtasks[j]
        T_max_i = max(env.T_max[i], 1e-9)

        if st.predecessors:
            data_ready = max(self._finish_times[i].get(p, 0.0) for p in st.predecessors)
        else:
            data_ready = 0.0

        # 紧迫度 = 数据就绪时间 / deadline + 已完成进度（避免长期饥饿）
        progress = cursor / max(1, len(topo))
        return data_ready / T_max_i + self.urgency_progress_weight * progress

    def _pick_next_task(self) -> Optional[Tuple[int, int]]:
        """从所有未完成 agent 中选出紧迫度最高的 (i, j)。"""
        if self.task_selection_mode == "round_robin":
            for offset in range(self.N):
                i = (self._round_robin_cursor + offset) % self.N
                if self._cursors[i] < len(self._topo_orders[i]):
                    self._round_robin_cursor = (i + 1) % self.N
                    j = self._topo_orders[i][self._cursors[i]]
                    return i, j
            return None
        if self.task_selection_mode != "urgency":
            raise ValueError(f"Unknown task_selection_mode: {self.task_selection_mode}")
        best_urgency = -2.0
        best_i = -1
        for i in range(self.N):
            u = self._urgency(i)
            if u >= 0.0 and u > best_urgency:
                best_urgency = u
                best_i = i
        if best_i < 0:
            return None
        j = self._topo_orders[best_i][self._cursors[best_i]]
        return best_i, j

    # ------------------------------------------------------------------ #
    #  内部：观测构造
    # ------------------------------------------------------------------ #

    def estimate_action_outcome(self, i: int, j: int, x: int) -> Dict[str, float]:

        env = self.env
        st = env.dag_tasks[i].subtasks[j]
        T_max_i = max(env.T_max[i], 1e-9)

        f_alloc = 0.0
        static_ratio = 0.0
        peak_ratio = 0.0
        exec_start = 0.0
        exec_end = 0.0

        if not st.predecessors:
            est_j = 0.0
        else:
            est_j = max(
                self._finish_times[i][p] + env.ch.intermediate_tran_delay(
                    delta_bits=env.dag_tasks[i].get_edge_data_bits(p, j),
                    x_src=self._decisions[i][p],
                    x_dst=x,
                    distance=env.distances[i],
                    home_m=env.home_servers[i],
                    M=env.M,
                    wire_bw_matrix=env.rm.bw_matrix,
                )
                for p in st.predecessors
            )

        cloud_exec_start: Optional[float] = None
        if 1 <= x <= self.M:
            m = x - 1
            t_tran = env._tran_delay(i, j, x)
            exec_start = est_j + t_tran
            f_alloc = self._compute_f_alloc(m, at_time=exec_start)
            F_max = max(env.rm.edge_servers[m].F_max, 1e-9)
            static_alloc = sum(
                f for dec_i, alloc_i in zip(self._decisions, self._f_allocs)
                for jj, act in dec_i.items()
                if 1 <= act <= self.M and act - 1 == m
                for f in [alloc_i.get(jj, 0.0)]
            )
            static_ratio = float((static_alloc + f_alloc) / F_max)
        elif x == self.M + 1:
            cloud_exec_start = est_j + env._tran_delay(i, j, x)

        # 计算总时延：边缘走标准接口；云端用动态共享时延
        if x == 0:
            t_total_j = env._local_exec_time(i, j)
        elif 1 <= x <= self.M:
            t_total_j = env.subtask_total_time(i, j, x, f_alloc)
        else:
            cloud_time = self._cloud_exec_time_dyn(i, j, at_time=cloud_exec_start)
            t_total_j = env._tran_delay(i, j, x) + cloud_time
        finish_j = est_j + t_total_j

        if 1 <= x <= self.M:
            m = x - 1
            exec_end = exec_start + env._edge_exec_time(i, j, f_alloc)
            F_max = max(env.rm.edge_servers[m].F_max, 1e-9)
            # 峰值负载估算
            ivs = list(self._edge_intervals[m]) + [(exec_start, exec_end, f_alloc)]
            evts = []
            for s, f_, g in ivs:
                evts.append((s, 1, g)); evts.append((f_, -1, g))
            evts.sort(key=lambda e: (e[0], e[1]))
            cur = 0.0; peak = 0.0
            for _, etype, g in evts:
                cur += etype * g; peak = max(peak, cur)
            peak_ratio = float(peak / F_max)

        energy = self._placement_energy(i, j, x, f_alloc, cloud_at_time=cloud_exec_start)
        local_energy_ref = max(
            1e-12, env.kappa * st.c * (env.f_local[i] ** (env.energy_cfg.zeta - 1))
        )

        return {
            "est": float(est_j),
            "finish": float(finish_j),
            "delay_ratio": float(finish_j / T_max_i),
            "energy": float(energy),
            "energy_ratio": float(energy / local_energy_ref),
            "f_alloc": float(f_alloc),
            "exec_start": float(exec_start),
            "exec_end": float(exec_end),
            "static_ratio": float(static_ratio),
            "peak_ratio": float(peak_ratio),
            "c1_violation": float(max(0.0, finish_j / T_max_i - 1.0)),
            "c2_violation": float(max(0.0, static_ratio - 1.0)),
            "c3_violation": float(max(0.0, peak_ratio - 1.0)),
        }

    def _heuristic_feature_dim(self) -> int:
        """启发式状态增强维度。"""
        if self.heuristic_state_mode in {"strong_outcome", "zero_outcome"}:
            return 6 * self.action_dim + 1
        if self.heuristic_state_mode in ("none", None):
            return 0
        raise ValueError(f"未知 heuristic_state_mode: {self.heuristic_state_mode}")

    def _make_heuristic_features(self, i: int, j: int) -> np.ndarray:
        """
        构造 strong_outcome 启发式状态特征。
        """
        if self.heuristic_state_mode in ("none", None):
            return np.zeros(0, dtype=np.float32)
        if self.heuristic_state_mode == "zero_outcome":
            return np.zeros(6 * self.action_dim + 1, dtype=np.float32)
        if self.heuristic_state_mode != "strong_outcome":
            raise ValueError(f"未知 heuristic_state_mode: {self.heuristic_state_mode}")

        env = self.env
        outcomes = [self.estimate_action_outcome(i, j, a) for a in range(self.action_dim)]
        delay_ratio = np.asarray(
            [o["delay_ratio"] for o in outcomes],
            dtype=np.float32,
        )
        energy_ratio = np.asarray(
            [o["energy_ratio"] for o in outcomes],
            dtype=np.float32,
        )
        c1_violation = np.asarray(
            [o["c1_violation"] for o in outcomes],
            dtype=np.float32,
        )
        c3_violation = np.asarray(
            [o["c3_violation"] for o in outcomes],
            dtype=np.float32,
        )
        costs = np.asarray(
            [env.alpha_w * o["finish"] + env.beta_w * o["energy"] for o in outcomes],
            dtype=np.float32,
        )
        costs = np.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)
        cost_max = float(costs.max())
        cost_min = float(costs.min())
        denom = max(cost_max - cost_min, 1e-6)
        heuristic_score = (cost_max - costs) / denom

        best_idx = int(np.argmin(costs))
        best_onehot = np.zeros(self.action_dim, dtype=np.float32)
        best_onehot[best_idx] = 1.0

        sorted_costs = np.sort(costs)
        if len(sorted_costs) > 1:
            score_margin = np.asarray([(sorted_costs[1] - sorted_costs[0]) / denom], dtype=np.float32)
        else:
            score_margin = np.asarray([0.0], dtype=np.float32)

        return np.concatenate(
            [
                np.clip(delay_ratio, 0.0, 5.0) / 5.0,
                np.clip(energy_ratio, 0.0, 5.0) / 5.0,
                np.clip(c1_violation, 0.0, 5.0) / 5.0,
                np.clip(c3_violation, 0.0, 5.0) / 5.0,
                heuristic_score.astype(np.float32),
                best_onehot,
                np.clip(score_margin, 0.0, 1.0),
            ]
        ).astype(np.float32)

    def _get_load_state(self):
        return self._get_load_avail()

    def _get_avail_actions(self) -> np.ndarray:
        avail = self._avail_actions.copy()[np.newaxis, :]  # [1, action_dim]
        if self.heuristic_action_masker is not None:
            mask = self.heuristic_action_masker(self).astype(np.float32)
            if mask.shape == avail.shape:
                avail = avail * mask
                # 至少保留一个动作（本地执行兜底）
                if avail.sum() < 0.5:
                    avail[0, 0] = 1.0
        return avail

    def _make_obs(self, i: int, j: int) -> np.ndarray:
        env = self.env
        st = env.dag_tasks[i].subtasks[j]
        T_max = env.T_max[i]
        n = max(1, len(self._topo_orders[i]))
        cursor = self._cursors[i]

        t_local = env._local_exec_time(i, j)
        t_ul = env.ch.upload_delay(st.d, env.distances[i])
        t_cloud = env._cloud_exec_time(i, j)

        if st.predecessors:
            data_ready = max(self._finish_times[i].get(p, 0.0) for p in st.predecessors)
        else:
            data_ready = 0.0

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = float(st.d) / max(self.d_scale, 1.0)
        obs[1] = float(st.c) / max(self.c_scale, 1.0)
        obs[2] = float(st.delta) / max(self.delta_scale, 1.0)
        obs[3] = t_local / max(T_max, 1e-9)
        obs[4] = t_ul / max(T_max, 1e-9)
        obs[5] = t_cloud / max(T_max, 1e-9)
        obs[6] = float(len(st.predecessors)) / n
        obs[7] = float(len(st.successors)) / n
        obs[8] = float(cursor) / n
        obs[9] = float(min(1.0, data_ready / max(T_max, 1e-9)))

        # agent one-hot
        if i < self.num_agents:
            obs[self._OBS_TASK + i] = 1.0

        # server states
        load, avail = self._get_load_avail()
        base = self._OBS_TASK + self.num_agents
        obs[base: base + self.M] = load
        obs[base + self.M: base + 2 * self.M] = avail

        h_features = self._make_heuristic_features(i, j)
        if h_features.size:
            h_base = self.base_obs_dim
            obs[h_base: h_base + h_features.size] = h_features

        return obs

    # ------------------------------------------------------------------ #
    #  内部：调度执行 & 即时奖励
    # ------------------------------------------------------------------ #

    def _schedule_and_reward(self, i: int, j: int, x: int) -> float:
        """调度子任务 (i, j) 到位置 x，返回即时奖励。"""
        env = self.env
        st = env.dag_tasks[i].subtasks[j]
        T_max_i = max(env.T_max[i], 1e-9)

        # EST (C4：数据依赖)
        if not st.predecessors:
            est_j = 0.0
        else:
            est_j = max(
                self._finish_times[i][p] + env.ch.intermediate_tran_delay(
                    delta_bits=env.dag_tasks[i].get_edge_data_bits(p, j),
                    x_src=self._decisions[i][p],
                    x_dst=x,
                    distance=env.distances[i],
                    home_m=env.home_servers[i],
                    M=env.M,
                    wire_bw_matrix=env.rm.bw_matrix,
                )
                for p in st.predecessors
            )

        f_alloc = 0.0
        cloud_exec_start: Optional[float] = None
        if 1 <= x <= self.M:
            m = x - 1
            exec_start_probe = est_j + env._tran_delay(i, j, x)
            f_alloc = self._compute_f_alloc(m, at_time=exec_start_probe)
        elif x == self.M + 1:
            cloud_exec_start = est_j + env._tran_delay(i, j, x)

        if x == 0:
            t_total_j = env._local_exec_time(i, j)
        elif 1 <= x <= self.M:
            t_total_j = env.subtask_total_time(i, j, x, f_alloc)
        else:
            cloud_time = self._cloud_exec_time_dyn(i, j, at_time=cloud_exec_start)
            t_total_j = env._tran_delay(i, j, x) + cloud_time
        finish_j = est_j + t_total_j

        # 更新状态
        self._decisions[i][j] = x
        self._f_allocs[i][j] = f_alloc
        self._finish_times[i][j] = finish_j

        # 更新边缘执行区间
        if 1 <= x <= self.M:
            m = x - 1
            t_tran = env._tran_delay(i, j, x)
            exec_start = est_j + t_tran
            exec_end = exec_start + env._edge_exec_time(i, j, f_alloc)
            self._edge_intervals[m].append((exec_start, exec_end, f_alloc))
        elif x == self.M + 1:
            cloud_exec_end = cloud_exec_start + self._cloud_exec_time_dyn(
                i, j, at_time=cloud_exec_start
            )
            self._cloud_intervals.append((float(cloud_exec_start), float(cloud_exec_end)))

        # 能耗
        energy = self._placement_energy(i, j, x, f_alloc, cloud_at_time=cloud_exec_start)
        self._energy[i] += energy

        # ---- 即时奖励----
        rt_ratio = finish_j / T_max_i
        e_ref = self._E_subtask_ref[i][j]
        en_ratio = energy / max(e_ref, 1e-12)

        rt_score = float(np.clip(1.0 - rt_ratio, -1.0, 1.0))
        en_score = float(np.clip(1.0 - en_ratio, -1.0, 1.0))

        reward = env.alpha_w * rt_score + env.beta_w * en_score
        if rt_ratio > 1.0:
            reward -= self.deadline_penalty * (rt_ratio - 1.0)

        return float(reward)

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #

    def reset(self) -> Dict[str, np.ndarray]:
        self.current_files = self._sample_files()
        self.env = self._build_env(self.current_files)
        N = self.env.N

        self._topo_orders = [self.env.dag_tasks[i].topological_sort() for i in range(N)]
        self._cursors = [0] * N
        self._decisions = [{} for _ in range(N)]
        self._f_allocs = [{} for _ in range(N)]
        self._finish_times = [{} for _ in range(N)]
        self._edge_intervals = [[] for _ in range(self.M)]
        self._cloud_intervals = []
        self._energy = [0.0] * N
        self._round_robin_cursor = 0

        # 预计算每个子任务的本地能耗参考
        env = self.env
        self._E_subtask_ref = []
        for i in range(N):
            refs = []
            for j in range(len(env.dag_tasks[i].subtasks)):
                st = env.dag_tasks[i].subtasks[j]
                e_local = max(
                    1e-12,
                    env.kappa * st.c * (env.f_local[i] ** (env.energy_cfg.zeta - 1))
                )
                refs.append(e_local)
            self._E_subtask_ref.append(refs)

        self.episode_step = 0

        # 选出第一个待调度任务
        nxt = self._pick_next_task()
        if nxt is None:
            raise RuntimeError("reset 后没有可调度的子任务")
        self._current_i, self._current_j = nxt

        obs = self._make_obs(self._current_i, self._current_j)
        return {
            "obs":           obs[np.newaxis, :],
            "state":         obs,
            "avail_actions": self._get_avail_actions(),
            "active_mask":   np.ones(1, dtype=np.float32),
        }

    def step(self, actions: np.ndarray) -> Dict[str, object]:
        assert self.env is not None, "请先调用 reset()"
        i, j = self._current_i, self._current_j
        x = int(np.clip(int(actions[0]), 0, self.M + 1))

        reward = self._schedule_and_reward(i, j, x)
        self._cursors[i] += 1
        self.episode_step += 1

        # 检查是否全部调度完
        all_done = all(self._cursors[ii] >= len(self._topo_orders[ii]) for ii in range(self.N))

        if all_done:

            info = self._terminal_info(reward)
            dones = np.ones(1, dtype=np.float32)

            obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            nxt = self._pick_next_task()
            if nxt is None:
                all_done = True
                info = self._terminal_info(reward)
                dones = np.ones(1, dtype=np.float32)
                obs = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                self._current_i, self._current_j = nxt
                obs = self._make_obs(self._current_i, self._current_j)
                dones = np.zeros(1, dtype=np.float32)
                info = {"terminal": False, "reward": reward}

        rewards = np.array([reward], dtype=np.float32)
        return {
            "obs":           obs[np.newaxis, :],
            "state":         obs,
            "avail_actions": self._get_avail_actions(),
            "active_mask":   np.ones(1, dtype=np.float32),
            "rewards":       rewards,
            "dones":         dones,
            "infos":         info,
        }

    # ------------------------------------------------------------------ #
    #  终局信息
    # ------------------------------------------------------------------ #

    def _terminal_info(self, last_step_reward: float) -> Dict:
        env = self.env
        N = env.N

        T_list = [
            max(self._finish_times[i].get(j, 0.0)
                for j in env.dag_tasks[i].get_exit_nodes())
            for i in range(N)
        ]
        E_list = self._energy[:]
        T_ratio = [T_list[i] / max(env.T_max[i], 1e-9) for i in range(N)]
        on_time_ratio = float(np.mean([1.0 if r <= 1.0 else 0.0 for r in T_ratio]))
        c1_excess = float(np.mean([max(0.0, r - 1.0) for r in T_ratio]))

        constraints = env.check_constraints(self._decisions, self._f_allocs, T_list)

        return {
            "terminal":          True,
            "reward":            float(last_step_reward),
            "agent_on_time_ratio": on_time_ratio,
            "c1_excess":         c1_excess,
            "ep_feasible":       1.0 if constraints.get("feasible", False) else 0.0,
            "T_list":            [float(t) for t in T_list],
            "avg_tmax":          float(np.mean(env.T_max)),
            "selected_files":    [str(p) for p in self.current_files],
        }
