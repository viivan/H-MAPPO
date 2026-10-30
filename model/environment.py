"""
任务卸载环境
============

决策变量
--------
    x_{i,j} ∈ {0, 1, ..., M, M+1}
        0      : 本地执行
        1 ~ M  : 卸载到第 x 个边缘服务器（1-indexed，对应0-indexed下标 x-1）
        M+1    : 卸载到云端

    f_{i,j}^alloc >= 0
        卸载到边缘服务器时分配的算力 (CPU cycles/s)
        本地/云端执行时无意义（置 0）

子任务总时延
------------
    x=0  : t_{i,j}^total = t_{i,j}^local
                         = c_{i,j} / f_i^local

    1<=x<=M : t_{i,j}^total = t_{i,j}^tran + t_{i,j}^edge
                             = t_{i,j}^ul [+ t_{m,m'}^wire]  +  c_{i,j} / f_{i,j}^alloc

    x=M+1: t_{i,j}^total = t_{i,j}^tran,cloud + t_{i,j}^cloud
                          = (t_{i,j}^ul + t^bh)  +  c_{i,j} / F^cloud

EST（最早开始时间，满足 DAG 依赖约束 C4）
-----------------------------------------
    EST(v_{i,j}) = 0,                          if pred(v_{i,j}) = ∅
    EST(v_{i,j}) = max_{v_{i,p} ∈ pred} [
                        EST(v_{i,p}) + t_{i,p}^total + t_{i,p→j}^tran
                   ],                          otherwise

    t_{i,p→j}^tran : 前驱 v_{i,p} 输出的中间结果 delta_{i,p} 传至 v_{i,j} 所在设备的时延
                     若两者在同一设备上则为 0

任务完成时间
------------
    T_i^finish = max_{v_{i,j} 是出口节点} [ EST(v_{i,j}) + t_{i,j}^total ]

能耗
----
    本地计算能耗 : E_{i,j}^local = kappa * c_{i,j} * (f_i^local)^(zeta-1)
    传输能耗     : E_{i,j}^tran  = p_i * t_{i,j}^ul
    任务总能耗   : E_i^total = sum_j E_{i,j}

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.config import DEFAULT_CONFIG, EnergyConfig, ObjectiveConfig
from model.task import DAGTask
from model.network import ChannelModel
from model.resource import ResourceManager

Decision = Dict[int, int]

FAlloc = Dict[int, float]

AllDecisions = List[Decision]

AllFAllocs = List[FAlloc]


# =========================================================================== #
#  任务卸载环境
# =========================================================================== #

class OffloadingEnvironment:

    def __init__(self,
                 dag_tasks: List[DAGTask],
                 resource_manager: ResourceManager,
                 channel_model: ChannelModel,
                 distances: List[float],
                 home_servers: List[int],
                 f_local_list: List[float],
                 kappa: Optional[float] = None,
                 alpha_w: Optional[float] = None,
                 beta_w: Optional[float] = None,
                 T_max_list: Optional[List[float]] = None,
                 energy_cfg: Optional[EnergyConfig] = None,
                 objective_cfg: Optional[ObjectiveConfig] = None):

        if not (len(dag_tasks) == len(distances) == len(home_servers) == len(f_local_list)):
            raise ValueError("dag_tasks / distances / home_servers / f_local_list 长度必须一致")

        self.dag_tasks = dag_tasks
        self.rm = resource_manager
        self.ch = channel_model
        self.distances = distances
        self.home_servers = home_servers
        self.f_local = f_local_list
        self.energy_cfg = energy_cfg if energy_cfg is not None else DEFAULT_CONFIG.energy
        self.objective_cfg = objective_cfg if objective_cfg is not None else DEFAULT_CONFIG.objective
        self.kappa = kappa if kappa is not None else self.energy_cfg.kappa
        self.alpha_w = alpha_w if alpha_w is not None else self.objective_cfg.alpha
        self.beta_w = beta_w if beta_w is not None else self.objective_cfg.beta
        if T_max_list is not None:
            self.T_max = T_max_list
        else:
            self.T_max = [self.objective_cfg.default_t_max_s] * len(dag_tasks)

        self.N = len(dag_tasks)       # 电梯数量
        self.M = resource_manager.M   # 边缘服务器数量

    # =================================================================== #
    #  单子任务时延计算
    # =================================================================== #

    def _local_exec_time(self, i: int, j: int) -> float:

        return self.dag_tasks[i].subtasks[j].c / self.f_local[i]

    def _edge_exec_time(self, i: int, j: int, f_alloc: float) -> float:

        if f_alloc <= 0.0:
            return float('inf')
        return self.dag_tasks[i].subtasks[j].c / f_alloc

    def _cloud_exec_time(self, i: int, j: int) -> float:

        return self.rm.cloud.execution_time(self.dag_tasks[i].subtasks[j].c)

    def _tran_delay(self, i: int, j: int, x_ij: int) -> float:

        if x_ij == 0:
            return 0.0

        d_bits   = self.dag_tasks[i].subtasks[j].d
        dist     = self.distances[i]
        home_m   = self.home_servers[i]

        if x_ij == self.M + 1:                      # 云端
            return self.ch.tran_delay_to_cloud(d_bits, dist)

        target_m = x_ij - 1
        if target_m == home_m:
            return self.ch.tran_delay_to_local_edge(d_bits, dist)
        else:
            bw = self.rm.inter_edge_bandwidth(home_m, target_m)
            return self.ch.tran_delay_to_remote_edge(d_bits, dist, bw)

    def subtask_total_time(self,
                           i: int,
                           j: int,
                           x_ij: int,
                           f_alloc: float) -> float:

        if x_ij == 0:
            return self._local_exec_time(i, j)
        elif 1 <= x_ij <= self.M:
            return self._tran_delay(i, j, x_ij) + self._edge_exec_time(i, j, f_alloc)
        else:   # x_ij == M+1
            return self._tran_delay(i, j, x_ij) + self._cloud_exec_time(i, j)

    # =================================================================== #
    #  EST 计算（约束 C4）
    # =================================================================== #

    def compute_est(self,
                    i: int,
                    decisions: Decision,
                    f_allocs: FAlloc,
                    topo_order: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:

        dag      = self.dag_tasks[i]
        dist     = self.distances[i]
        home_m   = self.home_servers[i]
        bw_mat   = self.rm.bw_matrix

        est:     Dict[int, float] = {}
        t_total: Dict[int, float] = {}

        for j in topo_order:
            x_j = decisions[j]
            f_j = f_allocs.get(j, 0.0)
            t_total[j] = self.subtask_total_time(i, j, x_j, f_j)

            preds = dag.subtasks[j].predecessors
            if len(preds) == 0:
                est[j] = 0.0
            else:
                candidates = []
                for p in preds:
                    x_p = decisions[p]
                    # 中间结果传输时延 t_{i,p→j}^tran
                    t_mid = self.ch.intermediate_tran_delay(
                        delta_bits   = dag.get_edge_data_bits(p, j),
                        x_src        = x_p,
                        x_dst        = x_j,
                        distance     = dist,
                        home_m       = home_m,
                        M            = self.M,
                        wire_bw_matrix = bw_mat,
                    )
                    candidates.append(est[p] + t_total[p] + t_mid)
                est[j] = max(candidates)

        return est, t_total

    # =================================================================== #
    #  任务完成时间
    # =================================================================== #

    def compute_finish_time(self,
                             i: int,
                             decisions: Decision,
                             f_allocs: FAlloc) -> Tuple[float, Dict]:

        dag        = self.dag_tasks[i]
        topo_order = dag.topological_sort()
        est, t_total = self.compute_est(i, decisions, f_allocs, topo_order)

        timing = {}
        for j in topo_order:
            x_j = decisions[j]
            f_j = f_allocs.get(j, 0.0)

            if x_j == 0:
                tran_time = 0.0
                exec_time = self._local_exec_time(i, j)
                exec_start = est[j]
                exec_finish = exec_start + exec_time
                target_edge = None
            elif 1 <= x_j <= self.M:
                tran_time = self._tran_delay(i, j, x_j)
                exec_time = self._edge_exec_time(i, j, f_j)
                exec_start = est[j] + tran_time
                exec_finish = exec_start + exec_time
                target_edge = x_j - 1
            else:
                tran_time = self._tran_delay(i, j, x_j)
                exec_time = self._cloud_exec_time(i, j)
                exec_start = est[j] + tran_time
                exec_finish = exec_start + exec_time
                target_edge = None

            timing[j] = {
                'decision': x_j,
                'f_alloc': f_j,
                'tran_time': tran_time,
                'exec_time': exec_time,
                'exec_start': exec_start,
                'exec_finish': exec_finish,
                'finish_time': est[j] + t_total[j],
                'target_edge': target_edge,
            }

        exit_nodes = dag.get_exit_nodes()
        T_finish = max(est[j] + t_total[j] for j in exit_nodes)

        return T_finish, {'est': est, 't_total': t_total, 'timing': timing}

    # =================================================================== #
    #  动态负载率
    # =================================================================== #

    def compute_dynamic_edge_load(self,
                                  all_decisions: AllDecisions,
                                  all_f_allocs: AllFAllocs) -> Dict[int, Dict[str, object]]:

        load_info: Dict[int, Dict[str, object]] = {}
        intervals_by_server = {m: [] for m in range(self.M)}

        for i in range(self.N):
            _, info = self.compute_finish_time(i, all_decisions[i], all_f_allocs[i])
            timing = info['timing']
            for j, detail in timing.items():
                target_edge = detail['target_edge']
                f_alloc = detail['f_alloc']
                if target_edge is None or f_alloc <= 0.0:
                    continue
                intervals_by_server[target_edge].append((
                    float(detail['exec_start']),
                    float(detail['exec_finish']),
                    float(f_alloc),
                    i,
                    j,
                ))

        for m in range(self.M):
            events = []
            for start, finish, f_alloc, _, _ in intervals_by_server[m]:
                events.append((start, 1, f_alloc))
                events.append((finish, 0, f_alloc))

            # 同一时刻先处理结束事件，再处理开始事件，表示区间为 [start, finish)
            events.sort(key=lambda item: (item[0], item[1]))

            current_alloc = 0.0
            peak_alloc = 0.0
            for _, event_type, f_alloc in events:
                if event_type == 0:
                    current_alloc -= f_alloc
                else:
                    current_alloc += f_alloc
                peak_alloc = max(peak_alloc, current_alloc)

            F_max = self.rm.edge_servers[m].F_max
            peak_load_ratio = peak_alloc / F_max if F_max > 0 else float('inf')
            load_info[m] = {
                'intervals': intervals_by_server[m],
                'peak_alloc': peak_alloc,
                'peak_load_ratio': peak_load_ratio,
                'feasible': peak_load_ratio <= 1.0 + 1e-9,
            }

        self.rm.update_dynamic_load_state(load_info)
        return load_info

    # =================================================================== #
    #  能耗计算
    # =================================================================== #

    def compute_energy(self, i: int, decisions: Decision) -> float:

        dag   = self.dag_tasks[i]
        dist  = self.distances[i]
        total = 0.0

        for j, x_j in decisions.items():
            st = dag.subtasks[j]
            if x_j == 0:
                # 本地计算能耗
                total += self.kappa * st.c * (self.f_local[i] ** (self.energy_cfg.zeta - 1))
                continue

            tx_energy = self.ch.transmission_energy(st.d, dist)
            upload_time = self.ch.upload_delay(st.d, dist)
            if 1 <= x_j <= self.M:
                f_alloc = self.rm.baseline_edge_alloc_hz
                exec_time = self._edge_exec_time(i, j, f_alloc)
                total += tx_energy + self.energy_cfg.edge_wait_power_w * (upload_time + exec_time)
            else:
                cloud_time = self._cloud_exec_time(i, j)
                total += (
                    tx_energy
                    + self.energy_cfg.cloud_wait_power_w * (upload_time + cloud_time)
                    + self.energy_cfg.cloud_backhaul_energy_per_bit * st.d
                )

        return total

    # =================================================================== #
    #  系统总代价
    # =================================================================== #

    def compute_system_cost(self,
                             all_decisions: AllDecisions,
                             all_f_allocs: AllFAllocs
                             ) -> Tuple[float, List[float], List[float]]:

        T_list: List[float] = []
        E_list: List[float] = []

        for i in range(self.N):
            T_i, _ = self.compute_finish_time(i, all_decisions[i], all_f_allocs[i])
            E_i    = self.compute_energy(i, all_decisions[i])
            T_list.append(T_i)
            E_list.append(E_i)

        total_cost = sum(
            self.alpha_w * T + self.beta_w * E
            for T, E in zip(T_list, E_list)
        )
        return total_cost, T_list, E_list

    # =================================================================== #
    #  静态资源分配统计
    # =================================================================== #

    def _compute_static_edge_alloc(self,
                                    all_decisions: AllDecisions,
                                    all_f_allocs: AllFAllocs) -> Dict[int, float]:

        alloc_per_server: Dict[int, float] = {m: 0.0 for m in range(self.M)}
        for i in range(self.N):
            for j, x in all_decisions[i].items():
                if 1 <= x <= self.M:
                    target_m = x - 1
                    f = all_f_allocs[i].get(j, 0.0)
                    alloc_per_server[target_m] += f
        return alloc_per_server

    # =================================================================== #
    #  约束检查
    # =================================================================== #

    def check_constraints(self,
                           all_decisions: AllDecisions,
                           all_f_allocs: AllFAllocs,
                           T_list: List[float]
                           ) -> Dict[str, object]:

        result: Dict[str, object] = {}

        c1 = {}
        if self.T_max is not None:
            for i, T_i in enumerate(T_list):
                c1[i] = (T_i <= self.T_max[i] + 1e-9)
        else:
            c1 = {i: True for i in range(self.N)}
        result['C1'] = c1

        static_alloc = self._compute_static_edge_alloc(all_decisions, all_f_allocs)
        c2 = {m: static_alloc[m] <= self.rm.edge_servers[m].F_max + 1e-9
              for m in range(self.M)}
        result['C2'] = c2
        result['static_alloc'] = static_alloc

        dynamic_load = self.compute_dynamic_edge_load(all_decisions, all_f_allocs)
        c3 = {m: dynamic_load[m]['feasible'] for m in range(self.M)}
        result['C3'] = c3
        result['dynamic_load'] = dynamic_load

        c5 = True
        for i in range(self.N):
            for j, x in all_decisions[i].items():
                if not (0 <= x <= self.M + 1):
                    c5 = False
                    break
            for j, f in all_f_allocs[i].items():
                if f < -1e-9:
                    c5 = False
                    break
        result['C5'] = c5

        feasible = (
            all(c1.values())
            and all(c2.values())
            and all(c3.values())
            and c5
        )
        result['feasible'] = feasible

        return result

    # =================================================================== #
    #  约束惩罚项
    # =================================================================== #

    def constraint_penalty(self,
                            all_decisions: AllDecisions,
                            all_f_allocs: AllFAllocs,
                            T_list: List[float],
                            penalty_coeff: Optional[float] = None) -> float:

        if penalty_coeff is None:
            penalty_coeff = self.objective_cfg.constraint_penalty_coeff

        penalty = 0.0

        if self.T_max is not None:
            c1_violations = []
            for i, T_i in enumerate(T_list):
                c1_violations.append(max(0.0, T_i / self.T_max[i] - 1.0))
            if c1_violations:
                penalty += penalty_coeff * (sum(c1_violations) / len(c1_violations))

        static_alloc = self._compute_static_edge_alloc(all_decisions, all_f_allocs)
        c2_violations = []
        for m in range(self.M):
            F_max = self.rm.edge_servers[m].F_max
            c2_violations.append(max(0.0, static_alloc[m] / F_max - 1.0) if F_max > 0 else 0.0)
        if c2_violations:
            penalty += penalty_coeff * (sum(c2_violations) / len(c2_violations))

        dynamic_load = self.compute_dynamic_edge_load(all_decisions, all_f_allocs)
        c3_violations = []
        for m in range(self.M):
            peak_ratio = float(dynamic_load[m]['peak_load_ratio'])
            c3_violations.append(max(0.0, peak_ratio - 1.0))
        if c3_violations:
            penalty += penalty_coeff * (sum(c3_violations) / len(c3_violations))

        return penalty

    # =================================================================== #
    #  RL 接口
    # =================================================================== #

    def get_observation(self, i: int) -> np.ndarray:

        dag        = self.dag_tasks[i]
        topo_order = dag.topological_sort()
        dist       = self.distances[i]

        task_feats = []
        for j in topo_order:
            st = dag.subtasks[j]
            t_local = st.c / self.f_local[i]
            t_ul    = self.ch.upload_delay(st.d, dist)
            t_cloud = self.rm.cloud.execution_time(st.c)
            task_feats.extend([
                st.d, st.c, st.delta,
                t_local, t_ul, t_cloud,
                float(len(st.predecessors)),
                float(len(st.successors)),
            ])

        load_state  = self.rm.load_state().tolist()
        avail_state = (self.rm.available_capacity_state() /
                       np.array([es.F_max for es in self.rm.edge_servers],
                                dtype=np.float32)).tolist()

        obs = np.array(task_feats + load_state + avail_state, dtype=np.float32)
        return obs

    def get_observation_tensor(self, i: int) -> torch.Tensor:

        return torch.from_numpy(self.get_observation(i))

    def step(self,
             i: int,
             decisions: Decision,
             f_allocs: FAlloc,
             penalty_coeff: Optional[float] = None
             ) -> Tuple[float, bool, Dict]:

        penalty_coeff = (
            penalty_coeff
            if penalty_coeff is not None
            else self.objective_cfg.constraint_penalty_coeff
        )

        self.rm.reset()

        allocation_failed = False
        failed_allocations = []

        # 更新边缘服务器资源分配
        for j, x in decisions.items():
            if 1 <= x <= self.M:
                m = x - 1
                f = f_allocs.get(j, 0.0)
                success = self.rm.edge_servers[m].allocate(i, j, f)
                if not success:
                    allocation_failed = True
                    failed_allocations.append((j, m, f))

        T_i, info_detail = self.compute_finish_time(i, decisions, f_allocs)
        E_i              = self.compute_energy(i, decisions)

        single_decisions = [{} for _ in range(self.N)]
        single_f_allocs = [{} for _ in range(self.N)]
        single_decisions[i] = decisions
        single_f_allocs[i] = f_allocs
        dynamic_load = self.compute_dynamic_edge_load(single_decisions, single_f_allocs)
        dynamic_overload = any(not dynamic_load[m]['feasible'] for m in range(self.M))

        # 约束惩罚
        pen = 0.0
        if self.T_max is not None and T_i > self.T_max[i]:
            pen = penalty_coeff * (T_i - self.T_max[i])
        if allocation_failed:
            pen += penalty_coeff
        if dynamic_overload:
            pen += penalty_coeff

        reward = -(self.alpha_w * T_i + self.beta_w * E_i) - pen

        info = {
            'T_finish':  T_i,
            'E_total':   E_i,
            'feasible':  (T_i <= self.T_max[i]) and (not allocation_failed) and (not dynamic_overload),
            'allocation_failed': allocation_failed,
            'failed_allocations': failed_allocations,
            'dynamic_load': dynamic_load,
            **info_detail,
        }
        return reward, False, info

    def reset(self) -> None:
        """重置环境（清空边缘服务器资源分配）"""
        self.rm.reset()

    # =================================================================== #
    #  基线策略
    # =================================================================== #

    def all_local_policy(self, i: int) -> Tuple[Decision, FAlloc]:

        dag = self.dag_tasks[i]
        return {j: 0 for j in dag.subtasks}, {}

    def all_cloud_policy(self, i: int) -> Tuple[Decision, FAlloc]:

        dag = self.dag_tasks[i]
        return {j: self.M + 1 for j in dag.subtasks}, {}

    def all_home_edge_policy(self, i: int,
                              f_alloc_per_task: Optional[float] = None) -> Tuple[Decision, FAlloc]:

        dag    = self.dag_tasks[i]
        home_x = self.home_servers[i] + 1
        if f_alloc_per_task is None:
            f_alloc_per_task = self.rm.baseline_edge_alloc_hz
        return (
            {j: home_x for j in dag.subtasks},
            {j: f_alloc_per_task for j in dag.subtasks},
        )

    def greedy_policy(self, i: int) -> Tuple[Decision, FAlloc]:

        dag      = self.dag_tasks[i]
        home_m   = self.home_servers[i]
        home_x   = home_m + 1
        n        = dag.num_subtasks
        f_share  = max(self.rm.edge_servers[home_m].available_capacity / max(n, 1), 1.0)

        decisions: Decision = {}
        f_allocs:  FAlloc   = {}

        for j, st in dag.subtasks.items():
            t_local = self._local_exec_time(i, j)
            t_edge  = self._tran_delay(i, j, home_x) + self._edge_exec_time(i, j, f_share)
            t_cloud = self._tran_delay(i, j, self.M + 1) + self._cloud_exec_time(i, j)

            best = min(t_local, t_edge, t_cloud)
            if best == t_local:
                decisions[j] = 0
            elif best == t_edge:
                decisions[j] = home_x
                f_allocs[j]  = f_share
            else:
                decisions[j] = self.M + 1

        return decisions, f_allocs

    # =================================================================== #
    #  调试输出
    # =================================================================== #

    def __repr__(self) -> str:
        return (f"OffloadingEnvironment("
                f"N={self.N}, M={self.M}, "
                f"alpha={self.alpha_w}, beta={self.beta_w})")

    def evaluate_policy(self,
                         all_decisions: AllDecisions,
                         all_f_allocs: AllFAllocs,
                         verbose: bool = False) -> Dict:

        total_cost, T_list, E_list = self.compute_system_cost(all_decisions, all_f_allocs)
        constraints = self.check_constraints(all_decisions, all_f_allocs, T_list)

        if verbose:
            print(f"{'='*50}")
            print(f"系统总代价 : {total_cost:.6f}")
            print(f"约束满足   : {constraints['feasible']}")
            for i in range(self.N):
                print(f"  电梯 {i:2d} | T_finish={T_list[i]:.4f}s | "
                      f"E_total={E_list[i]:.4e}J")
            print(f"{'='*50}")

        return {
            'total_cost':  total_cost,
            'T_list':      T_list,
            'E_list':      E_list,
            'constraints': constraints,
        }
