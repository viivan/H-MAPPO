from __future__ import annotations

import numpy as np
import pydotplus
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from model.config import DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
#  子任务节点
# --------------------------------------------------------------------------- #

@dataclass
class SubTask:

    task_id: int
    subtask_id: int
    d: float                                         # 输入数据量 (bits)
    c: float                                         # 计算量 (CPU cycles)
    delta: float                                     # 中间结果大小 (bits)
    predecessors: List[int] = field(default_factory=list)
    successors:   List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"SubTask(i={self.task_id}, j={self.subtask_id}, "
                f"d={self.d:.2e}, c={self.c:.2e}, delta={self.delta:.2e}, "
                f"pred={self.predecessors}, succ={self.successors})")


# --------------------------------------------------------------------------- #
#  DAG 任务图
# --------------------------------------------------------------------------- #

class DAGTask:

    def __init__(self, task_id: int):
        self.task_id: int = task_id
        self.deadline_s: Optional[float] = None
        self.scenario: Optional[str] = None
        self.subtasks: Dict[int, SubTask] = {}           # {j: SubTask}
        self.edges: List[Tuple[int, int]] = []           # [(src_j, dst_j)]
        self.edge_data_bits: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------ #
    #  构建接口
    # ------------------------------------------------------------------ #

    def add_subtask(self, subtask: SubTask) -> None:
        self.subtasks[subtask.subtask_id] = subtask

    def add_edge(self, src_j: int, dst_j: int, data_bits: Optional[float] = None) -> None:

        if data_bits is None:
            data_bits = self.subtasks[src_j].delta
        self.edges.append((src_j, dst_j))
        self.edge_data_bits[(src_j, dst_j)] = float(data_bits)
        self.subtasks[src_j].successors.append(dst_j)
        self.subtasks[dst_j].predecessors.append(src_j)

    def get_edge_data_bits(self, src_j: int, dst_j: int) -> float:

        return self.edge_data_bits.get((src_j, dst_j), self.subtasks[src_j].delta)

    # ------------------------------------------------------------------ #
    #  基本属性
    # ------------------------------------------------------------------ #

    @property
    def num_subtasks(self) -> int:
        return len(self.subtasks)

    def get_entry_nodes(self) -> List[int]:
        return [j for j, st in self.subtasks.items() if len(st.predecessors) == 0]

    def get_exit_nodes(self) -> List[int]:
        return [j for j, st in self.subtasks.items() if len(st.successors) == 0]

    def topological_sort(self) -> List[int]:
        in_degree: Dict[int, int] = {j: len(st.predecessors)
                                      for j, st in self.subtasks.items()}
        queue = sorted([j for j, deg in in_degree.items() if deg == 0])
        result: List[int] = []

        while queue:
            j = queue.pop(0)
            result.append(j)
            for succ in sorted(self.subtasks[j].successors):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != self.num_subtasks:
            raise ValueError(f"任务图 {self.task_id} 存在环路，无法拓扑排序。")
        return result

    @classmethod
    def from_dot_file(cls,
                      file_path: str,
                      task_id: int,
                      cpu_cycles_per_bit: Optional[float] = None) -> "DAGTask":

        dag = cls(task_id)
        dag_cfg = DEFAULT_CONFIG.dag
        if cpu_cycles_per_bit is None:
            cpu_cycles_per_bit = dag_cfg.cpu_cycles_per_bit
        dot_obj = pydotplus.graphviz.graph_from_dot_file(file_path)

        graph_attrs = dict(dot_obj.get_attributes() or {})
        for graph_node in dot_obj.get_node_list():
            if graph_node.get_name() == "graph":
                graph_attrs.update(graph_node.obj_dict.get('attributes', {}))
        deadline_raw = graph_attrs.get('deadline_s')
        if deadline_raw is not None:
            dag.deadline_s = float(str(deadline_raw).strip('"'))
        scenario_raw = graph_attrs.get('scenario')
        if scenario_raw is not None:
            dag.scenario = str(scenario_raw).strip('"')

        # 解析节点
        for node in dot_obj.get_node_list():
            name = node.get_name()
            if name in ('node', 'graph', 'edge', ''):
                continue
            clean_name = str(name).strip('"')
            if not clean_name.isdigit():
                continue
            attrs = node.obj_dict['attributes']

            j = int(clean_name) - 1                              # 0-indexed
            d_bytes = float(eval(attrs.get('size', '0')))
            delta_bytes = float(
                eval(attrs.get('expect_size', str(d_bytes * dag_cfg.default_expect_size_ratio)))
            )

            d = d_bytes * dag_cfg.bytes_to_bits                  # bytes -> bits
            delta = delta_bytes * dag_cfg.bytes_to_bits          # bytes -> bits
            cycles_str = attrs.get('cycles', None)               # 可选：直接指定计算量
            if cycles_str is not None:
                c = float(eval(cycles_str))                      # 使用显式 cycles
            else:
                c = d * cpu_cycles_per_bit                       # 回退到全局默认

            dag.add_subtask(SubTask(
                task_id=task_id,
                subtask_id=j,
                d=d,
                c=c,
                delta=delta,
            ))

        # 解析边
        for edge in dot_obj.get_edge_list():
            src = int(edge.get_source()) - 1
            dst = int(edge.get_destination()) - 1
            attrs = edge.obj_dict['attributes']
            edge_bytes = float(eval(attrs.get('size', str(dag.subtasks[src].delta / dag_cfg.bytes_to_bits))))
            edge_bits = edge_bytes * dag_cfg.bytes_to_bits
            dag.add_edge(src, dst, data_bits=edge_bits)

        return dag

    @classmethod
    def generate_random(cls,
                        task_id: int,
                        num_subtasks: int,
                        d_range: Optional[Tuple[float, float]] = None,
                        c_range: Optional[Tuple[float, float]] = None,
                        delta_ratio: Optional[float] = None,
                        edge_density: Optional[float] = None,
                        rng: Optional[np.random.Generator] = None) -> "DAGTask":

        if rng is None:
            rng = np.random.default_rng()
        dag_cfg = DEFAULT_CONFIG.dag
        if d_range is None:
            d_range = dag_cfg.random_d_range_bits
        if c_range is None:
            c_range = dag_cfg.random_c_range_cycles
        if delta_ratio is None:
            delta_ratio = dag_cfg.random_delta_ratio
        if edge_density is None:
            edge_density = dag_cfg.random_edge_density

        dag = cls(task_id)

        for j in range(num_subtasks):
            d = float(rng.uniform(*d_range))
            c = float(rng.uniform(*c_range))
            delta = d * delta_ratio
            dag.add_subtask(SubTask(
                task_id=task_id,
                subtask_id=j,
                d=d, c=c, delta=delta,
            ))

        # 按拓扑序添加边，确保 DAG 无环
        for dst in range(1, num_subtasks):
            for src in range(dst):
                if rng.random() < edge_density:
                    dag.add_edge(src, dst, data_bits=dag.subtasks[src].delta)

        # 保证每个非入口节点至少有一个前驱
        for j in range(1, num_subtasks):
            if len(dag.subtasks[j].predecessors) == 0:
                src = int(rng.integers(0, j))
                dag.add_edge(src, j, data_bits=dag.subtasks[src].delta)

        return dag

    # ------------------------------------------------------------------ #
    #  调试输出
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"DAGTask(i={self.task_id}, "
                f"|V|={self.num_subtasks}, |E|={len(self.edges)})")

    def summary(self) -> str:
        lines = [f"DAGTask i={self.task_id}  |V|={self.num_subtasks}  |E|={len(self.edges)}"]
        lines.append(f"  入口节点: {self.get_entry_nodes()}")
        lines.append(f"  出口节点: {self.get_exit_nodes()}")
        lines.append(f"  拓扑序:   {self.topological_sort()}")
        for j, st in sorted(self.subtasks.items()):
            lines.append(f"  {st}")
        return "\n".join(lines)
