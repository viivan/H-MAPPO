from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class DatasetConfig:

    project_root: Path = Path(__file__).resolve().parents[1]
    rl_root_name: str = "RLTaskOffloading-master"
    train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_random10",
        "offload_random15",
        "offload_random20",
        "offload_random25",
        "offload_random30",
        "offload_random35",
        "offload_random40",
        "offload_random45",
        "offload_random50",
    ])
    test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_random10_test",
        "offload_random15_test",
        "offload_random20_test",
        "offload_random25_test",
        "offload_random30_test",
        "offload_random35_test",
        "offload_random40_test",
        "offload_random45_test",
        "offload_random50_test",
    ])
    public_v3_train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_daggen_v3_10",
        "offload_daggen_v3_15",
        "offload_daggen_v3_20",
        "offload_daggen_v3_25",
        "offload_daggen_v3_30",
    ])
    public_v3_test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_daggen_v3_10_test",
        "offload_daggen_v3_15_test",
        "offload_daggen_v3_20_test",
        "offload_daggen_v3_25_test",
        "offload_daggen_v3_30_test",
    ])
    # 电梯场景数据集（固定拓扑 + 电梯真实参数）
    elevator_train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_elevator6",
        "offload_elevator8",
        "offload_elevator10",
        "offload_elevator12",
        "offload_elevator14",
        "offload_elevator16",
        "offload_elevator18",
        "offload_elevator20",
    ])
    elevator_test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_elevator6_test",
        "offload_elevator8_test",
        "offload_elevator10_test",
        "offload_elevator12_test",
        "offload_elevator14_test",
        "offload_elevator16_test",
        "offload_elevator18_test",
        "offload_elevator20_test",
    ])

    safety_train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_safety_monitor",
    ])
    safety_test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_safety_monitor_test",
    ])
    dispatch_train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_smart_dispatch_R2_E2",
        "offload_smart_dispatch_R2_E4",
        "offload_smart_dispatch_R4_E2",
        "offload_smart_dispatch_R4_E4",
        "offload_smart_dispatch_R6_E2",
        "offload_smart_dispatch_R6_E4",
        "offload_smart_dispatch_R8_E2",
        "offload_smart_dispatch_R8_E4",
    ])
    dispatch_test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_smart_dispatch_R2_E2_test",
        "offload_smart_dispatch_R2_E4_test",
        "offload_smart_dispatch_R4_E2_test",
        "offload_smart_dispatch_R4_E4_test",
        "offload_smart_dispatch_R6_E2_test",
        "offload_smart_dispatch_R6_E4_test",
        "offload_smart_dispatch_R8_E2_test",
        "offload_smart_dispatch_R8_E4_test",
    ])
    predmaint_train_subdirs: List[str] = field(default_factory=lambda: [
        "offload_predictive_maintenance",
    ])
    predmaint_test_subdirs: List[str] = field(default_factory=lambda: [
        "offload_predictive_maintenance_test",
    ])

    file_suffix: str = ".gv"

    @property
    def rl_root(self) -> Path:
        return self.project_root / self.rl_root_name

    @property
    def offloading_data_dir(self) -> Path:

        return self.project_root / "offloading_data"

    def resolve_subdir(self, subdir: str) -> Path:
        return self.offloading_data_dir / subdir


@dataclass
class DAGConfig:

    bytes_to_bits: float = 8.0
    default_expect_size_ratio: float = 0.4
    cpu_cycles_per_bit: float = 10.0
    random_delta_ratio: float = 0.4
    random_edge_density: float = 0.3
    random_d_range_bits: Tuple[float, float] = (1e6, 1e8)
    random_c_range_cycles: Tuple[float, float] = (1e8, 1e10)


@dataclass
class ChannelConfig:

    h_0: float = 1e-3
    d_0: float = 1.0
    path_loss_alpha: float = 2.5
    uplink_bandwidth_hz: float = 100.0e6
    noise_psd_w_hz: float = 1e-17
    tx_power_w: float = 1.258
    rx_power_w: float = 1.181
    backhaul_delay_s: float = 0.18
    default_inter_edge_bandwidth_bps: float = 100e6


@dataclass
class ResourceConfig:

    num_base_stations: int = 3
    elevators_per_station: List[int] = field(default_factory=lambda: [4, 4, 4])
    edge_cpu_hz: List[float] = field(default_factory=lambda: [1.5e9, 2.0e9, 2.5e9])
    cloud_cpu_hz: float = 4e9
    local_cpu_hz: float = 0.8e9
    overload_threshold: float = 0.9
    inter_edge_bandwidth_bps: float = 100e6
    baseline_edge_alloc_hz: float = 1e9

    def total_elevators(self) -> int:
        return int(sum(self.elevators_per_station))


@dataclass
class ElevatorConfig:

    floor_height_m: float = 3.0
    base_station_horizontal_distance_m: float = 50.0
    default_floor: int = 10


@dataclass
class EnergyConfig:

    kappa: float = 5e-27
    zeta: int = 3
    edge_wait_power_w: float = 0.15
    cloud_wait_power_w: float = 0.90
    cloud_backhaul_energy_per_bit: float = 1.0e-9


@dataclass
class ObjectiveConfig:

    alpha: float = 0.5
    beta: float = 0.5
    default_t_max_s: float = 30.0
    constraint_penalty_coeff: float = 1.0


@dataclass
class ModelConfig:

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    elevator: ElevatorConfig = field(default_factory=ElevatorConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    def build_inter_edge_bw_matrix(self) -> np.ndarray:

        M = self.resource.num_base_stations
        bw = np.full(
            (M, M),
            self.channel.default_inter_edge_bandwidth_bps,
            dtype=np.float64,
        )
        np.fill_diagonal(bw, 0.0)
        return bw

    def default_home_servers(self) -> List[int]:

        home_servers: List[int] = []
        for m, count in enumerate(self.resource.elevators_per_station):
            home_servers.extend([m] * count)
        return home_servers

    def default_local_cpu_list(self) -> List[float]:

        return [self.resource.local_cpu_hz] * self.resource.total_elevators()

    def default_distance_list(self) -> List[float]:

        floor = self.elevator.default_floor
        vertical = floor * self.elevator.floor_height_m
        horizontal = self.elevator.base_station_horizontal_distance_m
        distance = float(np.sqrt(horizontal ** 2 + vertical ** 2))
        return [distance] * self.resource.total_elevators()

    def default_tmax_list(self) -> List[float]:
        return [self.objective.default_t_max_s] * self.resource.total_elevators()


DEFAULT_CONFIG = ModelConfig()
