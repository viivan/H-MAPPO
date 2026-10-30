"""
网络与通信模型
"""

from __future__ import annotations

import numpy as np

from model.config import ChannelConfig, DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
#  无线信道 + 传输时延模型
# --------------------------------------------------------------------------- #

class ChannelModel:

    def __init__(self, cfg: ChannelConfig | None = None):
        self.cfg = cfg if cfg is not None else DEFAULT_CONFIG.channel

    # ------------------------------------------------------------------ #
    #  信道增益
    # ------------------------------------------------------------------ #

    def channel_gain(self, distance: float) -> float:

        d = max(distance, 1e-3)
        return self.cfg.h_0 * (self.cfg.d_0 / d) ** self.cfg.path_loss_alpha

    # ------------------------------------------------------------------ #
    #  上行速率与上传时延
    # ------------------------------------------------------------------ #

    def uplink_rate(self, distance: float) -> float:

        h = self.channel_gain(distance)
        snr = self.cfg.tx_power_w * h / (self.cfg.noise_psd_w_hz * self.cfg.uplink_bandwidth_hz)
        return self.cfg.uplink_bandwidth_hz * np.log2(1.0 + snr)

    def upload_delay(self, data_bits: float, distance: float) -> float:

        R = self.uplink_rate(distance)
        return data_bits / R

    # ------------------------------------------------------------------ #
    #  有线传输时延（边缘服务器间转发）
    # ------------------------------------------------------------------ #

    def wire_delay(self, data_bits: float, bandwidth_bps: float) -> float:

        if bandwidth_bps <= 0:
            return float('inf')
        return data_bits / bandwidth_bps

    # ------------------------------------------------------------------ #
    #  总传输时延
    # ------------------------------------------------------------------ #

    def tran_delay_to_local_edge(self,
                                  data_bits: float,
                                  distance: float) -> float:

        return self.upload_delay(data_bits, distance)

    def tran_delay_to_remote_edge(self,
                                   data_bits: float,
                                   distance: float,
                                   wire_bandwidth_bps: float) -> float:

        t_ul   = self.upload_delay(data_bits, distance)
        t_wire = self.wire_delay(data_bits, wire_bandwidth_bps)
        return t_ul + t_wire

    def tran_delay_to_cloud(self,
                             data_bits: float,
                             distance: float) -> float:

        t_ul = self.upload_delay(data_bits, distance)
        return t_ul + self.cfg.backhaul_delay_s

    # ------------------------------------------------------------------ #
    #  中间结果传输时延
    # ------------------------------------------------------------------ #

    def intermediate_tran_delay(self,
                                 delta_bits: float,
                                 x_src: int,
                                 x_dst: int,
                                 distance: float,
                                 home_m: int,
                                 M: int,
                                 wire_bw_matrix: np.ndarray) -> float:

        if x_src == x_dst:
            return 0.0

        def edge_idx(x: int) -> int:

            return x - 1

        # ---------- 本地 -> 其他 ----------
        if x_src == 0:
            if 1 <= x_dst <= M:
                dst_m = edge_idx(x_dst)
                if dst_m == home_m:
                    return self.tran_delay_to_local_edge(delta_bits, distance)
                else:
                    bw = wire_bw_matrix[home_m, dst_m]
                    return self.tran_delay_to_remote_edge(delta_bits, distance, bw)
            else:                                      # 云端
                return self.tran_delay_to_cloud(delta_bits, distance)

        # ---------- 边缘 -> 其他 ----------
        if 1 <= x_src <= M:
            src_m = edge_idx(x_src)
            if x_dst == 0:
                # 边缘 -> 本地
                return self.upload_delay(delta_bits, distance)
            elif 1 <= x_dst <= M:
                # 边缘 -> 边缘
                dst_m = edge_idx(x_dst)
                if src_m == dst_m:
                    return 0.0
                bw = wire_bw_matrix[src_m, dst_m]
                return self.wire_delay(delta_bits, bw)
            else:
                # 边缘 -> 云端
                return self.cfg.backhaul_delay_s

        # ---------- 云端 -> 其他 ----------

        if x_dst == 0:

            return self.cfg.backhaul_delay_s + self.upload_delay(delta_bits, distance)
        elif 1 <= x_dst <= M:

            return self.cfg.backhaul_delay_s
        else:
            return 0.0

    # ------------------------------------------------------------------ #
    #  传输能耗
    # ------------------------------------------------------------------ #

    def transmission_energy(self, data_bits: float, distance: float) -> float:

        t_ul = self.upload_delay(data_bits, distance)
        return self.cfg.tx_power_w * t_ul

    # ------------------------------------------------------------------ #
    #  楼层到距离的映射
    # ------------------------------------------------------------------ #

    @staticmethod
    def floor_to_distance(floor: int,
                           floor_height: float | None = None,
                           base_distance: float | None = None) -> float:

        elevator_cfg = DEFAULT_CONFIG.elevator
        if floor_height is None:
            floor_height = elevator_cfg.floor_height_m
        if base_distance is None:
            base_distance = elevator_cfg.base_station_horizontal_distance_m
        vertical = floor * floor_height
        return float(np.sqrt(base_distance ** 2 + vertical ** 2))
