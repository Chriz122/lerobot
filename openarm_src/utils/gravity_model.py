from __future__ import annotations

import logging
import os
import numpy as np

from openarm_src.utils.config import JOINT_SIGN

logger = logging.getLogger("openarm_gc.gravity_model")


class GravityModel:
    def __init__(self, urdf_path: str, side: str | None):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.side = side or "right"

        candidates = []
        if side:
            candidates.append([f"openarm_{side}_joint{i}" for i in range(1, 8)])
        candidates.append([f"openarm_joint{i}" for i in range(1, 8)])
        for names in candidates:
            if all(self.model.existJointName(n) for n in names):
                break
        else:
            raise ValueError(
                f"URDF 裡找不到 {candidates[0]}。現有關節: {list(self.model.names)}"
            )
        jids = [self.model.getJointId(n) for n in names]
        self.q_idx = [self.model.idx_qs[j] for j in jids]
        self.v_idx = [self.model.idx_vs[j] for j in jids]
        self.sign = np.array(JOINT_SIGN.get(self.side, [1] * 7), dtype=float)
        logger.info(f"[gc] {side}: using URDF joints {names}")

    def torque(self, q_deg: np.ndarray) -> np.ndarray:
        """q_deg: LeRobot 角度 (7,) -> 需要送給馬達的重力補償力矩 (Nm, 7)"""
        q_urdf = self.sign * np.radians(q_deg)
        self.q[self.q_idx] = q_urdf
        g = self.pin.computeGeneralizedGravity(self.model, self.data, self.q)
        return self.sign * g[self.v_idx]

