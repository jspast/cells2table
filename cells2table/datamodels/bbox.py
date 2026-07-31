from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    l: float
    t: float
    r: float
    b: float

    @staticmethod
    def from_array(bbox: NDArray[np.float32]) -> BoundingBox:
        return BoundingBox(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])

    def as_array(self) -> NDArray[np.float32]:
        return np.array([self.l, self.t, self.r, self.b], dtype=np.float32)
