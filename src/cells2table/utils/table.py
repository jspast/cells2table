from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class BoundingBox:
    l: float  # noqa: E741
    t: float
    r: float
    b: float

    @staticmethod
    def from_array(bbox: ArrayLike[float]) -> BoundingBox:
        return BoundingBox(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])

    def as_array(self) -> ArrayLike[float]:
        return np.array([self.l, self.t, self.r, self.b])


@dataclass
class Cell:
    bbox: BoundingBox
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


@dataclass
class Table:
    cells: list[Cell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
