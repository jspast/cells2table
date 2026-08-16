from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BoundingBox:
    """Rectangular bounding box with coordinates.

    Attributes:
        l: Left coordinate.
        t: Top coordinate.
        r: Right coordinate.
        b: Bottom coordinate.
    """

    l: np.float32
    t: np.float32
    r: np.float32
    b: np.float32

    @staticmethod
    def from_array(bbox: NDArray[np.float32]) -> BoundingBox:
        """Create a BoundingBox from a numpy array.

        Args:
            bbox: Array with [left, top, right, bottom] coordinates.

        Returns:
            BoundingBox instance.
        """
        return BoundingBox(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])

    def as_array(self) -> NDArray[np.float32]:
        """Convert bounding box to numpy array.

        Returns:
            Array with [left, top, right, bottom] coordinates.
        """
        return np.array([self.l, self.t, self.r, self.b], dtype=np.float32)
