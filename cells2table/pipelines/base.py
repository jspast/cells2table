from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePipeline(ABC):
    """Base interface for pipelines.

    A pipeline orchestrates multiple models to process input data end-to-end.
    """

    @abstractmethod
    def __init__(self, models_path: Path | str | None = None) -> None:
        """Initialize pipeline models.

        Args:
            models_path: Path to directory containing model weights.
        """

    @abstractmethod
    def __call__(self, input: Any, **kwargs):
        """Run the pipeline.

        Args:
            input: Pipeline input.
            **kwargs: Additional pipeline-specific arguments.

        Returns:
            Pipeline output.
        """
