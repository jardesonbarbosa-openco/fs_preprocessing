import pandas as pd

from abc import ABC, abstractmethod

class BasePipeline(ABC):

    @abstractmethod
    def load_dataset(self, dataset: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def export_output(self, output: str) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass