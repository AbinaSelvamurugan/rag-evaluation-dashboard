"""No surrogate implementations may masquerade as Project 6."""
from abc import ABC, abstractmethod
from rag_eval.config import ExperimentConfig, IndexInfo, Trace


class RAGAdapter(ABC):
    verified: bool = False

    @abstractmethod
    def capabilities(self) -> dict: ...

    @abstractmethod
    def prepare_index(self, config: ExperimentConfig) -> IndexInfo: ...

    @abstractmethod
    def run_question(self, question: str, config: ExperimentConfig, index: IndexInfo) -> Trace: ...
