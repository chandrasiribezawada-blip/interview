from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    @abstractmethod
    def save_local(self, path):
        raise NotImplementedError

    @abstractmethod
    def as_retriever(self, **kwargs):
        raise NotImplementedError
