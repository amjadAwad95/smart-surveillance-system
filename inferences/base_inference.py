from abc import ABC, abstractmethod


class BaseInference(ABC):
    """
    Abstract base class for inference pipelines.

    This class defines the interface for creating custom inference implementations.
    Subclasses must implement the `inference` method, which can accept any number
    of positional and keyword arguments. The class also implements the `__call__`
    method to allow instances to be called like functions, automatically invoking
    the `inference` method.
    """

    @abstractmethod
    def inference(self, *args, **kwargs):
        """
        Abstract method for performing inference.
        Must be implemented by subclasses to define the specific inference logic.

        :param args: Positional arguments for the inference method.
        :param kwargs: Keyword arguments for the inference method.
        :return: The output of the inference process (type depends on implementation).
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Makes the instance callable.
        Calling an instance of a subclass will invoke its `inference` method.

        :param args: Positional arguments passed to `inference`.
        :param kwargs: Keyword arguments passed to `inference`.
        :return: The result returned by the `inference` method.
        """
        return self.inference(*args, **kwargs)
