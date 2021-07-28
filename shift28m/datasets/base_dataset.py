import abc


class BaseDataset(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def load_dataset(
        self,
        train_size: int = 10000,
        test_size: int = 10000,
        covariate_shift: bool = False,
        target_shift: bool = False,
        train_mu: float = 50,
        train_sigma: float = 10,
        test_mu: float = 80,
        test_sigma: float = 10,
        random_seed: int = 128,
        max_iter: int = 100,
    ):

        pass
