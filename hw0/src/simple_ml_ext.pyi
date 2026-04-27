import numpy as np


def softmax_regression_epoch_cpp(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lr: float,
    batch: int,
) -> None: ...
