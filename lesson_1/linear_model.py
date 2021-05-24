import numpy as np


class GradientDescentRegressor:
    def __init__(
        self, intercept_samples: bool = True, copy_samples: bool = True
    ):
        self._intercept_samples = intercept_samples
        self._copy_samples = copy_samples

    @staticmethod
    def intercept_samples(X: np.array) -> np.array:
        return np.hstack((np.ones(shape=(X.shape[0], 1)), X.copy()))

    @staticmethod
    def initialize_weights(X: np.array) -> np.array:
        return np.zeros(shape=X.shape[1])

    @staticmethod
    def interpolate_targets(X: np.array, w: np.array) -> np.array:
        return X.dot(w)

    @staticmethod
    def error_function(X: np.array, y: np.array, w: np.array) -> np.array:
        # return np.linalg.norm(X.dot(w) - y)**2 / X.shape[0]
        return sum((X.dot(w) - y)**2) / X.shape[0]

    @staticmethod
    def error_flow_function(X: np.array, y: np.array, w: np.array) -> np.array:
        return 2 * np.dot(X.transpose(), X.dot(w) - y) / X.shape[0]

    @staticmethod
    def residual_function(w: list) -> float:
        # return np.linalg.norm(np.subtract(*w))
        return sum((w[0] - w[1])**2)**.5

    @classmethod
    def iterate(
        cls, X: np.array, y: np.array, w: np.array, s: float
    ) -> tuple:
        return w - s * cls.error_flow_function(X, y, w)

    @staticmethod
    def stop_criteria(
        r: float, i: int, r_min: float, r_max: float, i_max: int
    ) -> bool:
        return r <= r_min or r >= r_max or i >= i_max

    @staticmethod
    def print_log(data: tuple) -> None:
        print(
            ' '.join(
                [
                    format(_, '.6g') if isinstance(_, float) else str(_)
                    for _ in data
                ]
            )
        )

    @staticmethod
    def write_log(file, data: tuple) -> int:
        return file.write(','.join([str(_) for _ in data]) + '\n')

    def fit(
        self,
        samples: np.array,
        targets: np.array,
        step_size: float = 1e-1,
        weights_initial: np.array = None,
        residual_min: float = 1e-3,
        residual_max: float = 1e3,
        iteration_max: int = 1e3,
        print_frequency: int = 0,
        log_file=None,
        write_frequency: int = 1
    ) -> tuple:
        if self._intercept_samples:
            samples_ = self.intercept_samples(samples)
        elif self._copy_samples:
            samples_ = samples.copy()
        else:
            samples_ = samples

        if weights_initial is None:
            self.weights = self.initialize_weights(samples_)
        else:
            self.weights = weights_initial

        iteration = 0
        error = self.error_function(samples_, targets, self.weights)

        if print_frequency:
            self.print_log((iteration, error))

        if log_file is not None:
            self.write_log(log_file, (iteration, error))

        while True:
            self.weights, weights_previous = (
                self.iterate(samples_, targets, self.weights, step_size),
                self.weights.copy()
            )
            iteration += 1
            residual = self.residual_function((self.weights, weights_previous))
            error = self.error_function(samples_, targets, self.weights)

            if print_frequency and iteration % print_frequency == 0:
                self.print_log((iteration, error, residual))

            if log_file is not None and iteration % write_frequency == 0:
                self.write_log(log_file, (iteration, error, residual))

            if self.stop_criteria(
                residual, iteration, residual_min, residual_max, iteration_max
            ):
                break

        return iteration, error, residual

    def predict(self, samples: np.array) -> np.array:
        return self.interpolate_targets(
            self.intercept_samples(samples) if self._intercept_samples else samples,
            self.weights
        )
