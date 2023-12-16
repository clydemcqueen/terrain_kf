"""
Estimate terrain using a 3 dim Kalman filter
"""

import math

import filterpy.common
import filterpy.kalman
import numpy as np


class TerrainKF:
    def __init__(self, dt, measurement_var, process_var):
        self.dt = dt

        # State is p (observed), p' (hidden), p'' (hidden)
        self.x = np.array([0.0, 0.0, 0.0])

        # Initial covariance
        self.P = np.diag([50.0, 50.0, 50.0])

        # Process noise
        spectral_density = math.sqrt(process_var)
        # Q = filterpy.common.Q_continuous_white_noise(dim=3, dt=dt, spectral_density=spectral_density)
        # print('Possible process noise:')
        # print(Q)

        # Process noise (simple)
        self.Q = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, spectral_density * dt]])
        # print('Actual process noise:')
        # print(self.Q)

        # State transition function
        self.F = np.array([[1, dt, 0.5 * dt * dt],
                           [0, 1, dt],
                           [0, 0, 1]])

        # Measurement function
        self.H = np.array([[1.0, 0.0, 0.0]])

        # Measurement covariance
        self.R = np.array([[measurement_var]])

    def predict(self):
        self.x, self.P = filterpy.kalman.predict(self.x, self.P, F=self.F, Q=self.Q)

    def update(self, z: float):
        self.x, self.P = filterpy.kalman.update(self.x, self.P, z, self.R, H=self.H)

    def project(self, steps: int):
        """
        The filter runs at t = the time of the last measurement.
        To simulate a sensor delay we project forward using a multiple of dt.
        The results of the projection do not affect the state.

        E.g., to simulate a sensor with a 0.8s delay, try:
            kf.predict()
            kf.update(z)
            curr_x, curr_P = kf.project(8)

        This projection can get wild... probably need some filters on the result.
        """
        x = self.x
        P = self.P
        for _ in range(steps):
            x, P = filterpy.kalman.predict(x, P, self.F)
        return x, P
