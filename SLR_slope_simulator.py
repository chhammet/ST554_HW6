"""
SLR_slope_simulator.py
----------------------
ST 554 - Big Data with Python | Homework 6, Part II
Author: Cole Hammett (chhammet@ncsu.edu)

This module defines the SLR_slope_simulator class, which encapsulates the
simulation of the sampling distribution of the OLS slope estimator (beta_1)
in a Simple Linear Regression model:

    Yi = beta_0 + beta_1 * xi + Ei,   Ei ~ N(0, sigma^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn import linear_model


class SLR_slope_simulator:
    """
    Simulates the sampling distribution of the OLS slope estimator for a
    Simple Linear Regression (SLR) model.

    Parameters
    ----------
    beta_0 : float
        True intercept of the linear model.
    beta_1 : float
        True slope of the linear model.
    x : array-like
        Fixed predictor values used for every simulated dataset.
    sigma : float
        Standard deviation of the error term (sigma, not sigma^2).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, beta_0, beta_1, x, sigma, seed):
        # Store true model parameters as attributes
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.sigma  = sigma
        self.x      = np.array(x)          # ensure numpy array
        self.n      = len(self.x)           # number of observations
        self.rng    = default_rng(seed)     # seeded random number generator
        self.slopes = []                    # will hold slope estimates after run_simulations()

    # ------------------------------------------------------------------
    # generate_data
    # ------------------------------------------------------------------
    def generate_data(self):
        """
        Generate one dataset from the true SLR model.

        Returns
        -------
        x : np.ndarray
            The fixed predictor values (same every call).
        y : np.ndarray
            Simulated response values: beta_0 + beta_1*x + sigma * N(0,1).

        Note: We use rng.standard_normal() scaled by self.sigma, which exactly
        mirrors the rng.standard_normal(n) pattern from the HW5 key simulation.
        When sigma=1 the two are identical; scaling generalises to any sigma.
        """
        y = self.beta_0 + self.beta_1 * self.x + self.sigma * self.rng.standard_normal(self.n)
        return self.x, y

    # ------------------------------------------------------------------
    # fit_slope
    # ------------------------------------------------------------------
    def fit_slope(self, x, y):
        """
        Fit an OLS Simple Linear Regression model and return the slope estimate.

        Parameters
        ----------
        x : np.ndarray  (1-D)
        y : np.ndarray  (1-D)

        Returns
        -------
        float : Estimated slope (beta_1 hat).
        """
        reg = linear_model.LinearRegression()
        # sklearn requires x to be 2-D (n, 1)
        reg.fit(x.reshape(-1, 1), y)
        return reg.coef_[0]  # return the single slope coefficient

    # ------------------------------------------------------------------
    # run_simulations
    # ------------------------------------------------------------------
    def run_simulations(self, num_simulations):
        """
        Run the slope simulation `num_simulations` times.

        Calls generate_data() and fit_slope() in a loop and stores all
        slope estimates in self.slopes (replaces any previous results).

        Parameters
        ----------
        num_simulations : int
            Number of simulated datasets (and slope estimates) to generate.
        """
        # Pre-allocate array for efficiency then fill it
        slopes_array = np.zeros(num_simulations)

        for i in range(num_simulations):
            x, y = self.generate_data()          # simulate one dataset
            slopes_array[i] = self.fit_slope(x, y)  # estimate and store slope

        # Replace the slopes attribute with the completed array
        self.slopes = slopes_array

    # ------------------------------------------------------------------
    # plot_sampling_distribution
    # ------------------------------------------------------------------
    def plot_sampling_distribution(self):
        """
        Plot a histogram of the simulated slope estimates.

        If run_simulations() has not been called yet (slopes is empty),
        prints an informative error message instead.
        """
        # Guard: check that simulations have been run
        if len(self.slopes) == 0:
            print("Error: run_simulations() must be called first before plotting.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(self.slopes, bins=40, color="steelblue", edgecolor="white")
        plt.axvline(self.beta_1, color="red", linestyle="--",
                    label=f"True β₁ = {self.beta_1}")
        plt.xlabel("Estimated Slope (β̂₁)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Simulated Sampling Distribution of the OLS Slope Estimator",
                  fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # find_prob
    # ------------------------------------------------------------------
    def find_prob(self, value, sided):
        """
        Approximate a tail probability from the simulated slope distribution.

        Parameters
        ----------
        value : float
            The threshold value for the probability calculation.
        sided : str
            One of "above", "below", or "two-sided".
            - "above"     : P(β̂₁ > value)
            - "below"     : P(β̂₁ < value)
            - "two-sided" : 2 * P(β̂₁ > value) if value > median,
                            2 * P(β̂₁ < value) if value < median

        Returns
        -------
        float : Estimated probability, or None if slopes is empty.
        """
        # Guard: check that simulations have been run
        if len(self.slopes) == 0:
            print("Error: run_simulations() must be called first before finding probabilities.")
            return None

        slopes = self.slopes  # shorthand

        if sided == "above":
            # Proportion of slope estimates that exceed `value`
            prob = np.mean(slopes > value)

        elif sided == "below":
            # Proportion of slope estimates that fall below `value`
            prob = np.mean(slopes < value)

        elif sided == "two-sided":
            median_slope = np.median(slopes)
            if value > median_slope:
                # value is in the upper tail → double the upper tail probability
                prob = 2 * np.mean(slopes > value)
            else:
                # value is in the lower tail → double the lower tail probability
                prob = 2 * np.mean(slopes < value)

        else:
            raise ValueError(f"sided must be 'above', 'below', or 'two-sided'. Got: '{sided}'")

        return prob


# =============================================================================
# Demo / driver code (runs when this script is executed directly)
# =============================================================================
if __name__ == "__main__":

    # --- Create an instance of the simulator ---
    # beta_0=12, beta_1=2, x from linspace(0,10,11) repeated 3 times,
    # sigma=1, seed=10
    x_vals = np.array(list(np.linspace(start=0, stop=10, num=11)) * 3)

    sim = SLR_slope_simulator(
        beta_0=12,
        beta_1=2,
        x=x_vals,
        sigma=1,
        seed=10
    )

    # --- Attempt to plot before running simulations (should print error) ---
    print("Calling plot_sampling_distribution() before run_simulations():")
    sim.plot_sampling_distribution()

    # --- Run 10,000 simulations ---
    print("\nRunning 10,000 simulations...")
    sim.run_simulations(10000)
    print("Simulations complete.")

    # --- Plot the sampling distribution ---
    sim.plot_sampling_distribution()

    # --- Two-sided probability of the slope being larger than 2.1 ---
    prob = sim.find_prob(value=2.1, sided="two-sided")
    print(f"\nApproximate two-sided probability for slope > 2.1: {prob:.4f}")

    # --- Print the simulated slopes array ---
    print("\nSimulated slopes (first 10 shown):")
    print(sim.slopes[:10])
    print(f"... ({len(sim.slopes)} total slope estimates stored in sim.slopes)")
