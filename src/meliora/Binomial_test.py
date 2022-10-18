from scipy.stats import binom
import pandas as pd
import numpy as np


def binomial_test(ratings, default_flag, prob_default, alpha):
    """
    The Binomial Test evaluates whether the PD of a pool is correctly estimated.

    It does not take into account correlated defaults, and it generally yields
    an overestimate of the significance of deviations in the realized default
    rate from the forecast rate.

    If the number of default accounts per pool exceeds either the low limit
    (binomial test at 0.95 confidence) or high limit (binomial test at 0.99
    confidence), the test suggests that the model is poorly calibrated.

    The Confidence Interval indicates the confidence interval band of the actual
    PD or LGD for a pool. The Probability of Default (PD) report provides the PD
    that is estimated from the model and the actual PD with its confidence
    interval limits. If the PD that is estimated from the model is within the
    confidence interval limits of the actual PD, then the model outcomes are
    consistent with the actual outcomes.

    Parameters
    ----------
    ratings : pandas series
        Series with PD ratings
    default_flag : boolean flag
        Actual defaults in the dataset
    prob_default : pandas series
        predicted defaults for a given class
    alpha : scalar
        Confidence level

    Returns
    -------
    dataframe : pandas dataframe


    See Also
    --------
    Adjusted binomial test: Compute the binomial test assuming correlated defaults

    """

    df = pd.DataFrame(columns=["Rating", "Number of Obs", "Number of Defaults", "Average PD", "Binomial Test"])

    for rating in list(set(df.ratings)):
        # Calculation of binomial factors
        rating_df = df[df["ratings"] == rating]
        n_g = len(rating_df)
        n_1g = sum(rating_df["default_flag"])
        p_g = np.mean(rating_df["prob_default"])
        binom_factor1 = binom.cdf(n_1g, n_g, p_g)
        binom_factor2 = binom.cdf(n_1g - 1, n_g, p_g)

        # Binomial test
        flag = (binom_factor1 <= alpha / 2) | (1 - binom_factor2 <= alpha / 2)
        outcome = "reject" if flag else "Accept"

        # Store results in a dataframe
        results = {
            "Rating": rating,
            "Number of Obs": n_g,
            "Number of Defaults": n_1g,
            "Average PD": p_g,
            "Binomial Test": outcome,
        }

        results = results.append(results, ignore_index=True)

    return results
