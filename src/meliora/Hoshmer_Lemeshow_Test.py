from scipy.stats import chi2
import pandas as pd
import numpy as np


def hosmer_lemeshow(ratings, default_flag, prob_default, alpha, chi_stat):
    """
    A statistical test for goodness-of-fit for classification models.

    The test assesses whether the observed event rates match the expected
    event rates in pools. Models for which expected and observed event rates
    in pools are similar are well calibrated. The p-value of this test is a
    measure of the accuracy of the estimated default probabilities. The closer
    the p-value is to zero, the poorer the calibration of the model.

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

    frame = {'ratings': ratings,
             'default_flag': default_flag,
             'prob_default': prob_default
             }

    df = pd.DataFrame(frame)

    for rating in list(set(df.ratings)):
        rating_df = df[df['ratings'] == rating]
        n_g = len(rating_df)
        n_1g = sum(rating_df['default_flag'])
        p_g = np.mean(rating_df['prob_default'])
        e_g = n_g * p_g

        chi_stat += (n_1g-e_g)**2/(n_g*p_g*(1-p_g))

    # Outcomes
    p_value = 1 - chi2.cdf(chi_stat, df=len(df.ratings)-2)
    outcome = 'Pass' if p_value < alpha else 'Fail'

    return p_value, outcome
