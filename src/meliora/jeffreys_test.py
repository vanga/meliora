import numpy as np
import pandas as pd
from scipy.stats import norm, beta


def _jeffreys(pd, d, n, p):
    """
    Calculate Jeffrey's test outcome

    Parameters
    ----------
    pd : estimated default probability
    d : number of defaults
    n : number of obligors
    p : realised default rate

    Returns
    -------
    p_value : Jeffrey's p-value

    Notes
    -----
    Given the Jeffreys prior for the binomial proportion, the
    posterior distribution is a beta distribution with shape parameters a = D + 1/2 and
    b = N − D + 1/2. Here, N is the number of customers in the portfolio/rating grade and
    D is the number of those customers that have defaulted within that observation
    period. The p-value (i.e. the cumulative distribution function of the aforementioned
    beta distribution evaluated at the PD of the portfolio/rating grade) serves as a
    measure of the adequacy of estimated PD.
    """

    a = d + 0.5
    b = n - d + 0.5
    p_value = beta.cdf(pd, a, b)

    return p_value


def jeffreys_test(data, ratings, default_flag, predicted_pd, conf_level=0.05):
    """Calculate the Jeffrey's test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with three columns
            ratings : PD rating class of obligor
            default_flag : 1 for defaulted and 0 for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating : Contains the ratings of each class/group
        PD : predicted default rate in each group
        N : number of obligors in each group
        D : number of defaults in each group
        Default Rate : realised default rate per each group
        P-Value : Jeffreys p-value


    Notes
    -----
    The Jeffreys test compares forecasted defaults with observed defaults in a binomial
    model with independent observations under the null hypothesis that the PD applied
    in the portfolio/rating grade at the beginning of the relevant observation period is
    greater than the true one (one-sided hypothesis test). The test statistic is the PD of
    the portfolio/rating grade.

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 20-21, 2019.


    Examples
    --------

    >>> ratings = random.choices(['A', 'B', 'C'],  [0.4, 0.5, 0.1], k=1000)
    >>> default_flag = random.choices([0, 1],  [0.9, 0.1], k=1000)
    >>> probs_default = [np.clip(random.normalvariate(0.1, 0.05), 0, 1) for x in range(1000)]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import jeffreys_test
    >>> jeffreys_test(test_data, 'ratings', 'default_flag', 'probs_default')

            PD	        N	D	Default Rate p_value
    ratings
    A	    0.096349	406	42	0.103448	 0.307845
    B	    0.097796	500	53	0.106000	 0.264328
    C	    0.100591	94	5	0.053191	 0.946477

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, 1] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Jeffrey's test outcome for each rating
    df["p_value"] = _jeffreys(df["PD"], df["D"], df["N"], conf_level)

    return df
