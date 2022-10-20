from scipy.stats import binom
from scipy.stats import norm
import pandas as pd
from scipy.stats import chi2
from scipy.stats import beta


def _binomial(p, d, n):
    """

    Parameters
    ----------
    p : estimated default probability
    d : number of defaults
    n : number of obligors

    Returns
    -------
    p_value : Binomial test p-value

    Notes
    -----
    If the defaults are modeled as iid Bernoulli trials with success probability p,
    then the number of defaults d is a draw from Binomial(n, p).
    The one-sided p-value is the probability that such a draw
    would be at least as large as d.
    """

    p_value = 1 - binom.cdf(d - 1, n, p)

    return p_value


def binomial_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Binomial test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group
        N : number of obligors in each group
        D : number of defaults in each group
        Default Rate : realised default rate per each group
        p_value : Binomial test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Binomial test compares forecasted defaults with observed defaults in a binomial
    model with independent observations under the null hypothesis that the PD applied
    in the portfolio/rating grade at the beginning of the relevant observation period is
    greater than the true one (one-sided hypothesis test). The test statistic is the
    observed number of defaults.

    .. [1] "Studies on the Validation of Internal Rating Systems,"
            Basel Committee on Banking Supervision,
            p. 47, revised May 2005.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import binomial_test
    >>> binomial_test(test_data, 'ratings', 'default_flag', 'probs_default')

               PD    N   D  Default Rate   p_value  reject
    ratings
    A        0.10  401  36      0.089776  0.775347   False
    B        0.15  489  73      0.149284  0.537039   False
    C        0.20  110  23      0.209091  0.443273   False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Missing columns"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean",
                                    default_flag: ["count", "sum", "mean"]}).reset_index()
    df.columns = ["Rating class", "Predicted PD", "Total count", "Defaults", "Actual Default Rate"]

    # Calculate Binomial test outcome for each rating
    df["p_value"] = _binomial(df["Predicted PD"], df["Defaults"], df["Total count"])
    df["Reject H0"] = df["p_value"] < alpha_level

    return df


def _brier(predicted_values, realised_values):
    """

    Parameters
    ----------
    predicted_values : Pandas Series of predicted PD outcomes
    realised_values : Pandas Series of realised PD outcomes

    Returns
    -------
    mse : Brier score for the dataset

    Notes
    -----
    Calculates the mean squared error (MSE) between the outcomes
    and their hypothesized PDs. In this context, the MSE is
    also called the "Brier score" of the dataset.
    """

    # Calculate mean squared error
    errors = realised_values - predicted_values
    mse = (errors**2).sum()

    return mse


def brier_score(data, ratings, default_flag, predicted_pd):
    """Calculate the Brier score for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        brier_score : overall Brier score


    Notes
    -----
    The Brier score is the mean squared error when each default outcome
    is predicted by its PD rating. Larger values of the Brier score
    indicate a poorer performance of the rating system.

    .. [1] "Studies on the Validation of Internal Rating Systems,"
            Basel Committee on Banking Supervision,
            pp. 46-47, revised May 2005.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import brier_score
    >>> brier_score(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate brier_score
    ratings
    A        0.10000   401.0   36.0      0.089776        None
    B        0.15000   489.0   73.0      0.149284        None
    C        0.20000   110.0   23.0      0.209091        None
    total    0.13545  1000.0  132.0      0.132000    0.113112

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Brier score for the dataset
    b_score = _brier(df['PD'], df['Default Rate'])

    return b_score


def _herfindahl(df):
    """

    Parameters
    ----------
    df : Pandas DataFrame with first column providing the number
         of obligors and row index corresponding to rating labels

    Returns
    -------
    cv : coefficient of variation
    h : Herfindahl index

    Notes
    -----
    Calculates the coefficient of variation and the Herfindahl index,
    as defined in the paper [1] referenced in herfindahl_test's docstring.
    These quantities measure the dispersion of rating grades in the data.
    """

    k = df.shape[0]
    counts = df.iloc[:, 0]
    n_tot = counts.sum()
    terms = (counts / n_tot - 1 / k) ** 2
    cv = (k * terms.sum()) ** 0.5
    h = (counts**2).sum() / n_tot**2

    return cv, h


def herfindahl_multiple_period_test(data1, data2, ratings, alpha_level=0.05):
    """Calculate the Herfindahl test for a given probability of defaults buckets

    Parameters
    ----------
    data1 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor
    data2 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor

    ratings : column label for ratings
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        N_initial : number of obligors in each group and total
        h_initial : Herfindahl index for initial dataset
        N_current : number of obligors in each group and total
        h_current : Herfindahl index for current dataset
        p_value : overall Herfindahl test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Herfindahl test looks for an increase in the
    dispersion of the rating grades over time.
    The (one-sided) null hypothesis is that the current Herfindahl
    index is no greater than the initial Herfindahl index.
    The test statistic is a suitably standardized difference
    in the coefficient of variation, which is monotonically
    related to the Herfindahl index.
    If the Herfindahl index has not changed, then the
    test statistic has the standard Normal distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.
    (Note that the reference [1] has an uncommon defintion
    of Herfindahl index, whereas we return the common definition)

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 26-27, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings1 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data1 = pd.DataFrame({'ratings': ratings1})
    >>> ratings2 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data2 = pd.DataFrame({'ratings': ratings2})
    >>> from meliora import herfindahl_test
    >>> herfindahl_test(test_data1, test_data2, "ratings")

           N_initial h_initial  N_current h_current   p_value reject
    B            489      None        487      None      None   None
    A            401      None        414      None      None   None
    C            110      None         99      None      None   None
    total       1000   0.19291       1000  0.206819  0.475327  False

    """

    # Perform plausibility checks
    assert ratings in data1.columns and ratings in data2.columns, f"Ratings column {ratings} not found"
    assert max(len(data1[ratings].unique()), len(data2[ratings].unique())) < 40, "Number of PD ratings is excessive"

    # Transform input data into the required format
    df1 = pd.DataFrame({"N_initial": data1[ratings].value_counts()})
    df2 = pd.DataFrame({"N_current": data2[ratings].value_counts()})

    # Calculate the Herfindahl index for each dataset
    c1, h1 = _herfindahl(df1)
    c2, h2 = _herfindahl(df2)

    # Add a row of totals along with Herfindahl indices
    df1.loc["total"] = [df1["N_initial"].sum()]
    df1["h_initial"] = None
    df1.loc["total", "h_initial"] = h1
    df2.loc["total"] = [df2["N_current"].sum()]
    df2["h_current"] = None
    df2.loc["total", "h_current"] = h2

    # Put the results together into a single dataframe
    df = df1.join(df2)

    # Calculate Herfindahl test's p-value for the dataset
    k = df.shape[0] - 1
    z_stat = (k - 1) ** 0.5 * (c2 - c1) / (c2**2 * (0.5 + c2**2)) ** 0.5
    p_value = 1 - norm.cdf(z_stat)

    # Put the p-value and test result into the output
    df["p_value"] = None
    df.loc["total", "p_value"] = p_value
    if alpha_level:
        df["reject"] = None
        df.loc["total", "reject"] = p_value < alpha_level

    return df


def herfindahl_test(data1, ratings, alpha_level=0.05):
    """Calculate the Herfindahl test for a given probability of defaults buckets

    Parameters
    ----------
    data1 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor
    data2 : Pandas DataFrame with at least one column
            ratings : PD rating class of obligor

    ratings : column label for ratings
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        N_initial : number of obligors in each group and total
        h_initial : Herfindahl index for initial dataset
        N_current : number of obligors in each group and total
        h_current : Herfindahl index for current dataset
        p_value : overall Herfindahl test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Herfindahl test looks for an increase in the
    dispersion of the rating grades over time.
    The (one-sided) null hypothesis is that the current Herfindahl
    index is no greater than the initial Herfindahl index.
    The test statistic is a suitably standardized difference
    in the coefficient of variation, which is monotonically
    related to the Herfindahl index.
    If the Herfindahl index has not changed, then the
    test statistic has the standard Normal distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.
    (Note that the reference [1] has an uncommon defintion
    of Herfindahl index, whereas we return the common definition)

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 26-27, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings1 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data1 = pd.DataFrame({'ratings': ratings1})
    >>> ratings2 = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> test_data2 = pd.DataFrame({'ratings': ratings2})
    >>> from meliora import herfindahl_test
    >>> herfindahl_test(test_data1, test_data2, "ratings")

           N_initial h_initial  N_current h_current   p_value reject
    B            489      None        487      None      None   None
    A            401      None        414      None      None   None
    C            110      None         99      None      None   None
    total       1000   0.19291       1000  0.206819  0.475327  False

    """

    # Perform plausibility checks
    assert ratings in data1.columns, f"Ratings column {ratings} not found"
    assert len(data1[ratings].unique()) < 40, "Number of PD ratings is excessive"

    # Transform input data into the required format
    df1 = pd.DataFrame({"N_initial": data1[ratings].value_counts()})

    # Calculate the Herfindahl index for each dataset
    c1, h1 = _herfindahl(df1)

    return c1, h1


def _hosmer(p, d, n):
    """

    Parameters
    ----------
    p : Pandas Series of estimated default probabilities
    d : Pandas Series of number of defaults
    n : Pandas Series of number of obligors

    Returns
    -------
    p_value : Hosmer-Lemeshow Chi-squared test p-value

    Notes
    -----
    Calculates the Hosmer-Lemeshow test statistic, as defined in
    the paper [1] referenced in hosmer_test's docstring.
    If the hypothesized PDs are accurate and defaults are independent,
    this test statisitc is approximately Chi-squared distributed
    with degrees of freedom equal to the number of rating groups minus two.
    The p-value is the probability of such a draw being
    at least as large as the observed value of the statistic.
    """

    assert len(p) > 2, "Hosmer-Lemeshow test requires at least three groups"

    expected_def = n * p
    expected_nodef = n * (1 - p)
    if any(expected_def < 10) or any(expected_nodef < 10):
        print("Warning: a group has fewer than 10 expected defaults or non-defaults.")
        print("--> Chi-squared approximation is questionable.")

    terms = (expected_def - d) ** 2 / (p * expected_nodef)
    chisq_stat = terms.sum()
    p_value = 1 - chi2.cdf(chisq_stat, len(p) - 2)

    return p_value


def hosmer_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Hosmer-Lemeshow Chi-squared test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        p_value : overall Hosmer-Lemeshow test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Hosmer-Lemeshow Chi-squared test calculates a standardized sum
    of squared differences between the number of defaults and
    the expected number of defaults within each rating group.
    Under the null hypothesis that the PDs applied
    in the portfolio/rating grade at the beginning of the relevant observation period are
    equal to the true ones, the test statistic has an approximate Chi-squared distribution.
    Large values of this test statistic
    provide evidence against the null hypothesis.

    .. [1] "Backtesting Framework for PD, EAD and LGD - Public Version,"
            Bauke Maarse, Rabobank International,
            p. 43, 2012.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import hosmer_test
    >>> hosmer_test(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate   p_value reject
    ratings
    A        0.10000   401.0   36.0      0.089776      None   None
    B        0.15000   489.0   73.0      0.149284      None   None
    C        0.20000   110.0   23.0      0.209091      None   None
    total    0.13545  1000.0  132.0      0.132000  0.468902  False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Hosmer-Lemeshow test's p-value for the dataset
    p_value = _hosmer(df["PD"], df["D"], df["N"])

    return [p_value, p_value < alpha_level]


def _spiegelhalter(ratings, default_flag, df):
    """

    Parameters
    ----------
    ratings : Pandas Series of rating categories
    default_flag : Pandas Series of default outcomes (0/1 or False/True)
    df : Pandas DataFrame with ratings as rownames and a column of hypothesized 'PD' values

    Returns
    -------
    p_value : Spiegelhalter test p-value

    Notes
    -----
    Calculates the mean squared error (MSE) between the outcomes
    and their hypothesized PDs, which is approximately Normal.
    If the hypothesized PDs equal the true PDs, then the mean
    and standard deviation of that statistic are provided in
    the paper [1] referenced in spiegelhalter_test's docstring.
    The standardized statistic is approximately standard Normal.
    and leads to a "one-sided" p-value via the Normal cdf.
    """

    # Calculate mean squared error
    errors = default_flag - [df.loc[rating, "PD"] for rating in ratings]
    mse = (errors**2).mean()

    # Calculate null expectation and variance of MSE
    expectations = df["PD"] * (1 - df["PD"])
    variances = expectations * (1 - 2 * df["PD"]) ** 2
    n_tot = df["N"].sum()
    expected_mse = (expectations * df["N"]).sum() / n_tot
    variance_mse = (variances * df["N"]).sum() / n_tot**2

    # Calculate standardized MSE as test statistic, then its p-value
    chisq_stat = (mse - expected_mse) ** 2 / variance_mse
    p_value = 1 - chi2.cdf(chisq_stat, 1)

    return p_value


def spiegelhalter_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Spiegelhalter test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group and total
        N : number of obligors in each group and total
        D : number of defaults in each group and total
        Default Rate : realised default rate per each group and total
        p_value : overall Spiegelhalter test p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Spiegelhalter test compares forecasted defaults with observed defaults by analyzing
    the prediction errors. Under the null hypothesis that the PDs applied
    in the portfolio/rating grade at the beginning of the relevant observation period are
    equal to the true ones, the mean squared error can be standardized into
    an approximately standard Normal test statistic. Large values of this test statistic
    provide evidence against the null hypothesis.

    .. [1] "Backtesting Framework for PD, EAD and LGD - Public Version,"
            Bauke Maarse, Rabobank International,
            pp. 43-44, 2012.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import spiegelhalter_test
    >>> spiegelhalter_test(test_data, 'ratings', 'default_flag', 'probs_default')

                  PD       N      D  Default Rate   p_value reject
    ratings
    A        0.10000   401.0   36.0      0.089776      None   None
    B        0.15000   489.0   73.0      0.149284      None   None
    C        0.20000   110.0   23.0      0.209091      None   None
    total    0.13545  1000.0  132.0      0.132000  0.647161  False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean", default_flag: ["count", "sum", "mean"]})
    df.columns = ["PD", "N", "D", "Default Rate"]

    # Calculate Spiegelhalter test's p-value for the dataset
    p_value = _spiegelhalter(data[ratings], data[default_flag], df)

    # Add a row of totals
    n_tot = df["N"].sum()
    d_tot = df["D"].sum()
    pd_tot = sum(df["N"] * df["PD"]) / n_tot
    df.loc["total"] = [pd_tot, n_tot, d_tot, d_tot / n_tot]

    # Put the p-value and test result into the output
    df["p_value"] = None
    df.loc["total", "p_value"] = p_value
    if alpha_level:
        df["reject"] = None
        df.loc["total", "reject"] = p_value < alpha_level

    return df


def _jeffreys(p, d, n):
    """

    Parameters
    ----------
    p : estimated default probability
    d : number of defaults
    n : number of obligors

    Returns
    -------
    p_value : Jeffrey's "p-value" (The posterior probability of the null hypothesis)

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
    p_value = beta.cdf(p, a, b)

    return p_value


def jeffreys_test(data, ratings, default_flag, predicted_pd, alpha_level=0.05):
    """Calculate the Jeffrey's test for a given probability of defaults buckets

    Parameters
    ----------
    data : Pandas DataFrame with at least three columns
            ratings : PD rating class of obligor
            default_flag : 1 (or True) for defaulted and 0 (or False) for good obligors
            probs_default : predicted probability of default of an obligor

    ratings : column label for ratings
    default_flag : column label for default_flag
    probs_default : column label for probs_default
    alpha_level : false positive rate of hypothesis test (default .05)

    Returns
    -------
    Pandas DataFrame with the following columns :
        Rating (Index) : Contains the ratings of each class/group
        PD : predicted default rate in each group
        N : number of obligors in each group
        D : number of defaults in each group
        Default Rate : realised default rate per each group
        p_value : Jeffreys p-value
        reject : whether to reject the null hypothesis at alpha_level


    Notes
    -----
    The Jeffreys test compares forecasted defaults with observed defaults in a binomial
    model with independent observations under the null hypothesis that the PD applied
    in the portfolio/rating grade at the beginning of the relevant observation period is
    greater than the true one (one-sided hypothesis test). The test updates a Beta distribution
    (with Jeffrey's prior) in light of the number of defaults and non-defaults,
    then reports the posterior probability of the null hypothesis.

    .. [1] "Instructions for reporting the validation results
            of internal models - IRB Pillar I models for credit risks," ECB,
            pp. 20-21, 2019.


    Examples
    --------

    >>> import random, numpy as np
    >>> buckets = ['A', 'B', 'C']
    >>> ratings = random.choices(buckets,  [0.4, 0.5, 0.1], k=1000)
    >>> bucket_pds = {'A': .1, 'B': .15, 'C': .2}
    >>> probs_default = [bucket_pds[r] for r in ratings]
    >>> default_flag = [random.uniform(0, 1) < bucket_pds[r] for r in ratings]
    >>> test_data = pd.DataFrame({'ratings': ratings,
                                  'default_flag': default_flag,
                                  'predicted_pd' : probs_default})
    >>> from meliora import jeffreys_test
    >>> jeffreys_test(test_data, 'ratings', 'default_flag', 'probs_default')

               PD    N   D  Default Rate   p_value  reject
    ratings
    A        0.10  401  36      0.089776  0.748739   False
    B        0.15  489  73      0.149284  0.511781   False
    C        0.20  110  23      0.209091  0.397158   False

    """

    # Perform plausibility checks
    assert all(x in data.columns for x in [ratings, default_flag, predicted_pd]), "Not all columns are present"
    assert all(x in [0, False, 1, True] for x in data[default_flag]), "Default flag can have only value 0 and 1"
    assert len(data[ratings].unique()) < 40, "Number of PD ratings is excessive"
    assert all(x >= 0 and x <= 1 for x in data[predicted_pd]), "Predicted PDs must be between 0% and 100%"

    # Transform input data into the required format
    df = data.groupby(ratings).agg({predicted_pd: "mean",
                                    default_flag: ["count", "sum", "mean"]}).reset_index()
    df.columns = ["Rating class", "Predicted PD", "Total count", "Defaults", "Actual Default Rate"]

    # Calculate Binomial test outcome for each rating
    df["p_value"] = _jeffreys(df["Predicted PD"], df["Defaults"], df["Total count"])
    df["Reject H0"] = df["p_value"] < alpha_level

    return df
