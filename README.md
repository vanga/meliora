[![PyPI version](https://badge.fury.io/py/vangap-meliora1.svg)](https://badge.fury.io/py/vangap-meliora1) ![CI Build](https://github.com/vanga/meliora/actions/workflows/CI.yml/badge.svg?event=push)

About the package
------------------

**meliora** is a Python package that provides a set of statistical tests and tools to assess the performance of the credit risk models. All tests are covered with unit tests and algorithms have been replicated in other tools like R, MATLAB and SAS to avoid errors. Whenever possible, the definition of the test was retrieved from the authoritive source like the EBA, the ECB or the Basel Committee.

The main contributors started building their first statistical credit models back in 2003. Over the years, we have impemented similar set of tests in several different financial institutions. 

This package is standing on the shoulders of giants as it makes heavy use of the Python
ecosystem and especially Scikit-learn, Scipy and Statsmodels. Several functions are straightforward
wrappers using these resources and are provided to the user for convenience purposes. The authors
have taken great care to ensure that no part of this package contains proprietary code. 

Main aim
-----------------
The aim of the package is to provide all common tests used by today's modellers when developing, maintaining and validating their PD, LGD, EAD and prepayment models. The aim of this package is to provide credit risk practioners with the tools to develop their credit risk models without reinventing the wheel. 

Main Features
-----------------
  - tests cover both IFRS 9 and IRB models as well as non-regulatory models
  - the tool contains more than 30 tests
  - all test have been covered with unit tests 
  - the tests have been documented in detail
  - commonly accepted tresholds have been provided for convenience purposes

  For the list of all tests, see Overview > List of tests

Tests that are currently included in the package
--------------------------------------------------

| #  | Name                                               | Area             | Estimate |
|----|----------------------------------------------------|------------------|----------|
| 1  | Binomial test                                      | Calibration      | PD       |
| 2  | Chi-Square test (Hoshmer-Lemeshow test)            | Calibration      | PD       |
| 3  | Normal test                                        | Calibration      | PD       |
| 4  | Traffic lights approach                            | Calibration      | PD       |
| 5  | Spiegehalter test                                  | Calibration      | PD       |
| 6  | Redelmeier test                                    | Calibration      | PD       |
| 7  | Herfhindahl index / Concentration of rating grades | Concentration    | PD       |
| 8  | Brier score                                        | Discrimination   | PD       |
| 9  | Receiver Operating Characteristic                  | Discrimination   | PD       |
| 10 | Accuracy Ratio                                     | Discrimination   | PD       |
| 11 | Kendall’s τ                                        | Discrimination   | PD       |
| 12 | Somers’ D                                          | Discrimination   | PD       |
| 13 | The Pietra Index                                   | Discrimination   | PD       |
| 14 | Conditional Information Entropy Ratio              | Discrimination   | PD       |
| 15 | Kullback-Leibler distance                          | Discrimination   | PD       |
| 16 | Information value                                  | Discrimination   | PD       |
| 17 | Bayesian error rate                                | Discrimination   | PD       |
| 18 | Cumulative LGD accuracy ratio                      | Discrimination   | LGD      |
| 19 | Loss Capture Ratio                                 | Discrimination   | LGD      |
| 20 | Kolmogorov-Smirnov test                            | Discrimination   | PD       |
| 21 | Spearman’s rank correlation                        | Discrimination   | LGD      |
| 22 | Jeffrey's test                                     | Discrimination   | PD       |
| 23 | ELBE back-test using a t-test                      | Discrimination   | LGD      |
| 24 | Migration matrices test                            | Discrimination   | PD       |
| 25 | Loss Shortfall                                     | Predictive power | LGD      |
| 26 | Mean Absolute Deviation                            | Predictive power | LGD      |
| 27 | Bucket test                                        | Predictive power | LGD      |
| 28 | Transition matrix test                             | Predictive power | LGD      |
| 29 | Population Stability Index                         | Stability        | PD       |
| 30 | Stability of transition matrices                   | Stability        | PD       |

Full list of dependencies
---------------------------
- NumPy (https://www.numpy.org)
- Pandas (https://pandas.pydata.org/)
- Statsmodels (https://www.statsmodels.org/)
- Scikit-learn (https://scikit-learn.org/)
- Scipy (https://scipy.org/)


Getting Help
------------------

For usage questions, send an email to anton.treialt@aistat.com

License
----------------------
MIT License
