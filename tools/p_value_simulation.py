import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import barnard_exact, boschloo_exact, fisher_exact, chi2_contingency, poisson_means_test
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare

from .util import load_file, save_file
from .p_value_simulation_params import PValueSimulationParams


class PValueSimulation(object):
    def __init__(self, simulation_param_list, dir_list,
                 alternative, true_hypothesis):
        self.simulation_param_list = simulation_param_list
        self.dir_list = dir_list
        self.alternative = alternative
        self.true_hypothesis = true_hypothesis

    def _test_list(self):
        def g_test_adjusted(x_k, y_k, x_n, y_n):
            if x_k == 0 and y_k == 0:
                if x_n >= y_n:
                    x_k += 1
                    y_k += y_n / x_n
                else:
                    x_k += x_n / y_n
                    y_k += 1
            elif x_k == x_n and y_k == y_n:
                if x_n >= y_n:
                    x_k -= 1
                    y_k -= y_n / x_n
                else:
                    x_k -= x_n / y_n
                    y_k -= 1

            return chi2_contingency([[x_k, y_k],

                                     [x_n - x_k, y_n - y_k]],
                                    lambda_="log-likelihood")

        test_list = [{
            "name": "Barnard’s exact test",
            "p_value": lambda x_k, y_k, x_n, y_n, sampling_points=32:
                              barnard_exact([[x_k, y_k],
                                             [x_n - x_k, y_n - y_k]],
                                            n=sampling_points,
                                            alternative="two-sided").pvalue
        }, {
            "name": "Boschloo’s exact test",
            "p_value": lambda x_k, y_k, x_n, y_n, sampling_points=32: \
                              boschloo_exact([[x_k, y_k],
                                              [x_n - x_k, y_n - y_k]],
                                             n=sampling_points,
                                             alternative="two-sided").pvalue
        }, {
            "name": "Fisher’s exact test",
            "p_value": lambda x_k, y_k, x_n, y_n: \
                fisher_exact([[x_k, y_k],
                              [x_n - x_k, y_n - y_k]],
                             alternative="two-sided").pvalue
        }, {
            "name": "G-test",
            "p_value": lambda x_k, y_k, x_n, y_n: g_test_adjusted(x_k, y_k, x_n, y_n).pvalue
        }, {
            "name": "Poisson means test",
            "p_value": lambda x_k, y_k, x_n, y_n: \
                poisson_means_test(x_k, x_n, y_k, y_n,
                                   alternative="two-sided").pvalue
        }, {
            "name": "Z-test",
            "p_value": lambda x_k, y_k, x_n, y_n: \
                proportions_ztest([x_k, y_k],
                                  [x_n, y_n],
                                  alternative="two-sided")[1]
        }, {
            "name": "Chi-square test",
            "p_value": lambda x_k, y_k, x_n, y_n: \
                proportions_chisquare([x_k, y_k],
                                      [x_n, y_n])[1]
        }]

        # if test_name_list is None:
        #     test_list = default_test_list
        # else:
        #     test_list = list(filter(lambda test: test["name"] in test_name_list,
        #                             default_test_list))

        return test_list

    def _load_result(self, simulation_params):

    def run(self, random_state=None, test_name_list=None, rewrite_result=False):
        for simulation_params in self.simulation_param_list:
            sample_name = simulation_params.sample_name
            p_value_file_name = f"{sample_name}, p-value.csv"
            test_speed_file_name = f"{sample_name}, test speed.csv"

            p_value_data = load_file(p_value_file_name, dir_list=self.dir_list)
            test_speed = load_file(test_speed_file_name, dir_list=self.dir_list)

            test_name_list = [test["name"] for test in self._test_list()]
            if p_value_data is not None and not rewrite_result:
                test_name_list = list(filter(lambda test_name: test_name not in all_data.index.unique(),
                                             test_name_list))
            else:
                test_name_list = test_name_list