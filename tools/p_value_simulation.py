import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tqdm import tqdm
from time import time
from scipy.stats import binom, barnard_exact, boschloo_exact, fisher_exact, chi2_contingency, poisson_means_test
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare
from statsmodels.distributions.empirical_distribution import ECDF

from .util import load_file, save_file


class PValueSimulation(object):
    def __init__(self, simulation_param_list, dir_list):
        self.simulation_param_list = simulation_param_list
        self.dir_list = dir_list

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

        return test_list

    def _test_list_result(self, x_k, y_k, x_n, y_n, test_name):
        test_list = list(filter(lambda test: test["name"] == test_name,
                                self._test_list()))
        assert len(test_list) > 0, f"Test {test_name} did not found"
        test = test_list[0]

        return test["p_value"](x_k, y_k, x_n, y_n)

    def simulate(self, random_state=None, custom_test_name_list=None, rewrite_result=False, tqdm_disable=True):
        for i, simulation_params in enumerate(self.simulation_param_list):
            sample_name = simulation_params.sample_name
            p_value_file_name = f"{sample_name}, p-value.csv"
            test_speed_file_name = f"{sample_name}, test speed.csv"
            print(sample_name)

            if not rewrite_result:
                p_value_data = load_file(p_value_file_name,
                                         dir_list=self.dir_list,
                                         index_col_list=["name", "iter_num"])
                test_speed = load_file(test_speed_file_name,
                                       dir_list=self.dir_list,
                                       index_col_list=["name"])
            else:
                p_value_data = None
                test_speed = None

            default_test_name_list = [test["name"]
                                      for test in self._test_list()
                                      if custom_test_name_list is None
                                         or test["name"] in custom_test_name_list]
            if p_value_data is not None and test_speed is not None and not rewrite_result:
                test_name_list = list(filter(lambda test_name: test_name not in p_value_data.index.unique(),
                                             default_test_name_list))
            else:
                test_name_list = default_test_name_list

            if len(test_name_list) > 0:
                sample_random_state = random_state + 2 * i
                x = binom.rvs(simulation_params.x_sample_size,
                              simulation_params.x_p,
                              random_state=sample_random_state,
                              size=simulation_params.iter_size)
                y = binom.rvs(simulation_params.y_sample_size,
                              simulation_params.y_p,
                              random_state=sample_random_state+1,
                              size=simulation_params.iter_size)

                for test_name in test_name_list:
                    print(test_name)
                    start_time = time()
                    p_value_list = []
                    for j in tqdm(range(simulation_params.iter_size), disable=tqdm_disable):
                        p_value = self._test_list_result(x[j], y[j],
                                                         simulation_params.x_sample_size,
                                                         simulation_params.y_sample_size,
                                                         test_name)
                        p_value_list.append(p_value)

                    end_time = time()
                    mean_time = 1000 * (end_time - start_time) / simulation_params.iter_size

                    test_result_data = pd.DataFrame({
                        "name": test_name,
                        "iter_num": list(range(simulation_params.iter_size)),
                        "p_value": p_value_list
                    }) \
                        .set_index(["name", "iter_num"]) \
                        [["p_value"]]

                    test_speed_data = pd.DataFrame({
                        "name": [test_name],
                        "speed": [mean_time]
                    }) \
                        .set_index(["name"]) \
                        [["speed"]]

                    print(f"{mean_time:.3f} ms per sample")

                    if p_value_data is None or test_speed is None:
                        p_value_data = test_result_data
                        test_speed = test_speed_data
                    else:
                        p_value_data = pd.concat([p_value_data, test_result_data])
                        test_speed = pd.concat([test_speed, test_speed_data])

                save_file(p_value_data, p_value_file_name, dir_list=self.dir_list)
                save_file(test_speed, test_speed_file_name, dir_list=self.dir_list)

    def plot_compare(self, custom_test_name_list=None, max_alpha=1, corr_matrix=False):
        for i, simulation_params in enumerate(self.simulation_param_list):
            sample_name = simulation_params.sample_name
            p_value_file_name = f"{sample_name}, p-value.csv"
            test_speed_file_name = f"{sample_name}, test speed.csv"

            p_value_data = load_file(p_value_file_name,
                                     dir_list=self.dir_list,
                                     index_col_list=["name", "iter_num"])
            test_speed = load_file(test_speed_file_name,
                                   dir_list=self.dir_list,
                                   index_col_list=["name"])

            default_test_name_list = [test["name"]
                                      for test in self._test_list()
                                      if custom_test_name_list is None
                                      or test["name"] in custom_test_name_list]
            if p_value_data is not None and test_speed is not None:
                test_name_list = list(filter(lambda test_name: test_name not in p_value_data.index.unique(),
                                             default_test_name_list))
            else:
                test_name_list = default_test_name_list

            if len(test_name_list) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.linspace(0, max_alpha, num=100),
                                         y=np.linspace(0, max_alpha, num=100),
                                         line={"color": "grey"},
                                         name="y = x",
                                         mode="lines"))

                p_value_data = p_value_data.pivot_table(index="iter_num", columns="name", values="p_value")
                for test_name in test_name_list:
                    p_value_list = p_value_data[test_name]
                    p_value_ecdf = ECDF(p_value_list)

                    fig.add_trace(go.Scatter(x=p_value_ecdf.x[p_value_ecdf.x <= max_alpha],
                                             y=p_value_ecdf.y[p_value_ecdf.x <= max_alpha],
                                             name=test_name,
                                             mode="lines"))

                fig.update_layout(title=sample_name,
                                  xaxis_title="alpha",
                                  yaxis_title="ECDF p-value",
                                  legend_title="Criteria")
                fig.show()

                if corr_matrix:
                    corr_data = p_value_data.corr()
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=corr_data.columns,
                                             y=corr_data.index,
                                             z=np.array(corr_data),
                                             text=corr_data.values,
                                             texttemplate="%{text:.2f}"))
                    fig.show()