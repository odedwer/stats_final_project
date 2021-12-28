import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.api import anova_lm, AnovaRM
from statsmodels.formula.api import ols
import IPython.display as ipd
import ipywidgets as wd
import re
import sys
from tqdm import tqdm
from functools import lru_cache


class Utils:
    """
    This class provides statistic and parsing utilities to be used by the other classes
    """
    DATA_PATH = r"https://raw.githubusercontent.com/odedwer/stats_final_project_datasets/main/"

    @staticmethod
    @lru_cache(maxsize=None)
    def get_dataset(dataset_name) -> pd.DataFrame:
        """
        :return: pandas DataFrame that is the dataset with the given dataset_name.
        """
        return pd.read_csv(Utils.DATA_PATH + dataset_name + ".csv")

    @staticmethod
    def get_questions() -> tuple[dict, dict]:
        """
        Reads the data tagging CSV and parses it into a dictionary whose keys are different statistical tests
        and values are list of question parameters
        :return: questions parameters dictionary
        """
        # get from github
        questions = pd.read_csv(Utils.DATA_PATH + r"stat_final_project_data_tagging.csv")
        question_parameters = dict()
        question_extra_str = dict()
        # start parsing
        for test in questions["test"].unique():
            if test not in question_parameters.keys():
                question_parameters[test] = []
                question_extra_str[test] = []
            # subset just what we need
            test_df = questions.loc[questions["test"] == test, :]
            possible_combs = []
            extra_strs = []
            # for each row generate all possible questions
            for idx, row in test_df.iterrows():
                dependent_vars = [s for s in re.split("; *", str(row["dependent_var"])) if s and s != 'nan']
                independent_vars1 = [s for s in re.split("; *", str(row["independent_var1"])) if s and s != 'nan']
                independent_vars2 = [s for s in re.split("; *", str(row["independent_var2"])) if s and s != 'nan']
                extra_vars = [s for s in re.split("; *", str(row["extra_independent_vars"])) if s and s != 'nan']
                cur_extra_str = str(row["independent_var_str_addition"])
                cur_extra_str = (" " + cur_extra_str) if cur_extra_str != 'nan' else ""

                if dependent_vars:  # not chi-squared
                    if independent_vars1 and independent_vars2:  # 2-way anova
                        possible_combs += [(str(row["data"]), dep, indep1, indep2) for dep in dependent_vars for indep1
                                           in independent_vars1 for indep2 in
                                           independent_vars2 if dep != indep1 and dep != indep2 and indep1 != indep2]
                        extra_strs += [cur_extra_str for dep in dependent_vars for indep1
                                       in independent_vars1 for indep2 in
                                       independent_vars2 if dep != indep1 and dep != indep2 and indep1 != indep2]
                        if extra_vars:
                            print(
                                f"Error! Two way anova with an extra independent variable, "
                                f"meaning 3-way anova required! Row {idx}", file=sys.stderr)
                    else:
                        if extra_vars:  # one way anova framing that should be performed as anova 2-way
                            if 'ANOVA 1-way' not in question_parameters.keys():
                                question_parameters['ANOVA 1-way'] = []
                                question_extra_str['ANOVA 1-way'] = []
                            question_parameters['ANOVA 1-way'] += [(str(row["data"]), dep, indep1, extra) for dep in
                                                                   dependent_vars for indep1 in
                                                                   independent_vars1 for extra in extra_vars if
                                                                   dep != indep1 and dep != extra and indep1 != extra]
                            question_extra_str['ANOVA 1-way'] += [cur_extra_str for dep in
                                                                  dependent_vars for indep1 in
                                                                  independent_vars1 for extra in extra_vars if
                                                                  dep != indep1 and dep != extra and indep1 != extra]
                        else:  # any other test
                            possible_combs += [(str(row["data"]), dep, indep1) for dep in dependent_vars for indep1 in
                                               independent_vars1 if dep != indep1]
                            extra_strs += [cur_extra_str for dep in dependent_vars for indep1 in
                                           independent_vars1 if dep != indep1]

                else:  # chi squared
                    possible_combs += [(str(row["data"]), indep1, indep2) for indep1 in independent_vars1 for indep2 in
                                       independent_vars2 if indep1 != indep2]
                    extra_strs += [cur_extra_str for indep1 in independent_vars1 for indep2 in
                                   independent_vars2 if indep1 != indep2]
                    if extra_vars:
                        print(
                            f"Error! It doesn't make sense to have an "
                            f"extra variable for a 2-variable chi-square test! Row {idx}", file=sys.stderr)
            question_parameters[test] += possible_combs[:]
            question_extra_str[test] += extra_strs[:]
        return question_parameters, question_extra_str

    # scatter plot df[x] vs df[y], if color!=None, use it for hue
    @staticmethod
    def scatter(df, x, y, color=None):
        sns.scatterplot(x=df[x], y=df[y], hue=df[color])

    # regular plot df[x],df[y]
    @staticmethod
    def plot(df: pd.DataFrame, x, y):
        plt.plot(df[x], df[y])

    @staticmethod
    def hist(df, x):
        return plt.hist(df[x], bins=df[x].unique().size // 5)

    # perform bootstrap on df rows, calculating the statistic_func on the bootstrapped df and plotting the histogram
    @staticmethod
    def bootstrap(df: pd.DataFrame, statistic_func):
        boot_idx = np.random.choice(df.index, (10000, df.index.size))
        boot_dist = np.zeros(10000, dtype=float)
        for i in range(boot_dist.size):
            boot_dist[i] = statistic_func(pd.DataFrame(df.values[boot_idx[0]], columns=df.columns))
        plt.hist(boot_dist, bins=50)
        return boot_dist

    # calculate the statistic_func on the df with permuted df[to_permute] and return the created distribution
    @staticmethod
    def permutation(df: pd.DataFrame, to_permute, statistic_func):
        perm_dist = np.zeros(10000, dtype=float)
        for i in range(10000):
            perm_df = df.copy()
            perm_df[to_permute] = np.random.choice(perm_df[to_permute], perm_df[to_permute].size, False)
            perm_dist[i] = statistic_func(perm_df)
        n, bins, _ = plt.hist(perm_dist)
        statistic = statistic_func(df)
        plt.axvline(0, ymax=n.max() * 1.1, linestyle=":", color="k")
        return perm_dist, statistic

    # perform permutation test fo the statistic func where the permuted label is to_permute.
    @staticmethod
    def permutation_test(df: pd.DataFrame, to_permute, statistic_func):
        perm_dist, statistic = Utils.permutation(df, to_permute, statistic_func)
        print(f"Percentage of permutations below the statistic: f{np.mean(perm_dist < statistic)}")

    # reject entries based on SD threshold. Print how many were rejected and return the df without the rejected points
    @staticmethod
    def sd_reject(df: pd.DataFrame, reject_columns, thresh):
        numeric_df: pd.DataFrame = df[reject_columns].select_dtypes(np.number)
        outliers = (np.abs(stats.zscore(numeric_df)) < thresh).any(axis=1)
        print(f"{np.sum(outliers)} rejected based on rejection criteria of {thresh} standard deviations")
        return df.loc[~outliers,]

    @staticmethod
    def get_t_test_func(q, col1, col2, col2_val1, col2_val2, method, alternative):
        if not (Utils.validate_var_name(q, col1)
                and Utils.validate_var_name(q, col2) and
                Utils.validate_var_value(q, col2, col2_val1) and
                Utils.validate_var_value(q, col2, col2_val2)):
            return lambda df: None
        if method == "paired":
            return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, alternative=alternative: \
                stats.ttest_rel(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1],
                                alternative=alternative)
        elif method == "independent":
            return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, alternative=alternative: \
                stats.ttest_ind(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1],
                                alternative=alternative)
        else:
            print(f"t-test can only be for paired/independent!", file=sys.stderr)

    @staticmethod
    def get_t_test_func_for_subset(q, col1, col1_val, col2, col2_val1, col2_val2, method, alternative):
        if not (Utils.validate_var_name(q, col1)
                and Utils.validate_var_name(q, col2) and
                Utils.validate_var_value(q, col2, col2_val1) and
                Utils.validate_var_value(q, col2, col2_val2) and
                Utils.validate_var_value(q, col1, col1_val)):
            return lambda df: None
        if method == "paired":
            return lambda df, col1=col1, col1_val=col1_val, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, \
                          alternative=alternative: stats.ttest_rel(df.loc[(df[col2] == col2_val1).to_numpy() &
                                                                          (df[col1] == col1_val).to_numpy(), col1],
                                                                   df.loc[(df[col2] == col2_val2).to_numpy() &
                                                                          (df[col1] == col1_val).to_numpy(), col1],
                                                                   alternative=alternative)
        elif method == "independent":
            return lambda df, col1=col1, col1_val=col1_val, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, \
                          alternative=alternative: stats.ttest_ind(df.loc[(df[col2] == col2_val1).to_numpy() &
                                                                          (df[col1] == col1_val).to_numpy(), col1],
                                                                   df.loc[(df[col2] == col2_val2).to_numpy() &
                                                                          (df[col1] == col1_val).to_numpy(), col1],
                                                                   alternative=alternative)
        else:
            print(f"t-test can only be for paired/independent!", file=sys.stderr)

    @staticmethod
    def t_test(q, col1, col2, col2_val1, col2_val2, method, alternative):
        return Utils.get_t_test_func(q, col1, col2, col2_val1, col2_val2, method, alternative)(q.get_dataset())

    @staticmethod
    def t_test_for_specific_levels(q, col1, col1_val, col2, col2_val1, col2_val2, method, alternative):
        return Utils.get_t_test_func_for_subset(q, col1, col1_val, col2, col2_val1, col2_val2, method, alternative)(
            q.get_dataset())

    @staticmethod
    def one_way_anova(q, dependent_var, factor):
        return anova_lm(ols(f"{dependent_var}~{factor}", data=q.get_dataset()).fit())

    @staticmethod
    def two_way_anova(q, dependent_var, factor1, factor2):
        return anova_lm(ols(f"{dependent_var}~{factor1}*{factor2}", data=q.get_dataset()).fit())

    @staticmethod
    def rm_anova(q, dependent_var, subject, factor1):
        return AnovaRM(q.get_dataset(), f'{dependent_var}', f'{subject}', [f'{factor1}']).fit().anova_table

    @staticmethod
    def get_mann_whitney_func(q, col1, col2, col2_val1, col2_val2):
        if not (Utils.validate_var_name(q, col1)
                and Utils.validate_var_name(q, col2) and
                Utils.validate_var_value(q, col2, col2_val1) and
                Utils.validate_var_value(q, col2, col2_val2)):
            return lambda df: None
        return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2: \
            stats.mannwhitneyu(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1])

    @staticmethod
    def mann_whitney_test(q, col1, col2, col2_val1, col2_val2):
        return Utils.get_mann_whitney_func(q, col1, col2, col2_val1, col2_val2)(q.get_dataset())

    @staticmethod
    def get_regress_func(q, dep_var, indep_var):
        if not (Utils.validate_var_name(q, dep_var) and Utils.validate_var_name(q, indep_var)):
            return None
        return lambda df, dep_var=dep_var, indep_var=indep_var: stats.linregress(q[dep_var], q[indep_var])

    @staticmethod
    def regress(q, dep_var, indep_var):
        res = Utils.get_regress_func(q, dep_var, indep_var)(q.get_dataset())
        if res is not None:
            slope, intercept, rvalue, pvalue, stderr, intercept_stderr = res
            return slope, intercept, rvalue, pvalue
        else:
            return None

    @staticmethod
    def get_spearman_correlation_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var=var1, var2=var2: stats.spearmanr(q[var1], q[var2])

    @staticmethod
    def spearman_correlation(q, var1, var2):
        return Utils.get_spearman_correlation_func(q, var1, var2)(q.get_dataset())

    @staticmethod
    def get_pearson_correlation_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var=var1, var2=var2: stats.pearsonr(q[var1], q[var2])

    @staticmethod
    def pearson_correlation(q, var1, var2):
        return Utils.get_pearson_correlation_func(q, var1, var2)(q.get_dataset())

    # TODO: implement
    @staticmethod
    def get_chi_for_indep_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var1=var1, var2=var2: stats.chi2_contingency(pd.crosstab(df[var1], df[var2]))[:2]

    @staticmethod
    def chi_for_indep(q, var1, var2):
        return Utils.get_chi_for_indep_func(q, var1, var2)(q.get_dataset())

    # TODO: implement
    @staticmethod
    def cohens_d(q, col1, col2, col2_val1, col2_val2):
        if not (Utils.validate_var_name(q, col1)
                and Utils.validate_var_name(q, col2) and
                Utils.validate_var_value(q, col2, col2_val1) and
                Utils.validate_var_value(q, col2, col2_val2)):
            return None
        df = q.get_dataset()
        x = df[col1][df[col2] == col2_val1]
        y = df[col1][df[col2] == col2_val2]
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)

    @staticmethod
    def eta_squared(aov_res):
        return (aov_res["sum_sq"] / aov_res["sum_sq"].sum())[:-1].rename("eta^2")

    @staticmethod
    def r_squared_pearson_or_regression(q, var1, var2):
        res = Utils.pearson_correlation(q, var1, var2)
        if res is None:
            return None
        return res[0] ** 2

    @staticmethod
    def r_squared_spearman(q, var1, var2):
        res = Utils.spearman_correlation(q, var1, var2)
        if res is None:
            return None
        return res[0] ** 2

    # TODO: implement
    @staticmethod
    def multiple_hyp_thresh_perm():
        pass

    # TODO: implement
    @staticmethod
    def multiple_hyp_thresh_bon(alpha, number_of_comparisons):
        return alpha / number_of_comparisons

    # TODO: implement
    @staticmethod
    def multiple_hyp_thresh_fdr(p_values_list, alpha):
        p_values_list = np.array(p_values_list)
        m = p_values_list.size
        sorting_idx = np.argsort(p_values_list)[::-1]
        for i, idx in enumerate(sorting_idx):
            if p_values_list[idx] < (i + 1) * (alpha / m):
                return (i + 1) * (alpha / m)

    @staticmethod
    def validate_var_name(q, col1):
        if col1 not in q.get_dataset().columns:
            print(f"{col1} isn't a variable in the dataset!\n possible variables: {q.get_dataset().columns}",
                  file=sys.stderr)
            return False
        return True

    @staticmethod
    def validate_var_value(q, col, val):
        if val not in q.get_dataset()[col]:
            print(
                f"{val} isn't a value of in the dataset in column {col}!\n possible values: {q.get_dataset()[col].unique()}",
                file=sys.stderr)
            return False
        return True

    @staticmethod
    def _get_general_two_groups_func(q, col1, col2, col_func, groups_func):
        if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
            return None
        return lambda df, col1=col1, col2=col2, col_func=col_func, groups_func=groups_func: groups_func(
            col_func(df[col1]), col_func(df[col2]))

    @staticmethod
    def distance_of_means(q, col1, col2):
        return Utils._get_general_two_groups_func(q, col1, col2, np.mean, lambda a, b: a - b)

    @staticmethod
    def distance_of_medians(q, col1, col2):
        return Utils._get_general_two_groups_func(q, col1, col2, np.median, lambda a, b: a - b)


class GUI:
    """
    This class generates a GUI for a question - providing different plots, statistical tests etc. via
    manual_interact with Run button for choosing the parameters (if plot - which type? what are the x,y,color values?
    If test, between which rows? and so on)
    On "Run" press, will call the corresponding Utils function with the chosen options. The Utils function should
    validate the chosen options and print an error message if there is a problem with the options.
    """
    pass


class ProjectQuestion:
    OUTLIER_THRESH = 2.5  # outlier threshold in multiples of standard deviation
    INLIER_THRESH = 1.5  # outlier threshold in multiples of standard deviation
    DEF_NUM_ENTRIES = 100

    def __init__(self, group: int, q_type: str, q_params: tuple, extra_str: str, outliers: bool, idx):
        self._group = group
        self._idx = idx
        self._num_entries = ProjectQuestion.DEF_NUM_ENTRIES
        self._q_type = q_type
        self._full_dataset = Utils.get_dataset(q_params[0])
        self._vars = list(q_params[1:])
        self._extra_str = extra_str
        self._outliers = outliers
        self._dataset = None
        self._choose_dataset()
        self._filtered_dataset = None

    def get_dataset(self):
        return self._filtered_dataset if self._filtered_dataset else self._dataset

    def reject_outliers(self, columns, thresh):
        if not isinstance(columns, list):
            columns = [columns]
        if sum([Utils.validate_var_name(self, col) for col in columns]) != len(columns):
            return None
        self._filtered_dataset = Utils.sd_reject(self._dataset, columns, thresh)

    def _remove_filter(self):
        self._filtered_dataset = None

    def __repr__(self):
        if self._q_type == 'Chi squared':
            ret_str = f"Are %s{self._extra_str} and %s{self._extra_str} independent?"
            return ret_str % tuple(self._vars)
        elif self._q_type == 'ANOVA 2-way':
            ret_str = f"Do %s{self._extra_str} or %s{self._extra_str} have an effect on %s{self._extra_str}? " \
                      f"Does the effect of %s{self._extra_str} on %s{self._extra_str} for different levels of %s " \
                      f"{self._extra_str}?"
            return ret_str % (
                self._vars[1], self._vars[2], self._vars[0], self._vars[1], self._vars[0], self._vars[2])
        elif self._q_type == 'ANOVA 1-way' or self._q_type == 'Repeated Measures ANOVA':
            ret_str = f"Does %s{self._extra_str} have an effect on %s{self._extra_str}?"
            return ret_str % (self._vars[1], self._vars[0])
        elif self._q_type == 'Regression':
            ret_str = f"What is the relationship between %s{self._extra_str} and %s{self._extra_str} and is it significant?"
            return ret_str % tuple(self._vars)
        elif self._q_type == 'independent samples t-test' or self._q_type == 'paired t-test':
            unique_y = self.get_dataset()[self._vars[1]].unique()
            self._seed()
            ret_str = f"Is %s{self._extra_str} %s for %s=%s compared to %s=%s?"
            return ret_str % (
                self._vars[0], np.random.choice(['larger', 'smaller', 'different']), self._vars[1], unique_y[0],
                self._vars[1], unique_y[1])
        elif self._q_type == 'Mann Whitney':
            ret_str = f"Do %s{self._extra_str} distributions differ based on %s{self._extra_str}?"
            return ret_str % tuple(self._vars)

    def is_same(self, q_params):
        """
        :return: True if q_params will generate the same question as self, False otherwise
        """
        if self._vars == q_params[1:]:
            return True
        return False

    def _choose_dataset(self):
        relevant_columns_dataset = self._full_dataset[
            self._vars] if self._q_type != "Repeated Measures ANOVA" or "subject" not in self._full_dataset.columns else \
            self._full_dataset[["subject"] + self._vars]
        relevant_columns_dataset: pd.DataFrame = relevant_columns_dataset.dropna()
        relevant_columns_dataset = relevant_columns_dataset.reset_index(drop=True)
        # choose the number of entries
        if relevant_columns_dataset.shape[0] < ProjectQuestion.DEF_NUM_ENTRIES:
            self._num_entries = relevant_columns_dataset.shape[0]
        #####
        # create a dataset with outliers if required
        #####
        # find out/inliers if the data has no categorical variables
        if self._q_type != 'Chi squared':
            # both are numeric, no need to make sure all levels are present
            numeric_df = relevant_columns_dataset.select_dtypes(include=np.number)
            outlier_rows = \
                np.where((np.abs(stats.zscore(numeric_df)) > ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
            inlier_rows = np.where((np.abs(stats.zscore(numeric_df)) < ProjectQuestion.INLIER_THRESH).all(axis=1))[
                0]
            if self._num_entries > outlier_rows.size + inlier_rows.size:
                self._num_entries = inlier_rows.size
            self._seed()
            max_outliers = min([outlier_rows.size, self._num_entries // 10])
            num_outliers = np.random.randint(1, max_outliers) if self._outliers and max_outliers > 1 else 0
            num_inliers = self._num_entries - num_outliers
            # get a boolean vector the size of the data for inliers and outliers
            outlier_rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            outlier_rows_idx[np.random.choice(outlier_rows, num_outliers, replace=False)] = 1
            inlier_rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            if self._q_type == "Regression":
                inlier_rows_idx[np.random.choice(inlier_rows, num_inliers, replace=False)] = 1
            else:  # t-test, anova 1/2 way - need to make sure all levels are present
                if "2-way" not in self._q_type:
                    # only need vars[1]
                    unique_vals = relevant_columns_dataset[self._vars[1]].unique()
                    num_inliers_per_val = num_inliers // unique_vals.size
                    for val in unique_vals:
                        val_rows = numeric_df.index[relevant_columns_dataset[self._vars[1]] == val]
                        inlier_rows_idx[
                            np.random.choice(val_rows, min(num_inliers_per_val, val_rows.size), replace=False)] = 1

                else:
                    # need vars[1] and vars[2]
                    unique_vals1 = relevant_columns_dataset[self._vars[1]].unique()
                    unique_vals2 = relevant_columns_dataset[self._vars[2]].unique()
                    num_inliers_per_val_comb = num_inliers // (unique_vals1.size * unique_vals2.size)
                    for val1 in unique_vals1:
                        for val2 in unique_vals2:
                            val_rows = numeric_df.index[(relevant_columns_dataset[self._vars[1]] == val1).to_numpy() & (
                                    relevant_columns_dataset[self._vars[2]] == val2).to_numpy()]
                            inlier_rows_idx[np.random.choice(val_rows, min(num_inliers_per_val_comb, val_rows.size),
                                                             replace=False)] = 1
            # subset the dataset
            self._dataset = relevant_columns_dataset.loc[outlier_rows_idx | inlier_rows_idx, :]
        else:
            rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            rows_idx[np.random.choice(np.arange(relevant_columns_dataset.index.size), size=self._num_entries,
                                      replace=False)] = True
            self._dataset = relevant_columns_dataset.loc[rows_idx, :]
        self._used_index = self._dataset.index[:]
        self._dataset.reset_index(drop=True, inplace=True)

    def _seed(self) -> None:
        np.random.seed(self._group * self._idx)

    def print_dataset(self):
        ipd.display(self.get_dataset())


class Project:
    """
    This class represents the first part of the project.
    It contains NUM_QUESTIONS, where the type of tests required for each question can/can't repeat based on the value
    of REPEAT_Q_TYPES.
    """
    NUM_QUESTIONS = 3
    REPEAT_Q_TYPES = False

    def __init__(self, group: int, id: str):
        self._group = group
        self._id = id
        self._possible_questions, self._questions_extra_str = Utils.get_questions()
        self._seed()
        self._question_types = np.random.choice(list(self._possible_questions.keys()), replace=Project.REPEAT_Q_TYPES,
                                                size=Project.NUM_QUESTIONS)
        self._extra_var = None
        if 'Anova 1-way' in self._question_types:
            self._extra_var = bool(np.random.binomial(1, 0.5, 1))
        self._questions = []
        self._generate_questions()

    def _generate_questions(self):
        self._seed()
        self._outliers = np.random.binomial(1, 0.5, len(self._question_types)).astype(bool)
        for i, key in enumerate(self._question_types):
            if key == 'Anova 1-way':
                if self._extra_var:
                    q_options = [(q, self._questions_extra_str[key][i]) for i, q in
                                 enumerate(self._possible_questions[key])
                                 if len(q) == 4]
                else:
                    q_options = [(q, self._questions_extra_str[key][i]) for i, q in
                                 enumerate(self._possible_questions[key])
                                 if len(q) == 3]
            else:
                q_options = list(zip(self._possible_questions[key], self._questions_extra_str[key]))
            q_params = q_options[np.random.choice(np.arange(len(q_options)))]
            if Project.REPEAT_Q_TYPES:
                while sum([q.is_same(q_params) for q in self._questions]) > 0:
                    q_params = q_options[np.random.choice(np.arange(len(q_options)))]
            self._questions.append(
                ProjectQuestion(self._group, key, q_params[0], q_params[1], self._outliers[i], i + 1))

    def _seed(self) -> None:
        np.random.seed(self._group)

    def __getitem__(self, idx):
        if idx >= Project.NUM_QUESTIONS:
            raise IndexError(f"There are only {Project.NUM_QUESTIONS} questions in this part!"
                             f"Please provide a subscript between 0-{Project.NUM_QUESTIONS - 1}, and not {idx}!")
        return self._questions[idx]


# %% tests
questions, extra = Utils.get_questions()
q_list = []
qs = dict()
for key, value in tqdm(questions.items()):
    count = 1
    qs[key] = []
    for i, q_params in enumerate(value):
        q = ProjectQuestion(1, key, q_params, extra[key][i], False, count)
        qs[key].append(q)
        q_list.append({"test": key, "count": count, "variables": q_params, "question": q.__repr__()})
        count += 1

# %%
p = Project(111,"315594044")
p[2]
# pd.DataFrame(q_list).to_csv("all_questions.csv")
# q = qs["Repeated Measures ANOVA"][0]
# smoke_data = Utils.get_dataset("smoke_ban")
# smoke_data
