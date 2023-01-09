# %% imports
from tqdm.notebook import tqdm
import pickle
import pingouin as pg
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
from functools import lru_cache


# %% classes & functions
def set_font_style(size=20, color='black', weight='normal'):
    ipd.display(ipd.Javascript('''
  for (rule of document.styleSheets[0].cssRules){
    if (rule.selectorText=='body') {
      rule.style.fontSize = '%dpx'
      rule.style.color = '%s'
      rule.style.fontWeight = '%s'
      break
    }
  }
  ''' % (size, color, weight)))


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
    def get_questions():
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

    @staticmethod
    def validate_not_none(*args):
        return sum([a is None for a in args]) == 0

    # scatter plot df[x] vs df[y], if color!=None, use it for hue
    @staticmethod
    def scatter(df, x, y, color=None):
        try:
            if not Utils.validate_not_none(df, x, y):
                print("You must choose x and y!", file=sys.stderr)
                return None
            if color:
                sns.scatterplot(x=df[x], y=df[y], hue=df[color])
            else:
                sns.scatterplot(x=df[x], y=df[y])
            plt.title(f"Scatter plot of {y} vs {x}")
            plt.xlabel(x)
            plt.ylabel(y)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def hist(df, x):
        try:
            plt.hist(df[x], bins=25)
            plt.title(f"{x} histogram")
            plt.xlabel(x)
            plt.ylabel("Frequencies")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def multihist(df, val_col, hue_col):
        try:
            unique_hues = df[hue_col].unique()
            num_cols = min(3, unique_hues.size)
            num_rows = int(np.ceil(unique_hues.size / num_cols))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
            count = 0
            if num_rows == 1:
                axes = axes[None, :]
            for r in range(num_rows):
                for c in range(num_cols):
                    cur_hue = unique_hues[count]
                    count += 1
                    axes[r, c].hist(df[val_col][df[hue_col] == cur_hue])
                    axes[r, c].set(title=f"{hue_col}={cur_hue}", xlabel=val_col, ylabel="Frequencies")
                    if count == unique_hues.size:
                        break
                if count == unique_hues.size:
                    break
            fig.tight_layout()
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")
            # g.map(sns.histplot, val_col)
            # sns.displot(data=df,x=val_col,kind="hist",hue=hue_col)

    @staticmethod
    def scatter_by_level(q, val_col, level_col, level1, level2):
        try:
            if not (Utils.validate_var_name(q, val_col) and
                    Utils.validate_var_name(q, level_col) and
                    Utils.validate_var_value(q, level_col, level1) and
                    Utils.validate_var_value(q, level_col, level2)):
                return None
            df = q.get_dataset()
            plt.scatter(df[val_col][df[level_col] == level1], df[val_col][df[level_col] == level2])
            plt.title(f"Scatter plot of {val_col} for {level_col}={level2} vs {level_col}={level1}")
            plt.xlabel(f"{level_col}={level1}")
            plt.ylabel(f"{level_col}={level2}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    # perform bootstrap on df rows, calculating the statistic_func on the bootstrapped df and plotting the histogram
    @staticmethod
    def bootstrap(q, statistic_func):
        try:
            df = q.get_dataset()
            boot_idx = np.random.choice(df.index, (10000, df.index.size))
            boot_dist = np.zeros(10000, dtype=float)
            for i in range(boot_dist.size):
                boot_dist[i] = statistic_func(pd.DataFrame(df.values[boot_idx[i]], columns=df.columns))
            plt.hist(boot_dist, bins=50)
            return boot_dist
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    # calculate the statistic_func on the df with permuted df[to_permute] and return the created distribution
    @staticmethod
    def permutation(q, to_permute, statistic_func, **kwargs):
        try:
            df = q.get_dataset()
            perm_dist = np.zeros(10000, dtype=float)
            for i in range(10000):
                perm_df = df.copy()
                perm_df[to_permute] = np.random.choice(perm_df[to_permute], perm_df[to_permute].size, False)
                perm_dist[i] = statistic_func(perm_df, **kwargs)
            n, bins, _ = plt.hist(perm_dist)
            statistic = statistic_func(df)
            plt.axvline(statistic, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return perm_dist, statistic
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    # perform permutation test fo the statistic func where the permuted label is to_permute.
    @staticmethod
    def permutation_test(df, to_permute, statistic_func, **kwargs):
        try:
            perm_dist, statistic = Utils.permutation(df, to_permute, statistic_func, **kwargs)
            print(f"Percentage of permutations below the statistic: f{np.mean(perm_dist < statistic)}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

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
        elif method == "independent samples":
            return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, alternative=alternative: \
                stats.ttest_ind(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1],
                                alternative=alternative)
        else:
            set_font_style(color="red")
            print(f"t-test can only be for paired/independent!", file=sys.stderr)

    @staticmethod
    def get_column_ttest_func(q, col1, col2, method, alternative):
        if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
            return None
        if method == "independent samples":
            return lambda df, col1=col1, col2=col2, method=method, alternative=alternative: \
                stats.ttest_ind(df[col1], q.get_dataset()[col2], alternative=alternative)
        else:
            return lambda df, col1=col1, col2=col2, method=method, alternative=alternative: \
                stats.ttest_rel(df[col1], q.get_dataset()[col2], alternative=alternative)

    @staticmethod
    def column_t_test(q, col1, col2, method, alternative):
        try:
            res = Utils.get_column_ttest_func(q, col1, col2, method, alternative)(q.get_dataset())
            if res: print(f"t-statistic: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def column_by_group_t_test(q, col1, col2, col2_val1, col2_val2, method, alternative):
        try:
            res = Utils.get_t_test_func(q, col1, col2, col2_val1, col2_val2, method, alternative)(q.get_dataset())
            if res: print(f"t-statistic: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_one_way_anova_model(q, dependent_var, factor):
        try:
            return ols(f"{dependent_var}~C({factor})", data=q.get_dataset()).fit()
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_two_way_anova_model(q, dependent_var, factor1, factor2):
        try:
            return ols(f"{dependent_var}~C({factor1})*C({factor2})", data=q.get_dataset()).fit()
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_rm_anova_model(q, dependent_var, subject, factor):
        try:
            return AnovaRM(q.get_dataset(), f'{dependent_var}', f'{subject}', [f'{factor}']).fit()
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def one_way_anova(q, dependent_var, factor, eta):
        try:
            aov = anova_lm(Utils.get_one_way_anova_model(q, dependent_var, factor))
            if eta:
                Utils.eta_squared(aov)
            else:
                ipd.display(aov)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def two_way_anova(q, dependent_var, factor1, factor2, eta):
        try:
            aov = anova_lm(Utils.get_two_way_anova_model(q, dependent_var, factor1, factor2))
            if eta:
                Utils.eta_squared(aov)
            else:
                ipd.display(aov)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def rm_anova(q, dependent_var, subject, factor, eta):
        try:
            aov = Utils.get_rm_anova_model(q, dependent_var, subject, factor).anova_table
            if eta:  # need to calculate alone
                res = pg.rm_anova(dv=dependent_var, subject=subject, within=factor, data=q.get_dataset(), detailed=True)
                ss = res['SS'].to_numpy()
                eta_sq = (ss[:-1] / ss.sum())[0]
                return eta_sq
            else:
                ipd.display(aov)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_mann_whitney_func(q, col1, col2, col2_val1, col2_val2, alternative):
        try:
            if not (Utils.validate_var_name(q, col1)
                    and Utils.validate_var_name(q, col2) and
                    Utils.validate_var_value(q, col2, col2_val1) and
                    Utils.validate_var_value(q, col2, col2_val2)):
                return lambda df: None
            return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, alternative=alternative: \
                stats.mannwhitneyu(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1],
                                   alternative=alternative)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_column_mann_whitney_func(q, col1, col2, alternative):
        try:
            if not (Utils.validate_var_name(q, col1)
                    and Utils.validate_var_name(q, col2)):
                return lambda df: None
            return lambda df, col1=col1, col2=col2, alternative=alternative: \
                stats.mannwhitneyu(df[col1], df[col2], alternative)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def mann_whitney_test(q, col1, col2, col2_val1, col2_val2, alternative):
        try:
            res = Utils.get_mann_whitney_func(q, col1, col2, col2_val1, col2_val2, alternative)(q.get_dataset())
            if res: print(f"U-statistic: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def column_mann_whitney_test(q, col1, col2, alternative):
        try:
            res = Utils.get_column_mann_whitney_func(q, col1, col2, alternative)(q.get_dataset())
            if res: print(f"U-statistic: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def print_regression_equation(dep, indep, slope, intercept, rvalue, pvalue, residuals):
        try:
            from decimal import getcontext
            getcontext().prec = 5
            slope, intercept, rvalue, pvalue = list(
                map(lambda v: np.round(v, 5), [slope, intercept, rvalue ** 2, pvalue]))
            mat_str = "$" + dep + f" = {slope} * " + indep + (" +" if intercept >= 0 else " ") + f"{intercept}" + "$"
            mat2_str = "$r^2=" + f"{rvalue}, p={pvalue:.5g}$"
            ipd.display(ipd.Latex(mat_str))
            ipd.display(ipd.Latex(mat2_str))
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_regress_func(q, col1, col2):
        try:
            if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
                return None
            return lambda df, col1=col1, col2=col2: stats.linregress(df[col1], df[col2])
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def regress(q, col1, col2):
        try:
            res = Utils.get_regress_func(q, col1, col2)(q.get_dataset())
            if res is not None:
                Utils.print_regression_equation(col1, col2, *res)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_spearman_correlation_func(q, col1, col2):
        try:
            if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
                return lambda df: None
            return lambda df, col1=col1, col2=col2: stats.spearmanr(df[col1], df[col2])
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def spearman_correlation(q, col1, col2):
        try:
            res = Utils.get_spearman_correlation_func(q, col1, col2)(q.get_dataset())
            if res: print(f"Spearman correlation: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_pearson_correlation_func(q, col1, col2):
        try:
            if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
                return lambda df: None
            return lambda df, col1=col1, col2=col2: stats.pearsonr(df[col1], df[col2])
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def pearson_correlation(q, col1, col2):
        try:
            res = Utils.get_pearson_correlation_func(q, col1, col2)(q.get_dataset())
            if res: print(f"Pearson correlation: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def get_chi_for_indep_func(q, col1, col2):
        try:
            if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
                return lambda df: None
            return lambda df, col1=col1, col2=col2: stats.chi2_contingency(pd.crosstab(df[col1], df[col2]))[:2]
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def chi_for_indep(q, col1, col2):
        try:
            res = Utils.get_chi_for_indep_func(q, col1, col2)(q.get_dataset())
            if res: print(f"Chi squared statistic: {res[0]}\np-value: {res[1]:.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def cohens_d(q, col1, col2, col2_val1, col2_val2):
        try:
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
            d = (np.mean(x) - np.mean(y)) / np.sqrt(
                ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
            return d
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def cohens_d_columns(q, col1, col2):
        try:
            if not (Utils.validate_var_name(q, col1)
                    and Utils.validate_var_name(q, col2)):
                return None
            df = q.get_dataset()
            x = df[col1]
            y = df[col2]
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            d = (np.mean(x) - np.mean(y)) / np.sqrt(
                ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
            return d
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def eta_squared(aov_res):
        try:
            return pd.DataFrame((aov_res["sum_sq"] / aov_res["sum_sq"].sum())[:-1].rename("eta^2"))
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def r_squared_pearson_or_regression(q, col1, col2):
        try:
            res = Utils.get_pearson_correlation_func(q, col1, col2)(q.get_dataset())
            if res is None:
                return None
            return res[0] ** 2
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def r_squared_spearman(q, col1, col2):
        try:
            res = Utils.get_spearman_correlation_func(q, col1, col2)(q.get_dataset(), col1, col2)
            if res is None:
                return None
            return res[0] ** 2
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    # TODO: implement
    @staticmethod
    def multiple_hyp_thresh_perm():
        pass

    @staticmethod
    def multiple_hyp_thresh_bon(alpha, number_of_comparisons):
        try:
            print(f"Significant p-value threshold: {alpha / float(number_of_comparisons):.5g}")
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def multiple_hyp_thresh_fdr(p_values_list, alpha):
        try:
            p_values_list = [float(p) for p in re.split("[, ]", p_values_list) if p]
            p_values_list = np.array(p_values_list)
            m = p_values_list.size
            sorting_idx = np.argsort(p_values_list)[::-1]
            for i, idx in enumerate(sorting_idx):
                if p_values_list[idx] < (i + 1) * (alpha / m):
                    print(f"Significant p-value threshold: {(i + 1) * (alpha / float(m)):.5g}")
                    return None
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def validate_var_name(q, col1):
        if col1 not in q.get_dataset().columns:
            set_font_style(color='red')
            print(f"{col1} isn't a variable in the dataset!\n possible variables: {q.get_dataset().columns.tolist()}")
            return False
        return True

    @staticmethod
    def validate_var_value(q, col, val):
        if val not in q.get_dataset()[col].unique():
            set_font_style(color='red')
            print(
                f"{val} isn't a value of in the dataset in column {col}!\n possible values: {q.get_dataset()[col].unique().tolist()}")
            return False
        return True

    @staticmethod
    def _get_general_two_groups_func(q, col1, col2, col_func, groups_func):
        try:
            if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
                return None
            return lambda df, col1=col1, col2=col2, col_func=col_func, groups_func=groups_func: groups_func(
                col_func(df[col1]), col_func(df[col2]))
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def distance_of_means(q, col1, col2):
        try:
            return Utils._get_general_two_groups_func(q, col1, col2, np.mean, lambda a, b: a - b)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def distance_of_medians(q, col1, col2):
        try:
            return Utils._get_general_two_groups_func(q, col1, col2, np.median, lambda a, b: a - b)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def by_level_permutation(kwargs, stat_func, statistic, NUM_PERM=5000):
        try:
            # print("Calculating permutations... This might take a while")
            df: pd.DataFrame = kwargs["q"].get_dataset()[[kwargs["col1"], kwargs["col2"]]]
            perm_df = df.copy()
            perm_dist = np.zeros(NUM_PERM, dtype=float)
            for i in range(NUM_PERM):
                perm_df[kwargs["col2"]] = np.random.permutation(df[kwargs["col2"]].values)
                perm_dist[i] = stat_func(perm_df, **kwargs)
            real_stat = stat_func(df, **kwargs)
            if not isinstance(real_stat, np.number):
                real_stat = real_stat[0]
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # plt.title(
            #     f"Permutation test for {statistic} between {kwargs['col2']}={kwargs['col2_val1']} and {kwargs['col2']}={kwargs['col2_val2']}")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def regression_permutation(kwargs, stat_func, statistic, NUM_PERM=5000):
        try:
            # print("Calculating permutations... This might take a while")
            df: pd.DataFrame = kwargs["q"].get_dataset()[[kwargs["col1"], kwargs["col2"]]]
            perm_df = df.copy()
            perm_dist = np.zeros(NUM_PERM, dtype=float)
            for i in range(NUM_PERM):
                perm_df[kwargs["col2"]] = np.random.permutation(df[kwargs["col2"]].values)
                perm_dist[i] = stat_func(perm_df, **kwargs)
            real_stat = stat_func(df, **kwargs)
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # plt.title(
            #     f"Permutation test for {statistic} between {kwargs['col1']} and {kwargs['col2']}")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def between_col_permutation(kwargs, stat_func, statistic, num_perm=5000):
        try:
            # print("Calculating permutations... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["col1"], kwargs["col2"]]]
            num_per_col = len(df[kwargs["col1"]])
            both_cols = df[[kwargs["col1"], kwargs["col2"]]].to_numpy().flatten()
            perm_dist = np.zeros(num_perm, dtype=float)
            for i in range(num_perm):
                np.random.shuffle(both_cols)
                perm_dist[i] = stat_func(
                    pd.DataFrame({kwargs["col1"]: both_cols[:num_per_col], kwargs["col2"]: both_cols[num_per_col:]}),
                    **kwargs)
            real_stat = stat_func(df, **kwargs)
            if not isinstance(real_stat, np.number):
                real_stat = real_stat[0]
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # plt.title(f"Permutation test for {statistic} between {kwargs['col1']} and {kwargs['col2']}")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def chisq_permutation(NUM_PERM, kwargs, stat_func):
        try:
            NUM_PERM = 5000#
            # print("Calculating permutations... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["col1"], kwargs["col2"]]]
            perm_dist = np.zeros(NUM_PERM, dtype=float)
            perm_df = df.copy()
            for i in range(NUM_PERM):
                perm_df[kwargs["col1"]] = np.random.permutation(df[kwargs["col1"]].values)
                perm_df[kwargs["col2"]] = np.random.permutation(df[kwargs["col2"]].values)
                perm_dist[i] = stat_func(perm_df, **kwargs)
            real_stat = stat_func(df, **kwargs)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # plt.title("Permutation test for Chi Squared for independence")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def rm_anova_permutation(kwargs, stat_func, num_perm=2500):#
        try:
            # print("Calculating permutations... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["subject"], kwargs["dependent_var"], kwargs["factor"]]]
            pivoted_df = df.pivot(kwargs["subject"], kwargs["factor"], kwargs["dependent_var"])  # make wide
            perm_dist = np.zeros(num_perm, dtype=float)
            for i in range(num_perm):
                permutated_values = np.stack(pivoted_df.apply(np.random.permutation, axis=1))
                perm_dist[i] = pg.rm_anova(pd.DataFrame(permutated_values, columns=pivoted_df.columns))["F"].values[0]
            real_stat = stat_func(df, **kwargs).values[0]
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # plt.title("Permutation test for RM ANOVA F-statistic")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def two_way_anova_permutation(kwargs, stat_func, num_perm=2500):#
        try:
            # print("Calculating permutations... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["dependent_var"], kwargs["factor1"], kwargs["factor2"]]]
            perm_dist = []
            perm_df = df.copy()
            for i in range(num_perm):
                perm_df[kwargs["factor1"]] = np.random.permutation(df[kwargs["factor1"]].values)
                perm_df[kwargs["factor2"]] = np.random.permutation(df[kwargs["factor2"]].values)
                perm_dist.append(stat_func(perm_df, **kwargs))
            real_stats: pd.DataFrame = stat_func(df, **kwargs)
            hist_names = list(real_stats.index)
            perm_dist = pd.concat(perm_dist, axis=1)
            num_rows = len(perm_dist)
            # fig, axes = plt.subplots(1, num_rows, figsize=(20, 6))
            ret_stats = []
            for i in range(num_rows):
                ret_stats.append(np.round(np.mean(perm_dist.iloc[i] < real_stats.iloc[i]), 5))
                # print(f"{hist_names[i]} permutation % smaller: {np.mean(perm_dist.iloc[i] < real_stats.iloc[i])}")
                # n, bins, _ = axes[i].hist(perm_dist.iloc[i], bins=50)
                # axes[i].set(title=hist_names[i], xlabel="Statistic", ylabel="Frequency")
                # axes[i].axvline(real_stats.iloc[i], 0, ymax=n.max() * 1.1, linestyle=":", color="red", linewidth=4)
            # fig.tight_layout()
            return np.array(ret_stats)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def one_way_anova_permutation(kwargs, stat_func, num_perm=2500):#
        try:
            # print("Calculating permutations... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["dependent_var"], kwargs["factor"]]]
            perm_dist = np.zeros(num_perm, dtype=float)
            perm_df = df.copy()
            for i in range(num_perm):
                perm_df[kwargs["factor"]] = np.random.permutation(df[kwargs["factor"]].values)
                perm_dist[i] = stat_func(perm_df, **kwargs)
            real_stat = stat_func(df, **kwargs)
            # print(f"permutation % smaller: {np.mean(perm_dist < real_stat)}")
            # n, bins, _ = plt.hist(perm_dist, bins=50)
            # plt.title("Permutation test for One way ANOVA F-statistic")
            # plt.xlabel("Statistic")
            # plt.ylabel("Frequency")
            # plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
            return np.round(np.mean(perm_dist < real_stat), 5)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def simple_bootstrap_rows(kwargs, stat_func, statistic, num_boot=5000):#
        try:
            print("Calculating bootstrap... This might take a while")
            df = kwargs["q"].get_dataset()
            boot_dist = np.zeros(num_boot, dtype=float)
            for i in tqdm(range(num_boot), desc="Bootstrapping..."):
                boot_dist[i] = stat_func(df.sample(frac=1, replace=True), **kwargs)
            n, bins, _ = plt.hist(boot_dist, bins=50)
            plt.title(f"Bootstrap distribution for {statistic}")
            plt.xlabel("Statistic")
            plt.ylabel("Frequency")
            real_stat = stat_func(df, **kwargs)
            plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def simple_bootstrap_rows_paired(kwargs, stat_func, statistic, num_boot=2500):#
        try:
            print("Calculating bootstrap... This might take a while")
            df = kwargs["q"].get_dataset()
            boot_dist = np.zeros(num_boot, dtype=float)
            subject_choices = np.random.choice(df["subject"], len(df["subject"]) * num_boot, True).reshape(
                (num_boot, len(df["subject"])))
            subjects = df['subject']
            final_idx_list = []
            for subject_choice in tqdm(subject_choices, desc="Preparing bootstrap samples..."):
                subj_num_to_idx_list = [df.index[subj == subjects].to_numpy() for subj in subject_choice]
                final_idx_list.append(np.concatenate(subj_num_to_idx_list))
            for i in tqdm(range(num_boot), desc="Bootstrapping..."):
                boot_dist[i] = stat_func(df.iloc[final_idx_list[i]], **kwargs)
            n, bins, _ = plt.hist(boot_dist, bins=50)
            plt.title(f"Bootstrap distribution for {statistic}")
            plt.xlabel("Statistic")
            plt.ylabel("Frequency")
            real_stat = stat_func(df, **kwargs)
            plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def single_col_bootstrap(kwargs, stat_func, statistic, num_boot=10000):
        try:
            print("Calculating bootstrap... This might take a while")
            vals = kwargs["q"].get_dataset()[kwargs["col1"]].to_numpy()
            boot_dist = np.zeros(num_boot, dtype=float)
            for i in tqdm(range(num_boot), desc="Bootstrapping..."):
                boot_dist[i] = stat_func(np.random.choice(vals, vals.size, replace=True))
            n, bins, _ = plt.hist(boot_dist, bins=50)
            plt.title(f"Bootstrap distribution for {statistic}")
            plt.xlabel("Statistic")
            plt.ylabel("Frequency")
            real_stat = stat_func(vals)
            plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def two_way_anova_bootstrap(kwargs, stat_func, statistic, num_boot=2500):
        try:
            print("Calculating bootstrap... This might take a while")
            df = kwargs["q"].get_dataset()
            factor1_levels = df[kwargs["factor1"]].unique()
            factor2_levels = df[kwargs["factor2"]].unique()
            boot_dist = []
            for i in tqdm(range(num_boot), desc="Bootstrapping..."):
                boot_df = []
                for fac1_val in factor1_levels:
                    for fac2_val in factor2_levels:
                        boot_df.append(df[(df[kwargs["factor1"]] == fac1_val).to_numpy() & (
                                df[kwargs["factor2"]] == fac2_val).to_numpy()].sample(frac=1, replace=True))
                boot_df = pd.concat(boot_df)
                boot_dist.append(stat_func(boot_df, **kwargs))
            real_stats: pd.DataFrame = stat_func(df, **kwargs)
            hist_names = list(real_stats.index)
            boot_dist = pd.concat(boot_dist, axis=1)
            num_rows = len(boot_dist)
            fig, axes = plt.subplots(1, num_rows, figsize=(20, 6))
            for i in range(num_rows):
                n, bins, _ = axes[i].hist(boot_dist.iloc[i], bins=50)
                axes[i].set(title=hist_names[i], xlabel="Statistic", ylabel="Frequency")
                axes[i].axvline(real_stats.iloc[i], 0, ymax=n.max() * 1.1, linestyle=":", color="red", linewidth=4)
            fig.tight_layout()
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")

    @staticmethod
    def rm_anova_bootstrap(kwargs):
        try:
            print("Calculating bootstrap... This might take a while")
            df = kwargs["q"].get_dataset()[[kwargs["subject"], kwargs["dependent_var"], kwargs["factor"]]]
            pivoted_df = df.pivot(kwargs["subject"], kwargs["factor"], kwargs["dependent_var"])  # make wide
            num_boot = 2500
            perm_dist = np.zeros(num_boot, dtype=float)
            for i in tqdm(range(num_boot), desc="Bootstrapping..."):
                perm_dist[i] = pg.rm_anova(pivoted_df.sample(frac=1, replace=True))["F"].values[0]
            n, bins, _ = plt.hist(perm_dist, bins=50)
            plt.title("Bootstrap for RM ANOVA F-statistic")
            plt.xlabel("Statistic")
            plt.ylabel("Frequency")
            real_stat = pg.rm_anova(pivoted_df)["F"].values[0]
            plt.axvline(real_stat, 0, ymax=n.max() * 1.1, linestyle=":", color="r", linewidth=4)
        except Exception:
            print("\033[91m" + f"An error has occured! You cannot use the selected tool as specified!" + "\033[0m")


class ProjectQuestion:
    OUTLIER_THRESH = 2.5  # outlier threshold in multiples of standard deviation
    INLIER_THRESH = 1.5  # inlier threshold in multiples of standard deviation
    DEF_NUM_ENTRIES = 60

    def __init__(self, group: int, q_type: str, q_params: tuple, extra_str: str, outliers: bool, idx):
        self._dataset_desc_dict = {
            "bfi": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20Bfi.pdf?forcedownload=1",
            "morality_and_cleanliness": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20morality%20and%20cleanliness.pdf?forcedownload=1",
            "morality_and_cleanliness_long": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20morality%20and%20cleanliness.pdf?forcedownload=1",
            "pokedex": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20pokedex.pdf?forcedownload=1",
            "siegel_etal_2017_both_studies": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20siegel_etal_2017.pdf?forcedownload=1",
            "siegel_wormwood_quigley_barrett_data1": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20siegel_etal_2017.pdf?forcedownload=1",
            "siegel_wormwood_quigley_barrett_data2": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20siegel_etal_2017.pdf?forcedownload=1",
            "siegel_etal_2017_both_studies_b": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20siegel_etal_2017.pdf?forcedownload=1",
            "smoke_ban": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%E2%80%8F%E2%80%8F%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20SmokeBan.pdf?forcedownload=1",
            "states": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20states.pdf?forcedownload=1",
            "unit_asking_replication_data": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20Unit-Asking.pdf?forcedownload=1",
            "unit_asking_replication_data_long": r"https://moodle2.cs.huji.ac.il/nu21/pluginfile.php/530645/mod_folder/content/0/%D7%AA%D7%99%D7%90%D7%95%D7%A8%20%D7%9E%D7%90%D7%92%D7%A8%20%D7%94%D7%A0%D7%AA%D7%95%D7%A0%D7%99%D7%9D%20Unit-Asking.pdf?forcedownload=1"}
        self._group = group
        self._idx = idx
        self._num_entries = ProjectQuestion.DEF_NUM_ENTRIES
        self._q_type = q_type
        self._full_dataset = Utils.get_dataset(q_params[0])
        self._dataset_name = q_params[0]
        self._vars = list(q_params[1:])
        self._extra_str = extra_str
        self._outliers = outliers
        self._dataset = None
        self._choose_dataset()
        self._filtered_dataset = None

    def get_dataset(self):
        return self._filtered_dataset if self._filtered_dataset else self._dataset

    # Deprecated by choice
    # def reject_outliers(self, columns, thresh):
    #     columns=list(columns)
    #     if sum([Utils.validate_var_name(self, col) for col in columns]) != len(columns):
    #         return None
    #     self._filtered_dataset = Utils.sd_reject(self.get_dataset(), columns, thresh)

    def _remove_filter(self):
        self._filtered_dataset = None

    def print_dataset_desc(self):
        set_font_style()
        print(
            f"The following research question concerns the '{self._dataset_name}' dataset.\nTo learn more about the data and the variables it describes, see {self._dataset_desc_dict[self._dataset_name]}.\n\nThe research question is:")

    def __repr__(self):
        self._seed()
        if self._q_type == 'Chi squared':
            possible_q_phrasing = [
                f"Is there any relationship between {self._vars[0]}{self._extra_str} and {self._vars[1]}{self._extra_str}?",
                f"Is the distribution of {self._vars[0]}{self._extra_str} different for different levels of {self._vars[1]}{self._extra_str}?",
                f"Does the value of {self._vars[0]}{self._extra_str} depend on {self._vars[1]}{self._extra_str}?"]

            return np.random.choice(possible_q_phrasing)
        elif self._q_type == 'ANOVA 2-way':

            possible_q_phrasing = [
                f"Does the level of {self._vars[1]}{self._extra_str} predict differences in average {self._vars[0]}{self._extra_str}?",
                f"Does the level of {self._vars[2]}{self._extra_str} predict differences in average {self._vars[0]}{self._extra_str}?",
                # f"Is the {self._vars[1]}{self._extra_str} effect on {self._vars[0]}{self._extra_str} modulated by {self._vars[2]}{self._extra_str}?",
                f"Is there any influence of {self._vars[1]}{self._extra_str} on {self._vars[0]}{self._extra_str}?",
                f"Is there any influence of {self._vars[2]}{self._extra_str} on {self._vars[0]}{self._extra_str}?",
                f"Does {self._vars[1]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str}?",
                f"Does {self._vars[2]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str}?",
                f"Does {self._vars[0]}{self._extra_str} change for different levels of {self._vars[1]}{self._extra_str}? Do {self._vars[1]} and {self._vars[2]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str}?",
                f"Does {self._vars[0]}{self._extra_str} change for different levels of {self._vars[2]}{self._extra_str}? Do {self._vars[1]} and {self._vars[2]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str}?",
                f"Does the effect of {self._vars[1]}{self._extra_str} on {self._vars[0]}{self._extra_str} changes for different levels of {self._vars[2]}{self._extra_str}?"]
            return np.random.choice(possible_q_phrasing)

        elif self._q_type == 'ANOVA 1-way':
            possible_q_phrasing = [
                f"Does the level of {self._vars[1]}{self._extra_str} predict differences in average {self._vars[0]}{self._extra_str}?"
                f"When the level of {self._vars[1]}{self._extra_str} changes, does {self._vars[0]}{self._extra_str} change as well?",
                f"Does the value of {self._vars[0]}{self._extra_str} depend on {self._vars[1]}{self._extra_str}?",
                f"Does {self._vars[1]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str}?",
                f"Does {self._vars[0]}{self._extra_str} change for different levels of {self._vars[1]}{self._extra_str}?"]
            return np.random.choice(possible_q_phrasing)
        elif self._q_type == 'Repeated Measures ANOVA':
            possible_q_phrasing = [
                f"Does the level of {self._vars[1]}{self._extra_str} predict differences in average {self._vars[0]}{self._extra_str}?",
                f"When the value of {self._vars[1]}{self._extra_str} changes, does {self._vars[0]}{self._extra_str} changes as well?",
                f"When measuring {self._vars[0]}{self._extra_str} for the same people under different levels of {self._vars[1]}{self._extra_str}, is there any difference?",
                f"Does {self._vars[1]}{self._extra_str} have an effect on {self._vars[0]}{self._extra_str} across subjects?"]
            return np.random.choice(possible_q_phrasing)

        elif self._q_type == 'Regression' or self._q_type == 'Pearson correlation':
            possible_q_phrasing = [
                f"When {self._vars[1]}{self._extra_str} is increased by 1, can we expect a change in {self._vars[0]}{self._extra_str}?",
                f"Can a linear model predict {self._vars[0]}{self._extra_str} by {self._vars[1]}{self._extra_str}?"]
            return np.random.choice(possible_q_phrasing)

        elif self._q_type == "Spearman correlation":
            possible_q_phrasing = [
                f"Is there any relationship between {self._vars[1]}{self._extra_str} and {self._vars[0]}{self._extra_str}?",
                f"Is there a monotonous relationship between {self._vars[1]}{self._extra_str} and {self._vars[0]}{self._extra_str}?"]
            return np.random.choice(possible_q_phrasing)
        elif self._q_type == 'paired t-test':
            unique_y = self.get_dataset()[self._vars[1]].unique()
            unique_choices = np.random.choice(unique_y, 2, False)
            direction = np.random.choice(['larger', 'smaller', 'different'])
            if len(unique_y) == 2 and direction == "different":
                possible_q_phrasing = [
                    f"Does DV change across subjects between {self._vars[1]}={unique_choices[0]} and {self._vars[0]}={unique_choices[1]}?",
                    f"Does {self._vars[1]} have an effect on {self._vars[0]}{self._extra_str} across subjects?"]
            else:
                possible_q_phrasing = [
                    f"Is {self._vars[0]}{self._extra_str} {direction} for {self._vars[1]}={unique_choices[0]} compared to {self._vars[1]}={unique_choices[1]} across subjects?"]
            return np.random.choice(possible_q_phrasing)
        elif self._q_type == 'independent samples t-test' or self._q_type == "Mann Whitney":
            unique_y = self.get_dataset()[self._vars[1]].unique()
            unique_choices = np.random.choice(unique_y, 2, False)
            direction = np.random.choice(['larger', 'smaller', 'different'])
            if len(unique_y) == 2 and direction == "different":
                possible_q_phrasing = [
                    f"Does {self._vars[1]} have an effect on {self._vars[0]}{self._extra_str}?",
                    f"Does {self._vars[0]}{self._extra_str} change for different levels of {self._vars[1]}?",
                    f"Does {self._vars[0]}{self._extra_str} change between {self._vars[1]}={unique_choices[0]} and {self._vars[1]}={unique_choices[1]}?"]
            else:
                possible_q_phrasing = [
                    f"Is {self._vars[0]}{self._extra_str} {direction} for {self._vars[1]}={unique_choices[0]} compared to {self._vars[1]}={unique_choices[1]}?"]
            return np.random.choice(possible_q_phrasing)
        else:
            return "No question available for this dataset!"

    def is_same(self, q_params):
        """
        :return: True if q_params will generate the same question as self, False otherwise
        """
        return self._vars == list(q_params[0][1:])

    def _choose_dataset(self):
        relevant_columns_dataset = self._full_dataset[
            self._vars] if "subject" not in self._full_dataset.columns else \
            self._full_dataset[["subject"] + self._vars]
        if "subject" in relevant_columns_dataset.columns:
            nan_rows = relevant_columns_dataset.isnull().any(axis=1)
            nan_subjects = relevant_columns_dataset["subject"][nan_rows]
            relevant_columns_dataset = relevant_columns_dataset[~relevant_columns_dataset["subject"].isin(nan_subjects)]
        else:
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
            numeric_df = relevant_columns_dataset.select_dtypes(include=np.number).copy()
            cols = numeric_df.columns
            for col in cols:
                if numeric_df[col].unique().size < 10:
                    numeric_df = numeric_df.drop(col, axis=1)

            outlier_rows = \
                np.where((np.abs(stats.zscore(numeric_df)) > ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
            inlier_rows_idx = np.ones_like(relevant_columns_dataset.index).astype(bool)
            inlier_rows_idx[outlier_rows] = 0
            inlier_rows = np.where(
                (np.abs(stats.zscore(numeric_df.loc[inlier_rows_idx,])) < ProjectQuestion.INLIER_THRESH).all(axis=1))[
                0]
            if self._num_entries > outlier_rows.size + inlier_rows.size:
                self._num_entries = inlier_rows.size
            self._seed()
            max_outliers = min([outlier_rows.size, 8])
            num_outliers = np.random.randint(1, max_outliers) if self._outliers and max_outliers > 1 else 0
            num_inliers = self._num_entries - num_outliers
            # get a boolean vector the size of the data for inliers and outliers
            outlier_rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            inlier_rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            if self._q_type == "Regression" or self._q_type == "Spearman correlation":
                inlier_rows_idx[np.random.choice(inlier_rows, num_inliers, replace=False)] = 1
            else:  # t-test, anova 1/2 way - need to make sure all levels are present
                if self._q_type == "Repeated Measures ANOVA" or "subject" in relevant_columns_dataset.columns:
                    self._outliers = False
                    num_outliers = 0
                    num_inliers = self._num_entries if relevant_columns_dataset[
                                                           'subject'].unique().size > self._num_entries else \
                        relevant_columns_dataset['subject'].unique().size
                    subjects = np.random.choice(relevant_columns_dataset['subject'].unique(), num_inliers, False)
                    inlier_rows_idx = relevant_columns_dataset['subject'].isin(subjects).to_numpy()
                elif "2-way" not in self._q_type:
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
                            if val_rows.size == 0:
                                self._dataset = None
                                return
                            inlier_rows_idx[np.random.choice(val_rows, min(num_inliers_per_val_comb, val_rows.size),
                                                             replace=False)] = 1
            # subset the dataset
            inlier_dataset = numeric_df.loc[inlier_rows_idx]
            if self._outliers and "subject" not in relevant_columns_dataset.columns:
                mean, std = inlier_dataset.mean(), inlier_dataset.std()
                positive_outlier_rows = \
                    np.where((numeric_df > mean + std * ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
                negative_outlier_rows = \
                    np.where((numeric_df < mean - std * ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
                if positive_outlier_rows.size > negative_outlier_rows.size:
                    outliers = np.random.choice(positive_outlier_rows, min(num_outliers, positive_outlier_rows.size),
                                                replace=False)
                else:
                    outliers = np.random.choice(negative_outlier_rows, min(num_outliers, negative_outlier_rows.size),
                                                replace=False)
                outlier_rows_idx[outliers] = True
                relevant_columns_dataset[self._vars[0]][outlier_rows_idx] *= \
                    np.array(1000 * mean, dtype=relevant_columns_dataset[self._vars[0]].dtype)[0]

                if num_outliers == 0:
                    self._outliers = False

            self._dataset = relevant_columns_dataset.loc[outlier_rows_idx | inlier_rows_idx, :]
        else:
            rows_idx = np.zeros_like(relevant_columns_dataset.index).astype(bool)
            rows_idx[np.random.choice(np.arange(relevant_columns_dataset.index.size), size=self._num_entries,
                                      replace=False)] = True
            self._dataset = relevant_columns_dataset.loc[rows_idx, :]
            self._outliers = False
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
    NUM_QUESTIONS = 5
    NUM_NO_REPEAT = 3
    NUM_REPEAT = 2

    def __init__(self, group: int, id: str):
        self._group = group
        self._id = id
        self._possible_questions, self._questions_extra_str = Utils.get_questions()
        self._seed()
        self._question_types = np.concatenate([np.random.choice(list(self._possible_questions.keys()), replace=False,
                                                                size=Project.NUM_NO_REPEAT),
                                               np.random.choice(list(self._possible_questions.keys()), replace=True,
                                                                size=Project.NUM_REPEAT)])
        self._extra_var = None
        self._extra_var = [bool(np.random.binomial(1, 0.5, 1)) if qt == 'ANOVA 1-way' else False for qt in
                           self._question_types]
        self._questions = []
        self._generate_questions()

    def _generate_questions(self):
        self._seed()
        self._outliers = np.random.binomial(1, 0.5, len(self._question_types)).astype(bool)
        for i, key in enumerate(self._question_types):
            q = None
            q_param_choices = set()
            while q is None:
                if key == 'Anova 1-way':
                    if self._extra_var[i]:
                        q_options = [(q, self._questions_extra_str[key][i]) for i, q in
                                     enumerate(self._possible_questions[key])
                                     if len(q) == 4]
                    else:
                        q_options = [(q, self._questions_extra_str[key][i]) for i, q in
                                     enumerate(self._possible_questions[key])
                                     if len(q) == 3]
                else:
                    q_options = list(zip(self._possible_questions[key], self._questions_extra_str[key]))
                q_param_choice = np.random.choice(np.arange(len(q_options)))
                while q_param_choice in q_param_choices:
                    q_param_choice = np.random.choice(np.arange(len(q_options)))
                q_param_choices.add(q_param_choice)
                q_params = q_options[q_param_choice]
                if Project.NUM_REPEAT > 0:
                    break_flag = False
                    while sum([q.is_same(q_params) for q in self._questions]) > 0:
                        if len(q_options) == 1:
                            break_flag = True
                            break
                        q_params = q_options[np.random.choice(np.arange(len(q_options)))]
                    if break_flag:
                        self._question_types[i] = np.random.choice(list(self._possible_questions.keys()))
                        key = self._question_types[i]
                        q = None
                        continue
                q = ProjectQuestion(self._group, key, q_params[0], q_params[1], self._outliers[i], i + 1)
                if q._dataset is not None:
                    self._questions.append(q)
                else:
                    q = None

    def _seed(self) -> None:
        np.random.seed(self._group)

    def __getitem__(self, idx):
        if idx >= Project.NUM_QUESTIONS:
            raise IndexError(f"There are only {Project.NUM_QUESTIONS} questions in this part!"
                             f"Please provide a subscript between 0-{Project.NUM_QUESTIONS - 1}, and not {idx}!")
        return self._questions[idx]

    def __len__(self):
        return len(self._questions)


class GUI:
    """
    This class generates a GUI for a question - providing different plots, statistical tests etc. via
    manual_interact with Run button for choosing the parameters (if plot - which type? what are the x,y,color values?
    If test, between which rows? and so on)
    On "Run" press, will call the corresponding Utils function with the chosen options. The Utils function should
    validate the chosen options and print an error message if there is a problem with the options.
    """

    def __init__(self, q):
        self._test_type_widg = None
        self._widget_func_dict = {"t-test between columns": self._get_column_t_test_widgets,
                                  "t-test by level": self._get_column_by_group_t_test_widgets,
                                  "One way ANOVA": self._get_one_way_anova_widgets,
                                  "Two way ANOVA": self._get_two_way_anova_widgets,
                                  "Repeated Measures ANOVA": self._get_rm_anova_widgets,
                                  "Mann-Whitney test between columns": self._get_column_mann_widgets,
                                  "Mann-Whitney test by level": self._get_column_by_group_mann_widgets,
                                  "Linear regression": self._get_correlation_widgets,
                                  "Spearman correlation": self._get_correlation_widgets,
                                  "Pearson correlation": self._get_correlation_widgets,
                                  "Chi squared for independence": self._get_var1_var2_widgets,
                                  "mean": self._get_mean_bootstrap_wd}
        self.q = q
        self.funcs_dict = {"Choose a function": lambda: None,
                           "Show dataset": self.show_data,
                           "Scatter plot 2 columns": self.scatter,
                           "Scatter plot of variable by levels": self.scatter_by_values,
                           "Histogram": self.hist,
                           "Histogram of variable by levels": self.hist_by_values,
                           "Bootstrap": self.bootstrap,
                           "Permutation test": self.permutation_test,
                           "t-test between columns": self.column_t_test,
                           "t-test by level": self.column_by_group_t_test,
                           "One way ANOVA": self.one_way_anova,
                           "Two way ANOVA": self.two_way_anova,
                           "Repeated Measures ANOVA": self.rm_anova,
                           "Mann-Whitney test between columns": self.column_mann_whitney_test,
                           "Mann-Whitney test by level": self.mann_whitney_test,
                           "Linear regression": self.regress,
                           "Spearman correlation": self.spearman_correlation,
                           "Pearson correlation": self.pearson_correlation,
                           "Chi squared for independence": self.chi_for_indep,
                           "Cohen's d between columns": self.cohens_d_columns,
                           "Cohen's d by level": self.cohens_d,
                           "Eta squared for one-way ANOVA": self.eta_squared_for_oneway,
                           "Eta squared for two-way ANOVA": self.eta_squared_for_twoway,
                           "Eta squared for Repeated Measures ANOVA": self.eta_squared_for_rm,
                           "r squared for pearson correlation": self.r_squared_pearson_or_regression,
                           "r squared for linear regression": self.r_squared_pearson_or_regression,
                           "r squared for spearman correlation": self.r_squared_spearman,
                           "Bonferroni multiple-comparisons significance threshold": self.multiple_hyp_thresh_bon,
                           "Benjamini-Hochberg multiple-comparisons significance threshold": self.multiple_hyp_thresh_fdr, }
        self._statistic_fun_dict = {"t-test between columns": lambda df, **kwargs: (
            stats.ttest_rel if kwargs["method"] == "paired" else stats.ttest_ind)(df[kwargs["col1"]],
                                                                                  df[kwargs["col2"]],
                                                                                  alternative=kwargs["alternative"])[0],
                                    "t-test by level": lambda df, **kwargs: (
                                        stats.ttest_rel if kwargs["method"] == "paired" else stats.ttest_ind)(
                                        df[kwargs["col1"]][df[kwargs["col2"]] == kwargs["col2_val1"]],
                                        df[kwargs["col1"]][df[kwargs["col2"]] == kwargs["col2_val2"]],
                                        alternative=kwargs["alternative"])[0],
                                    "One way ANOVA": lambda df, **kwargs: anova_lm(
                                        ols(f'{kwargs["dependent_var"]}~C({kwargs["factor"]})', data=df).fit())[
                                        "F"].values[0],
                                    "Two way ANOVA": lambda df, **kwargs: anova_lm(
                                        ols(f'{kwargs["dependent_var"]}~C({kwargs["factor1"]})*C({kwargs["factor2"]})',
                                            data=df).fit())["F"].iloc[[0, 1, 2, ]],
                                    "Repeated Measures ANOVA": lambda df, **kwargs:
                                    AnovaRM(df, f'{kwargs["dependent_var"]}', f'{kwargs["subject"]}',
                                            [f'{kwargs["factor"]}']).fit().anova_table["F Value"],
                                    "Mann-Whitney test between columns": lambda df, **kwargs: stats.mannwhitneyu(
                                        df[kwargs["col1"]],
                                        df[kwargs["col2"]],
                                        alternative=kwargs["alternative"])[0],
                                    "Mann-Whitney test by level": lambda df, **kwargs: stats.mannwhitneyu(
                                        df[kwargs["col1"]][df[kwargs["col2"]] == kwargs["col2_val1"]],
                                        df[kwargs["col1"]][df[kwargs["col2"]] == kwargs["col2_val2"]],
                                        alternative=kwargs["alternative"])[0],
                                    "Linear regression": lambda df, **kwargs:
                                    stats.linregress(df[kwargs["col1"]], df[kwargs["col2"]])[0],
                                    "Spearman correlation": lambda df, **kwargs:
                                    stats.spearmanr(df[kwargs["col1"]], df[kwargs["col2"]])[0],
                                    "Pearson correlation": lambda df, **kwargs:
                                    stats.pearsonr(df[kwargs["col1"]], df[kwargs["col2"]])[0],
                                    "Chi squared for independence": lambda df, **kwargs:
                                    stats.chi2_contingency(pd.crosstab(df[kwargs["col1"]], df[kwargs["col2"]]))[0],
                                    "mean": np.mean}

    def get_interact(self, func_str):
        # data_table.disable_dataframe_formatter()
        ipd.display(ipd.HTML("""
      <style>
      #output-body {
          display: flex;
          align-items: center;
          justify-content: center;
      }
      </style>
      """))
        set_font_style()
        self.funcs_dict[func_str]()

    def show_data(self):
        # data_table.enable_dataframe_formatter()
        ipd.display(self.q.get_dataset())

    # return a dropdown widget for all possible columns
    def get_columns_dropdown(self, desc) -> wd.Dropdown:
        return wd.Dropdown(
            options=self.q.get_dataset().columns,
            value=None,
            description=desc,
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def get_column_selection(self, desc):
        return wd.SelectMultiple(
            options=self.q.get_dataset().columns,
            value=[],
            # rows=10,
            description=desc,
            disabled=False
        )

    # return a dropdown widget for all possible levels of a column
    def get_values_dropdown(self, widget, desc) -> wd.Dropdown:
        new_widget = wd.Dropdown(
            options=[],
            value=None,
            description=desc,
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

        def on_change(change, changed_widg=widget, widg=new_widget, df=self.q.get_dataset()):
            widg.options = df[change['new']].unique()

        widget.observe(on_change, names='value')
        return new_widget

    def get_stats_dropdown(self):
        return wd.Dropdown(
            options=["mean", "t-test between columns", "t-test by level", "One way ANOVA", "Two way ANOVA",
                     "Repeated Measures ANOVA", "Mann-Whitney test between columns", "Mann-Whitney test by level",
                     "Linear regression", "Spearman correlation", "Pearson correlation",
                     "Chi squared for independence"],
            value=None,
            description="Statistic:",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def get_alternative_dropdown(self):
        return wd.Dropdown(
            options=[("two-sided", "two-sided"), ("1 is greater than 2", "greater"), ("1 is smaller than 2", "less")],
            value=None,
            description="Hypothesis direction:",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def get_ttest_method_dropdown(self):
        return wd.Dropdown(
            options=["independent samples", "paired"],
            value=None,
            description="t-test type:",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def scatter(self):
        inter = wd.interact_manual(Utils.scatter, df=wd.fixed(self.q.get_dataset()),
                                   x=self.get_columns_dropdown("X:"),
                                   y=self.get_columns_dropdown("Y:"),
                                   color=self.get_columns_dropdown("Column to base color on"))
        inter.widget.children[3].description = "Run"

    def hist(self):
        inter = wd.interact_manual(Utils.hist, df=wd.fixed(self.q.get_dataset()),
                                   x=self.get_columns_dropdown("Column to calculate histogram:"))
        inter.widget.children[1].description = "Run"

    def hist_by_values(self):
        inter = wd.interact_manual(Utils.multihist, df=wd.fixed(self.q.get_dataset()),
                                   val_col=self.get_columns_dropdown("Column to calculate histogram for:"),
                                   hue_col=self.get_columns_dropdown("Column containing the levels to split by:"))
        inter.widget.children[2].description = "Run"

    def scatter_by_values(self):
        level_col = self.get_columns_dropdown("Column containing the levels to split by:")
        inter = wd.interact_manual(Utils.scatter_by_level, q=wd.fixed(self.q),
                                   val_col=self.get_columns_dropdown("Column containing values for scatter:"),
                                   level_col=level_col,
                                   level1=self.get_values_dropdown(level_col, "Level for X axis:"),
                                   level2=self.get_values_dropdown(level_col, "Level for Y axis:"))
        inter.widget.children[4].description = "Run"

    def _get_column_t_test_widgets(self):
        return {"q": wd.fixed(self.q), "col1": self.get_columns_dropdown("Group 1"),
                "col2": self.get_columns_dropdown("Group 2"),
                "method": self.get_ttest_method_dropdown(),
                "alternative": self.get_alternative_dropdown()}

    def _get_mean_bootstrap_wd(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Column:")}

    def _get_column_by_group_t_test_widgets(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Dependent variable column:"),
                "col2": col2_wd,
                "col2_val1": self.get_values_dropdown(col2_wd, "Value for group 1:"),
                "col2_val2": self.get_values_dropdown(col2_wd, "Value for group 2:"),
                "method": self.get_ttest_method_dropdown(),
                "alternative": self.get_alternative_dropdown()}

    def _get_one_way_anova_widgets(self, eta=False):
        return {"q": wd.fixed(self.q),
                "dependent_var": self.get_columns_dropdown("Dependent variable column:"),
                "factor": self.get_columns_dropdown("Independent variable:"),
                "eta": wd.fixed(eta)}

    def _get_two_way_anova_widgets(self, eta=False):
        return {"q": wd.fixed(self.q),
                "dependent_var": self.get_columns_dropdown("Dependent variable column:"),
                "factor1": self.get_columns_dropdown("Independent variable 1:"),
                "factor2": self.get_columns_dropdown("Independent variable 2:"),
                "eta": wd.fixed(eta)}

    def _get_rm_anova_widgets(self, eta=False):
        return {"q": wd.fixed(self.q),
                "dependent_var": self.get_columns_dropdown("Dependent variable column:"),
                "subject": self.get_columns_dropdown("Subject column:"),
                "factor": self.get_columns_dropdown("Independent variable:"),
                "eta": wd.fixed(eta)}

    def column_t_test(self):
        inter = wd.interact_manual(Utils.column_t_test, **self._get_column_t_test_widgets())
        inter.widget.children[4].description = "Run"

    def column_by_group_t_test(self):
        inter = wd.interact_manual(Utils.column_by_group_t_test, **self._get_column_by_group_t_test_widgets())
        inter.widget.children[6].description = "Run"

    def one_way_anova(self):
        inter = wd.interact_manual(Utils.one_way_anova, **self._get_one_way_anova_widgets())
        inter.widget.children[2].description = "Run"

    def two_way_anova(self):
        inter = wd.interact_manual(Utils.two_way_anova, **self._get_two_way_anova_widgets())
        inter.widget.children[3].description = "Run"

    def rm_anova(self):
        inter = wd.interact_manual(Utils.rm_anova, **self._get_rm_anova_widgets())
        inter.widget.children[3].description = "Run"

    def _get_column_mann_widgets(self):
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Group 1"),
                "col2": self.get_columns_dropdown("Group 2"),
                "alternative": self.get_alternative_dropdown()}

    def _get_column_by_group_mann_widgets(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Dependent variable column:"),
                "col2": col2_wd,
                "col2_val1": self.get_values_dropdown(col2_wd, "Value for group 1:"),
                "col2_val2": self.get_values_dropdown(col2_wd, "Value for group 2:"),
                "alternative": self.get_alternative_dropdown()}

    def mann_whitney_test(self):
        inter = wd.interact_manual(Utils.mann_whitney_test, **self._get_column_by_group_mann_widgets())
        inter.widget.children[5].description = "Run"

    def column_mann_whitney_test(self):
        inter = wd.interact_manual(Utils.column_mann_whitney_test, **self._get_column_mann_widgets())
        inter.widget.children[3].description = "Run"

    def _get_correlation_widgets(self):
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Dependent variable column:"),
                "col2": self.get_columns_dropdown("Independent variable column:")}

    def _get_var1_var2_widgets(self):
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Variable 1 column:"),
                "col2": self.get_columns_dropdown("Variable 2 column:")}

    def regress(self):
        inter = wd.interact_manual(Utils.regress, **self._get_correlation_widgets())
        inter.widget.children[2].description = "Run"

    def spearman_correlation(self):
        inter = wd.interact_manual(Utils.spearman_correlation, **self._get_correlation_widgets())
        inter.widget.children[2].description = "Run"

    def pearson_correlation(self):
        inter = wd.interact_manual(Utils.pearson_correlation, **self._get_correlation_widgets())
        inter.widget.children[2].description = "Run"

    def chi_for_indep(self):
        inter = wd.interact_manual(Utils.chi_for_indep, **self._get_var1_var2_widgets())
        inter.widget.children[2].description = "Run"

    def get_group_by_level_widgets(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Dependent variable column:"),
                "col2": col2_wd,
                "col2_val1": self.get_values_dropdown(col2_wd, "Value for group 1:"),
                "col2_val2": self.get_values_dropdown(col2_wd, "Value for group 2:")}

    def get_dep_indep_widgets(self):
        return {"q": wd.fixed(self.q),
                "col1": self.get_columns_dropdown("Group 1"),
                "col2": self.get_columns_dropdown("Group 2")}

    def cohens_d(self):
        inter = wd.interact_manual(Utils.cohens_d, **self.get_group_by_level_widgets())
        inter.widget.children[4].description = "Run"

    def cohens_d_columns(self):
        inter = wd.interact_manual(Utils.cohens_d_columns, **self.get_dep_indep_widgets())
        inter.widget.children[2].description = "Run"

    def eta_squared_for_oneway(self):
        inter = wd.interact_manual(Utils.one_way_anova, **self._get_one_way_anova_widgets(True))
        inter.widget.children[3].description = "Run"

    def eta_squared_for_twoway(self):
        inter = wd.interact_manual(Utils.two_way_anova, **self._get_two_way_anova_widgets(True))
        inter.widget.children[3].description = "Run"

    def eta_squared_for_rm(self):
        inter = wd.interact_manual(Utils.rm_anova, **self._get_rm_anova_widgets(True))
        inter.widget.children[3].description = "Run"

    def r_squared_pearson_or_regression(self):
        inter = wd.interact_manual(Utils.r_squared_pearson_or_regression, **self._get_var1_var2_widgets())
        inter.widget.children[2].description = "Run"

    def r_squared_spearman(self):
        inter = wd.interact_manual(Utils.r_squared_spearman, **self._get_var1_var2_widgets())
        inter.widget.children[2].description = "Run"

    def multiple_hyp_thresh_perm(self):
        pass

    def multiple_hyp_thresh_bon(self):
        inter = wd.interact_manual(Utils.multiple_hyp_thresh_bon,
                                   alpha=wd.FloatText(description=r"alpha", layout=wd.Layout(min_width='50%'),
                                                      style={'description_width': 'initial'}),
                                   number_of_comparisons=wd.FloatText(description=r"Number of comparisons",
                                                                      layout=wd.Layout(min_width='50%'),
                                                                      style={'description_width': 'initial'}))
        inter.widget.children[2].description = "Run"

    def multiple_hyp_thresh_fdr(self):
        inter = wd.interact_manual(Utils.multiple_hyp_thresh_fdr,
                                   alpha=wd.FloatText(description=r"alpha", layout=wd.Layout(min_width='50%'),
                                                      style={'description_width': 'initial'}),
                                   p_values_list=wd.Text(description=r"P values (separated by commas):",
                                                         layout=wd.Layout(min_width='50%'),
                                                         style={'description_width': 'initial'}))
        inter.widget.children[2].description = "Run"

    def _run_bootstrap(self, test, **kwargs):
        kwargs["test"] = test
        stat_func = self._statistic_fun_dict[test]
        if test == "t-test between columns":
            if kwargs["method"] == "paired":
                Utils.simple_bootstrap_rows_paired(kwargs, stat_func, "t-statistic")
            else:
                Utils.simple_bootstrap_rows(kwargs, stat_func, "t-statistic")
        elif test == "mean":
            Utils.single_col_bootstrap(kwargs, stat_func, "Mean")
        elif test == "t-test by level":
            if kwargs["method"] == "paired":
                Utils.simple_bootstrap_rows_paired(kwargs, stat_func, "t-statistic")
            else:
                Utils.simple_bootstrap_rows(kwargs, stat_func, "t-statistic")
        elif test == "Mann-Whitney test between columns":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "U-statistic")
        elif test == "Mann-Whitney test by level":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "U-statistic")
        elif test == "Linear regression":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "Linear-Regression slope")
        elif test == "Spearman correlation":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "Spearman r")
        elif test == "Pearson correlation":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "Pearson r")
        elif test == "One way ANOVA":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "One way F")
        elif test == "Two way ANOVA":
            Utils.two_way_anova_bootstrap(kwargs, stat_func, "Two way F")
        elif test == "Repeated Measures ANOVA":
            Utils.rm_anova_bootstrap(kwargs)
        elif test == "Chi squared for independence":
            Utils.simple_bootstrap_rows(kwargs, stat_func, "Chi squared for independence")

    def _run_permutation(self, test, **kwargs):
        stat_func = self._statistic_fun_dict[test]
        res = None
        if test == "t-test between columns":
            res = Utils.between_col_permutation(kwargs, stat_func, "t-statistic")
        elif test == "t-test by level":
            res = Utils.by_level_permutation(kwargs, stat_func, "t-statistic")
        elif test == "Mann-Whitney test between columns":
            res = Utils.between_col_permutation(kwargs, stat_func, "U-statistic")
        elif test == "Mann-Whitney test by level":
            res = Utils.by_level_permutation(kwargs, stat_func, "U-statistic")
        elif test == "Linear regression":
            res = Utils.regression_permutation(kwargs, stat_func, "Linear-Regression slope")
        elif test == "Spearman correlation":
            res = Utils.regression_permutation(kwargs, stat_func, "Spearman r")
        elif test == "Pearson correlation":
            res = Utils.regression_permutation(kwargs, stat_func, "Pearson r")
        elif test == "One way ANOVA":
            res = Utils.one_way_anova_permutation(kwargs, stat_func)
        elif test == "Two way ANOVA":
            res = Utils.two_way_anova_permutation(kwargs, stat_func)
        elif test == "Repeated Measures ANOVA":
            res = Utils.rm_anova_permutation(kwargs, stat_func)
        elif test == "Chi squared for independence":
            res = Utils.chisq_permutation(10000, kwargs, stat_func)
        return res

    def show_permutation_test_interact(self, change):
        ipd.clear_output()
        set_font_style()
        ipd.display(self._test_type_widg)
        test_type = change['new']
        wd.interact_manual(self._run_permutation, test=wd.fixed(test_type), **self._widget_func_dict[test_type]())

    def permutation_test(self):
        test_options = ["t-test between columns", "t-test by level",
                        "One way ANOVA", "Two way ANOVA", "Repeated Measures ANOVA",
                        "Mann-Whitney test between columns", "Mann-Whitney test by level", "Linear regression",
                        "Spearman correlation", "Pearson correlation", "Chi squared for independence"]
        self._test_type_widg = wd.Dropdown(
            options=test_options,
            value=None,
            description="Statistic for permutation: ",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )
        self._test_type_widg.observe(self.show_permutation_test_interact, names="value")
        ipd.display(self._test_type_widg)

    def show_bootstrap_test_interact(self, change):
        ipd.clear_output()
        set_font_style()
        ipd.display(self._test_type_widg)
        test_type = change['new']
        wd.interact_manual(self._run_bootstrap, test=wd.fixed(test_type), **self._widget_func_dict[test_type]())

    def bootstrap(self):
        test_options = ["mean", "t-test between columns", "t-test by level",
                        "One way ANOVA", "Two way ANOVA", "Repeated Measures ANOVA",
                        "Mann-Whitney test between columns", "Mann-Whitney test by level", "Linear regression",
                        "Spearman correlation", "Pearson correlation", "Chi squared for independence"]
        self._test_type_widg = wd.Dropdown(
            options=test_options,
            value=None,
            description="Statistic for bootstrap: ",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )
        self._test_type_widg.observe(self.show_bootstrap_test_interact, names="value")
        ipd.display(self._test_type_widg)


class QuestionReport:
    NUM_METHODS = 10

    def __init__(self, q):
        self._q = q
        self._gui = GUI(q)
        self._possible_methods = ["", "Show dataset", "Scatter plot 2 columns",
                                  "Scatter plot of variable by levels", "Histogram",
                                  "Histogram of variable by levels", "Bootstrap", "Permutation test",
                                  "t-test between columns", "t-test by level", "One way ANOVA",
                                  "Two way ANOVA", "Repeated Measures ANOVA", "Mann-Whitney test between columns",
                                  "Mann-Whitney test by level", "Linear regression",
                                  "Spearman correlation", "Pearson correlation", "Chi squared for independence",
                                  "Cohen's d between columns", "Cohen's d by level",
                                  "Eta squared for one-way ANOVA", "Eta squared for two-way ANOVA",
                                  "Eta squared for Repeated Measures ANOVA", "r squared for pearson correlation",
                                  "r squared for linear regression", "r squared for spearman correlation",
                                  "Bonferroni multiple-comparisons significance threshold",
                                  "Benjamini-Hochberg multiple-comparisons significance threshold"]
        self._possible_tests = ["", "Permutation test",
                                "t-test between columns", "t-test by level", "One way ANOVA",
                                "Two way ANOVA", "Repeated Measures ANOVA", "Mann-Whitney test between columns",
                                "Mann-Whitney test by level", "Linear regression",
                                "Spearman correlation", "Pearson correlation", "Chi squared for independence"]
        self._methods_report_widgets = [self._get_dropdown_widget(self._possible_methods, "Choose the method: ") for _
                                        in range(QuestionReport.NUM_METHODS)]
        self._test_report_widget = self._get_dropdown_widget(self._possible_tests, "Test: ")
        self._test_report_widget.observe(self._q_ans_report_specific_test_widget_show, names='value')
        self._test_param_report_widgets = None
        self._perm_test_report_widget = self._get_dropdown_widget(self._possible_tests, "Test: ")
        self._perm_test_report_widget.observe(self._show_test_report_widgets, names='value')
        self._free_text_widget = wd.Textarea(description="",
                                             layout=wd.Layout(min_width='50%', height='100px'),
                                             style={'description_width': 'initial'})

    def _get_dropdown_widget(self, options, desc):
        return wd.Dropdown(
            options=options,
            value=None,
            description=desc,
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def show_methods_report(self):
        set_font_style()
        print("The question was:")
        print("\x1B[3m" + self._q.__repr__() + "\x1B[0m")
        print(
            "\nWhich methods did you use to reach your conclusions?\nPlease write them in the order in which they should be used to help reach the conclusion")
        for widg in self._methods_report_widgets:
            ipd.display(widg)
        inter = wd.interact_manual(self._save_methods_ans, layout=wd.Layout(min_width='50%'),
                                   style={'description_width': 'initial'})
        inter.widget.children[0].description = "Save methods"

    def _get_wd(self, widget_class, desc):
        return widget_class(value=None,
                            description=desc,
                            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'})

    def _q_ans_report_specific_test_widget_show(self, change):
        self._test_param_report_widgets = []
        self._perm_test_report_widget.value = None
        ipd.clear_output()
        set_font_style()
        print("The question was:")
        print("\x1B[3m" + self._q.__repr__() + "\x1B[0m")
        ipd.display(self._test_report_widget)
        if change['new'] == "Permutation test":
            ipd.display(self._perm_test_report_widget)
        else:
            self._show_test_report_widgets(change)

    def _show_test_report_widgets(self, change):
        test = change['new']
        if test == "t-test between columns":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_column_t_test_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "t-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Effect size: "))

        elif test == "t-test by level":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_column_by_group_t_test_widgets().values() if
                 not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "t-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Effect size: "))


        elif test == "One way ANOVA":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_one_way_anova_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "F-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Effect size: "))

        elif test == "Two way ANOVA":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_two_way_anova_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 1 F-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 1 p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 1 Effect size: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 2 F-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 2 p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Independent variable 2 Effect size: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Interaction F-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Interaction p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Interaction Effect size: "))

        elif test == "Repeated Measures ANOVA":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_rm_anova_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._gui.get_columns_dropdown("Dependent variable column:"))
            self._test_param_report_widgets.append(self._gui.get_columns_dropdown("Independent variable:"))
            self._test_param_report_widgets.append(self._gui.get_columns_dropdown("Subject column:"))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "F-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Effect size: "))

        elif test == "Mann-Whitney test between columns":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_column_mann_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "U-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))

        elif test == "Mann-Whitney test by level":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_column_by_group_mann_widgets().values() if
                 not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "U-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))

        elif test == "Linear regression":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_correlation_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Slope: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Intercept: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "r^2: "))

        elif test == "Spearman correlation" or test == "Pearson correlation":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_correlation_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "r: "))

        elif test == "Chi squared for independence":
            self._test_param_report_widgets.extend(
                [widg for widg in self._gui._get_var1_var2_widgets().values() if not isinstance(widg, wd.fixed)])
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "Chi squared: "))
            self._test_param_report_widgets.append(self._get_wd(wd.FloatText, "p-value: "))

        for widg in self._test_param_report_widgets:
            ipd.display(widg)

        inter = wd.interact_manual(self._save_q_ans, layout=wd.Layout(min_width='50%'),
                                   style={'description_width': 'initial'})
        inter.widget.children[0].description = "Save question"

    def show_is_conclusion_valid(self):
        set_font_style()
        print("Are the results above valid?")
        self._valid_conclusion_wd = wd.Dropdown(options=[("Yes", True), ("No", False)],
                                                value=None,
                                                description="",
                                                layout=wd.Layout(min_width='50%'),
                                                style={'description_width': 'initial'})
        ipd.display(self._valid_conclusion_wd)
        print("\nWhich tools did you use to determine this?")
        data = {k: k for k in self._possible_methods}
        names = []
        checkbox_objects = []
        for key in data:
            checkbox_objects.append(wd.Checkbox(value=False, description=key,
                                                layout=wd.Layout(min_width='50%'),
                                                style={'description_width': 'initial'}))
            names.append(key)

        arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}

        ui = wd.VBox(children=checkbox_objects)

        self._selected_validation_methods = []

        def select_data(**kwargs):
            self._selected_validation_methods.clear()

            for key in kwargs:
                if kwargs[key] is True:
                    self._selected_validation_methods.append(key)
            print(f"Selected methods: {self._selected_validation_methods}")

        out = wd.interactive_output(select_data, arg_dict)
        ipd.display(ui, out)
        inter = wd.interact_manual(self._save_valid_ans)
        inter.widget.children[0].description = "Save"

    def show_question_answers_report(self):
        set_font_style()
        print("The question was:")
        print("\x1B[3m" + self._q.__repr__() + "\x1B[0m")
        print(
            "\nWhich test did you use to answer the question?\nAfter choosing the test, please fill the options that appear")
        self._perm_test_report_widget.value = None
        ipd.display(self._test_report_widget)

    def _save_valid_ans(self):
        with open(f"Q{self._q._idx}_valid_conclusion", "w") as ans_file:
            print("You chose: " + ("valid" if self._valid_conclusion_wd.value else "invalid") + " conclusion.")
            print("To determine this, you used:")
            ans_file.write(f"valid_conclusion,{self._valid_conclusion_wd.value}\n")
            for ans in self._selected_validation_methods:
                ans_file.write(f"used method,{ans}\n")
                print(ans)

    def _save_q_ans(self):
        with open(f"Q{self._q._idx}_question", "w") as ans_file:
            set_font_style()
            print("You answered:")
            if self._perm_test_report_widget.value:
                ans_file.write(f"test,{self._test_report_widget.value}\n")
                ans_file.write(f"test,{self._perm_test_report_widget.value}\n")
                print("Chosen test:", self._test_report_widget.value)
                print("Test in permutation:", self._perm_test_report_widget.value)
            else:
                ans_file.write(f"test,{self._test_report_widget.value}\n")
                print("Chosen test:", self._test_report_widget.value)
            for i, widg in enumerate(self._test_param_report_widgets):
                if widg.value is not None:
                    no_paren_desc = widg.description[:widg.description.find(":")]
                    save_str = f"{no_paren_desc}, {widg.value}\n"
                    ans_file.write(save_str)
                    print(save_str.replace(',', ": "), end="")
        with open(f"Q{self._q._idx}_object", "wb") as q_file:
            pickle.dump(self._q, q_file)

    def _save_methods_ans(self):
        with open(f"Q{self._q._idx}_methods", "w") as ans_file:
            set_font_style()
            print("\nYou answered:")
            for i, widg in enumerate(self._methods_report_widgets):
                if widg.value is not None:
                    save_str = f"Method {i + 1}, {widg.value}\n"
                    ans_file.write(save_str)
                    print(save_str.replace(',', ": "), end="")
        with open(f"Q{self._q._idx}_object", "wb") as q_file:
            pickle.dump(self._q, q_file)

    def show_free_text(self):
        set_font_style()
        print("You may describe your thought process here:")

        ipd.display(self._free_text_widget)
        inter = wd.interact_manual(self._save_free_text, layout=wd.Layout(min_width='25%'),
                                   style={'description_width': 'initial'})
        inter.widget.children[0].description = "Save question"

    def _save_free_text(self):
        with open(f"Q{self._q._idx}_freetxt", "w") as ans_file:
            ans_file.write(self._free_text_widget.value)
        print("\n\nYou entered:\n")
        print(self._free_text_widget.value)

    def show_answers(self):
        set_font_style()
        print("************")
        print("Methods:")
        print("************")
        for i, widg in enumerate(self._methods_report_widgets):
            if widg.value is not None:
                save_str = f"Method {i + 1}, {widg.value}\n"
                print(save_str.replace(',', ": "), end="")
        print("\n************")
        print("Test:")
        print("************")
        if self._perm_test_report_widget.value:
            print("Chosen test:", self._test_report_widget.value)
            print("Test in permutation:", self._perm_test_report_widget.value)
        else:
            print("Chosen test:", self._test_report_widget.value)
        for i, widg in enumerate(self._test_param_report_widgets):
            if widg.value is not None:
                no_paren_desc = widg.description[:widg.description.find(":")]
                save_str = f"{no_paren_desc}, {widg.value}\n"
                print(save_str.replace(',', ": "), end="")
        print("\n************")
        print("Conclusion validity:")
        print("************")
        print("You chose: " + ("valid" if self._valid_conclusion_wd.value else "invalid") + " conclusion.")
        print("To determine this, you used:")
        for ans in self._selected_validation_methods:
            print(ans)

        print("\n************")
        print("Thought process:")
        print("************")
        print(self._free_text_widget.value)
