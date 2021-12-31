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
# from google.colab import data_table

sns.set(rc={'figure.figsize':(10,10),"font.size":20, "axes.labelsize":20,
    "axes.titlesize":25, "xtick.labelsize":12,
    "ytick.labelsize":12, "legend.fontsize":15,
    "lines.markersize":10})
_=plt.rcParams["figure.figsize"] = (10,10)


# @title Personal details
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
        return None

    # regular plot df[x],df[y]
    @staticmethod
    def plot(df: pd.DataFrame, x, y):
        plt.plot(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)

    @staticmethod
    def hist(df, x):
        bins = np.linspace(0.9 * df[x].min(), 1.1 * df[x].max(), 50)
        plt.hist(df[x], bins=bins)
        plt.title(f"{x} histogram")
        plt.xlabel(x)
        plt.ylabel("Frequencies")

    # perform bootstrap on df rows, calculating the statistic_func on the bootstrapped df and plotting the histogram
    @staticmethod
    def bootstrap(df: pd.DataFrame, statistic_func):
        boot_idx = np.random.choice(df.index, (10000, df.index.size))
        boot_dist = np.zeros(10000, dtype=float)
        for i in range(boot_dist.size):
            boot_dist[i] = statistic_func(pd.DataFrame(df.values[boot_idx[i]], columns=df.columns))
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
        elif method == "independent samples":
            return lambda df, col1=col1, col2=col2, col2_val1=col2_val1, col2_val2=col2_val2, alternative=alternative: \
                stats.ttest_ind(df.loc[df[col2] == col2_val1, col1], df.loc[df[col2] == col2_val2, col1],
                                alternative=alternative)
        else:
            set_font_style(color="red")
            print(f"t-test can only be for paired/independent!", file=sys.stderr)

    @staticmethod
    def get_t_test_func_for_subset(q, dep, col1, val1, col2, val2, col3, val3, col4, val4, method, alternative):
        if not (Utils.validate_var_name(q, dep) and
                Utils.validate_var_name(q, col1) and
                Utils.validate_var_name(q, col2) and
                Utils.validate_var_name(q, col3) and
                Utils.validate_var_name(q, col4) and
                Utils.validate_var_value(q, col1, val1) and
                Utils.validate_var_value(q, col2, val2) and
                Utils.validate_var_value(q, col3, val3) and
                Utils.validate_var_value(q, col4, val4)):
            return lambda df: None
        if method == "paired":
            return lambda df, dep=dep, col1=col1, val1=val1, col2=col2, val2=val2, col3=col3, val3=val3, col4=col4,
                          val4=val4, \
                          alternative=alternative: stats.ttest_rel(df.loc[(df[col1] == val1).to_numpy() &
                                                                          (df[col2] == val2).to_numpy(), dep],
                                                                   df.loc[(df[col3] == val3).to_numpy() &
                                                                          (df[col4] == val4).to_numpy(), dep],
                                                                   alternative=alternative)
        elif method == "independent samples":
            return lambda df, dep=dep, col1=col1, val1=val1, col2=col2, val2=val2, col3=col3, val3=val3, col4=col4,
                          val4=val4, \
                          alternative=alternative: stats.ttest_ind(df.loc[(df[col1] == val1).to_numpy() &
                                                                          (df[col2] == val2).to_numpy(), dep],
                                                                   df.loc[(df[col3] == val3).to_numpy() &
                                                                          (df[col4] == val4).to_numpy(), dep],
                                                                   alternative=alternative)
        else:
            set_font_style(color="red")
            print(f"t-test can only be for paired/independent!")

    @staticmethod
    def column_t_test(q, col1, col2, method, alternative):
        if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
            return None
        if method == "independent samples":
            res = stats.ttest_ind(q.get_dataset()[col1], q.get_dataset()[col2], alternative=alternative)
        else:
            res = stats.ttest_rel(q.get_dataset()[col1], q.get_dataset()[col2], alternative=alternative)
        if res: print(f"t-statistic: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def column_by_group_t_test(q, col1, col2, col2_val1, col2_val2, method, alternative):
        res = Utils.get_t_test_func(q, col1, col2, col2_val1, col2_val2, method, alternative)(q.get_dataset())
        if res: print(f"t-statistic: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def t_test_for_specific_levels(q, dep, col1, val1, col2, val2, col3, val3, col4, val4, method, alternative):
        res = Utils.get_t_test_func_for_subset(q, dep, col1, val1, col2, val2, col3, val3, col4, val4, method,
                                               alternative)(
            q.get_dataset())
        if res: print(f"t-statistic: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def get_one_way_anova_model(q, dependent_var, factor):
        return ols(f"{dependent_var}~C({factor})", data=q.get_dataset()).fit()

    @staticmethod
    def get_two_way_anova_model(q, dependent_var, factor1, factor2):
        return ols(f"{dependent_var}~C({factor1})*C({factor2})", data=q.get_dataset()).fit()

    @staticmethod
    def get_rm_anova_model(q, dependent_var, subject, factor1):
        return AnovaRM(q.get_dataset(), f'{dependent_var}', f'{subject}', [f'{factor1}']).fit()

    @staticmethod
    def one_way_anova(q, dependent_var, factor, eta):
        aov = anova_lm(Utils.get_one_way_anova_model(q, dependent_var, factor))
        if eta:
            Utils.eta_squared(aov)
        else:
            ipd.display(aov)

    @staticmethod
    def two_way_anova(q, dependent_var, factor1, factor2, eta):
        aov = anova_lm(Utils.get_two_way_anova_model(q, dependent_var, factor1, factor2))
        if eta:
            Utils.eta_squared(aov)
        else:
            ipd.display(aov)

    @staticmethod
    def rm_anova(q, dependent_var, subject, factor1, eta):
        aov = Utils.get_rm_anova_model(q, dependent_var, subject, factor1).anova_table
        if eta:  # need to calculate alone
            res = pg.rm_anova(dv=dependent_var, subject=subject, within=factor1, data=q.get_dataset(), detailed=True)
            ss = res['SS'].to_numpy()
            eta_sq = (ss[:-1] / ss.sum())[0]
            ipd.display(ipd.Latex(r"$\eta^2 = %.5f$" % eta_sq))
        else:
            ipd.display(aov)

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
    def get_column_mann_whitney_func(q, col1, col2):
        if not (Utils.validate_var_name(q, col1)
                and Utils.validate_var_name(q, col2)):
            return lambda df: None
        return lambda df, col1=col1, col2=col2: \
            stats.mannwhitneyu(df[col1], df[col2])

    @staticmethod
    def mann_whitney_test(q, col1, col2, col2_val1, col2_val2):
        res = Utils.get_mann_whitney_func(q, col1, col2, col2_val1, col2_val2)(q.get_dataset())
        if res: print(f"U-statistic: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def column_mann_whitney_test(q, col1, col2):
        res = Utils.get_column_mann_whitney_func(q, col1, col2)(q.get_dataset())
        if res: print(f"U-statistic: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def print_regression_equation(dep, indep, slope, intercept, rvalue, pvalue, residuals):
        from decimal import getcontext
        getcontext().prec = 5
        slope, intercept, rvalue, pvalue = list(map(lambda v: np.round(v, 5), [slope, intercept, rvalue ** 2, pvalue]))
        mat_str = "$" + dep + f" = {slope} * " + indep + (" +" if intercept >= 0 else " ") + f"{intercept}" + "$"
        mat2_str = "$r^2=" + f"{rvalue}, p={np.format_float_scientific(pvalue, precision=3)}$"
        ipd.display(ipd.Latex(mat_str))
        ipd.display(ipd.Latex(mat2_str))

    @staticmethod
    def get_regress_func(q, dep_var, indep_var):
        if not (Utils.validate_var_name(q, dep_var) and Utils.validate_var_name(q, indep_var)):
            return None
        return lambda df, dep_var=dep_var, indep_var=indep_var: stats.linregress(df[dep_var], df[indep_var])

    @staticmethod
    def regress(q, dep_var, indep_var):
        res = Utils.get_regress_func(q, dep_var, indep_var)(q.get_dataset())
        if res is not None:
            Utils.print_regression_equation(dep_var, indep_var, *res)

    @staticmethod
    def get_spearman_correlation_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var=var1, var2=var2: stats.spearmanr(df[var1], df[var2])

    @staticmethod
    def spearman_correlation(q, var1, var2):
        res = Utils.get_spearman_correlation_func(q, var1, var2)(q.get_dataset())
        if res: print(f"Spearman correlation: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def get_pearson_correlation_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var=var1, var2=var2: stats.pearsonr(df[var1], df[var2])

    @staticmethod
    def pearson_correlation(q, var1, var2):
        res = Utils.get_pearson_correlation_func(q, var1, var2)(q.get_dataset())
        if res: print(f"Pearson correlation: {res[0]}\np-value: {np.format_float_scientific(res[1], precision=3)}")

    @staticmethod
    def get_chi_for_indep_func(q, var1, var2):
        if not (Utils.validate_var_name(q, var1) and Utils.validate_var_name(q, var2)):
            return lambda df: None
        return lambda df, var1=var1, var2=var2: stats.chi2_contingency(pd.crosstab(df[var1], df[var2]))[:2]

    @staticmethod
    def chi_for_indep(q, var1, var2):
        print(Utils.get_chi_for_indep_func(q, var1, var2)(q.get_dataset()))

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
        d = (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
        print(f"Cohen's d of {col1} between {col2}={col2_val1} and {col2}={col2_val2}: {d}")

    @staticmethod
    def cohens_d_columns(q, col1, col2):
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
        print(f"Cohen's d between {col1} and {col2}: {d}")

    @staticmethod
    def eta_squared(aov_res):
        ipd.display(pd.DataFrame((aov_res["sum_sq"] / aov_res["sum_sq"].sum())[:-1].rename("eta^2")))

    @staticmethod
    def r_squared_pearson_or_regression(q, var1, var2):
        res = Utils.get_pearson_correlation_func(q, var1, var2)(q.get_dataset())
        if res is None:
            return None
        print(f"r^2 = {res[0] ** 2:.5g}")

    @staticmethod
    def r_squared_spearman(q, var1, var2):
        res = Utils.get_spearman_correlation_func(q, var1, var2)(q.get_dataset(), var1, var2)
        if res is None:
            return None
        print(f"r^2 = {res[0] ** 2:.5g}")

    # TODO: implement
    @staticmethod
    def multiple_hyp_thresh_perm():
        pass

    @staticmethod
    def multiple_hyp_thresh_bon(alpha, number_of_comparisons):
        print(
            f"Significant p-value threshold: {np.format_float_scientific(alpha / number_of_comparisons, precision=3)}")

    @staticmethod
    def multiple_hyp_thresh_fdr(p_values_list, alpha):
        p_values_list = [float(p) for p in re.split("[, ]", p_values_list) if p]
        p_values_list = np.array(p_values_list)
        m = p_values_list.size
        sorting_idx = np.argsort(p_values_list)[::-1]
        for i, idx in enumerate(sorting_idx):
            if p_values_list[idx] < (i + 1) * (alpha / m):
                print(
                    f"Significant p-value threshold: {np.format_float_scientific((i + 1) * (alpha / m), precision=3)}")
                return None

    @staticmethod
    def validate_var_name(q, col1):
        if col1 not in q.get_dataset().columns:
            set_font_style(color='red')
            print(f"{col1} isn't a variable in the dataset!\n possible variables: {q.get_dataset().columns}")
            return False
        return True

    @staticmethod
    def validate_var_value(q, col, val):
        if val not in q.get_dataset()[col].unique():
            set_font_style(color='red')
            print(
                f"{val} isn't a value of in the dataset in column {col}!\n possible values: {q.get_dataset()[col].unique()}")
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

    def __init__(self, q):
        self.q = q
        self.funcs_dict = {"Scatter plot 2 columns": self.scatter,
                           "Show dataset": self.show_data,
                           "Plot": self.plot, "Histogram": self.hist,
                           "Bootstrap": self.bootstrap,
                           "t-test between columns": self.column_t_test,
                           "t-test by level": self.column_by_group_t_test,
                           "Simple effect t-test": self.simple_effect_t_test,
                           "One way ANOVA": self.one_way_anova,
                           "Two way ANOVA": self.two_way_anova,
                           "Repeated Measures ANOVA": self.rm_anova,
                           "Mann-Whitney test between columns": self.column_mann_whitney_test,
                           "Mann-Whitney test by level": self.mann_whitney_test,
                           "Linear regression": self.regress,
                           "Spearman correlation": self.spearman_correlation,
                           "Pearson correlation": self.pearson_correlation,
                           "Cohen's d between columns": self.cohens_d_columns,
                           "Cohen's d by level": self.cohens_d,
                           "Eta squared for one-way ANOVA": self.eta_squared_for_oneway,
                           "Eta squared for two-way ANOVA": self.eta_squared_for_twoway,
                           "Eta squared for Repeated Measures ANOVA": self.eta_squared_for_rm,
                           "r squared for pearson correlation": self.r_squared_pearson_or_regression,
                           "r squared for linear regression": self.r_squared_pearson_or_regression,
                           "r squared for spearman correlation": self.r_squared_spearman,
                           "Bonferroni multiple-comparisons significance threshold": self.multiple_hyp_thresh_bon,
                           "Benjamini-Hochberg multiple-comparisons significance threshold": self.multiple_hyp_thresh_fdr}

    def get_interact(self, func_str):
        data_table.disable_dataframe_formatter()
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
        data_table.enable_dataframe_formatter()
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
            if change['new'] in changed_widg.options:
                widg.options = df[change['new']].unique()

        widget.observe(on_change)
        return new_widget

    def get_stats_dropdown(self):
        return wd.Dropdown(
            options=["mean", "median", "standard-deviation"],
            value=None,
            description="Statistic:",
            layout=wd.Layout(min_width='50%'), style={'description_width': 'initial'},
        )

    def get_alternative_dropdown(self):
        return wd.Dropdown(
            options=["two-sided", "greater", "less"],
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

    def plot(self):
        inter = wd.interact_manual(Utils.plot, df=wd.fixed(self.q.get_dataset()),
                                   x=self.get_columns_dropdown("X:"),
                                   y=self.get_columns_dropdown("Y:"))
        inter.widget.children[2].description = "Run"

    def hist(self):
        inter = wd.interact_manual(Utils.hist, df=wd.fixed(self.q.get_dataset()),
                                   x=self.get_columns_dropdown("Column to calculate histogram:"))
        inter.widget.children[1].description = "Run"

    @staticmethod
    def bootstrap_helper(df, col, statistic_func):
        f = None
        if statistic_func == "mean":
            f = np.mean
        elif statistic_func == "median":
            f = np.median
        elif statistic_func == "standard-deviation":
            f = np.std
        _ = Utils.bootstrap(df, lambda df, col=col, f=f: f(df[col]))
        plt.xlabel(f"{col} {statistic_func}")
        plt.ylabel("Frequency")
        plt.title(f"Bootstrap of {col} {statistic_func}")

    def bootstrap(self):
        wd.interact_manual(GUI.bootstrap_helper, df=wd.fixed(self.q.get_dataset()),
                           col=self.get_columns_dropdown("Column to calculate bootstrap on:"),
                           statistic_func=self.get_stats_dropdown())

    def permutation(self, q: pd.DataFrame, to_permute, statistic_func):

        pass

    def permutation_test(self, q: pd.DataFrame, to_permute, statistic_func):
        pass

    def sd_reject(self):
        inter = wd.interact_manual(self.q.reject_outliers,
                                   columns=self.get_column_selection("Columns to reject by:"),
                                   thresh=wd.FloatText(value=0, description="SD threshold:"))
        inter.widget.children[2].description = "Run"

    def column_t_test(self):
        inter = wd.interact_manual(Utils.column_t_test, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Group 1"),
                                   col2=self.get_columns_dropdown("Group 2"),
                                   method=self.get_ttest_method_dropdown(),
                                   alternative=self.get_alternative_dropdown())
        inter.widget.children[4].description = "Run"

    def column_by_group_t_test(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        inter = wd.interact_manual(Utils.column_by_group_t_test, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Dependent variable column:"),
                                   col2=col2_wd,
                                   col2_val1=self.get_values_dropdown(col2_wd, "Value for group 1:"),
                                   col2_val2=self.get_values_dropdown(col2_wd, "Value for group 2:"),
                                   method=self.get_ttest_method_dropdown(),
                                   alternative=self.get_alternative_dropdown())
        inter.widget.children[6].description = "Run"

    def simple_effect_t_test(self):

        col1_wd = self.get_columns_dropdown("Column for first factor of cell 1:")
        col2_wd = self.get_columns_dropdown("Column for second factor of cell 1:")
        col3_wd = self.get_columns_dropdown("Column for first factor of cell 2:")
        col4_wd = self.get_columns_dropdown("Column for second factor of cell 2:")

        inter = wd.interact_manual(Utils.t_test_for_specific_levels, q=wd.fixed(self.q),
                                   dep=self.get_columns_dropdown("Dependent variable:"),
                                   col1=col1_wd,
                                   val1=self.get_values_dropdown(col1_wd, "Level of first factor of cell 1"),
                                   col2=col2_wd,
                                   val2=self.get_values_dropdown(col2_wd, "Level of second factor of cell 1"),
                                   col3=col3_wd,
                                   val3=self.get_values_dropdown(col3_wd, "Level of first factor of cell 2"),
                                   col4=col4_wd,
                                   val4=self.get_values_dropdown(col4_wd, "Level of second factor of cell 2"),
                                   method=self.get_ttest_method_dropdown(),
                                   alternative=self.get_alternative_dropdown())
        inter.widget.children[11].description = "Run"

    def one_way_anova(self):
        inter = wd.interact_manual(Utils.one_way_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   factor=self.get_columns_dropdown("Independent variable:"), eta=wd.fixed(False))
        inter.widget.children[2].description = "Run"

    def two_way_anova(self):
        inter = wd.interact_manual(Utils.two_way_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   factor1=self.get_columns_dropdown("Independent variable 1:"),
                                   factor2=self.get_columns_dropdown("Independent variable 2:"), eta=wd.fixed(False))
        inter.widget.children[3].description = "Run"

    def rm_anova(self):
        inter = wd.interact_manual(Utils.rm_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   subject=self.get_columns_dropdown("Subject column:"),
                                   factor1=self.get_columns_dropdown("Independent variable:"), eta=wd.fixed(False))
        inter.widget.children[3].description = "Run"

    def mann_whitney_test(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        inter = wd.interact_manual(Utils.mann_whitney_test, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Dependent variable column:"),
                                   col2=col2_wd,
                                   col2_val1=self.get_values_dropdown(col2_wd, "Value for group 1:"),
                                   col2_val2=self.get_values_dropdown(col2_wd, "Value for group 2:"))
        inter.widget.children[4].description = "Run"

    def column_mann_whitney_test(self):
        inter = wd.interact_manual(Utils.column_mann_whitney_test, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Group 1"),
                                   col2=self.get_columns_dropdown("Group 2"))
        inter.widget.children[2].description = "Run"

    def regress(self):
        inter = wd.interact_manual(Utils.regress, q=wd.fixed(self.q),
                                   dep_var=self.get_columns_dropdown("Dependent variable column:"),
                                   indep_var=self.get_columns_dropdown("Independent variable:"))
        inter.widget.children[2].description = "Run"

    def spearman_correlation(self):
        inter = wd.interact_manual(Utils.spearman_correlation, q=wd.fixed(self.q),
                                   var1=self.get_columns_dropdown("Variable 1 column:"),
                                   var2=self.get_columns_dropdown("Variable 2 column:"))
        inter.widget.children[2].description = "Run"

    def pearson_correlation(self):
        inter = wd.interact_manual(Utils.pearson_correlation, q=wd.fixed(self.q),
                                   var1=self.get_columns_dropdown("Variable 1 column:"),
                                   var2=self.get_columns_dropdown("Variable 2 column:"))
        inter.widget.children[2].description = "Run"

    def chi_for_indep(self):
        inter = wd.interact_manual(Utils.pearson_correlation, q=wd.fixed(self.q),
                                   var1=self.get_columns_dropdown("Variable 1 column:"),
                                   var2=self.get_columns_dropdown("Variable 2 column:"))
        inter.widget.children[2].description = "Run"

    def cohens_d(self):
        col2_wd = self.get_columns_dropdown("Column to divide by:")
        inter = wd.interact_manual(Utils.cohens_d, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Dependent variable column:"),
                                   col2=col2_wd,
                                   col2_val1=self.get_values_dropdown(col2_wd, "Value for group 1:"),
                                   col2_val2=self.get_values_dropdown(col2_wd, "Value for group 2:"))
        inter.widget.children[4].description = "Run"

    def cohens_d_columns(self):
        inter = wd.interact_manual(Utils.cohens_d_columns, q=wd.fixed(self.q),
                                   col1=self.get_columns_dropdown("Group 1"),
                                   col2=self.get_columns_dropdown("Group 2"))
        inter.widget.children[2].description = "Run"

    def eta_squared_for_oneway(self):
        inter = wd.interact_manual(Utils.one_way_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   factor=self.get_columns_dropdown("Independent variable:"), eta=wd.fixed(True))
        inter.widget.children[3].description = "Run"

    def eta_squared_for_twoway(self):
        inter = wd.interact_manual(Utils.two_way_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   factor1=self.get_columns_dropdown("Independent variable 1:"),
                                   factor2=self.get_columns_dropdown("Independent variable 2:"), eta=wd.fixed(True))
        inter.widget.children[3].description = "Run"

    def eta_squared_for_rm(self):
        inter = wd.interact_manual(Utils.rm_anova, q=wd.fixed(self.q),
                                   dependent_var=self.get_columns_dropdown("Dependent variable column:"),
                                   subject=self.get_columns_dropdown("Subject column:"),
                                   factor1=self.get_columns_dropdown("Independent variable:"), eta=wd.fixed(True))
        inter.widget.children[3].description = "Run"

    def r_squared_pearson_or_regression(self):
        inter = wd.interact_manual(Utils.r_squared_pearson_or_regression, q=wd.fixed(self.q),
                                   var1=self.get_columns_dropdown("Variable 1 column:"),
                                   var2=self.get_columns_dropdown("Variable 2 column:"))
        inter.widget.children[2].description = "Run"

    def r_squared_spearman(self):
        inter = wd.interact_manual(Utils.r_squared_spearman, q=wd.fixed(self.q),
                                   var1=self.get_columns_dropdown("Variable 1 column:"),
                                   var2=self.get_columns_dropdown("Variable 2 column:"))
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

    def distance_of_means(self, q, col1, col2):
        pass

    def distance_of_medians(self, q, col1, col2):
        pass


class ProjectQuestion:
    OUTLIER_THRESH = 4  # outlier threshold in multiples of standard deviation
    INLIER_THRESH = 1  # inlier threshold in multiples of standard deviation
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

    # Deprecated by choice
    # def reject_outliers(self, columns, thresh):
    #     columns=list(columns)
    #     if sum([Utils.validate_var_name(self, col) for col in columns]) != len(columns):
    #         return None
    #     self._filtered_dataset = Utils.sd_reject(self.get_dataset(), columns, thresh)

    def _remove_filter(self):
        self._filtered_dataset = None

    def __repr__(self):
        if self._q_type == 'Chi squared':
            ret_str = f"Are %s{self._extra_str} and %s{self._extra_str} independent?"
            return ret_str % tuple(self._vars)
        elif self._q_type == 'ANOVA 2-way':
            ret_str = f"Do %s{self._extra_str} or %s{self._extra_str} have an effect on %s{self._extra_str}? " \
                      f"Does the effect of %s{self._extra_str} on %s{self._extra_str} change for different levels of %s " \
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
            inlier_dataset = numeric_df.loc[inlier_rows_idx,]
            mean, std = inlier_dataset.mean(), inlier_dataset.std()
            positive_outlier_rows = \
                np.where((numeric_df > mean + std * ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
            negative_outlier_rows = \
                np.where((numeric_df < mean - std * ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
            if positive_outlier_rows.size > negative_outlier_rows.size:
                outlier_rows_idx[np.random.choice(positive_outlier_rows, min(num_outliers, positive_outlier_rows.size),
                                                  replace=False)] = 1
            else:
                outlier_rows_idx[np.random.choice(negative_outlier_rows, min(num_outliers, negative_outlier_rows.size),
                                                  replace=False)] = 1
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
    REPEAT_Q_TYPES = True

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


group = 111  # @param {type:"number"}
id = "315594044"  # @param {type:"string"}
#%%
questions, extra = Utils.get_questions()
key = 'ANOVA 2-way'
q_params = questions[key][0]
ProjectQuestion(1, key, q_params, extra[key][0], True, 1)
# %% tests
q_list = []
qs = dict()
for key, value in tqdm(questions.items()):
    count = 1
    qs[key] = []
    for i, q_params in enumerate(value):
        q = ProjectQuestion(1, key, q_params, extra[key][i], True, count)
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
