import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
import IPython.display as ipd
import ipywidgets as wd
import re
import sys
from tqdm import tqdm
from functools import lru_cache


# TODO: Block that has the different options for utility functions. Running the block with a specific choice should
# TODO: generate a widget with the required values/call the correct utility.
class Utils:
    DATA_PATH = r"https://raw.githubusercontent.com/odedwer/stats_final_project_datasets/main/"

    @staticmethod
    @lru_cache(maxsize=None)
    def get_dataset(dataset_name) -> pd.DataFrame:
        """
        :return: pandas DataFrame that is the dataset with the given dataset_name.
        """
        return pd.read_csv(Utils.DATA_PATH + dataset_name + ".csv")

    @staticmethod
    def get_questions() -> dict:
        """
        Reads the data tagging CSV and parses it into a dictionary whose keys are different statistical tests
        and values are list of question parameters
        :return: questions parameters dictionary
        """
        # get from github
        questions = pd.read_csv(Utils.DATA_PATH + r"stat_final_project_data_tagging.csv")
        test_variables = dict()
        # start parsing
        for test in questions["test"].unique():
            if test not in test_variables.keys():
                test_variables[test] = []
            # subset just what we need
            test_df = questions.loc[questions["test"] == test, :]
            possible_combs = []
            # for each row generate all possible questions
            for idx, row in test_df.iterrows():
                dependent_vars = [s for s in re.split("; *", str(row["dependent_var"])) if s and s != 'nan']
                independent_vars1 = [s for s in re.split("; *", str(row["independent_var1"])) if s and s != 'nan']
                independent_vars2 = [s for s in re.split("; *", str(row["independent_var2"])) if s and s != 'nan']
                extra_vars = [s for s in re.split("; *", str(row["extra_independent_vars"])) if s and s != 'nan']
                if dependent_vars:  # not chi-squared
                    if independent_vars1 and independent_vars2:  # 2-way anova
                        possible_combs += [(str(row["data"]), dep, indep1, indep2) for dep in dependent_vars for indep1
                                           in independent_vars1 for indep2 in
                                           independent_vars2 if dep != indep1 and dep != indep2 and indep1 != indep2]
                        if extra_vars:
                            print(
                                f"Error! Two way anova with an extra independent variable, "
                                f"meaning 3-way anova required! Row {idx}", file=sys.stderr)
                    else:
                        if extra_vars:  # one way anova framing that should be performed as anova 2-way
                            if 'ANOVA 1-way' not in test_variables.keys():
                                test_variables['ANOVA 1-way'] = []
                            test_variables['ANOVA 1-way'] += [(str(row["data"]), dep, indep1, extra) for dep in
                                                              dependent_vars for indep1 in
                                                              independent_vars1 for extra in extra_vars if
                                                              dep != indep1 and dep != extra and indep1 != extra]
                        else:  # any other test
                            possible_combs += [(str(row["data"]), dep, indep1) for dep in dependent_vars for indep1 in
                                               independent_vars1 if dep != indep1]

                else:  # chi squared
                    possible_combs += [(str(row["data"]), indep1, indep2) for indep1 in independent_vars1 for indep2 in
                                       independent_vars2 if indep1 != indep2]
                    if extra_vars:
                        print(
                            f"Error! It doesn't make sense to have an "
                            f"extra variable for a 2-variable chi-square test! Row {idx}", file=sys.stderr)
            test_variables[test] += possible_combs[:]
        return test_variables

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

    # reject entries based on SD threshold. Print how many were rejected and return the df without the rejected points
    @staticmethod
    def sd_reject(df: pd.DataFrame, reject_columns, thresh):
        numeric_df: pd.DataFrame = df[reject_columns].select_dtypes(np.number)
        outliers = (np.abs(stats.zscore(numeric_df)) < thresh).any(axis=1)
        print(f"{np.sum(outliers)} rejected based on rejection criteria of {thresh} standard deviations")
        return df.loc[~outliers,]

    # perform permutation test fo the statistic func where the permuted label is to_permute.
    @staticmethod
    def permutation_test(df: pd.DataFrame, to_permute, statistic_func):
        perm_dist, statistic = Utils.permutation(df, to_permute, statistic_func)
        print(f"Percentage of permutations below the statistic: f{np.mean(perm_dist < statistic)}")

    @staticmethod
    def get_t_test_func(q, col1, col2, method, alternative):
        if not (Utils.validate_var_name(q, col1) and Utils.validate_var_name(q, col2)):
            return None
        if method == "paired":
            return lambda df, col1=col1, col2=col2, alternative=alternative: stats.ttest_rel(df[col1], df[col2],
                                                                                             alternative=alternative)
        elif method == "independent":
            return lambda df, col1=col1, col2=col2, alternative=alternative: stats.ttest_ind(df[col1], df[col2],
                                                                                             alternative=alternative)
        else:
            print(f"t-test can only be for paired/independent!", file=sys.stderr)

    @staticmethod
    def validate_var_name(q, col1):
        if col1 not in q._dataset.columns:
            print(f"{col1} isn't a variable in the dataset!\n possible variables: {q._dataset.columns}",
                  file=sys.stderr)
            # return None

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


class ProjectQuestion:
    OUTLIER_THRESH = 2.5  # outlier threshold in multiples of standard deviation
    INLIER_THRESH = 1.5  # outlier threshold in multiples of standard deviation
    DEF_NUM_ENTRIES = 100

    def __init__(self, group: int, q_type: str, q_params: tuple, outliers: bool):
        self._group = group
        self._num_entries = ProjectQuestion.DEF_NUM_ENTRIES
        self._q_type = q_type
        self._full_dataset = Utils.get_dataset(q_params[0])
        self._vars = list(q_params[1:])
        self._outliers = outliers
        self._dataset = None
        self._choose_dataset()
        self._filtered_dataset = None

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
            return "Is %s and %s independent?" % tuple(self._vars)
        elif self._q_type == 'ANOVA 2-way':
            return "Does %s and %s have an effect on %s?" % (self._vars[1], self._vars[2], self._vars[0])
        elif self._q_type == 'ANOVA 1-way' or self._q_type == 'Repeated Measures ANOVA':
            return "Does %s have an effect on %s?" % (self._vars[1], self._vars[0])
        elif self._q_type == 'Regression':
            return "What is the relationship between %s and %s and is it significant?" % tuple(self._vars)
        elif self._q_type == 'independent samples t-test' or self._q_type == 'paired t-test':
            self._seed()
            return "Is %s %s than %s?" % (
                self._vars[0], np.random.choice(['larger', 'smaller', 'different']), self._vars[1])
        elif self._q_type == 'Mann Whitney':
            return "Do %s distributions differ based on %s?" % tuple(self._vars)

    def is_same(self, q_params):
        """
        :return: True if q_params will generate the same question as self, False otherwise
        """
        if self._vars == q_params[1:]:
            return True
        return False

    def _choose_dataset(self):
        relevant_columns_dataset = self._full_dataset[self._vars]
        relevant_columns_dataset: pd.DataFrame = relevant_columns_dataset.dropna()
        # choose the number of entries
        if relevant_columns_dataset.shape[0] < ProjectQuestion.DEF_NUM_ENTRIES:
            self._num_entries = relevant_columns_dataset.shape[0]
        #####
        # create a dataset with outliers if required
        #####
        # find out/inliers if the data has no categorical variables
        if self._q_type != 'Chi squared':
            numeric_df = relevant_columns_dataset.select_dtypes(include=np.number)
            outlier_rows = np.where((np.abs(stats.zscore(numeric_df)) > ProjectQuestion.OUTLIER_THRESH).any(axis=1))[0]
            inlier_rows = np.where((np.abs(stats.zscore(numeric_df)) < ProjectQuestion.INLIER_THRESH).all(axis=1))[0]
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
            inlier_rows_idx[np.random.choice(inlier_rows, num_inliers, replace=False)] = 1
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
        np.random.seed(self._group)

    def print_dataset(self):
        ipd.display(self._dataset)

    def plot(self):
        if self._q_type == "Regression":
            Utils.scatter(self._dataset, self._vars[0], self._vars[1])
        if self._q_type == "ANOVA 2-way":
            Utils.scatter(self._dataset, self._vars[1], self._vars[0], self._vars[2])


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
        self._possible_questions = Utils.get_questions()
        self._seed()
        self._question_types = np.random.choice(self._possible_questions.keys(), replace=Project.REPEAT_Q_TYPES,
                                                size=Project.NUM_QUESTIONS)
        self._extra_var = None
        if 'Anova 1-way' in self._question_types.keys():
            self._extra_var = bool(np.random.binomial(1, 0.5, 1))
        self._questions = []
        self._generate_questions()

    def _generate_questions(self):
        self._seed()
        self._outliers = np.random.binomial(1, 0.5, len(self._question_types.keys())).astype(bool)
        for i, key in enumerate(self._question_types):
            if key == 'Anova 1-way':
                if self._extra_var:
                    q_options = [q for q in self._possible_questions[key] if len(q) == 4]
                else:
                    q_options = [q for q in self._possible_questions[key] if len(q) == 3]
            else:
                q_options = self._possible_questions[key]
            q_params = np.random.choice(q_options)
            if Project.REPEAT_Q_TYPES:
                while sum([q.is_same(q_params) for q in self._questions]) > 0:
                    q_params = np.random.choice(q_options)
            self._questions.append(ProjectQuestion(self._group, key, q_params, self._outliers[i]))

    def _seed(self) -> None:
        np.random.seed(self._group)

    def __getitem__(self, idx):
        if idx >= Project.NUM_QUESTIONS:
            raise IndexError(f"There are only {Project.NUM_QUESTIONS} questions in this part!"
                             f"Please provide a subscript between 0-{Project.NUM_QUESTIONS - 1}, and not {idx}!")
        return self._questions[idx]


# %% tests
questions = Utils.get_questions()
q_list = []
qs = dict()
for key, value in tqdm(questions.items()):
    count = 1
    qs[key] = []
    for q_params in value:
        q = ProjectQuestion(1, key, q_params, False)
        qs[key].append(q)
        q_list.append({"test": key, "count": count, "variables": q_params, "question": q.__repr__()})
        count += 1

# %%
