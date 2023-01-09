from final_project_classes import Utils, QuestionReport, ProjectQuestion, GUI, Project
from ListsAndMaps import *
import sys
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from joblib import Parallel, delayed


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ProjectQuestion':
            from final_project_classes import ProjectQuestion
            return ProjectQuestion
        return super().find_class(module, name)


LOGGING = True

SUBMISSION_DIR = r'C:\Users\odedw\Downloads\FinalProjectSubmissions'


class QuestionTester:
    """
    This class contains all relevant data and performs the grading of a single question
    """
    VALUE_COMPARE_TOL = 1e-3

    def __init__(self, stud_name, stud_id, zf, q_idx):
        self.log = dict(log_dict)  # create copy
        self.check_dict = dict(check_dict)  # create copy
        self.stud_id = stud_id
        self.stud_name = stud_name
        self.zf = zf
        self.q_idx = q_idx
        self.grade = 0
        self.q_obj: ProjectQuestion = None
        self.q_methods = None
        self.q_test = None
        self.q_validity = None
        self.reported_test = None
        self.real_values = None
        self.reported_p_val = None
        self.reported_values = None
        self.test_kwargs = dict()
        self._read_reports()
        if self.q_obj is not None:
            self.is_outlier = (
                    self.q_obj.get_dataset() > 4 * self.q_obj.get_dataset().std() + self.q_obj.get_dataset().mean()).any().any()
            self.variables = self.q_obj._vars
            self.question = self.q_obj.__repr__()
            self.q_type = self.q_obj._q_type
            if self.q_type == 'ANOVA 1-way' and len(self.variables) == 3:
                self.q_type = self.q_obj._q_type = 'ANOVA 2-way'
            self._add_to_log("name", self.stud_name)
            self._add_to_log("id", self.stud_id)
            self._add_to_log("group", self.q_obj._group)
            self._add_to_log("q_idx", self.q_idx)
            self._add_to_log("question", self.question)
            self._add_to_log("outliers", self.is_outlier)
            self._add_to_log("q_type", self.q_type)
            self._add_to_log("variables", self.variables)

    def score(self):
        if self.q_obj:
            if self.q_methods is not None:
                self._check_methods_report()
            if self.q_test is not None:
                self._check_test_report()
            if self.q_validity is not None:
                self._check_validity_report()
        self.calculate_grade()

    def calculate_grade(self):
        self.grade = 0
        self.log["grading"] = ""
        for k in self.check_dict.keys():
            cur_grade = 0
            if k == "methods_contain_incorrect_test":
                if self.check_dict["methods_contain_correct_test"]:
                    cur_grade = max(-10, self.check_dict[k] * scoring_dict[k])
            elif k == "is_valid_conclusion_contains_irrelevant_tests":
                cur_grade = bool(self.check_dict[k]) * scoring_dict[k]
            elif self.check_dict[k] is not None:
                cur_grade = self.check_dict[k] * scoring_dict[k]

            self.log["grading"] += f"{k}:{cur_grade}, "
            self.grade += cur_grade
        self._add_to_log("question_grade", self.grade)
        return self.grade

    def _check_methods_report(self):
        self._check_methods_visualization()
        self._check_methods_contain_correct_test()
        self._check_methods_contain_incorrect_test()
        self._methods_contain_incorrect_effect_size()  # TODO: check here that correct effect size is correct for the reported test, not actual test
        self._check_methods_contain_correct_test_but_wrong_setting()
        self._check_methods_contain_multiple_hypothesis_testing()

    def _check_methods_contain_multiple_hypothesis_testing(self):
        self.check_dict["methods_contain_multiple_hypothesis_testing"] = bool(
            [1 for m in self.q_methods["value"] if "multiple-comparisons" in m])
        self._add_to_log("methods_multiple_hypothesis_testing", [m for m in self.q_methods["value"] if
                                                                 "multiple-comparisons" in m])

    def _check_methods_contain_correct_test_but_wrong_setting(self):
        directionality = self._get_hyp_direction()
        test_direction = hypothesis_direction_to_ttest_hypothesis_dict[directionality]
        if self.q_test is not None:
            if 't-test' in self.q_type and (
                    't-test' == self.q_test["name"]).any():  # make sure the reported settings are correct
                self.check_dict["methods_contain_correct_test_but_wrong_setting"] = ("between columns" in
                                                                                     self.q_methods[
                                                                                         "value"].values) and bool(
                    [1 for m in self.q_methods["value"] if ('t-test' in m or 'Mann-Whitney' in m)])
                method = "paired" if 'paired' in self.q_type else "independent samples"
                self.check_dict["methods_contain_correct_test_but_wrong_setting"] = \
                    self.check_dict["methods_contain_correct_test_but_wrong_setting"] or \
                    self.q_test[self.q_test["name"] == "t-test type"]["value"].values[0] != method or \
                    (self.q_test[self.q_test["name"] == "Hypothesis direction"]["value"].values[
                         0] != test_direction)
            elif 'Mann Whitney' == self.q_type and ('Mann-Whitney' == self.q_test["name"]).any():
                self.check_dict["methods_contain_correct_test_but_wrong_setting"] = ("between columns" in
                                                                                     self.q_methods[
                                                                                         "value"].values) and bool(
                    [1 for m in self.q_methods["value"] if 'Mann-Whitney' in m])
                self.check_dict["methods_contain_correct_test_but_wrong_setting"] = \
                    self.check_dict["methods_contain_correct_test_but_wrong_setting"] or \
                    (self.q_test[self.q_test["name"] == "Hypothesis direction"]["value"].values[
                         0] == test_direction)
            else:
                self.check_dict["methods_contain_correct_test_but_wrong_setting"] = False
        else:
            self.check_dict["methods_contain_correct_test_but_wrong_setting"] = False
        self._add_to_log("methods_contain_correct_test_but_wrong_setting", int(
            self.check_dict["methods_contain_correct_test_but_wrong_setting"]))

    def _methods_contain_incorrect_effect_size(self):
        self.check_dict["methods_contain_incorrect_effect_size"] = bool(
            [1 for m in self.q_methods["value"] if m in incorrect_effect_sizes[self.q_type]])
        self._add_to_log("methods_contain_incorrect_effect_size", [m for m in self.q_methods["value"] if
                                                                   m in incorrect_effect_sizes[self.q_type]])

    def _check_methods_contain_incorrect_test(self):
        self.check_dict["methods_contain_incorrect_test"] = sum(
            [1 for m in self.q_methods["value"].unique() if m in (set(all_tests) - set(
                correct_tests_dict[self.q_type] + [all_tests[0]]))])
        self._add_to_log("methods_incorrect_test",
                         [m for m in self.q_methods["value"].unique() if m in (set(all_tests) - set(
                             correct_tests_dict[self.q_type] + [all_tests[0]]))])

    def _check_methods_contain_correct_test(self):
        self.check_dict["methods_contain_correct_test"] = bool([1 for m in self.q_methods["value"] if m in
                                                                correct_tests_dict[self.q_type]])
        if self.q_test is not None:
            self.check_dict["methods_contain_correct_test"] = self.check_dict["methods_contain_correct_test"] or (
                    "Permutation test" in self.q_methods["value"].values and self.q_test["value"][0] in
                    correct_tests_dict[self.q_type])
        self._add_to_log("methods_contain_correct_test", int(self.check_dict["methods_contain_correct_test"]))

    def _check_methods_visualization(self):
        self.check_dict["methods_contain_visualization"] = bool(
            [1 for m in self.q_methods["value"] if m in visualization_methods])
        self._add_to_log("methods_visualizations", [m for m in self.q_methods["value"] if m in visualization_methods])

    def _check_test_report(self):
        # reported test is in reported methods
        if self.q_methods is not None:
            self._check_test_report_coherent_with_test_chosen_in_methods()
        self._check_test_report_values_are_correct()

    def _check_test_report_coherent_with_test_chosen_in_methods(self):
        self.check_dict["test_report_coherent_with_test_chosen_in_methods"] = (
                self.reported_test in self.q_methods["value"].values or (
                "Permutation test" in self.q_methods["value"].values))
        num_reported_tests = self.q_methods["value"].isin(
            set(all_tests) - {"Permutation test"}).sum()  # don't count permutation
        if num_reported_tests > 1:  # in this case, must contain the correct test
            self.check_dict["test_report_coherent_with_test_chosen_in_methods"] = \
                self.check_dict["test_report_coherent_with_test_chosen_in_methods"] and self.check_dict[
                    "methods_contain_correct_test"]
        self._add_to_log("test_report_coherent_with_test_chosen_in_methods")

    def _check_validity_report(self):
        if self.reported_p_val is None:
            return
        self._check_is_valid_conclusion_coherent_with_a_parametric_tests()
        valid_validity_tests = set(
            valid_validity_tests_dict[self.q_type] + visualization_methods + correct_tests_dict[self.q_type] + [
                self.reported_test] + (self.q_methods["value"].tolist() if self.q_methods is not None else []))
        self._add_to_log("valid_conclusion_irrelevant_tests",
                         [m for m in self.q_validity['value'][1:] if m not in valid_validity_tests])
        self.check_dict["is_valid_conclusion_contains_irrelevant_tests"] = bool(
            [1 for m in self.q_validity['value'][1:] if m not in valid_validity_tests])
        self._add_to_log("is_valid_conclusion_contains_irrelevant_tests")

    def _check_is_valid_conclusion_coherent_with_a_parametric_tests(self):
        self.reported_validity = (self.q_validity.iloc[0, 1] == "True")
        if LOGGING:
            self.log["a_parametric_tests_validity"] = [
                "Chose " + ("" if self.reported_validity else "in") + "valid conclusion"]
        # set up the possible validity answers. True indicates that "valid" is a possible answer, False indicated that "invalid" is a possible answer
        possible_validity_answers = set()
        if self.reported_test_aparametric:
            possible_validity_answers.add(True)
            if LOGGING:
                self.log["a_parametric_tests_validity"] += ["Reported a-parametric test, valid should be an option"]
        # add the validity indication of permutation test if it was used
        if bool([1 for m in self.q_validity["value"][1:] if "Permutation test" in m]) or \
                ((self.q_methods is not None) and bool(
                    [1 for m in self.q_methods["value"] if "Permutation test" in m])):
            perm_validity = None
            if np.isscalar(self.perm_p_val):  # in case the permutation was a scalar
                perm_validity = (self.perm_p_val < 0.05) == (self.reported_p_val < 0.05)
                possible_validity_answers.add(perm_validity)
            elif self.perm_p_val is not None:  # two way anova - 3 permutation p values to check
                perm_validity = ((np.array(self.perm_p_val) < 0.05) == (np.array(self.reported_p_val) < 0.05)).all()
                possible_validity_answers.add(
                    perm_validity)
            if LOGGING and perm_validity is not None:
                self.log["a_parametric_tests_validity"] += [
                    "Reported permutation test, " + ("" if perm_validity else "in") + "valid should be an option"]
        # Mann whitney is a viable a-parametric test for the question and was stated in the validity/methods:
        if self.q_type in ["Mann Whitney",
                           'independent samples t-test'] and bool(
            [1 for m in self.q_validity["value"][1:] if "Mann-Whitney" in m]) or (
                self.q_methods is not None and bool([1 for m in self.q_methods["value"] if "Mann-Whitney" in m])):
            if "method" in self.test_kwargs:  # actually chose a t-test for the reported test
                self.test_kwargs.pop("method")
            try:  # try to run the correct test to see if p-value matches the report
                vals = re.findall(f"{self.variables[1]}=(.*?)[ ?]", self.question)
                try:
                    vals = list(map(float, vals))  # if values are floats, not strings
                except:
                    pass  # keep as string if the levels are strings
                if not vals:  # only 2 levels exist, so the levels were not stated
                    vals = [v for v in self.q_obj.get_dataset()[self.variables[1]].unique()]
                mann_whitney_validity = (Utils.get_mann_whitney_func(self.test_kwargs["q"], self.variables[0],
                                                                     self.variables[1],
                                                                     vals[0],
                                                                     vals[1],
                                                                     hypothesis_direction_to_ttest_hypothesis_dict[
                                                                         self._get_hyp_direction()])(
                    self.q_obj.get_dataset())[
                                             1] < 0.05) == (self.reported_p_val < 0.05)
                possible_validity_answers.add(mann_whitney_validity)
                if LOGGING:
                    self.log["a_parametric_tests_validity"] += ["Reported Mann-Whitney, " + (
                        "" if mann_whitney_validity else "in") + "valid should be an option"]
            except Exception as e:
                print("Failed running mann whitney test! Error:", e, file=sys.stderr, flush=True)
        # add bootstrap validity indication to possibilities if it was reported in validity/methods
        if bool([1 for m in self.q_validity["value"][1:] if "Bootstrap" in m]) or \
                (self.q_methods is not None and bool([1 for m in self.q_methods["value"] if "Bootstrap" in m])):
            # the specific case of "states" dataset with area and Alaska.
            if self.q_obj._dataset_name == "states" and "area" in self.variables and (
                    (self.q_obj.get_dataset()['area'] > 400000) > 0).any():
                self.is_outlier = True
            possible_validity_answers.add(not self.is_outlier)
            if LOGGING:
                self.log["a_parametric_tests_validity"] += ["Reported Bootstrap, " + (
                    "" if not self.is_outlier else "in") + "valid should be an option"]
        # Mark all used aparametric tests
        self.check_dict["is_valid_conclusion_coherent_with_a_parametric_tests"] = (
                self.reported_validity in possible_validity_answers)
        self._add_to_log("is_valid_conclusion_coherent_with_a_parametric_tests")

    def _read_reports(self):
        self.q_obj = self._parse_question_file()
        self.q_methods = self._read_csv_file("methods")
        if self.q_methods is not None:
            self._add_to_log("reported_methods", self.q_methods)
        self.q_test = self._read_csv_file("question")
        if self.q_test is not None:
            self._add_to_log("reported_test", self.q_test)
        if self.q_test is not None:
            self.reported_test = self.q_test[self.q_test["name"] == "test"]["value"].values[-1]
            self.reported_test_aparametric = self.q_test[self.q_test["name"] == "test"]["value"].values[
                                                 0] in aparametric_tests
        self.q_validity = self._read_csv_file("valid_conclusion")
        if self.q_validity is not None:
            self._add_to_log("reported_validity", self.q_validity)
        try:
            with self.zf.open(f"Q{self.q_idx}_freetxt", 'r') as file:
                self._add_to_log("free_text", file.read().decode('utf-8-sig'))
        except:
            pass

    def _parse_question_file(self):
        try:
            with self.zf.open(f"Q{self.q_idx}_object", "r") as file:
                q_obj = CustomUnpickler(file).load()
        except KeyError:
            q_obj = None
        return q_obj

    def _read_csv_file(self, part):
        try:
            with self.zf.open(f"Q{self.q_idx}_{part}", "r") as file:
                read_file_df = pd.read_csv(file, header=None, names=["name", "value"], skipinitialspace=True)
                read_file_df.replace('nan', pd.NA)
                read_file_df.dropna(inplace=True)
                return read_file_df
        except KeyError:
            return None

    def _add_to_log(self, key, value=None):
        if LOGGING:
            if isinstance(value, list) and not value:
                value = None
            elif value is None:
                value = self.check_dict[key]
            self.log[key] = value

    @staticmethod
    def get_p_val(test, test_res):
        if "Repeated" in test:
            return test_res["Pr > F"].to_numpy()[0]
        if "ANOVA" in test:
            return test_res["PR(>F)"].to_numpy()[:-1]
        elif test == "Linear regression":
            return test_res[3]
        else:
            return test_res[1]

    def _check_test_report_values_are_correct(self):
        self.reported_p_val, self.perm_p_val, self.parsed_kwargs = [None] * 3
        # check if permutation was reported
        start_row = int(self.q_test.iloc[0][1] == "Permutation test")
        self.chosen_test = self.q_test.iloc[start_row][1]
        self.test_kwargs = {key: self.q_test.iloc[start_row + 1 + i, 1] for i, key in
                            enumerate(test_args_dict[self.chosen_test])}
        # transform the test function arguments to numbers if possible
        for k, v in self.test_kwargs.items():
            try:
                self.test_kwargs[k] = float(v)
            except:
                pass
        self.test_kwargs["q"] = self.q_obj
        # calculate permutation p value
        gui = GUI(self.q_obj)
        self.perm_p_val = gui._run_permutation(self.chosen_test, **self.test_kwargs)
        # if directional test
        if 't-test' in self.q_obj._q_type or self.q_obj._q_type == 'Mann Whitney' and self.perm_p_val is not None:
            direction = self._get_hyp_direction()  # -1 for smaller, 1 for greater, 0 for wo-sided

            if direction > 0:
                self.perm_p_val = 1 - self.perm_p_val
            elif direction == 0:
                self.perm_p_val *= 2  # two sided hypothesis - they should have multiplied by 2

        elif np.isscalar(self.perm_p_val):
            if self.perm_p_val >= 0.5:
                self.perm_p_val = 1 - self.perm_p_val
        elif self.perm_p_val is not None:
            for j, p in enumerate(self.perm_p_val):
                self.perm_p_val[j] = 1 - p

        try:
            test_res = test_func_getters_dict[self.chosen_test](**self.test_kwargs)(self.q_obj.get_dataset())
        except:
            print("Cannot run reported test!", file=sys.stderr)
            return

        if start_row > 0:  # if permutation was used, make sure p value is correct. The rest should be calculated by the actual test
            p_val = self.perm_p_val
        else:
            p_val = self.get_p_val(self.chosen_test, test_res)

        effect_size_func_dict = {
            "t-test between columns": lambda **kwargs: Utils.cohens_d_columns(kwargs["q"], kwargs["col1"],
                                                                              kwargs["col2"]),
            "t-test by level": lambda **kwargs: Utils.cohens_d(kwargs["q"], kwargs["col1"], kwargs["col2"],
                                                               kwargs["col2_val1"], kwargs["col2_val2"]),
            "One way ANOVA": lambda q, **kwargs: Utils.eta_squared(test_res).iloc[0][0],
            "Two way ANOVA": lambda q, **kwargs: Utils.eta_squared(test_res),
            "Repeated Measures ANOVA": lambda q, **kwargs: Utils.rm_anova(q=q, **kwargs, eta=True),
            "Mann-Whitney test between columns": lambda **kwargs: None,
            "Mann-Whitney test by level": lambda **kwargs: None,
            "Linear regression": Utils.r_squared_pearson_or_regression,
            "Spearman correlation": Utils.r_squared_pearson_or_regression,
            "Pearson correlation": Utils.r_squared_spearman,
            "Chi squared for independence": lambda **kwargs: None}
        effect_size = effect_size_func_dict[self.chosen_test](**self.test_kwargs)
        if self.chosen_test == "Two way ANOVA":
            self.reported_values = self.q_test.iloc[-9:, 1].to_numpy().astype(float)
            index = self.q_test.iloc[-9:, 0].to_numpy()
            self.real_values = np.array([test_res["F"][0], p_val[0], effect_size.iloc[0][0],
                                         test_res["F"][1], p_val[1], effect_size.iloc[1][0],
                                         test_res["F"][2], p_val[2], effect_size.iloc[2][0]])
            self.reported_p_val = [self.reported_values[1], self.reported_values[4], self.reported_values[7]]
        elif self.chosen_test == "Linear regression":
            self.reported_values = self.q_test.iloc[-4:, 1].to_numpy().astype(float)
            index = self.q_test.iloc[-4:, 0].to_numpy()
            self.reported_p_val = self.reported_values[-2]
            if start_row:
                self.reported_values[-2] = self.perm_p_val
            self.real_values = np.array([test_res[0], test_res[1], p_val, effect_size])
        elif self.chosen_test in ["Spearman correlation", "Pearson correlation"]:
            self.reported_values = float(self.q_test.iloc[-1, 1])
            index = [self.q_test.iloc[-1, 0]]
            self.reported_p_val = test_res[1]
            if start_row:
                self.reported_p_val = self.perm_p_val
            self.real_values = test_res[0]
        elif "Mann-Whitney" in self.chosen_test:
            self.reported_values = self.q_test.iloc[-2:, 1].to_numpy().astype(float)
            index = self.q_test.iloc[-2:, 0].to_numpy()
            if start_row:
                self.reported_values[-1] = self.perm_p_val
            # check if Mann whitney reported U is the matching one by mistake
            if not np.isclose(self.reported_values[0], test_res[0], rtol=0, atol=self.VALUE_COMPARE_TOL):
                test_res = np.array([test_res[0], test_res[1]])
                new_u = len(self.q_obj.get_dataset().loc[
                                self.q_obj.get_dataset()[self.test_kwargs["col2"]] == self.test_kwargs[
                                    "col2_val1"]]) * len(
                    self.q_obj.get_dataset().loc[
                        self.q_obj.get_dataset()[self.test_kwargs["col2"]] == self.test_kwargs["col2_val2"]]) - \
                        test_res[0]
                if np.isclose(self.reported_values[0], new_u, rtol=0, atol=self.VALUE_COMPARE_TOL):
                    test_res[0] = new_u
            self.real_values = np.array([test_res[0], p_val])
            self.reported_p_val = p_val
        else:
            if effect_size:
                self.reported_values = self.q_test.iloc[-3:, 1].to_numpy().astype(float)
                index = self.q_test.iloc[-3:, 0].to_numpy()
                if start_row:
                    self.reported_values[-2] = self.perm_p_val
                self.reported_p_val = self.reported_values[-2]
            else:
                self.reported_values = self.q_test.iloc[-2:, 1].to_numpy().astype(float)
                index = self.q_test.iloc[-2:, 0].to_numpy()
                if start_row:
                    self.reported_values[-1] = self.perm_p_val
                self.reported_p_val = self.reported_values[-1]

            if 'Repeated' in self.chosen_test:
                res = test_res['F Value'][0]
            elif 'ANOVA' in self.chosen_test:
                res = test_res['F'][0]
                p_val = p_val if np.isscalar(p_val) else p_val[0]
            else:
                res = test_res[0]
            if effect_size:
                self.real_values = np.array([res, p_val, effect_size])
            else:
                self.real_values = np.array([res, p_val])
            self.reported_p_val = p_val
        if self.reported_p_val is None:
            self.reported_p_val = p_val
        self._add_to_log("test_report_values_are_correct",
                         pd.DataFrame({"reported": self.reported_values, "actual": self.real_values}, index=index))
        self.check_dict["test_report_values_are_correct"] = np.isclose(np.array(self.real_values), np.array(
            self.reported_values), rtol=0, atol=self.VALUE_COMPARE_TOL).all()

    def _get_hyp_direction(self):
        if "smaller" in self.question:
            return -1
        if "larger" in self.question:
            return 1
        return 0


class StudentTester:
    def __init__(self, stud_name, stud_id, filename):
        self.stud_id = stud_id
        self.stud_name = stud_name
        self.zf = ZipFile(filename)
        self._q_testers = [QuestionTester(self.stud_name, self.stud_id, self.zf, q_idx) for q_idx in
                           range(1, 6)]
        self.grade = 0

    def get_question_logs_list(self):
        return [q_tester.log for q_tester in self._q_testers]

    def score_questions(self):
        for qt in self._q_testers:
            qt.score()

    def get_grade(self):
        self.grade = sum([qt.calculate_grade() for qt in self._q_testers])
        return self.grade


# %%
dir_list = os.listdir(SUBMISSION_DIR)
id_list = list(map(lambda s: re.findall("(\d+)", s)[0], dir_list))
name_list = list(map(lambda s: s[:s.find("_")], dir_list))
log_df = []
final_scores = []
testers = []
filenames = [os.path.join(SUBMISSION_DIR, dir_name,
                          [f for f in os.listdir(os.path.join(SUBMISSION_DIR, dir_name)) if f.endswith(".zip")][
                              0]) for dir_name in dir_list]
for i, dir_name in enumerate(tqdm(dir_list)):
    tester = StudentTester(name_list[i], id_list[i], filenames[i])
    tester.score_questions()
    testers.append(tester)
    final_scores.append({"name": name_list[i], "grade": tester.get_grade()})
    log_df.extend(tester.get_question_logs_list())

pd.DataFrame(final_scores).to_csv(r"G:\.shortcut-targets-by-id\1M4mQwvihwkeXFuKPJ7-0VY9pIh4eVRES\Statistical Methods in Psychological Research (BA) 51307\2021-2022\פרויקט סוף\בדיקה וציונים\final_scores_latest.csv", encoding='utf-8-sig')
if LOGGING:
    log_df_from_list = pd.DataFrame(log_df)
    log_df_from_list.to_csv(r"G:\.shortcut-targets-by-id\1M4mQwvihwkeXFuKPJ7-0VY9pIh4eVRES\Statistical Methods in Psychological Research (BA) 51307\2021-2022\פרויקט סוף\בדיקה וציונים\final_project_grading_latest.csv", encoding='utf-8-sig')
