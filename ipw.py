import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pprint  # pretty print for dicts

class InversePropensityWeightLearner():
    def __init__(self, df, action_cols, outcome_cols, observation_cols, tailor_function, max_is_good=True, debug_mode=True):
        '''

        :param df:
        :param action_cols:
        :param outcome_cols:
        :param observation_cols:
        :param tailor_function: (NON INTUITIVE, READ ME) callable, able of receiving any pandas dataframe with any
                                 subset of the columns of df forming a history, and return a pandas dataframe or series
                                 or list-like object with one single value per entry of the callable's input. In order
                                 words, this function should be able to receive, for each stage j, at least a df with
                                 the history outcome_cols[j] + outcome_cols[j-1] + ... + outcome_cols[0] (in case you
                                  want to leverage past knowledge, but you can ignore it in the behaviour of your
                                  callable) and return one single value for each row (subject) to be used as a decision
                                  feature by the policy (DTR).
        :param max_is_good:
        :param debug_mode:
        '''
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Only accepts pandas DataFrame")
        # ~self.df=df~  NOTE: WE SAVE df TO A FIELD AFTER SOME NECESSARY STEPS
        if not isinstance(action_cols, (tuple, list, np.ndarray)) or len(action_cols) == 0:
            raise ValueError("Unsupported action_cols, should be non-empty list/array/tuple")
        self.action_cols = action_cols
        self.T = len(action_cols)  # number of stages
        if len(outcome_cols) == self.T + 1:  # baseline outcome is present but will not be used AS AN OUTCOME in fit()
            outcome_cols.remove(0)
        self.outcome_cols = outcome_cols
        if not callable(tailor_function):
            raise ValueError('wrong type for tailor_function; should be callable')
        self.tailor_function = tailor_function
        self.max_is_good = max_is_good

        # preprocess observation/feature columns
        # accumulate observations into histories:
        self.observation_cols = []
        accumulate_list = []
        for l in observation_cols:  # recall: predictive_cols is a list of lists
            accumulate_list += l
            self.observation_cols.append(copy.deepcopy(accumulate_list))

        # transform observations and save to simplified data frame
        # it will have a single column tailor_function(H_j) summarizing the features in stage j, for each stage
        self.df = df[action_cols + outcome_cols]
        for stage, history_cols in enumerate(self.observation_cols):
            vals = tailor_function(df[history_cols])
            if isinstance(vals, pd.DataFrame):
                self.df[['X' + str(stage)]] = vals
            elif isinstance(vals, pd.Series) or isinstance(vals, list):
                self.df.insert(len(self.df.columns), 'X' + str(stage), vals)
            elif isinstance(vals, np.ndarray):  # XXX not tested
                self.df.insert(len(self.df.columns), 'X' + str(stage), vals)
            else:
                raise ValueError("non accepted output for tailor_function given df with columns", history_cols)
        self.tailored_features = ['X' + str(stage) for stage in range(len(self.observation_cols))]

        # preprocess action columns (check that actions are binary and change their values to -1/1)
        for col in action_cols:
            action_space = df[col].unique()
            if len(action_space) != 2:
                raise ValueError('Expected a binary action space (only two possible values)')
            self.df = self.df.assign(**{col: self.df[[col]].replace(action_space[0], -1)})
            self.df = self.df.assign(**{col: self.df[[col]].replace(action_space[1], 1)})

        # --------------------------------------------------------------------------------------------------------------
        # computation of propensity function
        # --------------------------------------------------------------------------------------------------------------

        self.prop_score_models = []  # will be a list of sklearn trained regressors XXX USE predict_proba, NOT predict
        # self.prop_score_params = []  # will be a list of pairs of params (intercept and "slope" on tailor_function(H_j)) XXX
        # we will also insert the values of 1/prop_score(Hj) to df to ease further computations
        for stage, (action_col, tailored_ft) in enumerate(zip(self.action_cols, self.tailored_features)):
            y = self.df[action_col]
            X = self.df[[tailored_ft]]
            model = LogisticRegression().fit(X, y)
            # self.prop_score_params.append((model.get_params()['coef_'], model.get_params()['intercept_'])) XXX
            self.prop_score_models.append(copy.deepcopy(model))
            # insert column of 1/(w+1E-10), 1E-10 only to avoid division by zero (which means feasibility assump not ok)
            vals = (1 / (model.predict_proba(X)[:, [1]] + 1E-10))  # [:,1] to get P(A=1)
            self.df.insert(len(self.df.columns), 'W' + str(stage), vals)

        if debug_mode:
            print(self.df.head())
            print(self.df.columns)

    def estimate_value(self, policy):
        '''
        :param policy: list of decision pairs (action, threshold) meaning "DO ACTION action IF TAILORED_FT > threshold"
        :return: float
        '''


if __name__ == '__main__':
    bmiData = pd.read_csv('data/bmiData.csv', index_col=0)

    def custom_tailor_func(df):
        # return baselineBMI for stage 0 and month4BMI for stage 1
        # changing the order of the if=blocks would spoils the behavior of the function because df represents a HISTORY
        # (so baselineBMI will certainly be present in df in stage 1 when this function is  called)
        if "month4BMI" in df.columns:
            return df['month4BMI']
        if "baselineBMI" in df.columns:
            return df['baselineBMI']
        raise ValueError("not expected")

    estimator = InversePropensityWeightLearner(bmiData, ['A1', 'A2'], ["month4BMI", "month12BMI"],
                                               [["baselineBMI"], ["month4BMI"]],  # dumb choice for testing
                                               custom_tailor_func,
                                               max_is_good=False,   # we want to minimize the BMI
                                               debug_mode=True
                                               )