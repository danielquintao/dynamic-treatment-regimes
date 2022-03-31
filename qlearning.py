import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import copy
import pprint  # pretty print for dicts

class LinearQLearner():
    def __init__(self, df, action_cols, outcome_cols, predictive_cols,
                 tailoring_cols='repeat', add_intercept=True, max_is_good=True):
        '''
        Initializer an object for learning a DTR policy through a Q-learn procedure where the Q-function is estimated
        with linear regression
        :param df: pandas dataframe with data to learn from. We expect the information corresponding to different stages
                   of the study to be present in different columns. TODO create auxiliary function to format the df
        :param action_cols: list of names of the columns containing the actions/treatments. Ordered with study stages
                            The actions themselves should be binary (strings and other objects are accepted, but only 2
                            unique values per action column)
        :param outcome_cols: list of names of columns containing the outcomes.  Ordered with study stages. The outcomes
                              should be real values.
        :param predictive_cols: list of lists. The i-th inner list contains the names of the columns to be inserted in
                                the predictive features (i.e. those that are NOT multiplied by the action in the linear
                                regression model) for estimating the i-th Q-function. Since the history of a subject
                                increases between each stage as H_i = (H_{i-1}, O_i, A_{i-1}), we take care of appending
                                the previous lists to the new ones so you only need to include the states O_i of the
                                i-th stage and which should appear in the predictive features of the i-th Q-function.
                                Do not add the action variable or the model will be wrong.
        :param tailoring_cols: list of lists or string 'repeat'. This variable follows te same logic as the the argument
                               predictive_cols, but for the tailoring features (i.e. those that ARE multiplied by the
                               action in the linear regression model). Once more, do NOT include the action variable.
                               Defaults to 'repeat', which uses the same value as provided for predictive_cols.
        :param add_intercept: bool. True if the code should add an intercept term to the features, False if it is
                              already included in the predictive_cols and tailoring_cols arguments. This variable
                              concerns both predictive and tailoring features at once. Defaults to True.

        :param max_is_good: bool. Whether the goal is to maximize (True) or to minimize the outcomes. Defaults to True.
        '''
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Only accepts pandas DataFrame")
        self.df = df
        if not isinstance(action_cols, (tuple, list, np.ndarray)) or len(action_cols) == 0:
            raise ValueError("Unsupported action_cols, should be non-empty list/array/tuple")
        self.action_cols = action_cols
        self.T = len(action_cols)  # number of stages
        self.outcome_cols = outcome_cols
        self.max_is_good = max_is_good
        self.policy_params = {
            'pred': {},  # dict, because we will insert elements backwards so indexing a list would fail
            'tailor': {}  # same
        }

        # preprocess action columns (check that actions are binary and change their values to -1/1)
        for col in action_cols:
            action_space = df[col].unique()
            if len(action_space) != 2:
                raise ValueError('Expected a binary action space (only two possible values)')
            self.df[[col]] = self.df[[col]].replace(action_space[0], -1)
            self.df[[col]] = self.df[[col]].replace(action_space[1], 1)

        # preprocess feature columns
        if add_intercept:  # should only be False if df has an intercept column and it is present in predictive_cols
            self.df['intercept'] = [1 for _ in range(len(df))]
        # predictive features:
        self.predictive_cols = []
        accumulate_list = ['intercept'] if add_intercept else []
        for l in predictive_cols:  # recall: predictive_cols is a list of lists
            accumulate_list += l
            self.predictive_cols.append(copy.deepcopy(accumulate_list))
        # tailoring features:
        if tailoring_cols == 'repeat':
            self.tailoring_cols = copy.deepcopy(self.predictive_cols)  # XXX deepcopy maybe not needed here
        else:
            self.tailoring_cols = []
            accumulate_list = ['intercept'] if add_intercept else []
            for l in tailoring_cols:  # recall: tailoring_cols is a list of lists
                accumulate_list += l
                self.tailoring_cols.append(copy.deepcopy(accumulate_list))

    def q_function(self, pred_cols_stage, tailor_cols_stage, params_pred, params_tailor, action):
        '''
        compute Q-function
        :param pred_cols_stage: list of str
        :param tailor_cols_stage: list of str
        :param params_pred: ndarray
        :param params_tailor: ndarray
        :param action: int or float or array
        :return: ndarray of shape (N_subjects, 1)
        '''
        # beta_pred^T @ H_pred
        prediction = np.dot(self.df[pred_cols_stage].to_numpy(), params_pred)
        # action * beta_tailor^T @ H_tailor:
        tailoring = action * np.dot(self.df[tailor_cols_stage].to_numpy(), params_tailor)
        return (prediction + tailoring).reshape(-1, 1)

    def opt_q_function(self, pred_cols_stage, tailor_cols_stage, params_pred, params_tailor):
        # normal workflow:
        args = (pred_cols_stage, tailor_cols_stage, params_pred, params_tailor)
        all_q_funcs = np.concatenate(
            [self.q_function(*args, -1), self.q_function(*args, 1)], axis=1
        )
        if self.max_is_good:
            return np.max(all_q_funcs, axis=1, keepdims=True)  # best value per subject
        else:
            return np.min(all_q_funcs, axis=1, keepdims=True)  # best value per subject

    def fit(self, debug_mode=True):
        for stage in range(self.T-1, -1, -1):  # stages supposed 0-indexed
            # create pseudo-outcome
            if stage == self.T - 1:
                Y = self.df[[self.outcome_cols[stage]]].to_numpy()
            else:
                Y = self.df[[self.outcome_cols[stage]]].to_numpy() +\
                    self.opt_q_function(self.predictive_cols[stage+1], self.tailoring_cols[stage+1],
                                        self.policy_params['pred'][stage+1], self.policy_params['tailor'][stage+1])
            if debug_mode:
                print('stage', stage, 'pseudo_outcome shape =', Y.shape)
            # get covariates
            X_pred = self.df[self.predictive_cols[stage]]   # H_pred
            X_tailor = self.df[[self.action_cols[stage]]].to_numpy() * self.df[self.tailoring_cols[stage]]  # action * H_tailor
            X = pd.concat([X_pred, X_tailor], axis=1)
            if debug_mode:
                print('features (head, before numpyfying):\n', X.head(3))
                print(X.columns)
                print('shape of features:', X.shape)
            X = X.to_numpy()
            # regression
            params, _, _, _ = linalg.lstsq(X, Y)
            self.policy_params['pred'][stage] = params[0:len(self.predictive_cols[stage])]
            self.policy_params['tailor'][stage] = params[len(self.predictive_cols[stage]):]
            if debug_mode:
                print('shape of computed params (pred and tailor resp.):', self.policy_params['pred'][stage].shape,
                      self.policy_params['tailor'][stage].shape)

        if debug_mode:
            print('POLICY:')
            pprint.pp(self.policy_params)






if __name__ == '__main__':
    bmiData = pd.read_csv('data/bmiData.csv', index_col=0)
    estimator = LinearQLearner(bmiData, ['A1', 'A2'], ["month4BMI", "month12BMI"],
                               [["gender", "race", "parentBMI", "baselineBMI"], []],
                               max_is_good=False)  # we want to minimize the BMI
    estimator.fit(debug_mode=True)