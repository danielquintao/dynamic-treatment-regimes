import numpy as np
import pandas as pd

from qlearning import LinearQLearner
import logging

logging.basicConfig(filename='./logs/simulation_QLeaning.log', level=logging.INFO)


def custom_tailor_func(df):
    # changing the order of the if=blocks would spoils the behavior of the function because df represents a HISTORY
    if "x2" in df.columns:
        return df['x2']
    if "x1" in df.columns:
        return df['x1']
    raise ValueError("not expected")


for i in range(1, 51):
    data = pd.read_csv('data/simulation' + str(i) + '.csv', index_col=0)
    logging.info('Read data set ' + 'data/simulation' + str(i) + '.csv')

    estimator = LinearQLearner(data, ['a1', 'a2'], ["y1", "y"],
                               [["x1"], ["x2"]],
                               max_is_good=True)  # we want to minimize the BMI
    estimator.fit(debug_mode=False)

    best_val = estimator.get_optimal_value()
    print("best_vl and optimal policy:", best_val)
    logging.info(best_val)