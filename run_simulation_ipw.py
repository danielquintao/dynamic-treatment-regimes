import numpy as np
import pandas as pd

from ipw import InversePropensityWeightLearner
import logging

logging.basicConfig(filename='./logs/simulation_IPW.log', level=logging.INFO)


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

    estimator = InversePropensityWeightLearner(data, ['a1', 'a2'], ["y1", "y"],
                                               [["x1"], ["x2"]],
                                               custom_tailor_func,
                                               max_is_good=True,
                                               debug_mode=False
                                               )
    best_val, optimal_policy = estimator.find_optimal_policy(debug=False)
    print("best_vl and optimal policy:", best_val, optimal_policy)
    logging.info(best_val)