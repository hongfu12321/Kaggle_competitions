import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.estimator import BoostedTreesRegressor
from tensorflow.feature_column import numeric_column

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

print('Finish import!')

class DataSet:
    def __init__(self, path, is_test=False):
        self.is_test = is_test
        nrows = None if is_test else 200000
        self.df = self.reduce_mem_usage(pd.read_csv(path, nrows=nrows))
        self.df_id = self.df['Id']
        self.deal_feature()
        if not is_test:
            self.split_data()
            self.columns = self.x_train.columns
    
    def reduce_mem_usage(self, df):
        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        return df
    
    def deal_feature(self):
        self.df['teamPlayers'] = self.df['groupId'].map(self.df['groupId'].value_counts())
        self.df['gamePlayers'] = self.df['matchId'].map(self.df['matchId'].value_counts())
        self.df['enemyPlayers'] = self.df['gamePlayers'] - self.df['teamPlayers']
        self.df['totalDistance'] = self.df['rideDistance'] + self.df['swimDistance'] + self.df['walkDistance']
        self.df['enemyDamage'] = self.df['assists'] + self.df['kills']
        
        totalKills = self.df.groupby(['matchId', 'groupId']).agg({'kills': lambda x: x.sum()})
        totalKills.rename(columns={'kills': 'squadKills'}, inplace=True)
        self.df = self.df.join(other=totalKills, on=['matchId', 'groupId'])
        
        self.df['medicKits'] = self.df['heals'] + self.df['boosts']
        self.df['killPlaceOverMaxPlace'] = self.df['killPlace'] / self.df['maxPlace']
        self.df['avgKills'] = self.df['squadKills'] / self.df['teamPlayers']
        self.df['distTravelledPerGame'] = self.df['totalDistance'] / self.df['matchDuration']
        self.df['killPlacePerc'] = self.df['killPlace'] / self.df['gamePlayers']
        self.df['playerSkill'] = self.df['headshotKills'] + self.df['roadKills'] + self.df['assists'] - (5 * self.df['teamKills'])
        self.df['gamePlacePerc'] = self.df['killPlace'] / self.df['maxPlace']
        
        drop_cols = ['killPoints', 'rankPoints', 'winPoints', 'maxPlace', 'Id', 'groupId', 'matchId', 'matchType']
        self.df.drop(drop_cols, axis=1, inplace=True)
        for feature in self.df.columns:
            if self.df[feature].isnull().sum() > 0:
                self.df[feature].fillna(0, inplace=True)
        self.df.replace([np.inf, -np.inf], 0)
        
    def split_data(self):
        train_set = self.df.sample(frac=0.8, replace=False, random_state=100)
        valid_set = self.df.loc[set(self.df.index) - set(train_set.index)]
        self.y_train = train_set[['winPlacePerc']]
        self.x_train = train_set.drop(['winPlacePerc'], 1)
        self.y_valid = valid_set[['winPlacePerc']]
        self.x_valid = valid_set.drop(['winPlacePerc'], 1)

def create_feature_columns(features):
    feature_columns = []
    for feature in features:
        feature_columns.append(
            tf.feature_column.numeric_column(
                feature,
                dtype=tf.float32))
    return feature_columns

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).    
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

train_data = DataSet("./dataSet/train_V2.csv")
train_data.df.head()
print('Finish load train dataset')

feature_columns = create_feature_columns(train_data.columns)
n_batches = 1

estimator = BoostedTreesRegressor(
    feature_columns,
    n_trees=50,
    max_depth=3,
    n_batches_per_layer=1,
    center_bias=True,
    model_dir='./tensorbord/boosted_tree'
)

NUM_EXAMPLES = len(train_data.y_train)
train_input_fn = make_input_fn(train_data.x_train, train_data.y_train)
eval_input_fn = make_input_fn(train_data.x_valid, train_data.y_valid, shuffle=False, n_epochs=1)

for _ in range(10):
    estimator.train(train_input_fn, max_steps=100)
    result = est.evaluate(valid_input_fn)
    print(result)

print('Finish training')

test = DataSet('../data/test.csv')

print('Finish load test dataset')

labels = test.df_id
# test_X = utils.scale_features(test[FEATURE_COLS], FEATURE_COLS, utils.tanh_scalar)
# test_X = utils.reduce_mem(test_X)

test_input_fn = lambda: Dataset.from_tensors(dict(test.df))

test_dicts = list(estimator.experimental_predict_with_explanations(test_input_fn))

print('Finish predict')

placements = pd.Series([round(p['predictions'][0], 4) for p in test_dicts])

submission = pd.DataFrame({'Id': labels.values, 'winPlacePerc': placements.values})
submission.to_csv('submission.csv', index=False)

print('Finish submission')