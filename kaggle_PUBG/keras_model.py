import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler

from time import time
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.callbacks import TensorBoard
print('Finish import')

train_path = './dataSet/train_V2.csv'
test_path = './dataSet/test_V2.csv'
print('Finish loading data')

# Get data
class DataSet:
    def __init__(self, path, is_test=False):
        self.is_test = is_test
        self.df = self.reduce_mem_usage(pd.read_csv(path, nrows=2500000))
        self.df_id = self.df['Id']
        self.deal_feature()
        if is_test:
            sc = StandardScaler()
            self.df = sc.fit_transform(self.df)
            self.df = sc.transform(self.df)
        else:
            self.split_data()
    
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
        cv_set = self.df.loc[set(self.df.index) - set(train_set.index)]
        self.y_train = train_set[['winPlacePerc']]
        self.x_train = train_set.drop(['winPlacePerc'], 1)
        self.y_valid = cv_set[['winPlacePerc']]
        self.x_valid = cv_set.drop(['winPlacePerc'], 1)

# Create model
class create_model:
    def __init__(
        self,
        shape,
        epochs=1,
        batch_size=100000,
    ):
        self.shape=shape
        self.epochs=epochs
        self.batch_size=batch_size
        
        self.model = Sequential()
        self.build_NN()
        self.compile_model()
        self.create_callback_fn()
        
    def build_NN(self):
        self.model.add(Dense(80,input_dim=self.shape,activation='selu'))
        self.model.add(Dense(160,activation='selu'))
        self.model.add(Dense(320,activation='selu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(160,activation='selu'))
        self.model.add(Dense(80,activation='selu'))
        self.model.add(Dense(40,activation='selu'))
        self.model.add(Dense(20,activation='selu'))
        self.model.add(Dense(1,activation='sigmoid'))
        
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def create_callback_fn(self):
        # learning_rate_reduction = ReduceLROnPlateau(
        #     monitor='val_acc',
        #     patience=3,
        #     verbose=1,
        #     factor=0.5,
        #     min_lr=0.0001
        # )

        # Build tensorboard
        tensorboard = TensorBoard(
            log_dir='./Graph1',
        )

        # Saving model callback function
        # checkpoint_path = './keras_model/model_' + str(time()) + '.ckpt'
        # checkpoint = ModelCheckpoint(
        #     checkpoint_path,
        #     save_weights_only=True,
        #     verbose=1,
        # )
        self.callbacks_fn =  [tensorboard]
#         self.callbacks_fn = [tensorboard]
    
    def train(self, x_train, y_train, x_valid, y_valid):
        path = './tensorboard/keras' + str(time())
        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[TensorBoard(log_dir=path)]
        )

print('Start getting train data')
train_data = DataSet(train_path)
print('Finish getting train data')

# Train the model
model = create_model(shape=train_data.x_train.shape[1], epochs=15)
model.train(train_data.x_train, train_data.y_train, train_data.x_valid, train_data.y_valid)

# # Predict the test dataset
# print('Start getting test data')
# test_data = DataSet(test_path, is_test=True)
# print('Finish getting test data')
# predict = model.model.predict(test_data.df)
# print('Finish prediction')

# prediction = predict.ravel()
# prediction_series = pd.Series(prediction, name='winPlacePerc')

# # Submission
# submit = pd.read_csv(test_path)
# submit['winPlacePerc'] = prediction_series
# submit = submit[['Id', 'winPlacePerc']]
# submit.to_csv('sample_submission.csv', index=False)
# print(submit)

# Tensorboard validation and train graph
# import tensorflow as tf

# writer_1 = tf.summary.create_file_writer("./tensorboard/keras/training")
# writer_2 = tf.summary.create_file_writer("./tensorboard/keras/validation")

# log_var = tf.Variable(0.0)
# tf.summary.scalar("loss", log_var)
# write_loss = tf.summary.merge_all()

# session = tf.InteractiveSession()
# session.run(tf.global_variables_initializer())

# i = 0
# for train, validate in zip(model.history['loss'], model.history['val_loss']):
#     summary = session.run(write_loss, {log_var: train})
#     writer_1.add_summary(summary, i)
#     writer_1.flush()

#     summary = session.run(write_loss, {log_var: validate})
#     writer_2.add_summary(summary, i)
#     writer_2.flush()
#     i += 1