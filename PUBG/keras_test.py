import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


print('Import finished!')

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

train = reduce_mem_usage(pd.read_csv("./dataSet/smallSet.csv"))
# test_orj  = reduce_mem_usage(pd.read_csv("./dataSet/test_V2.csv"))

# train.head(5000).to_csv("./dataSet/smallSet.csv")

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test_orj.shape))

# train.info()

import time

# Select optimizer
lr = 1e-4
sgd = keras.optimizers.SGD(lr=lr, momentum=0.9)
rms_prop = keras.optimizers.RMSprop(lr=lr)
adam = keras.optimizers.adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
adamax = keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

def build_callback_fn():
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=0.0001
    )

    # Build tensorboard
    tensorboard = TensorBoard(
        log_dir='./Graph',
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    # Saving model callback function
    checkpoint_path = './keras_model/model_' + str(time.time()) + '.ckpt'
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        verbose=1,
    )
    return [learning_rate_reduction, tensorboard, checkpoint]

def model(data, batch_size, epochs):
    # Separate data
    Y_train = data.winPlacePerc
    X_train = data.drop('winPlacePerc', axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

    print('\033[31;1m' + str(build_callback_fn()) + '\033[0m')
    # Sequential model
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=X_train.shape))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Load model
    model.compile(optimizer=adam, loss='mse', metrics=['mae', 'accuracy'])

    # Fit model
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=build_callback_fn()
    )

    # Evaluate model
    score = model.evaluate(X_val, Y_val)
    print('Test loss: {}, mse: {}, accuracy: {}'.format(score[0], score[1], score[2]))

    return model, history

def print_tensorboard(history):
    writer_1 = tf.summary.FileWriter("./4logs/training")
    writer_2 = tf.summary.FileWriter("./4logs/validation")

    log_var = tf.Variable(0.0)
    tf.summary.scalar("loss", log_var)
    write_loss = tf.summary.merge_all()

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    i = 0
    for train, validate in zip(history.history['loss'], history.history['val_loss']):
        summary = session.run(write_loss, {log_var: train})
        writer_1.add_summary(summary, i)
        writer_1.flush()

        summary = session.run(write_loss, {log_var: validate})
        writer_2.add_summary(summary, i)
        writer_2.flush()
        i += 1

# Preprocess the data set
simple_data = pd.DataFrame({
    'rideDistance': train.rideDistance,
    'kills': train.kills,
    'boosts': train.boosts,
    'walkDistance': train.walkDistance,
    'winPlacePerc': train.winPlacePerc,
})

model(simple_data, 5, 10)