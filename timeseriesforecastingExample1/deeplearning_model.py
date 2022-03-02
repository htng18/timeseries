from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed


def dense_model(num_dim, output_steps):

    '''
    Dense model
    '''
    model = Sequential()
    model.add(Dense(20,input_shape=(num_dim,)))
    model.add(Dropout(0.05))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def lstm_model(input_steps, num_features, output_steps):
    '''
    LSTM model
    '''
    model = Sequential()
    model.add(LSTM(10,input_shape=(input_steps, num_features),return_sequences=False))
    model.add(Dense(5, activation='tanh'))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def cnn_model(input_steps, num_features, output_steps):
    '''
    CNN model
    '''
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(input_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def cnnlstm_model(num_steps, num_features, output_steps):
    '''
    CNN-LSTM model
    '''
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), 
              input_shape=(None, num_steps, num_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(20, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model