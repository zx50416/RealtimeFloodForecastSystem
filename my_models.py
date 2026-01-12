import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.layers import Dense, Conv1D, GRU, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
from tensorflow.keras.callbacks import EarlyStopping
import joblib

class My_BPNN():
    def __init__(self, hidden_units=64):
        self.name = 'BPNN'
        self.hidden_units = hidden_units  # 儲存外部傳入的參數

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X, W1, W2):
        hidden = self.sigmoid(np.dot(X, W1))
        output = self.sigmoid(np.dot(hidden, W2))
        return hidden, output

    def backward(self, X_train, Y_train, output, hidden, W2):
        error_output = Y_train - output
        delta_output = error_output * output * (1 - output)
        error_hidden = np.dot(delta_output, W2.T)
        delta_hidden = error_hidden * hidden * (1 - hidden)
        dW2 = np.dot(hidden.T, delta_output)
        dW1 = np.dot(X_train.T, delta_hidden)
        return dW1, dW2

    def train(self, X_train, Y_train, epochs, learning_rate):
        num_input = X_train.shape[1]
        num_output = Y_train.shape[1]
        num_hidden = self.hidden_units  # 用使用者設定的值

        W1 = np.random.randn(num_input, num_hidden)
        W2 = np.random.randn(num_hidden, num_output)
        for epoch in range(epochs):
            hidden, output = self.forward(X_train, W1, W2)
            dW1, dW2 = self.backward(X_train, Y_train, output, hidden, W2)
            W1 += learning_rate * dW1
            W2 += learning_rate * dW2
        return W1, W2

    def predict(self, X_test, W1, W2):
        _, Y_predict = self.forward(X_test, W1, W2)
        return Y_predict

    def save(self, path, W1, W2):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ 自動建立儲存資料夾
        np.savez(path, W1=W1, W2=W2)

class My_LSTM():
    def __init__(self, steps, features, units=64, num_layers=4, dropout_rate=0.25):
        self.name = 'LSTM'
        self.model = Sequential()
        for i in range(num_layers):
            return_seq = True if i < num_layers - 1 else False
            if i == 0:
                self.model.add(LSTM(units, return_sequences=return_seq, input_shape=(steps, features)))
            else:
                self.model.add(LSTM(units, return_sequences=return_seq))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, Y_test), verbose=0)
        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.close('all')  # 避免圖像記憶體堆積
        return self.model

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def clear(self):
        from keras import backend as K
        del self.model
        K.clear_session()

class My_CNN():
    def __init__(self, steps, features, cnn_filters_1=64, cnn_filters_2=32, cnn_dense_units=32, dropout_rate=0.25):
        self.name = 'CNN'
        self.model = Sequential()
        self.model.add(Conv1D(filters=cnn_filters_1, kernel_size=5, activation='relu', input_shape=(steps, features)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Conv1D(filters=cnn_filters_2, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(cnn_dense_units, activation='relu'))
        self.model.add(Dense(1))

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, Y_test), verbose=0)
        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.close('all')  # 避免圖像記憶體堆積
        return self.model

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def clear(self):
        from keras import backend as K
        del self.model
        K.clear_session()

class My_SVM():
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.01):
        self.name = 'SVM'
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
   
    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        return self  # ✅ 回傳 My_SVM 物件本身

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ 自動建立目錄
        joblib.dump(self.model, path)

    def predict(self, X):
        return self.model.predict(X)

class My_GRU():
    def __init__(self, steps, features, units, num_layers, dropout_rate):
        self.name = 'GRU'
        self.model = Sequential()
        for i in range(num_layers):
            return_seq = True if i < num_layers - 1 else False
            if i == 0:
                self.model.add(GRU(units, return_sequences=return_seq, input_shape=(steps, features)))
            else:
                self.model.add(GRU(units, return_sequences=return_seq))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))  # 最終輸出為單一值

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(X_test, Y_test), callbacks=[early_stop],
                         verbose=0)

        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.close('all')  # 避免圖像記憶體堆積
        return self.model

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
    def clear(self):
        from keras import backend as K
        del self.model
        K.clear_session()

class My_XGBoost():
    def __init__(self, max_depth=3, learning_rate=0.05, n_estimators=500):
        self.name = 'XGBoost'
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='reg:squarederror',
            verbosity=0
        )

    def train(self, X_train, Y_train):
        Y_train = np.ravel(Y_train)  # ✅ 新增：避免 (N,1) 造成的 shape 問題
        self.model.fit(X_train, Y_train)
        return self  # 回傳本物件以便鏈式操作

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

        
class My_RandomForest():
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.name = 'RandomForest'
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class My_DecisionTree():
    def __init__(self, max_depth=None, random_state=42):
        self.name = 'DecisionTree'
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        
class My_LSTM_Seq2Seq():
    def __init__(self, input_timesteps, input_dim, output_timesteps, units=64, dropout_rate=0.25):
        self.name = 'LSTM_Seq2Seq'
        self.units = units
        self.dropout_rate = dropout_rate

        # Encoder
        encoder_inputs = Input(shape=(input_timesteps, input_dim))
        encoder = LSTM(units, return_state=True, dropout=dropout_rate)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(output_timesteps, input_dim))
        decoder_lstm = LSTM(units, return_sequences=True, return_state=True, dropout=dropout_rate)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(1)
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def train(self, X_encoder, X_decoder, Y_decoder, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit([X_encoder, X_decoder], Y_decoder,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 callbacks=[early_stop],
                                 verbose=0)

        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.close('all')  # 避免圖像記憶體堆積
        return self.model

    def predict(self, X_encoder, X_decoder):
        return self.model.predict([X_encoder, X_decoder])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
    def clear(self):
        from keras import backend as K
        del self.model
        K.clear_session()