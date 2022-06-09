from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

class BuilTensorflowModel:
    def __init__(self, name='LSTM', compile=True):
        dct = {
            'LSTM': self.LSTMModel(),
            'BiLSTM' : self.BiLSTMModel(),
            'Tensorflow suggest': self.TensorflowModel()
        }
        self.model = dct[name]
        if compile:
            self.model.compile(optimizer=Adam(1e-4), loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        return self.model

    def LSTMModel(self):
        model = Sequential()
        model.add(Embedding(voc_size, embedding_vector_feature, input_length=sent_len))
        model.add(Dropout(0.3))
        model.add(LSTM(100))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def BiLSTMModel(self):
        model = Sequential()
        model.add(Embedding(input_dim=voc_size, output_dim=64))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1))
        return model

    def TensorflowModel(self):
        model = Sequential()
        model.add(Embedding(voc_size, sent_len))
        model.add(Dropout(0.3))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def Fit(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, early_stop=True):
        if early_stop:
            early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, mode='max', restore_best_weights=True)
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callback=[early_stop])

    def SaveModel(self, path):
        self.model.save(path)