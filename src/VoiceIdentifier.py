import numpy as np
# for splitting dataset
from sklearn.model_selection import train_test_split
# for normalizing dataset
from sklearn.preprocessing import StandardScaler
# for model
from keras import models
from keras import layers
# for early stopping the model - попробуй это
from keras.callbacks import EarlyStopping
# for and loading weights
import csv

class VoiceIdentifier:
    def __init__(self, trainData = None, testData = None):
        self.model = models.Sequential()
        self.trainData = trainData
        self.testData = testData
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def SaveModel(self):
        self.model.save("../data/vi_model")

    def LoadModel(self):
        self.model = models.load_model("../data/vi_model")
            

    def CreateModel(self):
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(3, activation='softmax')) # мб не 3, а 10
        # ПРИ ДОБАВЛЕНИИ НОВОГО ЧЕЛОВЕКА УВЕЛИЧЬ КОЛИЧЕСТВО ВЫХОДНЫХ НЕЙРОНОВ!!!

        # Learning Process of a model
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    def FitModel(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train with early stopping to avoid overfitting
        history = self.model.fit(self.X_train,
                            self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=50,
                            batch_size=3)
        
        return history

    # Разделяет датасет на тренировочные, проверочный и тестовый
    def SplitAndNormalizeDataset(self):
        # Сначала разделяем датасет
        if (not(self.trainData is None)):
            X = np.array(self.trainData.iloc[:, :-1], dtype = float)
            y = self.trainData.iloc[:, -1]
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            print("Y from training data:", self.y_train.shape)
            print("Y from validation data:", self.y_val.shape)
        

        self.X_test = np.array(self.testData.iloc[:, :-1], dtype = float)
        self.y_test = self.testData.iloc[:, -1]

        print("Y from test data:", self.y_test.shape)

        # Теперь нормализуем входные значения
        scaler = StandardScaler()
        if (not(self.trainData is None)):
            self.X_train = scaler.fit_transform( self.X_train )
            self.X_val = scaler.transform( self.X_val )
        self.X_test = scaler.transform( self.X_test )

    def GetModel(self):
        return self.model
        
    def GetXTest(self):
        return self.X_test
        
    def GetYTest(self):
        return self.y_test

    def GetSpeakerNameStr(self, speaker_number):
        speaker_name = " "
        
        if speaker_number == 0:
            speaker_name = "Kazanin"
        elif speaker_number == 1:
            speaker_name = "Bednarsky"
        elif speaker_number == 2:
            speaker_name = "Zaikov"
        else: 
            speaker_name = "Unknown"

        return speaker_name
    
if __name__ == "__main__":
    pass