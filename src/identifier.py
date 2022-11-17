from FeatureExtractor import extractWavFeatures
from PreprocessData import preProcessData
from VoiceIdentifier import VoiceIdentifier

import numpy as np

import time

timer = time.time()

TRAIN_CSV_FILE = "../data/csv/train.csv"
# ТЕСТ НАДО ПЕРЕИМЕНОВАТЬ ВО ЧТО-НИБУДЬ ТИПА "current" И Т.П.
TEST_CSV_FILE = "../data/csv/test.csv"

# Экстрактим фичерсы (хаха нефтянные)
# uncomment if needed
#extractWavFeatures("../data/recordings/trainData", TRAIN_CSV_FILE)
extractWavFeatures("../data/recordings/testData", TEST_CSV_FILE)
print("Done1 ====================================================================================================================================================")
# Препроцессим датасеты
train_data = preProcessData(TRAIN_CSV_FILE)
test_data = preProcessData(TEST_CSV_FILE)
print("Done PREPROCESS ====================================================================================================================================================")

# Инициализируем модельку
voice_identifier = VoiceIdentifier(trainData=train_data, testData=test_data)#
print("Done INIT====================================================================================================================================================")

# Сплиттим (на трен., валидац. и тестовый) и нормализуем датасет
voice_identifier.SplitAndNormalizeDataset()
print("Done SPLITTING AND NORMALIZE ====================================================================================================================================================")

# Создаём модель (зайди внутрь чтобы посмотреть структуру)
# uncomment if needed to create a model again
#voice_identifier.CreateModel()

# Тренируем
# uncomment if needed to retrain
#voice_identifier.FitModel()

# uncomment if needed to save the model
#voice_identifier.SaveModel()
voice_identifier.LoadModel()
print("Done LOADING WEIGHTS ====================================================================================================================================================")

# Забираем модель
model = voice_identifier.GetModel()
X_test = voice_identifier.GetXTest()
y_test = voice_identifier.GetYTest()
print("Done TAKING OUT THE MODEL ====================================================================================================================================================")

def printPredictions(X_data, y_data, printDigit):
    print('\n# Generate predictions')
    for i in range(len(y_data)):
        #print("weight: ",model.predict(X_data[i:i+1]))
        prediction = voice_identifier.GetSpeakerNameStr(np.argmax(model.predict(X_data[i:i+1]), axis=-1)[0])
        #deprecetad:  prediction = getSpeaker(model.predict_classes(X_data[i:i+1])[0])

        speaker = voice_identifier.GetSpeakerNameStr(y_data[i])
        if printDigit == True:
            print("Number={0:d}, y={1:10s}- prediction={2:10s}- match={3}".format(i, speaker, prediction, speaker==prediction))
        else:
            print("y={0:10s}- prediction={1:10s}- match={2}".format(speaker, prediction, speaker==prediction))

printPredictions(X_test[0:100], y_test[0:100], False)

timer = time.time() - timer
print(timer)

# 1. Грузим веса в нейронку - LoadWeights()
# 2. Экстрактим ключевые особенности .wav файла с помощью FeatureExtractor
# 3. np.argmax(model.predict(ключевые особенности только что записанного глс), axis=-1)[0] - аргумент в predicted name = voice_identifier.GetSpeakerNameStr(аргумент)