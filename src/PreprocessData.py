import pandas as pd
import csv
from sklearn import preprocessing

# Нумерует говорящих людей в "виртуальной" таблице (data), добавляя отдельную колонку для этого
# Для работы функции необходимо предварительно иметь подготовленный .csv файл
# Содержащий в себе ключевые особенности для каждой аудиозаписи (тренировочные/тестовые - не важно)
def preProcessData(csvFileName):
    print(csvFileName + " will be preprocessed")
    data = pd.read_csv(csvFileName) # размеры: 150(0) x 28(1)
    # на данный момент 3 человека: 
    # 0: Дима
    # 1: Тема
    # 2: Матвей
    filenameArray = data['filename'] 
    speakerArray = []
    #print(filenameArray)
    for i in range(len(filenameArray)):
        speaker = filenameArray[i][0]
        if speaker == "k":
            speaker = 0
        elif speaker == "b":
            speaker = 1
        elif speaker == "z":
            speaker = 2
        else:
            speaker = 6
        speakerArray.append(speaker)
    
    data['number'] = speakerArray
    
    # Удаляем ненужные колонки
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    data = data.drop(['chroma_stft'],axis=1)
    data.shape

    print("Preprocessing is finished")
    print(data.head())
    return data

if __name__ == "__main__":
    pass