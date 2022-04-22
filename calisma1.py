import numpy as np 
import pandas as pd 
import os
from glob import glob
pd.set_option('display.max_colwidth', None)

dataFrame = pd.read_csv("E:/Github/Akciger-Hastaliklari-Tespit-Sistemi/a.csv")
globx = glob("E:/Github/Akciger-Hastaliklari-Tespit-Sistemi/images/000000*.png")
imgPaths = {os.path.basename(x): x for x in globx}
dataFrame['Path'] = dataFrame['Image Index'].map(imgPaths.get)
for i in list(imgPaths.values()):
    i = i.replace("\\", "/")
    imgPaths[i[-16:]] = i
    print(i)

dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

for label in dummy_labels:
    dataFrame[label] = dataFrame['Finding Labels'].map(lambda result: 1.0 if label in result else 0)

dataFrame['Target Vector'] = dataFrame.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])

for col in dataFrame.columns:
    if (col != "Path") and (col != "Target Vector"):
        dataFrame.drop(col, axis = 1, inplace = True)

#print(dataFrame)