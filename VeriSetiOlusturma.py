import pandas as pd
import shutil

def ayikla(resimKlasorYolu, csvDosyaYolu, hastalik, klasor, resimAdedi = 250):
    data = pd.read_csv(csvDosyaYolu, usecols = ['Image Index','Finding Labels'])
    hastaliklar = data["Finding Labels"].tolist()
    resimler = data["Image Index"].tolist()
    resimList = list()
    idx = 0
    for i in hastaliklar:
        if i == hastalik:
            resimList.append(resimler[idx])
            if len(resimList) == resimAdedi:
                break
        idx += 1

    for i in resimList:
        shutil.copy((resimKlasorYolu +'/'+ i), klasor)

ayikla("images", "Data_Entry_2017.csv", "Cardiomegaly", "Cardiomegaly", 250)
ayikla("images", "Data_Entry_2017.csv", "Pneumothorax", "Pneumothorax", 250)
ayikla("images", "Data_Entry_2017.csv", "Infiltration", "Infiltration", 250)
ayikla("images", "Data_Entry_2017.csv", "No Finding", "No Finding", 250)