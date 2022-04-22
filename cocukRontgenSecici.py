import pandas as pd
import shutil

data = pd.read_csv("Data_Entry_2017.csv")
saglikliCocukResimAdlari = data[data['Finding Labels'].str.contains("No Finding")][data['Patient Age'] < 15]["Image Index"].tolist()
for i in saglikliCocukResimAdlari:
        shutil.copy(("images/"+ i), "cocuk")