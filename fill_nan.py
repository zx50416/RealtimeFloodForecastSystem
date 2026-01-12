import numpy as np
import pandas as pd
import openpyxl

def get_values(sheet):
    arr = [] 
    for row in sheet:
        temp = []  
        for column in row:
            temp.append(column.value)
        arr.append(temp)
    return arr

gdrive_path = '/content/drive'
path = '/content/drive/MyDrive/Inundation_Forcasting/data/plotdata.xlsx'

# from google.colab import drive
# drive.mount(gdrive_path)

wb = openpyxl.load_workbook(path, data_only=True)
sheet = wb['工作表1']
df = pd.DataFrame(get_values(sheet))

rows, cols = df.shape[0], [1,4,7,10]

for i in cols:
  for j in range(1, rows):
    if(df[i][j]=='#N/A' or df[i][j]=='NULL'):
      first = df[i][j-1]
      m = j
      while df[i][m]=='#N/A' or df[i][m]=='NULL':
        m += 1
      last = df[i][m]
      delta = (last-first)/(m-j+1)
      for n in range(j, m):
        df[i][n] = first + delta*(n-j+1)
        sheet.cell(n+1, i+1).value = first + delta*(n-j+1)

wb.save('/content/drive/MyDrive/Inundation_Forcasting/data/modified_plotdata.xlsx')
