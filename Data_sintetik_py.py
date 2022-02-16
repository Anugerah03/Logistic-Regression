# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:21:07 2022

@author: Nisfu
"""

import numpy as np
import pandas as pd
from random import randint

np.random.seed(100)


data = {'Id': list(range(1,1001)), 
        'Usia': [np.random.randint(20,90) for i in range(1000)],
        'Status': [np.random.randint(0,4) for i in range(1000)],
        'Kelamin': [np.random.randint(0,2) for i in range(1000)],
        'Memiliki_Mobil': [np.random.randint(0,4) for i in range(1000)],
        'Penghasilan': [np.random.randint(72,400) for i in range(1000)],
        'Beli_Mobil': [np.random.randint(0,2) for i in range(1000)]}

df = pd.DataFrame(data)

df.to_csv('Data_Sintetik.csv',index=False)

# Membuat data Id
#Id = list(range(1,1001))
#df_id = pd.DataFrame(Id, Id, ['Id'])
    

# Membuat data umur
#Usia = []
#for i in range (1000):
#    Um = randint(20,90)
#    Usia.append(Um)   

#df_age = pd.DataFrame(Usia, Id, ['Usia'])

# Membuat data penghasilan   
#penghasilan = []
#for i in range(1000):
#    income = randint(175,400)
#    penghasilan.append(income)

#df_penghasilan = pd.DataFrame(penghasilan,Id,['Penghasilan'])
#a = df_penghasilan.describe()



