import numpy as np
import pandas as pd


df1 = pd.read_csv('calonpembeli_ch5.csv')
df2 = pd.read_csv('Data_Sintetik.csv')  # Data sintetik
df1_desc = df1.describe()


# Terdapat data yang tidak masuk akal pada kolom usia, dimana usia max 164 tahun.
# Oleh karena itu dilakukan filtering untuk membuang data tersebut agar tidak mengganggu training

df1 = df1[df1['Usia'] <= 100]  # Menghilangkan usia yang >= 100
df1_desc = df1.describe()
print('\n',df1_desc)  # Usia max jadinya 65

# Mencari tahu apakah ada data yang bernilai kosong(null)
df1_nul = df1.isnull().sum()
print('\n', df1_nul)

# Melihat berapa banyak yang memutuskan untuk membeli mobil
df_buy = df1['Beli_Mobil'].value_counts()
print('\n1 -> Beli mobil'
      '\n',df_buy)  # Terdapat 633 orang yang membeli mobil

desc_buyer = df1[df1['Beli_Mobil']==1]  #melihat karakterisktik dari pembeli mobil
print(desc_buyer)
by = desc_buyer.describe()

# Melakukan training & Model selection
import sklearn.model_selection as ms
X2 = df2 [['Usia','Status','Kelamin','Memiliki_Mobil','Penghasilan']]  # Data Sintetik
X = df1 [['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']]  # Kolom id tidak diperlukan, jadi tidak dimasukkan ke dalam training & test data set
y = df1.Beli_Mobil
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2,random_state=0)

# membuat model logistic regresion
import sklearn.linear_model as lm
model = lm.LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
print('\n',model.coef_)

# Menggunakan model untuk mengeluarkan seluruh hasil prediksi thdp tes dataset
y_prediksi = model.predict(X_test)
y2 = model.predict(X2)  # Data sintetik
print('\n', y_prediksi)
print('\n', y2)
a = X_test.head()
b = y_test.head()

print('\n',a)
print('\n',b)


# Mengukur kinerja model (using confusion matrix; True Positif, True Negatif, False Positif, False Negatif)

import sklearn.metrics as met
confusionmet = met.confusion_matrix(y_test, y_prediksi)  # Dari 200 data 9 miss prediction

''' rumus accuracy = (TP + TN) / (TP + TN + FP + FN)
'''
# Menggunakan score
accuracy = model.score(X_test , y_test)
print('\n', accuracy)

'''
Angka di atas mengukur keseluruhan akurasi model, tanpa membedakan error FP maupun FN.
Ini kurang informatif terutama pada penerapan model yang difokuskan pada mendeteksi hal-hal
yang sangat peka pada False Positif atau False Negatif saja.'''

# Mengecek kepresisian model
precision = met.precision_score(y_test, y_prediksi)
print('\n', precision)

# Action untuk data sintetik
df2['Beli_Mobil_Predict'] = y2
df2.to_csv('Data_Sintetik_predict.csv',index=False)


# Membuat visualisasi

#fig = plt.figure()
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.plot(y_prediksi, y_test)
#axes.set_xlabel('Sumbu X')
#axes.set_ylabel('Sumbu y')
#axes.set_title('Grafik');


