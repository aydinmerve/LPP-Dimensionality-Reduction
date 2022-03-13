# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:53:23 2021

@author: merve
"""
# kütüphaneleri import ediyoruz
#from sklearn.metrics import accuracy_score

#NumPy, Python programlama dili için bir kütüphanedir; 
#büyük, çok boyutlu diziler ve matrisler için destek ekler ve 
#bu dizilerde çalışacak geniş bir üst düzey matematiksel 
#işlev koleksiyonu sunar.
from PIL import Image
import numpy as np
import glob


#veri ölçeklendirilip normalize edildi.
def build_dataset():
	org_dataset = []
	labels = []
    
    #16 örnek verimiz var.
    #Glob modülü, Python’da belirli bir klasör içindeki dosyaları 
    #listelememize yardımcı olan  bir modüldür.
    #filelist dosyayı listeleme modülü
    #zfill sıfır ekleme 
    
    #1'dan 16'e kadar bir sayı dizisi oluşturun ve sırayla her öğeyi 
    #yazdırın:
	for i in range(1, 16):   
		filelist = glob.glob('./data/subject'+str(i).zfill(2)+"*")
		for fname in filelist:
			img = Image.open(fname)
            # image resize küçüt,yeniden boyutlandırma
			img = np.array(img.resize((32, 32), Image.ANTIALIAS))
			img = img.reshape(img.shape[0] * img.shape[1])
            # dönüştürdüğüm image data setine ekle
			org_dataset.append(img)
			labels.append(i)
	return np.array(org_dataset), np.array(labels)

data, labels = build_dataset()

## Normalize
data = data/255
print(len(data))   #165 datamız var bu değeri döndürür.

# verileri test ve eğitim olarak ayırma
#Her sınıf için 7 görüntü eğitim kalanları test düşünebilirsiniz
#her sınıfta 10 veri var 7 eğitim 3 test şeklinde ayırdık.
# shuffle Bir listeyi karıştırın (liste öğelerinin sırasını yeniden düzenleyin):
#random_state, sözde rastgele sayı üretecini kontrol eder. Kodun tekrarlanabilirliği için bir random_state belirtilmelidir.
#shuffle: True ise, bölmeden önce verileri karıştırır
#stratify : dizi benzeri veya Yok (varsayılan Yoktur)
#Hiçbiri değilse, veriler, bunu labels dizisi olarak kullanarak katmanlara ayrılmış bir şekilde bölünür.
#Bu stratify parametresi, üretilen numunedeki değerlerin oranı, parametre stratify'a sağlanan değerlerin oranıyla aynı olacak şekilde bir bölme yapar.

#Örneğin, y değişkeni 0 ve 1 değerlerine sahip bir ikili kategorik değişkense ve
# %25 sıfır ve %75 birler varsa, stratify=y rastgele bölmenizin 
#0'ların %25'ini ve 1'lerin %75'ini olmasını sağlar.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size = 0.33, shuffle=True, random_state=42, stratify=labels)


#KNN sınıflandırma 
## classification
 #sınıflandırma için kütüphane eklendi
 #metrik olarak minkovski kullanıldı 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy = '  + str(accuracy_score(y_test, y_pred)))
from sklearn.metrics import r2_score
print(r2_score(y_pred, y_test))

import lpproj
dims = list(range(5,250,10))
ks = list(range(1,15))
knn_ks = list(range(1,15))

for dim in dims:
	for k in ks:
		for knn_k in knn_ks: 
			lppModel = lpproj.LocalityPreservingProjection(n_components = dim, n_neighbors = k)
			selfObject = lppModel.fit(X_train)
			trainKlpp = np.dot(X_train, selfObject.projection_)
			testKlpp = np.dot(X_test, selfObject.projection_)

			## classification
			# default=2 Minkowski metric
			from sklearn.neighbors import KNeighborsClassifier
			model = KNeighborsClassifier(n_neighbors = knn_k)
			model.fit(trainKlpp, y_train)
			y_pred = model.predict(testKlpp)
			print('Accuracy with LPP = '  + str(accuracy_score(y_test, y_pred)))







