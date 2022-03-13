from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import glob

def build_dataset():
	org_dataset = []
	labels = []
	for i in range(1, 16):
		filelist = glob.glob('./data/subject'+str(i).zfill(2)+"*")
		for fname in filelist:
			img = Image.open(fname)
			img = np.array(img.resize((32, 32), Image.ANTIALIAS))
			img = img.reshape(img.shape[0] * img.shape[1])
			org_dataset.append(img)
			labels.append(i)
	return np.array(org_dataset), np.array(labels)

data, labels = build_dataset()

## Normalize
data = data/255
print(len(data))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, shuffle=True, random_state=42, stratify=labels)

## classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy = '  + str(accuracy_score(y_test, y_pred)))


## KLPP
from KLPP import constructWKLPP, KLPP, constructKernelKLPP
gnd = y_train #98
options={}
#options['NeighborMode'] = 'Supervised' #danışmanlı
options['NeighborMode'] = 'KNN'  #danışmansız
options['gnd'] = gnd
options['WeightMode'] = 'HeatKernel'
#options['bLDA'] = 1
#options['bNormalized'] = 1
options['t'] = 20
options['reducedDim'] = 35
options['k'] = 3
W = constructWKLPP(X_train, options)
options['KernelType'] = 'Gaussian'
options['Regu'] = 1
options['ReguAlpha'] = 0.001
eigvector, eigvalue = KLPP(W, options, X_train)
kTrain = constructKernelKLPP(X_train, [], options)
trainKlpp = np.dot(kTrain,eigvector)
kTest = constructKernelKLPP(X_test, X_train, options)
testKlpp = np.dot(kTest,eigvector)


# ## LPP
# from LPP import constructLPP, LPP
# gnd = y_train #98
# options={}
# options['NeighborMode'] = 'Supervised' #danışmanlı
# #options['NeighborMode'] = 'KNN'  #danışmansız
# options['gnd'] = gnd
# options['WeightMode'] = 'HeatKernel'
# #options['bLDA'] = 1
# #options['bNormalized'] = 1
# options['t'] = 20
# options['reducedDim'] = 35
# options['k'] = 125
# W = constructWKLPP(X_train, options)
# options['ReguAlpha'] = 0.001
# eigvector, eigvalue = LPP(W, options, X_train)
# trainKlpp = np.dot(kTrain,eigvector)
# testKlpp = np.dot(kTest,eigvector)





# ## LLE
# from sklearn.manifold import LocallyLinearEmbedding
# embedding = LocallyLinearEmbedding(n_neighbors = 40, n_components=108)
# trainKlpp = embedding.fit_transform(X_train)
# testKlpp = embedding.transform(X_test)





## classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(trainKlpp, y_train)
y_pred = model.predict(testKlpp)
print('Accuracy with KLPP = '  + str(accuracy_score(y_test, y_pred)))





