import pickle, pandas, numpy

with open('./dataset/TestData.pkl', 'rb') as file:
    X_test, y_test = pickle.load(file)

from sklearn.decomposition import PCA
pca = PCA(3)
X_norm = (X_test - X_test.min()) / (X_test.max() - X_test.min())
transformed = pandas.DataFrame(pca.fit_transform(X_norm))

from tensorflow.keras.models import load_model
model = load_model('./models/BiLSTM/')

pred = model.predict(X_test)
result = []
for i in range(len(pred)):
    if pred[i]>0.5:
        result.append(1)
    else:
        result.append(0)

import matplotlib.pyplot as plt
plt.scatter(
    transformed[result==0][0],
    transformed[result==0][1],
    transformed[result==0][2],
    label='Fake', c='red')

plt.scatter(
    transformed[result==1][0],
    transformed[result==1][1],
    transformed[result==1][2],
    label='Real', c='blue')
plt.show()