from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Last inn MNIST-datasettet
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist.data, mnist.target

# Skaler pikslene til verdier mellom 0 og 1
X = X / 255.0

# Del datasettet inn i opplærings- og testsett
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the size of the training and testing data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Velg en rekke verdier for k
k_values = list(range(1,10))

# Opprett lister for å lagre valideringsresultatene
validation_scores = []

# Utfør kryssvalidering for hvert k
for k in k_values:
    print(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5-fold kryssvalidering
    print(scores)
    mean_score = scores.mean()
    validation_scores.append(mean_score)


# Finn den beste k basert på valideringsresultatene
best_k = k_values[validation_scores.index(max(validation_scores))]

print(f'Best k: {best_k}')

# Opprett en KNeighborsClassifier med beste k
knn = KNeighborsClassifier(n_neighbors=best_k)

# Tren modellen
knn.fit(X_train, y_train)

# Gjør prediksjoner på testsettet
y_pred = knn.predict(X_test)

# Evaluer nøyaktigheten
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')