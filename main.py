

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# EXTRACT
df = pd.read_csv("train.csv")

print("Información inicial del dataset:")
df.info()

print("\nValores vacíos por columna:")
print(df.isna().sum())
print("\nRegistros duplicados:")
print(df.duplicated().sum())

print("\nDescripción de variables categóricas:")
print(df.describe(include='object'))


# TRANSFORM
categorical_cols = [
    'location',
    'subscription_type',
    'payment_plan',
    'num_subscription_pauses',
    'payment_method',
    'customer_service_inquiries'
]
numeric_cols = [
    'age',
    'weekly_hours',
    'average_session_length',
    'song_skip_rate',
    'weekly_songs_played',
    'weekly_unique_songs',
    'num_favorite_artists',
    'num_platform_friends',
    'num_playlists_created',
    'num_shared_playlists',
    'notifications_clicked'
]

for i in categorical_cols:
    print(f"\nDescripción sobre {i}")
    print(round(df[i].value_counts(normalize=True) * 100, 2))

df["location"] = df["location"].replace("Nebrasksa", "Nebraska")
df = df.set_index(['customer_id'], drop=True)
df['signup_date'] = pd.to_datetime(df['signup_date'])

print("\nDescripción de variables numéricas:")
print(df.describe())


# EDA
print("\nDistribución de churn:")
print(df['churned'].value_counts())
print(round(df['churned'].value_counts(normalize=True) * 100, 2))

sns.set_theme(style='whitegrid')
for col in categorical_cols:
    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.title(f'{col}')
    plt.xlabel(col)
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(7, 3))

    sns.boxplot(
        data=df,
        x=col
    )
    plt.title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.show()

temp = df.select_dtypes(include='number')
plt.figure(figsize=(12, 10))
sns.heatmap(temp.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.tight_layout()
plt.show()


# One Hot Encoding

encoding = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

#Definimos las variables a usar en el modelo
X = encoding.drop(
    columns=[
        'churned',
        'signup_date'
    ]
)
y = encoding['churned']

def split_train_test(X, y, porcentaje_test=0.3):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    cantidad_test = int(len(X) * porcentaje_test)
    cantidad_train = len(X) - cantidad_test

    for i in range(len(X)):
        if i < cantidad_train:
            X_train.append(X.iloc[i])
            y_train.append(y.iloc[i])
        else:
            X_test.append(X.iloc[i])
            y_test.append(y.iloc[i])

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test)
    )


X_train, X_test, y_train, y_test = split_train_test(X, y)


# Regresión logística

class Reg_logis:
    def __init__(self, alpha=0.001, iteracion=100):
        self.alpha = alpha
        self.iter = iteracion
        self.w = None
        self.costo = []

    def sigmoidal(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, sample):
        acum = 0
        for i in range(len(self.w)):
            acum += self.w[i] * sample[i]
        return self.sigmoidal(acum)

    def scaling_fit(self, samples):
        samples = np.asarray(samples, dtype=float).T.tolist()

        self.medias = []
        self.desviaciones = []

        for i in range(1, len(samples)):
            acum = 0
            for j in range(len(samples[i])):
                acum += samples[i][j]

            avg = acum / len(samples[i])

            acum = 0
            for j in range(len(samples[i])):
                acum += (samples[i][j] - avg) ** 2

            std = (acum / len(samples[i])) ** 0.5

            if std == 0:
                std = 1

            self.medias.append(avg)
            self.desviaciones.append(std)

            for j in range(len(samples[i])):
                samples[i][j] = (samples[i][j] - avg) / std

        return np.asarray(samples).T.tolist()

    def scaling_predict(self, samples):
        samples = np.asarray(samples, dtype=float).T.tolist()

        for i in range(1, len(samples)):
            avg = self.medias[i - 1]
            std = self.desviaciones[i - 1]

            for j in range(len(samples[i])):
                samples[i][j] = (samples[i][j] - avg) / std

        return np.asarray(samples).T.tolist()

    def cross_entropy(self, y, y_pred):
        eps = 1e-9
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        y1 = y * np.log(y_pred + eps)
        y2 = (1 - y) * np.log(1 - y_pred + eps)
        return -np.mean(y1 + y2)

    def GD(self, X, y):
        errores = []

        for i in range(len(X)):
            error = self.h(X[i]) - y[i]
            errores.append(error)

        temp = list(self.w)

        # Actuailzar cada peso
        for j in range(len(self.w)):
            acum = 0

            for i in range(len(X)):
                acum += errores[i] * X[i][j]

            temp[j] = self.w[j] - self.alpha * (1 / len(X)) * acum

        return temp

    def fit(self, X, y):
        X_train = X.copy()

        for i in range(len(X_train)):
            X_train[i] = [1] + X_train[i]

        X_train = self.scaling_fit(X_train)
        self.w = [0] * len(X_train[0])

        for i in range(self.iter):
            w_anterior = list(self.w)
            self.w = self.GD(X_train, y)
            y_pred = []

            for sample in X_train:
                y_pred.append(self.h(sample))

            error = self.cross_entropy(y, y_pred)
            self.costo.append(error)

            if w_anterior == self.w or error < 0.0001:
                break

    def predict(self, X, y):
        X_test = X.copy()
        y_test = y.copy()

        # Bias
        for i in range(len(X_test)):
            X_test[i] = [1] + X_test[i]

        X_test = self.scaling_predict(X_test)

        predicciones = []

        for sample in X_test:
            prob = self.h(sample)

            if prob >= 0.5:
                predicciones.append(1)
            else:
                predicciones.append(0)

        probabilidades = []

        for sample in X_test:
            prob = self.h(sample)
            probabilidades.append(prob)

        print("\nProbabilidad mínima:", round(min(probabilidades), 4))
        print("Probabilidad máxima:", round(max(probabilidades), 4))
        print("Probabilidad promedio:", round(sum(probabilidades) / len(probabilidades), 4))

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(y_test)):
            real = y_test[i]
            pred = predicciones[i]

            if real == 1 and pred == 1:
                TP += 1
            elif real == 0 and pred == 0:
                TN += 1
            elif real == 0 and pred == 1:
                FP += 1
            elif real == 1 and pred == 0:
                FN += 1

        recall = TP / (TP + FN)
        accuracy = (TP + TN) / len(y_test)
        # precision = TP / (TP + FP)

        print("True Positive:", round(TP, 4))
        print("True Negative:", round(TN, 4))
        print("Fasle Positive:", round(FP, 4))
        print("False Negative:", round(FN, 4))
        print("Recall clase 1:", round(recall, 4))
        print("Accuracy clase 1:", round(accuracy, 4))

        return predicciones



modelo = Reg_logis()
modelo.fit(X_train, y_train)

plt.figure(figsize=(8, 4))
plt.plot(modelo.costo)
plt.xlabel("Iteración")
plt.ylabel("Cross Entropy")
plt.title("Convergencia del modelo")
plt.tight_layout()

modelo.predict(X_test, y_test)

plt.show()
