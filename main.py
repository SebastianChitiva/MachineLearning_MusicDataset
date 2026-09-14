
#Librerías
#Exploración, transformación y visualización de datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Segunda parte de proyecto (Modelo mejorado)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, roc_auc_score, auc, average_precision_score)

# Abrir el archivo
df = pd.read_csv("train.csv")

df.info()

#Busqueda de valores vacíos y duplicados
print(df.isna().sum())
print("")
print(df.duplicated().sum())

#Descripción categórica
df.describe(include='object')

#Columnas Categóricas
categorical_cols = [
    'location',
    'subscription_type',
    'payment_plan',
    'num_subscription_pauses',
    'payment_method',
    'customer_service_inquiries'
]

#Columnas Numéricas
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

#Análisis de error gramatical en variables categóricas
for i in categorical_cols:
  print(f"Descripción sobre")
  print(round(df[i].value_counts(normalize=True) * 100,2))
  print("")

#Transform
#Corregir el texto de "Nebrasksa" a "Nebraska"
df["location"] = df["location"].replace("Nebrasksa", "Nebraska")

#Convertir ID de Cliente como Index
df = df.set_index(['customer_id'],drop=True)

# Convertir la columna "Signup Date" en fecha
df['signup_date'] = pd.to_datetime(df['signup_date'])

#Descripción numérica
df.describe()



# Exploratorio (EDA)
#Análisis de porcentaje de clientes que abandonaron
print(df['churned'].value_counts())
print(round(df['churned'].value_counts(normalize=True) * 100,2))

#Gráficas Numéricas
for col in numeric_cols:

    plt.figure(figsize=(7, 3))

    sns.boxplot(
        data=df,
        x=col
    )

    plt.title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.show()

#Matriz de correlación
temp =df.select_dtypes(include='number')
plt.figure(figsize=(10,10))
sns.heatmap(temp.corr(),annot=True,cmap='coolwarm',fmt='.2f')

# One-Hot Encoding

encoding = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

X = encoding.drop(
    columns=[
        'churned',
        'signup_date',
        'weekly_songs_played',
        'weekly_unique_songs',
        'num_favorite_artists'
    ])

y = encoding['churned']

def split_train_val_test(X, y, train=0.7, val=0.1,
                         test=0.2, random_state=42):

    # Convertir a arrays
    X_array = np.asarray(X)
    y_array = np.asarray(y)


    # Crear índices y mezclarlos
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X_array))
    rng.shuffle(indices)

    # Reordenar datos
    X_array = X_array[indices]
    y_array = y_array[indices]

    # Cantidades
    n = len(X_array)
    cantidad_train = int(n * train)
    cantidad_val = int(n * val)

    # Separación
    X_train = X_array[:cantidad_train]
    y_train = y_array[:cantidad_train]

    X_val = X_array[cantidad_train:cantidad_train + cantidad_val]
    y_val = y_array[cantidad_train:cantidad_train + cantidad_val]

    X_test = X_array[cantidad_train + cantidad_val:]
    y_test = y_array[cantidad_train + cantidad_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

#Función para determinar el porcentaje por clase en cada set
def revisar_distribucion(y_train, y_val, y_test):

    conjuntos = {
        "Train": y_train,
        "Validation": y_val,
        "Test": y_test
    }

    for nombre, y_set in conjuntos.items():

        conteo = pd.Series(y_set).value_counts().sort_index()
        porcentaje = pd.Series(y_set).value_counts(normalize=True).sort_index() * 100

        print(nombre)
        print(f"\n Total de observaciones: {len(y_set)}")

        for clase in [0, 1]:
            print(
                f"Clase {clase}: ({round(porcentaje.get(clase, 0),2)}%)")

revisar_distribucion(y_train, y_val, y_test)

#Objeto de regresión Logística
class Reg_logis:
  def __init__(self, alpha = 0.001, iteracion=100):
    self.alpha = alpha
    self.iter = iteracion
    self.w = None
    self.costo = []

  def sigmoidal(self,z):
    return 1 / (1+np.exp(-z))

  def h(self,sample):
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
        std=1

      self.medias.append(avg)
      self.desviaciones.append(std)

      for j in range(len(samples[i])):
        samples[i][j] = (samples[i][j] - avg) / std

    return np.asarray(samples).T.tolist()


  def scaling_predict(self, samples):

    samples = np.asarray(samples, dtype=float).T.tolist()

    for i in range(1, len(samples)):

      avg = self.medias[i-1]
      std = self.desviaciones[i-1]

      for j in range(len(samples[i])):
        samples[i][j] = (samples[i][j] - avg) / std

    return np.asarray(samples).T.tolist()

  def cross_entropy(self, y, y_pred):
    eps = 1e-9
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    y1 = y * np.log(y_pred + eps)
    y2 = (1-y) * np.log(1 - y_pred + eps)
    return -np.mean(y1+y2)

  def GD(self, X, y):

    errores = []
    for i in range(len(X)):
      error = self.h(X[i]) - y[i]
      errores.append(error)

    temp = list(self.w)

    # Actualizar cada peso
    for j in range(len(self.w)):
      acum = 0

      for i in range(len(X)):
        acum += errores[i] * X[i][j]

      temp[j] = self.w[j] - self.alpha * (1/len(X)) * acum

    return temp

  def fit(self, X, y):
    X_train = X.copy()
    for i in range(len(X_train)):
      X_train[i] = [1] + X_train[i]

    X_train = self.scaling_fit(X_train)
    self.w = [0] * len(X_train[0])

    # Gradient Descent
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


  def predict_proba(self, X):

    X_pred = X.copy()

    # Bias
    for i in range(len(X_pred)):
      X_pred[i] = [1] + X_pred[i]

    # Escalamiento usando media y desviación de X_train
    X_pred = self.scaling_predict(X_pred)

    probabilidades = []

    for sample in X_pred:
      prob = self.h(sample)
      probabilidades.append(prob)

    return np.asarray(probabilidades)


  def roc_manual(self, X, y):

    probabilidades = self.predict_proba(X)
    umbrales = np.sort(np.unique(probabilidades))[::-1]

    tpr = []
    fpr = []

    for umbral in umbrales:

      predicciones = (probabilidades >= umbral).astype(int)

      TP = np.sum((y == 1) & (predicciones == 1))
      TN = np.sum((y == 0) & (predicciones == 0))
      FP = np.sum((y == 0) & (predicciones == 1))
      FN = np.sum((y == 1) & (predicciones == 0))

      recall = TP / (TP + FN) if (TP + FN) > 0 else 0
      especifidad = TN / (TN + FP) if (TN + FP) > 0 else 0

      tpr.append(recall)
      fpr.append(1 - especifidad)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    orden = np.argsort(fpr)
    fpr = fpr[orden]
    tpr = tpr[orden]

    auc = 0
    for i in range(len(fpr) - 1):

      base = fpr[i+1] - fpr[i]
      altura = (tpr[i] + tpr[i+1]) / 2
      auc += base * altura

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}"
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificador aleatorio"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()
    return fpr, tpr, auc

  def pr_auc_manual(self, X, y):

    probabilidades = self.predict_proba(X)
    umbrales = np.sort(np.unique(probabilidades))[::-1]

    precision = []
    recall = []

    for umbral in umbrales:

      predicciones = (probabilidades >= umbral).astype(int)

      TP = np.sum((y == 1) & (predicciones == 1))
      FP = np.sum((y == 0) & (predicciones == 1))
      FN = np.sum((y == 1) & (predicciones == 0))

      p = TP / (TP + FP) if (TP + FP) > 0 else 1
      r = TP / (TP + FN) if (TP + FN) > 0 else 0

      precision.append(p)
      recall.append(r)

    precision = np.array(precision)
    recall = np.array(recall)

    orden = np.argsort(recall)
    recall = recall[orden]
    precision = precision[orden]

    pr_auc = 0

    for i in range(1, len(recall)):
      if recall[i] > recall[i - 1]:
        pr_auc += precision[i] * (recall[i] - recall[i - 1])


    print(f"PR-AUC: {round(pr_auc,4)}%")
    return pr_auc


  def predict(self, X, y=None):
    probabilidades = self.predict_proba(X)
    predicciones = []
    tpr = []
    fpr = []

    for probabilidad in probabilidades:
      if probabilidad >= 0.5:
        predicciones.append(1)
      else:
        predicciones.append(0)

    if y is not None:

      TP = 0
      TN = 0
      FP = 0
      FN = 0

    for i in range(len(y)):

      real = y[i]
      pred = predicciones[i]

      if real == 1 and pred == 1:
          TP += 1
      elif real == 0 and pred == 0:
          TN += 1
      elif real == 0 and pred == 1:
          FP += 1
      elif real == 1 and pred == 0:
          FN += 1

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / len(y)
    precision = (TP) / (TP + FP) if (TP + FP) > 0 else 0
    F1 = (2*recall * precision)/ (recall + precision) if (recall + precision) > 0 else 0

    print("\n Predict de Regresión Logística a Mano")
    print("Probabilidad mínima:", round(probabilidades.min(), 4))
    print("Probabilidad máxima:", round(probabilidades.max(), 4))
    print("Probabilidad promedio:", round(probabilidades.mean(), 4))

    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)

    print("Recall clase 1:", round(recall, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precisión:", round(precision, 4))
    print("F1 Score:", round(F1, 4))

    return np.asarray(predicciones)

# Modelo de Regresión Logística a Mano
modelo = Reg_logis()
modelo.fit(X_train,y_train)

y_hat1 = modelo.predict(X_val,y_val)
modelo.roc_manual(X_val,y_val)
modelo.pr_auc_manual(X_val,y_val)

plt.figure(figsize=(10,10))
plt.plot(modelo.costo)
plt.xlabel("Iteración")
plt.ylabel("Cross Entropy")
plt.title("Convergencia del modelo")
plt.show()

#Gráfico para observar probabilidad de set "Validation"
y_prob = modelo.predict_proba(X_val)

plt.figure(figsize=(8, 6))
sns.boxplot(
    x=y_val,
    y=y_prob
)

plt.axhline(0.5, color="black", linestyle="--",label="Límite = 0.5")

plt.xlabel("Clase real")
plt.ylabel("Probabilidad de pertenecer a clase 1")
plt.title("Probabilidad predicha vs clase real")
plt.legend()

plt.show()

# Scikit Learn Regresión Logística
# Entrenamiento de regresión con Scikit Learn

# Split 70/10/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.125,
    random_state=42,
    stratify=y_train
)

#Escalar datos
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_val_scaled =scaler.transform(X_val)
X_test_scaled =scaler.transform(X_test)

#Modelo de Scikit Learn
model = LogisticRegression(solver = 'lbfgs',
                           penalty='none',
                           max_iter=300,
                           class_weight='balanced')
model.fit(X_train_scaled, y_train)

def evaluar(y_real, y_pred, nombre):
  print(f"\n {nombre}")
  print("Accuracy:", round(accuracy_score(y_real, y_pred),4))
  print("Recall:", round(recall_score(y_real, y_pred),4))
  print("Precision:", round(precision_score(y_real, y_pred),4))
  print("F1:", round(f1_score(y_real, y_pred),4))
  print("PR-AUC:", round(average_precision_score(y_real, y_pred),4))
  print("Matriz de confusión:")
  print(confusion_matrix(y_real, y_pred))

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

evaluar(y_train, y_train_pred, "Train")
evaluar(y_val, y_val_pred, "Val")
evaluar(y_test, y_test_pred, "Test")

y_val_prob= model.predict_proba(X_val_scaled)[:, 1]

#AUC-ROC
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_val, y_val_prob)
auc_sklearn = auc(fpr_sklearn, tpr_sklearn)

plt.plot(fpr_sklearn, tpr_sklearn, label=f"AUC = {auc_sklearn:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Validation")
plt.legend()
plt.grid()

plt.show()

#Implementación de XGBoost Classifier
clf_1 = XGBClassifier(
  tree_method="hist",
  n_estimators=300,
  learning_rate=0.05,
  max_depth=4,
  random_state=94,
  eval_metric="aucpr"
)

clf_1.fit(
  X_train,
  y_train,
  eval_set=[(X_val, y_val)],
  verbose=False
)

y_val_pred_1 = clf_1.predict(X_val)
y_val_prob_1 = clf_1.predict_proba(X_val)[:, 1]
print("\n Modelo XGBoost 1")
print(f"Accuracy: {round(accuracy_score(y_val, y_val_pred_1),4)}")
print(f"Precision: {round(precision_score(y_val, y_val_pred_1),4)}")
print(f"Recall: {round(recall_score(y_val, y_val_pred_1),4)}")
print(f"F1: {round(f1_score(y_val, y_val_pred_1),4)}")
print(f"ROC-AUC: {round(roc_auc_score(y_val, y_val_prob_1),4)}")
print(f"PR-AUC: {round(average_precision_score(y_val, y_val_prob_1),4)}")

clf_2 = XGBClassifier(
  tree_method="hist",
  n_estimators=300,
  learning_rate=0.05,
  max_depth=5,
  scale_pos_weight=3.72,
  random_state=94,
  eval_metric="aucpr"
)

clf_2.fit(
  X_train,
  y_train,
  eval_set=[(X_val, y_val)],
  verbose=False
)
y_val_pred_2 = clf_2.predict(X_val)
y_val_prob_2 = clf_2.predict_proba(X_val)[:, 1]

print("\n Modelo XGBoost 2")
print(f"Accuracy: {round(accuracy_score(y_val, y_val_pred_2),4)}")
print(f"Precision: {round(precision_score(y_val, y_val_pred_2),4)}")
print(f"Recall: {round(recall_score(y_val, y_val_pred_2),4)}")
print(f"F1: {round(f1_score(y_val, y_val_pred_2),4)}")
print(f"ROC-AUC: {round(roc_auc_score(y_val, y_val_prob_2),4)}")
print(f"PR-AUC: {round(average_precision_score(y_val, y_val_prob_2),4)}")

clf_3 = XGBClassifier(
  tree_method="hist",
  n_estimators=300,
  learning_rate=0.03,
  max_depth=3,
  scale_pos_weight=3.72,
  eval_metric="aucpr"
)


clf_3.fit(
  X_train,
  y_train,
  eval_set=[(X_val, y_val)],
  verbose=False
)

y_val_pred_3 = clf_3.predict(X_val)
y_val_prob_3 = clf_3.predict_proba(X_val)[:, 1]

print("\n Modelo XGBoost 3")
print(f"Accuracy: {round(accuracy_score(y_val, y_val_pred_3),4)}")
print(f"Precision: {round(precision_score(y_val, y_val_pred_3),4)}")
print(f"Recall: {round(recall_score(y_val, y_val_pred_3),4)}")
print(f"F1: {round(f1_score(y_val, y_val_pred_3),4)}")
print(f"ROC-AUC: {round(roc_auc_score(y_val, y_val_prob_3),4)}")
print(f"PR-AUC: {round(average_precision_score(y_val, y_val_prob_3),4)}")

clf_4 = XGBClassifier(
  tree_method="hist",
  n_estimators=1000,
  learning_rate=0.03,
  max_depth=5,
  scale_pos_weight=3.72,
  eval_metric="aucpr"
)

clf_4.fit(
  X_train,
  y_train,
  eval_set=[(X_val, y_val)],
  verbose=False
)

y_val_pred_4 = clf_4.predict(X_val)
y_val_prob_4 = clf_4.predict_proba(X_val)[:, 1]

print("\n Modelo XGBoost 4")
print(f"Accuracy: {round(accuracy_score(y_val, y_val_pred_4),4)}")
print(f"Precision: {round(precision_score(y_val, y_val_pred_4),4)}")
print(f"Recall: {round(recall_score(y_val, y_val_pred_4),4)}")
print(f"F1: {round(f1_score(y_val, y_val_pred_4),4)}")
print(f"ROC-AUC: {round(roc_auc_score(y_val, y_val_prob_4),4)}")
print(f"PR-AUC: {round(average_precision_score(y_val, y_val_prob_4),4)}")

#Búsqueda de threshold con modelo 3
y_prob = clf_3.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.40, 0.65, 0.01)
resultados = []

for threshold in thresholds:

  y_pred = (y_prob >= threshold).astype(int)

  precision = precision_score(y_val, y_pred, zero_division=0)
  recall = recall_score(y_val, y_pred, zero_division=0)
  f1 = f1_score(y_val, y_pred, zero_division=0)

  tn, fp, fn, tp = confusion_matrix(
    y_val,
    y_pred
  ).ravel()


  resultados.append({
    "Threshold": threshold,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
  })


resultados_threshold = pd.DataFrame(resultados)

a = resultados_threshold.sort_values(
  by="F1",
  ascending=False
)
print(a.head(30))

y_train_prob = clf_3.predict_proba(X_train)[:, 1]
y_val_prob = clf_3.predict_proba(X_val)[:, 1]
y_test_prob = clf_3.predict_proba(X_test)[:, 1]

threshold = 0.56
y_train_pred = (y_train_prob >= threshold).astype(int)
y_val_pred = (y_val_prob >= threshold).astype(int)
y_test_pred = (y_test_prob >= threshold).astype(int)

def evaluar(y_real, y_pred, y_prob, nombre):
  print(f"\n{nombre}")

  print("Accuracy:", round(accuracy_score(y_real, y_pred), 4))
  print("Recall:", round(recall_score(y_real, y_pred), 4))
  print("Precision:", round(precision_score(y_real, y_pred), 4))
  print("F1:", round(f1_score(y_real, y_pred), 4))
  print("PR-AUC:", round(average_precision_score(y_real, y_prob), 4))
  print("ROC-AUC:", round(roc_auc_score(y_real, y_prob), 4))
  print("Matriz de confusión:")
  print(confusion_matrix(y_real, y_pred))


# Evaluar validación
evaluar(
  y_val,
  y_val_pred,
  y_val_prob,
  "Validation"
)

#Evaluar Test
evaluar(
  y_test,
  y_test_pred,
  y_test_prob,
  "Test"
)

# Gráfico de probabilidades
fig, ax = plt.subplots(figsize=(12, 10))

# Train
ax.scatter(np.arange(len(y_train_prob)), y_train_prob,c=y_train,
  cmap="plasma",alpha=0.6,marker="o",label="Train")

# Validacion
ax.scatter(
  np.arange(
    len(y_train_prob),
    len(y_train_prob) + len(y_val_prob)
  ), y_val_prob, c=y_val, cmap="plasma",
    alpha=0.6, marker="x", label="Validation")

# Test
ax.scatter(
  np.arange(
    len(y_train_prob) + len(y_val_prob),
    len(y_train_prob) + len(y_val_prob) + len(y_test_prob)
  ), y_test_prob, c=y_test, cmap="plasma",
    alpha=0.6, marker="^", label="Test")

#Umbral
ax.axhline(threshold, color="black", linestyle="--",
  linewidth=2, label=f"Umbral de Decisión = {threshold}")

ax.set_xlabel("Muestra")
ax.set_ylabel("Probabilidad de modelo")
ax.set_title("Predicciones de Modelo XGBoost")
ax.legend()
plt.tight_layout()
plt.show()

metricas = {
  "Accuracy": [
    accuracy_score(y_train, y_train_pred),
    accuracy_score(y_val, y_val_pred),
    accuracy_score(y_test, y_test_pred)
  ],
  "Precision": [
    precision_score(y_train, y_train_pred),
    precision_score(y_val, y_val_pred),
    precision_score(y_test, y_test_pred)
  ],
  "Recall": [
    recall_score(y_train, y_train_pred),
    recall_score(y_val, y_val_pred),
    recall_score(y_test, y_test_pred)
  ],
  "F1": [
    f1_score(y_train, y_train_pred),
    f1_score(y_val, y_val_pred),
    f1_score(y_test, y_test_pred)
  ],
  "PR-AUC": [
    average_precision_score(y_train, y_train_prob),
    average_precision_score(y_val, y_val_prob),
    average_precision_score(y_test, y_test_prob)
  ],
  "ROC-AUC": [
    roc_auc_score(y_train, y_train_prob),
    roc_auc_score(y_val, y_val_prob),
    roc_auc_score(y_test, y_test_prob)
  ]
}


x = np.arange(len(metricas))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4))

# Train
ax.bar(
    x - width,
    [v[0] for v in metricas.values()],
    width,
    label="Train"
)

# Validacion
ax.bar(
    x,
    [v[1] for v in metricas.values()],
    width,
    label="Validation"
)

# Test
ax.bar(
    x + width,
    [v[2] for v in metricas.values()],
    width,
    label="Test"
)

ax.set_xticks(x)
ax.set_xticklabels(metricas.keys())
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Evaluación de XGBoost")
ax.legend()

plt.tight_layout()
plt.show()
