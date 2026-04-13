import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams["figure.facecolor"] = "black"

# Cargar datos
df = pd.read_csv("Data Banking.csv")

# ---------------------------
# 1. Overview
# ---------------------------
print("Shape:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------
# 2. Identificar variables clave
# ---------------------------
print("\nColumnas:", df.columns.tolist())

# ---------------------------
# 3. Estadísticas generales
# ---------------------------
print(df.describe())

# ---------------------------
# 4. Distribuciones
# ---------------------------

plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"

df.hist(figsize=(12,10))
plt.suptitle("Distribución de variables")
plt.show()

# ---------------------------
# 5. Correlaciones
# ---------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlación")
plt.show()

# ================================
# FIX EDA - Pairplot seguro
# ================================

target = "Churned"  # ajustar si cambia

# detectar columnas numéricas automáticamente
features = df.select_dtypes(include=np.number).columns.tolist()

# remover target si está en features
if target in features:
    features.remove(target)

# usar solo algunas para no explotar el gráfico
features = features[:5]

print("Columnas reales:", df.columns.tolist())

sns.pairplot(
    df[features + [target]],
    hue=target,
    palette="Set2"
)

plt.suptitle("Relaciones entre variables clave", y=1.02)
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

from sklearn.cluster import KMeans

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,10), inertia, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sns.scatterplot(
    x="Account_Balance",
    y="Number_of_Transactions",
    hue="Cluster",
    data=df,
    palette="Set1"
)
plt.title("Segmentación de clientes")
plt.show()
plt.title("Segmentación de clientes")
plt.show()

cluster_profile = df.groupby("Cluster")[features].mean()
print(cluster_profile)

import seaborn as sns

sns.boxplot(x="Cluster", y="Account_Balance", data=df)
plt.title("Balance por Cluster")
plt.show()

sns.boxplot(x="Cluster", y="Number_of_Transactions", data=df)
plt.title("Estimated Salary por Cluster")   
plt.show()

df["CLV"] = (
    df["Account_Balance"] * 0.3 +
    df["Number_of_Transactions"] * 50 +
    df["Number_of_Products"] * 500
)

sns.boxplot(x="Cluster", y="CLV", data=df)
plt.title("Customer Lifetime Value por Cluster")    
plt.show()

print(df["CLV"].describe())

import seaborn as sns
sns.histplot(df["CLV"], bins=50)

plt.title("Distribución de Customer Lifetime Value")
plt.show()

clv_cluster = df.groupby("Cluster")["CLV"].mean()
print(clv_cluster)

df["ValueSegment"] = pd.qcut(df["CLV"], q=3, labels=["Low", "Medium", "High"])

sns.countplot(x="ValueSegment", hue="Churned", data=df)
plt.title("Segmentación de CLV vs Churn")   
plt.show()

sns.boxplot(x="Cluster", y="CLV", data=df)
plt.title("CLV por Cluster")
plt.show()

# Variable objetivo real
y = df["Churned"]

features = [
    "Account_Balance",
    "Number_of_Transactions",
    "Number_of_Products",
    "Credit_Score",
    "Age",
    "Tenure",
    "CLV",
    "Cluster"
]

from sklearn.model_selection import train_test_split

X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import xgboost as xgb
from sklearn.metrics import roc_auc_score

model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_proba)
print("AUC:", auc)

import matplotlib.pyplot as plt

xgb.plot_importance(model)
plt.show()

def calculate_profit_clv(y_true, y_pred, clv, contact_cost=20):

    tp = (y_true == 1) & (y_pred == 1)
    fp = (y_true == 0) & (y_pred == 1)

    profit = clv[tp].sum() - (contact_cost * fp.sum())

    return profit

thresholds = np.linspace(0, 1, 100)
profits = []

clv_test = X_test["CLV"].values

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    profit = calculate_profit_clv(y_test.values, y_pred, clv_test)
    profits.append(profit)

best_threshold = thresholds[np.argmax(profits)]
best_profit = max(profits)

print("Best Threshold:", best_threshold)
print("Max Profit:", best_profit)

import matplotlib.pyplot as plt

plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Profit ($)")
plt.title("Optimización de Revenue")
plt.axvline(best_threshold, linestyle="--")
plt.show()

# ================================
# BLOQUE 7 - Customer Targeting Strategy
# ================================

import pandas as pd
import numpy as np

# Probabilidades del modelo
y_proba = model.predict_proba(X_test)[:, 1]

# DataFrame base
df_targeting = X_test.copy()
df_targeting["y_true"] = y_test.values
df_targeting["proba"] = y_proba

# ================================
# 1. Expected Value por cliente
# ================================

BENEFIT_TP = 100   # ingreso por conversión
COST_CONTACT = 5   # costo contacto

df_targeting["expected_value"] = (
    df_targeting["proba"] * BENEFIT_TP - COST_CONTACT
)

# ================================
# 2. Ranking de clientes
# ================================

df_targeting = df_targeting.sort_values(by="expected_value", ascending=False)

# ================================
# 3. Selección top clientes (budget constraint)
# ================================

TOP_N = int(len(df_targeting) * 0.3)  # top 30%

df_targeting["target"] = 0
df_targeting.iloc[:TOP_N, df_targeting.columns.get_loc("target")] = 1

# ================================
# 4. Evaluación de estrategia
# ================================

targeted = df_targeting[df_targeting["target"] == 1]

tp = np.sum((targeted["y_true"] == 1))
fp = np.sum((targeted["y_true"] == 0))

total_profit = (tp * BENEFIT_TP) - (len(targeted) * COST_CONTACT)

print("Clientes contactados:", len(targeted))
print("True Positives:", tp)
print("False Positives:", fp)
print("Profit total:", total_profit)

BUDGET = 5000

df_targeting["cumulative_cost"] = np.arange(1, len(df_targeting)+1) * COST_CONTACT

df_targeting["target"] = (df_targeting["cumulative_cost"] <= BUDGET).astype(int)

df_targeting["segment"] = pd.qcut(df_targeting["proba"], 4, labels=["Low","Med","High","Top"])

df_targeting[df_targeting["target"] == 1].to_csv("campaign_targets.csv", index=False)

# ================================
# BLOQUE 8 - Explainability con SHAP
# ================================

import shap

# Crear explainer (TreeExplainer si usás XGBoost/LightGBM)
explainer = shap.Explainer(model, X_train)

# Calcular valores SHAP
shap_values = explainer(X_test)

# ================================
# 1. Feature Importance global
# ================================

shap.plots.bar(shap_values)

# ================================
# 2. Summary plot (impacto global)
# ================================

shap.plots.beeswarm(shap_values)

# ================================
# 3. Explicación individual
# ================================

# Elegimos un cliente
i = 0

shap.plots.waterfall(shap_values[i])

# ================================
# Top drivers por cliente
# ================================

def explain_customer(i):
    row = shap_values[i]
    feature_impact = pd.DataFrame({
        "feature": X_test.columns,
        "impact": row.values
    }).sort_values(by="impact", key=abs, ascending=False)
    
    return feature_impact.head(5)

# Ejemplo
explain_customer(0)

# Explicar SOLO clientes targeteados
# clientes targeteados
targeted_idx = df_targeting[df_targeting["target"] == 1].index

# convertir índices reales a posiciones
targeted_positions = [
    X_test.index.get_loc(idx)
    for idx in targeted_idx
]

# filtrar shap correctamente
shap_targeted = shap_values[targeted_positions]

# gráfico
shap.plots.beeswarm(shap_targeted)

# Ejemplo: impacto promedio por feature
mean_impact = np.abs(shap_values.values).mean(axis=0)

pd.DataFrame({
    "feature": X_test.columns,
    "importance": mean_impact
}).sort_values(by="importance", ascending=False)

# Export explicaciones top por cliente target
top_clients = df_targeting[df_targeting["target"] == 1].copy()

top_clients = df_targeting[df_targeting["target"] == 1].copy()

top_clients["top_driver"] = [
    explain_customer(X_test.index.get_loc(idx))["feature"].values[0]
    for idx in top_clients.index
]

top_clients.to_csv("target_with_explanations.csv", index=False)


# ================================
# BASELINE (TRAIN)
# ================================

train_dist = X_train.describe()

import numpy as np

def calculate_psi(expected, actual, bins=10):
    def scale_range(input, min_val, max_val):
        input = input.copy()
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        input = input * (max_val - min_val) + min_val
        return input

    breakpoints = np.linspace(0, 1, bins + 1)
    
    expected_percents = np.histogram(scale_range(expected, 0, 1), bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(scale_range(actual, 0, 1), bins=breakpoints)[0] / len(actual)

    psi = np.sum((expected_percents - actual_percents) * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6)))
    
    return psi

psi_values = {}

for col in X_train.columns:
    psi = calculate_psi(X_train[col], X_test[col])
    psi_values[col] = psi

psi_df = pd.DataFrame.from_dict(psi_values, orient='index', columns=['PSI'])
psi_df = psi_df.sort_values(by='PSI', ascending=False)

print(psi_df.head())

from sklearn.metrics import roc_auc_score

# Simulación: train vs test vs producción futura
auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

print("AUC Train:", auc_train)
print("AUC Test:", auc_test)

if auc_test < 0.75:
    print("⚠️ ALERTA: caída de performance del modelo")
    
performance_log = []

performance_log.append({
    "period": "2026-01",
    "auc": auc_test
})

perf_df = pd.DataFrame(performance_log)

# Si hay drift en features clave → revisar modelo

important_features = psi_df.head(5).index.tolist()

print("Features con mayor drift:", important_features)

# ================================
# BLOQUE 10 - MLflow Model Registry
# ================================

import mlflow
import mlflow.sklearn

mlflow.set_experiment("Customer_Intelligence_Project")

with mlflow.start_run():

    # Entrenar modelo (ya lo tenés, esto es conceptual)
    model.fit(X_train, y_train)

    # Log de métricas
    mlflow.log_metric("auc", auc_test)

    # Log del modelo
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="customer_conversion_model"
    )
    
# usar modelo actual entrenado como modelo productivo local
model_production = model

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Obtener última versión
latest_version = client.get_latest_versions(
    "customer_conversion_model", stages=["None"]
)[0].version

# Promover a producción
client.transition_model_version_stage(
    name="customer_conversion_model",
    version=latest_version,
    stage="Production"
)

def predict_customer(data, threshold=0.5):

    proba = model_production.predict_proba(data)[:, 1]

    decision = (proba >= threshold).astype(int)

    return pd.DataFrame({
        "proba": proba,
        "decision": decision
    })
    
sample = X_test.iloc[:5]

predict_customer(sample)

def predict_and_target(data, top_n=0.3):

    preds = predict_customer(data)
    df = data.copy()
    df["proba"] = preds["proba"]

    df = df.sort_values(by="proba", ascending=False)

    cutoff = int(len(df) * top_n)
    df["target"] = 0
    df.iloc[:cutoff, df.columns.get_loc("target")] = 1

    return df


