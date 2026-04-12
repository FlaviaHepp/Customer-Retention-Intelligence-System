# 🧠Customer Retention Intelligence System
End-to-End Machine Learning + Decision Engine + Generative AI

## 🚀Overview

Este proyecto desarrolla un sistema completo de predicción y optimización de churn en banca, evolucionando desde un modelo de Machine Learning hacia un motor inteligente de decisiones orientado a negocio.

A diferencia de enfoques tradicionales, el foco no está en predecir churn, sino en:

💥 maximizar el impacto económico de las estrategias de retención

## 🎯Problema de negocio

La pérdida de clientes genera impacto directo en ingresos.

La pregunta clave no es:

❌ “¿Quién se va?”

Sino:

✅ “¿A quién conviene retener, cómo y con qué estrategia?”

🧩 Solución desarrollada

El sistema integra múltiples capas:

## 🤖1. Machine Learning
Modelos: Logistic Regression, Random Forest, XGBoost
Evaluación con AUC (capacidad de ranking)
Feature engineering orientado a negocio

## 💰2. Profit Optimization
Definición de:
beneficio por cliente retenido
costo de contacto
Optimización de threshold para maximizar:

👉 Profit total de campaña

## 🧠3. Decision Engine

Sistema que transforma predicciones en acciones:

Segmentación de clientes
Estrategias diferenciadas:
clientes de alto valor
clientes inactivos
retención estándar

👉 El modelo deja de ser predictivo → pasa a ser accionable

## 🤖4. Generative AI Layer

Integración de IA generativa para:

Explicar churn en lenguaje natural
Generar acciones personalizadas de retención

👉 Conecta modelos con áreas de negocio

## ⚙️5. MLOps (MLflow)
Tracking de experimentos
Versionado de modelos
Registro de métricas:
AUC
Profit
Threshold óptimo

## 🌐6. Deployment (FastAPI)
API REST para predicción en tiempo real
Endpoint avanzado con:
probabilidad
explicación
acción recomendada

## 📊7. Business Simulation
Simulación de campañas de retención
Estimación de:
clientes contactados
clientes retenidos
tasa de conversión

## 🧪8. A/B Testing
Comparación entre:
estrategia random
estrategia basada en modelo

👉 Medición de uplift real

## 📉9. Monitoring & Risk
Detección de drift
Logging de predicciones
Evaluación de fairness (sesgos)

## 💳10. Customer Value Integration
Incorporación de CLV (Customer Lifetime Value)
Priorización de clientes de alto valor económico

## 🔍11. Explainability (SHAP)
Interpretación de variables
Explicabilidad del modelo

## 🏗️Arquitectura
Data → Feature Engineering → Model → Profit Optimization  
→ Decision Engine → API → Monitoring → Business Impact

## 🧪Stack tecnológico
Python
Pandas / NumPy
Scikit-learn
XGBoost
MLflow
FastAPI
SHAP
OpenAI (opcional)
💥 Diferencial clave

Este proyecto no se limita a Machine Learning.

👉 Integra:

Data Science
Business Strategy
MLOps
Generative AI

Para construir un sistema que:

toma decisiones optimizadas y accionables

## 📈Resultados
Modelo con alta capacidad de discriminación (AUC)
Optimización de campañas de retención
Mejora en ROI mediante selección inteligente de clientes
Sistema listo para integración en entornos productivos

## 🚀Cómo ejecutar
pip install -r requirements.txt
python train.py
mlflow ui
uvicorn app:app --reload

## 🧠Autor

Proyecto desarrollado con foco en aplicaciones reales de Machine Learning en banca, combinando analítica avanzada, MLOps e inteligencia artificial aplicada.
