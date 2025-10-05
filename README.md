# 🧠 Interpretabilidad de Scoring Crediticio

Proyecto de **modelado para scoring crediticio** con un pipeline claro y resultados evaluables.  
Se entrenan y comparan **Regresión Logística** (baseline y con regularización) y **Random Forest** con **GridSearchCV**.  


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Koke-Oliva/Interpretabilidad-de-Scoring-Crediticio/blob/main/Interpretabilidad_de_Scoring_Crediticio.ipynb)

---

## 🎯 Objetivo
Predecir si un cliente tendrá **alto riesgo crediticio** (clasificación binaria) y evaluar el desempeño con **Accuracy, Precision, Recall, F1 y ROC-AUC**.

---

## 🗃️ Dataset
- Fuente: **OpenML** (`credit`, versión 1), con variables numéricas típicas de riesgo.
- Objetivo: **`SeriousDlqin2yrs`** (1 = evento severo de morosidad en 2 años).
- Carga en el notebook:
```python
from sklearn.datasets import fetch_openml
df = fetch_openml("credit", version=1, as_frame=True).frame
