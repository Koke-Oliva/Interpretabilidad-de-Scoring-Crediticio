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

## 🧰 Librerías utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=plotly&logoColor=white)
![seaborn](https://img.shields.io/badge/Seaborn-9A9A9A?logoColor=white)

**Componentes clave de scikit-learn usados:** `train_test_split`, `StandardScaler`,  
`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `classification_report`,  
`ConfusionMatrixDisplay`, `RocCurveDisplay`, `roc_auc_score`.
