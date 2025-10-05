# 🧠 Interpretabilidad de Scoring Crediticio

Modelado de **scoring crediticio** con pipeline claro y evaluación estándar.  
Comparo **Regresión Logística** (baseline y regularización) vs **Random Forest** con **GridSearchCV**.  
> El detalle completo (EDA, tuning, métricas, curvas) está en el **notebook**.

[![Ver Notebook](https://img.shields.io/badge/Ver%20Notebook-000000?logo=jupyter&logoColor=white)](./Interpretabilidad_de_Scoring_Crediticio.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Koke-Oliva/Interpretabilidad-de-Scoring-Crediticio/blob/main/Interpretabilidad_de_Scoring_Crediticio.ipynb)

---

## 🧾 Resumen (60s)
- **Objetivo:** predecir alto riesgo crediticio (binario) con métricas **Accuracy / Precision / Recall / F1 / ROC-AUC**.
- **Datos:** OpenML `credit` (v1). Variables numéricas de comportamiento/ingresos.  
- **Técnicas:** split estratificado, escalado con `StandardScaler` para continuas.  
- **Modelos:** Logistic Regression (L1/L2) y Random Forest (**GridSearchCV**, `cv=5`, `scoring=roc_auc`).  
- **Resultado:** el **Random Forest optimizado** entrega el mejor compromiso en **ROC-AUC** y **F1**.  
- **Próximo paso:** añadir **SHAP/LIME** para explicabilidad global/local y *threshold tuning*.

---

## 🧰 Stack (PyData)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=plotly&logoColor=white)
![seaborn](https://img.shields.io/badge/Seaborn-9A9A9A?logoColor=white)

**Módulos clave usados:** `train_test_split`, `StandardScaler`, `LogisticRegression`,  
`RandomForestClassifier`, `GridSearchCV`, `classification_report`, `ConfusionMatrixDisplay`, `RocCurveDisplay`.

---

## 🗃️ Dataset
- **Fuente:** OpenML — `credit` (v1).  
- **Target:** `SeriousDlqin2yrs` (1 = evento severo de morosidad en 2 años).

```python
from sklearn.datasets import fetch_openml
df = fetch_openml("credit", version=1, as_frame=True).frame
