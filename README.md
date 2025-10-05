# 🧠 Interpretabilidad de Scoring Crediticio

Proyecto de **modelado para scoring crediticio** con foco en un pipeline limpio y resultados evaluables.  
Se entrenan y comparan **Regresión Logística (con y sin regularización)** y **Random Forest**.  
> En esta versión **no** se usan aún técnicas de interpretabilidad global/local (SHAP/LIME). Se dejan en “Trabajo futuro”.

---

## 🎯 Objetivo
Predecir si un cliente tendrá **alto riesgo crediticio** (binario) y evaluar el desempeño con métricas estándar (Accuracy, Precision, Recall, F1, ROC-AUC).

---

## 🗃️ Dataset
- Fuente: **OpenML** — dataset `Credit` (versión 1).
- Tamaño: **16.714** filas · **11** características + objetivo.
- Variable objetivo: **`SeriousDlqin2yrs`** (1=evento severo de morosidad en 2 años).
- Balance observado:
  - `0`: **8352**
  - `1`: **8273**

Carga usada en el notebook:
```python
from sklearn.datasets import fetch_openml
df = fetch_openml("Credit", version=1, as_frame=True).frame
