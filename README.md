# Predicción de default de tarjetas de crédito: Bayes ingenuo vs. PCA + Bayes ingenuo

Proyecto de ciencia de datos que predice si un cliente caerá en default el mes siguiente, usando el dataset *Default of Credit Card Clients* de UCI. Compara un clasificador Bayes ingenuo (Gaussian Naive Bayes) con y sin reducción de dimensionalidad (PCA), y ajusta el umbral de decisión para mejorar la detección de defaults.

## Pregunta de investigación

¿Mejora PCA el rendimiento de Bayes ingenuo en un dataset con variables fuertemente correlacionadas?

## Dataset

- **Fuente:** [UCI Machine Learning Repository, Default of Credit Card Clients (id 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tamaño:** 30.000 clientes, 23 variables predictoras, sin valores nulos
- **Origen:** clientes de tarjetas de crédito en Taiwán (2005)
- **Objetivo:** `default` (1 = cae en default el mes siguiente). Clase positiva: ~22%

| Grupo | Variables |
|---|---|
| Crédito | `LIMIT_BAL` |
| Demográficas | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Estado de pago (6 meses) | `PAY_1` ... `PAY_6` |
| Monto facturado (6 meses) | `BILL_AMT1` ... `BILL_AMT6` |
| Monto pagado (6 meses) | `PAY_AMT1` ... `PAY_AMT6` |

## Metodología

1. **Carga y limpieza**
   - Lectura del `.xls` original (`header=1`) y renombre de columnas (`PAY_0` → `PAY_1`, objetivo → `default`).
   - Categorías no documentadas agrupadas: `EDUCATION` {0, 5, 6} → 4; `MARRIAGE` {0} → 3.
   - Eliminación de `ID`.
2. **Análisis exploratorio**
   - Tasa de default por estado de pago, nivel educativo, sexo, estado civil y decil de límite de crédito.
   - Matriz de correlación: los seis `BILL_AMT` forman un bloque muy correlacionado, lo que motiva el uso de PCA y viola el supuesto de independencia de Bayes ingenuo.
3. **Preprocesamiento** (dentro de un `Pipeline`, ajustado solo con train para evitar data leakage)
   - Montos, edad y límite: `PowerTransformer` (Yeo-Johnson), que corrige colas largas y admite valores negativos.
   - `PAY_1..6`: `StandardScaler`.
   - Variables categóricas: `OneHotEncoder`.
4. **Modelos**
   - A: `GaussianNB`
   - B: `PCA(n_components=0.95)` + `GaussianNB`
   - Split estratificado 80/20, `random_state=42`.
5. **Evaluación:** ROC-AUC, PR-AUC, recall, precisión y F1. No se usa accuracy como métrica principal por el desbalance de clases.
6. **Ajuste de umbral:** probabilidades *out-of-fold* con validación cruzada estratificada en train; el umbral que maximiza F1 se evalúa una sola vez en test.

## Resultados

| Modelo | ROC-AUC | PR-AUC | Recall (default) |
|---|---|---|---|
| Naive Bayes sin PCA | **0,744** | **0,481** | **0,523** |
| PCA (17 componentes, 95% var.) + Naive Bayes | 0,708 | 0,407 | 0,339 |

- Umbral óptimo por F1 (validación cruzada en train): **0,39**.
- <!-- Completar con precisión / recall / F1 en test para umbral 0,5 vs. 0,39 -->

## Conclusiones

- **PCA empeoró el modelo.** PCA conserva varianza, no capacidad predictiva: al descartar el 5% restante se perdió señal útil, y las variables `PAY_x`, las más predictivas, quedaron repartidas entre componentes.
- Decorrelacionar las variables no compensó la pérdida de información.
- Bajar el umbral de 0,5 a 0,39 aumenta la detección de defaults a costa de más falsas alarmas, un trade-off razonable en riesgo crediticio, donde no detectar un default suele costar más que rechazar a un buen cliente.

## Limitaciones

- Datos de Taiwán de 2005: los resultados no necesariamente generalizan a otros mercados o períodos.
- Bayes ingenuo asume independencia condicional entre variables, supuesto que este dataset incumple.
- Las probabilidades de Bayes ingenuo suelen estar mal calibradas; el umbral óptimo es específico de este modelo y este dataset.
- No se compararon modelos más flexibles (regresión logística, árboles, gradient boosting).

## Cómo reproducirlo

1. Abrir el notebook en Google Colab (o Jupyter).
2. Ejecutar las celdas en orden. El dataset se descarga automáticamente desde UCI.

**Librerías:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xlrd`

## Autor

Lorenzo ciprés, estudiante de Ciencia de Datos (Universidad Nacional Guillermo Brown).
