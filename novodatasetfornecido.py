import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

from scipy import stats


vinho = pd.read_csv("group_15_winequality.csv")

#Transformando as strings em números para performar os algoritmos
for col in vinho.columns:
    if col == "type":
        continue
    vinho[col] = pd.to_numeric(vinho[col], errors='coerce')


#Dropando Nans    
vinho = vinho.dropna()

#Convertendo feature binária para int
vinho["type"] = (vinho["type"] == "red").astype(int)


#Tratamento de outliers usando z-score
z_scores = np.abs(stats.zscore(vinho.select_dtypes(include=np.number)))
df_clean = vinho[(z_scores < 3).all(axis=1)].copy()

'''features = vinho.columns.drop('quality') 

plt.figure(figsize=(24, 18))

for i, feature in enumerate(features):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x='quality', y=feature, data=vinho)
    plt.title(f'{feature} vs Quality', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.4)
plt.show()'''


# Matriz de correlação
'''plt.figure(figsize=(12,10))
sns.heatmap(vinho.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()'''

#Sem valores ausentes
'''print(vinho.isnull().sum())'''


#Aplicando padronização
X = df_clean.drop(["quality", "type"], axis=1) #Evitando a coluna target e as colunas já padronizadas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Setando o target
y = df_clean["quality"]

#Definindo o treino/teste ratio do dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)



######Regressão#######



modelos = {
    'Regressão Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR()
}

for name, model in modelos.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Modelo: {name}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")
    print('-'*30)
    

#Ajuste de hiperparâmetros para o RandomForestRegressor

'''param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred_best = best_rf.predict(X_test)

print("Melhores parâmetros encontrados:", grid_search.best_params_)
print(f"RMSE dps do GridSearch: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.3f}")
print(f"MAE dps do GridSearch: {mean_absolute_error(y_test, y_pred_best):.3f}")
print(f"R2 dps do GridSearch: {r2_score(y_test, y_pred_best):.3f}")'''

#Melhores parâmetros encontrados: {'bootstrap': False, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
#RMSE dps do GridSearch: 0.670
#MAE dps do GridSearch: 0.452
#R2 dps do GridSearch: 0.441

melhor_rf = RandomForestRegressor(
    bootstrap=False,
    max_depth=None,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=2,
    n_estimators=200,
)

melhor_rf.fit(X_train, y_train)
y_pred = melhor_rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Modelo: Random Forest com os hiperparâmetros tunados:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R2: {r2:.3f}")
print('-'*30)


#Coeficientes e feature importance

'''importances = melhor_rf.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_imp)

#A feature "alcohol" parece ser claramente a feature mais importante para o Random Forest já tunado, com 25,3% de importância, seguido de "density" e "volatile acidity", com 10,5% e 08,7%, respectivamente.

plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title('Importância das Variáveis no Random Forest Tunado')
plt.ylabel('Importância')
plt.show()'''





######Classificação#######




#Definindo as categorias e aplicando como uma coluna no modelo
def categorize(q):
    if q <= 4:
        return 'Baixa'
    elif q <= 6:
        return 'Média'
    else:
        return 'Alta'

df_clean['quality_categorizado'] = df_clean['quality'].apply(categorize)


#Plotando as categorias
sns.countplot(x='quality_categorizado', data=df_clean)
plt.title('Distribuição das classes de qualidade')
plt.show()


#Setando X e y
X_classificacao = df_clean.drop(["quality", "type", "quality_categorizado"], axis=1)
y_classificacao = df_clean["quality_categorizado"]

#Rótulos e padronização antes do split
le = LabelEncoder()
y_encoded = le.fit_transform(y_classificacao)

scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_classificacao)


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class_scaled, y_classificacao, test_size=0.2)

#Oversampling
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train_cls, y_train_cls)

#Checar o oversampling
print(y_train_res.value_counts())



modelos_classifiers = {
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, clf in modelos_classifiers.items():
    clf.fit(X_train_res, y_train_res)
    y_pred_cls = clf.predict(X_test_cls)
    print(f"Modelo: {name}")
    print(classification_report(y_test_cls, y_pred_cls, target_names=le.classes_))
    print(confusion_matrix(y_test_cls, y_pred_cls))
    print('-'*30)


#Novamente Random Forest foi o melhor modelo, tunando:

'''param_grid_cls = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

grid_search_cls = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_cls,
    cv=3,                    
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search_cls.fit(X_train_res, y_train_res)

print("Melhores parâmetros:", grid_search_cls.best_params_)
best_clf = grid_search_cls.best_estimator_

y_pred_best_cls = best_clf.predict(X_test_cls)
print("Random Forest com hiperparâmetros tunados:", classification_report(y_test_cls, y_pred_best_cls, target_names=le.classes_))'''

#Melhores parâmetros: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
#Random Forest com hiperparâmetros tunados:               precision    recall  f1-score   support

#        Alta       0.68      0.70      0.69       244
#       Baixa       0.00      0.00      0.00         7
#      Média       0.92      0.92      0.92      1004

#    accuracy                           0.87      1255
#   macro avg       0.53      0.54      0.54      1255
# weighted avg       0.87      0.87      0.87      1255


#Aplicando modelo tunado

melhor_rf_cls = RandomForestClassifier(
    bootstrap=False,
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
)

melhor_rf_cls.fit(X_train_res, y_train_res)
y_pred_final = melhor_rf_cls.predict(X_test_cls)

# Métricas

print("Modelo Random Forest Tunado: ", classification_report(y_test_cls, y_pred_final, target_names=le.classes_))

cm = confusion_matrix(y_test_cls, y_pred_final)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão Random Forest Tunado')
plt.show()


#Coeficientes e feature importance

importances = melhor_rf_cls.feature_importances_
feat_imp = pd.Series(importances, index=X_classificacao.columns).sort_values(ascending=False)
print(feat_imp)

plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title('Importância das Variáveis no Random Forest Classificador Tunado')
plt.ylabel('Importância')
plt.show()

#"alcohol" ainda continua sendo a feature mais importante com 13,3%, mas dessa vez, a distribuição da importância está bem mais homogênea, com "free sulfur dioxide" logo atrás com 12,3% e "density" com 12%.






