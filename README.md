# 📊 Predição e Classificação de Qualidade de Vinhos

Projeto de **predição e classificação da qualidade de vinhos tintos e brancos** utilizando regressão e classificação supervisionada para a disciplina de Machine Learning da UNISATC.  

## Objetivos

- Predizer a qualidade de vinhos com algoritmos de regressão.
- Categorizar a qualidade em três possíveis classes: **Baixa**, **Média** e **Alta**.
- Avaliar a performance de diferentes algoritmos.
- Otimização de hiperparâmetros.

## Dataset

Utilizamos dois datasets do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality):

- `winequality-red.csv`
- `winequality-white.csv`

Cada registro representa uma amostra de vinho com atributos físico-químicos (pH, álcool, acidez volátil, etc.) e a qualidade atribuída por especialistas (nota de 0 a 10).

---

## Etapas do projeto

1. **Pré-processamento**
   - Combinação dos datasets.
   - Conversão de tipos de dados.
   - Remoção de valores ausentes.
   - Transformação da variável `type` (0 = branco | 1 = tinto).
   - Padronização das variáveis numéricas com `StandardScaler`.
   - Criação da variável categórica `quality_categorizado` para classificação.

![pre_process](https://github.com/user-attachments/assets/c4b76910-ec82-4e9f-9e9b-ac67063e5247)

2. **Análise Exploratória**
   - Boxplots e gráficos de distribuição.
   - Matriz de correlação.

3. **Modelagem**
   - **Regressão:** 
     - Linear Regression
     - Random Forest Regressor (com ajuste de hiperparâmetros via `GridSearchCV`)
     - SVR (Support Vector Regressor)
     
    ![regression](https://github.com/user-attachments/assets/eb8d86f5-97c5-417d-b7bd-e013cbd244ad)

   - **Classificação:**
     - Random Forest Classifier
     - Oversampling com `SMOTE` para balanceamento de classes.
     - Ajuste de hiperparâmetros com `GridSearchCV`.
   
   ![class](https://github.com/user-attachments/assets/dabeb7b4-bf8c-4fef-8dd5-a413916a55d6)

4. **Métricas de Avaliação**
   - Regressão: RMSE, MAE, R².
   - Classificação: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

---

## Resultados

- **A variável `alcohol` se mostrou a mais importante** para predição e classificação, seguida de `density` e `volatile acidity`.
- O modelo **Random Forest(Classifier e Regressor)** obteve o melhor desempenho tanto em regressão quanto em classificação após os hiperparâmetros serem tunados com o auxílio do GridSearchCV.
  
---

## Como rodar o projeto

1. Clone o repositório ou baixe os arquivos e instale as dependências:

```bash```
pip install -r requirements.txt



## Link para o repo do projeto: https://github.com/wguii/trabalho_ml.git
