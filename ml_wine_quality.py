import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Carregar os dados (assumindo 'winequality-red.csv')
data = pd.read_csv('winequality-red.csv')

# Função para categorizar a qualidade em 'baixa', 'média', 'alta'
def categorize_quality(quality):
    if quality <= 4:
        return 0  # Baixa
    elif quality <= 6:
        return 1  # Média
    else:
        return 2  # Alta

# Aplicar a categorização
X = data.drop('quality', axis=1)
y = data['quality'].apply(categorize_quality)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo com hiperparâmetros ajustados
model = GradientBoostingClassifier(
    n_estimators=200,      # Mais árvores para maior capacidade
    learning_rate=0.05,    # Taxa de aprendizado menor para melhor generalização
    max_depth=4,           # Profundidade ajustada para capturar mais padrões
    random_state=42
)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo com métricas detalhadas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Baixa', 'Média', 'Alta']))

# Validação cruzada para uma avaliação mais robusta
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nAcurácia média com validação cruzada: {cv_scores.mean():.2f} (± {cv_scores.std():.2f})")

# Salvando o modelo
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
