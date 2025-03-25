import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o conjunto de dados (assumindo o arquivo 'winequality-red.csv' no diretório)
data = pd.read_csv('winequality-red.csv')

# Separar as características (X) e a variável alvo (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Dividir os dados em conjuntos de treinamento e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliar o modelo com o conjunto de teste
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE (Erro Quadrático Médio): {mse:.2f}')
print(f'R² (Coeficiente de Determinação): {r2:.2f}')

# Função para o usuário inserir as características e prever a qualidade
def predict_quality():
    print("Insira as características do seu vinho:")
    fixed_acidity = float(input("Acidez fixa: "))
    volatile_acidity = float(input("Acidez volátil: "))
    citric_acid = float(input("Ácido cítrico: "))
    residual_sugar = float(input("Açúcar residual: "))
    chlorides = float(input("Cloretos: "))
    free_sulfur_dioxide = float(input("Dióxido de enxofre livre: "))
    total_sulfur_dioxide = float(input("Dióxido de enxofre total: "))
    density = float(input("Densidade: "))
    pH = float(input("pH: "))
    sulphates = float(input("Sulfatos: "))
    alcohol = float(input("Teor alcoólico: "))

    # Criar um DataFrame com as características inseridas
    features = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                              free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                            columns=X.columns)

    # Fazer a previsão
    quality_pred = model.predict(features)
    print(f'A qualidade prevista do vinho é: {quality_pred[0]:.2f} (em uma escala de 0 a 10)')

# Chamar a função para permitir a previsão
predict_quality()