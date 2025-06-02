from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import tracemalloc

# Carregar o dataset Diabetes
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target

# Estatísticas e amostras
print("===== DIABETES: Estatísticas =====")
print(X.describe())
print("\nAmostras:")
print(X.head())

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo
model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

# Treinamento
tracemalloc.start()
start = time.time()
model.fit(X_train, y_train)
treino_time = time.time() - start
_, mem_pico_treino = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Previsão
tracemalloc.start()
start = time.time()
y_pred = model.predict(X_test)
pred_time = time.time() - start
_, mem_pico_pred = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Avaliação
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Gráfico
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("SVR - Regressão Diabetes")
plt.grid(True)
plt.show()

print("\nTempo e Memória:")
print(f"Tempo de treino: {treino_time:.4f}s")
print(f"Pico de memória no treino: {mem_pico_treino / 10**6:.2f} MB")
print(f"Tempo de previsão: {pred_time:.4f}s")
print(f"Pico de memória na previsão: {mem_pico_pred / 10**6:.2f} MB")