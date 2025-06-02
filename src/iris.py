from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc

# Carregar o dataset Iris
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Estatísticas e amostras
print("===== IRIS: Estatísticas =====")
print(X.describe())
print("\nAmostras:")
print(X.head())

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo
model = DecisionTreeClassifier(criterion="entropy", random_state=0)

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
print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nMatriz de Confusão:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Matriz de Confusão - Iris")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

print("\nTempo e Memória:")
print(f"Tempo de treino: {treino_time:.4f}s")
print(f"Pico de memória no treino: {mem_pico_treino / 10**6:.2f} MB")
print(f"Tempo de previsão: {pred_time:.4f}s")
print(f"Pico de memória na previsão: {mem_pico_pred / 10**6:.2f} MB")