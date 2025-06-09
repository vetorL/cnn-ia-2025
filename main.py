import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Criar pasta de saída
os.makedirs("saida_cnn", exist_ok=True)

# Carrega o MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compila
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Salva pesos iniciais
model.save_weights("saida_cnn/pesos_iniciais.weights.h5")

# Treinamento
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Salva pesos finais
model.save_weights("saida_cnn/pesos_finais.weights.h5")

# Salva hiperparâmetros e arquitetura
with open("saida_cnn/hiperparametros.txt", "w") as f:
    f.write("Otimizador: Adam\n")
    f.write("Função de perda: sparse_categorical_crossentropy\n")
    f.write("Épocas: 10\n")
    f.write("Arquitetura:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Salva erro por iteração
pd.DataFrame({
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}).to_csv("saida_cnn/erro_por_iteracao.csv", index=False)

# Predição
preds = model.predict(x_test)
y_pred = preds.argmax(axis=1)

# Salva saídas da rede para dados de teste
pd.DataFrame(preds).to_csv("saida_cnn/saidas_rede.csv", index=False)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv("saida_cnn/matriz_confusao.csv", index=False)

# Opcional: plot da matriz de confusão
plt.imshow(cm, cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig("saida_cnn/matriz_confusao.png")
