# Convolutuinal-Neural-Networks

Nesta seção, usaremos o famoso Dataset MNIST para construir duas Redes Neurais capazes de realizar a classificação de dígitos manuscritos. A primeira rede é um simples Perceptron Multi-layer (MLP) e a segunda é uma Rede Neural Convolucional (CNN a partir de agora). Em outras palavras, quando dada uma entrada, nosso algoritmo dirá, com algum erro associado, que tipo de dígito esta entrada representa.
Project Coursera - Tensorflow

### 1ª parte: classificar MNIST usando um modelo simples.

De acordo com o site da LeCun, o MNIST é um: "banco de dados de dígitos escritos à mão que tem um conjunto de treinamento de 60.000 exemplos e um conjunto de teste de 10.000 exemplos. É um subconjunto de um conjunto maior disponível no NIST. Os dígitos foram dimensionados- normalizado e centralizado em uma imagem de tamanho fixo "
```sh
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

```
### Normalização e codificação
Precisamos convertê-lo em um vetor codificado de um ponto. Em contraste com a representação binária, os rótulos serão apresentados de uma forma que para representar um número N, o bit 𝑁𝑡ℎ é 1 enquanto os outros bits são 0. Por exemplo, cinco e zero em um código binário seriam:
```sh
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# make labels one hot encoded
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

```
A nova API Dataset no TensorFlow 2.X permite definir tamanhos de lote como parte do conjunto de dados
```sh
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)
```
### Convertendo a imagem 2D em um vetor 1D
```sh
from tensorflow.keras.layers import Flatten
flatten = Flatten(dtype='float32')
```
