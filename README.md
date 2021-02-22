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
# make labels one hot encoded
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

```
A nova API Dataset no TensorFlow 2.X permite definir tamanhos de lote como parte do conjunto de dados,desta forma, otimizando
a utlização dos recursos computacionais.
```sh
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)
```
### Convertendo a imagem 2D em um vetor 1D
```sh
from tensorflow.keras.layers import Flatten
flatten = Flatten(dtype='float32')
flatten(x_train).shape
```
### Atribuição de viés e pesos a tensores nulos e construção do modelo
Agora vamos criar os pesos e vieses, para isso eles serão usados como arrays preenchidos com zeros.
```sh
# Weight tensor
W = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))
# função linear, modelo que é ajustado pelo otimizador.
def forward(x):
    return tf.matmul(x,W) + b
#Definindo a saida da rede neural com a função softmax
def activate(x):
    return tf.nn.softmax(forward(x))
#Criando a função model
def model(x):
    x = flatten(x)
    return activate(x)
#Definindo a função custo
def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))
```
### Otimizador e treinamento

Otimização com a descida do gradiente SGD
```sh
optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
```
treinamento
```sh
def train_step(x, y ):
    with tf.GradientTape() as tape:
        #compute loss function
        current_loss = cross_entropy( y, model(x))
        # compute gradient of loss 
        #(This is automatic! Even with specialized funcctions!)
        grads = tape.gradient( current_loss , [W,b] )
        # Apply SGD step to our Variables W and b
        optimizer.apply_gradients( zip( grads , [W,b] ) )     
    return current_loss.numpy()
#Training batches
loss_values=[]
accuracies = []
epochs = 10

for i in range(epochs):
    j=0
    # each batch has 50 examples
    for x_train_batch, y_train_batch in train_ds:
        j+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if j%500==0: #reporting intermittent batch statistics
            print("epoch ", str(i), "batch", str(j), "loss:", str(current_loss) ) 
    
    # collecting statistics at each epoch...loss function and accuracy
    #  loss function
    current_loss = cross_entropy( y_train, model( x_train )).numpy()
    loss_values.append(current_loss)
    correct_prediction = tf.equal(tf.argmax(model(x_train), axis=1),
                                  tf.argmax(y_train, axis=1))
    #  accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
    accuracies.append(accuracy)
    print("end of epoch ", str(i), "loss", str(current_loss), "accuracy", str(accuracy) ) 
```
### Teste e Visualização