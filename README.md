# Convolutuinal-Neural-Networks

Nesta seção, usaremos o famoso Dataset MNIST para construir duas Redes Neurais capazes de realizar a classificação de dígitos manuscritos. A primeira rede é um simples Perceptron Multi-layer (MLP) e a segunda é uma Rede Neural Convolucional (CNN a partir de agora). Em outras palavras, quando dada uma entrada, nosso algoritmo dirá, com algum erro associado, que tipo de dígito esta entrada representa.
Project Coursera - Tensorflow

## Perceptron Multi-layer (MLP)

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
A precisão de 84% não é ruim considerando a simplicidade do modelo, mas uma precisão de> 90% foi alcançada no passado.

![image](https://github.com/MilianoJunior/Convolutuinal-Neural-Networks/blob/main/plot/acurracies.png)

![image](https://github.com/MilianoJunior/Convolutuinal-Neural-Networks/blob/main/plot/loss.png)


## Rede Neural Convolucional (CNN)

Na primeira parte, aprendemos como usar uma RNA simples para classificar MNIST. Agora vamos expandir nosso conhecimento usando uma Rede Neural Profunda.

 > Entrada - conjunto de dados MNIST 
 > Convolucional e Max-Pooling 
 > Convolucional e Max-Pooling 
 > Camada totalmente conectada 
 > Processamento - Dropout
 > Camada de leitura - totalmente conectada 
 > Saídas - dígitos classificados

### Entrada - conjunto de dados MNIST
```sh
x_image_train = tf.reshape(x_train, [-1,28,28,1])  
x_image_train = tf.cast(x_image_train, 'float32') 

x_image_test = tf.reshape(x_test, [-1,28,28,1]) 
x_image_test = tf.cast(x_image_test, 'float32') 

train_ds2 = tf.data.Dataset.from_tensor_slices((x_image_train, y_train)).batch(50)
test_ds2 = tf.data.Dataset.from_tensor_slices((x_image_test, y_test)).batch(50)
x_image_train = tf.slice(x_image_train,[0,0,0,0],[10000, 28, 28, 1])
y_train = tf.slice(y_train,[0,0],[10000, 10])
```
### Convolucional e Max-Pooling (1)
```sh
W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=0))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

def convolve1(x):
    return(
        tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
def h_conv1(x): return(tf.nn.relu(convolve1(x)))

def conv1(x):
    return tf.nn.max_pool(h_conv1(x), ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')
```           
### Convolucional e Max-Pooling (2)
```sh
W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs
def convolve2(x): 
    return( 
    tf.nn.conv2d(conv1(x), W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
def h_conv2(x):  return tf.nn.relu(convolve2(x))
def conv2(x):  
    return(
    tf.nn.max_pool(h_conv2(x), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
```
### Camada totalmente conectada
```sh
def layer2_matrix(x): return tf.reshape(conv2(x), [-1, 7 * 7 * 64])
W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1, seed = 2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
def fcl(x): return tf.matmul(layer2_matrix(x), W_fc1) + b_fc1
def h_fc1(x): return tf.nn.relu(fcl(x))
```
### Processamento - Dropout
```sh
keep_prob=0.5
def layer_drop(x): return tf.nn.dropout(h_fc1(x), keep_prob)
```
### Camada de leitura - totalmente conectada 
```sh
W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1, seed = 2)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
def fc(x): return tf.matmul(layer_drop(x), W_fc2) + b_fc2
def y_CNN(x): return tf.nn.softmax(fc(x))
```
### Saídas - Treinamento e teste

```sh
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))
    
optimizer = tf.keras.optimizers.Adam(1e-4)


variables = [W_conv1, b_conv1, W_conv2, b_conv2, 
             W_fc1, b_fc1, W_fc2, b_fc2, ]

def train_step(x, y):
    with tf.GradientTape() as tape:
        current_loss = cross_entropy( y, y_CNN( x ))
        grads = tape.gradient( current_loss , variables )
        optimizer.apply_gradients( zip( grads , variables ) )
        return current_loss.numpy()

correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1), tf.argmax(y_train, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

loss_values=[]
accuracies = []
epochs = 1

for i in range(epochs):
    j=0
    # each batch has 50 examples
    for x_train_batch, y_train_batch in train_ds2:
        j+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if j%50==0: #reporting intermittent batch statistics
            correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
            #  accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
            print("epoch ", str(i), "batch", str(j), "loss:", str(current_loss),
                     "accuracy", str(accuracy)) 
            
    current_loss = cross_entropy( y_train, y_CNN( x_image_train )).numpy()
    loss_values.append(current_loss)
    correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1),
                                  tf.argmax(y_train, axis=1))
    #  accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
    accuracies.append(accuracy)
    print("end of epoch ", str(i), "loss", str(current_loss), "accuracy", str(accuracy) ) 

```
### Avaliação e Visualização

```sh
j=0
acccuracies=[]
# evaluate accuracy by batch and average...reporting every 100th batch
for x_train_batch, y_train_batch in train_ds2:
        j+=1
        correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        #accuracies.append(accuracy)
        if j%100==0:
            print("batch", str(j), "accuracy", str(accuracy) ) 
import numpy as np
print("accuracy of entire set", str(np.mean(accuracies))) 

kernels = tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1])

import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = [x_image_train[0]]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")

#ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
keep_prob=1.0
ActivatedUnits = convolve1(sampleimage)
                           
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
    
    #ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
ActivatedUnits = convolve2(sampleimage)
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

```