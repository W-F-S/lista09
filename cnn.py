import tensorflow as tf
import keras as k
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image


lista = ['le1.jpg',  
         'lo4.jpg', 
         'lo8.jpg',
         'le3.jpg', 
         'le6.jpg',
         'le7.jpg',
         'le9.jpg',
         'le10.jpg',
         'lo6.jpg',
         'lo7.jpg',
         'lo9.jpg',
         'le5.jpg', 
         'lo10.jpg',
         'lo1.jpg', 
         'le4.jpg', 
         'lo2.jpg', 
         'lo3.jpg', 
         'le2.jpg',
         'le8.jpg',
         'lo5.jpg']

classifier = Sequential()

#primeira camada
classifier.add(Conv2D(32, (3, 3), input_shape = ( 64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#segunda camada 
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#terceira camada de convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#convertendo a matriz anterior em um array
classifier.add(Flatten())

#conectamos as camadas usando uma funcação de ativacao retificadora e uma sigmóide para obter a probabilidade de conter um gato ou cachorro na imagem
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))

#compilando a rede, usamos um algoritmo de otimização chamado adam
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fazendo a base de treino, Com a função image.ImageDataGenerator() vamos ajustar a escala e zoom das imagens de treino e validação.
train_datagen = image.ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range  = 0.2,
                                   horizontal_flip = True)

validation_datagen = image.ImageDataGenerator(rescale = 1./255)


#treinando o dataset usando imgens de leões e leopardos
training_set = train_datagen.flow_from_directory('leopardo/data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('leopardo/data/val',
                                                        target_size = (64, 64),
                                                        batch_size = 32, # = quantidade de imagens por bach. isso é um "Mini batch gradient descent" pois é menor que a quantidade total de imagens 
                                                        class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 85, #quanto maior o valor, melhor a qualidade do treinamento
                         epochs = 10, # epochs = numero de batches 
                         validation_data = validation_set,
                         validation_steps = 2000)

#lista de arquivos que serão usados na parte de teste
n_acertos = 0
for nome in lista:
    test_image = image.load_img(nome, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices


    if result[0][0] == 1:
        prediction = 'Leao'
        if nome[1] == 'o':
            n_acertos = n_acertos + 1
    else:
        prediction = 'Leopardo'
        if nome[1] == 'e':
            n_acertos = n_acertos + 1
        
    #algoritimo classificará errado com um steps_per_epoch <= 10 e epochs = 3
    
    print(nome)
    print(prediction)

print("número de acertos do algoritmo: " + str(n_acertos) + "; testes realizados: " + str(len(lista)))
"""
    O autor recomenda:

        aumentar o número de epochs para 25
        aumentar a resolução das imagens
        aumentar o tamanho de lote (???) para 64
        usar uma técnica chamada DataSetAugmentation 
        adicionar mais uma camada convolucional 
        experimentar outros algoritmos de otimização
"""
"""
    ao tentar rodar o algoritmo com a base dos simpsons 
    recebo um erro, pois temos um data set pequeno. tentar ver nos slides o que pode ser feito.
    
    pegando uma imagem aleatória do homer e do bart o algoritmo consege classificar com perfeição
"""


"""
    Para o documento, talvez configurar o jupyter e fazer passo a passo as mudanças que ela pede no módulo 03 do exercício.
"""

