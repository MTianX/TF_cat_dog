import keras
import matplotlib.pyplot as plt
from data import create_dir_and_data
from keras.applications.inception_v3 import InceptionV3

if __name__ == "__main__":
    train_dir, validation_dir, test_dir = create_dir_and_data()
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                                                                 shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

    #创建网络
    base_model = InceptionV3(weights='imagenet',include_top=False,input_shape = (150,150,3))
    base_model.trainable = False
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation=keras.activations.relu))
    model.add(keras.layers.Dense(1,activation=keras.activations.sigmoid))

    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.RMSprop(lr=2e-5),metrics=['acc'])
    history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

    model.save('model.h5')


