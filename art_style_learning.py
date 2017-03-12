from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import keras.callbacks
import tensorflow as tf
import numpy as np
import random
import h5py
import os
import scipy.misc
import matplotlib.pyplot as plt


def class_to_genre(genres, class_vector):
    def conv(cl):
        try:
            genre = genres[cl]
        except:
            genre = "False"

        return genre

    genre_vector = [conv(t) for t in class_vector]

    return genre_vector


def metalnet(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, init='normal', border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Convolution2D(128, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Convolution2D(256, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Convolution2D(512, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Convolution2D(512, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3, init='normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Flatten())
    model.add(Dense(1024, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(666, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, init='glorot_uniform'))
    model.add(Activation('softmax'))

    return model


def metalnet_simple(input_shape):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(666, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(Activation('softmax'))

    return model


def extract_data(filename):
    h5file = h5py.File(filename, 'r')
    imgs = h5file.get("images")     # imgs.shape should be
    labels = h5file.get("labels")   # (# of imgs, IMAGE_SIZE, IMAGE_SIZE, 3)
    images = imgs[:(len(imgs))]
    images = images.astype(np.float32)
    images = images / 255.0
    images = images.astype(np.float32)
    labels = labels[:len(labels)]
    labels = labels.astype(np.float64)
    h5file.close()
    return images, labels


def class_to_de_genre(d_genres, class_vector):
    def conv(cl):
        try:
            genre = d_genres[cl]
        except:
            genre = "False"

        return genre

    genre_vector = [conv(t) for t in class_vector]

    return genre_vector



if __name__ == '__main__':
    genres = ["Heavy", "Hair", "Thrash", "Doom", "Death", "Black",
          "Symphonic", "Power", "Progressive", "Metalcore",
          "Grindcore", "Groove", "Goregrind", "Gothic", "Viking", "False"]

    working_dir = os.getcwd()

    filename = "metalcovers_train_s.h5"
    train_image, train_label = extract_data(filename)

    filename = "metalcovers_test_s.h5"
    test_image, test_label = extract_data(filename)

    filename = "metalcovers_show.h5"
    test_show_imgs, test_show_lbls = extract_data(filename)
    test_show_imgs = test_show_imgs * 255.0

    count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for l in train_label:
        count += l
    print count


    html = '<font size="5">Used Images Statistics<br><br><font>'
    for i in xrange(len(count)):
        stats = '<font size="4>%s - %d<font><br>' % (genres[i], int(count[i]))
        html += stats

    input_shape = (224,224,3)

    old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        model = metalnet(input_shape)

        #w_fname = "training/simplest_cnn_model_s.hdf5"
        #model.load_weights(w_fname)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        tb_cb = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True)
        cp_cb = keras.callbacks.ModelCheckpoint("./training/simplest_cnn_model_s.hdf5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
        cbks = [tb_cb, cp_cb]
        history = model.fit(train_image, train_label, shuffle=True, nb_epoch=40, batch_size=25, verbose=1, callbacks=cbks)

        loss_and_metrics = model.evaluate(test_image, test_label, batch_size=25)
        print(loss_and_metrics)

        pre_value = model.predict(test_show_imgs)
        pre_class = model.predict_classes(test_show_imgs)

        print(class_to_de_genre(genres, pre_class))

        json_string = model.to_json()
        with open(working_dir + '/training/simplest_cnn_model_s.json', 'w') as f:
            f.write(json_string)

    KTF.set_session(old_session)

    html += 'Accuracy: %d' % (int(loss_and_metrics[1] * 100))
    html += '%<br><font size="5">Result<br><br><font>'

    current_dir = os.getcwd()
    result_dir = current_dir + '/result'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    pred = class_to_genre(genres, pre_class)
    ans = class_to_genre(genres, test_show_lbls.nonzero()[1])


    for i in xrange(len(ans)):
        print "prediction:%s truth:%s" % (pred[i],ans[i])

        path = "/home/yashima/public_html/result13/" + str(i) + "_" + ans[i] + "_as_" + pred[i] + ".jpg"

        scipy.misc.imsave(path, np.asarray(test_show_imgs[i]))

        correct_genre = '<font size="4">[Answer]:%s<font><br>' % ans[i]
        predicted_genre = '<font size="4">[Predicted]:%s<font><br>' % pred[i]
        path2 = "result13/" + str(i) + "_" + ans[i] + "_as_" + pred[i] + ".jpg"
        Images = '<img src="%s"align="left">' % (path, )

        html += '<img src="%s"align="left">' % (path2, )
        html += '%s<br>' % correct_genre
        html += '%s<br>' % predicted_genre
        html += '<br><br clear="all"><br>'

    html_file = result_dir + "/result_13.html"
    print 'writing html result file to %s...' % (html_file, )
    open(html_file, 'w').write(html)
