{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_train shape: (60000, 28, 28, 1)\n",
      "60000 train samples\n",
      "10000 test samples\n",
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/1\n",
      "60000/60000 [==============================] - 32s 534us/step - loss: 0.1601 - accuracy: 0.9512 - val_loss: 0.0622 - val_accuracy: 0.9793\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D\n",
    "from keras import backend as K\n",
    "import os\n",
    "\n",
    "def model_train(epoch,n):\n",
    "    batch_size = 32\n",
    "    num_classes = 10\n",
    "    epochs = epoch\n",
    "\n",
    "    # input image dimensions\n",
    "    img_rows, img_cols = 28, 28\n",
    "\n",
    "    # the data, split between train and test sets\n",
    "    (x_train, y_train), (x_test, y_test) = mnist.load_data()\n",
    "\n",
    "    if K.image_data_format() == 'channels_first':\n",
    "        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)\n",
    "        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)\n",
    "        input_shape = (1, img_rows, img_cols)\n",
    "    else:\n",
    "        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n",
    "        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n",
    "        input_shape = (img_rows, img_cols, 1)\n",
    "\n",
    "    x_train = x_train.astype('float32')\n",
    "    x_test = x_test.astype('float32')\n",
    "    x_train /= 255\n",
    "    x_test /= 255\n",
    "    print('x_train shape:', x_train.shape)\n",
    "    print(x_train.shape[0], 'train samples')\n",
    "    print(x_test.shape[0], 'test samples')\n",
    "\n",
    "    # convert class vectors to binary class matrices\n",
    "    y_train = keras.utils.to_categorical(y_train, num_classes)\n",
    "    y_test = keras.utils.to_categorical(y_test, num_classes)\n",
    "\n",
    "    model = Sequential()\n",
    "    \n",
    "    model.add(Conv2D(32, kernel_size=(3, 3),\n",
    "                     activation='relu',\n",
    "                     input_shape=input_shape))\n",
    "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    if n>1:\n",
    "        model.add(Conv2D(32, kernel_size=(3, 3),\n",
    "                     activation='relu'))\n",
    "        model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(128, activation='relu'))\n",
    "    model.add(Dense(num_classes, activation='softmax'))\n",
    "\n",
    "    model.compile(loss=keras.losses.categorical_crossentropy,\n",
    "                  optimizer=keras.optimizers.Adadelta(),\n",
    "                  metrics=['accuracy'])\n",
    "\n",
    "    model.fit(x_train, y_train,\n",
    "              batch_size=batch_size,\n",
    "              epochs=epochs,\n",
    "              verbose=1,\n",
    "              validation_data=(x_test, y_test))\n",
    "    score = model.evaluate(x_test, y_test, verbose=0)\n",
    "    a=score[1]*100\n",
    "    model.save(\"MNIST.h5\")\n",
    "    os.system(\"mv /MNIST.h5 /mycode\")\n",
    "    return a\n",
    "\n",
    "no_epoch=1\n",
    "no_layer=1\n",
    "accuracy_train_model=model_train(no_epoch,no_layer)\n",
    "f = open(\"accuracy.txt\",\"w+\")\n",
    "f.write(str(accuracy_train_model))\n",
    "f.close()\n",
    "os.system(\"mv /accuracy.txt /mycode\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
