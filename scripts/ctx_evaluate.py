import matplotlib
import PIL
PIL.Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 2248656400
# from PIL import Image
import cv2
import numpy as np
print('op')
img = PIL.Image.open('MurrayLab_GlobalCTXMosaic_V01_E-176_N-16/MurrayLab_CTX_V01_E-176_N-16_Mosaic.tif')
img = np.array(img)
print('opened')
px = 47104
step=256 
newpx = 256
out_img = np.zeros((px,px))
print('init')
# import numpy as np 
# import os
# import tensorflow as tf
# import skimage.io as io
# import skimage.transform as trans
# import numpy as np
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
# from tensorflow.keras.losses import BinaryCrossentropy
# from keras import layers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model, load_model

import numpy as np
import os
import tensorflow as tf
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.losses import CategoricalCrossentropy

# from keras import layers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model, load_model
from PIL import Image, ImageOps
# from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred[0])
    y_true_f = tf.cast(y_true_f, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def loss_fn(y_true, y_pred):
    # y_pred = tf.expand_dims(y_pred, axis=-1)
    # print(y_pred.shape)
    # print(y_true.shape)
    # loss0 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[0], pos_weight=2, name=None)
    # loss1 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[1], pos_weight=2, name=None)
    # loss2 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[2], pos_weight=2, name=None)
    # loss3 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[3], pos_weight=2, name=None)
    # loss4 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[4], pos_weight=2, name=None)
    # loss5 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[5], pos_weight=2, name=None)
    # loss6 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[6], pos_weight=2, name=None)

    loss0 = cce(y_true, y_pred[0])
    loss1 = cce(y_true, y_pred[1])
    loss2 = cce(y_true, y_pred[2])
    loss3 = cce(y_true, y_pred[3])
    loss4 = cce(y_true, y_pred[4])
    loss5 = cce(y_true, y_pred[5])
    loss6 = cce(y_true, y_pred[6])
    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # return 1 - dice_coef(y_true,y_pred)

def REBNCONV(x, out_ch=3, dirate=1):
    #x = ZeroPadding2D((1*dirate,1*dirate))(x)
    x = tf.keras.layers.Conv2D(out_ch, 3, padding='same', dilation_rate = 1*dirate)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def _upsample_like(src, tar):
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = tf.keras.layers.UpSampling2D((h,w), interpolation='bilinear')(src)
    return src

def RSU7(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x4)

    x5 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x5)

    x6 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x6, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x6],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x5)

    x = REBNCONV(tf.concat([x,x5],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x4)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU6(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x4)

    x5 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x5],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x4)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU5(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x4],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x3)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU4(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)

    x = REBNCONV(x, mid_ch, 2)

    x = REBNCONV(tf.concat([x,x3],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x2)

    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 1)
    x = _upsample_like(x,x1)

    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def RSU4F(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x2 = REBNCONV(x1, mid_ch, 2)
    x3 = REBNCONV(x2, mid_ch, 4)

    x4 = REBNCONV(x3, mid_ch, 8)

    x = REBNCONV(tf.concat([x4,x3],axis=-1), mid_ch, 4)
    x = REBNCONV(tf.concat([x,x2],axis=-1), mid_ch, 2)
    x = REBNCONV(tf.concat([x,x1],axis=-1), out_ch, 1)

    return x + x0

def U2NET(x, out_ch=1):

    x1 = RSU7(x, 32, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 32, 128)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 64, 256)
    x = tf.keras.layers.MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 128, 512)
    x = tf.keras.layers.MaxPool2D(2, 2)(x4)

    x5 = RSU4F(x, 256, 512)
    x = tf.keras.layers.MaxPool2D(2, 2)(x5)

    x6 = RSU4F(x, 256, 512)
    x = _upsample_like(x6,x5)

    #-----------------decoder--------------------#

    x5 = RSU4F(tf.concat([x,x5],axis=-1),256, 512)
    x = _upsample_like(x5,x4)

    x4 = RSU4(tf.concat([x,x4],axis=-1),128, 256)
    x = _upsample_like(x4,x3)

    x3 = RSU5(tf.concat([x,x3],axis=-1),64, 128)
    x = _upsample_like(x3,x2)

    x2 = RSU6(tf.concat([x,x2],axis=-1),32, 64)
    x = _upsample_like(x2,x1)

    x1 = RSU7(tf.concat([x,x1],axis=-1),16, 64)

    #Side output
    x = tf.keras.layers.ZeroPadding2D((1,1))(x1)
    d1 = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d1 = tf.keras.layers.Activation('softmax')(d1)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x2)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 =tf.keras.layers.Activation('softmax')(d2)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x3)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = tf.keras.layers.Activation('softmax')(d3)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x4)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = tf.keras.layers.Activation('softmax')(d4)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x5)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = tf.keras.layers.Activation('softmax')(d5)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x6)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = tf.keras.layers.Activation('softmax')(d6)

    d0 = tf.keras.layers.Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = tf.keras.layers.Activation('softmax')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6])

def U2NETP(x, out_ch=1):

    x1 = RSU7(x, 16, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 16, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 16, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 16, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x4)

    x5 = RSU4F(x, 16, 64)
    x = tf.keras.layers.MaxPool2D(2, 2)(x5)

    x6 = RSU4F(x, 16, 64)
    x = _upsample_like(x6,x5)

    #---------------decoder--------------------
    x5 = RSU4F(tf.concat([x,x5],axis=-1),16, 64)
    x = _upsample_like(x5,x4)

    x4 = RSU4(tf.concat([x,x4],axis=-1),16, 64)
    x = _upsample_like(x4,x3)

    x3 = RSU5(tf.concat([x,x3],axis=-1),16, 64)
    x = _upsample_like(x3,x2)

    x2 = RSU6(tf.concat([x,x2],axis=-1),16, 64)
    x = _upsample_like(x2,x1)

    x1 = RSU7(tf.concat([x,x1],axis=-1),16, 64)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x1)
    d1 = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d1 = tf.keras.layers.Activation('sigmoid')(d1)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x2)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 = tf.keras.layers.Activation('sigmoid')(d2)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x3)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = tf.keras.layers.Activation('sigmoid')(d3)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x4)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = tf.keras.layers.Activation('sigmoid')(d4)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x5)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = tf.keras.layers.Activation('sigmoid')(d5)

    x = tf.keras.layers.ZeroPadding2D((1,1))(x6)
    x = tf.keras.layers.Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = tf.keras.layers.Activation('sigmoid')(d6)

    d0 = tf.keras.layers.Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = tf.keras.layers.Activation('sigmoid')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6])

net_input = tf.keras.layers.Input(shape=(512,512,1))

model_output = U2NETP(net_input)

model2 = tf.keras.models.Model(inputs = net_input, outputs = model_output)

lr = 1e-3

opt = tf.keras.optimizers.Adam(learning_rate = lr)

# bce = BinaryCrossentropy()
cce = tf.keras.losses.CategoricalCrossentropy()

model2.compile(optimizer = opt, loss = loss_fn, metrics = [dice_coef])
print('se')
model2.load_weights('u2net_dice_pyr_new_shuffle.h5')
print('re')
count = 0
for y in range(0,px,step): #no need to sub 512 b/c px are mult of 512
    for x in range(0,px,step):
        temp_img = img[x:x+newpx,y:y+newpx]
        # print(temp_img.shape)
        temp_img = cv2.resize(temp_img, (512,512))
        # print(temp_img.shape)
        preds = model2.predict(temp_img.reshape((1,512,512,1))/255,verbose=1)
        preds = preds[0].reshape((512,512))
        preds = cv2.resize(preds,(256,256))
        out_img[x:x+newpx,y:y+newpx] = preds*255
        count += 1
        if(count%5==0):
            print('processed',count)
        # break
    # break
print('ed')
cv2.imwrite('ctx_out_256.png',out_img)