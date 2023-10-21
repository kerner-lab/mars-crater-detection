import numpy as np
import os
import tensorflow as tf
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.keras.losses import BinaryCrossentropy
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import matplotlib.pyplot as plt

from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred[0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def loss_fn(y_true, y_pred):
    # y_pred = tf.expand_ dims(y_pred, axis=-1)
    # print(y_pred.shape)
    # print(y_true.shape)
    # loss0 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[0], pos_weight=2, name=None)
    # loss1 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[1], pos_weight=2, name=None)
    # loss2 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[2], pos_weight=2, name=None)
    # loss3 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[3], pos_weight=2, name=None)
    # loss4 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[4], pos_weight=2, name=None)
    # loss5 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[5], pos_weight=2, name=None)
    # loss6 = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred[6], pos_weight=2, name=None)

    # loss0 = bce(y_true, y_pred[0])
    # loss1 = bce(y_true, y_pred[1])
    # loss2 = bce(y_true, y_pred[2])
    # loss3 = bce(y_true, y_pred[3])
    # loss4 = bce(y_true, y_pred[4])
    # loss5 = bce(y_true, y_pred[5])
    # loss6 = bce(y_true, y_pred[6])
    # return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return 1 - dice_coef(y_true,y_pred)

def REBNCONV(x, out_ch=3, dirate=1):
    #x = ZeroPadding2D((1*dirate,1*dirate))(x)
    x = Conv2D(out_ch, 3, padding='same', dilation_rate = 1*dirate)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x

def _upsample_like(src, tar):
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = UpSampling2D((h,w), interpolation='bilinear')(src)
    return src

def RSU7(x, mid_ch=12, out_ch=3):

    x0 = REBNCONV(x, out_ch, 1)

    x1 = REBNCONV(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x4)

    x5 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x5)

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
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

    x4 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x4)

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
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x3)

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
    x = MaxPool2D(2, 2)(x1)

    x2 = REBNCONV(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

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
    x = MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 32, 128)
    x = MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 64, 256)
    x = MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 128, 512)
    x = MaxPool2D(2, 2)(x4)

    x5 = RSU4F(x, 256, 512)
    x = MaxPool2D(2, 2)(x5)

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
    x = ZeroPadding2D((1,1))(x1)
    d1 = Conv2D(out_ch, 3)(x)
    d1 = Activation('sigmoid')(d1)

    x = ZeroPadding2D((1,1))(x2)
    x = Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 = Activation('sigmoid')(d2)

    x = ZeroPadding2D((1,1))(x3)
    x = Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = Activation('sigmoid')(d3)

    x = ZeroPadding2D((1,1))(x4)
    x = Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = Activation('sigmoid')(d4)

    x = ZeroPadding2D((1,1))(x5)
    x = Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = Activation('sigmoid')(d5)

    x = ZeroPadding2D((1,1))(x6)
    x = Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = Activation('sigmoid')(d6)

    d0 = Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = Activation('sigmoid')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6])

def U2NETP(x, out_ch=1):

    x1 = RSU7(x, 16, 64)
    x = MaxPool2D(2, 2)(x1)

    x2 = RSU6(x, 16, 64)
    x = MaxPool2D(2, 2)(x2)

    x3 = RSU5(x, 16, 64)
    x = MaxPool2D(2, 2)(x3)

    x4 = RSU4(x, 16, 64)
    x = MaxPool2D(2, 2)(x4)

    x5 = RSU4F(x, 16, 64)
    x = MaxPool2D(2, 2)(x5)

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

    x = ZeroPadding2D((1,1))(x1)
    d1 = Conv2D(out_ch, 3)(x)
    d1 = Activation('sigmoid')(d1)

    x = ZeroPadding2D((1,1))(x2)
    x = Conv2D(out_ch, 3)(x)
    d2 = _upsample_like(x,d1)
    d2 = Activation('sigmoid')(d2)

    x = ZeroPadding2D((1,1))(x3)
    x = Conv2D(out_ch, 3)(x)
    d3 = _upsample_like(x,d1)
    d3 = Activation('sigmoid')(d3)

    x = ZeroPadding2D((1,1))(x4)
    x = Conv2D(out_ch, 3)(x)
    d4 = _upsample_like(x,d1)
    d4 = Activation('sigmoid')(d4)

    x = ZeroPadding2D((1,1))(x5)
    x = Conv2D(out_ch, 3)(x)
    d5 = _upsample_like(x,d1)
    d5 = Activation('sigmoid')(d5)

    x = ZeroPadding2D((1,1))(x6)
    x = Conv2D(out_ch, 3)(x)
    d6 = _upsample_like(x,d1)
    d6 = Activation('sigmoid')(d6)

    d0 = Conv2D(out_ch, 1)(tf.concat([d1,d2,d3,d4,d5,d6],axis=-1))
    d0 = Activation('sigmoid')(d0)

    return tf.stack([d0,d1,d2,d3,d4,d5,d6])

net_input = Input(shape=(512,512,1))

model_output = U2NETP(net_input)

model = Model(inputs = net_input, outputs = model_output)

lr = 1e-3

opt = tf.keras.optimizers.Adam(learning_rate = lr)

bce = BinaryCrossentropy()

model.compile(optimizer = opt, loss = loss_fn, metrics = [dice_coef])


from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import os
from glob import glob

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = glob(x_set+'/*.png'), glob(y_set+'/*.png')
        self.batch_size = batch_size
        self.indices = np.arange(np.asarray(self.x).shape[0])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(self.x)[inds]
        batch_y = np.array(self.y)[inds]

        # read your data here using the batch lists, batch_x and batch_y
        x = []
        y = []
        # print(batch_x)
        for filename in batch_x:
            # print(filename)
            # img = cv2.imread(filename,0)
            # if(img.shape[0]<512):
            #   img = cv2.copyMakeBorder(img, 128, 128, 128, 128, cv2.BORDER_CONSTANT)
            # else:
            #   img = cv2.resize(img,(512,512))
            img = plt.imread(filename)
            # img = img/255
            img = 2*np.array(img)-1 
            x.append(img)

        for filename in batch_y:
            # img = cv2.imread(filename,0)
            # if(img.shape[0]<512):
            #   img = cv2.copyMakeBorder(img, 128, 128, 128, 128, cv2.BORDER_CONSTANT)
            # else:
            #   img = cv2.resize(img,(512,512))
            img = plt.imread(filename)
            img = np.where(img>0, 1, 0).astype(np.float32)
            y.append(img)

        return np.array(x), np.array(y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
# tr_img,tr_mask = gen('train')
# train = zip(tr_img,tr_mask)
# vl_img,vl_mask = gen('val')
# val = zip(vl_img,vl_mask)

checkpoint = ModelCheckpoint(
    'trained_models/u2net_13_40epochs_edges_2014_old.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'

)
train = Mygenerator('themis_2014/themis_2014_split_edges/train/images','out_new_edges/train/mask',8)
val = Mygenerator('themis_2014/themis_2014_split_edges/val/images','out_new_edges/val/mask',8)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6,
    min_delta=0.05
)
# model = default_unet()
# model = unet1()
# model.compile(optimizer=Adam(lr=1e-3),
#                   loss='binary_crossentropy',
#                   metrics=[dice_coef])
# model.compile(optimizer = opt, loss = loss, metrics = [dice_coef])

STEP_SIZE_TRAIN=3375//8
STEP_SIZE_VALID=1800//8


# STEP_SIZE_TRAIN=tr_img.n//tr_img.batch_size
# STEP_SIZE_VALID=vl_img.n//vl_img.batch_size
history = model.fit_generator(train,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=val,validation_steps=STEP_SIZE_VALID,epochs=40, callbacks=[ reduce_lr,checkpoint])

img_data = []
pred_data = []
true_mask = []
for files in sorted(os.listdir('themis_2014/themis_2014_split_edges/test/images/')):
    # img = cv2.imread('out_new_edges/test/images/'+files,0)
    # img = cv2.resize(img,(512,512))
    img = plt.imread('themis_2014/themis_2014_split_edges/test/images/'+files)
    img = 2*img - 1
    preds = model.predict(img.reshape((1,512,512,1)))

    img_data.append(img)
    pred_data.append(preds[0].reshape((512,512)))
    truth = cv2.imread('themis_2014/themis_2014_split_edges/test/mask/'+files,0)
    true_mask.append(truth)
    
    
def create_tile(img_list):
    step = 512
    newpx = 512
    px = 7680
    count = 0
    tile = np.zeros((7680,7680))
    for y in range(0,px,step): #no need to sub 512 b/c px are mult of 512
        for x in range(0,px,step):
            # print(x,y)
            target = img_list[count]
            target[target >= 0.1] = 1
            target[target < 0.1] = 0
            tile[x:x+newpx,y:y+newpx] = target*255
            count += 1
            # count += 1
            # break
    cv2.imwrite('generated_masks/u2net_13_40epochs_edges_2014_old.png',tile)
    return tile

tile = create_tile(pred_data)