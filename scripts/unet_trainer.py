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
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def default_unet():
    z1 = Input(shape=(512,512,1))
    print('z1: {}'.format(z1.shape))

    z2 = Conv2D(16, 3, padding='same', activation='relu')(z1)
    p2 = AveragePooling2D(pool_size=2)(z2)
    print('z2: {}, \np2: {}'.format(z2.shape, p2.shape))

    z3 = Conv2D(24, 3, padding='same', activation='relu')(p2)
    p3 = AveragePooling2D(pool_size=2)(z3)
    print('z3: {}, \np3: {}'.format(z3.shape, p3.shape))

    z4 = Conv2D(32, 3, padding='same', activation='relu')(p3)
    d4 = Dropout(0.2)(z4)
    p4 = AveragePooling2D(pool_size=2)(d4)
    print('z4: {}, \np4: {}'.format(z4.shape, p4.shape))

    z5 = Conv2D(48, 3, padding='same', activation='relu')(p4)
    d5 = Dropout(0.2)(z5)
    p5 = AveragePooling2D(pool_size=2)(d5)
    print('z5: {}'.format(z5.shape))

    z6 = Conv2D(64, 3, padding='same', activation='relu')(p5)
    d6 = Dropout(0.3)(z6)
    p6 = AveragePooling2D(pool_size=2)(d6)

    z7 = Conv2D(96, 3, padding='same', activation='relu')(p6)
    d7 = Dropout(0.4)(z7)
    p7 = AveragePooling2D(pool_size=2)(d7)

    z8 = Conv2D(128, 3, padding='same', activation='relu')(p7)
    d8 = Dropout(0.5)(z8)
    p8 = AveragePooling2D(pool_size=2)(d8)

    z9 = Conv2D(128, 3, padding='same', activation='relu')(p8)
    d9 = Dropout(0.5)(z9)

    u9 = UpSampling2D(size=2)(d9)
    q9 = Conv2D(128, 3, padding='same', activation='relu')(u9)
    d9b = Dropout(0.5)(q9)
    a9 = Add()([d9b,z8])

    u8 = UpSampling2D(size=2)(a9)
    q8 = Conv2D(96, 3, padding='same', activation='relu')(u8)
    d8b = Dropout(0.4)(q8)
    a8 = Add()([d8b,z7])

    u7 = UpSampling2D(size=2)(a8)
    q7 = Conv2D(64, 3, padding='same', activation='relu')(u7)
    d7b = Dropout(0.3)(q7)
    a7 = Add()([d7b,z6])

    u6 = UpSampling2D(size=2)(a7)
    q6 = Conv2D(48, 3, padding='same', activation='relu')(u6)
    d6b = Dropout(0.2)(q6)
    a6 = Add()([d6b,z5])

    u5 = UpSampling2D(size=2)(a6)
    q5 = Conv2D(32, 3, padding='same', activation='relu')(u5)
    d5b = Dropout(0.2)(q5)
    a5 = Add()([d5b,z4])

    u4 = UpSampling2D(size=2)(a5)
    q4 = Conv2D(24, 3, padding='same', activation='relu')(u4)
    a4 = Add()([q4,z3])

    u3 = UpSampling2D(size=2)(a4)
    q3 = Conv2D(16, 3, padding='same', activation='relu')(u3)
    a3 = Add()([q3,z2])

    z_final = Conv2D(1, 3, padding='same', activation='sigmoid')(a3)
    #z8 activation = sigmoid or softmax

    return Model(inputs = z1, outputs = z_final)


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
    'trained_models/unet_13_500epochs_edges.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'

)
train = Mygenerator('out_new_edges/train/images','out_new_edges/train/mask',10)
val = Mygenerator('out_new_edges/val/images','out_new_edges/val/mask',10)
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor = 0.5,
#     patience=7,
#     verbose=1,
#     min_lr=1e-6,
#     min_delta=0.05
# )
model = default_unet()
# model = unet1()
model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=[dice_coef,'accuracy'])
# model.compile(optimizer = opt, loss = loss, metrics = [dice_coef])

STEP_SIZE_TRAIN=3375//10
STEP_SIZE_VALID=1800//10


# STEP_SIZE_TRAIN=tr_img.n//tr_img.batch_size
# STEP_SIZE_VALID=vl_img.n//vl_img.batch_size
history = model.fit_generator(train,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=val,validation_steps=STEP_SIZE_VALID,epochs=500, callbacks=[checkpoint])

img_data = []
pred_data = []
true_mask = []
for files in sorted(os.listdir('out_new_edges/test/images/')):
    # img = cv2.imread('out_new_edges/test/images/'+files,0)
    img = plt.imread('out_new_edges/test/images/'+files)
    img = 2*img - 1
    # img = cv2.resize(img,(512,512))
    preds = model.predict(img.reshape((1,512,512,1)))

    img_data.append(img)
    pred_data.append(preds[0].reshape((512,512)))
    truth = cv2.imread('out_new_edges/test/mask/'+files,0)
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
    cv2.imwrite('generated_masks/unet_13_500epochs_edges.png',tile)
    return tile

tile = create_tile(pred_data)