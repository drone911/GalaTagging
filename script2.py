import numpy as np
import pandas as pd
import cv2 as cv
import os

import matplotlib.pyplot as plt
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
from sklearn.utils import class_weight
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model, Sequential
from sklearn.metrics import classification_report
from keras.optimizers import Adam, Nadam, SGD
import pdb
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
IMAGES_DIR = os.curdir + '/Images'
ANNOTATIONS_DIR = os.curdir + '/Annotations'
TRAIN_IMAGES_DIR = IMAGES_DIR + '/Train Images'
TEST_IMAGES_DIR = IMAGES_DIR + '/Test Images'
CLASSES = {0:'Food', 1:'Attire', 2:'Decorationandsignage', 3:'misc'}
IMAGE_SIZE = (150, 150)
IMAGE_DIM = (150, 150, 3)
BATCH_SIZE = 32
    
def print_layers(model):
    for i, layer in enumerate(model.layers):
        print(i, layer, layer.trainable)
def plot_images(images, labels = [], image_tag=[]):
    fig, axes = plt.subplots(5, 5, figsize = (12, 8))
    for i in range(5):
        for j in range(5):
            k = i*5 + j
            if(images[k].any()):
                axes[i, j].set_axis_off()
                axes[i,j].imshow(images[k])
            if(labels):
                if(labels[k]):
                    axes[i,j].set_title(labels[k])  
                    if(image_tag):
                        if(image_tag[k]):
                            axes[i,j].set_title(image_tag[k].split('.')[0].lstrip("image") +" " + labels[k])
                            
    plt.plot()

def load_dataset(train_file_name = "", test_file_name=""):
    train_images = []
    test_images = []
    
    train_df = pd.read_csv(ANNOTATIONS_DIR + '/train.csv', index_col = 0)
    test_df = pd.read_csv(ANNOTATIONS_DIR + '/test.csv', index_col = 0)
    train_images_names = train_df.index
    train_labels = list(train_df['Class'])
    test_images_names = test_df.index
    
    
    if(train_file_name):
        train_images = list(np.load(train_file_name, allow_pickle = True))
    else:
        for train_images_name in train_images_names:
            image = cv.imread(TRAIN_IMAGES_DIR + '/' + train_images_name, cv.IMREAD_COLOR)
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            train_images.append(image)
    if(test_file_name):
        test_images = list(np.load(test_file_name, allow_pickle = True))
    else:
        for test_images_name in test_images_names:
            image = cv.imread(TEST_IMAGES_DIR + '/' + test_images_name, cv.IMREAD_COLOR)
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            test_images.append(image)
    
    return (train_images, train_labels, test_images, train_df, test_df)
    
def preprocess_labels(train_labels, smoothening_factor = 0.1):
    train_labels = list(map(lambda val:list(CLASSES.keys())[list(CLASSES.values()).index(val)], train_labels))
    train_labels= np.array(train_labels, dtype='int16')
    train_labels = to_categorical(train_labels)
    train_labels *= (1-smoothening_factor)
    train_labels += smoothening_factor / len(train_labels[0])
    return train_labels

def normalize_image(img):
    img = np.copy(img)
    for channel in range(img.shape[-1]):
        channeled_img = img[:,:,channel]
        img[:,:,channel] = (channeled_img  - np.min(channeled_img ))/(np.max(channeled_img )-np.min(channeled_img))
    return img
def preprocess_image(image):
    image = cv.resize(image, IMAGE_SIZE)
    return image

def get_accuracy(predictions, original):
    acc = 0
    tot = 0
    for i in range(len(predictions)):
        pred = np.argmax(predictions[i])
        actual = np.argmax(original[i])
        if pred == actual:
            acc+=1
        tot+=1
    return acc/tot

#dont use this function ever
def get_ZCA_matrix(images, epsilon = 1e-06):
    images= np.array(images)
    images= images.reshape((len(images), np.prod(images.shape[1:])))
    sigma = np.cov(images, rowvar = False)
    pdb.set_trace()
    U,S,V = np.linalg.svd(sigma)
    pdb.set_trace()
    return np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

def lr_scheduler():
    return 0.0001
#
#def apply_ZCA_matrix(images, zca):
#    for i in range(len(images));:
        
if __name__ == "__main__":
    
    (train_images_total, train_labels_total, final_test_images, train_df, test_df) = load_dataset()
    ct = Counter(train_labels_total)
    
    print(ct)
    
    class_weights = class_weight.compute_class_weight('balanced', np.array(['Food','Attire','Decorationandsignage','misc']),train_labels_total )
    #randomize
    train_data = list(zip(train_images_total, train_labels_total))
    np.random.shuffle(train_data)
    train_images_total, train_labels_total= zip(*train_data)
    train_images_total= list(train_images_total)
    train_labels_total = list(train_labels_total)
    
    #preprocessing
    train_images_total = list(map(preprocess_image, train_images_total))
    train_labels_oh_total = preprocess_labels(train_labels_total, smoothening_factor = 0.1)
    
    train_images_total = np.array(train_images_total, 'float32')
    #train_images_total = list(map(normalize_image, train_images_total))
    #split
    
    split_index = int(len(train_labels_total)*1)
    
    train_images = train_images_total[0:split_index]
    train_labels_oh = train_labels_oh_total[0:split_index]
    train_labels = train_labels_total[0:split_index]
    
    test_images = train_images_total[split_index:]
    test_labels_oh = train_labels_oh_total[split_index:]
    test_labels = train_labels_total[split_index:]
    
    #imageDataGen
    #zca = get_ZCA_matrix(train_images)
#    train_images = apply_ZCA_matrix(train_images, zca)
    
    #plot_images(train_images[25:50], train_labels[25:50])
    train_datagen_params = {
            "featurewise_center":False, 
            "samplewise_center":False, 
            "featurewise_std_normalization":False, 
            "samplewise_std_normalization":False, 
            "zca_whitening": False,
            "zca_epsilon": 1e-06,
            "rotation_range":15, 
            "width_shift_range":0.2,
            "height_shift_range":0.2, 
            "brightness_range":None,
            "shear_range":0.2,
            "zoom_range":0.1, 
            "channel_shift_range":0.0, 
            "fill_mode":'reflect', 
            "cval":0.0,
            "horizontal_flip":True, 
            "vertical_flip":False, 
            "rescale":None, 
            "preprocessing_function":None, 
            "data_format":'channels_last', 
            "validation_split":0.2}
    
    
    train_datagen = ImageDataGenerator(**train_datagen_params)
    #train_datagen.fit(np.array(train_images, 'float32'))
    
    #train_datagen.fit(np.array(train_images, 'float32'))
    #image_iterator = train_datagen.flow(np.array(train_images), train_labels, batch_size = 25, save_to_dir = "./Images/Preprocessed Images")
    #images = list(image_iterator[0][0])
    #labels= list(image_iterator[0][1])
    #plot_images(images[0:25], labels[0:25])
    
    
    #model
    input_layer = Input(shape = IMAGE_DIM, name = 'input-layer-1')
    intermediate_model = ResNet50(input_tensor = input_layer, include_top = False, weights = 'imagenet', pooling = 'max')
    
    features = intermediate_model.output
    
    #print(intermediate_model.summary())
    dp1 = Dropout(rate = 0.1)(features)
    dense2 = Dense(512,activation = 'relu', name = 'dense2')(dp1)
    dp2 = Dropout(rate = 0.1)(dense2)
    dense3 = Dense(256,activation = 'relu', name = 'dense3')(dp2)
    dense4 = Dense(4,activation = 'softmax', name = 'dense4')(dense3)
    model = Model(inputs = input_layer, outputs = dense4)
    
    #model = Model(inputs = input_layer, outputs = dense4)
    
    for layer in intermediate_model.layers:
        layer.trainable = False
    
    
    early_stopping = EarlyStopping(monitor= 'loss', patience = 3, min_delta = 0.01)
    model_checkpoint = ModelCheckpoint('./models/resNetWeightedSmoothMax.h5', monitor = 'loss', save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1, min_delta = 0.01)
    callbacks = [early_stopping , model_checkpoint,reduce_lr]
    
    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    
    #training
    hist1 = model.fit_generator(train_datagen.flow(np.array(train_images, 'float32'), train_labels_oh, batch_size = BATCH_SIZE), callbacks = callbacks, class_weight = class_weights,epochs = 10, steps_per_epoch = len(train_labels_oh)//BATCH_SIZE)
    
    for layer in model.layers[165:]:
        layer.trainable = True
    
    train_datagen_params = {
            "featurewise_center":False, 
            "samplewise_center":False, 
            "featurewise_std_normalization":False, 
            "samplewise_std_normalization":False, 
            "zca_whitening": False,
            "zca_epsilon": 1e-06,
            "rotation_range":20, 
            "width_shift_range":0.3,
            "height_shift_range":0.3, 
            "brightness_range":None,
            "shear_range":0.3,
            "zoom_range":0.3, 
            "channel_shift_range":0.0, 
            "fill_mode":'reflect', 
            "cval":0.0,
            "horizontal_flip":True, 
            "vertical_flip":False, 
            "rescale":None, 
            "preprocessing_function":None, 
            "data_format":'channels_last', 
            "validation_split":0.2}
    
    
    train_datagen = ImageDataGenerator(**train_datagen_params)
    
    model.compile(optimizer = SGD(lr=0.0001, momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    
    early_stopping = EarlyStopping(monitor= 'loss', patience = 3)
    model_checkpoint = ModelCheckpoint('./models/vggStdWeightedv2tweaked.h5', monitor = 'loss', save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)
    callbacks = [early_stopping , model_checkpoint,reduce_lr]
    
    hist2 = model.fit_generator(train_datagen.flow(np.array(train_images, 'float32'), train_labels_oh, batch_size = BATCH_SIZE), callbacks = callbacks, class_weight = class_weights,epochs = 10, steps_per_epoch = len(train_labels)//BATCH_SIZE)
    
    test_images = np.array(test_images, 'float32')
#    for images in test_images:
#        images = train_datagen.standardize(images)
#   
    #test_images = np.array(list(map(preprocess_image, test_images)))
    predictions = model.predict(test_images)
    
    accuracy = get_accuracy(predictions, test_labels_oh)
    print("acc: "+ str(accuracy))

    
    #final submission
    
    pr_final_test_images = np.array(list(map(preprocess_image, final_test_images)), 'float32')
#    pr_final_test_images  = list(map(normalize_image, pr_final_test_images))
    
    final_predictions = model.predict(np.array(pr_final_test_images))
    final_predictions = [np.argmax(x) for x in final_predictions]
    final_predictions = list(map(lambda x: CLASSES[x],final_predictions))
    final_df = pd.DataFrame({ 'Class': final_predictions}, index = test_df.index)
    final_df.to_csv('./submissions/resNetWeightedSmoothMax.csv')
    Counter(final_df['Class'])
    df = pd.read_csv('./temp1.csv')
    Counter(df['Class'])
    
    