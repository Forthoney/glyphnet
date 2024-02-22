import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

# TensorFlow and tf.keras
import tensorflow as tf
from keras.api.keras.utils import to_categorical
from keras.api.keras.models import Model
from keras.api.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Dropout, GlobalAveragePooling2D
from keras.api.keras.optimizers import Adam
from tensorflow.keras import regularizers

allowed_labels = ['D2', 'D21', 'D36', 'D4', 'D46', 'D58', 'E23', 'E34', 'F31', 'F35', 'G1', 'G17',
                  'G43', 'I10', 'I9', 'M17', 'M23', 'N35', 'O1', 'O34', 'O4', 'O49', 'Q1', 'Q3',
                  'R4', 'R8', 'S29', 'S34', 'U7', 'V13', 'V28', 'V30', 'V31', 'W11', 'W24', 'X1',
                  'X8', 'Y1', 'Y5', 'Z1']

label_enc = preprocessing.LabelEncoder()
label_enc.fit(allowed_labels)

def add_extra_dim(imgs, labels, n_classes):
    if imgs.ndim == 3:
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    else:
        print("error: imgs dataset dimension is " +str(imgs.ndim)+" instead 3")
        return
    if labels.ndim == 1:
        labels = to_categorical(labels, num_classes=n_classes)
    else:
        print("error: labels dataset dimension is " +str(labels.ndim)+" instead 1")
        return
    return imgs, labels

def create_filelist(path, file_extension):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if(file.endswith("." + file_extension)):
                #append the file name to the list
                filelist.append(os.path.join(root,file))
    return filelist

def load_dataset(path, file_extension):
    filelist = create_filelist(path, file_extension)
    X, y = [], []
    for file in filelist:
        img = cv2.imread(file, 0) # opens the image in grayscale
        X.append(img)
        y.append(path_to_label(file)) # gets the label from the filename
    return X, y

def get_labels_number_in_category(labels, label_enc=None, view=False, sort=True):
    if labels.ndim == 2:
        labels = categorical_to_decoded(labels, label_enc)
        all_labels = label_enc.classes_
    else:
        all_labels = list(set(labels))
    dict_labels = dict.fromkeys(set(all_labels), 0) 
    for l in labels:
        dict_labels[l] = dict_labels[l] + 1
    #sorted
    if sort == True:
        sorted_dict = {}
        sorted_keys = sorted(dict_labels, key=dict_labels.get, reverse=True) 
        for w in sorted_keys:
            sorted_dict[w] = dict_labels[w]
        dict_labels = sorted_dict
    #view  
    if view == True:
        for k, v in dict_labels.items():
            print(k, v)
        print("tot labels number: " + str(len(dict_labels)))
    return dict_labels

def get_prediction_data(predictions, X_test, y_test, label_enc, summary=False, details=False, 
                        plot=(None,0,None), y_train=None):
    print(f'Test dataset dimensions: {X_test.shape}')
    # plot = (plot_type, plot_n, only_label_to_print)
    # plot_type  0 - only corrected pred
    #            1 - only wrong pred
    #            2 - corrected + wrong pred
    #            None
    # plot_n     int - number of images to plot
    #            "all" plot all images
    # only_label_to_print     "S29" - label of images to plot
    #            None
    if plot[0] is not None:
        plot_type = plot[0]
    else:
        plot_type = None
    plot_n = plot[1]
    only_label_to_print = plot[2]
    pred_corr = 0
    pred_wrong = 0
    plot_counter = 0
    y_test_decoded = label_enc.inverse_transform(np.argmax(y_test, axis=1))
    d_pred_corr = dict.fromkeys(set(y_test_decoded), 0)
    d_tot_label = dict.fromkeys(set(y_test_decoded), 0)
    if y_train is not None:
        d_train_label = get_labels_number_in_category(y_train, label_enc)
        
    for i,prediction in enumerate(predictions):
        p = np.argmax(prediction)
        true_label_enc = np.argmax(y_test[i])
        true_label = label_enc.inverse_transform([true_label_enc])[0]
        d_tot_label[true_label] = d_tot_label[true_label] +1
        
        if plot_n == "all":
            plot_n = len(predictions)
        
        if p == true_label_enc:
            pred_corr = pred_corr + 1
            d_pred_corr[true_label] = d_pred_corr[true_label] +1
            if(summary is True):
                print("test " + str(i+1)+"/"+str(len(predictions)) + " corrected prediction  " + "prediction = "+ str(p) +"  true_label_enc = "+ str(true_label_enc))
            
            if plot_type == 0 or plot_type == 2 :  
                if plot_counter < plot_n:
                    if only_label_to_print is not None:
                        if str(true_label) == only_label_to_print:
                            plt.figure(i)
                            plt.imshow(X_test[i], cmap="gray")
                            plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_corr)+"° corrected - " +"label "+ str(true_label))
                            plot_counter = plot_counter + 1
                    else:        
                        plt.figure(i)
                        plt.imshow(X_test[i], cmap="gray")
                        plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_corr)+"° corrected - " +"label "+ str(true_label))
                        plot_counter = plot_counter + 1
                    
        else:
            pred_wrong = pred_wrong + 1
            if summary is True:
                print("test " + str(i+1)+"/"+str(len(predictions)) + " wrong prediction")
            if plot_type == 1 or plot_type == 2:
                if plot_counter < plot_n:   
                    if only_label_to_print is not None:
                        if str(true_label) == only_label_to_print:
                            plt.figure(i)
                            plt.imshow(X_test[i], cmap="gray")
                            plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_wrong)+"° wrong prediction - " +"true_label:" +str(true_label)+ " - pred_label:" +str(label_enc.inverse_transform([p])[0]))
                            plot_counter = plot_counter + 1
                    else:
                        plt.figure(i)
                        print(X_test[i].shape)
                        plt.imshow(X_test[i], cmap="gray")
                        plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_wrong)+"° wrong prediction - " +"true_label:" +str(true_label)+ " - pred_label:" +str(label_enc.inverse_transform([p])[0]))
                        plot_counter = plot_counter + 1
    
    if summary is True:
        print("-------")
        print("Corrected predictions: " + str(pred_corr) + "/" + str(len(predictions)) )
        print("\n")
    
    if details is True:
        if y_train is None:
            for key in d_tot_label:
                print(str(round(d_pred_corr[key]/d_tot_label[key]*100)) +"% "+ str(d_pred_corr[key])+"/"+str(d_tot_label[key]) +" "+ key )
        else:
            for key in d_train_label:
                if key in d_tot_label:
                    print(str(d_train_label[key]) + " "+ str(round(d_pred_corr[key]/d_tot_label[key]*100)) +"% "+ str(d_pred_corr[key])+"/"+str(d_tot_label[key]) +" "+ key )    
                else:
                    print(str(d_train_label[key]) + " No images in test of label "+ str(key))
        print("-------")
        print("Correct prediction: " +str(sum(d_pred_corr.values()))+"/"+str(sum(d_tot_label.values())))
  
    if plot_type == 0 or plot_type == 1 or plot_type == 2 :
        plt.show()
        
    return

def path_to_label(path):
    # the filename format is yourfilename_LABEL.format, the LABEL is the gardiner code
    # and should be one of the allowed_labels listed above (case sensitive)
    file_name_parts = path.split('/')
    img_name = file_name_parts[-1]
    img_name_parts = img_name.split('_')
    lable = img_name_parts[-1].split('.')[0]
    return lable

def build_model():
    model = ATCNet(shape=(100, 100, 1), n_classes=n_classes)

    top3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    model.compile(optimizer=Adam(), 
                loss='categorical_crossentropy', 
                metrics=['accuracy', top3, top5])
    return model

def ATCNet(shape, n_classes):  

    # INPUT BLOCK

    input = Input(shape=shape)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='input_block_conv1')(input)
    x = BatchNormalization(name='input_block_conv1_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name = "input_block_conv1_pool")(x)  
    x = Activation('relu', name='input_block_conv1_act')(x)  
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='input_block_conv2')(x)
    x = BatchNormalization(name='input_block_conv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name = "input_block_conv2_pool")(x)  
    x = Activation('relu', name='input_block_conv2_act')(x)


    # MIDDLE BLOCKS

    # MIDDLE BLOCK 1
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block1_sepconv1')(x)
    x = BatchNormalization(name='middle_block1_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block1_sepconv1_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block1_sepconv2')(x)
    x = BatchNormalization(name='middle_block1_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block1_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block1_sepconv2_act')(x)

    # MIDDLE BLOCK 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block2_sepconv1')(x)
    x = BatchNormalization(name='middle_block2_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block2_sepconv1_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block2_sepconv2')(x)
    x = BatchNormalization(name='middle_block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block2_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block2_sepconv2_act')(x)

    # MIDDLE BLOCK 3
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block3_sepconv1')(x)
    x = BatchNormalization(name='middle_block3_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block3_sepconv2')(x)
    x = BatchNormalization(name='middle_block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block3_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block3_sepconv2_act')(x)

    # MIDDLE BLOCK 4
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block4_sepconv1')(x)
    x = BatchNormalization(name='middle_block4_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block4_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block4_sepconv2')(x)
    x = BatchNormalization(name='middle_block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block4_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block4_sepconv2_act')(x)

    # EXIT BLOCK 
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='exit_block_sepconv')(x)
    x = BatchNormalization(name='exit_block_sepconv_bn')(x)
    x = Activation('relu', name='exit_block_sepconv_act')(x)

    # TOP
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.15, name="dropout")(x)
    output = Dense(n_classes, activation='softmax', name='predictions', kernel_regularizer=regularizers.l2(0.01))(x)
    #output = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(input, output, name="ATCNet")
    return model


