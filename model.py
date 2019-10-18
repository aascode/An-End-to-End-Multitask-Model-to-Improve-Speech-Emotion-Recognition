#!/usr/bin/env python
# coding: utf-8

# In[1]:

from utils import *
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Conv2D, Conv1D, MaxPool1D, AveragePooling1D, BatchNormalization
from keras.layers.merge import dot
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as k
from keras_self_attention import SeqSelfAttention
import os
from keras_multi_head import MultiHeadAttention
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[3]:


max_len = 256
features_number = 128
hidden_unit = 512
dropout_rate = 0.35
lstm_cells = 128
classes = 7
batch = 64
epochs = 300


# In[ ]:


def SER_model(tra_data, tra_label_emo, tra_label_spea, val_data, val_label_emo, val_label_spea, max_len, features_num, hidden_unit, dp_rate, lstm_cells, classes, epochs, batch_size, taskname):
#     tra_label = to_categorical(tra_label, num_classes=classes)
#     val_label = to_categorical(val_label, num_classes=classes)
    u_train, u_val = attention_init(tra_data.shape[0], val_data.shape[0], 256, 1.0/256)
    file_path = './results/weights_'+ str(taskname) + '.h5'
    try:
        model=load_model(file_path, custom_objects={'MultiHeadAttention': MultiHeadAttention})
        print('load model')
        notification='continue'
    except:
        notification='new'
        with k.name_scope('CNN_BLSTMLayer'):
            ipt_features = Input(shape=(max_len, features_num))
            x = Conv1D(512,3,activation='relu',kernel_initializer='uniform',bias_initializer='zeros')(ipt_features)
            x = Conv1D(1024,3,activation='relu',kernel_initializer='uniform',bias_initializer='zeros')(x)
            x = Conv1D(2048,3,activation='relu',kernel_initializer='uniform',bias_initializer='zeros')(x)
            x = BatchNormalization(axis=-1, momentum=0.9)(x)

            x = Dense(hidden_unit, activation='relu',kernel_initializer='uniform',bias_initializer='zeros')(x)
            x = Dropout(dp_rate)(x)
    
            x = Bidirectional(LSTM(lstm_cells, return_sequences=True, dropout=dp_rate,kernel_initializer='uniform',bias_initializer='zeros'))(x)
            sp_x = MultiHeadAttention(head_num=16,name='multi-head1')(x)
            x = Bidirectional(LSTM(lstm_cells, return_sequences=True, dropout=dp_rate,kernel_initializer='uniform',bias_initializer='zeros'))(sp_x)
            y = MultiHeadAttention(head_num=16,name='multi-head2')(x)
        with k.name_scope('AttentionLayer'):
            ipt_attention = Input(shape=(lstm_cells*2,))
            u = Dense(lstm_cells*2, activation='softmax',kernel_initializer='uniform',bias_initializer='zeros')(ipt_attention)
            alp = dot([u,y], axes=-1)
            alp = Activation('softmax')(alp)
        with k.name_scope('WeightPooling'):
            z = dot([alp, y], axes=1) #utterance-level

            
        emo_opt = Dense(classes, activation='softmax',kernel_initializer='uniform',bias_initializer='zeros', name='emo_label')(z)
        spea_opt = Dense(14, activation='softmax',kernel_initializer='uniform',bias_initializer='zeros', name='speaker_label')(z)
        
        model = Model(inputs=[ipt_attention, ipt_features],outputs=[emo_opt,spea_opt])
    
    model.summary()
    print(notification)
#     optimizer = optimizers.SGD(lr=0.001, decay=1e-6, 
#                                momentum=0.9, 
#                                nesterov=True)
    model.compile(optimizer='rmsprop', 
                  loss={'emo_label':'categorical_crossentropy','speaker_label':'categorical_crossentropy'},
                  loss_weights={'emo_label':1.,'speaker_label':1.},
                  metrics=['accuracy'])
    
    callback_list = [
                    EarlyStopping(
                        monitor='val_emo_label_acc',
                        patience=50,
                        verbose=1,
                        mode='auto'
                    ),
                    ModelCheckpoint(
                        filepath=file_path,
                        monitor='val_emo_label_acc',
                        save_best_only='True',
                        verbose=1,
                        mode='auto',
                        period=1
                    )
                    ]

    training = model.fit([u_train, tra_data], [tra_label_emo,tra_label_spea], batch_size=batch_size, epochs=epochs, verbose=1,
                             callbacks=callback_list, 
                             validation_data=([u_val,val_data], [val_label_emo, val_label_spea]))
    model.save('./results/weights_'+ str(taskname) + '_final.h5')
    
        
#     u_test, _ = attention_init(x_test.shape[0], x_test.shape[0], 256, 1.0/256)
    final_result = model.evaluate([u_val,val_data], [val_label_emo,val_label_spea], batch_size=batch_size, verbose=1)
#     score_2, accuracy_2 = model.evaluate([u_val2,x_test2], y_test2, batch_size=128, verbose=1)
    print('*******************************************************')
    print('                     ------------ loss -------- emo_label_loss -------- emo_label_acc -------- speaker_label_loss ---------- speaker_label_acc ----------')
    print("Final test validation result: %s" % final_result)
#     print("Final test2 validation accuracy: %s" % accuracy_2)
    print('*******************************************************')
    
    
    best_model=load_model(file_path, custom_objects={'MultiHeadAttention': MultiHeadAttention})
    print('load best model')
    best_result = best_model.evaluate([u_val,val_data], [val_label_emo, val_label_spea], batch_size=batch_size, verbose=1)
    print('                     ------------ loss -------- emo_label_loss -------- emo_label_acc -------- speaker_label_loss ---------- speaker_label_acc ----------')
    print("best model validation result: %s" % best_result)
    
    history = training.history
    
    with open(save_file_blstm, 'wb'):
        np.savetxt('./history.csv', history)

    return model

