import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

tfhub_handle_preprocess = 'data/bert_en_uncased_preprocess_3'
tfhub_handle_encoder = 'data/small_bert'

def lossNet():

    keras_emb = hub.KerasLayer(tfhub_handle_preprocess, name='BERT_preprocess_model',trainable=True) #bert_preprocess_model
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder_model')  # bert_model
    
    sentence_inp = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sen_inp')
    sen_emb = keras_emb(sentence_inp)
    sen_emb = encoder(sen_emb)['pooled_output']
    
    sen_dense = tf.keras.layers.Dense(32, activation="relu",name='sen_dense')(sen_emb)
    pred = tf.keras.layers.Dense(6, activation="softmax",name='pred_label')(sen_dense)   

    dense = tf.keras.layers.Dense(32, activation="relu",name='pred_loss_dense2')(sen_dense)
    predloss = tf.keras.layers.Dense(1,name='pred_loss')(dense)

    inp_wt = tf.keras.layers.Input(shape=(1,), name='weight_target_model')
    inp_wl = tf.keras.layers.Input(shape=(1,), name='weight_loss_module')

    pred_concat = tf.keras.layers.Concatenate(name='outputs')([pred, predloss,inp_wt,inp_wl])
    
    model = tf.keras.Model(inputs=[sentence_inp, inp_wt, inp_wl], outputs=pred_concat, name="lossNet")

    return model 

