#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import requests
import time
from functools import wraps
import traceback
import tqdm
import keras
from keras.models import save_model
from keras import backend as K
import tensorflow as tf

def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s


def download_file(url, file_path):
    if os.path.exists(file_path):
        return None
    else:
        r = requests.get(url, stream=True, allow_redirects=True, headers={'Accept-Encoding': None})
        total_size = int(r.headers.get('content-length'))
        bar = tqdm.tqdm_notebook(total=total_size, unit='B', unit_scale=True)
        bar.set_description(os.path.split(file_path)[-1])
        with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
            for chunk in r.iter_content(1 * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()
            f.close()
            
class ModelSaveCallback(keras.callbacks.Callback):

    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))
        
        

