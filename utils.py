import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import pandas as pd

import os
from os.path import join, basename, exists
from os import makedirs, listdir
import time

def load_data(path):
    data = pd.read_csv(path)
    data['Жалобы (ngramm)'] = data['Жалобы (ngramm)'].fillna('')
    
    return data

def get_most_popular_diagnoses(data, percent=.95):
    cumsums = np.cumsum(data.value_counts())
    most_classes = data.value_counts()[cumsums <= percent * len(data)]
    return list(most_classes.index)

def preproces_fit(data):
    # Complaints
    complaints = data['Жалобы (unigramm)']
    tfidf_complaints = TfidfVectorizer(ngram_range=(1,1), min_df=10, stop_words=stopwords.words('russian'))
    tfidf_complaints.fit(complaints)
    
    # Doctors
    doctors = train['Врач'].fillna('sss')
    doctors_voc, counts = np.unique(doctors, return_counts=True)
    pop_doctor = doctors_voc[np.argsort(counts)[::-1][0]]
    doctors[doctors == 'sss'] = pop_doctor
    
    vect_doctors = CountVectorizer()
    vect_doctors.fit(doctors)
    
    # Classes
    pop_diagnoses = set(utils.get_most_popular_diagnoses(data, percent=.8))
    most_pop_diagnose = scipy.stats.mode(data['Код_диагноза'])[0][0]
    y = data['Код_диагноза'].apply(
        lambda diag: diag if diag in pop_diagnoses else most_pop_diagnose
    )
    
    return tfidf_complaints, vect_doctors, pop_doctor, y

def preprocess_transform(tfidf_complaints, vect_doctors, pop_doctor, data):
    # Complaints
    complaints = data['Жалобы (unigramm)']
    
    # Doctors
    doctors = data['Врач'].fillna(pop_doctor)
    
    # Gender
    gender = data['Пол'].copy()
    gender[data['Пол'] == 1] = 0
    gender[data['Пол'] == 2] = 1
    
    # Repeats
    repeats = data['Повторный приём']
    
    # Age
    age = data['Возраст']
    
    print(doctors.shape, gender.shape, repeats.shape, age.shape)
    
    return np.hstack([
        tfidf_complaints.transform(complaints).todense(),
        vect_doctors.transform(doctors).todense(),
        np.expand_dims(gender, axis=1),
        np.expand_dims(repeats, axis=1),
        np.expand_dims(age, axis=1)
    ])

def get_next_model_id(experiment_dir):
    if not exists(experiment_dir):
        makedirs(experiment_dir)
    
    experiment_path = join(experiment_dir, '.model_ids.txt')
    if not exists(experiment_path):
        with open(experiment_path, 'w'):
            pass
        
    with open(experiment_path, 'r') as model_ids_f:
        model_ids = list(model_ids_f)
        model_ids = list(map(lambda str_id: int(str_id), model_ids))
        
    next_id = max(model_ids) + 1 if len(model_ids) > 0 else 0
    
    with open(experiment_path, 'a') as model_ids_f:
        model_ids_f.write(str(next_id) + '\n')
    
    return next_id

def get_model_full_path(models_path, model_name, experiment_postfix):
    time_str = '_'.join(time.ctime().split(' '))
    m_full_name = model_name + '_' + experiment_postfix + '_' + time_str
    model_full_name = join(models_path, m_full_name)
    if not exists(model_full_name):
        makedirs(model_full_name)
    
    return model_full_name

def get_model_fname_pattern(models_path, model_name):
    model_full_path = get_model_full_path(
        models_path, model_name, '')
    filepath = join(model_full_path, '{epoch:02d}_{val_acc:.2f}.h5')
    
    return filepath

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
