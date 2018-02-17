import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

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
