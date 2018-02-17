def get_most_popular_diagnoses(data, percent=95):
    diagnoses = list(np.sort(np.unique(data['Код_диагноза'])))
    diag_to_num = dict(zip(diagnoses, range(len(diagnoses))))
    num_to_diag = dict(zip(range(len(diagnoses)), diagnoses))
    
    data = data.copy()
    data['diag_idx'] = data['Код_диагноза'].apply(lambda code: diag_to_num[code])
    diag_idxs, counts = np.unique(data['diag_idx'], return_counts=True)
    sorted_idxs = np.argsort(counts)[::-1]
    
    popular_idxs = diag_idxs[counts >= np.percentile(counts, q=percent)]
    popular_diagnoses = list(map(lambda idx: num_to_diag[idx], popular_idxs))
    
    return popular_diagnoses

def get_doctor(string):
    string = string.replace(',', ' ')
    splitted_str = string.split('врача-')
    if len(splitted_str) > 1:
        return splitted_str[1].split(' ')[0]
    else:
        return 'нет'