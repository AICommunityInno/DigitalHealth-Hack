def get_most_popular_diagnoses(data, percent=95):
    cumsums = np.cumsum(train_df.Код_диагноза.value_counts())
    most_classes = data.Код_диагноза.value_counts()[cumsums <= percent * len(train_df)]
    return list(most_classes.index)

def get_doctor(string):
    string = string.replace(',', ' ')
    splitted_str = string.split('врача-')
    if len(splitted_str) > 1:
        return splitted_str[1].split(' ')[0]
    else:
        return 'нет'