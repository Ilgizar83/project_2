# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# pip install folium
import folium
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# pip install category_encoders
# pip install -q catboost shap
import catboost
import catboost.datasets
import shap
import sklearn.model_selection
from sklearn import metrics
from catboost import Pool
import plotly.express as px
import seaborn as sns

"""источник данных - https://www.kaggle.com/competitions/sf-crime/data"""

# from google.colab import drive
# drive.mount('/content/drive')
# path = 'drive/MyDrive/'

df_test = pd.read_csv('Dataset/test.csv')
df_train = pd.read_csv("Dataset/train.csv")

df_train.info(), df_test.info()

df_train.head(3)

df_test.head(3)

df_train.duplicated().value_counts()

df_test.duplicated().value_counts()

df_train.isna().sum()

df_test.isna().sum()

"""Тренировочные данные представлены в виде таблицы из 878049 строк и 9 колонок:
Dates - (objecе) дата и время преступления
Category - (objecе) целевая колонка, категория преступления
Descript - (objecе) описание преступления
DayOfWeek - (objecе) день недели, в который совершено преступление
PdDistrict - (objecе) наименование полицейского департамента
Resolution - (objecе) решение, принятое по данному преступлению
Address -(objecе) адрес совершенния преступления
X -(float64) географическая координата
Y - (float64) географическая координата
Тренировочные данные насчитывают 2323 записей дубликатов, что составляет 0,26 % от всего набора данных. Пропущенные значения отсутствуют.
Тестовые данные представлены в виде таблицы из 884262 строк и 7 колонок
Id -(int64)
Dates - (objecе) дата и время преступления
DayOfWeek - (objecе) день недели, в который совершено преступление
PdDistrict - (objecе) наименование полицейского департамента
Address -(objecе) адрес совершенния преступления
X -(float64) географическая координата
Y - (float64) географическая координата
Тестовые дынные не наcчитывают записей дубликатов, а так же пропущенных значений.
"""

# Очистка данных
df_train_clear=df_train
df_train_clear.drop_duplicates()

df_train_clear[['X', 'Y']].describe()

# мы видим странное значение в координате Y - 90
# нанесем две точки с максимальным и минимальным значением на карту

from folium import plugins
m = folium.Map(location=[1, 1], zoom_start=1)

figure = folium.FeatureGroup(name="Все метки")
m.add_child(figure)

group1 = plugins.FeatureGroupSubGroup(figure, "минимальное значение")
m.add_child(group1)

group2 = plugins.FeatureGroupSubGroup(figure, "максимальное значение")
m.add_child(group2)

folium.Marker([37.707879,-122.513642]).add_to(group1)
folium.Marker([90.0000, -120.50000]).add_to(group2)

folium.LayerControl(collapsed=False).add_to(m)

m

# удалим выбросы

df_train_clear = df_train[df_train['Y'] < 90]

df_train_clear = df_train[['Category', 'Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']]

df_train_clear.shape

# Манипуляция с данными

df_train['Dates'] = pd.to_datetime(df_train['Dates'])
df_train['Dау'] = df_train.Dates.dt.day
df_train['Month'] = df_train.Dates.dt.month
df_train['Year'] = df_train.Dates.dt.year
df_train['Hour'] = df_train.Dates.dt.hour

def return_address(address):
  '''Функция для формирования классификации улиц (перекресток, улица, авиню и пр.) '''
  if ' / ' in address:
    return 'cross'
  address_separate = address.split(' ')
  address = address_separate[-1].lower()
  return address

df_train['addres_type'] = df_train['Address'].apply(return_address)

df_train = df_train[['Dates','Category',
   'DayOfWeek',
   'PdDistrict',
   'Address',
   'Dау',
   'Month',
   'Year',
   'Hour','addres_type',
   'X', 'Y' ]]

df_train.info()

"""Колонка Dates в тренировочном датасете была подвержена изменению и разбита на колонки:
'Day' - (int64) день
'Month' - (int64) месяц
'Year' - (int64) год
'Hour' - (int64) час
"""
# Разведочный анализ данных (EDA)

df_train['Category'].value_counts().plot(kind='bar', figsize=(20, 10))
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений по категориям')

df_train['Category'].value_counts()

df_train['PdDistrict'].value_counts().plot(kind='bar', figsize=(20, 10))
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений по полицейскому департаментам')

df_train['PdDistrict'].value_counts().plot(kind='pie', figsize=(20, 10), autopct='%1.1f%%')

plt.title('Распределение количества преступлений по полицейскому департаментам')

df_train['addres_type'].value_counts().plot(kind='bar', figsize=(20, 10), fontsize=20)
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение по типу адреса')

df_train['Month'].value_counts().sort_index().plot(kind='bar', figsize=(20, 10), fontsize=20)
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений по месяцам')

df_train['Hour'].value_counts().sort_index().plot(kind='bar', figsize=(20, 10), fontsize=20)
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений по времени в течении суток')

df_train['Year'].value_counts().sort_index().plot(kind='bar', figsize=(20, 10), fontsize=20)
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений по годам')

df_train['DayOfWeek'].value_counts().plot(kind='bar', figsize=(20, 10), fontsize=20)
plt.ylabel('Количество преступлений',fontsize=20)
plt.title('Распределение количества преступлений годам')

"""
В тренировочном датасете представлены наблюдения за период с 2003 по 2015 гг. Самым распространенным преступлением является преступление категории «воровство», 
за весь период наблюдений их было совершено 174305 ед. или 19,9 %. Вторым по распространенности является категория «Прочие правонарушения» с показателями 125943 ед. 
или 14,4%. На третьем месте категория «Неуголовные преступления» 91911 ед. или 10,5%
Распределение по количеству преступлений по полицейским департаментам не равномерное и варьируется от 5,1% в районе «RICHMOND» до 17,9% в районе «SOUTHERN»
Наибольшее количество преступлений совершается на улицах, перекрестках и авеню. Их суммарное значение 827871 или 94,5%
Наблюдается равномерное распределение количества преступлений по месяцам, среднее значение 72971 ед.
При рассмотрении количества преступлений по времени в течении суток наблюдается снижение числа преступлений с 01:00 до 07:00 часов
В 2015 наблюдается снижение количества преступлений, но это связанно не со снижением уровня преступности, а неполноты данных за год.
"""

# Обучение модели
# CatBoost
# В модель подаем очищеный датасет


df_train_clear.head()

df_train_clear.info()

y = df_train_clear.Category
X = df_train_clear.drop('Category', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y)

features = list(X_train.columns)

cat_features = [
    'Dates',
    'DayOfWeek',
    'PdDistrict',
    'Address',
    ]

pool_t = Pool(X_train,y_train,cat_features=cat_features)
pool_ts = Pool(X_test,label=y_test,cat_features=cat_features)
Pool_test = Pool(df_test,cat_features=cat_features)

clf = catboost.CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='AUC',
    od_pval=0.05, # порог детектора переобучения
    od_wait=20, # сколько итераций еще ждать после превышения порога
    # early_stopping_rounds=20, # если ошибка на тесте не падает на протяжении 20 итераций, то останавливаем переобучение
    # random_seed=28,
    # train_dir=catboost_train_dir,
    # task_type='CPU',
    task_type='GPU',
    # devices='0',
    auto_class_weights='Balanced',
    max_ctr_complexity=2, # кол-во комбинаций категориальных фичей
    use_best_model=True
)
clf.fit(
    pool_t,
    eval_set=pool_ts,
    plot=True,
    # save_snapshot=True,
    # snapshot_file='snapshot.bkp',
    verbose=True
)

print('model is fitted:'+str(clf.is_fitted()))

def model_score (model):
    train_acc = model.score(X_train, y_train)
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    test_acc = model.score(X_test, y_test)
    test_loss = log_loss(y_test, model.predict_proba(X_test))
    print(f'{model} train_score = {train_acc}')
    print(f'{model} log_loss = {train_loss}')
    print(f'{model} test_score = {test_acc}')
    print(f'{model} test_log_loss = {test_loss}')

    return print()

model_score (clf)

y_pred = clf.predict_proba(X_test)

pred = clf.predict_proba(Pool_test)

submission = pd.DataFrame(columns=['Id'], data=df_test)

submission = pd.concat([submission, pd.DataFrame(pred, columns=["ARSON", "ASSAULT", "BAD CHECKS", "BRIBERY", "BURGLARY", "DISORDERLY CONDUCT",
"DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC", "DRUNKENNESS", "EMBEZZLEMENT", "EXTORTION",
"FAMILY OFFENSES", "FORGERY/COUNTERFEITING", "FRAUD", "GAMBLING", "KIDNAPPING", "LARCENY/THEFT",
"LIQUOR LAWS", "LOITERING", "MISSING PERSON", "NON-CRIMINAL", "OTHER OFFENSES", "PORNOGRAPHY/OBSCENE MAT",
"PROSTITUTION", "RECOVERED VEHICLE", "ROBBERY", "RUNAWAY", "SECONDARY CODES", "SEX OFFENSES FORCIBLE",
"SEX OFFENSES NON FORCIBLE", "STOLEN PROPERTY", "SUICIDE", "SUSPICIOUS OCC", "TREA", "TRESPASS",
"VANDALISM", "VEHICLE THEFT", "WARRANTS", "WEAPON LAWS"])], axis=1)
submission.head()

