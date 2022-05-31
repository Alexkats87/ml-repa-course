# Tutorial: Predict Mobile Device change 
Прогноз смены мобильного девайса в течение месяца 

# Описание задачи

### Задача:
- бинарная классификация
- таргет: событие смены девайса в течение месяца
- целевые метрики - Lift, Precision@K, Recall@K

### Данные:

**Get dataset**
- Download data from [storage](https://yadi.sk/d/m-ZDxoFUyThiSQ)
- Put data to `data/raw`

**Таргет: target.feather**

    — user_id - ID пользователя
    — month - округление date до последнего дня месяца
    — target - 1/0 (1 - пользователь сменил девайс, 0 - не было изменений)
    - отчетные даты (month): '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31' 

**Таблица признаков: user_features.feather**

    — user_id - ID пользователя
    — month - месяц, за который собраны фичи
    — feature_N - конкретная фича (всего 50 фичей, есть категориальные)
    - отчетные даты (month): '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31' 
    
**Данные для скоринга**

Используются для симуляции использования модели в Production и мониторинга (не используем для обучения)
    
    - scoring_user_features.feather - таблица признаков для скоринга
    - scoring_target.feather - таблица целевых событий 
    - отчетные даты (month): '2020-09-30'
   
**Jupyter Notebooks**
- Download noteboks from [storage](https://yadi.sk/d/m-ZDxoFUyThiSQ)
- Put data to `notebooks/` (should be ignored by Git!)
   

### Валидация:

 - модель должна показывать устойчивость на последних 4 месяцах
 - валидация с учетом временной структуры, т.е. 4 фолда (по месяцам)
 - выбираем лучшую модель по усредненным метрикам по фолдам - mean, std, min, max
 
 ### Целевые метрики 
 
 Для текущей задачи предполагается, что вся выборка (база клиентов) может быть большой, 
 сотни тысяч или миллионы. Поэтому, с точки зрения бизнеса целесообразно выбрать top-K
 клиентов с максимальной вероятностью целевого события.
 Соответственно, нам интересно, чтобы модель максимально хорошо работала для такой выборки
 из top-K клиентов. 
  
 Для расчета метрик для top-K (или @k) полученные прогнозы сначала сортируют по predict_proba  (по убыванию) 
 и потом считают нужную метрику для этих @k объектов  
 
 - Precision@K
 - Recall@K
 - Lift@K - показывает насколько модель лучше работает по сравнению с random выборкой
 
 Для тьюториала значение К = 5 % от базы клиентов


# How to Run 

## 1. Fork / Clone repo
- fork to your personal repo 
- clone to you local machine


## 2. Use a virtual environment

Сreate and activate virtual environment
```bash
python3 -m venv venv
echo "export PYTHONPATH=$PWD" >> venv/bin/activate
source venv/bin/activate
```

Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Add virtual environment to Jupyter Notebook
```bash
python -m ipykernel install --user --name=venv
``` 

Run Jupyter Notebook 
```bash
jupyter notebook
```

To deactivate virtual environment: 
```bash
deactivate 
```



## 3. Or run with Docker 

Create config/.env
```bash
GIT_CONFIG_USER_NAME=<git user>
GIT_CONFIG_EMAIL=<git email>
```
example:

```.env
GIT_CONFIG_USER_NAME=mnrozhkov
GIT_CONFIG_EMAIL=mnrozhkov@gmail.com
```

Build
```bash
ln -sf config/.env 
docker-compose build
```
or 

```
docker-compose  --env-file ./config/.env build
```

Run 
```bash
docker-compose up
```

Open tutorial notebook:
- open http://localhost:8888 in browser;
