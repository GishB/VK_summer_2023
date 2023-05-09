# VK Internship task for summer 2023

В данном репозитории представлен пример решения задачи предсказания рейтинга фильма https://grouplens.org/datasets/movielens/latest/

Для решения задачи был рассмотрен следующий порядок работы:
  1. Feature_engineering_preprocessing - это 0 стадия работы. В аналогичном по названию ноутбуки содержится подход по обработки данных, который позволил собрать информацию о каждом пользователе userId и фильме movieId. На выходе получили подготовленные данные в формате .csv -> df.to_csv('dask_raw_df/df_raw_pandas.csv') 

    Для пользоватлей на этапе №1 были собраны следующие данные: (user_data)
      1. Теги пользователя по всем просмотренным фильмам -> list('Rayan Gosling','Fast cars', 'Shooting',...)
      2. Жанры всех фильмов, который смотрел пользователь -> list('Drama','Criminal story',...)
      3. Теги пользователя по фильмам, где он поставил 5* -> list('Rayan Gosling','Silent Driver','Rayan and car',...)
      4. Жанры всех фильмов, когда пользователь поставил 5* -> list('Action','Detective',...)
      5. Cредняя оценка пользователя по всем фильмам -> list(np.mean())*N (N-кол-во фильмов)
      6. Вариация оценки пользователя по всем фильмам -> list(np.var())*N
      7. Среднеквадратическая оценка пользователя по всем фильмам -> list(np.std())*N

    Для фильмов на этапе №1 были собраны следующие данные: (movie_data)
      1. Теги фильма которые ему поставили пользователи -> list('Rayan Gosling','a handsome guy', 'good driver','MMA',...)
      2. Жанр фильма, которые ему присвоили пользователи -> list('Drama','Criminal story','Action',...)
      3. Теги фильма, где пользователь поставил 5* -> list('Rayan Gosling','Silent Driver','Rayan and car',...)
      4. Жанры фильма, когда пользователь поставил 5* -> list('Action','Detective',...)
      5. Cредняя оценка пользователей по фильму -> list(np.mean())*N (N-кол-во фильмов)
      6. Вариация оценки пользователей по фильму -> list(np.var())*N
      7. Среднеквадратическая оценка пользователей по фильму -> list(np.std())*N


![Screenshot from 2023-05-10 00-47-55](https://github.com/GishB/VK_summer_2023/assets/90556084/f8992ca5-eff9-4824-a02b-d7636809921f)

2. Feature_engineering_BERT.ipynb - это 1 стадия работы. В аналогичном по названию ноутбуки содержится подход к кодировки данных на основе модели BERT из библиотеки transformers. Данные из пунктов 1-4 были объединенны в одну строку и закодированны в столбцах для каждого "user_text" (пользователя) и "movie_text" (фильма) соответсвенно. На выходе данные были подготовленны в формате .parquet -> 
df.to_parquet('BERT_encoded_and_stat_features.parquet', index=False)

3. CNN_experiment.ipynb - это 2 и финальная стадия работы. В аналогичном по названию ноутбуки содержится подход к использованию модели CNN из библиотеки PyTorch для задачи регрессии (предективного анализа оценки фильма пользователем на основе подготовленных данных).

  Архитектура сети CNN:
  
    class CNN(nn.Module): 
      def __init__(self): 
          super().__init__()
          self.conv1 = nn.Conv2d(4, 4, 2)
          self.pool = nn.AvgPool2d(kernel_size=(2, 2))
          self.conv2 = nn.Conv2d(4,4,2) 
          self.fc1 = nn.Linear(472, 512)
          self.drop_0 = nn.Dropout(0.1)
          self.fc2 = nn.Linear(512, 512)
          self.drop_1 = nn.Dropout(0.3)
          self.fc3 = nn.Linear(512, 248)
          self.fc4 = nn.Linear(248,64)
          self.fc5 = nn.Linear(64, 1)

      def forward(self, x): 
          x = self.pool(F.relu(self.conv1(x)))
          x = F.relu(self.conv2(x))
          x = x.reshape(x.shape[0], -1)
          x = F.relu(self.fc1(x))
          x = self.drop_0(x)
          x = F.relu(self.fc2(x))
          x = self.drop_1(x)
          x = F.relu(self.fc3(x))
          x = self.drop_1(x)
          x = F.relu(self.fc4(x))
          x = self.fc5(x)
          return x

          
Результат обучения модели при следующих параметрах:

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    
![Untitled](https://github.com/GishB/VK_summer_2023/assets/90556084/a22c7871-4d5c-4403-83ac-65fb74791f9c)
![Screenshot from 2023-05-10 01-05-47](https://github.com/GishB/VK_summer_2023/assets/90556084/c43ac833-e762-438f-a583-2c049cf2257f)


# Результат:

  Модель CNN позволяет с степенью ошибки в среднем порядка 0.84 MSE и стандартным отклонением ~0.2 ед., предективно анализировать рейтинг, который поставить пользователь фильму. Для данного подхода достаточно представить входящие данные в следующем виде:
  
    1. Кодированный текст предпочтений пользователя;
    2. Кодированный текст предпочтений пользователей для данного фильма;
    3. Статестическую информацию оценок пользователя фильмам/оценки фильмов.
    
  Можно ли улучшить работу модели? Гипотетически мы можем взять полностью архитектуру модели GPT под данную задачу и/или модифицировать CNN при помощи слоев памяти LSTM, GRU и внимания MultiheadAttention, так как данные содержат в себе текстовую информацию, где такой подход может дать преимущество.
