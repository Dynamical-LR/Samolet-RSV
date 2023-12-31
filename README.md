# Модель предсказания вероятности покупки машиноместа

Особенность задачи - большое количество признаков, 99% пропусков, большинство информации содержится в меньшистве признаков. Учитывая мотивацию и особенности данных, мы рассматриваем задачу как рекомендательную систему, в которой пропуски в данных - неслучайное отсутствие взаимодействия с объектами-признаками. Целевая переменная же рассматривается не как абстрактный класс, а как один из объектов-признаков, а сама модель лишь заполняет пропуски основываясь на статистике. Такой подход позволяет не только интерпретируемо решить поставленную задачу машиномест, но и позволяет масштабировать систему для предсказания взаимодействия с другими возможными услугами.

Техничесие особенности: использование полных данных, EASE модель, подбор параметров с кросс-валидацией по месяцам и клиентам.

Уникальность решения: статистическая обоснованность и масштабируемость для дальнейших бизнес-задач.
