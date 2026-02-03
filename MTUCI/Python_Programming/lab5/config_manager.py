# config_manager.py
# ==================
# Расширенный менеджер конфигураций машинного обучения для аутентификации по поведенческим биометрическим данным
# 
# Описание модуля:
# ---------------
# Данный модуль представляет собой комплексную систему для экспериментов с различными алгоритмами машинного обучения
# в задаче аутентификации пользователей по поведенческим биометрическим признакам (динамика нажатий клавиш и движения мыши)
# 
# Основные возможности:
# ---------------------
# 1. Поддержка множества алгоритмов ML: логистическая регрессия, SVM, случайный лес, градиентный бустинг,
#    XGBoost, CatBoost, LSTM нейронные сети
# 2. Загрузка и предобработка данных: поддержка различных форматов датасетов, нормализация, обработка пропусков
# 3. Реальное обучение моделей с настраиваемыми параметрами
# 4. Вычисление метрик качества: accuracy, precision, recall, F1-score, ROC-AUC, EER (Equal Error Rate)
# 5. Логирование всех этапов работы
# 6. Сохранение и загрузка конфигураций экспериментов в JSON формате
# 
# Архитектура и принципы ООП:
# ----------------------------
# - Инкапсуляция: использование приватных атрибутов и свойств (properties)
# - Наследование: базовый класс ModelConfig с множеством подклассов для разных алгоритмов
# - Полиморфизм: единый интерфейс train_and_evaluate() для всех моделей
# - Композиция: класс Experiment объединяет ModelConfig, TrainingConfig и DatasetHandler
# - Магические методы: __str__, __repr__ для удобного представления объектов


import json
import os
import logging
import random
import pandas as pd
import numpy as np

# scikit-learn: классические алгоритмы машинного обучения и утилиты
from sklearn.model_selection import train_test_split  # Разделение данных на обучающую и тестовую выборки
from sklearn.metrics import (  # Метрики качества моделей
    accuracy_score,      # Точность классификации
    precision_score,     # Прецизионность (точность положительных предсказаний)
    recall_score,        # Полнота (чувствительность)
    f1_score,           # F1-мера (гармоническое среднее precision и recall)
    roc_auc_score       # Площадь под ROC-кривой
)
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия
from sklearn.svm import SVC  # Метод опорных векторов (Support Vector Machine)
from sklearn.ensemble import (  # Ансамблевые методы
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler  # Стандартизация признаков (z-score нормализация)
from sklearn.impute import SimpleImputer  # Заполнение пропущенных значений
from sklearn.metrics import roc_curve  # Построение ROC-кривой для вычисления EER

# XGBoost: градиентный бустинг с оптимизацией
import xgboost as xgb

# CatBoost: градиентный бустинг с обработкой категориальных признаков
import catboost as cb

# PyTorch: фреймворк для глубокого обучения
import torch
import torch.nn as nn  # Модули нейронных сетей
import torch.optim as optim  # Оптимизаторы для обучения
from torch.utils.data import DataLoader, TensorDataset  # Загрузчики данных для обучения

from scipy.optimize import brentq  # Поиск корня функции (для вычисления EER)
from scipy.interpolate import interp1d  # Интерполяция функций (для вычисления EER)

# Настройка системы логирования
# ------------------------------
# Настраивает глобальное логирование для всего модуля:
# - Уровень INFO: вывод информационных сообщений о ходе выполнения
# - Формат: время, уровень важности, сообщение
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Создание логгера для данного модуля (используется во всех классах)
logger = logging.getLogger(__name__)

class DatasetHandler:
    """
    Класс для загрузки, предобработки и управления наборами данных поведенческой биометрии.
    Реализует паттерн композиции и используется в классе Experiment для работы с данными.
    Поддерживает три режима работы с различными типами датасетов:
    1. 'keystroke' - данные о динамике нажатий клавиш (CMU Keystroke Dataset)
    2. 'mouse' - данные о движениях мыши (Balabit Mouse Dynamics Dataset)
    3. 'generic' - универсальный режим для произвольных CSV файлов
    
    Основные возможности:
    - Автоматическая загрузка данных из CSV файлов
    - Извлечение признаков по префиксам колонок или по указанным колонкам
    - Обработка пропущенных значений (импутация средними значениями)
    - Нормализация признаков (стандартизация: z-score)
    - Разделение данных на обучающую и тестовую выборки
    - Валидация данных перед использованием
    """
    def __init__(self, dataset_type: str, data_path: str, 
                 feature_prefixes: list = None, target_column: str = None,
                 normalize: bool = True, handle_missing: bool = True):
        """
        Инициализация обработчика датасета.
        
        Параметры:
        dataset_type : str
            Тип датасета:
            - 'keystroke': данные о динамике нажатий клавиш (CMU формат)
            - 'mouse': данные о движениях мыши (Balabit формат)
            - 'generic': универсальный режим для произвольных CSV файлов
            
        data_path : str
            Путь к CSV файлу с данными. Файл должен быть доступен для чтения.
            
        feature_prefixes : list, optional
            Список префиксов для поиска колонок с признаками.
            Для keystroke по умолчанию: ['H.', 'DD.', 'UD.']
            - H.* - времена удержания клавиш (Hold times)
            - DD.* - интервалы между нажатиями (Down-Down intervals)
            - UD.* - интервалы между отпусканием и нажатием (Up-Down intervals)
            
        target_column : str, optional
            Имя колонки с целевой переменной (метками пользователей).
            Обязательно для режима 'generic'.
            Для 'keystroke' по умолчанию используется 'subject'.
            Для 'mouse' по умолчанию используется 'user_id'.
            
        normalize : bool, default=True
            Применять ли нормализацию данных (стандартизацию).
            Нормализация преобразует признаки к среднему 0 и стандартному отклонению 1.
            Рекомендуется для алгоритмов, чувствительных к масштабу признаков (SVM, нейронные сети).
            
        handle_missing : bool, default=True
            Обрабатывать ли отсутствующие значения (NaN).
            Если True, пропущенные значения заполняются средними значениями соответствующих признаков.
        """
        # Сохранение параметров конфигурации
        self.dataset_type = dataset_type  # Тип датасета
        self.data_path = data_path  # Путь к файлу с данными
        self.feature_prefixes = feature_prefixes  # Префиксы для поиска признаков
        self.target_column = target_column  # Имя колонки с метками
        self.normalize = normalize  # Флаг нормализации
        self.handle_missing = handle_missing  # Флаг обработки пропусков
        
        # Инициализация атрибутов для данных (будут заполнены в load_data)
        self.X = None  # Массив признаков (features)
        self.y = None  # Массив меток (labels)
        
        # Инициализация объектов предобработки (будут созданы при необходимости)
        self.scaler = None  # Нормализатор (StandardScaler)
        self.imputer = None  # Импутер для пропущенных значений (SimpleImputer)
        
        # Автоматическая загрузка и предобработка данных при создании объекта
        self.load_data()

    def load_data(self):
        """
        Загрузить и предобработать набор данных из CSV файла.
        """
        # Шаг 1: Загрузка CSV файла
        # Попытка прочитать CSV файл с помощью pandas
        # pandas автоматически определяет разделители и типы данных
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Загружен CSV файл: {df.shape[0]} строк, {df.shape[1]} колонок")
        except Exception as e:
            # Если произошла ошибка (файл не найден, неправильный формат и т.д.)
            raise ValueError(f"Ошибка загрузки файла {self.data_path}: {e}")
        
        # Шаг 2: Извлечение признаков и меток в зависимости от типа датасета
        if self.dataset_type == 'keystroke':
            """
            Обработка датасета динамики нажатий клавиш (CMU Keystroke Dataset).
            
            Формат данных:
            - Признаки имеют префиксы: H.* (Hold times), DD.* (Down-Down), UD.* (Up-Down)
            - Метки находятся в колонке 'subject' (идентификатор пользователя)
            """
            # Определение префиксов для поиска признаков
            # Если не указаны явно, используются стандартные для keystroke датасета
            feature_prefixes = self.feature_prefixes or ['H.', 'DD.', 'UD.']
            
            # Поиск всех колонок, начинающихся с указанных префиксов
            # Это признаки, характеризующие временные характеристики нажатий клавиш
            feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
            
            # Валидация: проверка наличия признаков
            if not feature_cols:
                raise ValueError(f"Не найдены колонки с префиксами {feature_prefixes}")
            
            # Поиск колонки с метками (идентификаторы пользователей)
            # В CMU датасете это колонка 'subject'
            target_col = 'subject' if 'subject' in df.columns else None
            if target_col is None:
                raise ValueError("Не найдена колонка 'subject' для keystroke датасета")
            
            # Извлечение признаков (используется copy() для избежания предупреждений pandas)
            self.X = df[feature_cols].copy()
            
            # Извлечение меток: преобразование категориальных значений в числовые коды
            # astype('category') создает категориальный тип, cat.codes преобразует в числа
            self.y = df[target_col].astype('category').cat.codes
            
        elif self.dataset_type == 'mouse':
            """
            Обработка датасета динамики движений мыши (Balabit Mouse Dynamics Dataset).
            
            Формат данных:
            - Все колонки кроме целевой считаются признаками
            - Метки находятся в колонке 'user_id' (идентификатор пользователя)
            - Датасет должен быть предобработан (признаки уже извлечены из траекторий)
            """
            # Определение колонки с метками
            # По умолчанию используется 'user_id', но можно указать другую
            target_col = self.target_column or 'user_id'
            
            # Проверка наличия целевой колонки
            if target_col not in df.columns:
                # Попытка найти похожие колонки (содержащие 'user' или 'id' в названии)
                # Это позволяет работать с датасетами, где колонка называется по-другому
                possible_cols = [col for col in df.columns if 'user' in col.lower() or 'id' in col.lower()]
                if possible_cols:
                    target_col = possible_cols[0]
                    logger.warning(f"Используется колонка '{target_col}' вместо 'user_id'")
                else:
                    raise ValueError(f"Не найдена колонка '{target_col}' для mouse датасета")
            
            # Извлечение признаков: все колонки кроме целевой
            self.X = df.drop(target_col, axis=1).copy()
            
            # Извлечение меток как массив значений (не категориальный тип)
            self.y = df[target_col].values
            
        elif self.dataset_type == 'generic':
            """
            Универсальный режим для произвольных CSV файлов.
            
            Формат данных:
            - Все колонки кроме указанной целевой считаются признаками
            - Целевая колонка должна быть явно указана в параметре target_column
            """
            # Валидация: для generic режима обязательно указание целевой колонки
            if self.target_column is None:
                raise ValueError("Для generic режима необходимо указать target_column")
            
            # Проверка наличия указанной колонки в данных
            if self.target_column not in df.columns:
                raise ValueError(f"Колонка '{self.target_column}' не найдена в данных")
            
            # Извлечение признаков: все колонки кроме целевой
            self.X = df.drop(self.target_column, axis=1).copy()
            
            # Извлечение меток из указанной колонки
            self.y = df[self.target_column].values
            
        else:
            # Обработка неизвестного типа датасета
            raise ValueError(
                f"Неподдерживаемый тип датасета: {self.dataset_type}. "
                f"Используйте 'keystroke', 'mouse' или 'generic'."
            )
        
        # Шаг 3: Валидация извлеченных данных
        # Проверка на пустоту DataFrame (нет признаков)
        if self.X.empty:
            raise ValueError("Не найдены признаки для обучения")
        
        # Проверка на отсутствие строк (пустой датасет)
        if len(self.X) == 0:
            raise ValueError("Датасет пуст")
        
        # Шаг 4: Обработка отсутствующих значений (импутация)
        # Если включена обработка пропусков и они обнаружены
        if self.handle_missing:
            # Подсчет общего количества пропущенных значений
            missing_count = self.X.isnull().sum().sum()
            
            if missing_count > 0:
                logger.info(f"Обнаружено {missing_count} отсутствующих значений, применяется импутация")
                
                # Создание импутера со стратегией 'mean' (заполнение средними значениями)
                # Можно использовать 'median' или 'most_frequent' в зависимости от задачи
                self.imputer = SimpleImputer(strategy='mean')
                
                # Применение импутации: fit_transform обучает на данных и сразу преобразует
                # Результат преобразуется обратно в DataFrame для сохранения структуры
                self.X = pd.DataFrame(
                    self.imputer.fit_transform(self.X),
                    columns=self.X.columns,  # Сохранение имен колонок
                    index=self.X.index       # Сохранение индексов
                )
        
        # Шаг 5: Нормализация данных (стандартизация)
        # Нормализация преобразует признаки к среднему 0 и стандартному отклонению 1
        # Это важно для алгоритмов, чувствительных к масштабу (SVM, нейронные сети)
        if self.normalize:
            logger.info("Применяется нормализация данных")
            
            # Создание нормализатора (StandardScaler)
            self.scaler = StandardScaler()
            
            # Применение нормализации: fit_transform вычисляет среднее и ст.отклонение, затем нормализует
            # Результат преобразуется обратно в DataFrame для сохранения структуры
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns,  # Сохранение имен колонок
                index=self.X.index       # Сохранение индексов
            )
        
        # Шаг 6: Преобразование в numpy массивы
        # Преобразование DataFrame в numpy.ndarray для использования в ML моделях
        # Большинство библиотек ML (scikit-learn, PyTorch) работают с numpy массивами
        self.X = self.X.values
        
        # Логирование итоговой информации о загруженном датасете
        logger.info(
            f"Загружен датасет {self.dataset_type}: "
            f"{self.X.shape[0]} образцов, "
            f"{self.X.shape[1]} признаков, "
            f"{len(np.unique(self.y))} классов"
        )

    def split_data(self, test_size=0.2, random_state=None, shuffle=True):
        """
        Разделить данные на обучающую и тестовую выборки.
            
        Возвращает: tuple из 4 numpy.ndarray
        Кортеж (X_train, X_test, y_train, y_test), где:
        - X_train: массив признаков для обучения
        - X_test: массив признаков для тестирования
        - y_train: массив меток для обучения
        - y_test: массив меток для тестирования
        """
        # Использование функции train_test_split из scikit-learn
        # Она автоматически обрабатывает стратификацию и перемешивание
        return train_test_split(
            self.X,           # Признаки
            self.y,           # Метки
            test_size=test_size,        # Доля тестовой выборки
            random_state=random_state,  # Seed для воспроизводимости
            shuffle=shuffle             # Перемешивание данных
        )

class ModelConfig:
    """
    Базовый абстрактный класс для всех конфигураций моделей машинного обучения
    Каждый подкласс (LogisticRegressionConfig, SVMConfig, и т.д.) реализует
    методы build_model() и train_and_evaluate() для конкретного алгоритма
    
    Принципы ООП:
    1. Инкапсуляция:
       - Приватные атрибуты (_model_name, _input_size, _output_size)
       - Доступ через свойства (properties)
       
    2. Наследование:
       - Все конфигурации моделей наследуются от ModelConfig
       - Переиспользование общих методов (to_dict, from_dict, __str__, __repr__)
       
    3. Полиморфизм:
       - Единый интерфейс train_and_evaluate() для всех моделей
       - Различная реализация в каждом подклассе
       - Позволяет работать с разными моделями через один интерфейс

    Магические методы:
       - __str__ и __repr__ для удобного представления объектов
    """
    def __init__(self, model_name: str, input_size: int, output_size: int):
        """
        Инициализация базовой конфигурации модели
        """
        # Приватные атрибуты (инкапсуляция)
        self._model_name = model_name    # Название модели
        self._input_size = input_size    # Размерность входных признаков
        self._output_size = output_size  # Размерность выходного пространства (количество классов)

    @property
    def model_name(self) -> str:
        """
        Свойство для получения названия модели.
        """
        return self._model_name

    @property
    def input_size(self) -> int:
        """
        Свойство для получения размерности входных признаков.
        """
        return self._input_size

    @property
    def output_size(self) -> int:
        """
        Свойство для получения размерности выходного пространства.
        """
        return self._output_size

    def build_model(self):
        """
        Абстрактный метод для построения экземпляра модели машинного обучения
        Метод должен быть реализован в каждом подклассе ModelConfig.
        Созданная модель должна поддерживать методы fit() и predict() (или аналогичные).
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        """
        Абстрактный метод для обучения модели и оценки её качества.
        Каждый подкласс реализует свою логику обучения и оценки.
        Процесс работы метода:
        1. Создание модели через build_model()
        2. Обучение модели на обучающих данных
        3. Получение предсказаний на тестовых данных
        4. Вычисление метрик качества
        5. Возврат словаря с метриками
 
        Возвращает словарь с метриками качества модели:
        - 'accuracy': float - точность классификации (доля правильных предсказаний)
        - 'precision': float - прецизионность (точность положительных предсказаний)
        - 'recall': float - полнота (чувствительность, recall)
        - 'f1_score': float - F1-мера (гармоническое среднее precision и recall)
        - 'roc_auc': float - площадь под ROC-кривой (для бинарной классификации)
        - 'eer': float - Equal Error Rate (для бинарной классификации)
        """
        raise NotImplementedError("Subclasses must implement train_and_evaluate()")

    def calculate_params(self) -> int:
        """
        Оценить количество обучаемых параметров модели
        Метод возвращает оценку количества параметров, которые обучаются в модели
        Это полезно для анализа сложности модели и сравнения разных архитектур
        """
        return 0  # Базовая реализация (подклассы могут переопределить)

    def to_dict(self) -> dict:
        """
        Сериализовать конфигурацию модели в словарь.
        Метод преобразует объект конфигурации в словарь для сохранения в JSON файл или передачи между процессами
        Подклассы должны вызывать super().to_dict() и добавлять свои специфичные параметры
        """
        return {
            "class": self.__class__.__name__,      # Имя класса для десериализации
            "model_name": self.model_name,        # Название модели
            "input_size": self.input_size,        # Размерность входных признаков
            "output_size": self.output_size        # Размерность выходного пространства
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Десериализовать конфигурацию модели из словаря
        Классовый метод (classmethod) для создания объекта конфигурации из словаря
        Использует имя класса из словаря для определения правильного подкласса
        Если класс не найден в словаре подклассов, возвращается базовый класс ModelConfig
        """
        # Словарь соответствия имен классов и классов
        # Позволяет динамически создавать объекты нужного типа
        subclasses = {
            "LogisticRegressionConfig": LogisticRegressionConfig,
            "SVMConfig": SVMConfig,
            "RandomForestConfig": RandomForestConfig,
            "GradientBoostingConfig": GradientBoostingConfig,
            "XGBoostConfig": XGBoostConfig,
            "CatBoostConfig": CatBoostConfig,
            "LSTMConfig": LSTMConfig
        }
        
        # Получение класса по имени из словаря
        # Если класс не найден, используется базовый класс cls
        subclass = subclasses.get(data["class"], cls)
        
        # Создание экземпляра класса с параметрами из словаря
        # Исключаем ключ 'class', так как он не является параметром конструктора
        return subclass(**{k: v for k, v in data.items() if k != "class"})

    def __str__(self) -> str:
        """
        Возвращает неформальное строковое представление объекта
        Магический метод для преобразования объекта в строку
        Используется функциями str() и print()
        """
        return f"{self.__class__.__name__}({self.model_name})"

    def __repr__(self) -> str:
        """
        Возвращает формальное строковое представление объекта
        Магический метод для получения "официального" представления объекта
        Используется функцией repr() и в отладчиках
        """
        return f"<{self.__class__.__name__}: {self.model_name}>"

# ============================================================================
# ПОДКЛАССЫ ДЛЯ РАЗЛИЧНЫХ АЛГОРИТМОВ МАШИННОГО ОБУЧЕНИЯ
# ============================================================================
# Каждый подкласс реализует специфичную конфигурацию для конкретного алгоритма.
# Все подклассы наследуются от ModelConfig и реализуют методы build_model() и train_and_evaluate().

class LogisticRegressionConfig(ModelConfig):
    """
    Конфигурация для модели логистической регрессии.

    Логистическая регрессия - это линейный алгоритм классификации, который использует
    логистическую функцию (сигмоиду) для моделирования вероятности принадлежности к классу

    Преимущества:
    - Быстрое обучение и предсказание
    - Интерпретируемость (можно анализировать веса признаков)
    - Хорошо работает на линейно разделимых данных
    - Поддерживает регуляризацию для предотвращения переобучения

    Недостатки:
    - Предполагает линейную зависимость между признаками и логарифмом шансов
    - Может плохо работать на нелинейных данных
    """
    def __init__(self, model_name: str = "LogisticRegression", input_size: int = 31, output_size: int = 51,
                 max_iter: int = 100, C: float = 1.0, penalty: str = 'l2', 
                 solver: str = 'lbfgs', tol: float = 1e-4, class_weight: str = None):
        """
        Инициализация конфигурации логистической регрессии.
        """
        # Вызов конструктора базового класса
        super().__init__(model_name, input_size, output_size)
        
        # Сохранение параметров модели
        self.max_iter = max_iter        # Максимальное количество итераций
        self.C = C                      # Параметр регуляризации
        self.penalty = penalty          # Тип регуляризации
        self.solver = solver            # Алгоритм оптимизации
        self.tol = tol                  # Толерантность для остановки
        self.class_weight = class_weight  # Веса классов

    def build_model(self):
        """
        Создать экземпляр модели логистической регрессии.
        Для многоклассовой классификации (output_size > 2) используется 'multinomial'
        Для бинарной классификации используется 'auto' (автоматический выбор)
        Модель еще не обучена, обучение происходит в train_and_evaluate()
        """
        # Определение стратегии для многоклассовой классификации
        # 'multinomial' - для 3+ классов (использует softmax)
        # 'auto' - для бинарной классификации (использует sigmoid)
        multi_class = 'multinomial' if self.output_size > 2 else 'auto'
        
        # Создание и возврат модели с указанными параметрами
        return LogisticRegression(
            max_iter=self.max_iter,      # Максимальное количество итераций
            C=self.C,                    # Параметр регуляризации
            penalty=self.penalty,        # Тип регуляризации
            solver=self.solver,          # Алгоритм оптимизации
            tol=self.tol,               # Толерантность для остановки
            class_weight=self.class_weight,  # Веса классов
            multi_class=multi_class      # Стратегия для многоклассовой классификации
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        """
        Обучить модель логистической регрессии и оценить её качество

        Метод выполняет полный цикл обучения и оценки:
        1. Создание модели через build_model()
        2. Обучение модели на обучающих данных
        3. Получение предсказаний на тестовых данных
        4. Вычисление метрик качества

        Для бинарной классификации берется вероятность положительного класса
        Для многоклассовой классификации используются все вероятности классов
        """
        # Адаптация параметров из TrainingConfig
        # Если передан training_config, используем epochs как max_iter
        if training_config and hasattr(training_config, 'epochs'):
            original_max_iter = self.max_iter  # Сохранение оригинального значения
            self.max_iter = training_config.epochs  # Временное изменение для обучения
        
        # Создание модели
        model = self.build_model()
        
        # Восстановление оригинального значения max_iter
        if training_config and hasattr(training_config, 'epochs'):
            self.max_iter = original_max_iter
        
        # Обучение модели на обучающих данных
        # fit() находит оптимальные веса модели, минимизируя функцию потерь
        model.fit(X_train, y_train)
        
        # Получение предсказаний классов на тестовых данных
        # predict() возвращает класс с наибольшей вероятностью
        y_pred = model.predict(X_test)
        
        # Получение вероятностей классов на тестовых данных
        # predict_proba() возвращает вероятности для каждого класса
        if self.output_size == 2:
            # Для бинарной классификации берем вероятность положительного класса
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Для многоклассовой классификации используем все вероятности
            y_prob = model.predict_proba(X_test)
        
        # Вычисление и возврат метрик качества
        return self._compute_metrics(y_test, y_pred, y_prob)

    def _compute_metrics(self, y_true, y_pred, y_prob):
        """
        Вычислить метрики качества модели классификации.
        Этот метод используется не только LogisticRegressionConfig, но и другими конфигурациями
        через наследование или прямое обращение к методу
        """
        # Вычисление метрик качества
        metrics = {
            # Accuracy (точность): доля правильных предсказаний
            # Формула: (TP + TN) / (TP + TN + FP + FN)
            "accuracy": accuracy_score(y_true, y_pred),
            
            # Precision (прецизионность): точность положительных предсказаний
            # Формула: TP / (TP + FP)
            # 'macro': среднее значение precision по всем классам
            # zero_division=0: возвращает 0 вместо ошибки при делении на ноль
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            
            # Recall (полнота, чувствительность): доля найденных положительных примеров
            # Формула: TP / (TP + FN)
            # 'macro': среднее значение recall по всем классам
            "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            
            # F1-score: гармоническое среднее precision и recall
            # Формула: 2 * (precision * recall) / (precision + recall)
            # Балансирует между precision и recall
            "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            
            # ROC-AUC: площадь под ROC-кривой
            # Показывает способность модели различать классы
            # Для многоклассовой классификации используется стратегия 'ovr' (one-vs-rest)
            "roc_auc": (
                roc_auc_score(y_true, y_prob, multi_class='ovr') 
                if self.output_size > 2 
                else roc_auc_score(y_true, y_prob)
            ),
            
            # EER (Equal Error Rate): точка, где False Positive Rate = False Negative Rate
            # Важная метрика для задач биометрической аутентификации
            # Вычисляется только для бинарной классификации
            "eer": (
                self._compute_eer(y_true, y_prob) 
                if self.output_size == 2 
                else 0.0
            )
        }
        return metrics

    def _compute_eer(self, y_true, y_scores):
        """
        Вычислить Equal Error Rate (EER) для бинарной классификации.

        EER - это точка на ROC-кривой, где False Positive Rate (FPR) равен
        False Negative Rate (FNR). Это важная метрика для задач биометрической
        аутентификации, так как она показывает точку оптимального баланса между
        ошибками первого и второго рода.
        
        Алгоритм:
        1. Построение ROC-кривой (FPR vs TPR)
        2. Поиск точки, где FPR = 1 - TPR (что эквивалентно FPR = FNR)
        3. Использование численного метода brentq для нахождения корня
        """
        try:
            # Построение ROC-кривой
            # fpr: False Positive Rate (доля ложных срабатываний)
            # tpr: True Positive Rate (чувствительность, recall)
            # thresholds: пороги для классификации
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Поиск EER: точки, где FPR = FNR
            # FNR = 1 - TPR, поэтому ищем точку, где FPR = 1 - TPR
            # Это эквивалентно FPR + TPR = 1, или 1 - FPR - TPR = 0
            # Используем интерполяцию для получения непрерывной функции
            interp_func = interp1d(fpr, tpr)
            
            # Поиск корня уравнения: 1 - x - interp_func(x) = 0
            # где x - это FPR, а interp_func(x) - это TPR
            # brentq находит корень в интервале [0, 1]
            eer = brentq(lambda x: 1. - x - interp_func(x), 0., 1.)
            
            # Преобразование в float для сериализации
            return float(eer)
        except:
            # Если вычисление не удалось (например, недостаточно точек на ROC-кривой),
            # возвращаем 0.0 как значение по умолчанию
            return 0.0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "max_iter": self.max_iter,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "tol": self.tol,
            "class_weight": self.class_weight
        })
        return data

class SVMConfig(ModelConfig):
    """
    Конфигурация для метода опорных векторов (Support Vector Machine, SVM).
    
    SVM - мощный алгоритм классификации, который находит оптимальную разделяющую гиперплоскость
    между классами. Может работать с линейными и нелинейными данными через ядерный трюк (kernel trick).
    
    Преимущества:
    - Эффективен на данных с высокой размерностью
    - Хорошо работает с нелинейными данными (через ядра)
    - Устойчив к переобучению при правильной настройке C
    
    Недостатки:
    - Медленное обучение на больших датасетах
    - Требует нормализации данных
    - Чувствителен к выбору гиперпараметров (C, gamma, kernel)
    """
    def __init__(self, model_name: str = "SVM", input_size: int = 31, output_size: int = 51,
                 kernel: str = "rbf", C: float = 1.0, gamma: str = 'scale', 
                 degree: int = 3, coef0: float = 0.0, class_weight: str = None, tol: float = 1e-3):
        """
        Инициализация конфигурации SVM.
        
        Параметры:
        ----------
        kernel : str, default="rbf"
            Тип ядра: 'rbf' (радиально-базисная функция, рекомендуется),
            'linear' (линейное), 'poly' (полиномиальное), 'sigmoid'
        C : float, default=1.0
            Параметр регуляризации (меньше = сильнее регуляризация)
        gamma : str/float, default='scale'
            Параметр ядра ('scale', 'auto' или числовое значение)
        degree : int, default=3
            Степень полинома для 'poly' ядра
        coef0 : float, default=0.0
            Независимый член для 'poly' и 'sigmoid' ядер
        class_weight : str/dict, optional
            Веса классов ('balanced' или словарь)
        tol : float, default=1e-3
            Толерантность для остановки оптимизации
        """
        super().__init__(model_name, input_size, output_size)
        self.kernel = kernel              # Тип ядра
        self.C = C                        # Параметр регуляризации
        self.gamma = gamma                # Параметр ядра
        self.degree = degree              # Степень полинома
        self.coef0 = coef0                # Независимый член
        self.class_weight = class_weight  # Веса классов
        self.tol = tol                    # Толерантность

    def build_model(self):
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            class_weight=self.class_weight,
            tol=self.tol,
            probability=True
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # SVC не имеет прямого параметра epochs/max_iter в fit.
        # Используем модель как есть.
        model = self.build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if self.output_size == 2 else model.predict_proba(X_test)
        return LogisticRegressionConfig._compute_metrics(self, y_test, y_pred, y_prob)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "class_weight": self.class_weight,
            "tol": self.tol
        })
        return data

# ============================================================================
# КОНФИГУРАЦИИ ДЛЯ АНСАМБЛЕВЫХ МЕТОДОВ
# ============================================================================

class RandomForestConfig(ModelConfig):
    """
    Конфигурация для случайного леса (Random Forest).

    Random Forest - ансамблевый метод, объединяющий множество решающих деревьев.
    Каждое дерево обучается на случайной подвыборке данных и признаков.
    Финальное предсказание - среднее (регрессия) или голосование (классификация) всех деревьев.
    
    Преимущества:
    - Высокая точность на многих задачах
    - Устойчивость к переобучению
    - Оценка важности признаков
    - Работает с категориальными признаками
    
    Недостатки:
    - Медленное обучение на больших датасетах
    - Большой объем памяти
    - Менее интерпретируем, чем одно дерево
    """
    def __init__(self, model_name: str = "RandomForest", input_size: int = 31, output_size: int = 51,
                 n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: str = 'sqrt', 
                 bootstrap: bool = True, class_weight: str = None, random_state: int = None):
        """
        Инициализация конфигурации Random Forest.
        """
        super().__init__(model_name, input_size, output_size)
        self.n_estimators = n_estimators          # Количество деревьев
        self.max_depth = max_depth                # Максимальная глубина
        self.min_samples_split = min_samples_split  # Минимум образцов для разделения
        self.min_samples_leaf = min_samples_leaf   # Минимум образцов в листе
        self.max_features = max_features           # Количество признаков
        self.bootstrap = bootstrap                 # Бутстрэп-выборки
        self.class_weight = class_weight            # Веса классов
        self.random_state = random_state           # Seed для воспроизводимости

    def build_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # Используем n_estimators из TrainingConfig.epochs, если передан
        if training_config and hasattr(training_config, 'epochs'):
            original_n_estimators = self.n_estimators
            self.n_estimators = training_config.epochs
        
        model = self.build_model()
        
        if training_config and hasattr(training_config, 'epochs'):
            self.n_estimators = original_n_estimators # Восстанавливаем оригинальное значение
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if self.output_size == 2 else model.predict_proba(X_test)
        return LogisticRegressionConfig._compute_metrics(self, y_test, y_pred, y_prob)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "class_weight": self.class_weight,
            "random_state": self.random_state
        })
        return data

class GradientBoostingConfig(ModelConfig):
    """
    Конфигурация для градиентного бустинга (Gradient Boosting).
    
    Описание:
    ---------
    Gradient Boosting - ансамблевый метод, который последовательно добавляет слабые модели
    (обычно деревья), каждая из которых исправляет ошибки предыдущих.
    Использует градиентный спуск для минимизации функции потерь.
    
    Преимущества:
    - Очень высокая точность
    - Гибкость (разные функции потерь)
    - Работает с различными типами данных
    
    Недостатки:
    - Медленное обучение
    - Чувствителен к переобучению
    - Требует тщательной настройки гиперпараметров
    """
    def __init__(self, model_name: str = "GradientBoosting", input_size: int = 31, output_size: int = 51,
                 n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 min_samples_split: int = 2, min_samples_leaf: int = 1, subsample: float = 1.0,
                 loss: str = 'deviance', random_state: int = None):
        """
        Инициализация конфигурации Gradient Boosting.
        """
        super().__init__(model_name, input_size, output_size)
        self.n_estimators = n_estimators          # Количество деревьев
        self.learning_rate = learning_rate        # Скорость обучения
        self.max_depth = max_depth                # Максимальная глубина
        self.min_samples_split = min_samples_split  # Минимум для разделения
        self.min_samples_leaf = min_samples_leaf    # Минимум в листе
        self.subsample = subsample                 # Доля образцов
        self.loss = loss                          # Функция потерь
        self.random_state = random_state          # Seed

    def build_model(self, learning_rate=None):
        """Построить модель с возможностью переопределения learning_rate."""
        lr = learning_rate if learning_rate is not None else self.learning_rate
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            loss=self.loss,
            random_state=self.random_state
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # Используем n_estimators и learning_rate из TrainingConfig, если переданы
        if training_config:
            if hasattr(training_config, 'epochs'):
                original_n_estimators = self.n_estimators
                self.n_estimators = training_config.epochs
            if hasattr(training_config, 'learning_rate'):
                original_lr = self.learning_rate
                self.learning_rate = training_config.learning_rate
        
        model = self.build_model()
        
        if training_config:
            if hasattr(training_config, 'epochs'):
                self.n_estimators = original_n_estimators # Восстанавливаем оригинальное значение
            if hasattr(training_config, 'learning_rate'):
                self.learning_rate = original_lr # Восстанавливаем оригинальное значение
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if self.output_size == 2 else model.predict_proba(X_test)
        return LogisticRegressionConfig._compute_metrics(self, y_test, y_pred, y_prob)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "loss": self.loss,
            "random_state": self.random_state
        })
        return data

class XGBoostConfig(ModelConfig):
    """
    Конфигурация для XGBoost (eXtreme Gradient Boosting).

    XGBoost - оптимизированная реализация градиентного бустинга с дополнительными
    техниками регуляризации и эффективными алгоритмами. Один из самых популярных
    алгоритмов для соревнований по машинному обучению.
    
    Преимущества:
    - Очень высокая точность
    - Быстрое обучение (параллелизация)
    - Встроенная регуляризация
    - Работает с пропущенными значениями
    
    Недостатки:
    - Много гиперпараметров для настройки
    - Может переобучаться при неправильной настройке
    """
    def __init__(self, model_name: str = "XGBoost", input_size: int = 31, output_size: int = 51,
                 n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6,
                 min_child_weight: int = 1, subsample: float = 1.0, colsample_bytree: float = 1.0,
                 gamma: float = 0, reg_alpha: float = 0, reg_lambda: float = 1.0, random_state: int = None):
        """
        Инициализация конфигурации XGBoost.
        """
        super().__init__(model_name, input_size, output_size)
        self.n_estimators = n_estimators          # Количество деревьев
        self.learning_rate = learning_rate        # Скорость обучения
        self.max_depth = max_depth                # Максимальная глубина
        self.min_child_weight = min_child_weight  # Минимальный вес в листе
        self.subsample = subsample                # Доля образцов
        self.colsample_bytree = colsample_bytree  # Доля признаков
        self.gamma = gamma                        # Минимальное уменьшение потерь
        self.reg_alpha = reg_alpha                # L1 регуляризация
        self.reg_lambda = reg_lambda              # L2 регуляризация
        self.random_state = random_state          # Seed

    def build_model(self, learning_rate=None):
        """Построить модель с возможностью переопределения learning_rate."""
        lr = learning_rate if learning_rate is not None else self.learning_rate
        objective = 'multi:softprob' if self.output_size > 2 else 'binary:logistic'
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=lr,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective=objective,
            eval_metric='mlogloss' if self.output_size > 2 else 'logloss'
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # Используем n_estimators и learning_rate из TrainingConfig, если переданы
        if training_config:
            if hasattr(training_config, 'epochs'):
                original_n_estimators = self.n_estimators
                self.n_estimators = training_config.epochs
            if hasattr(training_config, 'learning_rate'):
                original_lr = self.learning_rate
                self.learning_rate = training_config.learning_rate
        
        model = self.build_model()
        
        if training_config:
            if hasattr(training_config, 'epochs'):
                self.n_estimators = original_n_estimators # Восстанавливаем оригинальное значение
            if hasattr(training_config, 'learning_rate'):
                self.learning_rate = original_lr # Восстанавливаем оригинальное значение
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if self.output_size == 2 else model.predict_proba(X_test)
        return LogisticRegressionConfig._compute_metrics(self, y_test, y_pred, y_prob)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state
        })
        return data

class CatBoostConfig(ModelConfig):
    """
    Конфигурация для CatBoost (Categorical Boosting).

    CatBoost - градиентный бустинг от Yandex, оптимизированный для работы с категориальными
    признаками. Автоматически обрабатывает категории и имеет встроенные механизмы
    предотвращения переобучения.
    
    Преимущества:
    - Отличная работа с категориальными признаками
    - Меньше переобучения (встроенные техники)
    - Хорошая точность из коробки
    - Меньше требуется настройки гиперпараметров
    
    Недостатки:
    - Медленнее XGBoost на некоторых задачах
    - Больше использование памяти
    """
    def __init__(self, model_name: str = "CatBoost", input_size: int = 31, output_size: int = 51,
                 iterations: int = 100, learning_rate: float = 0.1, depth: int = 6,
                 l2_leaf_reg: float = 3.0, border_count: int = 254, loss_function: str = 'MultiClass',
                 random_state: int = None):
        """
        Инициализация конфигурации CatBoost.
        
        Параметры:
        ----------
        iterations : int, default=100 - количество итераций
        learning_rate : float, default=0.1 - скорость обучения
        depth : int, default=6 - глубина деревьев
        l2_leaf_reg : float, default=3.0 - L2 регуляризация
        border_count : int, default=254 - количество границ квантования
        loss_function : str, default='MultiClass' - функция потерь
        random_state : int, optional - seed
        """
        super().__init__(model_name, input_size, output_size)
        self.iterations = iterations              # Количество итераций
        self.learning_rate = learning_rate        # Скорость обучения
        self.depth = depth                        # Глубина деревьев
        self.l2_leaf_reg = l2_leaf_reg            # L2 регуляризация
        self.border_count = border_count          # Количество границ
        self.loss_function = loss_function        # Функция потерь
        self.random_state = random_state          # Seed

    def build_model(self):
        loss_func = 'Logloss' if self.output_size == 2 else 'MultiClass'
        return cb.CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            loss_function=loss_func,
            random_state=self.random_state,
            verbose=0
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # Используем iterations и learning_rate из TrainingConfig, если переданы
        if training_config:
            if hasattr(training_config, 'epochs'):
                original_iterations = self.iterations
                self.iterations = training_config.epochs
            if hasattr(training_config, 'learning_rate'):
                original_lr = self.learning_rate
                self.learning_rate = training_config.learning_rate
        
        model = self.build_model()
        
        if training_config:
            if hasattr(training_config, 'epochs'):
                self.iterations = original_iterations # Восстанавливаем оригинальное значение
            if hasattr(training_config, 'learning_rate'):
                self.learning_rate = original_lr # Восстанавливаем оригинальное значение
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if self.output_size == 2 else model.predict_proba(X_test)
        return LogisticRegressionConfig._compute_metrics(self, y_test, y_pred, y_prob)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "loss_function": self.loss_function,
            "random_state": self.random_state
        })
        return data

class LSTMConfig(ModelConfig):
    """
    Конфигурация для LSTM (Long Short-Term Memory) нейронной сети.

    LSTM - тип рекуррентной нейронной сети (RNN), способной запоминать долгосрочные
    зависимости в последовательностях данных. Хорошо подходит для временных рядов
    и последовательностей, хотя в данном случае используется для классификации
    по признакам (данные преобразуются в последовательность длиной 1).
    
    Преимущества:
    - Способность запоминать долгосрочные зависимости
    - Хорошо работает с последовательностями
    - Может быть двунаправленным (bidirectional)
    
    Недостатки:
    - Медленное обучение
    - Требует много данных
    - Сложная настройка гиперпараметров
    - Требует GPU для больших моделей
    """
    def __init__(self, model_name: str = "LSTM", input_size: int = 31, output_size: int = 51,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0, 
                 bidirectional: bool = False):
        """
        Инициализация конфигурации LSTM.
        """
        super().__init__(model_name, input_size, output_size)
        self.hidden_size = hidden_size          # Размер скрытого состояния
        self.num_layers = num_layers            # Количество слоев
        self.dropout = dropout                  # Вероятность dropout
        self.bidirectional = bidirectional      # Двунаправленность

    def build_model(self):
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
                self.dropout_layer = nn.Dropout(dropout)
                self.fc = nn.Linear(lstm_output_size, output_size)

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                # Берем последний слой и последний временной шаг
                if self.lstm.bidirectional:
                    # Для bidirectional объединяем forward и backward скрытые состояния
                    h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    h_n = h_n[-1]
                h_n = self.dropout_layer(h_n)
                return self.fc(h_n)

        return LSTMModel(self.input_size, self.hidden_size, self.num_layers, 
                        self.output_size, self.dropout, self.bidirectional)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, training_config=None):
        # Используем параметры из TrainingConfig
        epochs = training_config.epochs if training_config else 10
        batch_size = training_config.batch_size if training_config else 32
        learning_rate = training_config.learning_rate if training_config else 0.001
        
        # Определяем optimizer из TrainingConfig
        optimizer_name = training_config.optimizer.lower() if training_config else 'adam'
        
        # Предполагаем временные ряды: приводим к форме (samples, timesteps=1, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = self.build_model()
        criterion = nn.CrossEntropyLoss()
        
        # Выбираем optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            logger.warning(f"Неизвестный optimizer '{optimizer_name}', используется Adam")

        # Обучение
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Оценка
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            all_outputs = []
            for inputs, _ in test_loader:
                outputs = model(inputs)
                all_outputs.append(outputs)
            outputs = torch.cat(all_outputs, dim=0)
            y_pred = torch.argmax(outputs, dim=1).numpy()
            y_prob = torch.softmax(outputs, dim=1).numpy()

        return LogisticRegressionConfig._compute_metrics(
            self, y_test, y_pred, 
            y_prob if self.output_size > 2 else y_prob[:, 1]
        )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional
        })
        return data

class TrainingConfig:
    """
    Класс конфигурации процесса обучения моделей.

    Данный класс инкапсулирует параметры, связанные с процессом обучения моделей.
    Используется в классе Experiment для передачи параметров обучения в модели.

    Для нейронных сетей (LSTM): все параметры используются напрямую
    Для классических ML моделей: epochs может использоваться как max_iter/n_estimators
    learning_rate используется для градиентных методов (GradientBoosting, XGBoost, CatBoost)
    """
    def __init__(self, epochs: int = 10, batch_size: int = 32, optimizer: str = "adam", learning_rate: float = 0.001):
        """
        Инициализация конфигурации обучения.
        """
        self.epochs = epochs              # Количество эпох
        self.batch_size = batch_size      # Размер батча
        self.optimizer = optimizer        # Оптимизатор
        self.learning_rate = learning_rate  # Скорость обучения

    def to_dict(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def __str__(self) -> str:
        return f"TrainingConfig(epochs={self.epochs}, batch_size={self.batch_size})"

class Experiment:
    """
    Класс для управления и выполнения экспериментов машинного обучения.

    Данный класс реализует паттерн композиции, объединяя три основных компонента:
    1. ModelConfig - конфигурация модели
    2. TrainingConfig - конфигурация обучения
    3. DatasetHandler - обработчик данных
    
    Класс обеспечивает:
    - Запуск экспериментов (реальных или симулированных)
    - Сохранение и загрузку конфигураций
    - Сравнение результатов нескольких экспериментов
    """
    def __init__(self, exp_id: str, model_config: ModelConfig, training_config: TrainingConfig, dataset_handler: DatasetHandler, simulate: bool = False):
        """
        Инициализация эксперимента
        """
        self.exp_id = exp_id                    # Идентификатор эксперимента
        self.model_config = model_config        # Конфигурация модели
        self.training_config = training_config  # Конфигурация обучения
        self.dataset_handler = dataset_handler  # Обработчик данных
        self.simulate = simulate                # Флаг симуляции
        self.results = {}                       # Результаты эксперимента (заполняются в run())

    def run(self):
        """
        Запустить эксперимент: обучить модель и получить метрики качества.

        Метод выполняет основной цикл эксперимента:
        1. Разделение данных на обучающую и тестовую выборки
        2. Обучение модели (реальное или симулированное)
        3. Вычисление метрик качества
        4. Сохранение результатов в self.results

        После выполнения метода self.results содержит словарь с метриками:
        - accuracy, precision, recall, f1_score, roc_auc, eer
        """
        # Логирование начала эксперимента
        logger.info(f"Запуск эксперимента {self.exp_id}")
        
        # Разделение данных на обучающую и тестовую выборки
        # По умолчанию: 80% обучающая, 20% тестовая
        X_train, X_test, y_train, y_test = self.dataset_handler.split_data()
        
        if self.simulate:
            """
            Режим симуляции: генерация случайных результатов.
            """
            # Базовое значение accuracy зависит от типа модели
            # Нейронные сети обычно показывают лучшие результаты
            base_acc = 0.9 if "NN" in self.model_config.model_name else 0.8
            
            # Генерация случайных метрик в реалистичных диапазонах
            self.results = {
                "accuracy": round(random.uniform(base_acc, 0.99), 4),      # Точность: 0.8-0.99
                "precision": round(random.uniform(0.7, 0.99), 4),          # Прецизионность: 0.7-0.99
                "recall": round(random.uniform(0.7, 0.99), 4),              # Полнота: 0.7-0.99
                "f1_score": round(random.uniform(0.7, 0.99), 4),           # F1-мера: 0.7-0.99
                "roc_auc": round(random.uniform(0.8, 0.99), 4),            # ROC-AUC: 0.8-0.99
                "eer": round(random.uniform(0.01, 0.1), 4)                  # EER: 0.01-0.1 (1%-10%)
            }
        else:
            """
            Реальный режим: обучение модели и вычисление метрик.
            
            Процесс:
            1. model_config.train_and_evaluate() создает модель
            2. Обучает её на X_train, y_train
            3. Оценивает на X_test, y_test
            4. Возвращает словарь с метриками
            """
            # Вызов метода обучения и оценки модели
            # training_config передается для настройки параметров обучения
            self.results = self.model_config.train_and_evaluate(
                X_train, y_train, X_test, y_test, 
                training_config=self.training_config
            )
        
        # Логирование результатов эксперимента
        logger.info(f"Результаты: {self.results}")

    def save_config(self, path: str):
        """
        Сохранить конфигурацию эксперимента в JSON файл.

        Метод сериализует всю конфигурацию эксперимента в JSON файл.
        Это позволяет воссоздать эксперимент позже или поделиться конфигурацией.
        """
        # Формирование словаря с данными эксперимента
        data = {
            "exp_id": self.exp_id,                                    # Идентификатор
            "model_config": self.model_config.to_dict(),              # Конфигурация модели
            "training_config": self.training_config.to_dict(),        # Конфигурация обучения
            "dataset_type": self.dataset_handler.dataset_type,       # Тип датасета
            "results": self.results                                    # Результаты
        }
        
        # Сохранение в JSON файл
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)  # indent=4 для читаемости

    @classmethod
    def load_config(cls, path: str, data_path: str, **dataset_kwargs):
        """
        Загрузить эксперимент из JSON файла.
 
        Классовый метод для восстановления эксперимента из сохраненной конфигурации.
        Полезно для воспроизведения экспериментов или анализа сохраненных результатов.
        """
        # Загрузка данных из JSON файла
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Восстановление конфигураций из словарей
        model_config = ModelConfig.from_dict(data["model_config"])
        training_config = TrainingConfig.from_dict(data["training_config"])
        
        # Создание обработчика данных с указанными параметрами
        dataset_handler = DatasetHandler(
            data["dataset_type"], 
            data_path, 
            **dataset_kwargs
        )
        
        # Создание объекта эксперимента
        exp = cls(
            data["exp_id"], 
            model_config, 
            training_config, 
            dataset_handler
        )
        
        # Восстановление результатов (если они были сохранены)
        exp.results = data.get("results", {})
        
        return exp

    @staticmethod
    def compare_experiments(experiments: list["Experiment"], output_path: str = None):
        """
        Сравнить результаты нескольких экспериментов.
        Создает таблицу сравнения и выводит её в лог и/или сохраняет в CSV.
        """
        # Создание DataFrame из результатов экспериментов
        # Каждый эксперимент добавляется как строка с его метриками
        df = pd.DataFrame([exp.results for exp in experiments])
        
        # Добавление колонки с названиями моделей для удобства сравнения
        df['model'] = [exp.model_config.model_name for exp in experiments]
        
        # Вывод таблицы в лог
        logger.info("\n" + df.to_string())
        
        # Сохранение в CSV файл, если указан путь
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Таблица сравнения сохранена в {output_path}")
        
        return df

    def __str__(self) -> str:
        """
        Возвращает строковое представление эксперимента.
        """
        return f"Experiment({self.exp_id}, {self.model_config})"