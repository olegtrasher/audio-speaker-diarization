# -*- coding: utf-8 -*-
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Явно указываем путь к .env (оставляем для других переменных, если потребуется)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# Настройки приложения
class Config:
    """Конфигурация приложения для диаризации аудио"""
    
    # Основные настройки
    APP_NAME = "AudioQA - Диаризация аудио"
    VERSION = "1.0.0"
    
    # Настройки Hugging Face
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Модель для диаризации
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    
    # Поддерживаемые форматы аудио
    SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    # Настройки экспорта
    EXPORT_FORMATS = ['csv', 'excel']
    
    # Настройки интерфейса
    # DEFAULT_WORK_FOLDER больше не используется - выбираем файлы напрямую
    
    # Настройки обработки
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))  # Максимальный размер файла в MB
    FORCE_GPU_USAGE = os.getenv("FORCE_GPU_USAGE", "true").lower() == "true"  # Принудительное использование GPU
    PREFERRED_SAMPLE_RATE = 16000  # Оптимальная частота дискретизации для модели
    
    # Настройки файнтюнинга диаризации
    class DiarizationTuning:
        """Настройки для файнтюнинга качества диаризации"""
        
        # Voice Activity Detection (VAD) - определение активности голоса
        VAD_ONSET = 0.5           # Порог для начала речи (0.1-0.9, меньше = более чувствительно)
        VAD_OFFSET = 0.35         # Порог для окончания речи (0.1-0.9, меньше = более чувствительно)
        VAD_MIN_DURATION_ON = 0.0  # Минимальная длительность речи в секундах
        VAD_MIN_DURATION_OFF = 0.0 # Минимальная длительность тишины в секундах
        
        # Сегментация - разбивка на фрагменты
        SEGMENTATION_ONSET = 0.5   # Порог для начала нового сегмента (0.1-0.9)
        SEGMENTATION_OFFSET = 0.5  # Порог для окончания сегмента (0.1-0.9)
        SEGMENTATION_MIN_DURATION_ON = 0.0   # Минимальная длительность сегмента
        SEGMENTATION_MIN_DURATION_OFF = 0.0  # Минимальная пауза между сегментами
        
        # Кластеризация спикеров - группировка по голосам
        CLUSTERING_METHOD = "centroid"  # Метод кластеризации: "centroid" или "closest"
        CLUSTERING_THRESHOLD = 0.7154   # Порог схожести спикеров (0.0-1.0, меньше = больше спикеров)
        
        # Дополнительные настройки
        MIN_SPEAKERS = None        # Минимальное количество спикеров (None = авто)
        MAX_SPEAKERS = None        # Максимальное количество спикеров (None = авто)
        
        @classmethod
        def get_hyperparameters(cls):
            """Возвращает гиперпараметры для модели"""
            return {
                # Voice Activity Detection
                "vad": {
                    "onset": cls.VAD_ONSET,
                    "offset": cls.VAD_OFFSET,
                    "min_duration_on": cls.VAD_MIN_DURATION_ON,
                    "min_duration_off": cls.VAD_MIN_DURATION_OFF,
                },
                # Segmentation
                "segmentation": {
                    "onset": cls.SEGMENTATION_ONSET,
                    "offset": cls.SEGMENTATION_OFFSET,
                    "min_duration_on": cls.SEGMENTATION_MIN_DURATION_ON,
                    "min_duration_off": cls.SEGMENTATION_MIN_DURATION_OFF,
                },
                # Clustering
                "clustering": {
                    "method": cls.CLUSTERING_METHOD,
                    "threshold": cls.CLUSTERING_THRESHOLD,
                    "min_cluster_size": 1,
                    "top_k": 10,
                },
            }
        
        @classmethod
        def get_presets(cls):
            """Возвращает готовые пресеты настроек"""
            return {
                "Стандартные": {
                    "vad_onset": 0.5, "vad_offset": 0.35,
                    "seg_onset": 0.5, "seg_offset": 0.5,
                    "clustering": 0.7154,
                    "description": "Базовые настройки модели"
                },
                "Высокая чувствительность": {
                    "vad_onset": 0.2, "vad_offset": 0.1,
                    "seg_onset": 0.3, "seg_offset": 0.3,
                    "clustering": 0.6,
                    "description": "Улавливает тихие голоса, больше сегментов"
                },
                "Низкая чувствительность": {
                    "vad_onset": 0.7, "vad_offset": 0.6,
                    "seg_onset": 0.7, "seg_offset": 0.7,
                    "clustering": 0.8,
                    "description": "Только громкие голоса, меньше ложных срабатываний"
                },
                "Много спикеров": {
                    "vad_onset": 0.3, "vad_offset": 0.2,
                    "seg_onset": 0.4, "seg_offset": 0.4,
                    "clustering": 0.5,
                    "description": "Для записей с 5+ спикерами"
                },
                "Мало спикеров": {
                    "vad_onset": 0.6, "vad_offset": 0.4,
                    "seg_onset": 0.6, "seg_offset": 0.6,
                    "clustering": 0.9,
                    "description": "Для записей с 2-3 спикерами"
                },
                "Шумная запись": {
                    "vad_onset": 0.8, "vad_offset": 0.7,
                    "seg_onset": 0.7, "seg_offset": 0.7,
                    "clustering": 0.6,
                    "description": "Для записей с фоновым шумом"
                }
            }
    
    # Настройки логирования
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Пути
    TEMP_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "temp"
    
    @staticmethod
    def create_temp_dir():
        """Создание временной папки"""
        Config.TEMP_DIR.mkdir(exist_ok=True)
        return Config.TEMP_DIR
    
    @staticmethod
    def get_output_filename(input_file: str, format: str = "csv") -> str:
        """Генерация имени выходного файла"""
        stem = Path(input_file).stem
        suffix = "csv" if format.lower() == "csv" else "xlsx"
        return f"{stem}_diarization.{suffix}"

# Проверка наличия токена Hugging Face
def check_huggingface_token():
    """Проверка наличия токена Hugging Face"""
    if not Config.HUGGINGFACE_TOKEN:
        return False, "Не найден токен Hugging Face."
    return True, "Токен Hugging Face найден."

# Проверка системных требований
def check_system_requirements():
    """Проверка системных требований"""
    try:
        import torch
        pytorch_available = True
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "Не доступно"
            
    except ImportError as e:
        pytorch_available = False
        gpu_available = False
        gpu_name = "PyTorch не установлен"
    
    requirements = {
        "Python": True,
        "PyTorch": pytorch_available,
        "GPU": gpu_available,
        "Hugging Face Token": bool(Config.HUGGINGFACE_TOKEN)
    }
    
    return requirements, gpu_name 