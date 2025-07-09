# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf
from config import Config

class DemoDataGenerator:
    """Генератор демонстрационных данных для тестирования приложения"""
    
    @staticmethod
    def create_demo_diarization_results() -> pd.DataFrame:
        """Создание демонстрационных результатов диаризации"""
        
        # Симулируем диалог между тремя спикерами
        demo_data = [
            {'Speaker': 'SPEAKER_00', 'Start (sec)': 0.0, 'End (sec)': 3.5, 'Duration (sec)': 3.5},
            {'Speaker': 'SPEAKER_01', 'Start (sec)': 3.8, 'End (sec)': 8.2, 'Duration (sec)': 4.4},
            {'Speaker': 'SPEAKER_00', 'Start (sec)': 8.5, 'End (sec)': 12.1, 'Duration (sec)': 3.6},
            {'Speaker': 'SPEAKER_02', 'Start (sec)': 12.4, 'End (sec)': 16.8, 'Duration (sec)': 4.4},
            {'Speaker': 'SPEAKER_01', 'Start (sec)': 17.1, 'End (sec)': 20.5, 'Duration (sec)': 3.4},
            {'Speaker': 'SPEAKER_00', 'Start (sec)': 20.8, 'End (sec)': 25.2, 'Duration (sec)': 4.4},
            {'Speaker': 'SPEAKER_02', 'Start (sec)': 25.5, 'End (sec)': 28.9, 'Duration (sec)': 3.4},
            {'Speaker': 'SPEAKER_01', 'Start (sec)': 29.2, 'End (sec)': 33.6, 'Duration (sec)': 4.4},
            {'Speaker': 'SPEAKER_00', 'Start (sec)': 33.9, 'End (sec)': 37.3, 'Duration (sec)': 3.4},
            {'Speaker': 'SPEAKER_02', 'Start (sec)': 37.6, 'End (sec)': 42.0, 'Duration (sec)': 4.4},
        ]
        
        df = pd.DataFrame(demo_data)
        return df
    
    @staticmethod
    def create_demo_audio_file(output_path: str, duration: float = 45.0, sample_rate: int = 16000):
        """Создание демонстрационного аудиофайла"""
        
        # Генерируем простой синусоидальный сигнал
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Создаем сигнал с разными частотами для разных "спикеров"
        frequency_base = 440  # Базовая частота (A4)
        signal = np.zeros_like(t)
        
        # Имитируем речь разных спикеров с разными частотами
        segments = [
            (0.0, 3.5, 440),    # SPEAKER_00 - A4
            (3.8, 8.2, 523),    # SPEAKER_01 - C5
            (8.5, 12.1, 440),   # SPEAKER_00 - A4
            (12.4, 16.8, 659),  # SPEAKER_02 - E5
            (17.1, 20.5, 523),  # SPEAKER_01 - C5
            (20.8, 25.2, 440),  # SPEAKER_00 - A4
            (25.5, 28.9, 659),  # SPEAKER_02 - E5
            (29.2, 33.6, 523),  # SPEAKER_01 - C5
            (33.9, 37.3, 440),  # SPEAKER_00 - A4
            (37.6, 42.0, 659),  # SPEAKER_02 - E5
        ]
        
        for start, end, freq in segments:
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            
            # Создаем модулированный сигнал для имитации речи
            segment_t = t[start_idx:end_idx]
            amplitude = 0.3 * np.sin(2 * np.pi * freq * segment_t)
            
            # Добавляем модуляцию для имитации речи
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * segment_t)
            signal[start_idx:end_idx] = amplitude * modulation
        
        # Добавляем небольшой шум
        noise = np.random.normal(0, 0.01, len(signal))
        signal = signal + noise
        
        # Нормализуем сигнал
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        # Сохраняем как WAV файл
        sf.write(output_path, signal, sample_rate)
        
        print(f"Демонстрационный аудиофайл создан: {output_path}")
        print(f"Длительность: {duration} секунд")
        print(f"Частота дискретизации: {sample_rate} Гц")
    
    @staticmethod
    def export_demo_results(df: pd.DataFrame, output_dir: str):
        """Экспорт демонстрационных результатов в файлы"""
        
        # CSV файл
        csv_path = os.path.join(output_dir, "demo_diarization_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Excel файл
        excel_path = os.path.join(output_dir, "demo_diarization_results.xlsx")
        df.to_excel(excel_path, index=False)
        
        print(f"Демонстрационные результаты экспортированы:")
        print(f"- CSV: {csv_path}")
        print(f"- Excel: {excel_path}")
    
    @staticmethod
    def create_demo_visualization(df: pd.DataFrame, output_dir: str):
        """Создание демонстрационной визуализации"""
        
        plt.figure(figsize=(12, 6))
        
        # Цвета для спикеров
        colors = {'SPEAKER_00': '#FF6B6B', 'SPEAKER_01': '#4ECDC4', 'SPEAKER_02': '#45B7D1'}
        
        for _, row in df.iterrows():
            plt.barh(
                row['Speaker'], 
                row['Duration (sec)'], 
                left=row['Start (sec)'], 
                color=colors.get(row['Speaker'], '#95A5A6'),
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
        
        plt.xlabel('Время (секунды)')
        plt.ylabel('Спикер')
        plt.title('Демонстрационная диаграмма диаризации')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем график
        plot_path = os.path.join(output_dir, "demo_diarization_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Демонстрационная визуализация сохранена: {plot_path}")
    
    @staticmethod
    def create_large_demo_file(output_path: str, size_mb: int = 150):
        """Создание демонстрационного файла заданного размера"""
        # Рассчитываем длительность для достижения нужного размера
        # WAV 16kHz моно ~ 32 kB/сек
        duration = size_mb * 1024 * 1024 / (16000 * 2)  # 2 байта на сэмпл
        
        print(f"Создание файла размером {size_mb} МБ (длительность: {duration/60:.1f} минут)...")
        
        # Создаем более длинный файл
        DemoDataGenerator.create_demo_audio_file(output_path, duration, 16000)

def create_demo_environment():
    """Создание полного демонстрационного окружения"""
    
    print("🎵 Создание демонстрационного окружения AudioQA")
    print("=" * 50)
    
    # Создаем папку для демо данных
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Создаем генератор
    generator = DemoDataGenerator()
    
    # Создаем демонстрационные результаты диаризации
    print("\n1. Создание демонстрационных результатов...")
    demo_df = generator.create_demo_diarization_results()
    generator.export_demo_results(demo_df, demo_dir)
    
    # Создаем демонстрационный аудиофайл
    print("\n2. Создание демонстрационного аудиофайла...")
    audio_path = demo_dir / "demo_audio.wav"
    generator.create_demo_audio_file(str(audio_path))
    
    # Создаем визуализацию
    print("\n3. Создание демонстрационной визуализации...")
    generator.create_demo_visualization(demo_df, demo_dir)
    
    print("\n✅ Демонстрационное окружение создано!")
    print(f"📁 Все файлы сохранены в папке: {demo_dir}")
    print("\nДля тестирования приложения:")
    print("1. Запустите приложение: streamlit run audio_diarization_app.py")
    print("2. Выберите файл demo_audio.wav через кнопку Browse или укажите путь:")
    print(f"   {demo_dir / 'demo_audio.wav'}")
    print("3. Сравните результаты с файлом demo_diarization_results.csv")
    
    # Опция создания большого файла
    print("\n" + "="*50)
    create_large = input("Создать большой файл для тестирования? (y/N): ").lower().strip()
    if create_large in ['y', 'yes', 'да']:
        try:
            size_mb = int(input("Размер файла в МБ (по умолчанию 150): ") or "150")
            large_audio_path = demo_dir / f"demo_large_{size_mb}mb.wav"
            generator.create_large_demo_file(str(large_audio_path), size_mb)
            print(f"✅ Большой файл создан: {large_audio_path}")
        except ValueError:
            print("❌ Неверный размер файла")
        except Exception as e:
            print(f"❌ Ошибка при создании файла: {e}")

if __name__ == "__main__":
    create_demo_environment() 