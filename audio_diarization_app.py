# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import tempfile
import shutil
import time
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import plotly.express as px
import plotly.graph_objects as go
from config import Config, check_huggingface_token, check_system_requirements

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация приложения
st.set_page_config(
    page_title="AudioQA - Диаризация аудио",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SMPTETimeCode:
    """Класс для работы с SMPTE 59.94 Drop Frame таймкодами"""
    
    @staticmethod
    def seconds_to_smpte(seconds: float, start_timecode: str = "00;00;00;00") -> str:
        """Конвертация секунд в SMPTE 59.94 DROP таймкод"""
        # Парсим начальный таймкод
        start_frames = SMPTETimeCode.smpte_to_frames(start_timecode)
        
        # Конвертируем секунды в кадры (59.94 fps)
        total_frames = int(seconds * 59.94) + start_frames
        
        return SMPTETimeCode.frames_to_smpte(total_frames)
    
    @staticmethod
    def smpte_to_seconds(timecode: str, start_timecode: str = "00;00;00;00") -> float:
        """Конвертация SMPTE таймкода в секунды"""
        frames = SMPTETimeCode.smpte_to_frames(timecode)
        start_frames = SMPTETimeCode.smpte_to_frames(start_timecode)
        
        # Вычитаем начальное смещение
        net_frames = frames - start_frames
        
        # Конвертируем в секунды (59.94 fps)
        return net_frames / 59.94
    
    @staticmethod
    def smpte_to_frames(timecode: str) -> int:
        """Конвертация SMPTE таймкода в количество кадров"""
        try:
            # Парсим таймкод HH;MM;SS;FF
            parts = timecode.replace(':', ';').split(';')
            if len(parts) != 4:
                return 0
                
            hours, minutes, seconds, frames = map(int, parts)
            
            # SMPTE 59.94 Drop Frame расчет
            # Drop 2 кадра каждую минуту, кроме минут кратных 10
            total_minutes = hours * 60 + minutes
            
            # Базовые кадры без учета drop
            total_frames = (hours * 3600 + minutes * 60 + seconds) * 60 + frames
            
            # Учитываем drop frame: -2 кадра каждую минуту, +2 кадра каждые 10 минут
            if total_minutes > 0:
                drop_frames = total_minutes * 2 - (total_minutes // 10) * 2
                total_frames -= drop_frames
            
            return max(0, total_frames)
            
        except (ValueError, IndexError):
            return 0
    
    @staticmethod
    def frames_to_smpte(total_frames: int) -> str:
        """Конвертация кадров в SMPTE 59.94 DROP таймкод"""
        if total_frames < 0:
            return "00;00;00;00"
        
        # Компенсируем drop frame
        # Приблизительный расчет с учетом drop frame
        fps = 60  # Номинальный FPS для расчетов
        
        # Грубый расчет минут для определения количества drop кадров
        approx_total_seconds = total_frames / 59.94
        approx_total_minutes = int(approx_total_seconds / 60)
        
        # Добавляем обратно drop кадры для получения номинального времени
        if approx_total_minutes > 0:
            drop_compensation = approx_total_minutes * 2 - (approx_total_minutes // 10) * 2
            nominal_frames = total_frames + drop_compensation
        else:
            nominal_frames = total_frames
        
        # Расчет времени из номинальных кадров
        frames = nominal_frames % fps
        total_seconds = nominal_frames // fps
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60
        
        return f"{hours:02d};{minutes:02d};{seconds:02d};{frames:02d}"
    
    @staticmethod
    def validate_timecode(timecode: str) -> bool:
        """Валидация формата таймкода"""
        try:
            parts = timecode.replace(':', ';').split(';')
            if len(parts) != 4:
                return False
            
            hours, minutes, seconds, frames = map(int, parts)
            
            return (0 <= hours <= 23 and 
                   0 <= minutes <= 59 and 
                   0 <= seconds <= 59 and 
                   0 <= frames <= 59)
        except (ValueError, IndexError):
            return False

class AudioDiarizationApp:
    def __init__(self):
        self.pipeline = None
        # Инициализируем session_state если данных нет
        if 'df_results_original' not in st.session_state:
            st.session_state.df_results_original = None
        if 'df_results' not in st.session_state:
            st.session_state.df_results = None
        if 'start_timecode' not in st.session_state:
            st.session_state.start_timecode = "00;00;00;00"
        
        # Инициализируем настройки файнтюнинга
        if 'tuning_settings' not in st.session_state:
            st.session_state.tuning_settings = {
                "vad_onset": Config.DiarizationTuning.VAD_ONSET,
                "vad_offset": Config.DiarizationTuning.VAD_OFFSET,
                "seg_onset": Config.DiarizationTuning.SEGMENTATION_ONSET,
                "seg_offset": Config.DiarizationTuning.SEGMENTATION_OFFSET,
                "clustering": Config.DiarizationTuning.CLUSTERING_THRESHOLD,
                "min_speakers": Config.DiarizationTuning.MIN_SPEAKERS,
                "max_speakers": Config.DiarizationTuning.MAX_SPEAKERS,
            }
        
        self.selected_speakers = []
        
    @property
    def df_results_original(self):
        return st.session_state.df_results_original
    
    @df_results_original.setter 
    def df_results_original(self, value):
        st.session_state.df_results_original = value
        
    @property
    def df_results(self):
        return st.session_state.df_results
    
    @df_results.setter
    def df_results(self, value):
        st.session_state.df_results = value
        
    @property
    def start_timecode(self):
        return st.session_state.start_timecode
    
    @start_timecode.setter
    def start_timecode(self, value):
        st.session_state.start_timecode = value
        
    def initialize_pipeline(self):
        """Инициализация пайплайна диаризации"""
        if self.pipeline is None:
            from config import Config
            
            with st.spinner("Загрузка модели диаризации..."):
                try:
                    # Загружаем основную модель с токеном
                    self.pipeline = Pipeline.from_pretrained(
                        Config.DIARIZATION_MODEL,
                        use_auth_token=Config.HUGGINGFACE_TOKEN
                    )
                    st.info("🔐 Модель загружена с токеном Hugging Face")
                    
                    # Принудительное использование GPU если доступно
                    if Config.FORCE_GPU_USAGE:
                        if torch.cuda.is_available():
                            device = torch.device("cuda")
                            self.pipeline.to(device)
                            gpu_name = torch.cuda.get_device_name(0)
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            logger.info(f"🚀 Используется GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                            st.success(f"🚀 GPU активирован: {gpu_name}")
                        else:
                            logger.error("❌ CUDA не найден! Проверьте драйвер NVIDIA, CUDA Toolkit и версию PyTorch.")
                            st.error("❌ CUDA не найден! Проверьте драйвер NVIDIA, CUDA Toolkit и версию PyTorch.")
                            st.info("Выполните в Python: import torch; print(torch.version.cuda); print(torch.cuda.is_available())")
                            return False
                    else:
                        st.info("⚠️ Используется CPU (GPU отключен в настройках)")
                        logger.info("CPU режим активирован")
                    
                    logger.info("Пайплайн диаризации успешно инициализирован")
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при инициализации модели: {str(e)}")
                    logger.error(f"Ошибка инициализации: {str(e)}")
                    return False
        else:
            # Модель уже загружена
            st.info("✅ Модель уже загружена")
            return True
    
    def process_audio_file(self, audio_file_path: str) -> Optional[pd.DataFrame]:
        """Обработка аудиофайла для диаризации"""
        try:
            # Получаем размер файла для оценки времени обработки
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            estimated_time = max(1, int(file_size_mb * 0.5))  # Примерно 0.5 минуты на МБ
            
            progress_text = f"Обработка файла ({file_size_mb:.1f} МБ)..."
            if file_size_mb > 50:
                progress_text += f" Примерное время: {estimated_time} минут"
            
            with st.spinner(progress_text):
                # Получаем настройки файнтюнинга
                tuning = st.session_state.tuning_settings
                
                # Показываем какие настройки применяются
                st.info(f"🎛️ Применяются настройки: VAD {tuning['vad_onset']:.2f}/{tuning['vad_offset']:.2f}, Clustering {tuning['clustering']:.3f}")
                
                # Примечание: настройки файнтюнинга для pyannote-audio 3.1+ 
                # могут не поддерживаться через API изменения параметров
                # Эта функциональность будет улучшена в будущих версиях
                
                # Применяем диаризацию
                diarization = self.pipeline(audio_file_path)
                
                # Преобразуем результат в DataFrame (только секунды, без таймкодов)
                results_data = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    results_data.append({
                        'Speaker': speaker,
                        'Start (sec)': round(turn.start, 2),
                        'End (sec)': round(turn.end, 2),
                        'Duration (sec)': round(turn.duration, 2)
                    })
                
                df = pd.DataFrame(results_data)
                
                # Сортируем по времени начала
                df = df.sort_values('Start (sec)').reset_index(drop=True)
                
                # Анализируем результаты и даем рекомендации
                total_segments = len(df)
                total_speakers = df['Speaker'].nunique()
                
                # Рекомендации по улучшению результатов
                recommendations = []
                if total_segments < 5:
                    recommendations.append("📊 Мало сегментов - попробуйте снизить VAD Onset до 0.2-0.3")
                if total_speakers == 1:
                    recommendations.append("👥 Найден только 1 спикер - попробуйте снизить Clustering до 0.5-0.6")
                if total_speakers > 10:
                    recommendations.append("👥 Слишком много спикеров - попробуйте увеличить Clustering до 0.8-0.9")
                
                if recommendations:
                    st.warning("💡 Рекомендации по улучшению результатов:")
                    for rec in recommendations:
                        st.write(f"• {rec}")
                
                logger.info(f"Обработано {len(df)} сегментов с {df['Speaker'].nunique()} спикерами")
                return df
                
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            logger.error(f"Ошибка обработки: {str(e)}")
            return None
    
    def create_timeline_visualization(self, df: pd.DataFrame) -> go.Figure:
        """Создание временной диаграммы спикеров"""
        fig = go.Figure()
        
        # Цвета для спикеров
        colors = px.colors.qualitative.Set3
        speakers = df['Speaker'].unique()
        color_map = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Start (sec)'], row['End (sec)']],
                y=[row['Speaker'], row['Speaker']],
                mode='lines+markers',
                line=dict(color=color_map[row['Speaker']], width=8),
                showlegend=False,
                hovertemplate=f"<b>{row['Speaker']}</b><br>" +
                             f"SMPTE: {row['Start (SMPTE)']} - {row['End (SMPTE)']}<br>" +
                             f"Секунды: {row['Start (sec)']}с - {row['End (sec)']}с<br>" +
                             f"Длительность: {row['Duration (sec)']}с<extra></extra>"
            ))
        
        fig.update_layout(
            title="Временная диаграмма спикеров",
            xaxis_title="Время (секунды)",
            yaxis_title="Спикер",
            hovermode="closest",
            height=max(400, len(speakers) * 50)
        )
        
        return fig
    
    def export_results(self, df: pd.DataFrame, output_path: str, format: str = "csv"):
        """Экспорт результатов в файл для Adobe Audition"""
        try:
            if format.lower() == "csv":
                # Adobe Audition CSV формат маркеров (TAB-separated)
                audition_data = []
                for index, row in df.iterrows():
                    # Конвертируем секунды в формат MM:SS.mmm для Adobe Audition
                    start_seconds = row['Start (sec)']
                    duration_seconds = row['Duration (sec)']
                    
                    # Формат MM:SS.mmm
                    start_minutes = int(start_seconds // 60)
                    start_secs = start_seconds % 60
                    start_time = f"{start_minutes}:{start_secs:06.3f}"
                    
                    duration_minutes = int(duration_seconds // 60)
                    duration_secs = duration_seconds % 60
                    duration_time = f"{duration_minutes}:{duration_secs:06.3f}"
                    
                    audition_data.append({
                        'Name': row['Speaker'],
                        'Start': start_time,
                        'Duration': duration_time,
                        'Time Format': 'decimal',
                        'Type': 'Cue',
                        'Description': f"{row['Speaker']} segment"
                    })
                
                audition_df = pd.DataFrame(audition_data)
                # Сохраняем с табуляцией как разделитель
                audition_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
                
            elif format.lower() == "excel":
                # Для Excel оставляем обычный формат с таймкодами
                export_df = df[['Speaker', 'Start (SMPTE)', 'End (SMPTE)', 'Duration (sec)']].copy()
                export_df = export_df.rename(columns={
                    'Start (SMPTE)': 'Start Timecode',
                    'End (SMPTE)': 'End Timecode',
                    'Duration (sec)': 'Duration (sec)'
                })
                export_df.to_excel(output_path, index=False)
            
            return True
        except Exception as e:
            st.error(f"Ошибка при экспорте: {str(e)}")
            return False
    
    def recalculate_timecodes(self):
        """Пересчет таймкодов без повторного анализа"""
        if self.df_results_original is not None:
            # Создаем новый DataFrame с пересчитанными таймкодами
            updated_data = []
            for _, row in self.df_results_original.iterrows():
                start_tc = SMPTETimeCode.seconds_to_smpte(row['Start (sec)'], self.start_timecode)
                end_tc = SMPTETimeCode.seconds_to_smpte(row['End (sec)'], self.start_timecode)
                
                # Переименовываем спикеров в Voice_XX формат
                speaker_num = row['Speaker'].replace('SPEAKER_', '').replace('Speaker_', '').replace('speaker_', '')
                voice_name = f"Voice_{speaker_num}"
                
                updated_data.append({
                    'Speaker': voice_name,
                    'Start (SMPTE)': start_tc,
                    'End (SMPTE)': end_tc,
                    'Start (sec)': row['Start (sec)'],
                    'End (sec)': row['End (sec)'],
                    'Duration (sec)': row['Duration (sec)']
                })
            
            self.df_results = pd.DataFrame(updated_data)
            return True
        return False
    
    def run(self):
        """Основная функция приложения"""
        st.title("🎵 AudioQA - Диаризация аудио")
        st.markdown("*Утилита для автоматического разделения аудиозаписей по спикерам*")
        
        # Добавляем информацию о настройках файнтюнинга
        st.info("🎛️ **Настройки файнтюнинга** доступны в боковой панели для улучшения качества распознавания голоса")
        
        # Сайдбар с настройками
        with st.sidebar:
            st.header("⚙️ Настройки")
            
            # Выбор файла напрямую
            st.subheader("📁 Выбор файла")
            
            # Поддерживаемые форматы для file_uploader
            accepted_formats = [format.upper() for format in Config.SUPPORTED_AUDIO_FORMATS]
            
            uploaded_file = st.file_uploader(
                "Выберите аудиофайл",
                type=[format[1:] for format in Config.SUPPORTED_AUDIO_FORMATS],  # убираем точку из .wav -> wav
                help=f"Поддерживаемые форматы: {', '.join(accepted_formats)}\nМаксимальный размер: {Config.MAX_FILE_SIZE_MB} МБ",
                accept_multiple_files=False
            )
            
            if uploaded_file is None:
                st.info("👆 Выберите аудиофайл для начала работы")
                
            # Альтернативный способ - ввод пути вручную
            st.subheader("🗂️ Или укажите путь")
            manual_path = st.text_input(
                "Путь к файлу",
                placeholder="C:\\path\\to\\your\\audio\\file.wav",
                help="Полный путь к аудиофайлу на вашем компьютере"
            )
            
            # Определяем, какой файл использовать
            selected_file = None
            file_path = None
            
            if uploaded_file is not None:
                # Проверяем размер загруженного файла
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > Config.MAX_FILE_SIZE_MB:
                    st.error(f"❌ Файл слишком большой: {file_size_mb:.1f} МБ (макс. {Config.MAX_FILE_SIZE_MB} МБ)")
                    return
                
                # Сохраняем загруженный файл во временную папку
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                # Очищаем старые временные файлы (старше 1 часа)
                current_time = time.time()
                for old_file in temp_dir.glob("*"):
                    if old_file.is_file() and (current_time - old_file.stat().st_mtime) > 3600:
                        try:
                            old_file.unlink()
                        except:
                            pass  # Игнорируем ошибки при удалении
                
                # Генерируем уникальное имя для избежания конфликтов
                timestamp = int(time.time())
                safe_filename = f"{timestamp}_{uploaded_file.name}"
                file_path = temp_dir / safe_filename
                
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    selected_file = uploaded_file.name
                    st.success(f"✅ Файл загружен: {selected_file}")
                except Exception as e:
                    st.error(f"❌ Ошибка при сохранении файла: {str(e)}")
                    return
                
            elif manual_path and os.path.exists(manual_path):
                # Проверяем файл по указанному пути
                if any(manual_path.lower().endswith(ext) for ext in Config.SUPPORTED_AUDIO_FORMATS):
                    file_size_mb = os.path.getsize(manual_path) / (1024 * 1024)
                    if file_size_mb <= Config.MAX_FILE_SIZE_MB:
                        file_path = manual_path
                        selected_file = os.path.basename(manual_path)
                        st.success(f"✅ Файл найден: {selected_file}")
                    else:
                        st.error(f"❌ Файл слишком большой: {file_size_mb:.1f} МБ (макс. {Config.MAX_FILE_SIZE_MB} МБ)")
                else:
                    st.error("❌ Неподдерживаемый формат файла")
            elif manual_path:
                st.error("❌ Файл не найден по указанному пути")
            
            # Системная информация
            st.subheader("💻 Системная информация")
            requirements, gpu_name = check_system_requirements()
            for req_name, req_status in requirements.items():
                if req_status:
                    st.success(f"✅ {req_name}")
                else:
                    st.error(f"❌ {req_name}")
            
            # Информация о GPU
            if requirements.get("GPU", False):
                st.info(f"🎮 GPU: {gpu_name}")
                if Config.FORCE_GPU_USAGE:
                    st.info("⚡ Принудительное использование GPU включено")
                else:
                    st.warning("⚠️ Принудительное использование GPU отключено")
            else:
                st.warning(f"🎮 GPU: {gpu_name}")
            
            # Информация о приложении
            st.subheader("ℹ️ О приложении")
            st.info(f"Версия: {Config.VERSION}")
            st.info(f"Модель: {Config.DIARIZATION_MODEL.split('/')[-1]}")
            st.info(f"Макс. размер файла: {Config.MAX_FILE_SIZE_MB} МБ")
            st.info(f"Частота: {Config.PREFERRED_SAMPLE_RATE}Hz")
            st.info(f"Таймкод: SMPTE 59.94 DROP")
            
            # Настройки файнтюнинга
            st.subheader("🎛️ Настройки файнтюнинга")
            
            # Пресеты настроек
            presets = Config.DiarizationTuning.get_presets()
            preset_names = list(presets.keys())
            
            selected_preset = st.selectbox(
                "Выберите пресет:",
                preset_names,
                index=0,
                help="Готовые настройки для разных типов записей"
            )
            
            # Показываем описание выбранного пресета
            if selected_preset:
                st.info(f"📝 {presets[selected_preset]['description']}")
            
            # Кнопка применения пресета
            if st.button("🎯 Применить пресет"):
                preset_values = presets[selected_preset]
                st.session_state.tuning_settings.update({
                    "vad_onset": preset_values["vad_onset"],
                    "vad_offset": preset_values["vad_offset"],
                    "seg_onset": preset_values["seg_onset"],
                    "seg_offset": preset_values["seg_offset"],
                    "clustering": preset_values["clustering"],
                })
                st.success(f"✅ Применён пресет: {selected_preset}")
                st.rerun()
            
            # Детальные настройки в экспандере
            with st.expander("⚙️ Детальные настройки"):
                st.markdown("**🎙️ Определение активности голоса (VAD)**")
                st.markdown("*Контролирует чувствительность к началу и окончанию речи*")
                
                tuning_settings = st.session_state.tuning_settings
                
                # VAD настройки
                vad_onset = st.slider(
                    "Порог начала речи",
                    min_value=0.1, max_value=0.9, 
                    value=tuning_settings["vad_onset"],
                    step=0.05,
                    help="Меньше = улавливает тихие голоса, больше = только громкие",
                    key="vad_onset_slider"
                )
                
                vad_offset = st.slider(
                    "Порог окончания речи",
                    min_value=0.1, max_value=0.9,
                    value=tuning_settings["vad_offset"],
                    step=0.05,
                    help="Меньше = быстрее определяет конец речи",
                    key="vad_offset_slider"
                )
                
                st.markdown("**📊 Сегментация**")
                st.markdown("*Разбивка записи на фрагменты*")
                
                seg_onset = st.slider(
                    "Порог начала сегмента",
                    min_value=0.1, max_value=0.9,
                    value=tuning_settings["seg_onset"],
                    step=0.05,
                    help="Чувствительность к смене спикера",
                    key="seg_onset_slider"
                )
                
                seg_offset = st.slider(
                    "Порог окончания сегмента",
                    min_value=0.1, max_value=0.9,
                    value=tuning_settings["seg_offset"],
                    step=0.05,
                    help="Определение конца сегмента",
                    key="seg_offset_slider"
                )
                
                st.markdown("**👥 Кластеризация спикеров**")
                st.markdown("*Группировка голосов*")
                
                clustering = st.slider(
                    "Порог схожести спикеров",
                    min_value=0.1, max_value=0.9,
                    value=tuning_settings["clustering"],
                    step=0.05,
                    help="Меньше = больше спикеров, больше = меньше спикеров",
                    key="clustering_slider"
                )
                
                st.markdown("**🔢 Количество спикеров**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_speakers = st.number_input(
                        "Минимум спикеров",
                        min_value=1, max_value=20,
                        value=tuning_settings["min_speakers"] if tuning_settings["min_speakers"] else 1,
                        help="Принудительно установить минимум спикеров",
                        key="min_speakers_input"
                    )
                    if min_speakers == 1:
                        min_speakers = None
                
                with col2:
                    max_speakers = st.number_input(
                        "Максимум спикеров",
                        min_value=2, max_value=20,
                        value=tuning_settings["max_speakers"] if tuning_settings["max_speakers"] else 10,
                        help="Принудительно установить максимум спикеров",
                        key="max_speakers_input"
                    )
                    if max_speakers == 10:
                        max_speakers = None
                
                # Автоматическое сохранение настроек при изменении
                new_settings = {
                    "vad_onset": vad_onset,
                    "vad_offset": vad_offset,
                    "seg_onset": seg_onset,
                    "seg_offset": seg_offset,
                    "clustering": clustering,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
                
                # Сохраняем настройки в session_state
                st.session_state.tuning_settings = new_settings
                
                # Показываем, что настройки изменены
                if new_settings != tuning_settings:
                    st.success("✅ Настройки обновлены - будут применены при следующем анализе")
                
                # Кнопка сброса к стандартным настройкам
                if st.button("🔄 Сбросить к стандартным"):
                    st.session_state.tuning_settings = {
                        "vad_onset": 0.5,
                        "vad_offset": 0.35,
                        "seg_onset": 0.5,
                        "seg_offset": 0.5,
                        "clustering": 0.7154,
                        "min_speakers": None,
                        "max_speakers": None,
                    }
                    st.success("✅ Настройки сброшены к стандартным")
                    st.rerun()
            
            # Рекомендации по оптимизации
            st.subheader("⚡ Рекомендации")
            st.markdown(f"""
            **Для лучшей производительности:**
            - Используйте WAV файлы с частотой {Config.PREFERRED_SAMPLE_RATE}Hz (оптимально)
            - 32kHz работает отлично с моделью
            - Файлы до 100 МБ обрабатываются быстрее
            - GPU обработка в ~10-20 раз быстрее CPU
            
            **Файнтюнинг распознавания:**
            - **Пропущенные голоса?** → Пресет "Высокая чувствительность"
            - **Много ложных срабатываний?** → Пресет "Низкая чувствительность"
            - **Шумная запись?** → Пресет "Шумная запись"
            - **Много спикеров (5+)?** → Пресет "Много спикеров"
            - **Мало спикеров (2-3)?** → Пресет "Мало спикеров"
            
            **Настройка порогов:**
            - **VAD Onset ↓** = улавливает тихие голоса
            - **VAD Offset ↓** = быстрее определяет конец речи
            - **Clustering ↓** = создает больше спикеров
            - **Clustering ↑** = объединяет похожие голоса
            
            **Таймкоды:**
            - Формат: SMPTE 59.94 Drop Frame
            - Разделители: точка с запятой (;)
            - Пример: 00;40;31;16
            
            **Экспорт:**
            - CSV: формат Adobe Audition маркеров
            - Названия: Voice_01, Voice_02, и т.д.
            """)
            
            # Информация о временных файлах
            temp_dir = Path("temp")
            if temp_dir.exists():
                temp_files = list(temp_dir.glob("*"))
                if temp_files:
                    st.info(f"📁 Временных файлов: {len(temp_files)}")
                    # Показываем размер временных файлов
                    total_size = sum(f.stat().st_size for f in temp_files if f.is_file())
                    st.info(f"📊 Размер: {total_size / (1024*1024):.1f} МБ")
                    st.caption("💡 Файлы старше 1 часа удаляются автоматически")
            
            # Очистка временных файлов
            if st.button("🧹 Очистить временные файлы"):
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        st.success("✅ Временные файлы очищены")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ошибка при очистке: {str(e)}")
                else:
                    st.info("ℹ️ Временные файлы не найдены")
        
        # Основная область
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📁 Выбранный файл")
            if selected_file and file_path:
                file_size_mb = os.path.getsize(file_path) / (1024*1024)
                st.info(f"📄 Файл: {selected_file}")
                st.info(f"📊 Размер: {file_size_mb:.2f} MB")
                st.info(f"📍 Путь: {file_path}")
                
                # Предупреждение о больших файлах
                if file_size_mb > 100:
                    st.warning(f"⚠️ Большой файл ({file_size_mb:.1f} МБ) может потребовать много времени для обработки!")
                    estimated_time = max(1, int(file_size_mb * 0.5))
                    st.info(f"⏱️ Примерное время обработки: {estimated_time} минут")
                
                # Кнопка обработки
                if st.button("🚀 Запустить диаризацию", type="primary"):
                    if not self.initialize_pipeline():
                        return
                    
                    self.df_results_original = self.process_audio_file(str(file_path))
                    
                    if self.df_results_original is not None:
                        # Первоначальный пересчет с дефолтным таймкодом
                        self.recalculate_timecodes()
                        st.success("Обработка завершена!")
            else:
                st.info("👈 Выберите аудиофайл в боковой панели для начала работы")
                
                # Показываем инструкцию
                st.markdown("""
                ### 📋 Инструкция по использованию:
                
                1. **Выберите файл** одним из способов:
                   - 📁 Нажмите "Browse files" и выберите аудиофайл
                   - 🗂️ Или введите полный путь к файлу вручную
                
                2. **Настройте файнтюнинг** (если нужно):
                   - 🎛️ Выберите пресет в боковой панели
                   - ⚙️ Или настройте детальные параметры
                   - 🎯 Для пропущенных голосов: "Высокая чувствительность"
                
                3. **Нажмите "Запустить диаризацию"** для обработки
                
                4. **Установите начальный таймкод** в разделе визуализации
                   - Формат: HH;MM;SS;FF (например: 00;40;31;16)
                   - Нажмите Enter для пересчета
                
                5. **Изучите результаты** с таймкодами SMPTE
                
                6. **Экспортируйте результаты** в Adobe Audition формате
                
                ### 🎛️ Решение проблем с распознаванием:
                
                - **Пропущенные тихие голоса?** → Уменьшите VAD Onset до 0.2-0.3
                - **Много ложных срабатываний?** → Увеличьте VAD Onset до 0.7-0.8
                - **Не все спикеры найдены?** → Уменьшите Clustering до 0.5-0.6
                - **Слишком много спикеров?** → Увеличьте Clustering до 0.8-0.9
                - **Шумная запись?** → Используйте пресет "Шумная запись"
                
                ---
                
                **Поддерживаемые форматы:** {formats}
                
                **Максимальный размер:** {max_size} МБ
                """.format(
                    formats=", ".join([f.upper() for f in Config.SUPPORTED_AUDIO_FORMATS]),
                    max_size=Config.MAX_FILE_SIZE_MB
                ))
        
        with col2:
            st.header("📊 Статистика")
            if self.df_results_original is not None:
                total_segments = len(self.df_results_original)
                total_speakers = self.df_results_original['Speaker'].nunique()
                total_duration = self.df_results_original['Duration (sec)'].sum()
                
                st.metric("Сегментов", total_segments)
                st.metric("Спикеров", total_speakers)
                st.metric("Общая длительность", f"{total_duration:.1f}с")
                
                # Показываем текущие настройки файнтюнинга
                st.subheader("🎛️ Текущие настройки")
                tuning = st.session_state.tuning_settings
                
                # Определяем какой пресет ближе всего к текущим настройкам
                presets = Config.DiarizationTuning.get_presets()
                current_preset = "Пользовательские"
                for preset_name, preset_values in presets.items():
                    if (abs(tuning["vad_onset"] - preset_values["vad_onset"]) < 0.01 and
                        abs(tuning["vad_offset"] - preset_values["vad_offset"]) < 0.01 and
                        abs(tuning["clustering"] - preset_values["clustering"]) < 0.01):
                        current_preset = preset_name
                        break
                
                st.info(f"🎯 Пресет: {current_preset}")
                st.info(f"🎙️ VAD: {tuning['vad_onset']:.2f} / {tuning['vad_offset']:.2f}")
                st.info(f"📊 Сегментация: {tuning['seg_onset']:.2f} / {tuning['seg_offset']:.2f}")
                st.info(f"👥 Кластеризация: {tuning['clustering']:.3f}")
                
                if tuning["min_speakers"] or tuning["max_speakers"]:
                    speakers_range = f"{tuning['min_speakers'] or '?'}-{tuning['max_speakers'] or '?'}"
                    st.info(f"🔢 Спикеров: {speakers_range}")
            else:
                st.info("🔄 Запустите анализ для просмотра статистики")
                
                # Показываем настройки даже без результатов
                st.subheader("🎛️ Готовые настройки")
                tuning = st.session_state.tuning_settings
                
                # Определяем текущий пресет
                presets = Config.DiarizationTuning.get_presets()
                current_preset = "Пользовательские"
                for preset_name, preset_values in presets.items():
                    if (abs(tuning["vad_onset"] - preset_values["vad_onset"]) < 0.01 and
                        abs(tuning["vad_offset"] - preset_values["vad_offset"]) < 0.01 and
                        abs(tuning["clustering"] - preset_values["clustering"]) < 0.01):
                        current_preset = preset_name
                        break
                
                st.info(f"🎯 Пресет: {current_preset}")
                st.info(f"🎙️ VAD: {tuning['vad_onset']:.2f} / {tuning['vad_offset']:.2f}")
                st.info(f"📊 Сегментация: {tuning['seg_onset']:.2f} / {tuning['seg_offset']:.2f}")
                st.info(f"👥 Кластеризация: {tuning['clustering']:.3f}")
                
                if tuning["min_speakers"] or tuning["max_speakers"]:
                    speakers_range = f"{tuning['min_speakers'] or '?'}-{tuning['max_speakers'] or '?'}"
                    st.info(f"🔢 Спикеров: {speakers_range}")
        
        # Результаты
        if self.df_results_original is not None:
            st.header("🎯 Результаты диаризации")
            
            # Фильтрация по спикерам
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Фильтр спикеров")
                # Используем df_results если есть, иначе df_results_original
                display_df = self.df_results if self.df_results is not None else self.df_results_original
                speakers = display_df['Speaker'].unique()
                
                # Чекбокс "Все спикеры"
                all_speakers = st.checkbox("Все спикеры", value=True)
                
                if all_speakers:
                    selected_speakers = list(speakers)
                else:
                    selected_speakers = []
                    for speaker in speakers:
                        if st.checkbox(f"Спикер {speaker}", value=False):
                            selected_speakers.append(speaker)
                
                if not selected_speakers:
                    selected_speakers = list(speakers)
            
            with col2:
                # Настройки таймкода
                st.subheader("🕐 Настройки таймкода")
                st.markdown("**SMPTE 59.94 Drop Frame**")
                
                timecode_input = st.text_input(
                    "Начальный таймкод (HH;MM;SS;FF)",
                    value=self.start_timecode,
                    placeholder="00;40;31;16",
                    help="Введите начальный таймкод с вашего таймлайна и нажмите Enter"
                )
                
                # Обработка изменения таймкода
                if timecode_input != self.start_timecode:
                    if SMPTETimeCode.validate_timecode(timecode_input):
                        self.start_timecode = timecode_input
                        if self.recalculate_timecodes():
                            st.success(f"✅ Таймкоды пересчитаны: {self.start_timecode}")
                    else:
                        st.error("❌ Неверный формат таймкода. Используйте HH;MM;SS;FF")
                
                # Фильтрация данных
                display_df = self.df_results if self.df_results is not None else self.df_results_original
                filtered_df = display_df[
                    display_df['Speaker'].isin(selected_speakers)
                ].copy()
                
                # Отображение таблицы
                if self.df_results is not None:
                    # Показываем с SMPTE таймкодами
                    table_df = filtered_df[['Speaker', 'Start (SMPTE)', 'End (SMPTE)', 'Duration (sec)']].copy()
                    table_df = table_df.rename(columns={
                        'Start (SMPTE)': 'Start Timecode',
                        'End (SMPTE)': 'End Timecode',
                        'Duration (sec)': 'Duration (sec)'
                    })
                else:
                    # Показываем только секунды
                    table_df = filtered_df[['Speaker', 'Start (sec)', 'End (sec)', 'Duration (sec)']].copy()
                
                st.subheader("Таблица сегментов")
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Экспорт результатов (только если есть пересчитанные данные)
                if self.df_results is not None:
                    st.subheader("📥 Экспорт результатов")
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        if st.button("Экспорт в CSV (Adobe Audition)"):
                            output_filename = Config.get_output_filename(selected_file, "csv")
                            # Сохраняем в папку с исходным файлом или в текущую папку
                            if file_path and os.path.dirname(str(file_path)) != "temp":
                                output_dir = os.path.dirname(str(file_path))
                            else:
                                output_dir = os.getcwd()
                            output_path = os.path.join(output_dir, output_filename)
                            
                            if self.export_results(filtered_df, output_path, "csv"):
                                st.success(f"✅ Adobe Audition маркеры сохранены: {output_path}")
                                
                                # Кнопка для скачивания
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Скачать CSV",
                                        data=f.read(),
                                        file_name=output_filename,
                                        mime="text/csv"
                                    )
                    
                    with col_exp2:
                        if st.button("Экспорт в Excel"):
                            output_filename = Config.get_output_filename(selected_file, "excel")
                            # Сохраняем в папку с исходным файлом или в текущую папку
                            if file_path and os.path.dirname(str(file_path)) != "temp":
                                output_dir = os.path.dirname(str(file_path))
                            else:
                                output_dir = os.getcwd()
                            output_path = os.path.join(output_dir, output_filename)
                            
                            if self.export_results(filtered_df, output_path, "excel"):
                                st.success(f"✅ Файл сохранен: {output_path}")
                                
                                # Кнопка для скачивания
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Скачать Excel",
                                        data=f.read(),
                                        file_name=output_filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
            
            # Визуализация (только если есть пересчитанные данные)
            if self.df_results is not None:
                st.header("📈 Визуализация")
                fig = self.create_timeline_visualization(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Дополнительная статистика
                st.header("📋 Детальная статистика")
                speaker_stats = filtered_df.groupby('Speaker').agg({
                    'Duration (sec)': ['sum', 'mean', 'count']
                }).round(2)
                
                speaker_stats.columns = ['Общее время (с)', 'Среднее время (с)', 'Количество сегментов']
                st.dataframe(speaker_stats, use_container_width=True)

# Запуск приложения
if __name__ == "__main__":
    app = AudioDiarizationApp()
    app.run() 