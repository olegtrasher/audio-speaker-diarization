#!/bin/bash

echo "🎵 AudioQA - Диаризация аудио"
echo "================================"

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

# Проверка наличия виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Ошибка при создании виртуального окружения"
        exit 1
    fi
fi

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей
if [ ! -f "venv/lib/python*/site-packages/streamlit" ]; then
    echo "Установка зависимостей..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Ошибка при установке зависимостей"
        exit 1
    fi
fi

# Проверка токена Hugging Face
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo ""
    echo "⚠️  ВНИМАНИЕ: Не найден токен Hugging Face"
    echo "Для работы приложения требуется токен Hugging Face."
    echo ""
    echo "Получите токен на: https://huggingface.co/settings/tokens"
    echo ""
    echo "Установите токен одним из способов:"
    echo "1. Создайте файл .env с содержимым: HUGGINGFACE_TOKEN=your_token_here"
    echo "2. Установите переменную окружения: export HUGGINGFACE_TOKEN=your_token_here"
    echo ""
    read -p "Нажмите Enter для продолжения..."
fi

# Запуск приложения
echo "Запуск приложения..."
echo "Откроется браузер с адресом: http://localhost:8501"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""
streamlit run audio_diarization_app.py 