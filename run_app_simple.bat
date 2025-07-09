@echo off
chcp 65001 > nul

echo AudioQA - Audio Diarization Tool
echo ==================================
echo.

REM Проверка Python
python --version
if %errorlevel% neq 0 (
    echo Python not found. Install Python 3.8+
    pause
    exit /b 1
)

REM Создание виртуального окружения
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активация
call venv\Scripts\activate.bat

REM Установка минимальных зависимостей
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements_minimal.txt --no-cache-dir

REM Создание .env если его нет
if not exist ".env" (
    echo Creating .env file...
    copy "env.example" ".env"
    echo Edit .env file and add your HUGGINGFACE_TOKEN
    pause
)

REM Запуск
echo Starting application...
echo Browser will open at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run audio_diarization_app.py

pause 