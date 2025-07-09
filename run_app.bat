@echo off
chcp 65001 > nul
echo AudioQA - Diarizatsiia audio
echo ================================

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Oshibka: Python ne naiden. Ustanovite Python 3.8+
    pause
    exit /b 1
)

REM Проверка наличия виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo Sozdanie virtualnogo okruzheniia...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Oshibka pri sozdanii virtualnogo okruzheniia
        pause
        exit /b 1
    )
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Установка зависимостей
if not exist "venv\Lib\site-packages\streamlit" (
    echo Ustanovka zavisimostei...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Oshibka pri ustanovke zavisimostei
        pause
        exit /b 1
    )
)

REM Проверка токена Hugging Face
if "%HUGGINGFACE_TOKEN%"=="" (
    echo.
    echo VNIMANIE: Ne naiden token Hugging Face
    echo Dlia raboty prilozheniia trebuetsia token Hugging Face.
    echo.
    echo Poluchite token na: https://huggingface.co/settings/tokens
    echo.
    echo Ustanovite token odnim iz sposobov:
    echo 1. Sozdaite fail .env s soderzhimym: HUGGINGFACE_TOKEN=your_token_here
    echo 2. Ustanovite peremennuiu okruzheniia: set HUGGINGFACE_TOKEN=your_token_here
    echo.
    pause
)

REM Запуск приложения
echo Zapusk prilozheniia...
echo Otkroetsia brauzer s adresom: http://localhost:8501
echo.
echo Dlia ostanovki nazhmite Ctrl+C
echo.
streamlit run audio_diarization_app.py

pause 