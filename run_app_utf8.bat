@echo off
chcp 65001 > nul
echo.
echo 🎵 AudioQA - Диаризация аудио
echo ================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ошибка: Python не найден. Установите Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python найден

REM Проверка наличия виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo 📦 Создание виртуального окружения...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Ошибка при создании виртуального окружения
        pause
        exit /b 1
    )
    echo ✅ Виртуальное окружение создано
)

REM Активация виртуального окружения
call venv\Scripts\activate.bat

REM Проверка и установка зависимостей
if not exist "venv\Lib\site-packages\streamlit" (
    echo 📚 Установка зависимостей...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Ошибка при установке зависимостей
        pause
        exit /b 1
    )
    echo ✅ Зависимости установлены
)

REM Переустановка PyTorch с CUDA поддержкой (если нужно)
echo 🚀 Проверка и переустановка PyTorch с CUDA...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade

REM Создание .env файла из примера если его нет
if not exist ".env" (
    if exist "env.example" (
        copy "env.example" ".env"
        echo 📝 Создан файл .env из примера
        echo ⚠️  ВНИМАНИЕ: Отредактируйте файл .env и укажите ваш HUGGINGFACE_TOKEN
        echo 🚀 GPU будет использоваться автоматически при наличии
        pause
    )
)

REM Токен Hugging Face прописан в коде, проверка не нужна
echo ✅ Токен Hugging Face: встроен в код

echo.
echo 🚀 Запуск приложения...
echo 🌐 Откроется браузер с адресом: http://localhost:8501
echo.
echo 🛑 Для остановки: закройте это окно или нажмите Ctrl+C в терминале
echo 💡 Или просто закройте браузер и это окно командной строки
echo.

REM Запуск приложения
streamlit run audio_diarization_app.py

echo.
echo Приложение завершено.
pause 