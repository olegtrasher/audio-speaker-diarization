# 📋 Changelog - AudioQA

## 🎯 Версия 2.0.0 - Новый интерфейс с прямым выбором файлов

### ✨ Новые возможности

#### 📁 Упрощенный выбор файлов
- **Кнопка Browse** - простой выбор файла из системы через стандартный диалог
- **Ввод пути вручную** - альтернативный способ указания полного пути к файлу
- **Автоматическая проверка** - размер, формат и доступность файла проверяются автоматически

#### 🧹 Управление временными файлами
- **Автоочистка** - файлы старше 1 часа удаляются автоматически
- **Информация о файлах** - отображение количества и размера временных файлов
- **Ручная очистка** - кнопка для немедленного удаления временных файлов

#### 📥 Улучшенный экспорт
- **Локальное сохранение** - файлы сохраняются рядом с исходным файлом
- **Прямое скачивание** - кнопки для скачивания результатов прямо из браузера
- **Два формата** - CSV и Excel с правильными MIME-типами

#### 🎨 Улучшенный интерфейс
- **Инструкции по использованию** - показываются при отсутствии файла
- **Расширенная информация** - путь к файлу, размер, оценка времени обработки
- **Предупреждения** - о больших файлах и времени обработки

### 🔧 Технические улучшения

#### 🗂️ Управление файлами
- **Уникальные имена** - временные файлы получают уникальные имена с timestamp
- **Безопасность** - проверка на существование и доступность файлов
- **Обработка ошибок** - корректная обработка ошибок при работе с файлами

#### ⚙️ Конфигурация
- **Удаление DEFAULT_WORK_FOLDER** - больше не нужно настраивать рабочую папку
- **Поддержка переменных окружения** - MAX_FILE_SIZE_MB теперь настраивается через .env
- **Улучшенная валидация** - проверка размера файла и формата

#### 🔄 Производительность
- **Оценка времени обработки** - показывается для больших файлов
- **Отображение размера файла** - в MB с точностью до 2 знаков
- **Оптимизация загрузки** - файлы сохраняются только при необходимости

### 🚫 Удаленные возможности

#### 🗂️ Работа с папками
- **Выбор рабочей папки** - больше не нужно указывать папку для поиска
- **Список файлов в папке** - замещено прямым выбором файла
- **Автоматическое сканирование** - файлы выбираются напрямую

### 📚 Обновленная документация

#### 📖 Новые файлы
- **INTERFACE_GUIDE.md** - подробное руководство по новому интерфейсу
- **CHANGELOG.md** - история изменений приложения

#### 📝 Обновленные файлы
- **README.md** - обновлены инструкции по использованию
- **QUICKSTART.md** - добавлена информация о новом интерфейсе
- **demo_data.py** - обновлены инструкции для демо-данных

### 🔧 Исправления

#### 🐛 Исправленные ошибки
- **Кодировка в Windows** - исправлены проблемы с UTF-8 в батч-файлах
- **Версии зависимостей** - использование гибких версий вместо жестких
- **Обработка ошибок** - улучшена обработка ошибок при работе с файлами

#### 🔄 Обновления
- **st.rerun()** - заменен устаревший st.experimental_rerun()
- **accept_multiple_files=False** - явно указано для предотвращения множественного выбора
- **Импорты** - добавлены необходимые импорты (shutil, time)

### 📦 Структура проекта

```
AudioQA/
├── 📄 audio_diarization_app.py    # Основное приложение (обновлено)
├── ⚙️ config.py                   # Конфигурация (обновлено)
├── 🧪 demo_data.py                # Демо-данные (обновлено)
├── 📝 requirements.txt            # Зависимости (обновлено)
├── 🚀 run_app_utf8.bat           # Запуск с UTF-8 (новое)
├── 🚀 run_app_simple.bat         # Простой запуск (новое)
├── 📖 README.md                   # Основная документация (обновлено)
├── 📋 QUICKSTART.md              # Быстрый старт (обновлено)
├── 🎯 INTERFACE_GUIDE.md         # Руководство по интерфейсу (новое)
├── 📜 CHANGELOG.md               # История изменений (новое)
├── 🔧 .env.example               # Пример настроек (обновлено)
├── 🚫 .gitignore                 # Исключения Git (обновлено)
└── 🗂️ temp/                     # Временные файлы (новое)
```

### 🎯 Влияние на пользователей

#### ✅ Преимущества
- **Проще использовать** - не нужно настраивать рабочую папку
- **Более гибко** - можно выбирать файлы из любого места
- **Безопаснее** - автоматическая очистка временных файлов
- **Удобнее** - прямое скачивание результатов

#### ⚠️ Изменения в использовании
- **Новый способ выбора файлов** - через кнопку Browse или ввод пути
- **Временные файлы** - загруженные файлы сохраняются в папке temp
- **Экспорт** - файлы сохраняются рядом с исходным файлом

### 🚀 Планы на будущее

#### 🔮 Возможные улучшения
- **Drag & Drop** - перетаскивание файлов в интерфейс
- **Пакетная обработка** - обработка нескольких файлов одновременно
- **Облачное хранение** - интеграция с облачными сервисами
- **Предварительный просмотр** - воспроизведение аудио в интерфейсе

#### 🎯 Оптимизации
- **Прогресс-бар** - более детальное отображение прогресса
- **Кэширование** - сохранение результатов для повторного использования
- **Параллельная обработка** - обработка нескольких файлов параллельно

---

**Версия 2.0.0** представляет собой значительное улучшение пользовательского опыта с сохранением всех основных функций диаризации! 🎉

*Дата выпуска: Декабрь 2024* 