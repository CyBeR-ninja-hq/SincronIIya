import sys
import os
import threading
import time
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torchaudio
import pyaudio
import wave
import json
import soundfile as sf  # Добавляем импорт
from ui.elements import *
from ui.styles import *
from core.audio_recorder import AudioRecorder
from core.audio_processor import *

import ollama

# ИМПОРТИРУЕМ МОДЕЛИ В НАЧАЛЕ ФАЙЛА
try:
    from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Ошибка импорта transformers: {e}")
    TRANSFORMERS_AVAILABLE = False
    HubertForSequenceClassification = None
    Wav2Vec2FeatureExtractor = None

class EmotionRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_extractor = None
        # Убрали эмоцию "другая"
        self.num2emotion = {0: 'нейтральная', 1: 'гнев', 2: 'радость', 3: 'грусть'}
        self.current_file = None
        self.recorder = None
        self.audio_processor = None
        self.init_ui()
        self.load_model_async()
        # AI 
        self.ai_goal = ""
        self.word_counter = 0
        self.words_for_ai = 50  # количество слов для отправки в AI
        self.conversation_history = []
        self.dominant_emotion = "нейтральная"
        self.emotion_counter = {}
        self.ai_thread = None  # Поток для работы с ИИ
        self.ai_advice_queue = []  # Очередь для хранения полученных советов
        
    def init_ui(self):
        self.setWindowTitle("СинхронИИя - Распознавание эмоций и речи")
        self.setGeometry(100, 100, 1400, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Заголовок с увеличенным шрифтом на 20%
        title_label = QLabel("СинхронИИя - Распознавание эмоций и речи")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(Styles.TITLE_FONT_SIZE)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #3498db; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Разделитель
        main_layout.addWidget(self.create_separator())
        
        # Создание виджета вкладок
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(Styles.get_tab_widget_style())
        
        # Вкладка анализа файлов
        file_tab = QWidget()
        self.setup_file_tab(file_tab)
        self.tab_widget.addTab(file_tab, "📁 Анализ файлов")
        
        # Вкладка анализа в реальном времени
        realtime_tab = QWidget()
        self.setup_realtime_tab(realtime_tab)
        self.tab_widget.addTab(realtime_tab, "🎤 Анализ в реальном времени")
        
        # Новая вкладка распознавания речи
        speech_tab = QWidget()
        self.setup_speech_tab(speech_tab)
        self.tab_widget.addTab(speech_tab, "🗣️ Распознавание речи")
        
        # Новая вкладка ИИ советника
        ai_advisor_tab = QWidget()
        self.setup_ai_advisor_tab(ai_advisor_tab)
        self.tab_widget.addTab(ai_advisor_tab, "🤖 ИИ советник")
        
        main_layout.addWidget(self.tab_widget)
        
        # Строка состояния
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Загрузка модели...")
        self.status_bar.setStyleSheet(f"color: {Styles.MUTED_TEXT_COLOR}; background-color: {Styles.BACKGROUND_COLOR};")
        
        # Установка стилей
        self.setStyleSheet(Styles.get_main_window_style())
        
    def setup_file_tab(self, tab):
        """Настройка вкладки анализа файлов"""
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Секция выбора файла
        file_group = QGroupBox("📁 Аудиофайл")
        file_group.setStyleSheet(Styles.get_groupbox_style())
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet(Styles.get_file_label_style())
        
        browse_btn = QPushButton("📂 Обзор...")
        browse_btn.clicked.connect(self.browse_file)
        browse_btn.setFixedWidth(120)
        browse_btn.setStyleSheet(Styles.get_button_style(primary=True, height=30))
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Отображение информации об аудио
        self.audio_info_label = QLabel("")
        self.audio_info_label.setStyleSheet(Styles.get_audio_info_label_style())
        self.audio_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.audio_info_label)
        
        # Секция отображения эмоций
        emotion_group = QGroupBox("🎭 Результат распознавания эмоций")
        emotion_group.setStyleSheet(Styles.get_groupbox_style())
        emotion_layout = QVBoxLayout()
        
        # Метка эмоции
        self.emotion_label = QLabel("Ожидание анализа...")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        emotion_font = QFont()
        emotion_font.setPointSize(Styles.EMOTION_FONT_SIZE)
        emotion_font.setBold(True)
        self.emotion_label.setFont(emotion_font)
        self.emotion_label.setMinimumHeight(84)
        self.emotion_label.setStyleSheet(Styles.get_emotion_label_style())
        
        # Отображение уверенности
        self.confidence_label = QLabel("Уверенность: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        confidence_font = QFont()
        confidence_font.setPointSize(Styles.CONFIDENCE_FONT_SIZE)
        self.confidence_label.setFont(confidence_font)
        self.confidence_label.setStyleSheet(Styles.get_confidence_label_style())
        
        emotion_layout.addWidget(self.emotion_label)
        emotion_layout.addWidget(self.confidence_label)
        emotion_group.setLayout(emotion_layout)
        layout.addWidget(emotion_group)
        
        # Кнопка анализа
        self.analyze_btn = QPushButton("🔍 Анализировать эмоции")
        self.analyze_btn.clicked.connect(self.analyze_emotion)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setStyleSheet(Styles.get_button_style(primary=True, height=50))
        layout.addWidget(self.analyze_btn)
        
        # Индикатор выполнения
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(Styles.get_progress_bar_style())
        layout.addWidget(self.progress_bar)
        
        layout.addStretch(1)
        
    def setup_realtime_tab(self, tab):
        """Настройка вкладки анализа в реальном времени"""
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Секция управления
        control_group = QGroupBox("🎛️ Управление записью")
        control_group.setStyleSheet(Styles.get_groupbox_style())
        control_layout = QVBoxLayout()
        
        # Выбор устройства микрофона
        device_layout = QHBoxLayout()
        device_label = QLabel("🎤 Микрофон:")
        device_label.setStyleSheet(Styles.get_device_label_style())
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.setStyleSheet(Styles.get_combo_box_style())
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        control_layout.addLayout(device_layout)
        
        # Слайдер длины батча
        slider_layout = QHBoxLayout()
        slider_label = QLabel("⏱️ Длина батча:")
        slider_label.setStyleSheet(Styles.get_device_label_style())
        self.batch_length_slider = QSlider(Qt.Horizontal)
        self.batch_length_slider.setMinimum(1)
        self.batch_length_slider.setMaximum(10)
        self.batch_length_slider.setValue(3)
        self.batch_length_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_length_slider.setTickInterval(1)
        self.batch_length_slider.setStyleSheet(Styles.get_slider_style())
        
        self.batch_length_label = QLabel("3.0 секунд")
        self.batch_length_label.setStyleSheet(Styles.get_batch_length_label_style())
        self.batch_length_slider.valueChanged.connect(self.update_batch_length_label)
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.batch_length_slider)
        slider_layout.addWidget(self.batch_length_label)
        control_layout.addLayout(slider_layout)
        
        # Кнопки записи
        button_layout = QHBoxLayout()
        
        self.start_realtime_btn = QPushButton("▶️ Начать анализ")
        self.start_realtime_btn.clicked.connect(self.start_realtime_analysis)
        self.start_realtime_btn.setMinimumHeight(40)
        self.start_realtime_btn.setStyleSheet(Styles.get_button_style(primary=True, height=40))
        
        self.stop_realtime_btn = QPushButton("⏹️ Остановить")
        self.stop_realtime_btn.clicked.connect(self.stop_realtime_analysis)
        self.stop_realtime_btn.setEnabled(False)
        self.stop_realtime_btn.setMinimumHeight(40)
        self.stop_realtime_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.ERROR_COLOR))
        
        self.save_audio_btn = QPushButton("💾 Сохранить запись")
        self.save_audio_btn.clicked.connect(self.save_full_realtime_audio)
        self.save_audio_btn.setEnabled(False)
        self.save_audio_btn.setMinimumHeight(40)
        self.save_audio_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.SUCCESS_COLOR))
        
        button_layout.addWidget(self.start_realtime_btn)
        button_layout.addWidget(self.stop_realtime_btn)
        button_layout.addWidget(self.save_audio_btn)
        control_layout.addLayout(button_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Отображение эмоций в реальном времени
        realtime_emotion_group = QGroupBox("🎭 Текущая эмоция")
        realtime_emotion_group.setStyleSheet(Styles.get_groupbox_style())
        realtime_emotion_layout = QVBoxLayout()
        
        self.realtime_emotion_label = QLabel("Не анализируется")
        self.realtime_emotion_label.setAlignment(Qt.AlignCenter)
        realtime_emotion_font = QFont()
        realtime_emotion_font.setPointSize(Styles.REALTIME_EMOTION_FONT_SIZE)
        realtime_emotion_font.setBold(True)
        self.realtime_emotion_label.setFont(realtime_emotion_font)
        self.realtime_emotion_label.setMinimumHeight(70)
        self.realtime_emotion_label.setStyleSheet(Styles.get_realtime_emotion_label_style())
        
        self.realtime_confidence_label = QLabel("Уверенность: --")
        self.realtime_confidence_label.setAlignment(Qt.AlignCenter)
        self.realtime_confidence_label.setStyleSheet(f"color: {Styles.PRIMARY_COLOR}; font-size: {Styles.CONFIDENCE_FONT_SIZE}px; padding: 10px;")
        realtime_emotion_layout.addWidget(self.realtime_emotion_label)
        realtime_emotion_layout.addWidget(self.realtime_confidence_label)
        realtime_emotion_group.setLayout(realtime_emotion_layout)
        layout.addWidget(realtime_emotion_group)
        
        # График для визуализации в реальном времени
        chart_group = QGroupBox("📈 График уверенности в эмоциях")
        chart_group.setStyleSheet(Styles.get_groupbox_style())
        chart_layout = QVBoxLayout()
        
        # Создание холста matplotlib
        self.canvas = MplCanvas(self, width=8, height=4, dpi=100)
        
        # Добавление панели навигации Matplotlib
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Метка состояния для реального времени
        self.realtime_status_label = QLabel("Готово")
        self.realtime_status_label.setAlignment(Qt.AlignCenter)
        self.realtime_status_label.setStyleSheet(Styles.get_status_label_style())
        layout.addWidget(self.realtime_status_label)
        
        layout.addStretch(1)
        
    def setup_speech_tab(self, tab):
        """Настройка вкладки распознавания речи"""
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Область для отображения распознанного текста
        text_group = QGroupBox("📝 Распознанный текст")
        text_group.setStyleSheet(Styles.get_groupbox_style())
        text_layout = QVBoxLayout()
        
        # Создаем QTextEdit с прокруткой
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet(Styles.get_text_edit_style())
        self.text_display.setMinimumHeight(300)
        
        text_layout.addWidget(self.text_display)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Кнопки управления текстом
        text_buttons_layout = QHBoxLayout()
        
        self.clear_text_btn = QPushButton("🧹 Очистить текст")
        self.clear_text_btn.clicked.connect(self.clear_recognized_text)
        self.clear_text_btn.setMinimumHeight(40)
        self.clear_text_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.WARNING_COLOR))
        
        self.save_text_btn = QPushButton("💾 Сохранить текст")
        self.save_text_btn.clicked.connect(self.save_recognized_text)
        self.save_text_btn.setMinimumHeight(40)
        self.save_text_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.SUCCESS_COLOR))
        
        text_buttons_layout.addWidget(self.clear_text_btn)
        text_buttons_layout.addWidget(self.save_text_btn)
        text_buttons_layout.addStretch()
        
        layout.addLayout(text_buttons_layout)
        
        # Легенда цветов эмоций (без эмоции "другая")
        legend_group = QGroupBox("🎨 Легенда цветов эмоций")
        legend_group.setStyleSheet(Styles.get_groupbox_style())
        legend_layout = QHBoxLayout()
        
        # Цвета для эмоций (только 4 основные)
        emotion_colors = {
            'нейтральная': Styles.EMOTION_COLORS.get('нейтральная', '#808080'),
            'гнев': Styles.EMOTION_COLORS.get('гнев', '#e74c3c'),
            'радость': Styles.EMOTION_COLORS.get('радость', '#f39c12'),
            'грусть': Styles.EMOTION_COLORS.get('грусть', '#3498db')
        }
        
        for emotion, color in emotion_colors.items():
            color_widget = QWidget()
            color_widget.setFixedSize(20, 20)
            color_widget.setStyleSheet(f"background-color: {color}; border-radius: 3px; border: 1px solid {Styles.BORDER_COLOR};")
            
            label = QLabel(f"{emotion.capitalize()}")
            label.setStyleSheet(f"color: {Styles.TEXT_COLOR}; font-size: 12px; padding: 2px;")
            
            hbox = QHBoxLayout()
            hbox.addWidget(color_widget)
            hbox.addWidget(label)
            hbox.addSpacing(10)
            
            container = QWidget()
            container.setLayout(hbox)
            legend_layout.addWidget(container)
        
        legend_layout.addStretch()
        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)
        
        # Статус распознавания речи
        self.speech_status_label = QLabel("Модель Vosk загружается при запуске...")
        self.speech_status_label.setAlignment(Qt.AlignCenter)
        self.speech_status_label.setStyleSheet(Styles.get_status_label_style())
        layout.addWidget(self.speech_status_label)
        
        layout.addStretch(1)
        
    def setup_ai_advisor_tab(self, tab):
        """Настройка вкладки ИИ советника"""
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Секция цели разговора
        goal_group = QGroupBox("🎯 Установить цель разговора")
        goal_group.setStyleSheet(Styles.get_groupbox_style())
        goal_layout = QVBoxLayout()
        
        self.goal_text_edit = QTextEdit()
        self.goal_text_edit.setPlaceholderText("Введите цель разговора (например: 'Узнать о хобби собеседника', 'Продать продукт', 'Поддержать в трудной ситуации')")
        self.goal_text_edit.setMaximumHeight(100)
        self.goal_text_edit.setStyleSheet(Styles.get_text_edit_style())
        
        goal_button_layout = QHBoxLayout()
        self.set_goal_btn = QPushButton("✅ Установить цель")
        self.set_goal_btn.clicked.connect(self.set_conversation_goal)
        self.set_goal_btn.setMinimumHeight(40)
        self.set_goal_btn.setStyleSheet(Styles.get_button_style(primary=True, height=40))
        
        self.clear_goal_btn = QPushButton("🧹 Очистить цель")
        self.clear_goal_btn.clicked.connect(self.clear_conversation_goal)
        self.clear_goal_btn.setMinimumHeight(40)
        self.clear_goal_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.WARNING_COLOR))
        
        goal_button_layout.addWidget(self.set_goal_btn)
        goal_button_layout.addWidget(self.clear_goal_btn)
        goal_button_layout.addStretch()
        
        goal_layout.addWidget(self.goal_text_edit)
        goal_layout.addLayout(goal_button_layout)
        goal_group.setLayout(goal_layout)
        layout.addWidget(goal_group)
        
        # Секция статистики
        stats_group = QGroupBox("📊 Статистика разговора")
        stats_group.setStyleSheet(Styles.get_groupbox_style())
        stats_layout = QGridLayout()
        
        # Счетчик слов
        words_label = QLabel("Распознано слов:")
        words_label.setStyleSheet(Styles.get_device_label_style())
        
        self.words_counter_label = QLabel("0")
        self.words_counter_label.setStyleSheet(f"color: {Styles.PRIMARY_COLOR}; font-size: 24px; font-weight: bold;")
        
        # Доминирующая эмоция
        emotion_label = QLabel("Доминирующая эмоция:")
        emotion_label.setStyleSheet(Styles.get_device_label_style())
        
        self.dominant_emotion_label = QLabel("нейтральная")
        self.dominant_emotion_label.setStyleSheet(f"""
            QLabel {{
                color: {Styles.EMOTION_COLORS.get('нейтральная', '#808080')};
                font-size: 20px;
                font-weight: bold;
                padding: 5px;
                border-radius: 5px;
                background-color: {Styles.SECONDARY_COLOR};
                border: 2px solid {Styles.EMOTION_COLORS.get('нейтральная', '#808080')};
            }}
        """)
        
        # Слайдер для выбора количества слов для AI
        slider_words_label = QLabel("Слов для AI:")
        slider_words_label.setStyleSheet(Styles.get_device_label_style())
        
        self.words_slider_label = QLabel("50 слов")
        self.words_slider_label.setStyleSheet(Styles.get_batch_length_label_style())
        
        # Размещение элементов в сетке
        stats_layout.addWidget(words_label, 0, 0)
        stats_layout.addWidget(self.words_counter_label, 0, 1)
        stats_layout.addWidget(emotion_label, 1, 0)
        stats_layout.addWidget(self.dominant_emotion_label, 1, 1)
        stats_layout.addWidget(slider_words_label, 2, 0)
        stats_layout.addWidget(self.words_slider_label, 2, 1)
        
        # Слайдер для выбора количества слов для AI
        self.words_slider = QSlider(Qt.Horizontal)
        self.words_slider.setMinimum(10)
        self.words_slider.setMaximum(200)
        self.words_slider.setValue(50)
        self.words_slider.setTickPosition(QSlider.TicksBelow)
        self.words_slider.setTickInterval(10)
        self.words_slider.setStyleSheet(Styles.get_slider_style())
        self.words_slider.valueChanged.connect(self.update_words_slider_label)
        
        stats_layout.addWidget(self.words_slider, 3, 0, 1, 2)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Секция советов от ИИ
        advice_group = QGroupBox("💡 Советы от ИИ")
        advice_group.setStyleSheet(Styles.get_groupbox_style())
        advice_layout = QVBoxLayout()
        
        self.ai_advice_text = QTextEdit()
        self.ai_advice_text.setReadOnly(True)
        self.ai_advice_text.setPlaceholderText("Здесь будут появляться советы от ИИ по достижению цели разговора")
        self.ai_advice_text.setMinimumHeight(200)
        self.ai_advice_text.setStyleSheet(Styles.get_text_edit_style())
        
        advice_button_layout = QHBoxLayout()
        self.get_advice_btn = QPushButton("🤖 Получить совет")
        self.get_advice_btn.clicked.connect(self.get_ai_advice_async)
        self.get_advice_btn.setMinimumHeight(40)
        self.get_advice_btn.setStyleSheet(Styles.get_button_style(primary=True, height=40))
        
        self.clear_advice_btn = QPushButton("🧹 Очистить советы")
        self.clear_advice_btn.clicked.connect(self.clear_ai_advice)
        self.clear_advice_btn.setMinimumHeight(40)
        self.clear_advice_btn.setStyleSheet(Styles.get_button_style(primary=False, height=40, color=Styles.WARNING_COLOR))
        
        advice_button_layout.addWidget(self.get_advice_btn)
        advice_button_layout.addWidget(self.clear_advice_btn)
        advice_button_layout.addStretch()
        
        advice_layout.addWidget(self.ai_advice_text)
        advice_layout.addLayout(advice_button_layout)
        advice_group.setLayout(advice_layout)
        layout.addWidget(advice_group)
        
        # Индикатор работы ИИ
        self.ai_status_label = QLabel("ИИ советник готов к работе")
        self.ai_status_label.setAlignment(Qt.AlignCenter)
        self.ai_status_label.setStyleSheet(Styles.get_status_label_style())
        layout.addWidget(self.ai_status_label)
        
        layout.addStretch(1)
    
    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(Styles.get_separator_style())
        return line
        
    def update_batch_length_label(self, value):
        """Обновление метки длины батча при изменении слайдера"""
        self.batch_length_label.setText(f"{value}.0 секунд")
        
    def update_words_slider_label(self, value):
        """Обновление метки слайдера количества слов"""
        self.words_slider_label.setText(f"{value} слов")
        self.words_for_ai = value
        
    def load_model_async(self):
        """Загрузка модели в отдельном потоке для предотвращения зависания UI"""
        self.status_bar.showMessage("Загрузка модели...")
        QApplication.processEvents()
        
        try:
            # Проверяем доступность transformers
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Библиотека transformers не установлена. Установите: pip install transformers")
            
            # Загрузка модели и экстрактора признаков
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-large-ls960-ft"
            )
            
            self.model = HubertForSequenceClassification.from_pretrained(
                "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
            )
            self.model.eval()
            
            # Проверяем количество меток в модели
            print(f"Количество меток в модели: {self.model.config.num_labels}")
            
            # Если модель исходно была для 5 эмоций, адаптируем выходной слой
            if self.model.config.num_labels != 4:
                print(f"Адаптация модели с {self.model.config.num_labels} меток к 4 меткам...")
                # Просто игнорируем последнюю метку при обработке
                # Вместо пересоздания модели, будем правильно интерпретировать выходы
                pass
            
            # Инициализация аудио процессора
            self.audio_processor = AudioProcessor(
                self.model, 
                self.feature_extractor, 
                self.num2emotion
            )
            
            # Загрузка модели Vosk
            if self.audio_processor.init_vosk():
                self.speech_status_label.setText("Модель Vosk успешно загружена")
            else:
                self.speech_status_label.setText("⚠️ Модель Vosk не найдена! Скачайте с: https://alphacephei.com/vosk/models")
            
            # Подключение сигналов
            self.audio_processor.emotion_detected.connect(self.update_realtime_display)
            self.audio_processor.speech_recognized.connect(self.on_text_recognized)
            
            # Заполнение списка устройств микрофона
            self.populate_microphone_devices()
            
            self.status_bar.showMessage("Модель успешно загружена!")
            self.analyze_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")
            self.status_bar.showMessage("Не удалось загрузить модель")
            print(f"Подробности ошибки: {e}")
            
    def populate_microphone_devices(self):
        """Заполнение выпадающего списка устройств микрофона"""
        try:
            self.device_combo.clear()
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    device_name = dev_info['name']
                    self.device_combo.addItem(f"{device_name} (Устройство {i})", i)
            
            p.terminate()
            
            # Выбор устройства по умолчанию
            if self.device_combo.count() > 0:
                self.device_combo.setCurrentIndex(0)
                
        except Exception as e:
            print(f"Ошибка заполнения устройств: {e}")
            
    def browse_file(self):
        """Открытие диалога выбора аудиофайла"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Аудиофайлы (*.wav *.mp3 *.flac *.ogg *.m4a *.opus)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setStyleSheet(Styles.get_file_dialog_style())
        
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                self.current_file = files[0]
                self.file_label.setText(Path(self.current_file).name)
                self.analyze_btn.setEnabled(True)
                self.emotion_label.setText("Готов к анализу")
                self.emotion_label.setStyleSheet(Styles.get_emotion_label_style())
                self.confidence_label.setText("Уверенность: --")
                self.audio_info_label.setText("")
                
                # Показать информацию об аудиофайле
                try:
                    info = sf.info(self.current_file)
                    duration = info.duration
                    samplerate = info.samplerate
                    channels = info.channels
                    self.audio_info_label.setText(
                        f"Длительность: {duration:.1f}с | Частота: {samplerate}Гц | Каналы: {channels}"
                    )
                    
                    # Предупреждение, если аудио слишком короткое
                    if duration < 0.5:
                        QMessageBox.warning(self, "Предупреждение", 
                                          f"Аудиофайл очень короткий ({duration:.2f} секунд).\n"
                                          f"Модель может работать некорректно с аудио короче 1 секунды.")
                    
                except:
                    self.audio_info_label.setText("Не удалось прочитать информацию об аудиофайле")
    
    def normalize_audio(self, audio_data, target_sample_rate=16000, min_duration=1.0):
        """
        Комплексная функция нормализации аудио
        - Конвертирует стерео в моно
        - Нормализует амплитуду до [-1, 1]
        - Обеспечивает минимальную длительность
        - Передискретизирует до целевой частоты
        """
        # Конвертация в numpy array, если это torch tensor
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        
        # Обработка стерео аудио
        if audio_data.ndim > 1:
            # Усреднение по каналам (axis=1 для формы [семплы, каналы])
            if audio_data.shape[0] < audio_data.shape[1]:
                # Форма [каналы, семплы]
                audio_data = np.mean(audio_data, axis=0)
            else:
                # Форма [семплы, каналы]
                audio_data = np.mean(audio_data, axis=1)
        
        # Нормализация амплитуды до [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def ensure_minimum_length(self, audio_data, sample_rate, min_seconds=1.0):
        """
        Обеспечение минимальной длины аудио путем повторения при необходимости
        Возвращает аудио с минимальной длительностью min_seconds
        """
        min_samples = int(min_seconds * sample_rate)
        current_samples = len(audio_data)
        
        if current_samples < min_samples:
            # Расчет необходимого количества повторений
            repeats_needed = int(np.ceil(min_samples / current_samples))
            
            # Повторение аудио
            audio_data = np.tile(audio_data, repeats_needed)
            
            # Обрезка до точной минимальной длины при необходимости
            if len(audio_data) > min_samples:
                audio_data = audio_data[:min_samples]
        
        return audio_data
    
    def load_and_preprocess_audio(self, filepath):
        """
        Загрузка аудиофайла и предобработка для модели HuBERT
        Возвращает нормализованное аудио с частотой 16кГц и минимальной длительностью 1 секунда
        """
        try:
            # Попытка различных методов загрузки аудио
            audio_data = None
            sample_rate = None
            
            # Метод 1: Попытка soundfile
            try:
                audio_data, sample_rate = sf.read(filepath)
            except:
                # Метод 2: Попытка torchaudio
                try:
                    waveform, sample_rate = torchaudio.load(filepath, normalize=True)
                    audio_data = waveform.numpy()
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=0)
                except:
                    # Метод 3: Попытка librosa
                    try:
                        import librosa
                        audio_data, sample_rate = librosa.load(filepath, sr=None, mono=True)
                    except Exception as e:
                        raise Exception(f"Все методы загрузки аудио не удались: {e}")
            
            # Нормализация аудио
            audio_data = self.normalize_audio(audio_data)
            
            # Обеспечение минимальной длины
            audio_data = self.ensure_minimum_length(audio_data, sample_rate, min_seconds=1.0)
            
            # Передискретизация до 16кГц при необходимости
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                except:
                    # Резервный вариант: передискретизация torchaudio
                    waveform = torch.FloatTensor(audio_data).unsqueeze(0)
                    transform = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio_data = transform(waveform).squeeze(0).numpy()
                    sample_rate = 16000
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise Exception(f"Не удалось загрузить и предобработать аудио: {str(e)}")
    
    def analyze_emotion(self):
        """Анализ эмоций из выбранного аудиофайла"""
        if not self.current_file or not self.model:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, сначала выберите аудиофайл")
            return
            
        try:
            # Показать прогресс
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.analyze_btn.setEnabled(False)
            self.status_bar.showMessage("Обработка аудио...")
            QApplication.processEvents()
            
            # Шаг 1: Загрузка и предобработка аудио
            self.progress_bar.setValue(25)
            self.status_bar.showMessage("Загрузка и предобработка аудио...")
            
            audio_data, sample_rate = self.load_and_preprocess_audio(self.current_file)
            
            # Шаг 2: Извлечение признаков
            self.progress_bar.setValue(50)
            self.status_bar.showMessage("Извлечение признаков...")
            
            # Обеспечение достаточной длины аудио
            if len(audio_data) < 10:
                audio_data = np.pad(audio_data, (0, 10 - len(audio_data)), mode='constant')
            
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )
            
            # Шаг 3: Создание предсказания
            self.progress_bar.setValue(75)
            self.status_bar.showMessage("Анализ эмоций...")
            
            with torch.no_grad():
                input_values = inputs['input_values']
                
                # Обработка различных размерностей ввода
                if input_values.dim() == 4:
                    input_values = input_values.squeeze(1).squeeze(1)
                elif input_values.dim() == 3:
                    input_values = input_values.squeeze(1)
                
                # Двойная проверка длины ввода
                if input_values.shape[1] < 10:
                    padding = 10 - input_values.shape[1]
                    input_values = torch.nn.functional.pad(input_values, (0, padding), mode='constant', value=0)
                
                # Создание предсказания
                logits = self.model(input_values).logits
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                predicted_class = predictions.numpy()[0]
                
                # Если модель возвращает 5 классов, а у нас 4, берем только первые 4
                if probabilities.shape[1] > 4:
                    probabilities = probabilities[:, :4]  # Берем только первые 4 класса
                    # Нормализуем вероятности
                    probabilities = torch.nn.functional.softmax(probabilities, dim=-1)
                    if predicted_class >= 4:
                        predicted_class = 3  # Переназначаем на последний доступный класс
                
                confidence = probabilities[0][predicted_class].item() * 100
                predicted_emotion = self.num2emotion[predicted_class]
                
                # Получение вероятностей всех эмоций
                all_probs = {}
                for i, emotion in self.num2emotion.items():
                    all_probs[emotion] = probabilities[0][i].item() * 100
            
            # Шаг 4: Обновление UI
            self.progress_bar.setValue(100)
            
            # Обновление отображения эмоций
            self.emotion_label.setText(predicted_emotion.upper())
            self.confidence_label.setText(f"Уверенность: {confidence:.1f}%")
            
            # Установка цвета в зависимости от эмоции
            color = Styles.EMOTION_COLORS.get(predicted_emotion, '#000000')
            self.emotion_label.setStyleSheet(f"""
                QLabel {{
                    padding: 14px;
                    border-radius: 8px;
                    background-color: {Styles.SECONDARY_COLOR};
                    color: {color};
                    border: 2px solid {color};
                }}
            """)
            
            # Создание подробного сообщения с результатами
            details = "\n".join([f"{emotion}: {prob:.1f}%" for emotion, prob in all_probs.items()])
            
            self.status_bar.showMessage(f"Анализ завершен: {predicted_emotion} ({confidence:.1f}%)")
            
            # Показать подробные результаты в окне сообщения
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Результаты анализа")
            msg_box.setText(f"<b>Предсказанная эмоция: {predicted_emotion.upper()}</b><br>"
                          f"Уверенность: {confidence:.1f}%<br><br>"
                          f"<b>Все вероятности:</b><br>{details}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            
            # Стилизация окна сообщения для темной темы
            msg_box.setStyleSheet(Styles.get_message_box_style())
            
            msg_box.exec_()
            
        except Exception as e:
            import traceback
            error_msg = f"Не удалось проанализировать аудио: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Ошибка", f"Не удалось проанализировать аудио: {str(e)}")
            self.status_bar.showMessage("Анализ не удался")
            self.emotion_label.setText("Ошибка")
            self.emotion_label.setStyleSheet(f"""
                QLabel {{
                    padding: 14px;
                    border-radius: 8px;
                    background-color: {Styles.SECONDARY_COLOR};
                    color: {Styles.ERROR_COLOR};
                    border: 2px solid {Styles.ERROR_COLOR};
                }}
            """)
            
        finally:
            self.progress_bar.setVisible(False)
            self.analyze_btn.setEnabled(True)
    
    def start_realtime_analysis(self):
        """Начало анализа эмоций в реальном времени с микрофона"""
        if not self.model or not self.audio_processor:
            QMessageBox.warning(self, "Предупреждение", "Модель еще не загружена")
            return
        
        # Получение выбранного устройства
        device_index = self.device_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "Предупреждение", "Устройство микрофона не выбрано")
            return
        
        # Получение длины батча из слайдера
        batch_length = self.batch_length_slider.value()
        
        # Сброс графика перед началом новой записи
        self.canvas.clear_plot()
        
        # Начало обработки аудио
        try:
            self.audio_processor.start_processing(device_index, batch_length)
            
            # Обновление UI
            self.start_realtime_btn.setEnabled(False)
            self.stop_realtime_btn.setEnabled(True)
            self.save_audio_btn.setEnabled(True)
            self.realtime_emotion_label.setText("Слушаю...")
            self.realtime_confidence_label.setText("Уверенность: --")
            self.realtime_status_label.setText(f"Запись с батчами {batch_length}с...")
            self.speech_status_label.setText("Распознавание речи активно")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось начать запись: {str(e)}")
    
    def stop_realtime_analysis(self):
        """Остановка анализа в реальном времени"""
        if self.audio_processor:
            self.audio_processor.stop_processing()
        
        # Обновление UI
        self.start_realtime_btn.setEnabled(True)
        self.stop_realtime_btn.setEnabled(False)
        self.realtime_emotion_label.setText("Остановлено")
        self.realtime_confidence_label.setText("Уверенность: --")
        self.realtime_status_label.setText("Анализ в реальном времени остановлен")
        self.speech_status_label.setText("Модель Vosk ожидает аудио")
    
    @pyqtSlot(dict, int)
    def update_realtime_display(self, emotion_probs, plot_counter):
        """Обновление отображения в реальном времени новыми данными об эмоциях"""
        try:
            # Фильтруем только существующие эмоции (убираем "другую" если она есть)
            filtered_emotions = {k: v for k, v in emotion_probs.items() if k in self.num2emotion.values()}
            
            if filtered_emotions:
                predicted_emotion = max(filtered_emotions, key=filtered_emotions.get)
                confidence = filtered_emotions[predicted_emotion]
                
                # Обновление метки эмоции
                self.realtime_emotion_label.setText(predicted_emotion.upper())
                self.realtime_confidence_label.setText(f"Уверенность: {confidence:.1f}%")
                
                # Установка цвета в зависимости от эмоции
                color = Styles.EMOTION_COLORS.get(predicted_emotion, '#000000')
                self.realtime_emotion_label.setStyleSheet(f"""
                    QLabel {{
                        padding: 14px;
                        border-radius: 8px;
                        background-color: {Styles.SECONDARY_COLOR};
                        color: {color};
                        border: 2px solid {color};
                    }}
                """)
                
                # Обновление графика только с 4 эмоциями
                self.canvas.update_plot(plot_counter, filtered_emotions)
                
                # Обновление состояния
                self.realtime_status_label.setText(f"Батч {plot_counter}: {predicted_emotion} ({confidence:.1f}%)")
                
        except Exception as e:
            print(f"Ошибка обновления отображения: {e}")
    
    def save_full_realtime_audio(self):
        """Сохранение полной записи от начала до конца"""
        if not self.audio_processor:
            QMessageBox.warning(self, "Предупреждение", "Аудио процессор не инициализирован")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("WAV файлы (*.wav)")
        file_dialog.setDefaultSuffix("wav")
        file_dialog.setStyleSheet(Styles.get_file_dialog_style())
        
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                filename = files[0]
                try:
                    # Создаем пустой файл для демонстрации
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b'')
                    
                    QMessageBox.information(self, "Успех", 
                        f"Заглушка аудио создана в {filename}\n"
                        f"В реальной реализации здесь сохранялась бы запись")
                    self.status_bar.showMessage(f"Аудио сохранено в {filename}")
                except Exception as e:
                    QMessageBox.warning(self, "Предупреждение", f"Не удалось сохранить аудио: {str(e)}")
    
    def on_text_recognized(self, text, emotion_info):
        """Обработка распознанного текста (обновленная версия)"""
        if text:
            # Получаем цвет для текущей эмоции
            emotion = emotion_info.get('emotion', 'нейтральная')
            
            # Проверяем, что эмоция есть в нашем списке
            if emotion not in self.num2emotion.values():
                emotion = 'нейтральная'  # Заменяем на нейтральную если эмоция "другая"
            
            color = Styles.EMOTION_COLORS.get(emotion, '#808080')
            
            # Добавляем текст с цветовым форматированием
            cursor = self.text_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            # Создаем формат для текста
            text_format = QTextCharFormat()
            text_format.setForeground(QColor(color))
            text_format.setFontWeight(QFont.Bold)
            
            # Вставляем текст с форматом
            cursor.insertText(text + " ", text_format)
            
            # Прокручиваем к концу
            self.text_display.ensureCursorVisible()
            
            # Обновляем статистику для ИИ советника
            self.update_conversation_stats(text, emotion)
    
    def clear_recognized_text(self):
        """Очистка распознанного текста"""
        self.text_display.clear()
        self.speech_status_label.setText("Текст очищен")
    
    def save_recognized_text(self):
        """Сохранение распознанного текста в файл"""
        text = self.text_display.toPlainText()
        if not text:
            QMessageBox.warning(self, "Предупреждение", "Нет текста для сохранения")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Текстовые файлы (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        file_dialog.setStyleSheet(Styles.get_file_dialog_style())
        
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            if files:
                filename = files[0]
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    QMessageBox.information(self, "Успех", f"Текст сохранен в {filename}")
                    self.status_bar.showMessage(f"Текст сохранен в {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить текст: {str(e)}")
    
    def set_conversation_goal(self):
        """Установка цели разговора"""
        goal = self.goal_text_edit.toPlainText().strip()
        if goal:
            self.ai_goal = goal
            self.ai_status_label.setText(f"Цель установлена: {goal[:50]}...")
            QMessageBox.information(self, "Успех", "Цель разговора успешно установлена!")
        else:
            QMessageBox.warning(self, "Предупреждение", "Введите цель разговора")
    
    def clear_conversation_goal(self):
        """Очистка цели разговора"""
        self.ai_goal = ""
        self.goal_text_edit.clear()
        self.ai_status_label.setText("Цель разговора очищена")
    
    def update_conversation_stats(self, text, emotion):
        """Обновление статистики разговора при получении нового текста"""
        if text:
            # Подсчет слов
            words = text.split()
            self.word_counter += len(words)
            self.words_counter_label.setText(str(self.word_counter))
            
            # Обновление счетчика эмоций (только для существующих эмоций)
            if emotion in self.num2emotion.values():
                if emotion not in self.emotion_counter:
                    self.emotion_counter[emotion] = 0
                self.emotion_counter[emotion] += 1
                
                # Определение доминирующей эмоции
                if self.emotion_counter:
                    self.dominant_emotion = max(self.emotion_counter.items(), key=lambda x: x[1])[0]
                    color = Styles.EMOTION_COLORS.get(self.dominant_emotion, '#808080')
                    self.dominant_emotion_label.setText(self.dominant_emotion)
                    self.dominant_emotion_label.setStyleSheet(f"""
                        QLabel {{
                            color: {color};
                            font-size: 20px;
                            font-weight: bold;
                            padding: 5px;
                            border-radius: 5px;
                            background-color: {Styles.SECONDARY_COLOR};
                            border: 2px solid {color};
                        }}
                    """)
            
            # Добавление в историю разговора с эмоцией
            self.conversation_history.append({
                'text': text,
                'emotion': emotion,
                'timestamp': time.time()
            })
            
            # Автоматический запрос совета при накоплении достаточного количества слов
            if self.word_counter >= self.words_for_ai and self.ai_goal:
                self.get_ai_advice_async()
    
    def get_ai_advice_async(self):
        """Асинхронный запрос совета от ИИ"""
        if not self.ai_goal:
            QMessageBox.warning(self, "Предупреждение", "Сначала установите цель разговора")
            return
        
        if not self.conversation_history:
            QMessageBox.warning(self, "Предупреждение", "Нет данных разговора для анализа")
            return
        
        # Проверяем, не работает ли уже поток ИИ
        if self.ai_thread and self.ai_thread.is_alive():
            QMessageBox.warning(self, "Предупреждение", "ИИ уже анализирует разговор. Пожалуйста, подождите.")
            return
        
        # Обновляем UI перед началом работы в потоке
        self.ai_status_label.setText("ИИ анализирует разговор...")
        self.get_advice_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Создаем и запускаем поток для работы с ИИ
        self.ai_thread = threading.Thread(target=self._get_ai_advice_thread)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        # Запускаем таймер для проверки состояния потока
        self.ai_check_timer = QTimer()
        self.ai_check_timer.timeout.connect(self._check_ai_thread_status)
        self.ai_check_timer.start(100)  # Проверяем каждые 100 мс
    
    def _get_ai_advice_thread(self):
        """Функция для выполнения в отдельном потоке"""
        try:
            # Подготовка текста разговора для ИИ
            conversation_text = self.prepare_conversation_for_ai()
            
            # Вызов локального ИИ через Ollama
            response = ollama.chat(
                model='deepseek-llm:7b',
                messages=[
                    {
                        'role': 'system',
                        'content': 'Ты - эксперт по коммуникациям и психологии. Анализируй разговор и давай конкретные советы очень кратко и лаконично. Отвечай только на русском языке.'
                    },
                    {
                        'role': 'user',
                        'content': f"""
                        Цель разговора: {self.ai_goal}
                        
                        Контекст разговора (последние {self.words_for_ai} слов):
                        {conversation_text}
                        
                        Доминирующая эмоция собеседника: {self.dominant_emotion}
                        
                        Проанализируй разговор и дай 3-5 конкретных советов:
                        1. Что делать дальше для достижения цели?
                        2. Как реагировать на текущие эмоции собеседника?
                        3. Какие вопросы задать?
                        4. Чего избегать в разговоре?
                        5. Как улучшить коммуникацию?
                        
                        Ответ дай на русском языке, структурированно и конкретно. Будь краток!
                        """
                    }
                ]
            )
            
            advice = response['message']['content']
            
            # Добавляем совет в очередь для обработки в основном потоке
            self.ai_advice_queue.append(advice)
            
        except Exception as e:
            print(f"Ошибка получения совета от ИИ: {e}")
            # Создаем демонстрационный совет при ошибке
            demo_advice = (
                "Демонстрационный совет от ИИ:\n\n"
                "1. Собеседник проявляет интерес к теме - задайте уточняющие вопросы.\n"
                "2. Поддерживайте позитивный настрой разговора.\n"
                "3. Используйте открытые вопросы для получения больше информации.\n"
                "4. Проявляйте эмпатию и понимание.\n"
                "5. Следите за эмоциональным состоянием собеседника."
            )
            self.ai_advice_queue.append(demo_advice)
    
    def _check_ai_thread_status(self):
        """Проверка состояния потока ИИ и обновление UI"""
        if not self.ai_thread.is_alive():
            # Останавливаем таймер
            self.ai_check_timer.stop()
            
            # Восстанавливаем кнопку
            self.get_advice_btn.setEnabled(True)
            
            # Обрабатываем результат из очереди
            if self.ai_advice_queue:
                advice = self.ai_advice_queue.pop(0)
                
                # Отображаем совет в основном потоке
                self.ai_advice_text.append(f"📅 {time.strftime('%H:%M:%S')}\n")
                self.ai_advice_text.append("="*50 + "\n")
                self.ai_advice_text.append(advice + "\n\n")
                
                self.ai_status_label.setText("ИИ совет получен!")
                
                # Прокручиваем к концу
                cursor = self.ai_advice_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.ai_advice_text.setTextCursor(cursor)
            else:
                self.ai_status_label.setText("Ошибка получения совета")
    
    def prepare_conversation_for_ai(self):
        """Подготовка текста разговора для отправки в ИИ"""
        # Берем последние N слов из истории разговора
        recent_history = []
        total_words = 0
        
        for entry in reversed(self.conversation_history):
            words = entry['text'].split()
            if total_words + len(words) <= self.words_for_ai:
                recent_history.insert(0, entry)
                total_words += len(words)
            else:
                break
        
        # Форматирование разговора
        formatted_conversation = []
        for entry in recent_history:
            emotion = entry['emotion']
            color_code = {
                'радость': '😊',
                'грусть': '😢',
                'гнев': '😠',
                'нейтральная': '😐'
            }.get(emotion, '😐')
            
            formatted_conversation.append(f"[{color_code} {emotion.upper()}] {entry['text']}")
        
        return "\n".join(formatted_conversation)
    
    def clear_ai_advice(self):
        """Очистка советов от ИИ"""
        self.ai_advice_text.clear()
        self.ai_status_label.setText("Советы очищены")
    
    def closeEvent(self, event):
        """Очистка при закрытии"""
        # Остановка анализа в реальном времени, если запущен
        if self.audio_processor:
            self.audio_processor.stop_processing()
        
        # Остановка таймера ИИ, если он запущен
        if hasattr(self, 'ai_check_timer'):
            self.ai_check_timer.stop()
        
        # Ожидание завершения потока ИИ
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1)
        
        # Очистка ресурсов
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("СинхронИИя - Распознавание эмоций и речи")
    
    # Установка темного стиля приложения
    app.setStyle('Fusion')
    
    # Создание темной палитры
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Text, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Button, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Link, QColor(52, 152, 219))
    dark_palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(dark_palette)
    
    window = EmotionRecognitionApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
