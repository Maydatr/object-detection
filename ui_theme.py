"""Shared dark theme styles for PyQt6 inspection apps."""

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a1a;
    color: #ddd;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #333;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #aaa;
}
QPushButton {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 6px 12px;
    color: #ddd;
}
QPushButton:hover {
    background-color: #333;
    border-color: #555;
}
QPushButton:pressed {
    background-color: #222;
}
QPushButton:disabled {
    background-color: #1e1e1e;
    color: #666;
    border-color: #2a2a2a;
}
QPushButton#primaryButton {
    background-color: #2F6DF6;
    border-color: #2F6DF6;
    color: #fff;
    font-weight: bold;
}
QPushButton#primaryButton:hover {
    background-color: #4a82f7;
}
QPushButton#primaryButton:disabled {
    background-color: #1a3055;
    color: #666;
    border-color: #1a3055;
}
QPushButton#dangerButton {
    background-color: #3a2020;
    border-color: #663333;
    color: #e08080;
}
QListWidget {
    background-color: #141414;
    border: 1px solid #333;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 4px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #2F6DF6;
    color: #fff;
}
QListWidget::item:hover {
    background-color: #252525;
}
QScrollArea {
    background: transparent;
    border: none;
}
QPlainTextEdit {
    background-color: #0d0d0d;
    border: 1px solid #333;
    border-radius: 6px;
    color: #aaa;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #333;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: #2F6DF6;
    border-radius: 7px;
}
QProgressBar {
    background-color: #141414;
    border: 1px solid #333;
    border-radius: 4px;
    text-align: center;
    color: #aaa;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2F6DF6;
    border-radius: 3px;
}
QStatusBar {
    background-color: #141414;
    color: #888;
    border-top: 1px solid #333;
}
QLabel#mutedLabel {
    color: #888;
    font-size: 11px;
}
QLabel#statValue {
    font-size: 28px;
    font-weight: bold;
}
QLabel#pageTitle {
    font-size: 22px;
    font-weight: bold;
}
QLabel#pageSubtitle {
    font-size: 12px;
    color: #888;
}
QLabel#badgeLabel {
    font-weight: bold;
    font-size: 11px;
}
"""

PREVIEW_STYLE = "background-color: #0d0d0d; border-radius: 8px; color: #888;"
SCHEMA_BG_STYLE = "background-color: #111; border-radius: 6px;"
