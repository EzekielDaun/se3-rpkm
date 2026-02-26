from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication, QLabel, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

GAMEPAD_WEB_PANEL_DIR: Path = Path(__file__).resolve().parent
GAMEPAD_ENTRY_HTML: Path = GAMEPAD_WEB_PANEL_DIR / "gamepad_svg_local.html"


class GamepadWebPanelController:
    def __init__(self) -> None:
        self.web_view = None
        self.style_hints = QApplication.styleHints()
        color_scheme_changed = getattr(self.style_hints, "colorSchemeChanged", None)
        if color_scheme_changed is not None:
            color_scheme_changed.connect(self._on_color_scheme_changed)  # type: ignore[attr-defined]

    def _theme_name(self) -> str:
        color_scheme_getter = getattr(self.style_hints, "colorScheme", None)
        if callable(color_scheme_getter):
            color_scheme = color_scheme_getter()
            if color_scheme == Qt.ColorScheme.Dark:
                return "dark"
            if color_scheme == Qt.ColorScheme.Light:
                return "light"

        window_color = QApplication.palette().window().color()
        if window_color.lightness() < 128:
            return "dark"
        return "light"

    def _sync_theme_to_web_view(self) -> None:
        if self.web_view is None:
            return

        theme_name = self._theme_name()
        self.web_view.page().runJavaScript(
            (
                "if (window.setGamepadTheme) { "
                f'window.setGamepadTheme("{theme_name}");'
                " }"
            ),
        )

    def _on_color_scheme_changed(self, _color_scheme: Qt.ColorScheme) -> None:
        self._sync_theme_to_web_view()

    def _on_page_load_finished(self, success: bool) -> None:
        if not success:
            return
        self._sync_theme_to_web_view()

    def build_widget(self, parent: QWidget) -> QWidget:
        if QWebEngineView is None:
            fallback = QLabel(
                "Qt WebEngine is unavailable in this environment.",
                parent,
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.web_view = None
            return fallback

        entry_html = GAMEPAD_ENTRY_HTML.resolve()
        if not entry_html.exists():
            fallback = QLabel(
                f"Local gamepad tester html not found:\n{entry_html}",
                parent,
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.web_view = None
            return fallback

        try:
            web_view = QWebEngineView(parent)
            self.web_view = web_view
            web_view.loadFinished.connect(self._on_page_load_finished)
            index_url = QUrl.fromLocalFile(str(entry_html))
            web_view.setUrl(index_url)
            return web_view
        except Exception:
            fallback = QLabel(
                "Failed to initialize local gamepad tester view.",
                parent,
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.web_view = None
            return fallback
