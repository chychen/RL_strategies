from kivy.app import App
from kivy.core.window import Window


class WindowFileDropExampleApp(App):
    def build(self):
        Window.bind(on_dropfile=self._on_file_drop)
        return

    def _on_file_drop(self, window, file_path):
        print(file_path)
        return

if __name__ == '__main__':
    WindowFileDropExampleApp().run()