import re


class LimpiadorTexto:
    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "\uFE0F"
            "]+", flags=re.UNICODE)
        
        self.emoticonos_pattern = re.compile(
            r'(:-\)|:\)|:-D|:D|:-\(|:\(|;-?\)|:\'||:-P|:P|:-O|:O|:-/|:/|:-\\|:\\|:-\||:\||<3|:\*)'
        )
    
    def eliminar_emojis(self, texto):
        return self.emoji_pattern.sub(r'', texto)

    def eliminar_emoticonos(self, texto):
        return self.emoticonos_pattern.sub('', texto)

    def limpiar_texto(self, texto):
        texto = str(texto)
        texto = re.sub(r'http[s]?://\S+', '', texto)
        texto = re.sub(r'@\w+', '[AIRLINE]', texto)
        texto = self.eliminar_emojis(texto)
        texto = self.eliminar_emoticonos(texto)
        texto = re.sub(r'#(\w+)', r'\1', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()