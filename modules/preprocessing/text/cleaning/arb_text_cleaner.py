import re


class ArbTextCleaner(object):
    def __init__(self, is_punct=False) -> None:
        super().__init__()
        self._whitespace_re = re.compile(r'\s+')
        self._html_re = re.compile(r'<[^>]*>')
        self._chars = set()
        self._is_punct = is_punct
        self._pre_replace_map = {
            "آ": "آ", "أ": "أ", "إ": "إ", "ؤ": "ؤ", "ىٔ": "ئ",
            "ﺀ": "ء", "ﺁ": "آ", "ﺂ": "آ", "ﺃ": "أ", "ﺄ": "أ", "ﺅ": "ؤ", "ﺆ": "ؤ", "ﺇ": "إ", "ﺈ": "إ", "ﺉ": "ئ",
            "ﺊ": "ئ", "ﺋ": "ئ", "ﺌ": "ئ", "ﺍ": "ا", "ﺎ": "ا", "ﺏ": "ب", "ﺐ": "ب", "ﺑ": "ب", "ﺒ": "ب", "ﺓ": "ة",
            "ﺔ": "ة", "ﺕ": "ت", "ﺖ": "ت", "ﺗ": "ت", "ﺘ": "ت", "ﺙ": "ث", "ﺚ": "ث", "ﺛ": "ث", "ﺜ": "ث", "ﺝ": "ج",
            "ﺞ": "ج", "ﺟ": "ج", "ﺠ": "ج", "ﺡ": "ح", "ﺢ": "ح", "ﺣ": "ح", "ﺤ": "ح", "ﺥ": "خ", "ﺦ": "خ", "ﺧ": "خ",
            "ﺨ": "خ", "ﺩ": "د", "ﺪ": "د", "ﺫ": "ذ", "ﺬ": "ذ", "ﺭ": "ر", "ﺮ": "ر", "ﺯ": "ز", "ﺰ": "ز", "ﺱ": "س",
            "ﺲ": "س", "ﺳ": "س", "ﺴ": "س", "ﺵ": "ش", "ﺶ": "ش", "ﺷ": "ش", "ﺸ": "ش", "ﺹ": "ص", "ﺺ": "ص", "ﺻ": "ص",
            "ﺼ": "ص", "ﺽ": "ض", "ﺾ": "ض", "ﺿ": "ض", "ﻀ": "ض", "ﻁ": "ط", "ﻂ": "ط", "ﻃ": "ط", "ﻄ": "ط", "ﻅ": "ظ",
            "ﻆ": "ظ", "ﻇ": "ظ", "ﻈ": "ظ", "ﻉ": "ع", "ﻊ": "ع", "ﻋ": "ع", "ﻌ": "ع", "ﻍ": "غ", "ﻎ": "غ", "ﻏ": "غ",
            "ﻐ": "غ", "ﻑ": "ف", "ﻒ": "ف", "ﻓ": "ف", "ﻔ": "ف", "ﻕ": "ق", "ﻖ": "ق", "ﻗ": "ق", "ﻘ": "ق", "ﻙ": "ك",
            "ﻚ": "ك", "ﻛ": "ك", "ﻜ": "ك", "ﻝ": "ل", "ﻞ": "ل", "ﻟ": "ل", "ﻠ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻣ": "م",
            "ﻤ": "م", "ﻥ": "ن", "ﻦ": "ن", "ﻧ": "ن", "ﻨ": "ن", "ﻩ": "ه", "ﻪ": "ه", "ﻫ": "ه", "ﻬ": "ه", "ﻭ": "و",
            "ﻮ": "و", "ﻯ": "ى", "ﻰ": "ى", "ﻱ": "ي", "ﻲ": "ي", "ﻳ": "ي", "ﻴ": "ي", "ﻵ": "لآ", "ﻷ": "لأ", "ﻸ": "لأ",
            "ﻹ": "لإ", "ﻺ": "لإ", "ﻻ": "لا", "ﻼ": "لا",
            "ﭐ": "ا", "ﭫ": "ف", "ﭭ": "ف", "ﭼ": "ج", "ﮐ": "ك", "ﮫ": "ه", "ﮭ": "ه", "ﯽ": "ى", "ﯾ": "ي",
            "٪": "%",
            "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۷": "7", "۸": "8", "۹": "9",
            "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
            "  ": " "
        }

        self._pre_replace_punct_map = {"?": "؟", "۔": ".", ",": "،"}
        if self._is_punct:
            self._pre_replace_map.update(self._pre_replace_punct_map)

        self._allowed_chars = {" ", "$", "%", "&", "/", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "@", "a",
                               "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                               "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "ء", "آ", "أ", "ؤ", "إ",
                               "ئ", "ا", "ب", "ة", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط",
                               "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ى", "ي", "ً", "ٌ", "ٍ", "ّ", "ﷲ",
                               "ﷺ"}

        self._allowed_punct = {"؟", ".", "،"}
        if self._is_punct:
            self._allowed_chars.update(self._allowed_punct)

        self._post_replace_map = {
            "صلى الله عليه وسلم": "ﷺ",
            "الله": "ﷲ",
            "صلى الله عليه وسلّم": "ﷺ",
            "اللّه": "ﷲ",
            "اللّٰهُ": "ﷲ",
            "اللّٰه": "ﷲ",
            "  ": " "
        }

    def capture_text(self, text: str):
        for ch in text:
            self._chars.add(ch)

    def get_chars(self):
        return sorted(list(self._chars))

    def clean_text(self, text: str):
        text = text.strip()
        text = re.sub(self._whitespace_re, ' ', text)
        text = re.sub(self._html_re, '', text)

        for i in range(5):
            for key, val in self._pre_replace_map.items():
                text = text.replace(key, val)

        chars = []
        for ch in text:
            if ch in self._allowed_chars:
                chars.append(ch)

        text = "".join(chars)

        if self._is_punct:
            has_end_punct = False
            for p in self._allowed_punct:
                for i in range(5):
                    text = text.replace(p + p, p + " ")
                    text = text.replace(" " + p, p + " ")
                    text = text.replace(p, p + " ")
                    text = text.replace("  ", " ")

                text = text.strip()
                if p in text and text.endswith(p):
                    has_end_punct = True

            if not has_end_punct:
                text = text + "."

        for i in range(5):
            for key, val in self._post_replace_map.items():
                text = text.replace(key, val)
            text = text.strip()

        for i in range(15):
            for pn in ["\n", "\t", "\r", "-", "/", "$$", "&", "@", "%%"]:
                text = text.strip(pn)
                text = text.strip()

        return text
