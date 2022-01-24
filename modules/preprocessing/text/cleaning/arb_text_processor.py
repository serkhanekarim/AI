from camel_tools.utils.stringutils import force_encoding, force_unicode
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_unicode, normalize_alef_ar, \
    normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
import six
from .arb_text_cleaner import ArbTextCleaner


class ArbTextProcessor(object):

    def __init__(self) -> None:
        super().__init__()
        self.sec_cleaner = None
        self.clean_mapper = CharMapper.builtin_mapper('arclean')
        self.ar2bw_mapper = CharMapper.builtin_mapper('ar2bw')
        self.bw2ar_mapper = CharMapper.builtin_mapper('bw2ar')

    def clean(self, ar_raw_text: str, normalize_teh_marbuta: bool = False, normalize_alef: bool = True,
              is_punct: bool = False, is_de_diacritize: bool = True, is_to_lower: bool = True) -> str:
        clean_text = ar_raw_text.strip("\n").strip()

        if is_to_lower:
            clean_text = clean_text.lower()

        clean_text = force_unicode(clean_text)
        if six.PY3:
            clean_text = self.clean_mapper.map_string(clean_text)
        else:
            clean_text = force_encoding(clean_text)
            clean_text = self.clean_mapper.map_string(clean_text)

        clean_text = self._get_sec_cleaner(is_punct=is_punct).clean_text(clean_text)

        clean_text = self._normalize(arb_text=clean_text, normalize_teh_marbuta=normalize_teh_marbuta,
                                     normalize_alef=normalize_alef)
        if is_de_diacritize:
            clean_text = self._de_diacritize(arb_text=clean_text)

        for i in range(5):
            clean_text = clean_text.replace("  ", " ")

        return clean_text.strip()

    def clean_and_convert2bw(self, ar_raw_text: str) -> str:
        clean_text = self.clean(ar_raw_text=ar_raw_text)
        bw_text = self.ar2bw_mapper.map_string(clean_text)
        return bw_text

    def convert2bw(self, ar_clean_text: str) -> str:
        bw_text = self.ar2bw_mapper.map_string(ar_clean_text)
        return bw_text

    def convert2ar(self, bw_text: str) -> str:
        ar_text = self.bw2ar_mapper.map_string(bw_text)
        return ar_text

    def _normalize(self, arb_text: str, normalize_teh_marbuta: bool, normalize_alef: bool) -> str:
        norm_text = normalize_unicode(arb_text)
        if normalize_alef:
            norm_text = normalize_alef_ar(norm_text)

        norm_text = normalize_alef_maksura_ar(norm_text)

        if normalize_teh_marbuta:
            norm_text = normalize_teh_marbuta_ar(norm_text)

        return norm_text

    def _de_diacritize(self, arb_text: str) -> str:
        de_diac_text = dediac_ar(arb_text)
        return de_diac_text

    def _get_sec_cleaner(self, is_punct: bool):
        if self.sec_cleaner is None:
            self.sec_cleaner = ArbTextCleaner(is_punct=is_punct)
        return self.sec_cleaner
