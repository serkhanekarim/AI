""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
#from text.text_ar import cmudict

#_punctuation = '!\'",.:;? '
#_math = '#%&*+-/[]()'
#_special = '_@©°½—₩€$'
#_accented = 'áçéêëñöøćž'
#_numbers = '0123456789'
#_letters = 'إYۛءqHيدةؤDقEآن؟ِKخtُۘڨVkIeٌۚۗwuWQ☭ّه»،ىغlMز”؛اـjPaئxOحذicضۖpﺃgdsbGF“رثیمھبhvS«mﻻ…RLCrلJَفصجfXnسoٍْظچyAطکعكTتشًNوأUٰB'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
#symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet

symbols = [' ', '!', ',', '-', '.', '/', '?', 'a', 'aː', 'b', 'd', 'dˤ', 'd͡ʒ', 'e', 'f', 'h', 'i', 'iː', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 'sˤ', 't', 'tˤ', 't͡ʃ', 'u', 'uː', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ðˤ', 'ħ', 'ŋ', 'ɑ', 'ɔ', 'ə',
        'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɹ', 'ɹ̩', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʕ', 'θ', '،', '؟', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة',
        'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ـ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن',
        'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٓ', 'ٔ', 'ٕ']
