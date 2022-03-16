""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text.text_en import cmudict

_punctuation = ["!", "'", ",", ".", ":", ";", "?", "،ۖ"]
_math = ['#', '%', '&', '*', '+', '-', '/', '[', ']', '(', ')']
_special = ['', '_', '@', '©°', '½', '—', '₩', '€', '$']
_accented = ['']
_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']
_letters = [' ', '!', '$', '&', "'", '*', ',', '-', '.', ':', ';', '<', '>', '?', 'A', 'D', 'E', 'F', 'H', 'J', 'K', 'N', 'S', 'T', 'Y', 'Z', '_', '`', 'a', 'b', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '}', '~', '،', '؛', '؟', 'ک', 'ی', 'ۖ', 'ۗ', 'ۘ', 'ۚ', 'ﺃ']

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet
