""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from modules.preprocessing.text_normalizer import cmudict

_punctuation = '!\'",.:;? '
_math = '#%&*+-/[]()'
_special = '_@©°½—₩€$'
_accented = 'áçéêëñöøćž'
_numbers = '0123456789'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet
