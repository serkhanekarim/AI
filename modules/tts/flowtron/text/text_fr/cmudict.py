""" from https://github.com/keithito/tacotron """

import re


valid_symbols = ['a',
 'a-',
 'b',
 'd',
 'dʒ',
 'e',
 'e-',
 'f',
 'h',
 'i',
 'j',
 'k',
 'kʰcFʀ',
 'l',
 'm',
 'n',
 'o',
 'p',
 'pʰcFʀ',
 's',
 't',
 'tʃ',
 'u',
 'v',
 'w',
 'x',
 'y',
 'y-',
 'z',
 'ð',
 'ø',
 'ŋ',
 'œ',
 'œ̃',
 'ɐ',
 'ɑ̃',
 'ɔ',
 'ɔ̃',
 'ə',
 'ə-',
 'əʊ',
 'ɛ',
 'ɛ̃',
 'ɡ',
 'ɡʲ',
 'ɪ',
 'ɲ',
 'ʁ',
 'ʃ',
 'ʌ',
 'ʒ',
 'θ']

_valid_symbol_set = set(valid_symbols)


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path, keep_ambiguous=True):
    if isinstance(file_or_path, str):
      with open(file_or_path, encoding='utf-8') as f:
        entries = _parse_cmudict(f)
    else:
      entries = _parse_cmudict(file_or_path)
    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries
    #print(self._entries)
    #exit()

  def __len__(self):
    return len(self._entries)


  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    return self._entries.get(word.upper())



_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  #for part in parts:
    #if part not in _valid_symbol_set:
    #  return None
  return ' '.join(parts)
