import re
from .cmudict import CMUDict

_letter_to_arpabet = {

    'A': 'a',
    'B': 'b e',
    'C': 's e',
    'D': 'd e',
    'E': 'ə',
    'F': 'ɛ f',
    'G': 'ʒ e',
    'H': 'a ʃ',
    'I': 'i',
    'J': 'ʒ i',
    'K': 'k a',
    'L': 'ɛ l',
    'M': 'ɛ m',
    'N': 'ɛ n',
    'O': 'o',
    'P': 'p e',
    'Q': 'k y',
    'R': 'ɛ ʁ',
    'S': 'ɛ s',
    'T': 't e',
    'U': 'y',
    'V': 'v e',
    'W': 'd u b l ə v e',
    'X': 'i k s',
    'Y': 'i ɡ ʁ ɛ k',
    'Z': 'z ɛ d'
}

# must ignore roman numerals
_acronym_re = re.compile(r'([A-Z][A-Z]+)s?|([A-Z]\.([A-Z]\.)+s?)')
cmudict = CMUDict('data/ipa_dictionary_ar-AR', keep_ambiguous=False)


def _expand_acronyms(m, add_spaces=True):
    acronym = m.group(0)

    # remove dots if they exist
    acronym = re.sub('\.', '', acronym)

    acronym = "".join(acronym.split())
    arpabet = cmudict.lookup(acronym)

    if arpabet is None:
        acronym = list(acronym)
        arpabet = ["{" + _letter_to_arpabet[letter] + "}" for letter in acronym]
        # temporary fix
        if arpabet[-1] == '{Z}' and len(arpabet) > 1:
            arpabet[-2] = arpabet[-2][:-1] + ' ' + arpabet[-1][1:]
            del arpabet[-1]

        arpabet = ' '.join(arpabet)
    else:
        arpabet = "{" + arpabet[0] + "}"

    return arpabet


def normalize_acronyms(text):
    text = re.sub(_acronym_re, _expand_acronyms, text)
    return text
