import re
from .cmudict import CMUDict

_letter_to_arpabet = {
    'A': 'AA',
    'B': 'B EY',
    'C': 'S EY',
    'D': 'D EY',
    'E': 'EU',
    'F': 'EH F',
    'G': 'ZH EY',
    'H': 'AA SH',
    'I': 'IY',
    'J': 'ZH IY',
    'K': 'K AA',
    'L': 'EH L',
    'M': 'EH M',
    'N': 'EH N',
    'O': 'OW',
    'P': 'P EY',
    'Q': 'K UH',
    'R': 'EH R',
    'S': 'EH S',
    'T': 'T EY',
    'U': 'UH',
    'V': 'V EY',
    'X': 'IY K S',
    'Y': 'IY G R EH K',
    'W': 'D UW B L EU V EY',
    'Z': 'Z EH D',
    's': 'S'
}

# must ignore roman numerals
_acronym_re = re.compile(r'([A-Z][A-Z]+)s?|([A-Z]\.([A-Z]\.)+s?)')
cmudict = CMUDict('data/cmudict_dictionary', keep_ambiguous=False)


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
