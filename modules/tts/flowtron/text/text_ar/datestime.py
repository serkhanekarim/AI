import re
_ordinal_re = re.compile(r'([0-9]|0[0-9]|1[0-9]|2[0-3])([Hh\:])([0-5][0-9])?')

def normalize_datestime(text):
    return re.sub(_ordinal_re,r"\1 heure \3",text)
