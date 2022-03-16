""" from https://github.com/keithito/tacotron """

from num2words import num2words
import re

_large_numbers = '(trillion|billion|million|mille|cent)'

_measurements = '(f|c|k)'
_measurements_key = {'f': 'fahrenheit', 'c': 'celsius', 'k': 'mille'}

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')

_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+[ ]?{}?)'.format(_large_numbers), re.IGNORECASE)

_measurement_re = re.compile(r'([0-9\.\,]*[0-9]+(\s)?{}\b)'.format(_measurements), re.IGNORECASE)

_ordinal_re = re.compile(r'(\d+)(i?[éèe]mes?|i?ere?s?|st|nde?s?|th|es?)')
_number_re = re.compile(r"(\d+)")

_percent_re = re.compile(r'\d(\s?%)')

def _remove_commas(m):
  return m.group(1).replace(',', ' virgule ')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)

    # check for million, billion, etc...
    parts = match.split(' ')
    if len(parts) == 2 and len(parts[1]) > 0 and parts[1] in _large_numbers:
        return "{} {} {} ".format(parts[0], parts[1], 'dollars')

    parts = parts[0].split('.')
    if len(parts) > 2:
        return match + " dollars"    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'centime' if cents == 1 else 'centimes'
        return "{} {}, {} {} ".format(
            num2words(str(dollars),lang="ar"), dollar_unit,
            num2words(str(cents),lang="ar"), cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return "{} {} ".format(num2words(str(dollars),lang="ar"), dollar_unit)
    elif cents:
        cent_unit = 'centime' if cents == 1 else 'centimes'
        return "{} {} ".format(num2words(str(cents),lang="ar"), cent_unit)
    else:
        return 'zéro dollars'


def _expand_ordinal(m):
    return num2words(m.group(0),lang="ar", to="ordinal")


def _expand_measurement(m):
    _, number, measurement = re.split('(\d+(?:\.\d+)?)', m.group(0))
    number = num2words(number,lang="ar")
    measurement = "".join(measurement.split())
    measurement = _measurements_key[measurement.lower()]
    return "{} {}".format(number, measurement)

def _expand_percents(m):   
    if re.match(r'\s',m.group(1)[0]) is None:
        #if there is a space between the digit and the percent symbol
        return m.group(0).replace("%","pour cent")
    else:
        #Else add a space
        return m.group(0).replace("%"," pour cent")


def _expand_number(m):
    return num2words(m.group(0),lang="ar")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_percent_re, _expand_percents, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
