import re

_domain = '(com|gov|org|ae|fr)'

_url_re = re.compile(r'\.{}\b'.format(_domain))
_mail_re = re.compile(r'[\w.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

def _expand_url(m):
    return m.group(0).replace("."," dot ")

def _expand_mail(m):
    mail = m.group(0)
    mail = mail.replace("."," dot ")
    mail = mail.replace("_"," underscore ")
    mail = mail.replace("-"," hyphen ")
    mail = mail.replace("@"," at ")
    return mail

def normalize_url(text):
    text = re.sub(_url_re, _expand_url, text)
    text = re.sub(_mail_re, _expand_mail, text)
    return text
    
