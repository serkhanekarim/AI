# import json
# import logging
import nltk
import re

# nltk.download()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("MLT")

# src_path = "../../dwl/manifest_train.json"
# tgt_path = "../../dwl/manifest_train_lm.txt"


def process_line(line, min_words=4, language='english'):
    # sentences = []
    # sents = nltk.sent_tokenize(line.strip(), language=language)
    # for sentence in sents:
    #     sentence_processed = process_sentence(sentence, min_words)
    #     if sentence_processed:
    #         sentences.append(sentence_processed)

    # return sentences

    return [process_sentence(sentence, min_words) for sentence in nltk.sent_tokenize(line.strip(), language=language)]

def process_sentence(sent, min_words=4):
    return ' '.join([normalize_word(word) for word in nltk.word_tokenize(sent, language='english')])
    # if len(words) >= min_words:
    #     return ' '.join(w for w in words if w).strip()  # prevent multiple spaces
    # return ''


def replace_numeric(text, numeric_pattern=re.compile('[0-9]+'), digit_pattern=re.compile('[0-9]'), repl='#',
                    by_single_digit=False):
    return re.sub(numeric_pattern, repl, text) if by_single_digit else re.sub(digit_pattern, repl, text)


def contains_numeric(text):
    return any(char.isdigit() for char in text)


def normalize_word(token):
    #_token = token
    # _token = unidecode_keep_umlauts(token)
    # _token = remove_punctuation(_token)  # remove any special chars
    token = replace_numeric(token, by_single_digit=True)
    token = '<num>' if token == '#' else token  # if token was a number, replace it with <num> token
    return token.strip().lower()


# with open(src_path, encoding="utf-8") as src_json:
#     with open(tgt_path, encoding="utf-8", mode="w") as tgt_text:
#         for jn in src_json:
#             try:
#                 jd = json.loads(jn)
#                 audio_filepath = jd["audio_filepath"]
#                 txt = str(jd["text"]).strip("\n").strip()
#                 duration = float(jd["duration"])
#                 proc_texts = process_line(txt)
#                 for proc_text in proc_texts:
#                     logger.info(proc_text)
#                     tgt_text.write("{}\n".format(proc_text))
#             except Exception as e:
#                 logger.exception(e)
