
import re
import string

ascii_letters = set(string.ascii_letters)
digits = set(string.digits)
whitespace = set(string.whitespace)

"""
Punctuation set contains the following:
    Brackets and Braces
        {} [] () <>

    Quotations
        " ' `

    Periods, Commas, Marks and Colons, etc.
        . , ! ? : ;
    
    Dashes, Slashes, Bars and Tilda
        - _ / \ | ~
    
    "Mathematical"
        = + - * /
    
    Symbols
        @ ^ # & $ %
"""
punctuation = set(string.punctuation)
ascii_chars = ascii_letters | digits | whitespace | punctuation
ascii_remove = {'{', '}', '[', ']', '<', '>', '~', '|', '\\',
                '=', '+', '*', '^', '$', '%', '\t'}
ascii_chars = ascii_chars - ascii_remove

def is_ascii(line):
    return all(c in ascii_chars for c in line)


def clean_line(line):
    line = expand_contractions(line)
    line = strip_punctuation(line, {'-', "'"})
    line = clean_whitespace(line)
    return line



def clean_whitespace(string):
    """
    Ensures only single spaces exist between words, removes extra whitespace 
    characters as well (such as \n and \t)
    """
    return clean_whitespace.regex.sub(' ', string).strip()
clean_whitespace.regex = re.compile(r'\s\s+')


def strip_punctuation(string, keep = set(), replacement = ''):
    if strip_punctuation.regex is None or strip_punctuation.keep != keep:
        strip_punctuation.keep = keep
        match_set = punctuation - keep
        pattern = r"[{:s}]".format(re.escape(''.join(match_set)))
        strip_punctuation.regex = re.compile(pattern)
    
    return strip_punctuation.regex.sub(replacement, string).strip()

strip_punctuation.regex = None
strip_punctuation.keep = None



contr_substring_replacements = {"'re" : " are",
                                "n't" : " not",
                                "'ve" : " have",
                                "'ll" : " will",
                                "in'" : "ing",
                                "'d" : " would"}

contr_is_replacements = {"it's", "that's", "he's", "let's", "what's", "there's", \
                         "she's", "who's", "where's", "here's", "how's", "nobody's", \
                         "everything's", "something's", "today's"}

contr_word_replacements = {"i'm" : "i am",
                           "'cause" : "because",
                           "'em" : "them",
                           "'scuse" : "excuse",
                           "starv" : "starve",
                           "'kay" : "okay",
                           "'k" : "okay",
                           "c'mon" : "come on", 
                           "d'you" : "do you",
                           "'til" : "until",
                           "hav" : "have",
                           "'ere" : "here",
                           "o'" : "oh",
                           "'bout" : "about",
                           "i'ma" : "i am going to",
                           "'round" : "around",
                           "wha'evah" : "whatever"}


def expand_contractions(string):
    
    def expand(word):
        if word in contr_word_replacements:
            return contr_word_replacements[word]
        elif word in contr_is_replacements:
            return word.replace("'s", " is")
        else:
            for pattern, repl in contr_substring_replacements.items():
                word = word.replace(pattern , repl)
            return word
    
    return ' '.join([expand(word) for word in string.split()])




def main():
    # Testing Clean Whitespace
    test_clean_whitespace()
    test_strip_punctutation()

def test_strip_punctutation():
    test_str = "This string's 2 + 3 = 4 {} fdsa <{[()]}> boat-tree__cow"
    print("'{}'  ==>  '{}'".format(test_str, strip_punctuation(test_str, {'-', "'"})))
    print

def test_clean_whitespace():
    print("=== Clean Whitespace Test ===")
    test_str = "text   more text with     more    extra white\t\t\tspace\n"
    print("'{}'  ==>  '{}'".format(test_str, clean_whitespace(test_str)))
    print 

if __name__ == "__main__":
    main()


