from AtomVector import AtomVector

"""mekano.Textual

Textual services like tokenization, stemming, stop word removal

Tokenizers: Convert a string into a stream of tokens. All tokenizers
            lower() the string and return a generator.
            Currently available:
            - BasicTokenizer
            - WordRegexTokenizer         [this is mostly what you want]
            - WordNumberRegexTokenizer

"""

import re
from itertools import ifilter

wordsplitter_rex = re.compile("\W+")

word_regex = re.compile(r"\b[a-z]{3,}\b")

# todo: broken: $700 is not parsed.                             
#word_number_regex = re.compile(r"\b[a-z][a-z0-9]{3,}\b|\$?[0-9]*\.?[0-9]+")
#word_number_regex = re.compile(r"\b[a-z][a-z0-9]{3,}\b|\$?[0-9]*([0-9]+[0-9x]+[0-9]+)?(\.[0-9]+)?")
# todo: "12," shouldn't get parsed. Basically, comma or dot can't end a number

word_number_regex = re.compile(r"\b[a-z][a-z0-9]{3,}\b|(\$|\b)[0-9]+(,[0-9]{3})*(\.[0-9]+)?\b")

def BasicTokenizer(s, minlen=1):
    """Split on any non-word letter.
    
    Words need not start with [a-z]
    """
    for token in wordsplitter_rex.split(s.lower()):
        if len(token) >= minlen:
          yield token

    
def WordRegexTokenizer(s):
    """Find 3 or more letter words.
    
    Words must start with [a-z]
    """
    for match in word_regex.finditer(s.lower()):
        yield match.group()


def WordNumberRegexTokenizer(s):
    """Find 4 or more letter words or numbers/currencies.
    
    Words must start with [a-z]
    """
    for match in word_number_regex.finditer(s.lower()):
        yield match.group()


def Vectorize(s, af, tokenizer = WordRegexTokenizer):
    """Create an AtomVector from a string.
    
    Tokenizes string 's' using tokenizer, creating
    atoms using AtomFactory 'af'.
    """
    av = AtomVector()
    for word in tokenizer(s):
        atom = af[word]
        av[atom] += 1
    return av


