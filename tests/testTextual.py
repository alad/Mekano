import mekano as mk
from nose.tools import *

shortstr = ""
longstr = ""
words_and_numbers = ""

def setup():
  global shortstr, longstr, words_and_numbers

  shortstr = " Santa's helpers are subordinate clauses. ... !@"

  longstr = "There are 10^11 stars in the galaxy.  \
  That used to be a huge number.                   \
  But it's only a hundred billion.                 \
  It's less than the national deficit!             \
  We used to call them astronomical numbers.       \
  Now we should call them economical numbers.      \
                                   - Richard Feynman"

  words_and_numbers = "This is a $700,000,000,000 bail-out       \
  but 7upper is not a word. 100,000's a big number, but not ,12, \
  120.434 is a number,additional. 12,34 56,67a 66,333a"


def testBasicTokenizer():
  tokens = list(mk.textual.BasicTokenizer(shortstr))
  assert_equal(tokens, ["santa", "s", "helpers", "are",
                    "subordinate", "clauses"])

  tokens = list(mk.textual.BasicTokenizer(shortstr, 4))
  assert_equal(len(tokens), 4)

  tokens = list(mk.textual.BasicTokenizer(longstr, 4))
  assert_equal(len(tokens), 26)


def testWordRegexNumberTokenizer():
  tokenizer = mk.textual.WordNumberRegexTokenizer
  tokens = list(tokenizer(words_and_numbers))
  assert_equal (tokens, ['this', '$700,000,000,000', 'bail', 'word',
                    '100,000', 'number', '12', '120.434', 'number',
                    'additional', '12', '34', '56', '66'])

  assert_equal(list(tokenizer("$700")), ['$700'])
  

