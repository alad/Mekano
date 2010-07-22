import cPickle
import sys, re

def loadpickleitems(filename, progressinterval=None):
    """
    Return a generator that yields each item in a pickled
    file.
    """
    fin = open(filename)
    i = 0
    while(True):
        try:
            r = cPickle.load(fin)
            i += 1
            if progressinterval is not None and i%progressinterval == 0:
                sys.stdout.write("[%s]\r" % i)
                sys.stdout.flush()
            yield r
        except EOFError:
            fin.close()
            if progressinterval is not None:
                print "[%s]" % i
            return

def dump(obj, filename):
    """
    Pickle an object to a file
    """
    with open(filename, "w") as fout:
        cPickle.dump(obj, fout, -1)


def load(filename):
    """
    Load a single object from a pickle file.
    """
    with open(filename) as fin:
        return cPickle.load(fin)


class StateMachineFileParser:
    """A state machine based line-oriented parser.

    You should subclass this and implement 'state'
    functions that begin with '_on', e.g., C{_onStart(self, line)}

    Each such function:
        - takes a single argument 'line' which
          contains the next line read from input string.
        - returns the next state to go into, e.g. "End",
          or returns None (i.e. returns nothing) to stay
          in the same state.
    
    Additionally, two special functions must be defined
    with no parameters:
        1. C{_onInit(self)}
        2. C{_onFinish(self)}
    to handle initialization (local state) and end of file
    (useful for formats that use no ending tag).
    
    """

    def __init__(self):
        self._stateFunctions = {}
        self._introspect()
        self.state = self._onInit()

    def _introspect(self):
        """
        Find all functions that begin with 'on' and create
        a map from the rest of their name to the function object.
        """
        for fn in filter(lambda x: x.startswith("_on"), dir(self)):
            self._stateFunctions[fn[3:]] = getattr(self, fn)

    def parse(self, line):
        fn = self._stateFunctions[self.state]
        newstate = fn(line)
        if newstate is not None:
            self.state = newstate
        
    def parseFile(self, fin):
        self._onInit()
        for line in fin:
            self.parse(line)
        self.close()
    
    def parseFileName(self, fileName):
        with open(fileName) as fin:
            self.parseFile(fin)

    def close(self):
        self._onFinish()
    
    def _onFinish(self):
        pass


class TrecParser(StateMachineFileParser):
    """TREC file parser

    tp = TrecParser(fin, callback)

    The callback function receives (docid, text).

    To begin parsing, use one of:
        - C{parse(line)}                # call repeatedly for each line.
        - C{parseFile(fin)}
        - C{parseFileName(fileName)}
    
    """
    
    def __init__(self, callback):
        StateMachineFileParser.__init__(self)
        self.callback = callback
    
    def _onInit(self):
        self.docid = None
        self.textlines = []
        return "Misc"
        
    def _onMisc(self, line):
        """Waiting for a new doc to start"""
        if line.startswith("<DOCNO>"):
            p = line.find("<", 7)
            self.docid = line[7:p].strip()
            self.textlines = []
            return "GotDocId"

    def _onGotDocId(self, line):
        """Waiting for the TEXT section to start"""
        if line.startswith("<TEXT>"):
            return "Text"

    def _onText(self, line):
        """Seeing text lines"""
        if line.startswith("</TEXT>"):
            self.callback(self.docid, "".join(self.textlines))
            return "Misc"
        else:
            self.textlines.append(line)
    
class SMARTParser(StateMachineFileParser):
    """SMART file parser

        >>> tp = SMARTParser(callback, sections = None)

    The callback function receives (docid, cats, text).
    
    sections:    None: Read all sections, or ["T", "W"]: Only read sections .T and .W

    To begin parsing, use one of:
        - C{parse(line)}                # call repeatedly for each line.
        - C{parseFile(fin)}
        - C{parseFileName(fileName)}

    """

    def __init__(self, callback, sections = None):
        StateMachineFileParser.__init__(self)
        self.allowedsections = sections
        self.callback = callback
        self.cat_regex = re.compile("([^ ]+) 1")
    
    def _onInit(self):
        self.docid = None
        self.cats = None
        self.textlines = []
        self.sectionheader = None
        return "Misc"
    
    def _onMisc(self, line):
        """Waiting for a new doc to start"""
        if line.startswith(".I"):
            self.docid = line[2:].strip()
            self.textlines = []
            self.cats = None
            self.sectionheader = None
            return "GotDocId"
    
    def _onGotDocId(self, line):
        """Waiting for the .C section to start"""
        if line.startswith(".C"):
            return "SawDotC"
    
    def _onSawDotC(self, line):
        """Reading the categories line"""
        self.cats = [c.group(1) for c in self.cat_regex.finditer(line)]
        return "SectionHeader"
    
    def _onSectionHeader(self, line):
        """Reading the section header line"""
        self.sectionheader = line[1:2]
        return "SectionText"

    def _onSectionText(self, line):
        """Seeing text lines"""
        if line.startswith("."):
            if line.startswith(".I"):
                self._doCallBack()
                return self._onMisc(line)
            else:
                return self._onSectionHeader(line)
        
        if self.allowedsections is not None and self.sectionheader not in self.allowedsections:
            return None
        
        self.textlines.append(line)
    
    def _onFinish(self):
        self._doCallBack()
    
    def _doCallBack(self):
        if len(self.textlines):
            self.callback(self.docid, self.cats, "".join(self.textlines))
    
