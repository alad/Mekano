"""Provides an interface to the Indri binaries.

The most important function is L{runquery}.
Results are represented using the L{Result} class.

@var binaryLocation         : Directory containing runquery, buildindex, and dumpindex binaries. 
                              By default, use the search path.
"""

import re, subprocess

# Location of the runquery, buildindex, and dumpindex binaries.
# If None, the shell is used to look for them in the path.
binaryLocation = None


class Result(object):
    """Represents a result (one document) returned by Indri."""
    
    def __init__(self):
        self.score = 0
        self.docid = None
        self.text = None
    
    def __repr__(self):
        ret = "Doc: %s        Score: %f" % (self.docid, self.score)
        if self.text:
            ret += ("\n" + self.text)
        return ret
        
def runquery(index, query, printDocuments = False, printSnippets = False):
    """Run a query against an Indri index.
    
    @param index            : A directory pointing to an Indri index
    @param query            : A string query
    @param printDocuments   : Whether to include document text (available in C{.text} attribute of the returned objects)
    @param printSnippets    : Whether to include document snippets (available in C{.text} attribute) (Only one of these should be true)
    @return                 : A list of L{Result} objects.
    """
    multiline = False
    ret = []
    if binaryLocation: cmd = "%s/runquery"
    else: cmd = "runquery"
    cmd += " -index=%s -query='%s'" % (index, query)
    if printDocuments:
        cmd += " -printDocuments=1"
        multiline = True
    elif printSnippets:
        cmd += " -printSnippets=1"
        multiline = True
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if not multiline:
        for line in stdout.split("\n"):
            if not line: break
            score, docid, _, _ = line.split()
            result = Result()
            result.score = float(score)
            result.docid = docid
            ret.append(result)
    else:
        pat = "-([0-9.]+)\s+([^\s]+)\s+([0-9]+)\s+([0-9]+)$"
        result = None
        for line in stdout.split("\n"):
            line = line.rstrip()
            if re.match(pat, line):
                result = Result()
                score, docid, _, _ = line.split()
                result.score = float(score)
                result.docid = docid
                result.text = ""
                ret.append(result)
            else:
                result.text += line + " "
    return ret
                
                    
    
