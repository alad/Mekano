import sys

def progress(n, s=None):
  if s is None:
    sys.stdout.write("[%d]" % n)
  else:
    sys.stdout.write("[%d %s]\n" % (n,s))
  sys.stdout.flush()

