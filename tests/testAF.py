
import mekano as mk

af = mk.AtomFactory()
af["a"]
af["b"]
af["c"]
af["d"]
af["e"]
af["f"]

a = mk.AtomVector()
a[af["a"]] = 1
a[af["b"]] = 2
a[af["c"]] = 3
a[af["e"]] = 4

print a
print a.tostring(af)

nf = af.remove(["c", "f"])

b = mk.af.convertAtomVector(af, nf, a)

print b
print b.tostring(nf)
