import peasoup_tools
import sys
import numpy as np

FILENAME = sys.argv[1]

x = peasoup_tools.OverviewFile(FILENAME)
ar = x.as_array()
print "    ".join([name for name in ar.dtype.names])
ar = np.sort(ar,order="period")

for row in ar:
    print row
