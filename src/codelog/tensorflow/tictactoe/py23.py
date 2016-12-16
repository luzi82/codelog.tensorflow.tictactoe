import sys
import os, errno
import builtins

if sys.version_info.major == 2:

    def makedirs(name, mode=0o777, exist_ok=False):
        if exist_ok:
            try:
                os.makedirs(name,mode)
            except OSError as e:
                if e.errno != errno.EEXIST or not os.path.isdir(name):
                    raise
        else:
            os.makedirs(name,mode)
    
    range_ = builtins.range

elif sys.version_info.major == 3:

    makedirs = os.makedirs
    range_ = range
