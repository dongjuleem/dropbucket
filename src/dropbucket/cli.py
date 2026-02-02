import sys
from importlib.metadata import version as pkg_version
from dropbucket.dropbucket import main

def app():
    if "--version" in sys.argv or "-V" in sys.argv:
        print(pkg_version("dropbucket"))
        raise SystemExit(0)
    main()