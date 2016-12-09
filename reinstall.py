import os
import sys

if __name__ == '__main__':
    pkgname = sys.argv[1]

    os.system('pip uninstall -y {0}'.format(pkgname))
    os.system('conda install {0}'.format(pkgname))
    os.system('conda env export > environment.yml')
