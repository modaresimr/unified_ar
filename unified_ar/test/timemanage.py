
import logging
import time  # line number 1
from random import random

from pyprof_timer import Profiler, Timer, Tree

logger=logging.getLogger('a')

# automatic_example.py




def f1():
    time.sleep(1)
def show(p):
    logger.warning(Tree(p.root))

import random
#@Profiler(depth=4, on_disable=show)
def f2():
    time.sleep(.6+random.random())

def mysleep(t):
    time.sleep(t)

def fact(i):
    f2()
    if(i==1):
        return 1
    return i*fact(i-1)

@Profiler(depth=4, on_disable=show)
def main():
    f1()
    for i in range(5):
        f2()

    fact(3)


if __name__ == '__main__':
    main()
    main()
