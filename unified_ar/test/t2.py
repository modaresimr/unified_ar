

import logging
import time  # line number 1
import random

from pyprof_timer import Profiler, Timer, Tree


def f1():
    mysleep(.3+random.random())


def f2():
    mysleep(.6+random.random())

def mysleep(t):
    time.sleep(t)

def fact(i):
    f2()
    if(i==1):
        return 1
    return i*fact(i-1)


def show(p):
    print(Tree(p.root, threshold=0.5))

@Profiler(depth=4, on_disable=show)
def main():
    f1()
    for i in range(5):
        f2()

    fact(3)


if __name__ == '__main__':
    main()    


