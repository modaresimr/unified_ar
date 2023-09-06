
#
import unified_ar.general.utils
import sys
import os
path = os.getcwd()


def trace_full(frame, event, arg):
    print('ali')


def trace_full2(frame, event, arg):
    pass  # print('ali2')


sys.setprofile(trace_full)
sys.setprofile(trace_full2)
print('ali3')
