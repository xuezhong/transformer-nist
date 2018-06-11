#!/usr/bin/python
"""
"""
import sys
import time
sys.path.append(r'thirdparty/model/transformer/')
import nmt_fluid

if __name__ == '__main__':
    '''
    try:
       nmt_fluid.main()
    except:
       time.sleep(1000000)
    '''
    nmt_fluid.main()
