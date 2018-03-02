#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/06/09 @Northrend
# updated 2017/06/09 @Northrend
#
# convert caffe-list file to mxnet.lst:
# Places365_val_00000001.jpg 165
# -->
# 0	165	Places365_val_00000001.jpg
#
# e.g.
# python make_recordio_lst.py /path/to/places365_val.txt /path/to/places365_val.lst
#

import sys
import random


def mklst(input, output, basename=False):
    lst = []
    file_in = open(input,'r')
    file_lst = open(output, 'w')
    i = 0
    for buff in file_in:
        if basename: 
            lst.append(os.path.basename(buff.split()[0]), int(buff.split()[1]))
            i += 1
        else:
            lst.append((buff.split()[0], int(buff.split()[1])))
            i += 1
    print 'total: ' + str(i)
    lst_index = range(i)
    random.shuffle(lst_index)
    for index in lst_index:
        file_lst.write('{}\t{}\t{}\n'.format(index, lst[index][1], lst[index][0]))
    file_lst.close()
    file_in.close()
    print 'done'


if __name__ == '__main__':
    mklst(sys.argv[1], sys.argv[2])
