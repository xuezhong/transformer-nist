#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
####################################################################
"""
Brief: paddle_feednnq dataprovider
File: dataprovider.py
Author: wangsijiang01@baidu.com
Date: Wed Jun  7 16:48:02 CST 2017
"""
import sys
import random
import util
import gzip


DICT_FILE = "dict.txt"
SCHEMA_FILE = "schema.conf"
MAX_SIZE = 1000


class CTRData(object): 
    """
    Feed CTR Reader
    """
    
    def __init__(self, dict_file=DICT_FILE, schema_file=SCHEMA_FILE):
        """
        init
        """
        
        #self.logger.info("hook")
        word_dict = util.load_dict(dict_file)
        schema_pos, schema_output = util.get_parse_shitu_conf(schema_file) 
    
    
        self.word_dict = word_dict
        self.schema_pos = schema_pos
        self.schema_output = schema_output
    
        dict_size = len(word_dict)
        schema_pos_size = len(schema_pos)
        schema_output_size = len(schema_output)
    
        #self.logger.info("dict_size is %d" % dict_size)
        #self.logger.info("schema_pos_size is %d" % schema_pos_size)
        #self.logger.info("schema_output_size is %d" % schema_output_size)
        #self.logger.info("lda_ad_size is %d" % lda_ad_size)
        #self.logger.info("lda_news_size is %d" % lda_news_size)
    
        #self.input_types = [
        #        integer_value(9),
        #        integer_value(3),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value_sequence(dict_size + 1),
        #        dense_vector(2),
        #        dense_vector(4),
        #        dense_vector(4),
        #        integer_value(3),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value(3),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value_sequence(dict_size + 1),
        #        integer_value(2)]
   
    def train_default(self, num=1024):
        """
        """
        def reader():
            """"""
            num = 10240
            while num > 0:
                num -= 1
                d = int(random.random() * MAX_SIZE)
                yield [d], d % 2
        return reader
    
    def train(self, filelist):
        """
        """
        def reader():
            """"""
            for filename in filelist:
                ins = self.reader_file(filename)
                for data in ins:
                    yield data
        return reader

    def reader_file(self, file_name):
        """
        process feednnq shitu
        """
        try:
            for line in util.smart_open(file_name):
                line = line.strip('\n')
                
                data = util.parse_shitu(line, \
                        self.schema_pos, \
                        self.schema_output)
                ret = util.merge_extractor(data, \
                        self.word_dict)
                
                if ret is None:
                    continue
                #print "hello" 
                yield ret
        except Exception as e:
            print >> sys.stderr, "error: %s" % e
            #self.logger.info("error: %s" % e)
   
if __name__ == "__main__":
    d = CTRData()
    reader = d.train([sys.argv[1]] * 20)
    #r = d.train_default(1000)
    count = 0
    for data in reader():
        print data
        count += 1
        if count % 10 == 0:
            print >> sys.stderr, "count = %d" % count 
