#!/usr/bin/env python
# -*- coding: utf8 -*-
####################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
####################################################################
"""
Brief: Feed特征抽取类
File: util.py
Author: wangsijiang01@baidu.com
Date: Wed Jun  7 16:18:37 CST 2017
"""

import sys
import gzip
import math
import json
import ConfigParser

SEQ_LEN = 128
EMB_LEN = 128
EPS = 0.000001

MAX_SHOW=10000

COD_SIM_SCALE = 5.0


def cos_sim(user_vec, 
        ad_vec, 
        user_vec_sq_sum=0.0, 
        ad_vec_sq_sum=0.0):
    """
    @Brief: 计算两个向量的COS距离
    @Args: user_vec,  用户向量
           ad_vec, 广告向量
           user_vec_sq_sum, 用户向量平方和开方
           ad_vec_sq_sum, 广告向量平方和开方
    @Returns:
           相似度
    """
    sumq = 0.0
    sumx = 0.0
    sumy = 0.0
    if len(user_vec) != EMB_LEN:
        return None
    if len(ad_vec) != EMB_LEN:
        return None
    for x, y in zip(user_vec, ad_vec):
        sumq += x * y
        sumx += x * x
        sumy += y * y
    
    if sumx * sumy < EPS:
        return None
    return COD_SIM_SCALE * sumq / math.sqrt(sumx * sumy)


def calc_vec_sq_sum(vec):
    """
    @Brief: 计算向量平方和开方
    """
    if len(vec) != EMB_LEN:
        return "-"
    vec_sq_sum = 0
    for x in vec:
        vec_sq_sum += x * x
    return math.sqrt(vec_sq_sum)


def load_ad_lda_selected_fea():
    """
    @Brief: 加载广告lda特征词典
    """
    selected_fea_dict = {}
    for line in open("./data/lda_top_ad_fea.txt", 'r'):
        segs = line.strip('\r\n').split('\t')
        if len(segs) != 2:
            logging.warning("fields error")
        fea_id = segs[0]
        slot = segs[1]
        selected_fea_dict[fea_id] = slot
    return selected_fea_dict


def load_news_lda_selected_fea():
    """
    @Brief: 加载新闻lda词典
    """
    selected_fea_dict = {}
    for line in open("./data/lda_top_news_fea.txt", 'r'):
        segs = line.strip('\r\n').split('\t')
        if len(segs) != 2:
            logging.warning("fields error")
        fea_id = segs[0]
        slot = segs[1]
        selected_fea_dict[fea_id] = slot
    return selected_fea_dict


def load_dict(dict_path):
    """
    @Brief: Load dictionary from dict_path.
    """
    out_dict = {}
    for line in open(dict_path, 'r'):
        word, idx  = line.decode('gb18030').strip('\n').split('\t')
        out_dict[word] = int(idx)
    return out_dict


def get_parse_shitu_conf(conf_file):
    """
    @Brief: 加载parse_shitu配置 
    """
    cf = ConfigParser.RawConfigParser()
    cf.read(conf_file)
    parse_conf = {}
    parse_conf["af_map_schema"] = cf.get("parse_shitu", "af_map_schema")
    parse_conf["af_map_output_key"] = cf.get("parse_shitu", "af_map_output_key")
    
    schema_pos = {}
    line_count = 0
    for schema in parse_conf['af_map_schema'].split(','):
        schema_pos[schema] = line_count
        line_count += 1
    schema_output = {}
    line_count = 0
    for schema in parse_conf['af_map_output_key'].split(','):
        if schema not in schema_pos:
            print >> sys.stderr, "parse shitu error" 
            exit(1)
        schema_output[schema] = schema_pos[schema] 
    return schema_pos, schema_output


def get_extractor_conf(conf_file):
    """
    @Brief: 加载parse_shitu配置 
    """
    cf = ConfigParser.RawConfigParser()
    cf.read(conf_file)
    parse_conf = {}
    parse_conf["add_token"] = cf.get("extractor", "add_token")
    
    return parse_conf["add_token"].split(",")


def smart_open(file_name):
    """
    @Brief: 适配打开.gz文件
    """
    try:
        if file_name[-3:] == ".gz":
            return gzip.open(file_name, 'r')
        else:
            return open(file_name, 'r')
    except Exception as e:
        print >> sys.stderr, "error: %s" % e
    return None


def parse_shitu(line, schema_pos, schema_output):
    """
    @Brief: Parse_shitu 
    """
    try:
        data = line.strip('\n').decode('gb18030', 'ignore').split('\t')
        res = {}
        for key in schema_output:
            pos = schema_pos.get(key, -1)
            if pos < 0:
                print >> sys.stderr, "output key error: %s" % key
                return None
            if pos < len(data):
                res[key] = data[pos]
            else:
                res[key] = "-"
    except Exception as e:
        return None
    return res


def text_to_seq(text, word_dict):
    """
    @Brief: 文本按字粒度切分
    """
    output = []
    for char in text:
        if char in word_dict:
            output.append(word_dict[char])
        else:
            output.append(0)
        if len(output) >= SEQ_LEN:
            break
    if len(output) == 0:
        output = [0]
    return output


def check_range(x, y):
    """
    @Brief: 
    """
    if x < 0 or x > y:
        return 0
    return x


def D_title(json_text, word_dict):
    """
    @Brief: 从json获取brand特征
    """
    #print json_text.encode('gb18030')
    res = [0]
    try:
        d = json.loads(json_text)
        title = d[0].get('title', '-')
        #print title.encode('gb18030') 
        res = text_to_seq(title, word_dict)
    except Exception as e:
        return res
    return res 


def D_brand(json_text, word_dict):
    """
    @Brief: 从json获取brand特征
    """
    res = [0]
    try:
        d = json.loads(json_text)
        brand = d[0].get('user_name', '-')
        
        if brand == "-":
            brand = d[0].get('brand', '-')
        res = text_to_seq(brand, word_dict)
    except Exception as e:
        return res
    return res 


def D_int(fea, up):
    """
    @Brief: id化
    """
    res = 0
    try:
        res = int(fea)
    except Exception as e:
        res = 0
    if res > up:
        res = up

    return res


def D_text2ids(fea, word_dict):
    """
    @Brief:
    """
    return text_to_seq(fea, word_dict)


def D_cmatch(fea):
    """
    @Brief:
    """
    if fea == "545":
        return 1
    if fea == "546":
        return 2
    return 0


def D_mtid(fea):
    """
    @Brief
    """
    if fea == "3176":
        return 1
    if fea == "3177":
        return 2
    return 0


def D_ip(ip):
    """
    @Brief:
    """
    res = [0] * 32
    try:
        data = ip.split('.')
        if len(data) < 4:
            return res
        for i in range(4):
            item = int(data[i])
            for j in range(8):
                res[i * 8 + j] = float(item & 1)
                item = item / 2
    except Exception as e:
        res = [0] * 32
    return res


def D_showclick(show, click):
    """
    @Brief:
    """
    res = [0.0] * 2
    click = int(click)
    res[0] = max(min(show, MAX_SHOW), 0) * 1.0 / MAX_SHOW
    res[1] = click * 1.0 / max(min(show, MAX_SHOW), 1)
    return res 


def D_totalprofile(total_profile):
    """
    @Brief:
    """
    res = [0.0] * 2
    try:
        show, click = total_profile.split('-')
        show = int(show)
        click = int(click)
        a, b = D_showclick(show, click)
        res[0] = a
        res[1] = b
    except Exception as e:
        return [0.0] * 2
    return res


def D_totalprofile_disperse(total_profile, show_clk_dict):
    """
    @Brief:
    """
    res = 0
    try:
        show_click_pair = total_profile.split('-')
        if show_click_pair in show_clk_dict:
            res = show_clk_dict[show_click_pair]
    except Exception as e:
        pass
    return res
   

def D_cmatchprofile(cmatch_profile):
    """
    seq:545, 546
    """
    res = [0.0] * 4 
    try:
        segs = cmatch_profile.split("%02")
        for seg in segs:
            fields = seg.split(":")
            if len(fields) < 2:
                continue
            cmatch = fields[0]
            show, click = fields[1].split('-')
            show = int(show)
            click = int(click)
            a, b = D_showclick(show, click)
            if cmatch == "545":
                res[0] = a
                res[1] = b
            if cmatch == "546":
                res[2] = a
                res[3] = b
    except Exception as e:
        pass
    return res


def D_mtidprofile(mtid_profile):
    """
    seq:3176, 3177
    """
    res = [0.0] * 4 
    try:
        segs = mtid_profile.split("%02")
        for seg in segs:
            fields = seg.split(":")
            if len(fields) < 2:
                continue
            mtid = fields[0]
            show, click = fields[1].split('-')
            show = int(show)
            click = int(click)
            a, b = D_showclick(show, click)
            if mtid == "3176":
                res[0] = a
                res[1] = b
            if mtid == "3177":
                res[2] = a
                res[3] = b
    except Exception as e:
        pass
    return res


def D_titleprofile(title_profile, word_dict, is_click=True):
    """
    @Brief
    """
    click_title_seq = []
    try:
        segs = title_profile.split("%02")
        for seg in segs:
            fields = seg.split(":")
            if len(fields) < 2:
                continue
            title = fields[0]
            show_click = fields[1]
            show, click = show_click.split("-")
            show = int(show)
            click = int(click)
            if ((click > 0) == is_click):
                for char in title:
                    click_title_seq.append(char)

    except Exception as e:
        pass
   
    #text = "".join(click_title_seq)
    #print >> sys.stderr, text.encode('utf8')

    return text_to_seq(click_title_seq, word_dict) 
        

def D_wordprofile(word_profile, word_dict):
    """
    Brief
    """
    return text_to_seq(word_profile, word_dict)


def D_ua(ua_profile, word_dict):
    """
    @ua
    """
    return text_to_seq(ua_profile, word_dict)


def D_user_lda_vector(lda_profile):
    """
    @user lda
    """
    res = []
    segs = lda_profile.lstrip('(').rstrip(')').split(') (')
    for seg in segs:
        fields = seg.split(', ')
        slot = int(fields[0])
        val = float(fields[1])
        res.append((slot, val))
    return res


def D_lda_text_to_vector(text, selected_fea_dict):
    """
    @Brief
    """
    output = [0.0] * (len(selected_fea_dict) + 1)
    segs = text.split(' ')
    for seg in segs:
        fields = seg.split(':')
        if len(fields) < 2:
            continue
        if fields[0] in selected_fea_dict:
            output[int(selected_fea_dict[fields[0]])] = float(fields[1])
    return output  


def D_text_to_sparse_vector(text, selected_fea_dict):
    """
    @Brief
    """
    output = []
    segs = text.split(' ')
    for seg in segs:
        fields = seg.split(':')
        if len(fields) < 2:
            continue
        if fields[0] in selected_fea_dict:
            tup = (int(selected_fea_dict[fields[0]]), float(fields[1]))
            output.append(tup)
        if len(output) >= SEQ_LEN:
            break
    if len(output) == 0:
        default_dup = (0, 0.0)
        output = [default_dup]
    return output  


def D_word(word, word_dict):
    """
    word
    """
    return text_to_seq(word, word_dict)


def merge_extractor(datas, word_dict):
    """
    merge_extractor
    """
    try: 
        # user
        click = D_int(datas.get('clk', 0), 1)
        ageid = D_int(datas.get('age_id', 0), 7)
        genderid = D_int(datas.get('gender_id', 0), 2)
        title_list_ids = D_text2ids(datas.get('gs_wise_title_list', '-'), word_dict)
        multi_preq_ids = D_text2ids(datas.get('gs_wise_multi_preq', '-'), word_dict)
        queryids = D_text2ids(datas.get('query_profile', ''), word_dict)
        total_profile_fea = D_totalprofile(datas.get('total_profile', '-'))
        cmatch_profile_fea = D_cmatchprofile(datas.get('cmatch_profile', '-'))
        mtid_profile_fea = D_mtidprofile(datas.get('mtid_profile', '-'))
        click_word_profile_fea = D_titleprofile(datas.get('word_profile', '-'), word_dict, True)
        word_profile_fea = D_titleprofile(datas.get('word_profile', '-'), word_dict, False)
        click_title_profile_fea = D_titleprofile(datas.get('title_profile', '-'), word_dict, True)
        title_profile_fea = D_titleprofile(datas.get('title_profile', '-'), word_dict, False)
        
        
        # position
        cmatch_fea = D_cmatch(datas.get('cmatch', '-'))
        ip_fea = D_ip(datas.get('ip', '-'))
        ua_fea = D_ua(datas.get('ua', '-'), word_dict)
        
        # ad
        mtid_fea = D_mtid(datas.get('mt_id', '-'))
        #title_ids = D_text2ids(datas.get('title', '-'), word_dict)
        title_ids = D_title(datas.get('json', '-'), word_dict)
        brand_ids = D_brand(datas.get('json', '-'), word_dict)
       
        # pair
        word_ids = D_text2ids(datas.get('word', '-'), word_dict)
    except Exception as e:
        print >> sys.stderr, e
        return None
     
    ret = [
            ageid, genderid, \
            multi_preq_ids, \
            queryids, \
            click_word_profile_fea, \
            click_title_profile_fea, \
            total_profile_fea, \
            cmatch_profile_fea, \
            mtid_profile_fea, \
            cmatch_fea, ua_fea, \
            mtid_fea, title_ids, brand_ids, \
            word_ids, \
            click
            ]
    return ret
  


def asq_extractor(datas, word_dict, lda_ad_fea_dict, lda_news_fea_dict):
    """
    asq_extractor
    """
    try: 
        # user
        click = D_int(datas.get('clk', 0), 1)
        ageid = D_int(datas.get('age_id', 0), 7)
        genderid = D_int(datas.get('gender_id', 0), 2)
        title_list_ids = D_text2ids(datas.get('gs_wise_title_list', '-'), word_dict)
        multi_preq_ids = D_text2ids(datas.get('gs_wise_multi_preq', '-'), word_dict)
        queryids = D_text2ids(datas.get('query_profile', ''), word_dict)
        total_profile_fea = D_totalprofile(datas.get('total_profile', '-'))
        cmatch_profile_fea = D_cmatchprofile(datas.get('cmatch_profile', '-'))
        mtid_profile_fea = D_mtidprofile(datas.get('mtid_profile', '-'))
        word_profile_fea = D_wordprofile(datas.get('word_profile', '-'), word_dict)
        title_profile_fea = D_titleprofile(datas.get('title_profile', '-'), word_dict)
        
        # lda
        #user_ad_lda = D_text_to_sparse_vector(datas.get('ad_title_lda', '-'), lda_ad_fea_dict)
        user_ad_lda = D_lda_text_to_vector(datas.get('ad_title_lda', '-'), lda_ad_fea_dict)
        #user_news_lda = D_text_to_sparse_vector(datas.get('news_title_lda', '-'), lda_news_fea_dict)
        user_news_lda = D_lda_text_to_vector(datas.get('news_title_lda', '-'), lda_news_fea_dict)

        # ad
        mtid_fea = D_mtid(datas.get('mt_id', '-'))
        title_ids = D_text2ids(datas.get('title', '-'), word_dict)
        brand_ids = D_brand(datas.get('json', '-'), word_dict)
        
        # pair
        cmatch_fea = D_cmatch(datas.get('cmatch', '-'))
        ip_fea = D_ip(datas.get('ip', '-'))
        word_ids = D_text2ids(datas.get('word', '-'), word_dict)
        ua_fea = D_ua(datas.get('ua', '-'), word_dict)
    except Exception as e:
        print >> sys.stderr, e
        return None
     
    ret = [
            ageid, genderid, \
            multi_preq_ids, multi_preq_ids, multi_preq_ids, \
            queryids, queryids, queryids, \
            title_profile_fea, title_profile_fea, title_profile_fea, \
            word_profile_fea, word_profile_fea, word_profile_fea, \
            user_ad_lda, user_news_lda, \
            mtid_fea, \
            title_ids, title_ids, title_ids, \
            cmatch_fea, \
            word_ids, word_ids, word_ids, \
            ua_fea, \
            click
            ]
    return ret
    

def bsq_extractor(datas, word_dict, lda_ad_fea_dict, lda_news_fea_dict):
    """
    bsq_extractor
    """
    try: 
        # user
        click = D_int(datas.get('clk', 0), 1)

        ageid = D_int(datas.get('age_id', 0), 7)
        genderid = D_int(datas.get('gender_id', 0), 2)

        cmatch_fea = D_cmatch(datas.get('cmatch', '-'))

        queryids = D_text2ids(datas.get('query_profile', ''), word_dict)
        multi_preq_ids = D_text2ids(datas.get('gs_wise_multi_preq', '-'), word_dict)
#
        word_profile_fea = D_wordprofile(datas.get('word_profile', '-'), word_dict)
        total_profile_fea = D_totalprofile(datas.get('total_profile', '-'))
        cmatch_profile_fea = D_cmatchprofile(datas.get('cmatch_profile', '-'))
        mtid_profile_fea = D_mtidprofile(datas.get('mtid_profile', '-'))
        title_profile_fea = D_titleprofile(datas.get('title_profile', '-'), word_dict)
        
        #lda
        user_ad_lda = D_text_to_sparse_vector(datas.get('lda_user_ad_click', '-'), 
                lda_ad_fea_dict)
        user_news_lda = D_text_to_sparse_vector(datas.get('lda_user_news_click', '-'), 
                lda_news_fea_dict)

        # ad
        title_fea = D_text2ids(datas.get('title', '-'), word_dict)
        brand_fea = D_brand(datas.get('json', '-'), word_dict)
        
    except Exception as e:
        print >> sys.stderr, e
        return None
     
    ret = [ 
            ageid, genderid, queryids, multi_preq_ids, cmatch_fea, \
            word_profile_fea, total_profile_fea, cmatch_profile_fea, \
            mtid_profile_fea, title_profile_fea, \
            user_ad_lda, user_news_lda, title_fea, brand_fea, click
            ]   
    return ret


def bsq_extractor_from_ins(line, word_dict):
    """
    bsq_extractor_from_ins(line, word_dict):
    """
    datas = line.strip('\n').split('\t')
    if len(datas) < 15:
        return None
    try: 
        # user
        click = D_int(datas[0], 1)
        print click

        ageid = D_int(datas[1], 7)
        genderid = D_int(datas[2], 2)

        cmatch_fea = D_cmatch(datas[5])

        
        queryids = [int(ids) for ids in datas[3].split(' ')]
        multi_preq_ids = [int(ids) for ids in datas[4].split(' ')] 

        word_profile_fea = [int(ids) for ids in datas[6].split(' ')]
        total_profile_fea = [float(val) for val in datas[7].split(' ')]
        cmatch_profile_fea = [float(val) for val in datas[8].split(' ')]
        mtid_profile_fea = [float(val) for val in datas[9].split(' ')]
        title_profile_fea = [int(ids) for ids in datas[10].split(' ')]
        
        #lda
        user_ad_lda = D_user_lda_vector(datas[11])
        user_news_lda = D_user_lda_vector(datas[12])

        # ad
        title_fea = [int(ids) for ids in datas[13].split(' ')]
        brand_fea = [int(ids) for ids in datas[14].split(' ')]
        
    except Exception as e:
        print >> sys.stderr, e
        return None
     
    ret =  [ageid, genderid, queryids, multi_preq_ids, cmatch_fea, \
            word_profile_fea, total_profile_fea, \
            cmatch_profile_fea, mtid_profile_fea, title_profile_fea, \
            user_ad_lda, user_news_lda, \
            title_fea, brand_fea, \
            click
            ]
    return ret


def print_ins(ins, data, add_token):
    """
    pirnt ins
    """
    tokens = []
    feas = []
    show = "1"
    click = str(ins[-1])
    feas.append(show)
    feas.append(click)
    for i in range(len(ins)):
        item = ins[i]
        slot_idx = i + 1
        fea = ""
        if isinstance(item, list):
            fea = ",".join([str(i) for i in item])
        else:
            fea = str(item)
        feas.append(":".join([fea, str(slot_idx)]))
    
    tokens.append(" ".join(feas))
    
    for schema in add_token:
        tokens.append(data.get(schema, "-"))

    text = "\t".join(tokens)
    print text.encode('gb18030')


if __name__ == "__main__":
    word_dict = load_dict('./data/dict.txt')
    schema_pos, schema_output = get_parse_shitu_conf("./conf/schema.conf") 
    add_token = get_extractor_conf("./conf/extractor.conf") 
    
    print >> sys.stderr, "add_token = %s" % (",".join(add_token))
    
    lda_ad_fea_dict = load_ad_lda_selected_fea()
    lda_news_fea_dict = load_news_lda_selected_fea() 
    

    for line in sys.stdin:
        line = line.strip('\n')
        data = parse_shitu(line, schema_pos, schema_output)
        ins = merge_extractor(data, \
                word_dict)
        print_ins(ins, data, add_token)

