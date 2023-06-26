import sys
sys.path.append('Final_codes_and_data/')
import os

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from urllib.parse import urlparse
from urllib.parse import unquote
#from PlotUtils import addDatetimeLabels, add_vspans
from GraphUtils import buildGraphSqlite
from datetime import datetime
from collections import Counter
import json
import random
from tqdm import tqdm


import time
import pickle
import graph_tool.all as gt
import networkx as nx


from general_utilities import build_CI_rank,add_CI_to_graph



official_twitter_clients = ['Twitter for iPhone',
'Twitter for Android',
'Twitter Web Client',
'Twitter for iPad',
'Mobile Web (M5)',
'TweetDeck',
'Facebook',
'Twitter for Windows',
'Mobile Web (M2)',
'Twitter for Windows Phone',
'Mobile Web',
'Google',
'Twitter for BlackBerry',
'Twitter for Android Tablets',
'Twitter for Mac',
'iOS',
'Twitter for BlackBerry®',
'OS X']

urls_db_file = '**'
tweet_db_file1 = "**"
tweet_db_file2 = "**"


sql_query_urls = """SELECT tweet_id FROM urls.urls
                         WHERE final_hostname IN (
                                 SELECT hostname FROM hosts_{med_type}_rev_stat
                                 WHERE perccum > 0.01)
                 """
                         

media_types = ['fake', 'far_right', 'right', 'lean_right',
               'center', 'lean_left', 'left','far_left']

gran_path = '**'
edgelist_path = gran_path + 'edgelist'
categories_network = gran_path + 'categories_networks'
section1_data_path = gran_path + 'section_one/' 

#import users accouns info
users_account = json.load(open(section1_data_path+'users_accounts.txt'))
users = pd.read_csv(section1_data_path+"all_users.csv", index_col="user_id",
                    usecols =["user_id", "is_IRA"], dtype={"user_id": str, "is_IRA": int})
IRA = users[users.is_IRA > 0]
IRA = set([int(i) for i in IRA.index if len(i)!=64])

Suspended = set([int(i) for i in users_account if (users_account[i]=='Suspended' and int(i) not in IRA)])
Not_Found = set([int(i) for i in users_account if (users_account[i]=='Not found' and int(i) not in IRA)])
Verified = set([int(i) for i in users_account if( users_account[i]=='Verified' and int(i) not in IRA)])
Not_Verified = set([int(i) for i in users_account if (users_account[i]=='Not verified' and int(i) not in IRA)])

node_to_supernode = {'IRA':111111,'Suspended':222222,'Not Found':333333,'Verified':444444,'Not Verified':555555 }


start_date = datetime(2016,6,1)
stop_date = datetime(2016,11,9)

        
# get edges list
def build_edgelist(start_date,stop_date,media_types):
    edges_db_file = dict()
    for tweet_db_file in [tweet_db_file1,tweet_db_file2]:
        print(tweet_db_file)

        edges_db_file[tweet_db_file] = dict()
        with sqlite3.connect(tweet_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

            c = conn.cursor()
            c.execute("ATTACH '{urls_db_file}' AS urls".format(urls_db_file=urls_db_file))



            for media_type in media_types:
                print(media_type)
                edges_db_file[tweet_db_file][media_type] = buildGraphSqlite(conn, 
                                        graph_type='retweet', 
                                        start_date=start_date,
                                        stop_date=stop_date,
                                        additional_sql_select_statement=sql_query_urls.format(med_type=media_type),
                                        graph_lib='edge_list')

                print(time.time() - t0)
    # save
    with open(edgelist_path+'categories_edge_lists.pickle', 'wb') as fopen:
        pickle.dump(edges_db_file, fopen)
    
def get_original_id():
    sql_select = """SELECT tweet_id,retweet_id from tweet_to_retweeted_uid""" 
    for db_file in [tweet_db_file1,tweet_db_file2]:
        print(db_file)
        with sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
            c = conn.cursor()
            c.execute(sql_select)
            d = c.fetchall()
            if db_file == tweet_db_file2:
                for i in d: 
                    if i[0] not in dic: 
                        dic[i[0]] = i[1] 
            else:
                dic = {i[0]:i[1] for i in d}
    return dic

#%% build graphs
def build_graph_from_edges(edge_list, graph_name):

    G = gt.Graph(directed=True)
    G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
    G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    G.edge_properties['source_id'] = G.new_edge_property('int64_t')
    G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id])

    G.gp['name'] = G.new_graph_property('string',graph_name)

    return G

def add_vertex_properties(G):
    
    # compute some vertex properties
    

    G.vp['k_out'] = G.degree_property_map('out')
    G.vp['k_in'] = G.degree_property_map('in')
    
    
# build and process each graph    
def build_retweet_networks():

    edges_db_file = []
    with (open(edgelist_path+'categories_edge_lists.pickle', "rb")) as openfile:
        edges_db_file.append(pickle.load(openfile))
    edges_db_file=edges_db_file[0]
    for media_type in edges_db_file[tweet_db_file1].keys():
        graph_type = 'simple'
        period = 'june-nov'
            
        print(media_type)
    
        edges_array = np.concatenate((edges_db_file[tweet_db_file1][media_type],
                                          edges_db_file[tweet_db_file2][media_type]))

        G = build_graph_from_edges(edges_array, media_type)

        if graph_type == 'simple':
            gt.remove_parallel_edges(G)
            gt.remove_self_loops(G)

        G.gp['period'] = G.new_graph_property('string',period)
        G.gp['graph_type'] = G.new_graph_property('string',graph_type)

        add_vertex_properties(G)
        
        #adding CI
        #add_CI_to_graph(G)
        
        #save graph
        print('saving graph...')
        filename = categories_network + '/' +'retweet_graph_' + media_type + '_' + graph_type + '_' + period + '.gt'
        filename_graphml = categories_network + '/' +'retweet_graph_' + media_type + '_' + graph_type + '_' + period + '.graphml'
        print(filename)
        G.save(filename)
        
def collapsed_edge_list(edges_array, n_times = 100):

    t0 = time.time()
    tweet_ids  = np.transpose(edges_array)[1]

    for sample in tqdm(range(n_times)):

        from_ =  np.transpose(edges_array)[0]
        to_  = np.transpose(edges_array)[1]
        if sample == 0:
            all_ = set(np.concatenate([from_,to_])) 
            sample_size = len([i for i in IRA if i in all_])
            #print('Sample size ',sample_size)

        for group,grp in zip([IRA,Suspended,Not_Found,Verified,Not_Verified],
                             ['IRA','Suspended','Not Found','Verified','Not Verified']):

            group = list(all_ & group)

            group = random.sample(group,sample_size)    
            
            if len(group) != sample_size:
                print('Whaaat')
                break
            
            from_ = [node_to_supernode[grp]  if i in group else i for i in from_]
            to_ = [node_to_supernode[grp]  if i in group else i for i in to_]

        final_edge_list = np.transpose([from_,to_,tweet_ids])
        yield final_edge_list

def sample_retweet_networks(n_times):
      
        edges_db_file = []

        with (open(edgelist_path+'categories_edge_lists.pickle', "rb")) as openfile:
            edges_db_file.append(pickle.load(openfile))
        edges_db_file=edges_db_file[0]

        for media_type in edges_db_file[tweet_db_file1].keys():
            group_pos = {}
            graph_type = 'simple'
            period = 'june-nov'

            print(media_type)

            edges_array = np.concatenate((edges_db_file[tweet_db_file1][media_type],
                                              edges_db_file[tweet_db_file2][media_type]))
            
            print('Analyzing influencers group for...',media_type)
            
            for final_edge_list in collapsed_edge_list(edges_array,n_times = n_times):
                G = build_graph_from_edges(final_edge_list, media_type)

                if graph_type == 'simple':
                    gt.remove_parallel_edges(G)
                    gt.remove_self_loops(G)



                #adding CI
                G = add_CI_to_graph(G)
                rst = build_CI_rank(G)

                for grp in ['IRA','Suspended','Not Found','Verified','Not Verified']:
                    if grp not in group_pos:
                        group_pos[grp] = {}
                    for dire in ['in','out']:
                        if dire not in group_pos[grp]:
                            group_pos[grp][dire] = []
                        group_pos[grp][dire].append(rst[dire + '_rank'][node_to_supernode[grp]])

            with open(section1_data_path+f'sampled_influencers_{media_type}.txt','w') as file:
                file.write(json.dumps(group_pos)+'\n')
                
def original_ids()
    categories = []
    
    dic = get_original_id()
    
    start_date = '2016-06-01'
    end_date = '2016-11-09'
    
    for media_type in media_types:

        data=pd.read_pickle(section1_data_path+f'{media_type}.pkl')
        data=data[(data.datetime_EST>=start_date) & (data.datetime_EST<=end_date)]

        data.drop(columns=['datetime_EST','source_content'],inplace=True)

        type_ = []
        for i in data['user_id']:
            c = 0
            for typology,nm in zip([Suspended,Not_Found,Verified,Not_Verified,IRA],
                               ['Suspended','Not Found','Verified','Not Verified','IRA']):
                if i in typology:
                    type_.append(nm)
                    c+=1
            if c == 0:
                type_.append('None') 

        original_id  = []
        for id_ in data['tweet_id']:
            if id_ in dic:
                original_id.append(1)
            else:
                original_id.append(0)

        data['type_'] = type_
        data['is_retweet'] = original_id
        data['category'] = [media_type for i in range(len(data))]

        categories.append(data)
    all_tweets = pd.concat(categories)
    all_tweets.to_csv(section1_data_path+'all_tweets.csv')
        
if __name__ == "__main__":
    
    #t0 = time.time()
    
    #4/5/2023
    #build_edgelist(start_date,stop_date,media_types)
    
    #print('Edge lists for the categories created in ', time.time()-t0,' seconds')
    
    #t0 = time.time()
    #9/5/2023
    #build_retweet_networks()
    #print('Categories networks created in ', time.time()-t0,' seconds')
    
    #9/5/2023
    #t0 = time.time()
    #sample_retweet_networks(n_times = 100)
    #print('Categories networks created in ', time.time()-t0,' seconds')    
    
    
    #10/5/2023
    t0 = time.time()
    original_ids()
    print('Categories networks created in ', time.time()-t0,' seconds')    
    
