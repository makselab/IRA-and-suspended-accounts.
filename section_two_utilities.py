import pandas as pd
import json
import sqlite3
import time
from tqdm import tqdm
import networkx as nx
from collections import Counter
import numpy as np
import pickle

from scipy import sparse
import scipy.sparse as sp

from pygenstability import run

#TO DO: improve this part
import sys
sys.path.append('/home/matteo/IRA_paper/fake_news_during_election/')


from fake_identify import Are_you_IRA

putin = Are_you_IRA()


gran_path = '/sdf/MatteoPaper/'
raw_data_path = gran_path + 'rawdata/'
edgelist_path = gran_path + 'edgelist/'

section_one_path = gran_path + 'section_one/'
section_two_path = gran_path + 'section_two/'
section_three_path = gran_path + 'section_three/'

start_date = '2016-06-01'
end_date = '2016-11-09'


tweet_db_file1 = "/disk2/US2016_alex/complete_trump_vs_hillary_db.sqlite"
tweet_db_file2 = "/disk2/US2016_alex/complete_trump_vs_hillary_sep-nov_db.sqlite"

graph_type_table_map = {'retweet': 'tweet_to_retweeted_uid',
                        'reply' : 'tweet_to_replied_uid',
                        'mention' : 'tweet_to_mentioned_uid',
                        'quote' : 'tweet_to_quoted_uid'}
    
all_ = ['all/' +"all_retweet_links.txt",
        'all/' +"all_mention_links.txt",
        'all/' +"all_reply_links.txt",
        'all/' +"all_quote_links.txt"]


ego = ["ego_networks/ego_retweet_links.txt",
       "ego_networks/ego_mention_links.txt",
       "ego_networks/ego_reply_links.txt",
       "ego_networks/ego_quote_links.txt"]

epanded = ["expanded_networks/expanded_retweet_links.txt",
           "expanded_networks/expanded_mention_links.txt",
           "expanded_networks/expanded_reply_links.txt",
           "expanded_networks/expanded_quote_links.txt"]

IRA_f = ["IRA/ira-men.txt",
         "IRA/ira-quo.txt",
         "IRA/ira-rep.txt",
         "IRA/ira-ret.txt"]

def connections_among_IRA():
    all_ira = pd.read_csv(raw_data_path+'ira_tweets_csv_hashed.csv',dtype=str)

    #Focus between start and end date. Only Tweets in English
    all_ira_e=all_ira[all_ira['tweet_language']=='en']
    IRA_data=all_ira_e[(all_ira_e['tweet_time']>start_date) & (all_ira_e['tweet_time']<end_date)]


    men_file = open(edgelist_path + "ira-men.txt", "w")
    ret_file = open(edgelist_path + "ira-ret.txt", "w")
    rep_file = open(edgelist_path + "ira-rep.txt", "w")
    quo_file = open(edgelist_path + "ira-quo.txt", "w")

    rep_ira_tweets = IRA_data[IRA_data.in_reply_to_tweetid.notnull()]
    quo_ira_tweets = IRA_data[IRA_data.quoted_tweet_tweetid.notnull()]
    ret_ira_tweets = IRA_data[IRA_data.retweet_tweetid.notnull()]
    men_ira_tweets = IRA_data[IRA_data.user_mentions.notnull()]


    #import users accouns info
    users_account = json.load(open(section1_data_path+'users_accounts.txt'))
    users = pd.read_csv(section1_data_path+"all_users.csv", index_col="user_id",
                        usecols =["user_id", "is_IRA"], dtype={"user_id": str, "is_IRA": int})
    IRA = users[users.is_IRA > 0]


    rep_file.write("tweet_id,user_id,o_tweet_id,o_user_id\n")
    for i, row in rep_ira_tweets.iterrows():
        rep_file.write(",".join([
            row["tweetid"],
            putin.uncover(row["userid"]),
            row["in_reply_to_tweetid"],
            putin.uncover(row["in_reply_to_userid"])
        ]) + "\n")

    cnt = 0
    quo_file.write("tweet_id,user_id,o_tweet_id,o_user_id\n")
    for i, row in quo_ira_tweets.iterrows():
        try:
            quo_file.write(",".join([
                row["tweetid"],
                putin.uncover(row["userid"]),
                row["quoted_tweet_tweetid"],
                putin.uncover(row["retweet_userid"])
            ]) + "\n")
        except:
            print(row["retweet_userid"])
            print(row["in_reply_to_userid"])
            cnt += 1
    print(len(quo_ira_tweets), cnt)

    ret_file.write("tweet_id,user_id,o_tweet_id,o_user_id\n")
    for i, row in ret_ira_tweets.iterrows():
        ret_file.write(",".join([
            row["tweetid"],
            putin.uncover(row["userid"]),
            row["retweet_tweetid"],
            putin.uncover(row["retweet_userid"])
        ]) + "\n")

    men_file.write("tweet_id,user_id,to_tweet_id,to_user_id\n")
    for i, row in men_ira_tweets.iterrows():
        mentions = row["user_mentions"]
        us = mentions[1:-1].split(", ")
        for u in us:
            men_file.write(",".join([
                row["tweetid"],
                putin.uncover(row["userid"]),
                putin.uncover(u)
            ]) + "\n")
            
            
def all_conections():
    for type_ in graph_type_table_map:
        out_file = open(edgelist_path+f'all_{type_}_links.txt', "w")
        print(type_,'...')
        conn = sqlite3.connect(tweet_db_file1)
        c = conn.cursor()
        c.execute(f'''SELECT * FROM {graph_type_table_map[type_]}''')

        for d in c.fetchall():
            out_file.write(" ".join([str(i) for i in d]) + "\n")
        conn.close()


        conn = sqlite3.connect(tweet_db_file2)
        c = conn.cursor()

        c.execute(f'''SELECT * FROM {graph_type_table_map[type_]}''')

        for d in c.fetchall():
            out_file.write(" ".join([str(i) for i in d]) + "\n")
        conn.close()
        
        
def ego_edgelist(default='ego'):
    """
    """
    
    if default=='ego':
        print('Analyzing the ego networks...\n')
        
        IRAs = putin.IRA_user_set
        
    elif default=='expanded':
        print('Analyzing the expanded networks...\n')
        
        users_account = json.load(open(section_one_path + 'users_accounts.txt'))
        ego_nodes = set(list(nx.read_gpickle(section_two_path + 'aggregated_ego.gp').nodes()))
        
        IRAs = set([i for i,j in users_account.items() if \
                    (len(i)!=64 and \
                     i in ego_nodes and \
                     users_account[i]=='Suspended')]) \
                    .union(putin.IRA_user_set)

    for file in all_:
        
        name=file.split('/')[-1]
        print(name,'...')
        
        if default == 'ego':
            
            out_file = edgelist_path + 'ego_networks/'+name.replace('all','ego').replace('.txt','.')+'txt'
        
        elif default == 'expanded':
            
            out_file = edgelist_path + 'expanded_neworks/'+name.replace('all','expanded').replace('.txt','.')+'txt'
  
        with open(out_file, "w") as f:
            for line in tqdm(open(edgelist_path+file)):
                w = line.strip().split()
                if w[1] in IRAs or w[2] in IRAs:
                    f.write(line)

def get_right_files(default):
    files_per_type={}
    for interaction in ['men','ret','rep','quo']:
        files_per_type[interaction]=[]
        if default == 'ego':
            
            files = ego
        elif default == 'expanded':
            
            files = epanded

        for file in files:
            if interaction in file:
                files_per_type[interaction].append(file)
        for file in IRA_f:
            if interaction in file:
                files_per_type[interaction].append(file)
    return files_per_type

def get_hashtag():
    hash_={}
    for db in [tweet_db_file1,tweet_db_file2]:
        conn = sqlite3.connect(db)
        c = conn.cursor() 
        c.execute(f'''SELECT tweet_id,hashtag FROM hashtag_tweet_user''')
        for d in c.fetchall():
            if d:
                if d[0] not in hash_:
                    hash_[d[0]]=[]
                hash_[d[0]].append(d[1])
    return hash_

def build_pickle_graph(default='ego'):
    int_files=get_right_files(default)

    hashtags=[]
    hash_=get_hashtag()

    if default=='ego':
        out_file = open(section_two_path + "aggregated_hashtags.txt", "w")
        
    elif default=='expanded':
        out_file = open(section_three_path + "expanded_hashtags.txt", "w")

    for interaction in int_files: 
        print('Building the pickle network for ', interaction)
        G = nx.DiGraph()
        men_graph = Counter()
        men_rec =set()
        print(int_files[interaction])
        for file in int_files[interaction]:

            if file.split('/')[-1].startswith('ira') and interaction=='men':
                n1_=2
                n2_=1
                cntmin=1
            elif file.split('/')[-1].startswith('ira') and interaction !='men':
                n1_=3
                n2_=1   
                cntmin=1

            for cnt,line in enumerate(open(edgelist_path + file)):
                if  file.split('/')[-1].startswith('ira'):
                    if cnt==0:
                        continue
                    w = line.strip().split(",")
                    t_id = w[0]
                    n1 = w[n1_]
                    n2 = w[n2_]
                else:
                    w = line.strip().split()
                    t_id = w[0]
                    n1 = w[1]
                    n2 = w[2]

                if int(t_id) in hash_:
                    for ln in hash_[int(t_id)]:
                        out_file.write(n1+','+n2+','+ln+ "\n")

                men_rec.add(t_id + "-" + n2)
                men_graph[(n1, n2)] += 1
                
        for e in men_graph:
            w = men_graph[e]
            G.add_edge(*e, weight=w)
        
        if default=='ego':
            
            nx.write_gpickle(G, section_two_path + f"ira_{interaction}_ego.gp")
        
        elif default=='expanded':
            
            nx.write_gpickle(G, section_three_path + f"expanded-{interaction}_ego.gp")

    out_file.close()
        
def aggregated_network(default='ego'):
    
    edge_list=set()
    
    edge_list_weight={}
    
    n_=[]
    
    if default=='ego':
        for interaction in ['men','ret','rep','quo']:
            n_.append(nx.read_gpickle(section_two_path + f"ira_{interaction}_ego.gp"))
    
    elif default=='expanded':
        
        for interaction in ['men','ret','rep','quo']:
            n_.append(nx.read_gpickle(section_three_path + f"expanded-{interaction}_ego.gp"))

    for n in n_: 
        for edge in n.edges(data=True):
            if edge[:2] not in edge_list:
                edge_list.add(edge[:2])
            lb=edge[0]+'_'+edge[1] 
            if lb not in edge_list_weight:
                edge_list_weight[lb]=0
            edge_list_weight[lb]+=edge[2]['weight']

    G = nx.DiGraph()
    G.add_weighted_edges_from([(i[0],i[1],edge_list_weight[i[0]+'_'+i[1] ]) for i in edge_list])

    G.remove_edges_from(nx.selfloop_edges(G,data=True))
    
    print(G.number_of_nodes(), G.number_of_edges())
    print('\n')
    print('Saving gp file...')
    
    if default=='ego':
    
        nx.write_gpickle(G,section_two_path + "aggregated_ego.gp")
    
    elif default=='expanded':
        
        nx.write_gpickle(G,section_three_path + "expanded_ego.gp")    
        
        
def find_all_voters(p=0.5):
    users = {}
    conn1 = sqlite3.connect(tweet_db_file1)
    c1 = conn1.cursor()
    c1.execute('''SELECT user_id, p_pro_hillary_anti_trump FROM class_proba''')
    for d in tqdm(c1.fetchall()):
        if d[0] not in users:
            users[d[0]] = [0, 0]
        if d[1] >= p:
            users[d[0]][0] += 1
        elif d[1] < 1-p:
            users[d[0]][1] += 1
    conn1.close()

    conn2 = sqlite3.connect(tweet_db_file2)
    c2 = conn2.cursor()
    c2.execute('''SELECT user_id, p_pro_hillary_anti_trump FROM class_proba''')
    for d in tqdm(c2.fetchall()):
        if d[0] not in users:
            users[d[0]] = [0, 0]
        if d[1] >= p:
            users[d[0]][0] += 1
        elif d[1] < 1-p:
            users[d[0]][1] += 1
    conn2.close()

    with open(section_two_path + 'users_support.txt','w') as file:
        file.write(json.dumps(users)+'\n')
        
        

def run_community_detect(path):
    #G = nx.read_gpickle(path)
    #to undirect
    #G = G.to_undirected()
    #largest connected component
    #largest_cc = max(nx.connected_components(G), key=len)
    #extract it larggest_cc from G
    #G = G.subgraph(largest_cc).copy()    
    
    #print('Is directed ',G.is_directed())
    #print('Is weighted',nx.is_weighted(G))
    #print('N° nodes',len(G.nodes()))
    
    #gettingadjacency
    #adjacency = nx.to_scipy_sparse_array(G)
    adjacency = sparse.load_npz("/home/matteo/IRA_paper/Final_codes_and_data/expanded_matrix.npz")
    #free some memory
    print('loaded')
    #del G
    
    #performing mutiscale community detection
    for n_scale in [60]:
        for max_scale_ in [1]:
            for min_scale_ in [-0.75]:
                print('n scale: ',n_scale, ',max_scale: ', max_scale_, ',min_scale:', min_scale_)
                if 'expanded' in path:
                    path_ =  section_three_path + f"communnity_det/results_{n_scale}_{str(max_scale_).replace('.',' ')}_{str(abs(min_scale_)).replace('.',' ')}.pkl"
                else:
                    path_ =  section_two_path + f"communnity_det/results_{n_scale}_{str(max_scale_).replace('.',' ')}_{str(abs(min_scale_)).replace('.',' ')}.pkl"                    
                print(path_)
                
                all_results = run(adjacency, min_scale=min_scale_,
                              max_scale = max_scale_, n_scale=n_scale,
                              method='leiden',
                              result_file=path_)
                              #n_workers=4)#constructor='linearized_directed')
                
        
if __name__ == "__main__":
    
    #10/05/2023
    #t0 = time.time()
    #connections_among_IRA()
    #print('Connections between IRA computed in ', time.time()-t0,' seconds')
    
    #10/05/2023
    #t0 = time.time()
    #all_conections()
    #print('Connections per type computed in ', time.time()-t0,' seconds')
    
    #10/05/2023
    #t0 = time.time()
    #ego_edgelist()
    #print('Ego edgelists computed in ', time.time()-t0,' seconds')
    
    #10/05/2023
    #t0 = time.time()
    #build_pickle_graph(default='ego')
    #print('Networks generated in ', time.time()-t0,' seconds')
    
    #10/05/2023
    #t0 = time.time()
    #aggregated_network(default='ego')
    #print('Aggragted network ', time.time()-t0,' seconds')
    
    #11/05/2023
    #t0 = time.time()
    #find_all_voters(p=0.5)
    #print('Supporter files created in ', time.time()-t0,' seconds')
    
    #15/05/2023
    #t0 = time.time()
    #path = section_two_path + "aggregated_ego.gp"
    #run_community_detect(path)
    #print('Mutiscale communiy detection ended in ', time.time()-t0,' seconds')    
    
    #EXPANDED NETWORKS
    
    #16/05/2023
    #t0 = time.time()
    #ego_edgelist(default='expanded')
    #print('Expanded edgelists computed in ', time.time()-t0,' seconds')  
    
    #16/05/2023
    #t0 = time.time()
    #build_pickle_graph(default='expanded')
    #print('Networks generated in ', time.time()-t0,' seconds')
    
    #16/05/2023
    #t0 = time.time()
    #aggregated_network(default='expanded')
    #print('Aggragted network ', time.time()-t0,' seconds')
    
    t0 = time.time()
    path = section_three_path + "expanded_ego.gp"
    run_community_detect(path)
    print('Mutiscale communiy detection ended in ', time.time()-t0,' seconds')    