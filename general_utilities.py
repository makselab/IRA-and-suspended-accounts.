import os
# mode
original_umask = os.umask(0)
mode = 0o777

import networkx as nx
from tqdm import tqdm
import time
import json
import graph_tool.all as gt

        
import sqlite3

from collections import Counter
import glob
import pendulum

from pytwitter import Api 
import numpy as np

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

import CIcython

path='*/Final_codes_and_data/'


#add CI values to graph
def add_CI_to_graph(graph,graph_file=None):
    for direction in [ 'out', 'in']:
        t0 = time.time()
        CIranks, CImap = CIcython.compute_graph_CI(graph, rad=2,
                                                   direction=direction,
                                                   verbose=False)
        graph.vp['CI_' + direction] = graph.new_vertex_property('int64_t', vals=0)
        graph.vp['CI_' + direction].a = CImap
    if graph_file:
        graph.save(graph_file.replace('gp','gt'))
    else:
        return graph

def build_CI_rank(graph ,graph_file  = None):
    rst = {}
    if graph:
        g = graph
    else:
        print(f"------------------{graph_file}------------------")
        g = gt.load_graph(graph_file)
    user_CI = {g.vp.user_id[v]: g.vp.CI_out[v] for v in g.vertices()}
    rst["out_CI"] = user_CI
    st_user_CI = sorted(user_CI.items(), key=lambda d: d[1], reverse=True)
    rank = {d[0]: i + 1 for i, d in enumerate(st_user_CI)}

    rst["out_id"] = st_user_CI
    rst["out_rank"] = rank
    rst["out_top50"] = st_user_CI[:50]

    user_CI = {g.vp.user_id[v]: g.vp.CI_in[v] for v in g.vertices()}
    rst["in_CI"] = user_CI
    st_user_CI = sorted(user_CI.items(), key=lambda d: d[1], reverse=True)
    rank = {d[0]: i + 1 for i, d in enumerate(st_user_CI)}

    rst["in_id"] = st_user_CI
    rst["in_rank"] = rank
    rst["in_top50"] = st_user_CI[:50]
    return rst


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    print("converting ...")
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node, data in list(nxG.nodes(data=True)):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst, data in list(nxG.edges(data=True)):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)  # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node, data in list(nxG.nodes(data=True)):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value  # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in list(nxG.edges(data=True)):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value  # ep is short for edge_properties

    # Done, finally!
    return gtG

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # if isinstance(key, unicode):
    #     # Encode the key as ASCII
    #     key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    # elif isinstance(value, unicode):
    #     tname = 'string'
    #     value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key

def get_get_outliers(data,lower = False):
    
    
    # Calculate Q3 and IQR using numpy
    q1 = np.percentile(list(data.values()), 25)
    q3 = np.percentile(list(data.values()), 75)
    iqr = q3 - q1

    # Calculate the upper limit
    upper_limit = q3 + 1.5*iqr
    
    lower_limit = q1 - 1.5*iqr

    # Identify the upper outliers
    outliers_up = [x for x in data if data[x] > upper_limit]
    if lower:
        
        normal = [x for x in data if (data[x] <= upper_limit and data[x] >= lower_limit)]
        outliers_down = [x for x in data if data[x] < lower_limit]
        return outliers_up,normal,outliers_down
    
    else:
        
        normal = [x for x in data if data[x] <= upper_limit]
        return outliers_up,normal


class get_status:
    def __init__(self):
        self.hist_api = {'API Key':'AyX0Cg7KCxeCD39Gb6qyzNpgF',
                         'API Key Secret':'lzq7RHJzQZ1ObgYTpAmsnd2WXk8swP8jdDocibc5yYheD6baq4',
                         'Bearer Token':'AAAAAAAAAAAAAAAAAAAAAHPzkgEAAAAACxhp1u0fi3adnG86f4j%2FZ6Vl1E8%3DN71bGYjPN1L5SedXbx2CWEVu36BQ3XlbefMeIdiQ22VN9ot2a4'}
        self.path = path+"data/users/"
        #self.users_info = json.load(open(path+'data/users_info.txt'))    
        
    def get_users_info(self,user_name,user_id=False):
        api = Api(bearer_token=self.hist_api['Bearer Token'])
        if user_id:
            user_id=user_name
        else:
            user_id=list(user_info[user_info.username==user_name]['user_id'])[0]
        try: 
            data=api.get_users(ids=user_id,user_fields=['created_at,description,verified,protected,withheld,entities'],
                               return_json=True)
        except Exception as e:
            data=e.message

        return data
    
    @staticmethod
    def account_type(list_):
        res=[]
        if type(list_)==dict and 'data' not in list_:
            if 'User has been suspended:' in list_['detail']:
                res.append([list_['resource_id'], 'Suspended'])
            elif 'Could not find user with id' in list_['detail']:
                res.append([list_['resource_id'], 'Not found']) 
            return res
        if 'data' in list_:
            for dt in list_['data']:
                if dt['verified']:
                    res.append([dt['id'],'Verified'])
                else:
                    res.append([dt['id'],'Not verified'])
        if 'errors' in list_:
            for dt in  list_['errors']:
                if 'User has been suspended:' in dt['detail']:
                    res.append([dt['resource_id'], 'Suspended'])
                elif 'Could not find user with id' in dt['detail']:
                    res.append([dt['resource_id'], 'Not found']) 
        return res

    def user_status(self,ids,c,user_id=True,return_=False):
        users_info={}
        to_analyze=[i for i in ids if (len(i)!=64 and i not in users_info)]
        chunks = [to_analyze[x:x+100] for x in range(0, len(to_analyze), 100)]
        for chunk in tqdm(chunks):
            while True:
                try:
                    accounts=self.account_type(self.get_users_info(chunk,user_id=user_id))
                    for line in accounts:
                        users_info[line[0]]=line[1]
                    break
                except:
                    time.sleep(60*15)
                    continue
                    
        #path_to_save=os.path.join(self.path, f'iteration_{n_iteration}')
        #newfolder=os.makedirs(path_to_save,mode=0o777, exist_ok=True) 
        with open(self.path+f'all_user_accounts_3.txt','w') as file:
            file.write(json.dumps(users_info)+'\n')
        if return_:
            return users_info
        
        

class network_edgelist:
    
    """
    All tweets
    ------------------------------ TABLE tweet_to_quoted_uid ------------------------------
    count： (2712599,)
    tweet_id		771131463383285761
    quoted_uid		457984599
    author_uid		3192506298

    ------------------------------ TABLE tweet_to_replied_uid ------------------------------
    count： (10064719,)
    tweet_id		771131465287528449
    replied_uid		1339835893
    author_uid		754080446078676993

    ------------------------------ TABLE tweet_to_retweeted_uid ------------------------------
    count： (71935828,)
    tweet_id		771131463924199424
    retweeted_uid		1339835893
    author_uid		55956218
    retweet_id		771037992811163648

    ------------------------------ TABLE tweet_to_mentioned_uid ------------------------------
    count： (40694821,)
    tweet_id		771131463345565696
    mentioned_uid		73657657
    author_uid		2429107224
    """
    
    
    """
    Ira tweets
    
    Rep.
    tweet_id,user_id,o_tweet_id,o_user_id
    Ret.
    tweet_id,user_id,o_tweet_id,o_user_id
    Men.
    tweet_id,user_id,o_user_id
    Quo.
    tweet_id,user_id,to_tweet_id,to_user_id
    """
    def __init__(self):
        self.all=["all_connections/all-ret-links.txt",
                  "all_connections/all-men-links.txt",
                  "all_connections/all-rep-links.txt",
                  "all_connections/all-quo-links.txt"]
        
        self.iractivity=["IRA_in_connections/ira-men.txt",
                   "IRA_in_connections/ira-rep.txt",
                   "IRA_in_connections/ira-ret.txt",
                   "IRA_in_connections/ira-quo.txt"]
        
        self.ego=["ego_networks/ego-ret-links.txt",
                  "ego_networks/ego-men-links.txt",
                  "ego_networks/ego-rep-links.txt",
                  "ego_networks/ego-quo-links.txt"]
        
        self.expanded_file=[f"expanded_networks/expanded-ret-links.txt",
                          f"expanded_networks/expanded-men-links.txt",
                          f"expanded_networks/expanded-rep-links.txt",
                          f"expanded_networks/expanded-quo-links.txt"]
        
        
        self.labels= ["fake","extreme bias (right)","right","right leaning",
                      "center","left leaning","left",
                      "extreme bias (left)","local"]#
        
        self.default_ira=set(json.load(open(path+'data/users/IRA_users.txt')))
        self.edgelist_path=path+"data/edgelists/"
        self.networks_path=path+"data/networks/"
        self.users_path=path+"data/users/"
        self.com_path=path+'data/communities/'
        self.get_status=get_status()
        
        
    def expanded(self,c):
        return [f"expanded_networks/C{c}-ret-links.txt",
              f"expanded_networks/C{c}-men-links.txt",
              f"expanded_networks/C{c}-rep-links.txt",
              f"expanded_networks/C{c}-quo-links.txt"]
       
    @staticmethod
    def get_hashtag():
        #tweet_id='737848312003366913'
        DB1_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_db.sqlite"
        DB2_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_sep-nov_db.sqlite"

        hash_={}
        for db in [DB1_NAME,DB2_NAME]:
            conn = sqlite3.connect(db)
            c = conn.cursor() 
            c.execute(f'''SELECT tweet_id,hashtag FROM hashtag_tweet_user''')
            for d in c.fetchall():
                if d:
                    if d[0] not in hash_:
                        hash_[d[0]]=[]
                    hash_[d[0]].append(d[1])
        return hash_
    
    def valid_activity(self,layer='aggregated'):

        if layer =='expanded':
            users_info=json.load(open('/home/matteo/IRA_paper/Final_codes_and_data/data/users/aggregated_user_accounts.txt'))
            IRAs=set([i for i,j in users_info.items() if (len(i)!=64 and users_info[i]=='Suspended')]).union(self.default_ira) 
        else:
            IRAs=self.default_ira
        in_com_tweets=set()
        for file in gg.all:
            for line in tqdm(open(gg.edgelist_path+file)):
                w = line.strip().split()
                if w[1] in IRAs or w[2] in IRAs:
                    if len(w)==3:
                        in_com_tweets.add(w[0])
                    elif len(w)==4:
                        in_com_tweets.add(w[0])
                        in_com_tweets.add(w[3])
        with open(self.users_path+f'valid_tweets_{layer}.txt','w') as file:
            for line in in_com_tweets:
                file.write(line+'\n')

    def find_tweets_by_users(self,netname):
        """
        collecting tweets over time
        """
        DB1_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_db.sqlite"
        DB2_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_sep-nov_db.sqlite"
        conn1 = sqlite3.connect(DB1_NAME)
        conn2 = sqlite3.connect(DB2_NAME)
        c1 = conn1.cursor()
        c2 = conn2.cursor()
        d=[]
        for c in [c1,c2]:
            c.execute('''SELECT tweet_id, user_id, datetime_EST,source_url_id,source_content_id FROM retweeted_status''') 
            d+=c.fetchall()
            c.execute('''SELECT tweet_id, user_id, datetime_EST,source_url_id,source_content_id FROM tweet''')
            d+=c.fetchall()
        c1.close()
        c2.close()
        if netname=='all':
            uids=set([i.strip() for i in open('/home/matteo/IRA_paper/Final_codes_and_data/data/users/complete_GC_uers.txt')])
            with open(self.users_path+f"user_time_{netname}.txt", "w") as f:
                for line in tqdm(d):
                    if str(line[1]) in uids:
                        f.write(f"{line[0]} {line[1]} {line[2]}  {line[3]}  {line[4]}\n")  
            print('\n')
        else:    
            for type_ in netname:
                uids=set(list(nx.read_gpickle(self.networks_path+f'aggregated/{type_}.gp').nodes()))
                for inside in [True, False]:         
                    good_ids=set()
                    in_='out'
                    if inside:
                        in_='in'
                        for line in open(self.users_path+f'valid_tweets_{type_}.txt'):
                            good_ids.add(line.strip())
                    print(len(uids))
                    print(len(d))
                    print(len(good_ids))
                    with open(self.users_path+f"user_time_{type_}_{in_}.txt", "w") as f:
                        for line in d:
                            if inside:
                                if str(line[1]) in uids and str(line[0]) in good_ids:
                                    f.write(f"{line[0]} {line[1]} {line[2]}\n")
                            else:
                                if str(line[1]) in uids:
                                    f.write(f"{line[0]} {line[1]} {line[2]}\n")  
                    print('\n')
       
    @staticmethod
    def get_15min(dt):
        _dt = pendulum.parse(dt)
        t0 = pendulum.parse(_dt.format("YYYY-MM-DD HH:00:00"))
        t1 = t0.add(minutes=15)
        t2 = t0.add(minutes=30)
        t3 = t0.add(minutes=45)

        if t0 <= _dt < t1:
            return t0
        elif _dt < t2:
            return t1
        elif _dt < t3:
            return t2
        else:
            return t3
    
    def get_15min_file(self,netname):
        for type_ in netname:
            #for in_ in ['out','in']:
                with open(self.users_path+f"user_time_15_{type_}.txt", "w") as f:
                    for line in tqdm(open(self.users_path+f"user_time_{type_}.txt")):
                        w = line.strip().split()
                        u = w[1]
                        _dt = w[2] + " " + w[3]
                        _dt = self.get_15min(_dt).to_datetime_string()
                        f.write(f"{w[0]} {w[1]} {_dt} {w[4]} {w[5]}\n")
    
    def search_connections(self,default='ego',partition=None):
        """
        Set c to 'ira' for the ego networrk.
        Set c to the community number for the other case and defaul 'non ego'
        c can take values 0,1,2
        """
        if default=='ego':
            print('Analyzing the ego network...')
            IRAs=self.default_ira
        elif default=='expanded_all':
            print('Analyzing the expanded network...')
            users_info=json.load(open('/home/matteo/IRA_paper/Final_codes_and_data/data/users/aggregated_user_accounts.txt'))
            IRAs=set([i for i,j in users_info.items() if (len(i)!=64 and users_info[i]=='Suspended')]).union(self.default_ira)
        else:
            users_info=json.load(open('/home/matteo/IRA_paper/Final_codes_and_data/data/users/aggregated_user_accounts.txt'))
            communities=json.load(open(self.com_path+partition+'.txt'))
            IRAs=set([i for i,j in communities.items() if (j==f'{default}' and i in users_info and users_info[i]=='Suspended')]).union(self.default_ira)
            print(len(IRAs))
            
        for file in self.all:
            name=file.split('/')[-1]
            if default=='ego':
                out_file=self.edgelist_path+'ego_networks/'+name.replace('all','ego').replace('.txt','.')+'txt'
            elif default=='expanded_all':
                out_file=self.edgelist_path+'expanded_networks/'+name.replace('all','expanded').replace('.txt','.')+'txt'
            else:
                out_file=self.edgelist_path+'expanded_networks/'+name.replace('all',f'C{default}').replace('.txt','.')+'txt'
               
            with open(out_file, "w") as f:
                for line in tqdm(open(self.edgelist_path+file)):
                    w = line.strip().split()
                    if w[1] in IRAs or w[2] in IRAs:
                        f.write(line)
                        
    def get_right_files(self,default):
        files_per_type={}
        for interaction in ['men','ret','rep','quo']:
            files_per_type[interaction]=[]
            if default=='ego':
                files=self.ego
            elif default=='expanded_all':
                files=self.expanded_file
            else:
                files=self.expanded(default)
                
            for file in files:
                if interaction in file:
                    files_per_type[interaction].append(file)
            for file in self.iractivity:
                if interaction in file:
                    files_per_type[interaction].append(file)
        return files_per_type
    
    def build_graph_gp(self,default='ego'):
        int_files=self.get_right_files(default)
        hashtags=[]
        hash_=self.get_hashtag()
        if default=='ego':
            out_file = open(self.users_path+"aggragted_hashtags.txt", "w")
        elif default=='expanded_all':
            out_file = open(self.users_path+"expanded_hashtags.txt", "w")
        else:
            out_file = open(self.users_path+f"C{default}_hashtags.txt", "w")
        for interaction in int_files: 
            print('Building network for IRA', interaction)
            G = nx.DiGraph()
            men_graph = Counter()
            men_rec =set()
            print(int_files[interaction])
            for file in int_files[interaction]:
                
                if file.split('/')[-1].startswith('ira') and interaction=='men':
                    n1_=2
                    n2_=1
                    cntmin=1
                elif file.split('/')[-1] and interaction !='men':
                    n1_=3
                    n2_=1   
                    cntmin=1

                for cnt,line in enumerate(open(self.edgelist_path+file)):
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
                nx.write_gpickle(G, self.networks_path+f"interactions_ego_net/ira-{interaction}.gp")
            elif default=='expanded_all':
                nx.write_gpickle(G, self.networks_path+f"expanded/expanded-{interaction}.gp")
            else:
                nx.write_gpickle(G, self.networks_path+f"expanded/C{default}-{interaction}.gp")
        out_file.close()
                                 
    def aggregated_network(self,default='ego'):
        edge_list=set()
        edge_list_weight={}
        n_=[]
        if default=='ego':
            for interaction in ['men','ret','rep','quo']:
                n_.append(nx.read_gpickle(f"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/interactions_ego_net/ira-{interaction}.gp"))
        elif default=='expanded_all':
            for interaction in ['men','ret','rep','quo']:
                n_.append(nx.read_gpickle(f"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/expanded/expanded-{interaction}.gp"))
        else:
            for interaction in ['men','ret','rep','quo']:
                n_.append(nx.read_gpickle(f"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/expanded/C{default}-{interaction}.gp"))
              
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
            nx.write_gpickle(G,"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/aggregated/aggregated.gp")
        elif default=='expanded_all':
            nx.write_gpickle(G,"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/aggregated/expanded.gp")         
        else:
            nx.write_gpickle(G,f"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/aggregated/C{default}.gp")
            

    def find_ira_in_all(self,iteration=0):
        """
        For this function the iterations must be run sequantially, i.e.,
        forst iteration 0, then 1,2,3 etc
        """
        for category in self.labels:
            if iteration == 0:
                IRA=self.default_ira
            else: 
                path_to_read=os.path.join(self.users_path+f"iteration_{iteration-1}/{category}.txt")
                isExist = os.path.exists(path_to_read)
                if not isExist:
                    print('File not found...')
                    break
                IRA=set([i for i,j in json.load(open(path_to_read)).items() if j=='Suspended']).union(self.default_ira)

            path=self.networks_path+"categories/full/{}.gpickle".format(category)
            G = nx.read_gpickle(path)

            nodes=set()
            for i in G.edges(data=True):
                if i[0] in IRA or i[1] in IRA:
                    nodes.add(i[0])
                    nodes.add(i[1])
            G_=G.subgraph(nodes)
            path_=os.path.join(self.networks_path+'categories/', f'iteration_{iteration}')
            newfolder=os.makedirs(path_,mode=0o777, exist_ok=True) 
            nx.write_gpickle(G_.copy(),path_+f"/{category}.gpickle")
            self.get_status.user_status('iteration'+'_'+category+f"_{iteration}",list(nodes),user_id=True)
    
    def compute_gt_plot(self,net_path,partition=None):
            G = nx.read_gpickle(self.networks_path+f"aggregated/{net_path}.gp")
            if partition:
                communities=json.load(open(self.com_path+partition+'.txt'))
                comm_0=[i for i,j in communities.items() if j=='0']
                comm_1=[i for i,j in communities.items() if j=='1']
                for lst,nm in zip([comm_0,comm_1],['C0','C1']):
                    C=G.subgraph(lst).copy()
                    #compute CI
                    print('\n')
                    print('Computing CI on gt file and saving the file for', path.split('/')[-1])
                    nx.write_gpickle(C,self.networks_path+f"Communities_C_networks/{nm}.gp")
                    C=nx2gt(C)
                    add_CI_to_graph(C,self.networks_path+f"Communities_C_networks/{nm}.gp")
            else:
                    G=nx2gt(G)

                    add_CI_to_graph(G,self.networks_path+f"aggregated/{net_path}.gp")  
                
            
            
def compute_gt_plot():
    #file='/home/matteo/IRA_paper/Final_codes_and_data/data/networks/aggregated/expanded.gp'
    import glob
    for file in glob.glob('/home/matteo/IRA_paper/Final_codes_and_data/data/networks/categories_new/*'):
        print(file)
        g = gt.load_graph(file)
        add_CI_to_graph(g,file)
    #G = nx.read_gpickle(file)
    #communities=json.load(open('/home/matteo/IRA_paper/Final_codes_and_data/data/communities/partition_expresults_60_1_75_37.txt'))
    #val,cnt=np.unique(list(communities.values()),return_counts=True)
    #for com,cnt_ in zip(val,cnt):
    #    if cnt_/len(communities)>0.05:
    #        lst=[i for i,j in communities.items() if j==com]
    #        C=G.subgraph(lst).copy()
    #        #compute CI
    #        print('\n')
    #        print('Computing CI on gt file and saving the file for', com, 'with n° ', len(C.nodes()), ' nodes')
    #        nx.write_gpickle(C,f"/home/matteo/IRA_paper/Final_codes_and_data/data/networks/Communities_C_networks/C{com}_37_expanded.gp")
    #        C=nx2gt(C)
    #        file_=f'/home/matteo/IRA_paper/Final_codes_and_data/data/networks/Communities_C_networks/C{com}_37_expanded.gp'
    #        add_CI_to_graph(C,file_)
        

        


