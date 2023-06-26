import networkx as nx
import json
import time
from datetime import timedelta, datetime
import pandas as pd
import pendulum
import numpy as np
from tqdm import tqdm
import sqlite3
import pickle
from collections import Counter
import glob
from statsmodels.tsa.seasonal import STL

import graph_tool.all as gt

from general_utilities import get_get_outliers


#TO DO: improve this part
import sys
sys.path.append('/home/matteo/IRA_paper/fake_news_during_election/')


from fake_identify import Are_you_IRA

Putin = Are_you_IRA()



urls_db_file = '/sdf/IRA/urls_db.sqlite'
DB1_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_db.sqlite"
DB2_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_sep-nov_db.sqlite"


gran_path = '/sdf/MatteoPaper/'
section_four_path = gran_path + 'section_four/'
section1_data_path = gran_path + 'section_one/' 
section_two_path = gran_path + 'section_two/'
raw_data_path = gran_path+'rawdata/'

resample_freq = '15min'


periods = ['until_sep','after_sep','IRA']


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

start_date = '2016-06-01'
end_date = '2016-11-09'

type_cnt = ['df_tid_count_off_cli']#'df_tid_count

type_cnt_names = ['OfficialClient']#'AllClient'


groups = [ ['IRA','S','WT','WC','ST','SC','U'],
           ['IandS','WT','WC','ST','SC','U'] ]

class_type = 'classes_2'
timeseries = 'timeseries2'
residuals = 'residuals2'
#user acconts info 
users_account = json.load(open(section1_data_path+'users_accounts.txt'))

    
def compute_res(dfrm):
    tsts = dfrm.copy()
    resid = pd.DataFrame(index=tsts.index)
    seasonal = pd.DataFrame(index=tsts.index)
    trend = pd.DataFrame(index=tsts.index)
    print("Loaded!")
    for col in tsts:
        print(col)
        tsts[col] = tsts[col].fillna(0)
        stl =  STL(tsts[col].values,
                      period=96,
                      seasonal=95,
                      trend=None,
                      low_pass=None,
                      seasonal_deg=1,
                      low_pass_deg=1,
                      trend_deg=1,
                      robust=True
                     ).fit(inner_iter=2, outer_iter=5)
        resid[col]=stl.resid
        trend[col]=stl.seasonal
        seasonal[col]=stl.trend
    return resid,seasonal,trend

def get_day_start_end(ts, cuthour=4, cutmin=0, cutsec=0):
    # get 4am before:
    if ts.hour >= 4:
        start = ts - timedelta(hours=ts.hour-cuthour,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
        end = ts - timedelta(hours=ts.hour-cuthour-24,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
    else:
        start = ts - timedelta(days=1,
                               hours=ts.hour-cuthour,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
        end = ts - timedelta(days=1,
                               hours=ts.hour-cuthour-24,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
    return (start, end)


def load_should_remove():
    should_remove_15Min = []

    for line in open("/sdf/IRA/russian_trolls/data/should_be_removed_in_timeseries.txt"):
        _dt = line.strip()
        _start = datetime.strptime(_dt, '%Y-%m-%d %H:%M:%S')
        for _dt in pd.date_range(start=_start, periods=4 * 24, freq="15Min"):
            should_remove_15Min.append(pd.to_datetime(_dt))

    return should_remove_15Min


def find_all_voters(p=0.5):
    users = {}
    conn1 = sqlite3.connect(DB1_NAME)
    c1 = conn1.cursor()
    c1.execute('''SELECT user_id, p_pro_hillary_anti_trump FROM class_proba''')
    for d in tqdm(c1.fetchall()):
        if str(d[0]) not in users:
            users[str(d[0])] = [0, 0]
        if d[1] >= p:
            users[str(d[0])][0] += 1
        elif d[1] < 1-p:
            users[str(d[0])][1] += 1
    conn1.close()

    conn2 = sqlite3.connect(DB2_NAME)
    c2 = conn2.cursor()
    c2.execute('''SELECT user_id, p_pro_hillary_anti_trump FROM class_proba''')
    for d in tqdm(c2.fetchall()):
        if str(d[0]) not in users:
            users[str(d[0])] = [0, 0]
        if d[1] >= p:
            users[str(d[0])][0] += 1
        elif d[1] < 1-p:
            users[str(d[0])][1] += 1
    conn2.close()

    return users

def removekey(d, key):
    del d[key]
    return d

def build_users_classes(p = 0.5):
    
    supporters_info = find_all_voters(p)
    
    user_strump2 = {}
    for user in supporters_info:
        S = supporters_info[user][1] - supporters_info[user][0]
        user_strump2[user] = S
    
    G = nx.read_gpickle(section_two_path + 'aggregated_ego.gp')
    
    IRA = set([i for i in G.nodes() if (Putin.fuck(i) and i in user_strump2)])
    
    Suspended = set([i for  i in G.nodes() if (i in users_account 
                                                and users_account[i] == 'Suspended'
                                                and Putin.fuck(i)==False
                                                and i in user_strump2)])
    
    
    if class_type == 'classes_2':
        
        temp = set(list(users_account.keys())) 
        cnt = 0
        exclude = set()
        for user in user_strump2:
            if ( user in temp and  users_account[user] in ['Suspended','Not found'] ):
                exclude.add(user)
                cnt += 1 
        print('removed ', cnt, ' accounts')
    else:
        exclude = set()
    
    Tr = {i:user_strump2[i] for i in user_strump2 if  
          (user_strump2[i]>0 
           and i not in IRA 
           and i not in Suspended
           and i not in exclude)}
    
    Cl = {i:abs(user_strump2[i]) for i in user_strump2 if 
          ( user_strump2[i]<0 
           and i not in IRA and 
           i not in Suspended
           and i not in exclude)}
    
    U = [i for i in user_strump2 if 
         ( user_strump2[i]==0 
          and i not in IRA 
          and i not in Suspended
          and i not in exclude)]
    
    for lower in [False,True]:
        
        if  not lower:

            ST,WT = get_get_outliers(Tr)
            SC,WC = get_get_outliers(Cl)

        else:

            ST,WT,UT = get_get_outliers(Tr, lower = lower)
            SC,WC,UC = get_get_outliers(Cl, lower = lower)
            U = np.concatenate([U,UT,UC])

        users_classes = {}

        for users, class_ in zip([IRA,Suspended,ST,WT,SC,WC,U],['IRA','S','ST','WT','SC','WC','U']):
            for user in users: 
                users_classes[user] = class_

        with open(section_four_path +f"{class_type}/user_classes{str(p).replace('0.','_')}_lower{str(lower)}.txt",'w') as file:
            file.write(json.dumps(users_classes)+'\n')

def clients():
    
    mapping={}

    conn1 = sqlite3.connect(DB1_NAME)
    c1 = conn1.cursor()
    c1.execute('''SELECT * FROM source_content''') 
    d = c1.fetchall()
    c1.close()
    
    mapping[DB1_NAME]={i[0]:i[1] for i in d}

    conn2 = sqlite3.connect(DB2_NAME)
    c2 = conn2.cursor()
    c2.execute('''SELECT * FROM source_content''') 
    d = c2.fetchall()
    c2.close()
    
    mapping[DB2_NAME]={i[0]:i[1] for i in d}
        
    return mapping

def get_user_activity_supporters():
    
    mapping = clients()
    
    for db_file,filename in zip([DB1_NAME,DB2_NAME],['until_sep','after_sep']):
        print(db_file+'\n')
        
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        
        d=[]
        #c.execute("""SELECT tweet_id, user_id, datetime_EST,source_content_id FROM retweeted_status""")
        #d += c.fetchall()
        
        c.execute("""SELECT tweet_id, user_id, datetime_EST,source_content_id FROM tweet""")
        d += c.fetchall()
        
        c.close()
        
        for cnt,i in enumerate(d):
            
            d[cnt] = (i[0],i[1],i[2],mapping[db_file][i[3]])


        pd.DataFrame(d,columns=['tweet_id', 'user_id', 'datetime_EST','source_content']).to_csv(section_four_path+f'all_{filename}.csv',index=False)        
        
def IRA_activity():
    
    all_ira = pd.read_csv(raw_data_path+'ira_tweets_csv_hashed.csv')
    all_ira_e = all_ira[all_ira['tweet_language']=='en']
    print('Tweets in English: ',len(all_ira_e))

    all_ira_e = all_ira_e[(all_ira_e['tweet_time']>start_date) & (all_ira_e['tweet_time']<end_date)]
    
    all_ira_e = all_ira_e[['tweetid', 'userid','tweet_time','tweet_client_name']]

    all_ira_e.rename(columns={'tweetid':'tweet_id', 'userid':'user_id',
                             'tweet_client_name': 'source_content',
                             'tweet_time': 'datetime_EST'}, inplace=True)

    to_est = [pendulum.parse(i).add(hours=-4).to_datetime_string() for i in all_ira_e['datetime_EST']]
    
    all_ira_e['datetime_EST'] = to_est
    
    all_ira_e.to_csv(section_four_path + 'all_IRA.csv',index=False)
    
def get_df_counts_supporters(df, filename=None, resample_freq='D',official_twitter_clients=official_twitter_clients):
    
    # count daily tweets
    df_tid_count = df.resample(resample_freq, on='datetime_EST').tweet_id.nunique()
    
    # only official clients
    df_tid_count_off_cli = df.loc[df.source_content.isin(official_twitter_clients)].resample(resample_freq, on='datetime_EST').tweet_id.nunique()
    
    # count daily tweets
    df_uid_count = df.resample(resample_freq, on='datetime_EST').user_id.nunique()
    
    # only official clients
    df_uid_count_off_cli = df.loc[df.source_content.isin(official_twitter_clients)].resample(resample_freq, on='datetime_EST').user_id.nunique()
    
    # global measures
    df_global = pd.DataFrame(data = {'num_tweet' : df.tweet_id.nunique(),
                                     'num_users' : df.user_id.nunique(),
                                     'num_tweet_off_cli' : df.loc[df.source_content.isin(official_twitter_clients)].tweet_id.nunique(),
                                     'num_users_off_cli' : df.loc[df.source_content.isin(official_twitter_clients)].user_id.nunique()
                                     },
                            index = [1])
    
    res = {'df_tid_count':df_tid_count,
            'df_uid_count' : df_uid_count,
            'df_tid_count_off_cli': df_tid_count_off_cli,
            'df_uid_count_off_cli': df_uid_count_off_cli,
            'df_global':df_global}
    
    if filename is not None:
        #save
        with open(filename, 'wb') as fopen:
            pickle.dump(res, fopen)
    else:
        return res
    
def compute_classes_supporters_activity(p=0.5,rp=0,clients=official_twitter_clients,remove_top_inf=False,method='extreme'):
    
    clients_ = 'clients_standard'
    
    for lower in [False]:#True
        
        if method == 'extreme':

            supporters = json.load(open(section_four_path +f"{class_type}/user_classes{str(p).replace('0.','_')}_lower{str(lower)}.txt"))

        else:
            #TO do: change
            method = 'bovet_paper'
            supporters = compute_support(p=p,rp=2)

        if remove_top_inf:

            #TO do: change
            top_inf='top_inf_NO'
            users_CG_all=set([i[0] for i in get_top_inf(top=100)])

        else:

            top_inf = 'top_inf_YES'
            users_CG_all = set()

        #upload activity data
        df=[]
        for filename in periods:

            df.append(pd.read_csv(section_four_path+f'all_{filename}.csv',dtype=str))

        df = pd.concat(df)

        supp_index = np.unique(list(supporters.values()))

        for supp_type in supp_index:

            node_list = [i for i in supporters if
                         (len(i)!=64 
                          and supporters[i]==supp_type 
                          and i not in users_CG_all)]

            temp_ = df[df['user_id'].isin(node_list)]

            temp_['datetime_EST'] = pd.to_datetime(temp_['datetime_EST'])

            get_df_counts_supporters(temp_,
                                     filename = section_four_path + f"{timeseries}/" +method+'_'+supp_type+'_'+clients_+'_rp'+str(rp)+'_'+top_inf+str(p).replace('0.','_')+f"_lower{str(lower)}"'.pickle',
                                     resample_freq = resample_freq,
                                     official_twitter_clients = clients)

        #add I + S
        node_list = [i for i in supporters 
                     if (len(i)!=64 
                     and i not in users_CG_all 
                     and (supporters[i]=='IRA' 
                     or  supporters[i]=='S'))]

        temp_ = df[df['user_id'].isin(node_list)]

        temp_['datetime_EST'] = pd.to_datetime(temp_['datetime_EST'])

        get_df_counts_supporters(temp_,
                                 filename = section_four_path + f"{timeseries}/"  +method+'_'+'IandS'+'_'+clients_+'_rp'+str(rp)+'_'+top_inf+str(p).replace('0.','_')+f"_lower{str(lower)}"+'.pickle',
                                 resample_freq = resample_freq,
                                 official_twitter_clients = clients)        

def prepare_for_tigramite_categories(p):
    
    should_remove_15Min = load_should_remove()
    
    for group,gr in zip(groups,['IvsS', 'IandS']):
        
        for cnt,clnt in zip(type_cnt,type_cnt_names):
            
                print('+++++++++',gr,resample_freq,clnt)
               
                df_num_tweets = pd.DataFrame()
                for type_ in group:
                    
                    read_path = section_four_path + f"{timeseries}/extreme_{type_}_clients_standard_rp0_top_inf_YES_{p}_lowerFalse.pickle"
                    
                    print(read_path)
                    
                    if type_ in ['WT','WC','ST','SC','U']:
                        
                        df_num_tweets[type_]=pd.read_pickle(read_path)[cnt]
                    
                    else:
                        
                        df_num_tweets[type_]=pd.read_pickle(read_path)['df_tid_count']
                        
                df_num_tweets.index = pd.to_datetime(df_num_tweets.index)

                if gr == 'IvsS':
                    
                    df_tot_tweets = df_num_tweets.WT + df_num_tweets.WC + df_num_tweets.ST+ df_num_tweets.SC+ df_num_tweets.U+ df_num_tweets.IRA+ df_num_tweets.S 
                    
                else:
                    
                    df_tot_tweets = df_num_tweets.WT + df_num_tweets.WC + df_num_tweets.ST+ df_num_tweets.SC+df_num_tweets.U+df_num_tweets.IandS
    

                df_tot_tweets.index = pd.to_datetime(df_tot_tweets.index)
                
                #remove from Zhenkun
                df_num_tweets = df_num_tweets[~df_num_tweets.index.isin(should_remove_15Min)]
                df_tot_tweets = df_tot_tweets[~df_tot_tweets.index.isin(should_remove_15Min)]
                
                #% resample  
                df_num_tweets = df_num_tweets.resample(resample_freq).sum()
                df_tot_tweets = df_tot_tweets.resample(resample_freq).sum()

                #prepare data
                df = pd.DataFrame(columns=['tot'], data=df_tot_tweets.copy())
                for key in df_num_tweets.columns:
                    df[key] = df_num_tweets[key].copy()

                # drop datetimes at both ends
                start_day = datetime(2016,6,1)
                stop_day = datetime(2016,11,9)


                df.drop(df.loc[np.logical_or(df.index < start_day, 
                                             df.index >= stop_day)].index,
                                            inplace=True)            

                # replace null values by nan
                for col in df.columns:
                    df[col].loc[df['tot'].isna()] = np.nan


                # deal with missing values:
                all_time_index = df.index
                missing_ts = df.loc[df['tot'].isna()].index


                # mask with values to remove
                ts_mask = np.zeros_like(all_time_index, dtype=bool)
                for ts in missing_ts:
                    start, end = get_day_start_end(ts)
                    ts_mask = np.logical_or(ts_mask,
                                            np.logical_and(
                                                    all_time_index >= start,
                                                    all_time_index < end)
                                            )

                df.loc[ts_mask] = np.nan

                #%% remove nan values:
                dfdrop = df.dropna()

                dfdrop.drop(dfdrop.loc[np.logical_or(dfdrop.index < start_day, 
                                             dfdrop.index >= stop_day)].index,
                                            inplace=True)     



                #%% LOESS smoothing
                resid,seasonal,trend = compute_res(dfdrop)

                filename_residuals = section_four_path  + f"{residuals}/UsersCat_" + resample_freq + '_' + clnt + '_' + gr +  f"_residuals_{p}_rp"+'.pickle'

                filename_seasonal =  section_four_path  + f"{residuals}/UsersCat_" + resample_freq + '_' + clnt + '_' + gr + f"_seasonal_{p}_rp"+'.pickle'

                filename_trend =     section_four_path  + f"{residuals}/UsersCat_" + resample_freq + '_' + clnt + '_' + gr + f"_trend_{p}_rp"+'.pickle'     

                resid.to_pickle(filename_residuals)
                seasonal.to_pickle(filename_seasonal)
                trend.to_pickle(filename_trend)
                

                
if __name__ == "__main__":
    
    #17/05/2023
    #t0 = time.time()
    #for p in [0.5,0.6,0.7]: 
    #    build_users_classes(p = p)
    #print('Users classes\' file generated in ', time.time()-t0,' seconds')    
    
    #17/05/2023
    #t0 = time.time()
    #get_user_activity_supporters()
    #IRA_activity()
    #print('Users supports file generated in ', time.time()-t0,' seconds') 
    
    #17/05/2023
    #t0 = time.time()
    #for p in [0.5,0.6,0.7]:
    #    compute_classes_supporters_activity(p)
    #print('Classes activity files generated in ', time.time()-t0,' seconds') 
    
    #18/05/2023
    #t0 = time.time()
    #for p in [5,6,7]:
    #    prepare_for_tigramite_categories(p)
    #print('Time series prepared for tigramite in ', time.time()-t0,' seconds') 
        
