import sqlite3
import time
import json
import pandas as pd

gran_path = '/sdf/MatteoPaper/'
section1_data_path = gran_path+'section_one/'

urls_db_file = '/sdf/IRA/urls_db.sqlite'
DB1_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_db.sqlite"
DB2_NAME = "/disk2/US2016_alex/complete_trump_vs_hillary_sep-nov_db.sqlite"


start_date = '2016-06-01'
end_date = '2016-11-09'

t0=time.time()

#saving the tweets info for section 1
sql_condition_dict = {}
for media_type in ['fake', 'far_right', 'right', 'lean_right', 'center', 'lean_left', 'left','far_left']:
    
    sql_condition_dict = """SELECT hostname FROM hosts_{med_type}_rev_stat
                                        WHERE perccum > 0.01""".format(med_type=media_type)
                                                                                                           
                                        
    with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
        c = conn.cursor()
        c.execute("""SELECT tweet_id, user_id, datetime_EST, source_content FROM urls
                     WHERE final_hostname IN
                     (""" + '\nUNION\n'.join([sql_condition_dict]) + ")")

        columns = [col[0] for col in c.description]
        news_num_tweets = c.fetchall()
        pd.DataFrame(news_num_tweets,columns=columns).to_pickle(section1_data_path+f'{media_type}.pkl')
     
        print('Tweets mapping for',media_type,' finished in ', time.time()-t0,' seconds')
        t0=time.time()
