import json
from tqdm import tqdm
import numpy as np
import pandas as pd
# from SQLite_handler import find_tweet

class Who_is_fake(object):
    def __init__(self):
        self.NEW_HOST_1 = {}
        # for k, v in json.load(open("data/sources.json")).items():
        #     hostname = k.lower()
        #     _type = v["type"]
        #     if _type in ["fake", "conspiracy", "hate"]:
        #         self.NEW_HOST_1[hostname] = "FAKE"
        #     elif _type == "bias":
        #         self.NEW_HOST_1[hostname] = "BIAS"

        # self.NEW_HOST_2 = {k.lower(): v for k, v in json.load(open("data/mbfc_host_label.json")).items()}
        self.NEW_HOST_2 = {k.lower(): v for k, v in json.load(open("data/mbfc_dict.json")).items()}
        self.NEW_HOST_3 = json.load(open("data/fake_dict_science.json"))
        self.NEW_HOST_4 = json.load(open("data/align_dict_science.json"))

        alex_c = json.load(open("data/alex_category.json"))
        ira_c = json.load(open("data/ira_category.json"))
        self.HOST = {**alex_c, **ira_c}

        for ht, color in self.NEW_HOST_3.items():
            if ht not in self.HOST and color == "Black":
                self.HOST[ht] = "fake"


    def identify(self, ht):
        ht = ht.lower()

        if ht in self.HOST:
            return self.HOST[ht]
        else:
            return -1


    def identify_v2(self, ht):
        ht = ht.lower()
        # if ht in self.NEW_HOST_1:
        #     labels.append(self.NEW_HOST_1[ht])
        # else:
        #     labels.append("GOOD")

        if ht in self.NEW_HOST_2:
            bias = self.NEW_HOST_2[ht].lower()
            if bias in ["fake-news", "conspiracy", "satire"]:
                bias = "questionable sources"
            elif bias == "right-center":
                bias = "right leaning"
            elif bias == "leftcenter":
                bias = "left leaning"
            elif bias == "pro-science":
                bias = "-1"
            # fact = self.NEW_HOST_2[ht][1].lower()
            return bias
        else:
            return "-1"

    def identify_science_fake(self, ht):
        ht = ht.lower()
        # if ht in self.NEW_HOST_1:
        #     labels.append(self.NEW_HOST_1[ht])
        # else:
        #     labels.append("GOOD")

        if ht in self.NEW_HOST_3:
            color = self.NEW_HOST_3[ht]
            return color
        else:
            return "-1"

    def identify_science_align(self, ht):
        ht = ht.lower()
        # if ht in self.NEW_HOST_1:
        #     labels.append(self.NEW_HOST_1[ht])
        # else:
        #     labels.append("GOOD")

        if ht in self.NEW_HOST_4:
            score = self.NEW_HOST_4[ht]
            return score
        else:
            return -1

    def is_fake(self, ht):
        if self.identify(ht)[0] == "FAKE":
            return True
        else:
            return False


class IRA_identify(object):

    def __init__(self):
        self._map = json.load(open("data/IRA_map.json"))
        # self.IRA_users_before_set = pd.read_csv("data/ira_users_csv_hashed.csv", usecols=["userid"], dtype=str)["userid"]
        self.IRA_user_set = set(json.load(open("data/IRA_user_list.json"))) # all IRA (匿名或非匿名) included

    def uncover(self, uid):
        if uid in self._map:
            uid = str(self._map[uid])
        return uid

    def check(self, ht):
        return ht in self.IRA_user_set

    def find_IRA_tweets(self):
        with open("data/IRA-tweets-in-SQLite-v2.json", "w") as f:
            for line in tqdm(open("data/IRA-tweets-in-SQLite.json")):
                tid, uid = line.strip().split(",")
                real_uid = str(find_tweet(tid)["user_id"])
                f.write("{},{},{}\n".format(tid, real_uid, uid))

    def find_IRA_retweets(self):
        with open("data/IRA-retweets-in-SQLite-v2.json", "w") as f:
            for line in tqdm(open("data/IRA-retweets-in-SQLite.json")):
                tid, uid = line.strip().split(",")
                real_uid = str(find_tweet(tid)["user_id"])
                f.write("{},{},{}\n".format(tid, real_uid, uid))


    def cal_IRA_map(self):
        data = []
        for line in open("data/IRA-(re)tweets-in-SQLite.json"):
            d = json.loads(line.strip())
            data.append(d)

        IRA_map = {}
        for d in data:
            if len(d["IRA_userid"]) == 64:
                IRA_map[str(d["IRA_userid"])] = str(d["user_id"])


        IRA_user_list = []
        data_ira_users = pd.read_csv("data/ira_users_csv_hashed.csv", usecols=["userid"], dtype=str)
        for _, row in tqdm(data_ira_users.iterrows()):
            uid = row["userid"]
            IRA_user_list.append(uid)
            if uid in IRA_map:
                IRA_user_list.append(IRA_map[uid])

        json.dump(IRA_user_list, open("data/IRA_user_list.json", "w"), ensure_ascii=False, indent=2)
        json.dump(IRA_map, open("data/IRA_map.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # who = Who_is_fake()
    # print(who.identify("baidu.com"))
    putin = Are_you_IRA()
    # putin.find_IRA_retweets()
    # putin.find_IRA_tweets()
    # putin.cal_IRA_map()
    # print(putin._map)

    tmp = []
    who = Who_is_fake()
    for k in who.NEW_HOST_3:
        if k in who.HOST:
            print(k, who.HOST[k])
            tmp.append(who.HOST[k])
    tmp = pd.Series(tmp)
    print(tmp.value_counts())