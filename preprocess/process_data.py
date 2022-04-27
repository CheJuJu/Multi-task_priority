
from datetime import datetime
import json
import os
import markdown
from bs4 import BeautifulSoup
import re
import nltk
from tqdm import tqdm
import random
from gensim.models import word2vec
import logging
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
import a
import csv
# stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
files_path = []
key_value = {}
word = []
def get_file_path(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            files_path.append(os.path.join(root, file))
def text_clean(text):
    # 去掉超链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
    text = text.replace("\"\"","'").replace("\"","'")
    text = text.replace("\\","/")
    # 去掉多余的空格
    text = re.sub(r'\s\s+', ' ', text)
    text = text.lstrip(' ')
    sentence = a.tokenizeRawTweetText(text)

    tmp = []
    for sent in sentence:
        tokens = a.tokenize(sent)
        filtered_words = [w.strip() for w in tokens]
        tmp.extend(filtered_words)
    final = ' '.join(tmp)
    return final
def tongji_label(dataset_name):
    df1 = pd.DataFrame(columns = ['1', '2', '3'])
    for file_path in tqdm(files_path):
        file = open(file_path)
        str1 = file.read()
        str_json = json.loads(str1)
        for label in str_json["labels"]:
            if label["name"].find("P") >= 0 :
                if label["name"] == "P3: When Possible" :
                    for temp in str_json["labels"]:
                        if temp["name"] not in df1.index.values.tolist():
                            df1.loc[temp["name"],'3'] = 1
                        elif np.isnan(df1.loc[temp["name"],'3']):
                            df1.loc[temp["name"],'3'] = 1
                        else :
                            df1.loc[temp["name"],'3'] += 1
                elif label["name"] == "P2: Soon":
                    for temp in str_json["labels"]:
                        if temp["name"] not in df1.index.values.tolist():
                            df1.loc[temp["name"],'2'] = 1
                        elif np.isnan(df1.loc[temp["name"],'2']):
                            df1.loc[temp["name"],'2'] = 1
                        else :
                            df1.loc[temp["name"],'2'] += 1
                elif label["name"] == "P1: High Priority" or label["name"] == "P0: Drop Everything":
                    for temp in str_json["labels"]:
                        if temp["name"] not in df1.index.values.tolist():
                            df1.loc[temp["name"],'1'] = 1
                        elif np.isnan(df1.loc[temp["name"],'1']):
                            df1.loc[temp["name"],'1'] = 1
                        else :
                            df1.loc[temp["name"],'1'] += 1
    if not os.path.exists(dataset_name+"\data"):  # 如果路径不存在
        os.makedirs(dataset_name+"\data")
    df1.to_csv(dataset_name+"\data\\"+dataset_name+'_label.csv')       
                        
def get_value(dataset_name):    #提取带有优先级的问题报告
    a1 = 0
    a2 = 0
    a3 = 0
    def process_body(str_json):
        aa = str_json['body']
        if aa == None:
            return ""
        aa = markdown.markdown(aa, extensions=['markdown.extensions.toc', 'markdown.extensions.fenced_code',
                                               'markdown.extensions.tables'])
        soup = BeautifulSoup(aa, "lxml")
        # print(soup)
        for code in soup.findAll("code"):
            code.string = " : " + str(code.string) +' . '
        for a in soup.findAll("a"):
            a.string = "<URL>"
        return soup.get_text(" ", strip=True)
    author = []
    time = []
    text = []
    pro = []
    kind =[]
    for file_path in tqdm(files_path):
        file = open(file_path)
        str1 = file.read()
        str_json = json.loads(str1)
        for label in str_json["labels"]:
            
            if label["name"]== "P3: When Possible" or label["name"] == "P2: Soon" or label["name"] == "P1: High Priority" or label["name"] == "P0: Drop Everything":
                kindtemp = []
                for temp in str_json["labels"]:
                        if temp["name"] == 'Type: Bug':
                            kindtemp.append(0)
                        elif temp['name'] =='Type: Feature Request':
                            kindtemp.append(1)
                        elif temp['name'] == 'Type: Discussion/Question' or temp['name'] == 'Related to: Documentation':
                            kindtemp.append(2)
                if len(kindtemp) == 0:
                    break
                # print(label["name"],str_json["title"])
                temp = process_body(str_json)
                if(len(temp)>1700):
                    break
                au = str_json["user"]["login"]
                de = str_json["title"]
                ti = str_json["created_at"]
                author.append(au)
                ti = datetime.strptime(ti, '%Y-%m-%dT%H:%M:%SZ')
                time.append(ti)
                s =de+' . '+temp
                s= text_clean(s)
                text.append(s)
                
                if label["name"] == "P3: When Possible" :
                    key_value = 2
                    pro.append(key_value)
                    a3 = a3 + 1
                    if 2 in kindtemp:
                        kind.append(2)
                    elif 1 in kindtemp:
                        kind.append(1)
                    else:
                        kind.append(0)
                elif label["name"] == "P2: Soon":
                    
                    key_value = 1
                    pro.append(key_value)
                    a2 = a2 + 1
                    if 2 in kindtemp:
                        kind.append(2)
                    elif 1 in kindtemp:
                        kind.append(1)
                    else:
                        kind.append(0)
                elif label["name"] == "P1: High Priority" or label["name"] == "P0: Drop Everything":
                    key_value = 0
                    pro.append(key_value)
                    a1 = a1 + 1
                    if 0 in kindtemp:
                        kind.append(0)
                    elif 2 in kindtemp:
                        kind.append(2)
                    else:
                        kind.append(1)
                continue
    if not os.path.exists(dataset_name+"\data\\"):  # 如果路径不存在
        os.makedirs(dataset_name+"\data\\")
    print(len(author),len(time),len(text),len(pro),len(kind))
    dic = {"author":author,"creat_time":time,"text":text,"priority":pro,'kind':kind}
    df = pd.DataFrame(dic)
    df.text.to_csv(dataset_name+"\data\\"+dataset_name+'_text.txt',header =0 ,index = 0,quoting =csv.QUOTE_ALL)
    df.to_csv(dataset_name+"\data\\"+dataset_name+'_raw.csv',quoting =csv.QUOTE_ALL)
    print(len(pro))
    print(a1,a2,a3)
    return [dataset_name+"\data\\"+dataset_name+'_raw.csv',dataset_name+"\data\\"+dataset_name+'_text.txt']

def get_feature_file(dataset_name,file_path):
    df = pd.read_csv(file_path, usecols=['author','creat_time','text','priority','kind'],encoding='utf-8')
    df['creat_time']= pd.to_datetime(df['creat_time'])
    df = df.sort_values(by="creat_time" , ascending=True)
    df=df.reset_index(drop=True)
    def temporal1():
        df['temporal1'] = 0
        for index, row in tqdm(df.iterrows()):
            # temp = row['creat_time']# 输出每行的索引值
            # for index1,row1 in df.iterrows():
            #     if (temp - row1['creat_time']) <= pd.Timedelta(days=1) and (temp - row1['creat_time']).total_seconds()>0:
            #         df.loc[index,'temporal1'] +=1   
            previous = df['creat_time'] < row['creat_time']
            x_range = df['creat_time'] >= row['creat_time'] - pd.Timedelta(days=1)
            count = sum(previous & x_range)  
            df.loc[index,'temporal1'] =count  
    def temporal3():
        df['temporal3'] = 0
        for index, row in tqdm(df.iterrows()):
            previous = df['creat_time'] < row['creat_time']
            x_range = df['creat_time'] >= row['creat_time'] - pd.Timedelta(days=3)
            count = sum(previous & x_range)  
            df.loc[index,'temporal3'] =count  
    def temporal7():
        df['temporal7'] = 0
        for index, row in tqdm(df.iterrows()):
            previous = df['creat_time'] < row['creat_time']
            x_range = df['creat_time'] >= row['creat_time'] - pd.Timedelta(days=7)
            count = sum(previous & x_range)  
            df.loc[index,'temporal7'] =count  
    def temporal30():
        df['temporal30'] = 0
        for index, row in tqdm(df.iterrows()):
            previous = df['creat_time'] < row['creat_time']
            x_range = df['creat_time'] >= row['creat_time'] - pd.Timedelta(days=30)
            count = sum(previous & x_range)  
            df.loc[index,'temporal30'] =count 
    def author():
        df['authormean'] = 0
        df['authorcnt'] = 0
        df['authormedian']  = 0
        for index, row in tqdm(df.iterrows()):
            re = []
            temptime = row['creat_time']
            temp = row["author"]
            tempdf = (df.loc[:,'creat_time']<temptime) &(df.loc[:,'author']==temp)
            # for index1,row1 in df.iterrows():
            #     if temptime > row1['creat_time'] and temp == row1["author"]:
            #         re.append(int(row1['priority']))
            tempdf = df.loc[tempdf,'priority']
            re = tempdf.tolist()
            df.loc[index,'authorcnt'] = len(re)
            if len(re) != 0:
                df.loc[index,'authormean'] = sum(re) / len(re)
            if len(re) != 0:
                re = sorted(re)
                df.loc[index,'authormedian'] = np.median(re)
    def related_report():
        df['top1mean'] = 0
        df['top1median'] = 0
        df['top5mean'] = 0
        df['top5median'] = 0
        df['top10mean'] = 0
        df['top10median'] = 0
        df['top20mean'] = 0
        df['top20median'] = 0
        vec = []
        pri = {}
        wtv = word2vec.Word2Vec.load(dataset_name+"/embedding_model/"+dataset_name+"_w2v.model")
        def cos_sim(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a,b)/(a_norm * b_norm)
            return cos
        cntt = 0
        for index, row in df.iterrows(): 
            re = np.zeros((300))
            a = df.loc[index,'text']
            temp = a.split(" ")
            
            for i in temp:
                try:
                    re+=wtv[i]
                except KeyError:
                    cntt+=1
                    print ("not in vocabulary:{} {}".format(cntt,i))
            re/=len(temp)
            vec.append(re)
            pri[index] = df.loc[index,'priority']
        
        vec = cosine_similarity(vec)
        for index, row in tqdm(df.iterrows()):
            tempvec = vec[index]
            vecser = pd.Series(tempvec)
            del vecser[index]
            l=vecser.sort_values(ascending = False,inplace = False) 
            top1=[]
            top1_cnt = 1
            top3=[]
            top3_cnt = 3
            top5=[]
            top5_cnt = 5
            top10=[]
            top10_cnt = 10
            top20=[]
            top20_cnt = 20
            cnt = 0
            for key,value in l.items():
                top1.append(pri[key])
                cnt+=1
                if(cnt>=top1_cnt):
                    break
            df.loc[index,'top1mean'] = top1[0]
            df.loc[index,'top1median'] = top1[0]

            cnt = 0
            for key,value in l.items():
                top3.append(pri[key])
                cnt+=1
                if(cnt>=top3_cnt):
                    break
            top3 = sorted(top3)
            df.loc[index,'top3mean'] = sum(top3) / len(top3)
            df.loc[index,'top3median'] = np.median(top3)
             
            cnt = 0
            for key,value in l.items():
                top5.append(pri[key])
                cnt+=1
                if(cnt>=top5_cnt):
                    break
            top5 = sorted(top5)
            df.loc[index,'top5mean'] = sum(top5) / len(top5)
            df.loc[index,'top5median'] = np.median(top5)

            cnt = 0
            for key,value in l.items():
                top10.append(pri[key])
                cnt+=1
                if(cnt>=top10_cnt):
                    break
            top10 = sorted(top10)
            df.loc[index,'top10mean'] = sum(top10) / len(top10)
            df.loc[index,'top10median'] = np.median(top10)

            cnt = 0
            for key,value in l.items():
                top20.append(pri[key])
                cnt+=1
                if(cnt>=top20_cnt):
                    break
            top20 = sorted(top20)
            df.loc[index,'top20mean'] = sum(top20) / len(top20)
            df.loc[index,'top20median'] = np.median(top20)
            

    related_report()
    author()        
    temporal1()
    temporal3()
    temporal7()
    temporal30()
    
    
    del df['author']
    del df['creat_time']
    df_p = df.priority
    df_t = df.text
    df_k = df.kind
    del df['priority']
    del df['text']
    del df['kind']
    df.insert(17,'text',df_t)
    df.insert(18,'priority',df_p)
    df.insert(19,'kind',df_k)
    df = shuffle(df)
    fe = open(dataset_name+"\data\\"+dataset_name+'_features.txt', 'w', encoding='utf-8')
    for index, row in df.iterrows():
        line = str(row[0])
        for i in range(1,20):
            if i== 17 or i == 18 or i == 19:
                line+='\t'+str(row[i])
            else :
                line+=' '+str(row[i])
        fe.write(line)   
        fe.write('\n')
    return [dataset_name+"\data\\"+dataset_name+'_features.txt',dataset_name+"\data\\"+dataset_name+'_text.txt']

    
def get_train_test_dev(file_path,dataset_name):
    file = open(file_path[0],encoding='UTF-8')
    cnt = len(file.readlines())
    print(cnt)
    trainc = int(cnt * 0.8)
    devc = trainc + int(0.1 * cnt)
    train = open(dataset_name+"/data/"+"train.txt", 'w', encoding='utf-8')
    dev = open(dataset_name+"/data/"+"/dev.txt", 'w', encoding='utf-8')
    test = open(dataset_name+"/data/"+"/test.txt", 'w', encoding='utf-8')

    temp = 0
    file2 = open(file_path[0],encoding='UTF-8')
    for l in file2.readlines():
        if temp in range(0, trainc):
            train.write(l)
        elif temp in range(trainc, devc):
            dev.write(l)
        else:
            test.write(l)
        temp += 1

def get_w2vmodel(file_path,dataset_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentence = word2vec.Text8Corpus(file_path[1])
    file = open(file_path[1],encoding='UTF-8')
    sentences = []
    cnt = 0
    for l in file.readlines():
        print(cnt)
        l = eval(l)
        l = l.split(' ')
        sentences.append(l)
        cnt+=1
    model = word2vec.Word2Vec(sentences=sentences, size=300, window=5, min_count=1, workers=4)
    if not os.path.exists(dataset_name+"\embedding_model"):  # 如果路径不存在
        os.makedirs(dataset_name+"\embedding_model")
    model.save(dataset_name+"/embedding_model/"+dataset_name+"_w2v.model")
def tongji(path,path1):
    # file = open(path,encoding='UTF-8')
    # max = 0
    # min = 700
    # sum = 0
    # cnt = 0
    # for l in file.readlines():
    #     cnt+=1
    #     l = eval(l)
    #     l = l.split(' ')
    #     leng = len(l)
    #     if(leng>max):
    #         max = leng
    #     if(leng<min):
    #         min = leng
    #     sum+=leng
    # print(min,max,sum/cnt)
    df = pd.read_csv(path1,delimiter="\t",names=["cc", "text", "pri", "kind"])
    print(df['pri'].value_counts())
    print(df['kind'].value_counts())

if __name__ =="__main__":
    # get_file_path("D:\桌面\学习\科研\标签优先级\数据\\github_issue_new\\ampproject/amphtml")
    dataset ="zephyr"
    # tongji_label(dataset)
    # path = get_value(dataset)
    # get_w2vmodel([dataset+"\data\\"+dataset+'_raw.csv',dataset+"\data\\"+dataset+'_text.txt'],dataset)
    # raw_path = get_feature_file(dataset,dataset+"\data\\"+dataset+'_raw.csv')
    get_train_test_dev([dataset+"\data\\"+dataset+'_features.txt',dataset+"\data\\"+dataset+'_text.txt'],dataset)
    tongji(dataset+"\data\\"+dataset+'_text.txt',dataset+"\data\\"+dataset+'_features.txt')