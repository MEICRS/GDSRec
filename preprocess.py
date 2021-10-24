# -*- coding: utf-8 -*-
"""
create on 01 Apr, 2020

@author: chenjiajia
"""

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import math

random.seed(1234)

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Epinions', help='dataset name: Ciao/Epinions')
parser.add_argument('--sigma', default='0', help='social strength definition')
parser.add_argument('--test_prop', default=0.2, help='the proportion of data used for test')
args = parser.parse_args()

# load data
if args.dataset == 'Ciao':
	click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
	trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
	click_f = loadmat(workdir + 'Epinions/rating.mat')['rating']
	trust_f = loadmat(workdir + 'Epinions/trustnetwork.mat')['trustnetwork']
else:
	pass 

click_list = []
trust_list = []

u_avg_list = []
u_items_list = []
u_items_divlist = []

u_users_simlist = []
u_users_items_list = []
u_users_items_divlist = []

i_avg_list = []
i_users_list = []
i_users_divlist = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	uid = s[0]   
	iid = s[1]
	if args.dataset == 'Ciao':
		label = s[3]
	elif args.dataset == 'Epinions':
		label = s[3]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	click_list.append([uid, iid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))  
# train, valid and test data split
random.shuffle(pos_list) 
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open(workdir + args.dataset + '/dataset_'+ args.sigma +'.pkl', 'wb') as f: 
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns=['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns=['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis=0, ascending=True, by='uid')  
all_avg = train_df['label'].mean()


for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_ratings = hist['label'].tolist()
	if i_ratings == []:
		i_avg_list.append(all_avg)
	else:
		i_avg_list.append(hist['label'].mean())

for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_ratings = hist['label'].tolist()
	if u_ratings == []:
		u_avg_list.append(all_avg)
	else:
		u_avg_list.append(hist['label'].mean())

for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_divlist.append([(0, 0)])
	else:
		u_items_divlist.append([(iid, round(abs(rating-i_avg_list[iid]))) for iid, rating in zip(u_items, u_ratings)])

for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

train_df = train_df.sort_values(axis=0, ascending=True, by='iid')


for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_divlist.append([(0, 0)])
	else:
		i_users_divlist.append([(uid, round(abs(rating-u_avg_list[uid]))) for uid, rating in zip(i_users, i_ratings)])

for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_list.append([(0, 0)])
	else:
		i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue

	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])  
trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')


u_users_similar = []
for u in tqdm(range(user_count + 1)):
	u_u_similar = []
	u_info = dict(u_items_list[u])
	hist = trust_df[trust_df['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_similar.append([(0, 0)])
		u_users_items_list.append([[(0, 0)]])
		u_users_items_divlist.append([[0, 0]])
	else:
		for user in u_users:
			user_info = dict(u_items_list[user])
			inter_list = list(set(user_info.keys()).intersection(set(u_info.keys())))
			inter_count = len(inter_list)
			for item in inter_list:
				if abs(u_info[item]-user_info[item]) > int(args.sigma):
					inter_count = inter_count - 1
			u_u_similar.append((user, inter_count+1))
		if u_u_similar == []:
			u_users_similar.append([(0, 0)])
			u_users_items_list.append([[(0, 0)]])
			u_users_items_divlist.append([[(0, 0)]])
		else:
			u_users_similar.append(u_u_similar)
			uu_items = []
			uu_items_div = []
			for (uid, inter_count) in u_u_similar:
				uu_items.append(u_items_list[uid])
				uu_items_div.append(u_items_divlist[uid])
			u_users_items_list.append(uu_items)
			u_users_items_divlist.append(uu_items_div)


with open(workdir + args.dataset +'/list_'+ args.sigma +'.pkl', 'wb') as f:
	pickle.dump(u_items_divlist, f, pickle.HIGHEST_PROTOCOL)  
	pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)  
	pickle.dump(u_avg_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_similar, f, pickle.HIGHEST_PROTOCOL)  
	pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)  
	pickle.dump(u_users_items_divlist, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_avg_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)  
	pickle.dump(i_users_divlist, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


