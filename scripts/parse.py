import os
import numpy as np
import random
files = os.listdir('../train/');
len_of_folders = len(files);
train_csv = []
i=0
for name in files:
	images = os.listdir('../train/'+name);
	if len(images)>1:
		i = i+1
	tmp = []
	tmp.append(name);
	l = len(images)
	if l>1:
		tmp.append(images[0]);
		tmp.append(images[1]);
		tmp.append(1);
		train_csv.append(tmp);
		flg = 1
		while(flg == 1):
			rnd = random.randint(0,len_of_folders-1)
			rand_folder = files[rnd]
			random_files = os.listdir('../train/'+rand_folder)
			if len(random_files)>0:	
				boom = []
				boom.append(rand_folder)
				boom.append(images[0])
				boom.append(random_files[0])
				boom.append(0)
				train_csv.append(boom)
				flg = 0
# print(train_csv)				
train_csv = np.asarray(train_csv)
np.savetxt('../train_similar.csv', train_csv, delimiter=',', fmt='%s');
print(train_csv.shape, i);