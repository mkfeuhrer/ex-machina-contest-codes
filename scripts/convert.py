import pandas as pd
import numpy as np

test = pd.read_csv("../new_test_1.csv")
new_test = []
for ind,row in test.iterrows():
	tmp = []
	tmp.append(row['image1'])
	tmp.append(row['image2'])
	if row['l1'] > 430:
		tmp.append(0)
	else: 
		tmp.append(1)
	new_test.append(tmp)

new_test = np.asarray(new_test)
np.savetxt("../final3.csv",new_test,delimiter=',',fmt="%s")