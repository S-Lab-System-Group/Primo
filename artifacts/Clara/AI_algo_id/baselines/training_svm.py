import warnings

warnings.filterwarnings("ignore")
import pickle
from sklearn import svm

with open("../embedding.pickle", "rb") as f:
    data = pickle.load(f)
source_reformatted_train = data[0]
target_reformatted_train = data[1]
source_reformatted_ptest = data[2]
target_reformatted_ptest = data[3]
source_reformatted_ntest = data[4]
target_reformatted_ntest = data[5]
click_dict = data[6]
# Create a svm Classifier
neigh = svm.SVC(kernel="linear")  # Linear Kernel kernel='linear'
neigh.fit(source_reformatted_train, target_reformatted_train)

summation_n = 0
total_n = 0
for item in source_reformatted_ntest[:]:
    total_n += 1
    if neigh.predict([item]) == [1]:
        # print(neigh.predict([item]))
        summation_n += 1
# print(summation_n)
summation_p = 0
total_p = 0
for item in source_reformatted_ptest[:]:
    total_p += 1
    if neigh.predict([item]) == [1]:
        summation_p += 1

for key in click_dict:
    item = click_dict[key][2]
    if neigh.predict([item]) == [1]:
        print(key[: key.find("-")], "crc hash accelerating opportunity found!!!!")
# print(summation, total)
print("Precision: %.3f" % (float(summation_p) / (summation_n + summation_p)))
print("Recall: %.3f" % (float(summation_p) / total_p))

"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
with open("SPE.pickle",'rb') as f:
    data = pickle.load(f)

source_reformatted = data[0]
target_reformatted = data[1]

pca = PCA(n_components=2)
X = source_reformatted[:]
pca.fit(X)
print("PCA: {}".format(pca.explained_variance_ratio_))
#print(pca.singular_values_)
X_new = pca.transform(X)

positive = []
negative = []
pos_set = set()
neg_set = set()
positive_t = []
negative_t = []
pos_set_t = set()
neg_set_t = set()

for index in range(-75,0):
    #print(X_new[index], target_reformatted[index], neigh.predict([source_reformatted[index]]))
    if neigh.predict([source_reformatted[index]])[0] == 1:
        positive.append([X_new[index][0], X_new[index][1]])
        pos_set.add((X_new[index][0], X_new[index][1]))
    else:
        negative.append([X_new[index][0], X_new[index][1]])
        neg_set.add((X_new[index][0], X_new[index][1]))

for index in range(-14000,-8000):
    if neigh.predict([source_reformatted[index]])[0] == 1:
        positive_t.append([X_new[index][0], X_new[index][1]])
        pos_set_t.add((X_new[index][0], X_new[index][1]))
    else:
        negative_t.append([X_new[index][0], X_new[index][1]])
        neg_set_t.add((X_new[index][0], X_new[index][1]))
        
        
POS = np.array(positive)
NEG = np.array(negative)
POS_T = np.array(positive_t)
NEG_T = np.array(negative_t)

plt.scatter(POS[:, 0], POS[:, 1],marker='o')
plt.scatter(NEG[:, 0], NEG[:, 1],marker='v')
try:
    plt.scatter(POS_T[:, 0], POS_T[:, 1],marker='>')
except:
    pass
plt.scatter(NEG_T[:, 0], NEG_T[:, 1],marker='^')

x1 = [-1, 0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]
y1 = [-0.62 * item + 1.98 for item in x1]

for index in range(len(x1)):
    print(x1[index], y1[index])
plt.plot(x1,y1,'r--',label='type1')
#plt.show()
#plt.savefig('pca.png')
"""
# for item in neg_set:
# print(item[0], item[1])
