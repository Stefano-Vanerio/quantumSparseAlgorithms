from QOMP import OMP
import numpy as np
from sklearn.preprocessing import normalize

np.random.seed(0)
size = 300
q_err = 0.00
L_thresh = size-1
error_type = ''
SCI_bound = 1

# TRAIN SET
train = np.loadtxt('./N-grams_experiments/X12_train.txt')
labels_train = np.loadtxt('./N-grams_experiments/y12_train.txt')

good_train = list()
bad_train = list()

for i in range(0, len(labels_train)):
    if labels_train[i] == 0:
        good_train.append(train[i])
    else:
        bad_train.append(train[i])

good_train = np.array(good_train)
bad_train = np.array(bad_train)
lower = bad_train.shape[0]
upper = lower + good_train.shape[0]

print(lower)
print(upper)

D = np.concatenate((bad_train.transpose(), good_train.transpose()), axis=1)
D = normalize(D, axis=0, norm='l2')

# TEST AND VALIDATION SET
test = np.loadtxt('./N-grams_experiments/X12_test.txt')
labels_test = np.loadtxt('./N-grams_experiments/y12_test.txt')

good = 0
bad=0

for i in range(0, len(labels_test)):
    if labels_test[i] == 0:
        good = good+1
    else:
        bad = bad+1
print(good)
print(bad)

num_samples = test.shape[0]

k = []
SCI = []
xc = 0
kc = 0
correct_pred = 0
total = 0
not_known = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for i in range(0, num_samples, 1):
    total += 1
    elem = test[i]/np.linalg.norm(test[i])
    tolerance = np.linalg.norm(elem)/1.5
    xc, kc = OMP(elem, D, tolerance, error_type, L_thresh, error=q_err, info='it')

    k.append(kc)

    x_ben = np.zeros(xc.shape)
    x_mal = np.zeros(xc.shape)

    for j in range(int(lower), int(upper), 1):
        x_ben[j] = xc[j]
    for j in range(0, int(lower), 1):
        x_mal[j] = xc[j]

    res_mal = D @ x_mal
    res_ben = D @ x_ben

    if np.linalg.norm(elem-res_mal) < np.linalg.norm(elem-res_ben):
        check = 1
    else:
        check = 0

    loc_SCI = (2 * max(np.linalg.norm(x_mal, 1), np.linalg.norm(x_ben, 1)) / np.linalg.norm(xc, 1)) - 1

    if loc_SCI < SCI_bound:
        not_known += 1
        total -= 1
    else:
        if labels_test[i] == check:
            correct_pred += 1
            if check == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if check == 1:
                false_pos += 1
            else:
                false_neg += 1

    SCI.append(loc_SCI)
    print(i)

acc = correct_pred/total
pre = true_pos/(true_pos+false_pos)
rec = true_pos/(true_pos+false_neg)
f1 = 2 * pre * rec / (pre+rec)
mean_SCI = np.mean(np.array(SCI))
not_class = not_known/num_samples

print('Here is reported some data about the run:')
print('Accuracy: {}%'.format(acc*100.00))
print('Precision: {}%'.format(pre*100.00))
print('Recall: {}%'.format(rec*100.00))
print('F1 score: {}%'.format(f1*100.00))
print('Mean SCI parameter: {}%'.format(mean_SCI*100.00))
print('Excluded samples: {}%'.format(not_class*100.00))
print('The experiment was runned with ' + str(SCI_bound) + ' SCI bound, ' + str(q_err) + ' error and ' + str(size) + ' atoms in the dictionary')

rsl = np.array([acc, pre, rec, f1, mean_SCI, not_class])
filename = 'res_'+str(SCI_bound)+'_'+str(q_err)+'_'+str(size)+'.txt'
np.savetxt(filename, rsl)

