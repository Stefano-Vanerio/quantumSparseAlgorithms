from QOMP import OMP
import numpy as np
from sklearn.preprocessing import normalize

np.random.seed(1234)
size = 300
q_err = 0.0
error_type = ''
k = 5
lam =1.0

# TRAIN SET
data_1 = np.loadtxt('./N-grams_experiments/X12_train.txt')
labels_data_1 = np.loadtxt('./N-grams_experiments/y12_train.txt')

good_samples = list()
bad_samples = list()

for i in range(0, len(labels_data_1)):
    if labels_data_1[i] == 0:
        good_samples.append(data_1[i])
    else:
        bad_samples.append(data_1[i])

print(len(good_samples))
print(len(bad_samples))
good_samples = np.array(good_samples)
bad_samples = np.array(bad_samples)
np.random.shuffle(good_samples)
np.random.shuffle(bad_samples)
dimension = (good_samples.shape[0]-30+bad_samples.shape[0]-1)//k ##34

print(dimension)
count = 0
count_good = 0
count_bad = 0
folds = list()

for j in range(0, k, 1):
    fold = list()

    for i in range(0, dimension, 1):
        if i < dimension/2:
            fold.append(np.array(good_samples[count_good]))
            count_good += 1
        else:
            fold.append(np.array(bad_samples[count_bad]))
            count_bad += 1
    folds.append(fold)

print(len(folds))

index = np.array(range(0, k, 1))
indexes = list()
indexes.append(index)

for i in range(0, k-1, 1):
    index = np.roll(index, -1)
    indexes.append(index)

par_1_list = []
par_2_list = []
acc_list = []
f1_list = []
pre_list = []
rec_list = []
sci_list = []
exc_list = []
count_add = 0

for id in indexes:
    test = folds[id[0]]
    validation = folds[id[1]]
    train = list()
    for s in range(2, k, 1):
        train = train + folds[id[s]]

    ordered_train = list()
    for i in range(0, k-2, 1):
        ordered_train = ordered_train + train[(int(i*dimension)):(int(i*dimension+dimension/2))]
    for i in range(0, k-2, 1):
        ordered_train = ordered_train + train[(int(i*dimension+dimension/2)):(int((i+1)*dimension))]

    D = np.array(ordered_train).transpose()
    D = normalize(D, axis=0, norm='l2')


    max_parameter = 0

    for p in range(0, 21, 1): # from 0.00 to 1.00 with step 0.05 (0,21)
        SCI_bound = 0.05 * p
        tol = 1.5
        for h in range(0, 10, 1): # from (1/2) to (1/10) with step (1/(x+0.5)) (4,21)

            SCI = []
            correct_pred = 0
            total = 0
            not_known = 0
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0

            for i in range(0, len(validation), 1):
                total += 1
                elem = np.array(validation[i]) / np.linalg.norm(np.array(validation[i]))
                tolerance = np.linalg.norm(elem)/tol
                xc, kc = OMP(elem, D, tolerance, error_type, (size-1), error=0.00, info='it')

                x_ben = np.zeros(xc.shape)
                x_mal = np.zeros(xc.shape)

                for j in range(0, int(len(train)/2), 1):
                    x_ben[j] = xc[j]
                for j in range(int(len(train)/2), len(train), 1):
                    x_mal[j] = xc[j]

                res_mal = D @ x_mal
                res_ben = D @ x_ben

                if np.linalg.norm(elem - res_mal) < np.linalg.norm(elem - res_ben):
                    check = 1 # malware
                else:
                    check = 0 # software

                loc_SCI = (2 * max(np.linalg.norm(x_mal, 1), np.linalg.norm(x_ben, 1)) / np.linalg.norm(xc, 1)) - 1

                if loc_SCI < SCI_bound:
                    not_known += 1
                    total -= 1
                else:
                    if (not (i < dimension/2)) == check:
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

            acc = correct_pred / total
            pre = true_pos / (true_pos + false_pos)
            rec = true_pos / (true_pos + false_neg)
            f1 = 2 * pre * rec / (pre + rec)
            not_class = not_known / len(validation)

            parameter = f1 * lam + (1-lam) * (1 - not_class)

            if parameter >= max_parameter:
                max_f1 = f1
                best_not = not_class
                best_tol = tol
                best_SCI = SCI_bound
                max_parameter = parameter

            tol = 0.5 + tol
    print('Fold '+str(id)+': f1='+str(max_f1 * 100) + ' tol=' + '1/' + str(best_tol) + ' exc=' + str(best_not * 100) + ' sci =' + str(best_SCI))

    SCI = []
    correct_pred = 0
    total = 0
    not_known = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    # TEST
    print(len(test))
    for i in range(0, len(test), 1):
        total += 1
        elem = np.array(test[i]) / np.linalg.norm(np.array(test[i]))
        tolerance = np.linalg.norm(elem)/ best_tol
        xc, kc = OMP(elem, D, tolerance, error_type, size-1, error=q_err, info='it')
        x_ben = np.zeros(xc.shape)
        x_mal = np.zeros(xc.shape)

        for j in range(0, int(len(train) / 2), 1):
            x_ben[j] = xc[j]
        for j in range(int(len(train) / 2), len(train), 1):
            x_mal[j] = xc[j]

        res_mal = D @ x_mal
        res_ben = D @ x_ben

        if np.linalg.norm(elem - res_mal) < np.linalg.norm(elem - res_ben):
            check = 1
        else:
            check = 0

        loc_SCI = (2 * max(np.linalg.norm(x_mal, 1), np.linalg.norm(x_ben, 1)) / np.linalg.norm(xc, 1)) - 1

        if loc_SCI < best_SCI:
            not_known += 1
            total -= 1
        else:
            if (not (i < dimension/2)) == check:
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

    if total==0:
        acc=0
    else:
        acc = correct_pred / total
    if (true_pos + false_pos) == 0:
        pre = 0
    else:
        pre = true_pos / (true_pos + false_pos)
    if (true_pos + false_neg) == 0:
        rec = 0
    else:
        rec = true_pos / (true_pos + false_neg)
    if (pre + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * pre * rec / (pre + rec)
    mean_SCI = np.mean(np.array(SCI))
    not_class = not_known / len(test)

    print('Here is reported some data about the test of '+str(id)+' fold:')
    print('Accuracy: {}%'.format(acc * 100.00))
    print('Precision: {}%'.format(pre * 100.00))
    print('Recall: {}%'.format(rec * 100.00))
    print('F1 score: {}%'.format(f1 * 100.00))
    print('Mean SCI parameter: {}%'.format(mean_SCI * 100.00))
    print('Excluded samples: {}%'.format(not_class * 100.00))
    print('The experiment was runned with ' + str(best_SCI) + ' SCI bound, ' + str(q_err) + ' error and 1/' +str(best_tol) + ' tolerance for QOMP\n')

    acc_list.append(acc)
    f1_list.append(f1)
    pre_list.append(pre)
    rec_list.append(rec)
    sci_list.append(mean_SCI)
    exc_list.append(not_class)
    par_1_list.append(best_SCI)
    par_2_list.append(best_tol)

full = [acc_list, pre_list, rec_list, f1_list, sci_list, exc_list, par_1_list, par_2_list]
rsl = [np.mean(acc_list), np.mean(pre_list), np.mean(rec_list), np.mean(f1_list),  np.mean(sci_list), np.mean(exc_list),  np.mean(par_1_list), np.mean(par_2_list)]
dev = [np.std(acc_list), np.std(pre_list), np.std(rec_list), np.std(f1_list),  np.std(sci_list), np.std(exc_list), np.std(par_1_list), np.std(par_2_list)]
print('Here is reported some data about the overall test ')
print('Accuracy: {}%'.format(rsl[0] * 100.00))
print('Precision: {}%'.format(rsl[1] * 100.00))
print('Recall: {}%'.format(rsl[2] * 100.00))
print('F1 score: {}%'.format(rsl[3] * 100.00))
print('Mean SCI parameter: {}%'.format(rsl[4] * 100.00))
print('Excluded samples: {}%'.format(rsl[5] * 100.00))

filename = 'full_'+str(k)+'_'+str(q_err)+'_'+str(lam)+'_folds.txt'
np.savetxt(filename, full)

filename = 'mean_'+str(k)+'_'+str(q_err)+'_'+str(lam)+'_folds.txt'
np.savetxt(filename, rsl)

filename = 'std_'+str(k)+'_'+str(q_err)+'_'+str(lam)+'_folds.txt'
np.savetxt(filename, dev)





