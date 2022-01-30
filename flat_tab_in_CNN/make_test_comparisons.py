import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import midi2tab_utils as m2t
# from sklearn.metrics import precision_recall_fscore_support

# load guitar data
with open('..' + os.sep + 'data' + os.sep + 'flat_tablature_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)
x_test = d['x_test'].T
y_test = d['y_test'].T
# load augmented/random data
with open('..' + os.sep + 'data' + os.sep + 'flat_tablature_rand_dataset.pickle', 'rb') as handle:
    d = pickle.load(handle)
x_rand_test = d['x_test'].T
y_rand_test = d['y_test'].T

# load guitar model
model = keras.models.load_model( '../models/tab_flat_CNN_out/tab_flat_CNN_out_current_best.hdf5' )
# load augmented model
model_rand = keras.models.load_model( '../models/tab_rand_flat_CNN_out/tab_rand_flat_CNN_out_current_best.hdf5' )

# keep some random samples from test data
idxs2keep = 5000
idx = np.random.permutation(len(x_test))
x_test = x_test[ idx[:idxs2keep] ]
y_test = y_test[ idx[:idxs2keep] ]
idx = np.random.permutation(len(x_rand_test))
x_rand_test = x_rand_test[ idx[:idxs2keep] ]
y_rand_test = y_rand_test[ idx[:idxs2keep] ]

print('len(x_test): ', len(x_test))
print('len(x_rand_test): ', len(x_rand_test))

print('x_test[0]: ', x_test[0])
print('x_rand_test[0]: ', x_rand_test[0])

def get_matches( x, y ):
    nomatch = 0
    partial = 0
    match = 0
    x = np.array(x)
    y = np.array(y)
    if np.all( x*y == y ):
        match = 1
    else:
        if np.any( x * y ):
            partial = 1
        else:
            nomatch = 1
    return nomatch, partial, match
# end get_matches

# simple model - simple data
x_in = x_test
y_true = y_test
simple_model_simple_data = {
    'nomatch':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'partial':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'match':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0}
}
for i in range(len(x_in)):
    print('simple_model_simple_data: ', i)
    # predict
    y_pred = model.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    n, p, m = get_matches(y_true[i], decision_bin)
    key = max( np.sum(decision_bin) , np.sum(y_true[i]) )
    simple_model_simple_data['nomatch'][key] += n
    simple_model_simple_data['partial'][key] += p
    simple_model_simple_data['match'][key] += m
    # print('========================================================')
    # print('y_true[i]: ', y_true[i])
    # print('decision_bin: ', decision_bin)
    # print('n, p, m: ', n, p, m)
    # print('========================================================')

# simple model - augmented data
x_in = x_rand_test
y_true = y_rand_test
simple_model_aug_data = {
    'nomatch':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'partial':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'match':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0}
}
for i in range(len(x_in)):
    print('simple_model_aug_data: ', i)
    # predict
    y_pred = model.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    n, p, m = get_matches(y_true[i], decision_bin)
    key = max( np.sum(decision_bin) , np.sum(y_true[i]) )
    simple_model_aug_data['nomatch'][key] += n
    simple_model_aug_data['partial'][key] += p
    simple_model_aug_data['match'][key] += m

# aug model - simple data
x_in = x_test
y_true = y_test
aug_model_simple_data = {
    'nomatch':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'partial':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'match':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0}
}
for i in range(len(x_in)):
    print('aug_model_simple_data: ', i)
    # predict
    y_pred = model_rand.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    n, p, m = get_matches(y_true[i], decision_bin)
    key = max( np.sum(decision_bin) , np.sum(y_true[i]) )
    aug_model_simple_data['nomatch'][key] += n
    aug_model_simple_data['partial'][key] += p
    aug_model_simple_data['match'][key] += m

# aug model - augmented data
x_in = x_rand_test
y_true = y_rand_test
aug_model_aug_data = {
    'nomatch':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'partial':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0},
    'match':{1:0, 2:0, 3:0 ,4:0, 5:0, 6:0}
}
for i in range(len(x_in)):
    print('aug_model_aug_data: ', i)
    # predict
    y_pred = model_rand.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    n, p, m = get_matches(y_true[i], decision_bin)
    key = max( np.sum(decision_bin) , np.sum(y_true[i]) )
    aug_model_aug_data['nomatch'][key] += n
    aug_model_aug_data['partial'][key] += p
    aug_model_aug_data['match'][key] += m

original_stdout = sys.stdout
with open('comparison_results.txt', 'w') as f:
    sys.stdout = f
    print('\n')
    print('simple_model_simple_data:')
    comparison = simple_model_simple_data
    print('=======================================================')
    for i in range(6):
        if i == 0:
            print('n. pitchs:  &\t', end='')
        print(str(i+1) + ' &\t', end='')
    print('sum') 
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['nomatch'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('nomatch:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['partial'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('partial:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('match:   &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    for i in range(6):
        mtrc = comparison['nomatch'][i+1] + comparison['partial'][i+1] + comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('sum:     &\t', end='')
        print(str(mtrc) + ' &\t', end='')
    print('')

    print('\n')
    print('simple_model_aug_data:')
    comparison = simple_model_aug_data
    print('=======================================================')
    for i in range(6):
        if i == 0:
            print('n. pitchs:  &\t', end='')
        print(str(i+1) + ' &\t', end='')
    print('sum') 
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['nomatch'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('nomatch:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['partial'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('partial:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('match:   &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    for i in range(6):
        mtrc = comparison['nomatch'][i+1] + comparison['partial'][i+1] + comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('sum:     &\t', end='')
        print(str(mtrc) + ' &\t', end='')
    print('')

    print('\n')
    print('aug_model_simple_data:')
    comparison = aug_model_simple_data
    print('=======================================================')
    for i in range(6):
        if i == 0:
            print('n. pitchs:  &\t', end='')
        print(str(i+1) + ' &\t', end='')
    print('sum') 
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['nomatch'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('nomatch:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['partial'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('partial:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('match:   &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    for i in range(6):
        mtrc = comparison['nomatch'][i+1] + comparison['partial'][i+1] + comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('sum:     &\t', end='')
        print(str(mtrc) + ' &\t', end='')
    print('')

    print('\n')
    print('aug_model_aug_data:')
    comparison = aug_model_aug_data
    print('=======================================================')
    for i in range(6):
        if i == 0:
            print('n. pitchs:  &\t', end='')
        print(str(i+1) + ' &\t', end='')
    print('sum') 
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['nomatch'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('nomatch:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['partial'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('partial:  &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    tmp_sum = 0
    for i in range(6):
        mtrc = comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('match:   &\t', end='')
        print(str(mtrc) + ' &\t', end='')
        tmp_sum += mtrc
    print(tmp_sum)
    for i in range(6):
        mtrc = comparison['nomatch'][i+1] + comparison['partial'][i+1] + comparison['match'][i+1]
        mtrc = np.round(mtrc/idxs2keep*100)/100
        if i == 0:
            print('sum:     &\t', end='')
        print(str(mtrc) + ' &\t', end='')
    print('')

sys.stdout = original_stdout

os.makedirs( '../data/results/', exist_ok=True )
comparison_results = {
    'simple_model_simple_data': simple_model_simple_data,
    'simple_model_aug_data': simple_model_aug_data,
    'aug_model_simple_data': aug_model_simple_data,
    'aug_model_aug_data': aug_model_aug_data,
}
with open('../data/results/' + os.sep + 'comparison_results.pickle', 'wb') as handle:
    pickle.dump(comparison_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
# simple model - simple data
x_in = x_test
y_true = y_test
simple_model_simple_data = {'precision':[], 'recall':[], 'fscore':[]}
for i in range(len(x_in)):
    print('simple_model_simple_data: ', i)
    # predict
    y_pred = model.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true[i], decision_bin, average='binary')
    simple_model_simple_data['precision'].append( precision )
    simple_model_simple_data['recall'].append( recall )
    simple_model_simple_data['fscore'].append( fscore )

# simple model - augmented data
x_in = x_rand_test
y_true = y_rand_test
simple_model_aug_data = {'precision':[], 'recall':[], 'fscore':[]}
for i in range(len(x_in)):
    print('simple_model_aug_data: ', i)
    # predict
    y_pred = model.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true[i], decision_bin, average='binary')
    simple_model_aug_data['precision'].append( precision )
    simple_model_aug_data['recall'].append( recall )
    simple_model_aug_data['fscore'].append( fscore )

# aug model - simple data
x_in = x_test
y_true = y_test
aug_model_simple_data = {'precision':[], 'recall':[], 'fscore':[]}
for i in range(len(x_in)):
    print('aug_model_simple_data: ', i)
    # predict
    y_pred = model_rand.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true[i], decision_bin, average='binary')
    aug_model_simple_data['precision'].append( precision )
    aug_model_simple_data['recall'].append( recall )
    aug_model_simple_data['fscore'].append( fscore )

# aug model - augmented data
x_in = x_rand_test
y_true = y_rand_test
aug_model_aug_data = {'precision':[], 'recall':[], 'fscore':[]}
for i in range(len(x_in)):
    print('aug_model_aug_data: ', i)
    # predict
    y_pred = model_rand.predict( x_in )
    midi = x_in[i, :128]
    decision = m2t.midi_and_flat_tab_decision(midi, y_pred[0])
    decision_bin = np.reshape( decision.astype(bool), y_true[i].shape )
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true[i], decision_bin, average='binary')
    aug_model_aug_data['precision'].append( precision )
    aug_model_aug_data['recall'].append( recall )
    aug_model_aug_data['fscore'].append( fscore )

print('\n')
print('simple_model_simple_data:')
comparison = simple_model_simple_data
mtrc = comparison['precision']
print('precision: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['recall']
print('recall: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['fscore']
print('fscore: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')

print('\n')
print('simple_model_aug_data:')
comparison = simple_model_aug_data
mtrc = comparison['precision']
print('precision: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['recall']
print('recall: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['fscore']
print('fscore: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')

print('\n')
print('aug_model_simple_data:')
comparison = aug_model_simple_data
mtrc = comparison['precision']
print('precision: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['recall']
print('recall: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['fscore']
print('fscore: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')

print('\n')
print('aug_model_aug_data:')
comparison = aug_model_aug_data
mtrc = comparison['precision']
print('precision: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['recall']
print('recall: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
mtrc = comparison['fscore']
print('fscore: ' + str(np.mean(mtrc)) + ' (' + str(np.std(mtrc)) + ')')
'''
