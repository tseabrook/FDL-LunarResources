#Written by Yarin Gal
#yg279@cam.ac.uk
#Contributions by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import os, time
import numpy as np
from data_loader import load_data, create_dataset, shuffle, split
from model import get_neon_set, fit_model, test_model
from neon.models import Model

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC')
NACEquatorialDir = os.path.join(NACDir, 'Equatorial')
NACPolarDir = os.path.join(NACDir, 'South_Pole')

log_name = os.path.join(os.getcwd(), time.strftime('exp_%Y%m%d_%H%M%S') + '.log')

def log(text):
    print(text)
    with open(log_name, 'a') as f:
        f.write(text + '\n')


log('Creating log file ' + log_name)

#LROC NAC: Lunar Reconnaissance Orbiter Camera - Narrow Angled Camera
eq_pos_train = load_data(os.path.join(NACEquatorialDir,'Crater'), end_row=1008)
eq_pos_test = load_data(os.path.join(NACEquatorialDir,'Crater'), start_row=1024, is_augment=False)
eq_neg_train = load_data(os.path.join(NACEquatorialDir,'Not Crater'), end_row=1008)
eq_neg_test = load_data(os.path.join(NACEquatorialDir,'Not Crater'), start_row=1024, is_augment=False)
pol_pos_train = load_data(os.path.join(NACPolarDir,'Crater'), end_row=1008)
pol_pos_test = load_data(os.path.join(NACPolarDir,'Crater'), start_row=1024, is_augment=False)
pol_neg_train = load_data(os.path.join(NACPolarDir,'Not Crater'), end_row=1008)
pol_neg_test = load_data(os.path.join(NACPolarDir,'Not Crater'), start_row=1024, is_augment=False)
empty_test = np.zeros((0, 32*32))

print 'Test / Train ratios:'
print 1. * eq_pos_test.shape[0] / (eq_pos_train.shape[0]//4 + eq_pos_test.shape[0])
print 1. * eq_neg_test.shape[0] / (eq_neg_train.shape[0]//4 + eq_neg_test.shape[0])
print 1. * pol_pos_test.shape[0] / (pol_pos_train.shape[0]//4 + pol_pos_test.shape[0])
print 1. * pol_neg_test.shape[0] / (pol_neg_train.shape[0]//4 + pol_neg_test.shape[0])

# each eval_pair is a tuple of train set tuple and test sets list:
#    (
#      (name, epochs, train_pos, train_neg),
#      [
#        (name, test_pos1, test_neg1),
#        (name, test_pos2, test_neg2)
#      ]
#    )
# Num epochs found via search wrt val accuracy
eval_pairs = [
              (
               ('baseline equatorial train set', 25, eq_pos_train, eq_neg_train),
               [
                ('eq->eq (train on equatorial and test on equatorial)', eq_pos_test, eq_neg_test),
#                 ('eq->pol1 (train on equatorial and test on poles (with eq negs))', pol_pos_test, eq_neg_test),
#                 ('eq->pol2 (train on equatorial and test on poles (with no negs))', pol_pos_test, empty_test),
                ('eq->pol3 (train on equatorial and test on poles)', pol_pos_test, pol_neg_test)
               ]
              ),
              (
               ('experiment polar train set', 15, pol_pos_train, pol_neg_train),
               [
#                 ('pol->pol1 (train on poles and test on poles (with eq negs))', pol_pos_test, eq_neg_test),
#                 ('pol->pol2 (train on poles and test on poles (with no negs))', pol_pos_test, empty_test),
                ('pol->pol3 (train on poles and test on poles)', pol_pos_test, pol_neg_test)
               ]
              ),
              (
               ('experiment all train set', 25, np.concatenate([eq_pos_train, pol_pos_train]),
                                               np.concatenate([eq_neg_train, pol_neg_train])),
               [
#                 ('all->pol1 (train on all and test on poles (with eq negs))', pol_pos_test, eq_neg_test),
#                 ('all->pol2 (train on all and test on poles (with no negs))', pol_pos_test, empty_test),
                ('all->pol3 (train on all and test on poles)', pol_pos_test, pol_neg_test)
               ]
              )
]

# for (name, (train_pos, train_neg), (test_pos, test_neg)) in eval_pairs:
for train_tuple, test_tuples in eval_pairs:
    name, epochs, train_pos, train_neg = train_tuple
    log('Exp: ' + name + ', epochs: ' +  str(epochs))
    X, y = create_dataset(train_pos, train_neg)
    X, y = shuffle(X, y)
    train_split = int(0.8 * X.shape[0])
    train_set = get_neon_set(X[:train_split], y[:train_split])
    val_set = get_neon_set(X[train_split:], y[train_split:])
    model = fit_model(train_set, val_set, num_epochs=epochs)
    train_error = test_model(model, train_set)
    log('Train Misclassification error = %.2f%%' % train_error)
    val_error = test_model(model, val_set)
    log('Val Misclassification error = %.2f%%' % val_error)
    for test_tuple in test_tuples:
        name, test_pos, test_neg = test_tuple
        log('Exp: ' + name)
        X_test, y_test = create_dataset(test_pos, test_neg)
        test_set = get_neon_set(X_test, y_test)
        test_error = test_model(model, test_set)
        log('  Test Misclassification error = %.2f%%' % test_error)
    log('')

    model.get_description()
    model.save_params('eq_polar_params.p')