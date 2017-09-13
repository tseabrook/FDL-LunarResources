#Written by Yarin Gal
#yg279@cam.ac.uk
#Contributions by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import os, time
import numpy as np
from data_loader import load_data, create_dataset, shuffle, split
from model import get_neon_set, fit_model, test_model
from neon.models import Model
from neon.data import ArrayIterator
from shutil import copyfile


thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC')
NACEquatorialDir = os.path.join(NACDir, 'Equatorial')
NACPolarDir = os.path.join(NACDir, 'South_Pole')

model_param_name = 'eq_polar_params.p'

paramsFilename = os.path.join(thisDir, model_param_name)
model = Model(paramsFilename)
print('Model Parameters loaded from file: ')
print(model.get_description())

if (not os.path.isdir(os.path.join(NACPolarDir, 'Unlabelled Regions'))):
    print('Please check that you have extracted the \'Unlabelled Regions.zip\' data file provided.')
else:

    log_name = os.path.join(os.getcwd(), 'predictor' + time.strftime('exp_%Y%m%d_%H%M%S') + '.log')

    def log(text):
        print(text)
        with open(log_name, 'a') as f:
            f.write(text + '\n')

    log('Creating log file ' + log_name)

    polar_predict, filenames = load_data(os.path.join(NACPolarDir,'Unlabelled Regions'), is_augment=False)

    files = []
    locations = []
    for filename in filenames:
        files.append(filename.split('/')[-1])
        locations.append(filename.split(files[-1])[0])

    log('Data loaded from ' + os.path.join(NACPolarDir,'Unlabelled Regions'))

    classes = ["Crater", "Not Crater"]
    inference_set = ArrayIterator(polar_predict, None, nclass=2, lshape=(1, 32, 32))
    out = model.get_outputs(inference_set)

    labels = np.argmax(out, axis = 1)

    count = []
    threshold = []
    threshold = np.subtract(1, np.divide(np.multiply(-400, np.power((np.arange(100).astype(float)), 2)),
                             10 * np.min(np.multiply(-400, np.power((np.arange(100).astype(float)), 2)))))

    for i in range(100):
        count.append(float(len(np.nonzero(out[np.arange(len(labels)),labels[:]] > threshold[i])[0])))
    perc = np.divide(count, len(filenames))
    ind = np.min(np.nonzero(perc > 0.6))

    NotCraterInd = np.nonzero((out[np.arange(len(labels)),labels[:]] > threshold[ind]) & (np.argmax(out, axis=1) == 0))[0]
    CraterInd = np.nonzero((out[np.arange(len(labels)),labels[:]] > threshold[ind]) & (np.argmax(out, axis=1) == 1))[0]
    NotSureInd = np.nonzero(out[np.arange(len(labels)),labels[:]] < threshold[ind])[0]

    filenames = np.array(filenames)
    files = np.array(files)

    if (not os.path.isdir(os.path.join(NACPolarDir, 'Not Crater', 'DNN'))):  # SUBJECT TO RACE CONDITION
        os.makedirs(os.path.join(NACPolarDir, 'Not Crater', 'DNN'))
    if (not os.path.isdir(os.path.join(NACPolarDir, 'Crater', 'DNN'))):  # SUBJECT TO RACE CONDITION
        os.makedirs(os.path.join(NACPolarDir, 'Crater', 'DNN'))
    if (not os.path.isdir(os.path.join(NACPolarDir, 'Ambiguous', 'DNN'))):  # SUBJECT TO RACE CONDITION
        os.makedirs(os.path.join(NACPolarDir, 'Ambiguous', 'DNN'))

    #Move to Not Crater
    for ind in CraterInd:
        copyfile(filenames[ind],
             os.path.join(NACPolarDir, 'Crater', 'DNN', files[ind]))
    #filenames[np.argmax(out, axis = 1) == 0]
    #Move to Crater
    for ind in NotCraterInd:
        copyfile(filenames[ind],
             os.path.join(NACPolarDir, 'Not Crater', 'DNN', files[ind]))

    for ind in NotSureInd:
        copyfile(filenames[ind],
             os.path.join(NACPolarDir, 'Ambiguous', 'DNN', files[ind]))

    #filenames[np.argmax(out, axis=1) == 1]
    #classes[labels]
    #print ('')


