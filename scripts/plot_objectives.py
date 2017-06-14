import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle
import time
import sys

filename = sys.argv[1]
file = open(filename)

last_chunk = -1
training_errors = []
validation_errors = []
training_f2 = []
validation_f2 = []
training_idcs = []
validation_idcs=[]

for line in file:
	if 'Chunk' in line :
		last_chunk = int(line.split()[1].split('/')[0])
	if 'Validation loss' in line:
		validation_errors.append(float(line.split(':')[1].rsplit()[0]))
		validation_f2.append(float(line.split(':')[1].rsplit()[2]))
		validation_idcs.append(last_chunk)
	if 'Mean train loss' in line:
		training_errors.append(float(line.split(':')[1].rsplit()[0]))
		training_f2.append(float(line.split(':')[1].rsplit()[2]))
		training_idcs.append(last_chunk)


#print 'training errors'
#print training_errors
#print training_idcs
#print 'validation errors'
#print validation_errors
#print validation_idcs


print 'min training error', np.amin(np.array(training_errors)), 'at', np.argmin(np.array(training_errors)), '/', len(training_errors)
print 'min validation error', np.amin(np.array(validation_errors)), 'at', np.argmin(np.array(validation_errors)), '/', len(validation_errors)


print 'max training f2 score', np.amax(np.array(training_f2)), 'at', np.argmax(np.array(training_f2)), '/', len(training_f2)
print 'max validation f2 score', np.amax(np.array(validation_f2)), 'at', np.argmax(np.array(validation_f2)), '/', len(validation_f2)

print np.amin(np.array(training_errors)), 
print str(np.argmin(np.array(training_errors))) + '/' + str(len(training_errors)),
print np.amax(np.array(training_f2)),
print str(np.argmax(np.array(training_f2))) + '/' + str(len(training_f2)),


print np.amin(np.array(validation_errors)),
print str(np.argmin(np.array(validation_errors))) + '/' + str(len(validation_errors)),
print np.amax(np.array(validation_f2)),
print str(np.argmax(np.array(validation_f2))) + '/' + str(len(validation_f2))





fig = plt.figure()
plt.title(sys.argv[1])
ax = fig.add_subplot(111)
ax.plot(training_errors, label='training errors')
ax.plot(validation_errors, label='validation errors')

ax2 = ax.twinx()
ax2.plot(training_f2, label='training f2 score')
ax2.plot(validation_f2, label='validation f2 score')

ax.legend(loc="center right")
ax2.legend(loc="center left")
ax.grid()
ax.set_xlabel('Epoch')
ax.set_ylabel(r"Mean BCE")
ax2.set_ylabel(r"F2 score")
plt.savefig(sys.argv[2])


