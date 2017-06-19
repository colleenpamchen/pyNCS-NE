import sys, time
import numpy as np
import matplotlib.pylab as plt
from expSetup import *
from expPop import *
import pylab
from scipy.optimize import curve_fit


pop_exc1=pyNCS.Population(name='core0')
pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,0] for i in range(256)])

pop_exc2=pyNCS.Population(name='core3')
pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,3] for i in range(256)])

mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

if __name__ == '__main__':

	c0.load_parameters('setupfiles/biases_killconnections')
	c = nsetup.chips['dynapse_u0']
	nsetup.mapper.clear_cam_chip_core(0,0)

	nsetup.prepare() # this sets the connections 
	time.sleep(8)

	tls1 = []
	sls1 = []
	tls2 = []
	sls2 = []

	nsetup.stimulate(duration=1000)   
	tls1.append(mon_core1.copy())
	sls1.append(mon_core1.sl.mean_rates())
	tls2.append(mon_core2.copy())
	sls2.append(mon_core2.sl.mean_rates())
	sls1 = np.array(sls1) # 1 x 256
	sls2 = np.array(sls2)
	
	
	plot(sls2.T,'o') 
	# pylab.title('Exc weight = %s '%(ww))
	pylab.title('')
	plt.xlabel('Exc weight', fontsize=18)
	plt.ylabel('estimated output firing rates', fontsize=18)

# clear all connections on core 3 and see firing rate is.
# do an experiment with the excitatory and see what the difference in firing rates for core 3 based on setting weights
# 1) record step 1 2) subtract from firing rates found in step 2
# do an experiment with the inhibitory and see what the difference in firing rates for core 3 based on setting weights
# take recordings from step one and subtract from step 3
# change the dcp 
