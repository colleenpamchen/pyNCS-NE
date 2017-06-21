import sys
import numpy as np
import matplotlib.pylab as plt
from expSetup import *
from expPop import *
# from expCom import * 
import pylab
from scipy.optimize import curve_fit

def sigmoid(x, a,b,C):
     y = C / (1 + np.exp( (a * (x + b) ) ))
     return y

# def sigmoid(x,a,b,C):
#      y = C / (1 + np.exp( (a * (x + b ) ))
#      	return y

# def doublelog(x,a,b):
# 	y=a*(x**b)
# 	return y 

pop_exc1=pyNCS.Population(name='core0')
pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,3] for i in range(256)])
# pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,0] for i in range(256)])
# pop_exc2=pyNCS.Population(name='core2')
# pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,2] for i in range(256)])

mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
# mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

if __name__ == '__main__':

	# c0.load_parameters('setupfiles/biases_inhibitory')
	c0.load_parameters('setupfiles/biases_killconnections')
	# c0.load_parameters('setupfiles/biasgreenblue')	
	c = nsetup.chips['dynapse_u0']
	nsetup.mapper.clear_cam_chip_core(0,0)

	tls = []
	sls = []
	xspace32 = []

	for i in range(0,255,32):
	    # c.configurator.set_parameter('C0_IF_DC_P.fineValue', i)
	    c.configurator.set_parameter('C3_IF_DC_P.fineValue', i)
	    # nsetup.run(None, duration = 1000) 
	    nsetup.stimulate(duration=1000)   
	    tls.append(mon_core1.copy())
	    #sls.append(mon_core1.mean_rates())
	    sls.append(mon_core1.sl.mean_rates())
	    xspace32.append(i) 

	sls=np.array(sls)
	plt.plot(sls,'g')	
	xspace32=np.array(xspace32)

	xspace = np.linspace(0, 1,32) 
	ydata = sls 
	show()

	xspace=np.zeros((32,256))
	for i in range(0,32):
		n= i * 8
		xspace[i,]=n
	
	xspace=xspace.flatten(order='F')
	ydata=ydata.flatten(order='F')


	popt, pcov = curve_fit(sigmoid, xspace, ydata)
	print popt


	x = np.linspace(0, 255,256)
	y = sigmoid(x, *popt)
	y32 = sigmoid(xspace32,*popt)
	est32 = np.tile(y32,(256,1))
	est32 = est32.T 

	mse = ( (sls - est32) **2).mean(axis=None)
	rmse=np.sqrt(mse)

	pylab.plot(xspace, ydata, 'o', label='data')
	pylab.plot(x,y, label='fit')
	# pylab.ylim(0, 2.05)
	# pylab.legend(loc='upper left')
	pylab.grid(True)
	pylab.show()   






