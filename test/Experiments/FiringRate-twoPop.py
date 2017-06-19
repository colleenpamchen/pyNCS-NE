import sys, time
import numpy as np
import matplotlib.pylab as plt
from expSetup import *
from expPop import *
# from expCom import * 
import pylab
from scipy.optimize import curve_fit

#Gorgios 
def sigmoid(x,a,b,d):
     y = a / (1 + np.exp(-b*x/a)*(a-c)/a)- d 
     return y

pop_exc1=pyNCS.Population(name='core0')
pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,0] for i in range(256)])

pop_exc2=pyNCS.Population(name='core3')
pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,3] for i in range(256)])


mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

if __name__ == '__main__':

	c0.load_parameters('setupfiles/biases_calibrations')
	# c0.load_parameters('setupfiles/biasgreenblue')	
	c = nsetup.chips['dynapse_u0']
	nsetup.mapper.clear_cam_chip_core(0,0)

	tls1 = []
	sls1 = []
	tls2 = []
	sls2 = []

	pyNCS.Connection(pop_exc1, pop_exc2, synapse='exc_fast',fashion='one2one')
	nsetup.prepare() # this sets the connections 
	time.sleep(8)

	for j in range(0,255,15):
	    c.configurator.set_parameter('C0_IF_DC_P.fineValue', j)
	    # nsetup.run(None, duration = 1000) 
	    nsetup.stimulate(duration=1000)   
	    tls1.append(mon_core1.copy())
	    sls1.append(mon_core1.sl.mean_rates())
	    tls2.append(mon_core2.copy())
	    sls2.append(mon_core2.sl.mean_rates())

	sls1=np.array(sls1)
	sls2=np.array(sls2)
	sls1flat = sls1.flatten(order='F')
	sls2flat = sls2.flatten(order='F')
	
	maxrate = np.array(sls2[16,:])*1.05
	z = np.zeros((256,2))

## PolyFit	
	num=0
	for i in range(0,4352,17):
	    first=i
	    last= i+17
	    input_firingRate = sls1flat[ first:last]
	    output_firingRate = sls2flat[ first:last]
	    estimatedX = 1. / ( -(1./maxrate[num]) + (1./(output_firingRate))) 
	    wb = np.polyfit(input_firingRate, estimatedX ,1) # choose points that are linear
	    # take out the first part 

	    z[num,0]=wb[0]
	    z[num,1]=wb[1]
  
	    estimatedY = 1./( (1./maxrate[num]) + (1./(wb[0]*input_firingRate + wb[1])) )
	    pylab.plot(input_firingRate, estimatedY, label='fit')
	    num=num+1

	pylab.plot(sls1, sls2, 'o', label='data')
	# pylab.plot(x,y, label='fit')
	# pylab.ylim(0, 2.05)
	# pylab.legend(loc='upper left')
	# pylab.grid(True)
	pylab.show()   

## CurveFit
	
	popt, pcov = curve_fit(sigmoid,x ,y )
	sigmoid(x, *popt)

	





