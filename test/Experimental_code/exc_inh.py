import sys
import numpy as np
import matplotlib.pylab as plt
from expSetup import *
from expPop import *
# from expCom import * 
import pylab
from scipy.optimize import curve_fit

# def sigmoid(x, a,b,C):
#      y = C / (1 + np.exp( (a * (x + b) ) ))
#      return y
# def sigmoid(x,a,b,C):
#      y = C / (1 + np.exp( (a * (x + b ) ))
#      	return y
def doublelog(x,a,b):
	y=a*(x**b)
	return y 

pop_exc1=pyNCS.Population(name='core0')
pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,1] for i in range(256)])
pop_exc2=pyNCS.Population(name='core2')
pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,3] for i in range(256)])

mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

if __name__ == '__main__':

	c0.load_parameters('setupfiles/biasesboth')
	c = nsetup.chips['dynapse_u0']
	nsetup.mapper.clear_cam_chip_core(0,3)


	R_in = np.zeros((256,8)) # 256 (neurons) x 8 (wgt) 
	R_out = np.zeros((256,8)) 

	num8=0;
	for i in range(0,255,32):
		c.configurator.set_parameter('C1_IF_DC_P.fineValue', 150)
		c.configurator.set_parameter('C3_IF_DC_P.fineValue', i)

		c.configurator.set_parameter('C3_PS_WEIGHT_EXC_F_N.fineValue', 224)
		c.configurator.set_parameter('C3_PS_WEIGHT_INH_F_N.fineValue', 224)
		nsetup.stimulate(duration=1000)
		nsetup.prepare() # this sets the connections
		time.sleep(8)

		tls1 = []
		sls1 = []
		tls2 = []
		sls2 = []
		
		tls1.append(mon_core1.copy())
		sls1.append(mon_core1.sl.mean_rates())
		tls2.append(mon_core2.copy())
		sls2.append(mon_core2.sl.mean_rates())

		sls1 = np.array(sls1)  # 1 x 256
		sls2 = np.array(sls2)

		R_in[:, num8] = sls1
		R_out[:, num8] = sls2
		num8 = num8 + 1
	

	estimatedY = np.zeros((256,8))
	maxrate = np.zeros((256))	
	RinNew=np.multiply(224,R_in)
	for ii in range(0,255):
		maxrate[ii] = np.amax(R_out[ii,:])*1.05256 # ~90

		# estimatedX[ii,:] = 1. / ( -(1./maxrate[ii]) + (1./( R_out[ii,:] ))) 

		# wb = np.polyfit( RinNew[ii,:], estimatedX[ii,:] ,1) # wb has two elements 
		# w_wgt[ii,0]=wb[0] # this column is the w(wgt), sweep through the wgt to find w. Start with w_wgt then estimate W using a line   
		# w_wgt[ii,1]=wb[1]

		# used w values obtained from previous simulations with only Exc or Inh alone 
		estimatedY[ii,:] = 1./ ( (1./maxrate[ii]) + (1./ ((1.75544911e-01* RinNew[ii,:]) + (-1.29813283e-02 * RinNew[ii,:]) + 1.53803812e+03 - 4.08716272e+02) ) ) # estimated output Firing rate using estimated w_wgt 
		
	ett=estimatedY.T 
	meanet=np.mean(ett, axis=1)

	plot(R_out.T, 'o')
	plot(meanet)
	show()


# # 	# plt.figure()
# # 	# pylab.plot(wgt, estimatedY[:,:], label='fit')
# # 	# pylab.plot(R_in[ii,:],R_out[ii,:])
# # 	# pylab.title('Exc weight = %s '%(ww))
# # 	# plt.xlabel('Input firing rates', fontsize=18)
# # 	# # plt.ylabel('estimated output firing rates', fontsize=18)
# # 	# pylab.show()
# # 	plot(R_out.T,'o')
# # 	plot(estimatedY.T)
# # 	# pylab.title('Exc weight = %s '%(ww))
# # 	pylab.title('')
# # 	plt.xlabel('Exc weight', fontsize=18)
# # 	plt.ylabel('estimated output firing rates', fontsize=18)

	  

## CurveFit
	
	# ydata needs to be flat
	# x seems to be the spacing? 
	popt, pcov = curve_fit(doublelog,x ,y )
	sigmoid(x, *popt)
	


