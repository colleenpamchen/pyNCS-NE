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
pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,0] for i in range(256)])
# pop_exc2=pyNCS.Population(name='core2')
# pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,2] for i in range(256)])

mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
# mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

if __name__ == '__main__':

	c0.load_parameters('setupfiles/biases_inhibitory')
	# c0.load_parameters('setupfiles/biasgreenblue')	
	c = nsetup.chips['dynapse_u0']
	nsetup.mapper.clear_cam_chip_core(0,0)

	tls = []
	sls = []

	for i in range(0,255,8):
	    c.configurator.set_parameter('C0_IF_DC_P.fineValue', i)
	    # nsetup.run(None, duration = 1000) 
	    nsetup.stimulate(duration=1000) 
	    nsetup.prepare() # this sets the connections 
	    time.sleep(8)
	    tls.append(mon_core1.copy())
	    #sls.append(mon_core1.mean_rates())
	    sls.append(mon_core1.sl.mean_rates())

	sls=np.array(sls)
	plt.plot(sls,'g')	

	xspace = np.linspace(0, 1,32) 
	ydata = sls 

# Dan's double log fit: 
	# a_doublelog=110
	# b_doublelog=0.4
	# r_doublelog=doublelog(xspace,a_doublelog,b_doublelog)

	# plt.plot(r_doublelog,'k')
	show()

	# plt.plot(sls,'g')
	# plt.plot(r_sigmoid,'k')
	# plot.show()


	xspace=np.zeros((32,256))
	for i in range(0,32):
		n= i * 8
		xspace[i,]=n


	# ydata=ydata[1:32,:]
	# xspace=xspace[1:32,:]
	
	# xspace=xspace.flatten(order='F')
	# ydata=ydata.flatten(order='F')

	x =zeros((256,256))
	y=zeros((256,256))

	for jj in range (256):
		popt, pcov = curve_fit(sigmoid, xspace[:,jj], ydata[:,jj])
		x = np.linspace(0, 255,256)
		y[jj,:] = sigmoid(x, *popt)
		pylab.plot(x,y[jj,:], label='fit')

		#print popt
	# popt, pcov= curve_fit(doublelog,xdata,ydata)

	# a_sigmoid= popt[0]
	# b_sigmoid= popt[1]
	# C_sigmoid= popt[2]
	# r_sigmoid=sigmoid(xspace,a_sigmoid,b_sigmoid,C_sigmoid)

	pylab.plot(xspace, ydata, 'o', label='data')
	# pylab.ylim(0, 2.05)
	# pylab.legend(loc='upper left')
	pylab.grid(True)
	pylab.show()   

rinexperiment=y[:,150]

#Gorgios' modified sigmoid 
# def sigmoid(x,a,b,d):
#      y = a / (1 + np.exp(-b*x/a)*(a-c)/a)- d 
#      return y

# pop_exc1=pyNCS.Population(name='core0')
# pop_exc1.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,0] for i in range(256)])

# pop_exc2=pyNCS.Population(name='core3')
# pop_exc2.populate_by_addr_list(nsetup, 'dynapse_u0', 'neuron',[[i,3] for i in range(256)])

# mon_core1 = nsetup.monitors.import_monitors_otf(pop_exc1)
# mon_core2 = nsetup.monitors.import_monitors_otf(pop_exc2)

# if __name__ == '__main__':

# 	c0.load_parameters('setupfiles/biases_calibrations')
# 	c = nsetup.chips['dynapse_u0']
# 	nsetup.mapper.clear_cam_chip_core(0,0)
# 	pyNCS.Connection(pop_exc1, pop_exc2, synapse='exc_fast',fashion='one2one')
# 	nsetup.prepare() # this sets the connections 
# 	time.sleep(8)

# 	w_wgt = np.zeros((256,8))
# 	wgt = np.array([0,32,64,96,128,160,192,224]) # same as ww 

# 	w_params = np.zeros((256,2))

# 	R_in = np.zeros((256,8)) # 256 (neurons) x 8 (wgt) 
# 	R_out = np.zeros((256,8)) 
# 	maxrate = np.zeros((256))
# 	estimatedX = np.zeros((256,8))
# 	estimatedY = np.zeros((256,8))
# 	# RinWgt = np.zeros((256,8))

# 	num8=0
# 	for ww in range(0,256,32): # 8 wgt 
# 		c.configurator.set_parameter('C3_PS_WEIGHT_EXC_F_N.fineValue', ww )
# 		tls1 = []
# 		sls1 = []
# 		tls2 = []
# 		sls2 = []
# 		c.configurator.set_parameter('C0_IF_DC_P.fineValue', 150)
# 		nsetup.stimulate(duration=1000)   
# 		tls1.append(mon_core1.copy())
# 		sls1.append(mon_core1.sl.mean_rates())
# 		tls2.append(mon_core2.copy())
# 		sls2.append(mon_core2.sl.mean_rates())
# 		sls1 = np.array(sls1) # 1 x 256
# 		sls2 = np.array(sls2)
# 		R_in[:,num8] = sls1
# 		R_out[:,num8] = sls2
# 		num8 = num8+1

# 	RinNew=np.multiply(wgt,R_in)
# 	## PolyFit	
# 	for ii in range(0,255):
# 		maxrate[ii] = np.amax(R_out[ii,:])*1.05256 # ~90

# 		estimatedX[ii,:] = 1. / ( -(1./maxrate[ii]) + (1./( R_out[ii,:] ))) 

# 		wb = np.polyfit( RinNew[ii,:], estimatedX[ii,:] ,1) # wb has two elements 
# 		w_wgt[ii,0]=wb[0] # this column is the w(wgt), sweep through the wgt to find w. Start with w_wgt then estimate W using a line   
# 		# w_wgt[ii,1]=wb[1]
# 		estimatedY[ii,:] = 1./ ( (1./maxrate[ii]) + (1./ (wb[0] * RinNew[ii,:] + wb[1]) ) ) # estimated output Firing rate using estimated w_wgt 
		
# 	# plt.figure()
# 	# pylab.plot(wgt, estimatedY[:,:], label='fit')
# 	# pylab.plot(R_in[ii,:],R_out[ii,:])
# 	# pylab.title('Exc weight = %s '%(ww))
# 	# plt.xlabel('Input firing rates', fontsize=18)
# 	# # plt.ylabel('estimated output firing rates', fontsize=18)
# 	# pylab.show()	
# 	plot(R_out.T,'o') 
# 	plot(estimatedY.T)
# 	# pylab.title('Exc weight = %s '%(ww))
# 	pylab.title('')
# 	plt.xlabel('Exc weight', fontsize=18)
# 	plt.ylabel('estimated output firing rates', fontsize=18)

	
		









	# for i in range(256)
	

		    # num=num+1
 #    allweights=z[:,0]
    	
 #    	# inputFiringRatesPopulation[:,:,num2=sls1
 #    	# outputFiringRatesPopulation[:,:,num2]=sls2

	# for kk in range(256):
	# 	ft=np.polyfit(inputWeights,allweights[kk,:], 1)
	# 	w_params=[kk,0]=ft[0] # function params
	# 	w_params=[kk,1]=ft[1] #
	

	# est_outputFiringRate = 1. / ( (1./maxrate[num]) + ( 1./ ( (weightfparams[-1,0]*inputWeights[7]+weightfparams[-1,1]) * input_firingRate + wb[1]) )  )
	# pylab.plot(input_firingRate, est_outputFiringRate)
	# pylab.plot(sls1, sls2, 'o', label='data')
	# pylab.show() 

	# w=(weightfparams[-1,0]*inputWeights+weightfparams[-1,1])
	# pylab.plot(inputWeights,w) # for the last neuron
	# pylab.show() 

	# plot wgt x outputfiring rates
	# 




	# pylab.plot(sls1, sls2, 'o', label='data')
	# pylab.plot(x,y, label='fit')
	# pylab.ylim(0, 2.05)
	# pylab.legend(loc='upper left')
	# pylab.grid(True)
	  

## CurveFit
	
	# ydata needs to be flat
	# x seems to be the spacing? 
#	popt, pcov = curve_fit(sigmoid,x ,y )
#	sigmoid(x, *popt)
	


