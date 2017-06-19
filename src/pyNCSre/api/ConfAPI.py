#-----------------------------------------------------------------------------
# Purpose:
#
# Author: <authors name>
#
# Copyright : University of Zurich, Giacomo Indiveri, Emre Neftci, Sadique Sheik, Fabio Stefanini
# Licence : GPLv2
#-----------------------------------------------------------------------------
#ConfAPI
#Biases and mapper
#Api for modules having pyAMDA-like functionality
#Api for modules having pyAEX-like functionality

from __future__ import absolute_import
from .BaseConfAPI import *
from .ComAPI import ResourceManagerBase



def load_parameter_definitions(self, filename, name):
    '''
    Parse xml file or element tree to generate the object
    '''
    #IMPLEMENT (REQUIRED)
    raise NotImplementedError



def add_parameter(self, param_elm):
    '''
    Add a parameter to the configurator
    param: dictionary with all attributes of parameter or an xml element or
    file name with the <parameter /> element
    '''
    assert isinstance(param_elm, etree._Element)
    parameter = Parameter({'SignalName': ''}, self)
    parseNHML__(param)
    return parameter

class ConfiguratorBase(ResourceManagerBase):
    def __init__(self):
        '''
        ConfiguratorBase()
        Base class for managing parameters
        Contains functions
        - set_parameter (required)
        - get_parameter (required)
        - add_parameter (required)
        - get_parameter_names (required)
        - reset (optional)
        - set_parameters (optional)
        - get_parameters (optional)
        - context_get_param (optional)

        Parameters should be stored in the _parameter dictionary.
        The dictionary's keys should be the parameter names.
        Inherits ResourceManagerBase
        '''
        self.parameters = {}
        ResourceManagerBase.__init__(self)

    def get_parameter(self, param_name):
        #IMPLEMENT
        '''
        Gets parameter param_name.
        '''
        return self.parameters[param_name]

    def get_parameters(self, param_names=None):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Returns parameters (dictionary of name-value pairs).
        Input:
        *param_names:* A list of parameter names
        If param_names is None, then this function returns all the parameters
        (using self.get_param_names())
        '''
        if param_names is None:
            param_names = self.get_param_names()

        if not isinstance(param_names, (list, tuple, GeneratorType)):
            raise TypeError('param_names should be a list, tuple or generator, not {0}'.
                format(type(param_names)))

        b = dict()
        for i, name in enumerate(param_names):
            b[name] = self.get_parameter(name)
        return b

    def set_parameter(self, param_name, param_value):
        #IMPLEMENT
        '''
        Sets parameter param_name with param_value
        '''
        self.parameters[param_name] = param_value
        return None

    def set_parameters(self, param_dict):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Set several parameters using a dictionary.
        Input:
        *param_dict*: dictionary of parameter names (str) - value (float) pairs.
        '''
        for name, value in param_dict.iteritems():
            self.set_parameter(name, value)
        self.get_parameters(param_dict.keys())
        return None

    def update_parameter(self, param_name, param_value):
        #CONVENIENCE FUNCTION. IMPLEMENTATION NOT REQUIRED
        '''
        Update/Inform the object of changes made from other clients.
        Input:
            *param_name*: Parameter name
            *param_value*: Parameter value
        Ideal to use when the parameters can be changed from multiple clients
        simultaneously.
        '''
        self.parameters[param_name].v = param_value
        return

    def get_param_names(self):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Returns names of all the parameters
        '''
        import numpy as np
        return np.sort(self.parameters.keys()).tolist()

    def save_parameters(self, filename, *kwargs):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Saves parameters to a file
        '''
        d = self.get_parameters()
        with open(filename, 'w') as f:
            for key in sorted(d.keys()):
                v = d[key]
                f.write("%s\t%d\n"%(key,v))
        print('Parameters have been saved to the file {0}'.format(filename))
        return None

    def load_parameters(self, filename, *kwargs):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Saves parameters to a file
        '''
        name_value_pairs = {}
        with open(filename, 'r') as f:
            while True:
                s = f.readline()
                if len(s) == 0:
                    break
                else:
                    s = s.strip()

                if s[0] == '%' or s[0] == '#':
                    continue
                if s.find(' ')!=-1:
                    sp = s.split(' ')
                elif s.find('\t')!=-1:
                    sp = s.split('\t')
                else:
                    raise Exception('Unknown delimiter. Reads spaces or tabs.')

                name_value_pairs[sp[0]] = sp[1]
        self.set_parameters(name_value_pairs)




        return None

    def reset(self):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Resets all the parameters to default values
        '''
        return None

    @contextmanager
    def context_get_param(self):
        #CONVENIENCE FUNCTION. IMPLEMENTATION IS NOT REQUIRED
        '''
        Convenience contextmanager:
        Context used when getting parameter object
        '''
        #This implementation raises an informative exception
        try:
            yield
        except KeyError as e:
            raise KeyError('There is no parameter {0} in the configurator'.
                format(e.message))


class MappingsBase(ResourceManagerBase):
    def __init__(self):
        '''
        MappingsBase()
        Base class for managing mappings

        Contains methods:
        - add_mappings() (required)
        - set_mappings(mappings) (optional)
        - get_mappings()
        - clear_mappings()
        - del_mappings() (optional, not used by pyNCS by default)
        '''
        ResourceManagerBase.__init__(self)

    def add_mappings(self, mappings):
        #IMPLEMENT (REQUIRED)
        '''
        Adds *mappings* to the mappings table.

        Inputs:
        *mappings*: a two-dimenstional iterable
        '''
        pass

    def get_mappings(self):
        #IMPLEMENT (REQUIRED)
        '''
        Returns an array representing the mappings
        '''
        return None

    def clear_mappings(self):
        #IMPLEMENT (REQUIRED)
        '''
        Clears the mapping table. No inputs
        '''
        return None

    def set_mappings(self, mappings):
        #CONVIENCE FUNCTION, IMPLEMENTATION NOT REQUIRED
        '''
        Clears the mapping table and adds *mappings* to the mappings table.

        Inputs:
        *mappings*: a two-dimenstional iterable
        '''
        self.clear_mappings()
        self.add_mappings(mappings)

    def del_mappings(self):
        #IMPLEMENT (OPTIONAL)
        '''
        Clears the mapping table. No inputs
        '''
        raise NotImplementedError('del_mappings has not been implemented')

    def filter_events(self, events):
        #CONVIENCE FUNCTION, IMPLEMENTATION NOT REQUIRED 
        '''
        Before Neurosetup sends the physical events are setup, they are first passed through this function. Useful if there are no virtual neurons.
        '''
        return events

# Default blank initializations
# Override these classes in custom API as required
Configurator = ConfiguratorBase
Mappings = MappingsBase
