import mace
import numpy as np
import ase.io
from mace.calculators.foundations_models import mace_mp
import matplotlib.pyplot as plt
import glob
import os
import runnerase 
from ase.visualize import view
import pandas as pd


class ValidationAnalyser:
    
    def __init__(self,validation_config_paths,model_path,fmt='data'):
        """
        Parameters
        ----------
        list_of_config_paths: list (str) of paths to .input files for different configs. Each config may have multiple frames.
                              I assume that the format for the config files is .input .
        
        model: the .model file for the model you are validating 

        fmt: format of validation config files, assumed '.data' for now.
        
        """
        #Initialising Class attributes
        self.config_paths = validation_config_paths 
        self.model_calculator = gen_2_model_calc = mace_mp(model=model_path, dispersion=True)
        self.config_names= []
        self.config_topologies = []
        self.ref_configs = [] #Atoms object for each structure in form: [conf1,conf2,...] where conf1 = [struct1,struct2,...]
        self.atom_to_struct=[]
        self.atom_to_element=[]
        self.struct_to_conf=[]
        self.ref_forces = [] #index i is reference force on that atom
        self.model_forces = []
        self.force_error_vectors= np.array([]) # force error vector i is the vector ref_F -model_F for atom i 
        self.elements = np.array([]) #set of elements in the validation set 
        self.atom_location = [] #[path to .xyz file, frame_num, atom index (within frame)]
        
        
        #This should be a getter 
        self.component_to_element = []
        self.force_error_components= []
        self.model_force_components = []
        self.ref_force_components = []

        
        
        #Extracts configs from .data files
        for config_path in self.config_paths: 
            config_name = config_path.split('.')[0].split('/')[-1]
            self.config_names. append(config_name)
            config_topology = config_name.split('_')[0]
            self.config_topologies.append(config_topology)
            ref_calc_structure=runnerase.io.ase.read(config_path, ':')
            self.ref_configs.append(ref_calc_structure)

        #Extracting ref forces and other info
        config_counter = 0
        struct_counter = 0
        num_configs = len(self.ref_configs)
        for config_index in range(num_configs):
            config =self.ref_configs[config_index]
            num_structs = len(config)
            for struct_index in range(num_structs):
                struct = config[struct_index]
                self.struct_to_conf.append(config_counter)
                num_atoms = len(struct)
                forces = struct.get_forces()
                for atom_index in range(num_atoms):
                    location_info = [config_index,struct_index,atom_index]
                    self.atom_location.append(location_info)
                    self.atom_to_struct.append(struct_counter)
                    self.atom_to_element.append(struct[atom_index].symbol)
                    self.ref_forces.append(forces[atom_index])
         
                    
                struct_counter+=1
            config_counter+=1

        #list of elements in system 
        self.num_atoms = np.sum([np.sum([len(struct) for struct in config]) for config in self.ref_configs]) 
        self.elements = np.unique(self.atom_to_element)
        
        #Computing model forces
        print('Calculating model forces')
        for config_path in self.config_paths: 
            #loading a new config such that does not change forces of ref_config
            config = runnerase.io.ase.read(config_path, ':') 
            
            for struct in config:
                num_atoms = len(struct)
                struct.set_calculator(self.model_calculator)
                forces=struct.get_forces()
                for atom_index in range(num_atoms):
                    self.model_forces.append(forces[atom_index])
                    
                struct_counter+=1
            config_counter+=1
        print('Forces Computed')
        
        #computing errors:
        self.force_error_vectors = np.array(self.ref_forces) - np.array(self.model_forces) 

            
        """ 
        for atom_index in range(num_atoms):
            element = self.atom_to_element[atom_index]
            model_force = self.model_forces[atom_index]
            ref_force = self.ref_forces[atom_index]
            force_error = self.force_error_vectors[atom_index]
            for component_index in range(3):
                self.component_to_element.append(element)
                self.ref_force_components.append(ref_force[component_index])
                self.model_force_components.append(model_force[component_index])
                self.force_error_components.append(force_error[component_index])
        """

    def forces_to_components(self,forces):
        components = []
        for force in forces:
            for i in range(3):
                components.append(force[i])
            
        return components

    
    def find_erroneous_ref_forces(self, tolerance_factor = 5):

        erroneous_force_atom_indices = []
        erroneous_forces =[]
        cleansing_mask = np.full(self.num_atoms,True)
        ref_force_magnitudes = np.array([np.linalg.norm(force) for force in self.ref_forces])
        
        Q1=np.percentile(ref_force_magnitudes,25)
        Q3=np.percentile(ref_force_magnitudes,75)
        IQR=Q3-Q1
        
        lower_bound=Q1-tolerance_factor*IQR
        upper_bound=Q3+tolerance_factor*IQR      

        for atom_index in range(self.num_atoms):
            force_mag = ref_force_magnitudes[atom_index]
            
            if force_mag < lower_bound or upper_bound < force_mag:
                erroneous_force_atom_indices.append(atom_index)
                erroneous_forces.append(force_mag)
                cleansing_mask[atom_index] = False
                
        return erroneous_force_atom_indices,erroneous_forces,cleansing_mask 

    
    def find_erroneous_model_forces(self,ref_forces_cleansing_mask ,tolerance_factor = 5): 
        high_error_atom_incices = []
        force_errors = [] 
        
        if ref_forces_cleansing_mask is None:
            ref_forces_cleansing_mask = self.find_erroneous_ref_forces()[2]

        force_error_magnitudes = np.array([ np.linalg.norm(force) for force in self.force_error_vectors ])
        cleansed_force_error_magnitudes=force_error_magnitudes[ref_forces_cleansing_mask]
        
        model_error_cleansing_mask = ref_forces_cleansing_mask

        Q1=np.percentile(cleansed_force_error_magnitudes,25)
        Q3=np.percentile(cleansed_force_error_magnitudes,75)
        IQR=Q3-Q1
        
        lower_bound=Q1-tolerance_factor*IQR
        upper_bound=Q3+tolerance_factor*IQR      

        
        for atom_index in range(self.num_atoms):
            if ref_forces_cleansing_mask[atom_index]:
                error_mag = force_error_magnitudes[atom_index]
                
                if error_mag < lower_bound or upper_bound < error_mag:
                    high_error_atom_incices.append(atom_index)
                    force_errors.append(error_mag)
                    model_error_cleansing_mask[atom_index] = False

        return high_error_atom_incices, force_errors, model_error_cleansing_mask

    def cleanse_validation_structures(self,ref_tolerance_factor=5,model_tolerance_factor=5):
        """
        Returns
        -------
        -model_error_cleansing_mask: (bool) index i is True if atom i is not erroneous
        -high_error_atom_incices: atom indices where model errors were high
        -ref_erroneous_force_atom_indices: atom indices where reference forces were extremely high

        """

        ref_erroneous_force_atom_indices,erroneous_forces,ref_cleansing_mask=self.find_erroneous_ref_forces(ref_tolerance_factor)
        high_error_atom_incices, force_errors, model_error_cleansing_mask = self.find_erroneous_model_forces(ref_cleansing_mask ,model_tolerance_factor)

        return model_error_cleansing_mask , high_error_atom_incices, ref_erroneous_force_atom_indices



    def visualise_atom(self,atom_index,view=True):
        conf_index,struct_index,atom_struct_index = self.atom_location[atom_index] #
        conf_path = self.config_paths[conf_index] 
        struct = runnerase.io.ase.read(conf_path, str(struct_index) )
        element = self.atom_to_element[atom_index]
        position = struct[atom_struct_index].position
        print('model error:', np.linalg.norm(self.force_error_vectors[atom_index]) )
        print('ref force magnitude:', np.linalg.norm(self.ref_forces[atom_index]) )
        print('position',position)
        print('index',atom_struct_index)
        print('element',element)
        if view:
            struct.edit()


    def get_force_component_data(self):
        ref_components
        model_components
        error_components
        cleanse_mask
        pass
        
    def get_component_error_hist_data(self):
        bin_centres  
        bin_heights 
        pass

    def get_U_map_data(self):
        atom_to_element
        atom_to_topology
        atom_to_config
        atom_force_error #magnitude of force difference vector 
        cleansed_mask
        ref_force_error_atoms
        model_force_error_atoms

        #one np array with everything
        
        pass

    