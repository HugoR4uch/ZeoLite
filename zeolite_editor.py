import ase.io
import ase
import numpy as np


import matplotlib.pyplot as plt

class StructureEditor:
    def __init__(self,pure_zeolite):
        self.zeolite = pure_zeolite

        
    def set_periodicity(self,cell,pbc=[True,True,True]):
        self.zeolite.set_pbc(pbc)
        self.zeolite.set_cell(cell)
    
    def remove_atoms(self,indices):
        del self.zeolite[indices]
    
    def set_system(self,new_system):
        self.zeolite=new_system
        
    def get_system(self):
        return self.zeolite
    
    def find_neighbours(self,atom_index,r_cutoff):
        neighbours=np.intersect1d(np.where( self.zeolite.get_all_distances(mic=True)[atom_index] <r_cutoff),
                        np.where( self.zeolite.get_all_distances(mic=True)[atom_index] != 0))      
        return neighbours
      
    def atom_selector(self,atom_list,atom):
        """
        Outputs np.array of subset of indices from np.array 'atom_list' of type 'atom'
        """
        filtered_list = atom_list[self.zeolite[atom_list].symbols==atom]
        return filtered_list
    
    
    def find_max_water_loading(self,r_cutoff=10,successive_loading_failiures_cutoff=100):
        unloaded_system_size=len(self.zeolite)
 
        LJ_sigmas=np.loadtxt("../LJ_sigmas.csv",delimiter=",",dtype="float",usecols=1,skiprows=1)
        LJ_symbols=np.loadtxt("../LJ_sigmas.csv",delimiter=",",dtype="str",usecols=0,skiprows=1)
        
        n_added=0            
        successive_failiure_counter=0
        filling=True
        
        while(filling):
            success=False
            
            atoms=self.get_system()
            rand_coefs=np.random.rand(3)[:, np.newaxis]
            trial_point=np.sum(rand_coefs * atoms.get_cell()[:],axis=0)
            self.add_H2O(trial_point)
            

            O_index=len(atoms)-1
            H1_index=len(atoms)-2
            H2_index=len(atoms)-3

            #'Neighbours' are atoms withn r_cutoff radius of the trial O atom
            trial_point_distances=atoms.get_distances(O_index,list(range(0,H2_index)), mic=True, vector=False)
            neighbours=np.array(range(0,H2_index))[trial_point_distances<r_cutoff]
            if(len(neighbours)==0):
                success=True
            else:
                O_distances=atoms.get_distances(O_index,neighbours, mic=True, vector=False)
                H1_distances=atoms.get_distances(H1_index,neighbours, mic=True, vector=False)
                H2_distances=atoms.get_distances(H2_index,neighbours, mic=True, vector=False)
                
                #Sigma is the Lenard Jones potential parameter
                neighbour_sigmas=np.zeros(len(neighbours))
                for i in range(len(neighbours)):
                    neighbour_sigmas[i] = LJ_sigmas[LJ_symbols==atoms[neighbours[i]].symbol] [0]

                sigma_H_neighbour= 0.5 * (neighbour_sigmas + LJ_sigmas[LJ_symbols=="H"]) 
                sigma_O_neighbour= 0.5 * (neighbour_sigmas + LJ_sigmas[LJ_symbols=="O"]) 

                scaled_O_distances=O_distances / sigma_O_neighbour
                scaled_H1_distances=H1_distances / sigma_H_neighbour
                scaled_H2_distances=H2_distances / sigma_H_neighbour


                if np.all(np.array([scaled_O_distances,scaled_H1_distances,scaled_H2_distances]) >2) :
                    success=True
            
                if(not success):
                    self.remove_atoms([-1,-2,-3])
                    successive_failiure_counter+=1
                else:
                    n_added+=1
                    successive_failiure_counter=0

            
            if successive_failiure_counter >= successive_loading_failiures_cutoff:
                    max_loading=n_added
                    filling = False
            
            
        #Removing the added test waters 
        added_water_indices = np.arange(unloaded_system_size,len(atoms))
        self.remove_atoms(added_water_indices)
        
        return(max_loading)
    
    
    
    def defect_site_info(self,site_index):
        r_SiO_cutoff=1.7
        suitability=True
        site_neighbours=self.find_neighbours(site_index,r_SiO_cutoff)
        site_neighbours_O=self.atom_selector(site_neighbours,"O")
        
        if(not len(site_neighbours_O) == 4 ):
            suitability=False
        
        site_neighbours_Si=[]
        for i in site_neighbours_O:
            O_neighbours=self.find_neighbours(i,r_SiO_cutoff)
            O_neighbours_Si=self.atom_selector(O_neighbours,"Si")
        
            O_neighbours_Si = np.delete( O_neighbours_Si , np.where(O_neighbours_Si==site_index)[0][0] )# This sometimes breaks
            if (not len(O_neighbours_Si) == 1 ):
                suitability=False
            else:
                site_neighbours_Si.append(O_neighbours_Si[0])
            
        return suitability,site_neighbours_O,site_neighbours_Si
    
    def add_partially_hydrolysed_site_1():
        pass
        
        
    
    def add_H2O(self,O_pos,H_pos=False,molecule_index=1):
        """
        Params
        ------
        -H_pos: list of numpy arrays of both H positions 
        -O_pos: O atom position, numpy array
        """

        if(H_pos!=False):
            H2O_disp = [H_pos[0],H_pos[1],np.array([0,0,0])] 
        else:
            H2O_r = 0.957 #OH bond length
            H2O_angle = 37.75/180 * np.pi #angle between x-axis and O-H displacemnet 
            H2O_disp = [H2O_r*np.array([-np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        H2O_r*np.array([np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        np.array([0,0,0])]

        water = ase.Atoms('H2O',positions = H2O_disp,tags=[molecule_index,molecule_index,molecule_index])
        euler_angles = np.random.rand(3) * 180 * [2,1,2]
        water.euler_rotate(euler_angles[0],euler_angles[1],euler_angles[2]) 
        new_pos = water.get_positions() + [ O_pos , O_pos , O_pos ]
        water.set_positions(new_pos)
        self.zeolite.extend(water)

                
    def fill_H2O(self,n_add,n_trials,f_cutoff=2,r_cutoff=10,printing=False):
        
        if n_add == 0 :
            return
        
        LJ_sigmas=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteRevolution/LJ_sigmas.csv",delimiter=",",dtype="float",usecols=1,skiprows=1)
        LJ_symbols=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteRevolution/LJ_sigmas.csv",delimiter=",",dtype="str",usecols=0,skiprows=1)
        
        n_attempts=0
        n_added=0
        filling=True
            

        while(filling):
            success=False
            
            atoms=self.get_system()
            rand_coefs=np.random.rand(3)[:, np.newaxis]
            trial_point=np.sum(rand_coefs * atoms.get_cell()[:],axis=0)
            self.add_H2O(trial_point)
            

            O_index=len(atoms)-1
            H1_index=len(atoms)-2
            H2_index=len(atoms)-3

            #'Neighbours' are atoms withn r_cutoff radius of the trial O atom
            trial_point_distances=atoms.get_distances(O_index,list(range(0,H2_index)), mic=True, vector=False)
            neighbours=np.array(range(0,H2_index))[trial_point_distances<r_cutoff]
            if(len(neighbours)==0):
                success=True
            else:
                O_distances=atoms.get_distances(O_index,neighbours, mic=True, vector=False)
                H1_distances=atoms.get_distances(H1_index,neighbours, mic=True, vector=False)
                H2_distances=atoms.get_distances(H2_index,neighbours, mic=True, vector=False)
                
                #Sigma is the Lenard Jones potential parameter
                neighbour_sigmas=np.zeros(len(neighbours))
                for i in range(len(neighbours)):
                    neighbour_sigmas[i] = LJ_sigmas[LJ_symbols==atoms[neighbours[i]].symbol] [0]

                sigma_H_neighbour= 0.5 * (neighbour_sigmas + LJ_sigmas[LJ_symbols=="H"]) 
                sigma_O_neighbour= 0.5 * (neighbour_sigmas + LJ_sigmas[LJ_symbols=="O"]) 

                scaled_O_distances=O_distances / sigma_O_neighbour
                scaled_H1_distances=H1_distances / sigma_H_neighbour
                scaled_H2_distances=H2_distances / sigma_H_neighbour


                if np.all(np.array([scaled_O_distances,scaled_H1_distances,scaled_H2_distances]) >2) :
                    success=True
            
            if(not success):
                self.remove_atoms([-1,-2,-3])
            else:
                n_added+=1
                if(printing):
                    print(n_added," have now been added, after ",n_attempts," trials")
                if(n_added==n_add):
                    filling=False

            n_attempts+=1
            if(n_attempts==n_trials):
                filling=False
                if(printing):
                    print("Failed to fit enough waters")



    def add_Al_defect(self,site_index):
        
         
        suitable,site_neighbours_O,site_neighbours_Si=self.defect_site_info(site_index)   
        
        if(not suitable):
            return

        site_choice=np.random.randint(0,4)
        OH_site=site_neighbours_O[site_choice]
        adjacent_Si=site_neighbours_Si[site_choice]

        r_OSi=self.get_system().get_distance(OH_site,adjacent_Si,mic=True,vector=True)
        r_AlO=self.get_system().get_distance(site_index,OH_site,mic=True,vector=True)
        r_OH=np.cross(r_OSi,r_AlO)
        defect_site_pos=self.get_system()[site_index].position
        H_pos=1.1*r_OH/np.linalg.norm(r_OH)+self.get_system()[OH_site].position
        self.zeolite.extend(ase.Atoms('H',positions = [H_pos]))
        #Replace O at Al_site with Al
        del self.zeolite[site_index]
        self.zeolite.extend(ase.Atoms('Al',positions = [defect_site_pos]))
    
    def add_Al_defect_pair(self,site_index):
        suitable,neighbouring_O_indices,neighbouring_Si_indices=self.defect_site_info(site_index)

        if(not suitable):
            return False

        neighbour_query_choice=np.random.randint(0,4)
        neighbours_queried=0
        
        while (neighbours_queried<4):
            neighbour_Si_index=neighbouring_Si_indices[neighbour_query_choice]
            
            neighbours_neighbouring_Si_indices=self.defect_site_info(neighbour_Si_index)[2]
            neighbour_suitability=self.defect_site_info(neighbour_Si_index)[0]
            if(not neighbour_suitability):
                return False
            neighbours_neighbour_query_choice=np.random.randint(0,4) 
            next_nearest_neighbours_queried=0
            while (next_nearest_neighbours_queried<4): 
                
                next_nearest_neighbour_Si=neighbours_neighbouring_Si_indices[neighbours_neighbour_query_choice]
                
                next_nearest_neighbour_suitable=self.defect_site_info(next_nearest_neighbour_Si)[0] 
                if(next_nearest_neighbour_suitable==False):
                    
                    neighbours_neighbour_query_choice = (neighbours_neighbour_query_choice +1 ) % 4
                    next_nearest_neighbours_queried+=1
                else:
                    self.add_Al_defect(next_nearest_neighbour_Si) 
                    #Index of original site Si reduced by 1 if index of removed atom larger than site index, due to heteroatom deletion
                    if(next_nearest_neighbour_Si<site_index):
                        self.add_Al_defect(site_index -1) 
                    else:
                        self.add_Al_defect(site_index) 
                    return True
            neighbours_queried+=1
        return False
        

    def fill_neighbour_Al_defects(self,n_add,n_trials=100):

        if n_add == 0: 
            return
        
        possible_defect_sites=self.atom_selector(np.array(range(0,len(self.zeolite))),"Si")
        n_attempts=0
        n_success=0
        filling=True
        while(filling):
        
            n_attempts+=1
            trial_site= possible_defect_sites [ np.random.randint(0,len(possible_defect_sites)) ]
            
            
            #Removing from list of possible defect sites
            possible_defect_sites=np.delete(possible_defect_sites,
                                            np.where(possible_defect_sites == trial_site))
            
            suitable = self.add_Al_defect_pair(trial_site)
                        
            if(suitable):
                
                n_success+=1
            
            success = n_success == n_add 
            failure = n_attempts == n_trials or len(possible_defect_sites) == 0
            if(success or failure):
                filling = False
        if(success):
            print("Successully added neighbour Al defects")
        else:
            print("Failed after adding ",n_success ," neighbour Al defects.")    
    
    
    def fill_Al_defects(self,n_add,n_trials=100):
        
        if n_add == 0: 
            return
        
        possible_defect_sites=self.atom_selector(np.array(range(0,len(self.zeolite))),"Si")
        n_attempts=0
        n_success=0
        filling=True
        while(filling):
        
            n_attempts+=1
            trial_site= possible_defect_sites [ np.random.randint(0,len(possible_defect_sites)) ]
            
            
            #Removing from list of possible defect sites
            possible_defect_sites=np.delete(possible_defect_sites,
                                            np.where(possible_defect_sites == trial_site))
            
            suitable = self.defect_site_info(trial_site)[0]
                        
            if(suitable):
                self.add_Al_defect(trial_site)
                n_success+=1
            
            success = n_success == n_add 
            failure = n_attempts == n_trials or len(possible_defect_sites) == 0
            if(success or failure):
                filling = False
        if(success):
            print("Successully added Al defects")
        else:
            print("Failed after adding ",n_success ," Al defects.")        
        
        
    def add_silanol_defect(self,site_index):
        length_OH=1.1#O-H bond length
        suitable,site_neighbours_O,site_neighbours_Si=self.defect_site_info(site_index)   
        
        if(not suitable):
            return
        
        
        for i in range(4):
            r_SiO=self.get_system().get_distance(site_neighbours_Si[i],site_neighbours_O[i],mic=True,vector=True)#Vect from Si to O
        
            r_site_O=self.get_system().get_distance(site_index,site_neighbours_O[i],mic=True,vector=True)
            
            r_OH=np.cross(r_SiO,r_site_O)
            

            #Need to shake H atoms until they are far enough from eachother
            H_pos=length_OH*r_OH/np.linalg.norm(r_OH)+self.get_system()[site_neighbours_O[i]].position
            self.zeolite.extend(ase.Atoms('H',positions = [H_pos]))
        
        del self.zeolite[site_index]
        
        
    def fill_silanol_defects(self,n_add,n_trials=100):
        
        if n_add==0: 
            return 

        possible_defect_sites=self.atom_selector(np.array(range(0,len(self.zeolite))),"Si")
        n_attempts=0
        n_success=0
        filling=True
        while(filling):
        
            n_attempts+=1
            trial_site= possible_defect_sites [ np.random.randint(0,len(possible_defect_sites)) ]
            
            
            #Removing from list of possible defect sites
            possible_defect_sites=np.delete(possible_defect_sites,
                                            np.where(possible_defect_sites == trial_site))
            
            suitable = self.defect_site_info(trial_site)[0]
                        
            if(suitable):
                self.add_silanol_defect(trial_site)
                n_success+=1
            
            success = n_success == n_add 
            failure = n_attempts == n_trials or len(possible_defect_sites) == 0
            if(success or failure):
                filling = False
        if(success):
            print("Successully added Silanol defects")
        else:
            print("Failed after adding ",n_success ," Silanol defects.")        
            
        
        
    def get_image_replicas(self,atom_index):
        """
        Gives indices of all atoms who's minimal image is within cutoff distance of atom with index=atom_index
        """
        cutoff=0.01
        distances_vs_cutoff_matrix=np.array( self.zeolite.get_all_distances(mic=True)<cutoff )*np.logical_not( np.eye(len(self.zeolite),dtype=bool) ) 
        return np.nonzero(distances_vs_cutoff_matrix[atom_index])[0]
    
    def delete_image_replicas(self,printing=False):
        
        deleted_atoms = []
        for i in range( len(self.zeolite) ):
            if not i in deleted_atoms:
                replica_images = self.get_image_replicas(i)
                if not len(replica_images) == 0: 
                    for j in replica_images: deleted_atoms.append(j)

        del self.zeolite[deleted_atoms]
        if printing:
            print(len(deleted_atoms)," mirror replica atoms deleted.")

        

        
