import ase.io
import ase
import numpy as np
import matplotlib.pyplot as plt



class StructureEditor:
    def __init__(self,pure_zeolite):
        self.zeolite = pure_zeolite
        #Adding tags to atoms
        self.zeolite.set_tags(np.arange(0,len(self.zeolite),1))

        
    def set_periodicity(self,cell,pbc=[True,True,True]):
        self.zeolite.set_pbc(pbc)
        self.zeolite.set_cell(cell)
    
    def remove_atoms(self,indices):
        del self.zeolite[indices]
    
    def set_system(self,new_system):
        self.zeolite=new_system
        
    def get_system(self):
        return self.zeolite
    
    def find_neighbours(self, atom_index, r_cutoff):
        """
        Find the indices of the neighboring atoms within a specified distance cutoff. (Includes MIC).

        Parameters:
        - atom_index (int): The index of the atom for which to find neighbors.
        - r_cutoff (float): The distance cutoff for determining neighbors.

        Returns:
        - neighbours (numpy.ndarray): An array of indices representing the neighboring atoms.

        """
        neighbours = np.intersect1d(
            np.where(self.zeolite.get_all_distances(mic=True)[atom_index] < r_cutoff),
            np.where(self.zeolite.get_all_distances(mic=True)[atom_index] != 0)
        )
        return neighbours
    
      
    def atom_selector(self,atom_list,atom):
        """
        Outputs np.array of subset of indices from np.array 'atom_list' of type 'atom'.
        """
        filtered_list = atom_list[self.zeolite[atom_list].symbols==atom]
        return filtered_list
    
    
    def find_max_water_loading(self,r_cutoff=10,successive_loading_failiures_cutoff=100):
        unloaded_system_size=len(self.zeolite)
 
        LJ_sigmas=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteEditor/LJ_sigmas.csv",delimiter=",",dtype="float",usecols=1,skiprows=1)
        LJ_symbols=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteEditor/LJ_sigmas.csv",delimiter=",",dtype="str",usecols=0,skiprows=1)
        
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
    
    
    
    def defect_site_info(self, site_index):
        """
        Returns information about a defect site in the zeolite structure.

        Parameters:
            site_index (int): The index of the defect site (i.e. any lattice Si, Al, etc. atom)

        Returns:
            tuple: A tuple containing the following information:
                - suitability (bool): True if the site is suitable for a defect (obeys Lowenstein's rule), False otherwise.
                - site_neighbours_O (list): A list of indices of oxygen atoms that are neighbors of the defect site.
                - site_neighbours_Si (list): A list of indices of silicon atoms that are neighbors of the oxygen atoms 
                                             neighboring the defect site.
        """
        r_SiO_cutoff = 1.7 
        suitability = True  # Suitable site for defect (obeys Lowenstein's rule)
        site_neighbours = self.find_neighbours(site_index, r_SiO_cutoff)
        site_neighbours_O = self.atom_selector(site_neighbours, "O")

        if self.zeolite[site_index].symbol == "Al":
            suitability = False

        if not len(site_neighbours_O) == 4:
            suitability = False

        site_neighbours_Si = []
        for i in site_neighbours_O:
            O_neighbours = self.find_neighbours(i, r_SiO_cutoff)  # O atom neighbours of Si
            O_neighbours_Si = self.atom_selector(O_neighbours, "Si")

            location_of_input_atom=np.where(O_neighbours_Si == site_index)[0]
            if len(location_of_input_atom)==0:
                pass #If site is not Si, does not need to remove input index
            else:
                O_neighbours_Si = np.delete(O_neighbours_Si, np.where(O_neighbours_Si == site_index)[0][0])  
            if not len(O_neighbours_Si) == 1:
                suitability = False
            else:
                site_neighbours_Si.append(O_neighbours_Si[0])

        return suitability, site_neighbours_O, site_neighbours_Si
    
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
        
        LJ_sigmas=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteEditor/LJ_sigmas.csv",delimiter=",",dtype="float",usecols=1,skiprows=1)
        LJ_symbols=np.loadtxt("/data/fast-pc-02/hr492/ZeoliteEditor/LJ_sigmas.csv",delimiter=",",dtype="str",usecols=0,skiprows=1)
        
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



    def add_Al_defect(self, site_index, tag=None):
        """
        Substitutes an Al atom at a Si site in the zeolite structure and adds an H atom to a neighbouring O atom.
        Will only add the defect if the site is suitable (i.e. obeys Lowenstein's rule).
        Note: the H atom is added first and then the Al atom.

        Parameters:
        - site_index (int): The index of the site where the Al defect will be added.
        - tag (int): The tag to assign to the H atom (tag+1 assigned to Al). (By default, the tag is not set.)


        Returns:
        - bool: True if the defect was successfully added, False otherwise.
        """

        suitable, site_neighbours_O, site_neighbours_Si = self.defect_site_info(site_index)

        if not suitable:
            print('site not suitable for AlOH defect')
            return False

        site_choice = np.random.randint(0, 4)
        OH_site = site_neighbours_O[site_choice]
        adjacent_Si = site_neighbours_Si[site_choice]

        r_OSi = self.get_system().get_distance(OH_site, adjacent_Si, mic=True, vector=True)
        r_AlO = self.get_system().get_distance(site_index, OH_site, mic=True, vector=True)
        r_OH = np.cross(r_OSi, r_AlO)
        defect_site_pos = self.get_system()[site_index].position
        H_pos = 1.1 * r_OH / np.linalg.norm(r_OH) + self.get_system()[OH_site].position
        if tag is None:
            new_H = ase.Atoms('H', positions=[H_pos])
        else:
            new_H=ase.Atoms('H', positions=[H_pos],tags=tag)
        self.zeolite.extend(new_H)
        # Replace O at Al_site with Al
        del self.zeolite[site_index]
        if tag is None:
            new_Al=ase.Atoms('Al', positions=[defect_site_pos])
        else:
            new_Al=ase.Atoms('Al', positions=[defect_site_pos],tags=tag+1)
        self.zeolite.extend(new_Al)
        return True
    
    def remove_Al_defect(self, site_index, tag=None):
        """
        Removes an Al heteroatom and corresponding H from the zeolite structure and replaces it with a Si atom.

        Args:
            site_index (int): The index of the Al atom of the defect site in the zeolite structure.
            tag (str, optional): An optional tag for the new Si atom. Defaults to None (adds no tag).

        Returns:
            None
        """
        # Turns Al into Si
        Al_pos = self.zeolite[site_index].position

        if tag is None:
            new_Si = ase.Atoms('Si', positions=[Al_pos])
        else:
            new_Si = ase.Atoms('Si', positions=[Al_pos], tags=tag)

        self.zeolite.extend(new_Si)

        # Remove Al atom and H atom
        self.remove_atoms([site_index])
        self.remove_atoms([site_index-1])
        pass
    
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
    
    
    def fill_Al_defects(self,n_add,n_trials=100,printing=False):
        
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

        if printing==True:
            if(success):
                print("Successully added Al defects")
            else:
                print("Failed after adding ",n_success ," Al defects.")        
    

    def fill_majority_Al_defects(self, Al_to_max_Al_ratio=None,printing=False):
        """
        Fills a purely silicious zeolite structure with chains of NN neighbour Al defects. Then removes random Al sites to reach desired Al filling.

        Parameters:
            Al_to_max_Al_ratio (float): The desired fraction of added Al sites to remove after filling.

        Returns:
            int or tuple: If `Al_to_max_Al_ratio` is None, returns the number of Al defects added to the zeolite structure.
                         If `Al_to_max_Al_ratio` is not None, returns a tuple containing the maximum number of Al defects,
                         the initial number of Si sites, and the number of Al defects removed to reach the desired Al filling.
        """

        def start_new_chain(possible_sites):
            
            #If cannot start new chain - breaks while loop
            #Returns target site index for new chain start
            if printing==True:
                print("Starting new chain")
            
            unsuitable_Si_site_indices=self.get_indices(unsuitable_Si_sites)
            possible_sites = possible_sites[~np.isin(possible_sites, unsuitable_Si_site_indices)]
            
            found_site_for_new_chain=False

            for site in possible_sites:
                site_suitability = self.defect_site_info(site)[0]
                if site_suitability:
                    if printing==True:
                        print('Found new chain start.')
                    return site
                
            
            if printing == True and found_site_for_new_chain==False:
                print('Could not find new chain start. Zeolite maximally filled.')
            #If no suitable site found, zeolite maximally filled
            return None    
        
        #Initialising variables
        num_Al = 0
        possible_sites = self.atom_selector(np.arange(0, len(self.zeolite), 1), "Si")
        initial_num_Si=len(possible_sites) 
        num_atoms=len(self.zeolite)
        initial_Si_sites = self.atom_selector(np.arange(0,num_atoms,1), "Si") 
        target_site = np.random.choice(initial_Si_sites)
        filling = True # True when not reached desired Al/Si ratio
        unsuitable_Si_sites = np.array([])
        possible_Al_sites = np.array([])

        while filling:
            
            #Adding Al at target site 
            if printing == True:
                print('Trial stie element: ',self.zeolite[target_site].symbol)
                print('Trial site position: ',self.zeolite[target_site].position)
            successful_Al_substitution = self.add_Al_defect(target_site, tag=num_atoms)
            if not successful_Al_substitution: 
                #This might happen as a previously suitable site may have become unsuitable due to new Al atoms
                possible_Al_sites=possible_Al_sites[possible_Al_sites!=self.get_tags(target_site)]
                unsuitable_Si_sites=np.append(unsuitable_Si_sites,self.get_tags(target_site))
            else:             
                num_Al += 1   
                num_atoms+=1   
                if printing==True:
                    print(num_Al," Al sites added") 
            
            #Adding neighbours of added Al to unsuitable site list
            added_Al_index=num_atoms-1
            neighbours=self.get_tags(self.defect_site_info(added_Al_index)[2])
            unsuitable_Si_sites=np.append(unsuitable_Si_sites,neighbours)

            #Finding (Lowenstein-suitable) indices of NN neighbours of added Al
            new_NN_neighbours = self.find_NN_neighbours(added_Al_index)
            if len(new_NN_neighbours) == 0: 
                pass
            else:
                suitable_sites=np.array([self.defect_site_info(index)[0]==True for index in new_NN_neighbours])  
                suitable_new_NN_neighbours=new_NN_neighbours[suitable_sites]
                unsuitable_new_NN_neighbours=new_NN_neighbours[~suitable_sites]

                #Adding non-Lowenstein NN neighobours to unsuitable Si sites list
                unsuitable_new_NN_neighbour_tags=self.get_tags(unsuitable_new_NN_neighbours)
                unsuitable_Si_sites=np.append(unsuitable_Si_sites,unsuitable_new_NN_neighbour_tags)
                unsuitable_Si_sites=np.unique(unsuitable_Si_sites)

                #Adding Lowenstein compatible NN neighbours to possible Al sites
                new_possible_Al_sites= self.get_tags(suitable_new_NN_neighbours)
                possible_Al_sites=np.append(possible_Al_sites,new_possible_Al_sites)
                possible_Al_sites=np.unique(possible_Al_sites)

            #Picking new target; if no new available targets, tries to start new chain
            if len(possible_Al_sites) == 0: 
                target_site = start_new_chain(possible_sites)
                if target_site==None:
                    filling = False
            else:
                target_site_tag = np.random.choice(possible_Al_sites)
                target_site = self.get_indices([target_site_tag])[0]
                possible_Al_sites=possible_Al_sites[possible_Al_sites!=target_site_tag]
        

        #Finishing or removing some Al sites
        if Al_to_max_Al_ratio is None:
            return num_Al
        else:
            #Removing Al atoms to reach desired Al/Si ratio
            max_Al=num_Al
            num_Al_removed=int(max_Al-max_Al*Al_to_max_Al_ratio)

            added_Al_site_indices=self.atom_selector(np.array(range(0,len(self.zeolite))),"Al")
            Al_sites_to_remove_indices=np.random.choice(added_Al_site_indices,num_Al_removed)
            Al_sites_to_remove_tags=self.get_tags(Al_sites_to_remove_indices)
            Al_sites_to_remove_tags=np.unique(Al_sites_to_remove_tags)

            for tag in Al_sites_to_remove_tags:
                self.remove_Al_defect(self.get_indices([tag])[0])


        return max_Al,initial_num_Si,num_Al_removed
        
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
            

    def find_NN_neighbours(self, atom_index):
        """
        Finds the next nearest (NN) neighbours of a given atom.

        Parameters:
        - atom_index (int): The index of the atom for which to find NN neighbours.

        Returns:
        - NN_neighbours_Si (numpy.ndarray): An array containing the (int) indices of the NN neighbours.

        """

        site_neighbours_Si = self.defect_site_info(atom_index)[2]
        NN_neighbours_Si = np.array([])

        for site in site_neighbours_Si:
            site_NN_neighbours = self.defect_site_info(site)[2]
            NN_neighbours_Si = np.append(NN_neighbours_Si, site_NN_neighbours)

        NN_neighbours_Si = np.unique(np.array(NN_neighbours_Si))

        # Deleting input site from list
        if len(np.where(NN_neighbours_Si == atom_index)[0]) == 0:
            pass  # If site is not Si, does not need to remove input index
        else:
            NN_neighbours_Si = np.delete(NN_neighbours_Si, np.where(NN_neighbours_Si == atom_index)[0][0])

        return NN_neighbours_Si.astype(int)


        
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

        
    def get_tags(self, indices):
        """
        Returns the tags of the zeolite at the specified indices.

        Parameters:
        - indices (array): An arrray of indices specifying the zeolite(s) to retrieve tags from.

        Returns:
        - array: An array of tags corresponding to the zeolite(s) at the specified indices.
        """
        return self.zeolite.get_tags()[indices]

    def get_indices(self, tags):
        """
        Returns the indices of the given tags in the zeolite object.

        Parameters:
        tags (array): An array of tags to search for.

        Returns:
        list: An array of indices corresponding to the given tags.
        """

        tags=np.array(tags)
        return np.array( [np.where(self.zeolite.get_tags() == tag)[0][0] for tag in tags] ) 

    
    

