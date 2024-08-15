import ase.io
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
import matplotlib.animation as animation
from collections import defaultdict
from copy import deepcopy


"""Module Functions"""

def cartesian_to_cylindrical(cartesian_coods):
    #input is cartesian coord vector as np.array
    x,y,z= cartesian_coods
    theta= np.arctan(y/x)
    r = np.sqrt(x**2+y**2)
    return np.array([r,theta,z])
    

def get_cylindrical_coordinates(analyser,indices):
    positions = analyser.get_positions(indices) # one Nx3 list per frame
    cylindrical_coords = []
    for frame in positions:
        frame_positions= []    
        num_atoms = len(frame)
        for atom_index in range(num_atoms):
            atomic_cartesian_coords = frame[atom_index]
            atomic_cylindrical_coords = cartesian_to_cylindrical(atomic_cartesian_coords)
            frame_positions.append(atomic_cylindrical_coords)
        cylindrical_coords.append(frame_positions)
    cylindrical_coords = np.array(cylindrical_coords,dtype=object)
    return cylindrical_coords


def water_cylindrical_heat_map(analyser,r_range = None):
    cylindrical_coords = get_cylindrical_coordinates(analyser,analyser.water_O_indices)
    all_cylindrical_coords = []
    for frame in cylindrical_coords:
        for coordinate in frame:
            all_cylindrical_coords.append(coordinate)
    plotting_coordinates = []
    if r_range is not None:
        for coordinate in all_cylindrical_coords:
            if r_range[0]<coordinate[0] and coordinate[0]<r_range[1]:
                plotting_coordinates.append(coordinate)
    else:
        plotting_coordinates = all_cylindrical_coords
    plotting_coordinates=np.array(plotting_coordinates)
    x=plotting_coordinates[:,2]
    y=plotting_coordinates[:,1]
    plt.hist2d(x,y,bins=(300,300))
    
        
    
def is_in_z_bounds(z,z_bounds):
    in_bounds=False
    for z_bound in z_bounds:
        if z_bound[0] < z < z_bound[1]:
            in_bounds=True
    return in_bounds
    




def proton_2D_heat_map(analyser):
    proton_indices = analyser.free_proton_indices
    proton_positions = analyser.get_positions(proton_indices)
    x=np.array([])
    y=np.array([])
    z=np.array([])
    for i in range(analyser.num_frames):
        x=np.append(x, proton_positions[i][:,0])
        y=np.append(y, proton_positions[i][:,1])
        z=np.append(z ,proton_positions[i][:,2]) 
    plt.hist2d(x,y,bins=(300,300))



def plot_water_heat_map(analyser,coords=[0,1],plot_name=None,plot_framework_stride=None,z_bounds = None,starting_frame=0):

    #Plotting water
    print('Plotting Water')
    water_O_pos=analyser.get_positions(analyser.water_O_indices)[starting_frame:]
    x=np.array([])
    y=np.array([])
    z=np.array([])
    num_frames = len(water_O_pos[starting_frame:])
    for i in range(num_frames):
        
        if z_bounds is not None:
            for pos in water_O_pos[i]:
                if is_in_z_bounds(pos[2],z_bounds):
                    x=np.append(x, pos[coords[0]])
                    y=np.append(y, pos[coords[1]])
        else:
            x=np.append(x, water_O_pos[i][:,coords[0]])
            y=np.append(y, water_O_pos[i][:,coords[1]])

    print(len(x),' coordinates plotted.')
    cmap = plt.cm.viridis
    cmap.set_under('white')
    plt.hist2d(x,y,bins=(100,100),cmap=cmap, vmin=0.01)

    #Plotting framework O
    if plot_framework_stride is not None:
        print('Plotting Framework O')
        framework_pos=analyser.get_positions(analyser.framework_O_indices)[::plot_framework_stride]
        x_framework=np.array([])
        y_framework=np.array([])
        z_framework=np.array([])
        for i in range(len(framework_pos)):
            if z_bounds is not None:
                for pos in framework_pos[i]:
                    if is_in_z_bounds(pos[2],z_bounds):
                        x_framework=np.append(x_framework, pos[coords[0]])
                        y_framework=np.append(y_framework, pos[coords[1]])
            else:
                x_framework=np.append(x_framework, framework_pos[i][:,coords[0]])
                y_framework=np.append(y_framework, framework_pos[i][:,coords[1]])

        plt.scatter(x_framework,y_framework,color='red')
        plt.xlim(x_framework.min(), x_framework.max())
        plt.ylim(y_framework.min(), y_framework.max())

    #Plotting details
    plt.gca().set_aspect('equal', adjustable='box')
    if coords==[0,1]:
    
        plt.xlabel(f'x [$\AA$]')
        plt.ylabel(f'y [$\AA$]')
        
    if plot_name is not None:
        plt.savefig(plot_name+'.png',dpi=300)
    
    plt.show()



def plot_H_bond_network(analyser, coords=[0,1], alpha = 0.1,stride=1 ,plot_name=None, plot_framework=False, color_gradient = 'inferno'):

    #Getting simulation aggregated positions of H-bonded O atoms 
    num_O=len(analyser.O_indices)
    O_1_positions = []
    O_2_positions = [] 

    #Plotting framework O atoms
    if plot_framework:
        framework_pos=analyser.get_positions(analyser.framework_O_indices)
        x_framework=np.array([])
        y_framework=np.array([])
        z_framework=np.array([])
        for i in range(analyser.num_frames):
            x_framework=np.append(x_framework, framework_pos[i][:,coords[0]])
            y_framework=np.append(y_framework, framework_pos[i][:,coords[1]])

        plt.scatter(x_framework,y_framework,color='red')
        plt.xlim(x_framework.min(), x_framework.max())
        plt.ylim(y_framework.min(), y_framework.max())


    #Plotting H-bonds 
    for frame_index in range(analyser.num_frames)[::stride]:

        connectivity = analyser.get_H_bond_connectivity(frame_index)

        for i in range(num_O):
            for j in range(i):
                if connectivity[i][j]:
                    index_i = analyser.O_indices[i]
                    index_j = analyser.O_indices[j]
                    pos_i = analyser.trajectory[frame_index][index_i].position
                    pos_j = analyser.trajectory[frame_index][index_j].position
                    O_1_positions.append(pos_i)
                    O_2_positions.append(pos_j)

    x = np.array(O_1_positions)
    y = np.array(O_2_positions)

    #Colours bond based on its component into plane

    #vector is r1 + dr12 where dr12 is displacemnet computed using MIC
    num_H_bonds = len(x)
    plt.gca().set_aspect('equal', adjustable='box')
    for i in range(num_H_bonds):
        plt.plot([x[i][coords[0]], y[i][coords[0]]], [x[i][coords[1]], y[i][coords[1]]], alpha=alpha,color='grey')

    if coords==[0,1]:
    
        plt.xlabel(f'x [$\AA$]')
        plt.ylabel(f'y [$\AA$]')

    
    if plot_name is not None:
        plt.savefig(plot_name+'.png',dpi=300)
    
    plt.show()
    
def VFI_coordinate_transform(position,cell):
    #NOTE: THIS MAY BE SPECIFIC TO THIS TYPE OF ZEOLITE
    #This might just work for VFI as the four corners r... all represent the centre of the channel
    
    x,y,z =  position[0],position[1],position[2]
    
    v1=cell[0]
    v2=cell[1]
    r0=np.array([0,0,0])
    r1=v1
    r2=v2
    r3=v1+v2
    centre_points = [r0,r1,r2,r3]
    
    distances = [ ( (x-r[0])**2 + (y-r[1])**2 ) for r in centre_points]
    shortest_distance  = min(distances)
    nearest_corner_index=np.where(distances==shortest_distance)[0][0]
    nearest_corner_position = centre_points[nearest_corner_index]
    return np.array( [x-nearest_corner_position[0] , y-nearest_corner_position[1] ,z] )



def animate_trajectory(trajectory_list,color_list,animation_name,coord_indices=[0,1]):
    
    
    fig, ax = plt.subplots()
   
    scatter_artists=[] 
    num_trajectories  = len(trajectory_list)
    num_frames=len(trajectory_list[0])
 
    for frame_index in range(num_frames):
        frame_scatter_artist=[]
        for trajectory_index in range(num_trajectories):
            trajectory = trajectory_list[trajectory_index]
            coods1=trajectory[frame_index][:, coord_indices[0]]
            coords2 =trajectory[frame_index][:, coord_indices[1]]
            scatter_artist = ax.scatter(coods1, coords2, color = color_list[trajectory_index])
            frame_scatter_artist.append(scatter_artist)
        progress = frame_index / num_frames
        title = ax.text(0.5, 1.05, f't={progress:.2f}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        scatter_artists.append(frame_scatter_artist+[title])
        

    ani = animation.ArtistAnimation(fig, scatter_artists, interval=1, blit=True)
                                
    
    ani.save(filename=animation_name+".gif", fps=60,writer="pillow")
    print('success')


def get_simulation_aggregated_positions(analyser,indices):
    positions= analyser.get_positions(indices)
    x=np.array([])
    y=np.array([])
    z=np.array([])
    for i in range(analyser.num_frames):
        x=np.append(x, positions[i][:,0])
        y=np.append(y, positions[i][:,1])
        z=np.append(z ,positions[i][:,2]) 
    return x,y,z
        


def histogram_to_plot(data, bins=10, range=None, density=False, **kwargs):
    """
    Convert np.histogram data to a plot using plt.plot.

    Parameters:
    - data: array-like, the input data to be histogrammed.
    - bins: int or sequence, optional, number of bins or bin edges.
    - range: tuple, optional, the lower and upper range of the bins.
    - density: bool, optional, if True, the result is the value of the probability density function.
    - kwargs: additional keyword arguments for plt.plot.

    Returns:
    - None
    """
    # Generate histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot data
    plt.plot(bin_centers, counts, **kwargs)
    plt.xlabel('Bin Centers')
    plt.ylabel('Counts')
    plt.title('Histogram to Plot')
    plt.show()


    
class TrajectoryAnalyser:
    
    def __init__(self,input_trajectory,stride=1,coordinate_transform_function=None):
        
        """Initialisation"""
        self.coordinate_transform_function=coordinate_transform_function
        self.trajectory = input_trajectory.copy()[::stride]
        self.num_atoms = len( self.trajectory[0] ) 
        self.num_frames = len(self.trajectory)
        self.cell=self.trajectory[0].cell

        """Transforming coordinates"""
        if self.coordinate_transform_function is None:
            #default is to wrap to unit cell
            print('No transformation function given.')
            for frame in self.trajectory:
                frame.wrap()
        else:
            print('Using ',coordinate_transform_function ,' to transform coordinates.')
            for frame in self.trajectory:
                frame.wrap()
                new_positions=[]
                for atom in frame:
                    position = atom.position
                    new_positions.append(VFI_coordinate_transform(position,self.cell))
                frame.positions=new_positions

        """Finding indices of important species""" 
        #These indices are constant for the whole simulation, and is an N-dim array
        self.O_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'O' for atom in self.trajectory[0]]]
        self.H_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'H' for atom in self.trajectory[0]]]
        self.Si_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'Si' for atom in self.trajectory[0]]]
        self.Al_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'Al' for atom in self.trajectory[0]]]
        self.framework_O_indices = self.find_framework_O_indices()
        self.water_O_indices = np.setdiff1d(self.O_indices,self.framework_O_indices)


        #These indices are frame dependent, and thus TxN-dim arrays
        if len(self.Al_indices) != 0:
            #Don't need these if there are no AlOH
            self.voronoi_dicts=  []
            self.free_proton_indices = []
            self.hydronium_O_indices = []
            self.framework_hydroxyl_O_indices = []  
            self.water_H_indices = []


            for frame_index in range( len(self.trajectory) ):
                distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)
                
                frame_voronoi_dict = self.find_voronoi_dict(frame_index,distance_matrix)
                frame_hydronium_O_indices = self.find_hydronium_O_indices(frame_index,frame_voronoi_dict,distance_matrix)
                frame_framework_hydroxyl_O_indices=self.find_framework_hydroxyl_O_indices(frame_voronoi_dict)
                frame_free_proton_indices= self.find_free_proton_indices(frame_index,frame_voronoi_dict,frame_framework_hydroxyl_O_indices,frame_hydronium_O_indices,distance_matrix)
                frame_water_H_indices= np.setdiff1d(self.H_indices,frame_free_proton_indices)
                #print('frame_voronoi_dict',frame_voronoi_dict)
                #print('frame_hydronium_O_indices',frame_hydronium_O_indices)
                #print('frame_framework_hydroxyl_O_indices',frame_framework_hydroxyl_O_indices)
                


                
                self.voronoi_dicts.append(frame_voronoi_dict)
                self.free_proton_indices.append(frame_free_proton_indices)
                self.hydronium_O_indices.append(frame_hydronium_O_indices)
                self.framework_hydroxyl_O_indices.append(frame_framework_hydroxyl_O_indices)
                self.water_H_indices.append(frame_water_H_indices)
            
            self.water_H_indices=np.array(self.water_H_indices)    
            self.free_proton_indices = np.array(self.free_proton_indices)
            self.hydronium_O_indices = np.array(self.hydronium_O_indices,dtype=object)
            self.framework_hydroxyl_O_indices = np.array(self.framework_hydroxyl_O_indices,dtype=object)

    
    def find_voronoi_dict(self,frame_index,distance_matrix=None):
        if distance_matrix is None:
            distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)
        
        voronoi_dictionary = {i: [] for i in self.O_indices}
        
        for H_index in self.H_indices:                       
            distances=distance_matrix[H_index][self.O_indices]
            
            smallest = np.min(distances)
            tolerance=1e-7
            closest_O_index = np.where( abs (distance_matrix[H_index] - smallest )< tolerance )[0][0]
            voronoi_dictionary[closest_O_index].append(int(H_index))

        return voronoi_dictionary


    def find_hydronium_O_indices(self,frame_index,voronoi_dict,distance_matrix=None):
        hydronium_mask = [ len(voronoi_dict[i]) > 2 for i in self.water_O_indices ]
        hydronium_O_indices = self.water_O_indices[hydronium_mask]
        return hydronium_O_indices

    def find_framework_hydroxyl_O_indices(self,voronoi_dict):
        AlOH_mask = [ len(voronoi_dict[i]) > 0 for i in self.framework_O_indices ]
        AlOH_O_indices = self.framework_O_indices [AlOH_mask]
        return AlOH_O_indices
        
    
    def find_free_proton_indices(self,frame_index,voronoi_dict,framework_hydroxyl_O_indices,hydronium_O_indices,distance_matrix=None):
        if distance_matrix is None:
            distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)

        
        #Add all H attached to framework O
        AlOH_protons = np.array([]) 
        for i in framework_hydroxyl_O_indices:
            AlOH_protons=np.append(AlOH_protons,np.array(voronoi_dict[i]))

        #Attach furtherst H in voronoi region of each O
        hydronium_protons = np.array([])
        for i in hydronium_O_indices:
            H_indices = voronoi_dict[i]
            distances = [distance_matrix[i][j] for j in H_indices]
            largest_distance = max(distances)
            tolerance=1e-7
            furthest_H_index = np.where( abs (distance_matrix[i] - largest_distance )< tolerance )[0][0]
            hydronium_protons =np.append(hydronium_protons,furthest_H_index)
        

        free_protons = np.append(hydronium_protons,AlOH_protons)

        #print('frame_index',frame_index)
        ##print('AlOH_protons',AlOH_protons)
        #print('hydronium_protons',hydronium_protons)
        #print('')
        
        
        return free_protons


    def find_neighbours(self, atom_index, r_cutoff,frame_index=0, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
        
        neighbours = np.intersect1d(
            np.where(distance_matrix[atom_index] < r_cutoff),
            np.where(distance_matrix[atom_index] != 0)
        )
        return neighbours

    
    def find_framework_O_indices(self):
        distance_matrix = self.trajectory[0].get_all_distances(mic=True)
        framework_O_indices= []
        for O_index in self.O_indices:
            neighbours=self.find_neighbours(O_index, 1.7, distance_matrix=distance_matrix)   
            if sum( [ self.trajectory[0][index].symbol == 'Si' for index in neighbours]  ) != 0:
                framework_O_indices.append(O_index)
        return np.array(framework_O_indices)
        

    
    """ Methods for H-bond network analyis """


    def H_bond_geometry_check(self,frame_index,O_D_index,H_index,O_A_index,r_OO_c = 3.5, r_OH_c = 2.4, theta_c = 30):
        """Takes indices of O_donor , H , O_acceptor and tells you whether the 3 have the geometry of a H bond """

        v_1 = self.trajectory[frame_index].get_distances(O_D_index,H_index,mic=True,vector=True) [0]
        v_2 = self.trajectory[frame_index].get_distances(O_D_index,O_A_index,mic=True,vector=True) [0]
        v_3 = self.trajectory[frame_index].get_distances(H_index,O_A_index,mic=True,vector=True) [0]
        
        r_OH = np.linalg.norm(v_3)
        r_OO = np.linalg.norm(v_2)
        cos_theta = np.dot(v_2,v_1) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        cos_theta_c = np.cos(30 * np.pi / 180)
    
        return r_OH < r_OH_c and cos_theta > cos_theta_c and r_OO<r_OO_c

    
    
    def is_H_bonded(self,frame_index,O_index_1,O_index_2,distance_matrix=None, r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):
        #Note, O_1 is donor and O_2 is acceptor
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)

        
        def find_local_H_indices(O_index):
            distances = distance_matrix[O_index]
            H_distances = distances[self.H_indices]
            local_H_distances = H_distances[[ 1e-7 < distance and distance < r_OH_c for distance in H_distances]]
            local_H_indices = np.where(np.isin(distances,local_H_distances))[0]
            return local_H_indices

        #Finding protons in between O atoms
        local_O_H_indices_1 = find_local_H_indices(O_index_1)
        local_O_H_indices_2 = find_local_H_indices(O_index_2)
        
        if len(local_O_H_indices_1) ==0 or len(local_O_H_indices_2) ==0:
            return False
        common_H_indices = np.intersect1d(local_O_H_indices_1,local_O_H_indices_2)
        if len(common_H_indices)==0:
            return False

        #Finding 'bonding proton' which has shortest OHO path length
        path_lengths= []

        for H_index in common_H_indices:
            common_H_pos=self.trajectory[frame_index][H_index].position
            path_length = distance_matrix[O_index_1][H_index] + distance_matrix[H_index][O_index_2]
            path_lengths.append(path_length)
   
        smallest_path_length = min(path_lengths)
        candidate_H_index = common_H_indices[ np.where( abs(path_lengths - smallest_path_length)<1e-7 )[0][0] ]       
        
        is_H_bond = self.H_bond_geometry_check(frame_index,O_index_1,candidate_H_index,O_index_2,r_OO_c,r_OH_c,theta_c)

        
        return is_H_bond
                    

    def get_H_bond_connectivity(self,frame_index,distance_matrix=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):
        
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
                            
        num_Os=len(self.O_indices)
        connectivity_matrix=np.full((num_Os, num_Os), False, dtype=bool)
        for i in range(num_Os):
            for j in range(i):
                O_index_i=self.O_indices[i]
                O_index_j=self.O_indices[j]
                H_bond=self.is_H_bonded(frame_index,O_index_i,O_index_j,distance_matrix,r_OO_c , r_OH_c , theta_c )
                if H_bond:
                    connectivity_matrix[i][j] = True
                    connectivity_matrix[j][i] = connectivity_matrix[i][j]
                    
        return connectivity_matrix
    

    def get_H_bond_clusters(self,frame_index,distance_matrix=None,H_bond_connectivity=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):

        clusters = []
        unassigned = self.O_indices
        
        def assign_to_cluster(index):
            """Returns whether index is part of a cluster"""

            
            nonlocal new_cluster
            nonlocal unassigned

            #removing index from unassigned
            location_of_index=np.where(unassigned==index)[0]
            unassigned = np.delete(unassigned,location_of_index)
            
            selection_index = np.where(self.O_indices == index)[0][0] 
            neighbours_mask = H_bond_connectivity[selection_index]
            
  
            if sum(neighbours_mask) == 0:
   
                return False

            
            neighbours= self.O_indices[neighbours_mask]
   
            new_cluster=np.append(new_cluster,index)
         
            #Removing index from unassigned lis
            
            #selection index is index in list of O indices (as opposed to list of all atoms)
            unassigned_neighbours = np.intersect1d(neighbours,unassigned)
            for unassigned_neighbour_index in unassigned_neighbours:
                assign_to_cluster(unassigned_neighbour_index)
            return True

    
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
        
        if H_bond_connectivity is None:
            H_bond_connectivity = self.get_H_bond_connectivity(frame_index,distance_matrix,r_OO_c , r_OH_c , theta_c  )
        

       
        while len(unassigned)>0:
     
                
            O_index = unassigned[0]
            
            new_cluster =np.array([])
            belongs_to_cluster = assign_to_cluster(O_index)
        
            
            if belongs_to_cluster: 
                clusters.append(new_cluster)
            else:
                continue
            
        return clusters

    
    
    """Getters"""  

    def get_positions(self,atom_indices,frame_indices=None):
        
        
 
        
        if frame_indices is None:
            frame_indices=np.arange(0,self.num_frames,1) #if no frame indices specified, assume whole trajectory
            

        
        positions=[]
        for frame_index in frame_indices:
            if np.issubdtype(atom_indices.dtype, np.integer):
                #Indended for cases where indices are constant throughout sim
                #Be careful, if inhomogeneous, indices will be 1-dim with N-dim elements
                frame_atom_indices=atom_indices
            else:
                frame_atom_indices=atom_indices[frame_index].astype(int)
            for atom_index in frame_atom_indices:
                frame_positions = np.array([ self.trajectory[frame_index][atom_index].position for atom_index in frame_atom_indices] )
            positions.append(frame_positions)

        return np.array(positions,dtype=object)
         