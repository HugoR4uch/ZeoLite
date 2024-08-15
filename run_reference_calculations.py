import ase.io 
import numpy as np
import time
import os
import sys
from itertools import islice
import aml
import glob
import argparse


#sys.path.append('/data/fast-pc-02/hr492/software/packages/aml')
#import aml


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--code_name', type=str, required=True, help='Code name for config (also name of .pdb file, but without the .pdb bit)')
    parser.add_argument('--sim_dir', type=str, required=True, help='Path to directory containing .pdb file')
    args = parser.parse_args()

    #Checks to see if time from md.log matches desired end time
    conf = ase.io.read(args.sim_dir+'traj.xyz','::100')
    num_frames = len(conf)*100
    print(num_frames)
    
    #If simulation not finished, will start new sim by changing .pdb
    run_ref_calcs(args.sim_dir,args.code_name,num_frames)
    


def run_ref_calcs(dir_name,code_name,num_frames):
    init_config_pdb = dir_name+code_name+'.pdb'
    num_frames_collected = 3 #They also take .pdb 
    stride_trj = num_frames//num_frames_collected

    fn_positions = dir_name+'traj.xyz'
    fn_pdb = dir_name+code_name+'.pdb'

    frames = aml.read_frames_mdtraj(fn_in=fn_positions, top=fn_pdb, stride=stride_trj)
    structures = aml.Structures.from_frames(frames)

    # construct calculator
    launcher = aml.ProcessLauncher(mode='OpenMPI', n_slots=2, n_core_task=8)
    cp2k = aml.CP2K(fns_input_in=['/data/fast-pc-02/hr492/ProductionRunsAnalysis/revPBE-D3.inp'], cmd_cp2k='/data/fast-pc-01/software/cp2k/exe/fast-pc/cp2k.popt', directory='./ref-calc', launcher=launcher)
    
    
    
    # use the calculator for all the selected structures
    print('running calculations')
    cp2k.run(structures, label='revPBE-D3')
    print('Done running calculation.')
    print()
    
    # check that the structure now has energy and forces
    E = structures[0].properties['revPBE-D3'].energy
    print(f'{structures[0]}, {structures[0].properties}, E={E}')
    
    # saves structures
    structures.to_file(code_name+'.data', label_prop='revPBE-D3')
    

if __name__ == '__main__':
    main()
