from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io import read, write
import numpy as np
import time
import os
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pdb_name', type=str, required=True, help='Name of .pdb file')
    parser.add_argument('--sim_dir', type=str, required=True, help='Path to directory containing .pdb file')
        
    args = parser.parse_args()
    
    run_md(args.sim_dir,args.pdb_name)
        

def run_md(dir_name,pdb_name):
    init_config_pdb = dir_name+pdb_name

    from mace.calculators.foundations_models import mace_mp

    calculator = mace_mp(model='model_2_swa.model',  dispersion=True) 

    print("")
    print("Loaded Model")
    print("")
    init_conf = read(init_config_pdb,'0') # why is the ,'0' here?  
    init_conf.set_calculator(calculator)

    print("Loaded Config")

    MaxwellBoltzmannDistribution(init_conf, temperature_K=300)

    dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=300, friction=5e-2)
    def write_frame():
        dyn.atoms.write(dir_name+'traj.xyz', append=True)
    dyn.attach(write_frame, interval=4)
    dyn.attach(MDLogger(dyn, init_conf, dir_name+'md.log', header=True, stress=False,peratom=True, mode="a"), interval=1)
    dyn.run(100)
    print("MD finished!")


if __name__ == '__main__':
    main()
