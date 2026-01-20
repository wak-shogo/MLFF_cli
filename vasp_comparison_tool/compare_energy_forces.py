# compare_energy_forces.py
import argparse
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import random
import subprocess
import tempfile
import shutil
from pathlib import Path

import torch

from ase.io import read, write
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.outputs import Vasprun
# --- 追加: 補正用モジュール ---
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

def get_mlff_calculator(model_name, use_device='cuda'):
    """
    Returns an ASE calculator for the specified MLFF model.
    """
    if not torch.cuda.is_available():
        use_device = 'cpu'
        print("Warning: CUDA not available. Using CPU for MLFF calculation.")

    if model_name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=use_device)
    elif model_name == "MatterSim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=use_device)
    elif model_name == "Orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_inf_omat(device=use_device, precision="float32-high")
        return ORBCalculator(orbff, device=use_device)
    elif model_name == "MatRIS":
        sys.path.insert(0, str(Path(__file__).parent / 'MatRIS'))
        from matris.applications.base import MatRISCalculator
        return MatRISCalculator(model='matris_10m_oam', task='efsm', device=use_device)
    elif model_name == "NequipOLM":
        raise NotImplementedError("NequipOLM model support is not yet implemented in this tool.")
    else:
        raise ValueError(f"Unknown MLFF model specified: {model_name}")

def calculate_mlff(atoms, mlff_model):
    """
    Calculates energy and forces using a Machine Learning Force Field.
    """
    print(f"--- Calculating with MLFF model: {mlff_model} ---")
    try:
        calculator = get_mlff_calculator(mlff_model)
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print("--- MLFF Calculation Finished ---")
        return energy, forces
    except Exception as e:
        print(f"Error during MLFF calculation: {e}")
        return None, None

def prepare_and_submit_vasp(atoms, structure_name, output_dir):
    """
    Prepares VASP input files and submits the job to the PBS queue.
    Returns the job ID and the working directory path.
    """
    
    vasp_calc_base_dir = Path(output_dir) / "vasp_calcs"
    tmpdir_path = vasp_calc_base_dir / f"vasp_{structure_name}"
    tmpdir_path.mkdir(parents=True, exist_ok=True)
    # print(f"Created VASP directory: {tmpdir_path}")

    try:
        # 1. Convert ASE Atoms to Pymatgen Structure
        structure = AseAtomsAdaptor.get_structure(atoms)
        
        # 2. Write VASP input files using MPRelaxSet (Standard MP settings)
        vasp_input_set = MPRelaxSet(structure, user_incar_settings={"NSW": 0, "IBRION": -1})
        vasp_input_set.write_input(tmpdir_path)
        
        # --- Dynamic Resource Allocation ---
        num_atoms = len(atoms)
        ATOMS_PER_NODE = 160
        PER_NODE = 16 # Keep consistent with original script's density (16 procs per 64-core node)
        
        num_nodes = max(1, (num_atoms + ATOMS_PER_NODE - 1) // ATOMS_PER_NODE)
        total_np = num_nodes * PER_NODE
        
        print(f"--- Preparing VASP calculation for {structure_name} ({num_atoms} atoms) ---")
        print(f"    -> Dynamic Resources: {num_nodes} Nodes, {total_np} MPI Ranks")

        # 3. Generate the PBS job script (Memory optimized version)
        job_script = f"""#!/bin/sh
#PBS -q J2
#PBS -l nodes={num_nodes}:ppn=64,walltime=02:00:00
#PBS -N vasp_{structure_name}
#PBS -j oe
#PBS -V

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > ./pbsnodelist

TOTAL_NP={total_np}
PER_NODE={PER_NODE}

. ~/Vasp-exe/module-18.sh

mpirun -bootstrap ssh -genv I_MPI_FABRICS shm:tcp \\
       -machinefile ./pbsnodelist \\
       -np $TOTAL_NP \\
       -ppn $PER_NODE \\
       $HOME/run_vasp.sh > vasp.out
"""
        job_script_path = tmpdir_path / "cluster_job.sh"
        with open(job_script_path, 'w') as f:
            f.write(job_script)

        # 4. Submit the VASP job
        print(f"Submitting VASP job for {structure_name}...")
        submit_result = subprocess.run(['qsub', 'cluster_job.sh'], cwd=tmpdir_path, capture_output=True, text=True, check=True)
        job_id = submit_result.stdout.strip().split('.')[0]
        print(f"VASP job submitted with ID: {job_id}")
        
        return job_id, tmpdir_path

    except Exception as e:
        print(f"Error submitting VASP job for {structure_name}: {e}")
        return None, tmpdir_path

def wait_for_jobs(job_ids):
    """
    Waits for a list of PBS job IDs to finish.
    """
    if not job_ids:
        return

    print(f"Waiting for {len(job_ids)} VASP jobs to complete: {job_ids}")
    username = os.getlogin()
    
    while True:
        try:
            qstat_result = subprocess.run(['qstat', '-u', username], capture_output=True, text=True)
            current_output = qstat_result.stdout
            
            # Check which jobs are still running
            remaining_jobs = [jid for jid in job_ids if jid in current_output]
            
            if not remaining_jobs:
                print("All monitored jobs have finished.")
                break
            else:
                # print(f"Jobs still running: {remaining_jobs}. Waiting 30 seconds...")
                time.sleep(30)
        except FileNotFoundError:
            print("Error: 'qstat' command not found. Cannot monitor job status.")
            break
        except Exception as e:
            print(f"Error during job monitoring: {e}")
            time.sleep(30)

def parse_vasp_result(structure_name, tmpdir_path):
    """
    Parses the VASP output (vasprun.xml) from the specified directory.
    Returns (raw_energy, corrected_energy, forces).
    """
    vasprun_path = tmpdir_path / "vasprun.xml"
    if not vasprun_path.exists() or vasprun_path.stat().st_size == 0:
        print(f"Error: vasprun.xml not found or empty for {structure_name}.")
        return None, None, None

    # Check if complete
    is_complete = False
    with open(vasprun_path, 'rb') as f:
        try:
            f.seek(-200, 2)
            if b'</modeling>' in f.read():
                is_complete = True
        except OSError:
            f.seek(0)
            if b'</modeling>' in f.read():
                is_complete = True
    
    if not is_complete:
        print(f"Error: {vasprun_path} appears to be incomplete.")
        return None, None, None
        
    try:
        vr = Vasprun(vasprun_path)
        if not vr.converged:
            print(f"Warning: VASP calculation for {structure_name} did not converge.")

        # --- Extract Raw Results ---
        raw_energy = vr.final_energy
        forces = vr.ionic_steps[-1]['forces']

        # --- Apply MP2020 Corrections ---
        corrected_energy = raw_energy # Default to raw if correction fails
        try:
            # Create a ComputedStructureEntry from the run
            entry = vr.get_computed_entry(inc_structure=True)
            
            # Apply compatibility scheme
            compat = MaterialsProject2020Compatibility()
            is_compatible = compat.process_entry(entry)
            
            if is_compatible:
                corrected_energy = entry.energy
                print(f"[{structure_name}] MP2020 Correction Applied: {raw_energy:.4f} -> {corrected_energy:.4f} eV")
            else:
                print(f"[{structure_name}] Entry incompatible with MP2020 scheme. Using Raw Energy.")

        except Exception as e:
            print(f"[{structure_name}] Error applying corrections: {e}. Using Raw Energy.")
        
        return raw_energy, corrected_energy, np.array(forces)

    except Exception as e:
        print(f"Error parsing VASP results for {structure_name}: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Compare MLFF and VASP energy/forces for given structures.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input .cif or .xyz file.")
    parser.add_argument("--mlff-model", type=str, required=True, help="MLFF model to use for comparison (e.g., CHGNet, MatRIS).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output CSV files.")
    parser.add_argument("--num-snapshots", type=int, default=1, help="Number of snapshots to select from an .xyz trajectory. Defaults to 1.")
    
    args = parser.parse_args()

    # --- 1. Setup Output Directory and Files ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energies_csv_path = output_dir / "energies.csv"
    # Write header initially
    pd.DataFrame(columns=["structure_name", "snapshot_num", f"{args.mlff_model}_energy", "VASP_raw_energy", "VASP_corrected_energy"]).to_csv(energies_csv_path, index=False)

    # --- 2. Load Structures ---
    input_path = Path(args.input_path)
    structures = []
    if input_path.suffix == '.cif':
        atoms = read(input_path)
        structures.append((input_path.stem, 0, atoms)) # snapshot_num = 0 for CIF
    elif input_path.suffix == '.xyz':
        all_frames = read(input_path, index=':')
        total_frames = len(all_frames)
        
        if total_frames <= args.num_snapshots:
            print(f"Warning: Trajectory has {total_frames} frames, requested {args.num_snapshots}. Using all frames.")
            indices = range(total_frames)
        else:
            # Equidistant sampling (isotropic)
            # Example: 100 frames, request 5.
            # step = 100 // 5 = 20
            # indices = 19, 39, 59, 79, 99 (0-based)
            step = total_frames // args.num_snapshots
            indices = [ (i + 1) * step - 1 for i in range(args.num_snapshots) ]
            # Ensure indices are within bounds (just in case)
            indices = [idx for idx in indices if idx < total_frames]
        
        print(f"Selected frame indices: {indices}")
        for i in indices:
            structures.append((f"{input_path.stem}_snapshot_{i}", i, all_frames[i]))
    else:
        print(f"Error: Unsupported file format '{input_path.suffix}'. Please use .cif or .xyz.")
        sys.exit(1)

    print(f"--- Found {len(structures)} structure(s) to process ---")

    # --- 3. Process in Batches ---
    BATCH_SIZE = 5
    all_results = [] # To store dicts of results

    for i in range(0, len(structures), BATCH_SIZE):
        batch = structures[i : i + BATCH_SIZE]
        print(f"\n>>> Processing Batch {i//BATCH_SIZE + 1} ({len(batch)} structures) <<<")
        
        batch_job_ids = []
        batch_dirs = {} # name -> path
        
        # 3a. Submit VASP Jobs
        for name, snap_idx, atoms_obj in batch:
            job_id, tmp_path = prepare_and_submit_vasp(atoms_obj, name, args.output_dir)
            if job_id:
                batch_job_ids.append(job_id)
                batch_dirs[name] = tmp_path
        
        # 3b. Run MLFF Calculations (while waiting)
        print("--- Running MLFF calculations while VASP jobs are in queue/running ---")
        batch_mlff_results = {}
        for name, snap_idx, atoms_obj in batch:
            e, f = calculate_mlff(atoms_obj, args.mlff_model)
            batch_mlff_results[name] = (e, f)
            
        # 3c. Wait for VASP Jobs
        if batch_job_ids:
            wait_for_jobs(batch_job_ids)
        
        # 3d. Parse VASP Results & Aggregate
        for name, snap_idx, atoms_obj in batch:
            # Get MLFF results
            mlff_energy, mlff_forces = batch_mlff_results.get(name, (None, None))
            
            # Get VASP results
            tmp_path = batch_dirs.get(name)
            if tmp_path:
                vasp_raw, vasp_corrected, vasp_forces = parse_vasp_result(name, tmp_path)
            else:
                vasp_raw, vasp_corrected, vasp_forces = None, None, None
            
            # Store if we have at least some data (or mark as failed)
            all_results.append({
                "name": name,
                "snapshot_num": snap_idx,
                "atoms": atoms_obj,
                "mlff_energy": mlff_energy,
                "mlff_forces": mlff_forces,
                "vasp_raw": vasp_raw,
                "vasp_corrected": vasp_corrected,
                "vasp_forces": vasp_forces
            })

    # --- 4. Save Final Results ---
    print("\n--- Saving All Results ---")
    
    for res in all_results:
        name = res['name']
        snap_num = res['snapshot_num']
        atoms_obj = res['atoms']
        
        # Save Energy
        if res['mlff_energy'] is not None or res['vasp_raw'] is not None:
            new_energy_row = pd.DataFrame([{
                "structure_name": name,
                "snapshot_num": snap_num,
                f"{args.mlff_model}_energy": res['mlff_energy'],
                "VASP_raw_energy": res['vasp_raw'],
                "VASP_corrected_energy": res['vasp_corrected']
            }])
            new_energy_row.to_csv(energies_csv_path, mode='a', header=False, index=False)
        
        # Save Forces
        num_atoms = len(atoms_obj)
        mlff_forces = res['mlff_forces']
        vasp_forces = res['vasp_forces']

        mlff_fx = mlff_forces[:, 0] if mlff_forces is not None else [None] * num_atoms
        mlff_fy = mlff_forces[:, 1] if mlff_forces is not None else [None] * num_atoms
        mlff_fz = mlff_forces[:, 2] if mlff_forces is not None else [None] * num_atoms
        
        vasp_fx = vasp_forces[:, 0] if vasp_forces is not None else [None] * num_atoms
        vasp_fy = vasp_forces[:, 1] if vasp_forces is not None else [None] * num_atoms
        vasp_fz = vasp_forces[:, 2] if vasp_forces is not None else [None] * num_atoms

        forces_df = pd.DataFrame({
            "atom_index": range(num_atoms),
            "symbol": atoms_obj.get_chemical_symbols(),
            "MLFF_Fx": mlff_fx,
            "MLFF_Fy": mlff_fy,
            "MLFF_Fz": mlff_fz,
            "VASP_Fx": vasp_fx,
            "VASP_Fy": vasp_fy,
            "VASP_Fz": vasp_fz,
        })
        forces_df.index.name = "atom_index"
        forces_csv_path = output_dir / f"{name}_forces.csv"
        forces_df.to_csv(forces_csv_path, index=False)
        print(f"Saved results for {name}")

    print("\n--- All structures processed and saved successfully! ---")

if __name__ == "__main__":
    main()