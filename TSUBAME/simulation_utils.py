# simulation_utils.py
#update
# --- Monkey-patch for importlib.metadata in Python 3.9 ---
import sys
if sys.version_info < (3, 10):
    import importlib_metadata
    import importlib
    importlib.metadata = importlib_metadata
# --- End of patch ---

import numpy as np
import pandas as pd
import torch
import gc
import os
from joblib import Parallel, delayed
from ase.io import read, write
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase import units, Atoms

def get_calculator(model_name, use_device='cuda'):
    print(f"DEBUG: get_calculator called for {model_name} on {use_device}")
    # Check if CUDA is actually available if requested
    if use_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        use_device = 'cpu'

    # メモリ節約のため、CHGNetのインスタンス作成時にメモリ管理を厳格化
    if model_name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        try:
            return CHGNetCalculator(use_device=use_device)
        except IndexError:
            # Fallback for the "list index out of range" error in determine_device
            if use_device == 'cuda':
                print("Warning: CHGNet auto-device detection failed (IndexError). Attempting 'cuda:0'.")
                try:
                    return CHGNetCalculator(use_device='cuda:0')
                except Exception:
                    pass
            
            print("Warning: CHGNet device initialization failed. Falling back to CPU.")
            return CHGNetCalculator(use_device='cpu')
    elif model_name == "MatterSim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=use_device)
    elif model_name == "Orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_inf_omat(device=use_device, precision="float32-high")
        return ORBCalculator(orbff, device=use_device)
    elif model_name == "MatRIS":
        from matris.applications.base import MatRISCalculator
        return MatRISCalculator(model='matris_10m_oam', task='efsm', device=use_device)
    elif model_name == "NequipOLM":
        from nequip.ase import NequIPCalculator
        model_path = "/workspace/NequipOLM_model/nequip-oam-l.nequip.pt2"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NequipOLM model not found at '{model_path}'.")
        return NequIPCalculator.from_compiled_model(model_path, device=use_device)
    elif model_name == "CHGNet_r2SCAN":
        import matgl
        # DGL backend is required for CHGNet models in matgl
        try:
            matgl.set_backend("DGL")
        except Exception:
            pass # Backend might already be set or not switchable
            
        from matgl.ext.ase import PESCalculator
        # Search for the model in potential locations
        possible_paths = [
            os.environ.get("CHGNET_R2SCAN_PATH"),
            os.path.join(os.getcwd(), "CHGNet_r2SCAN"),
            os.path.join(os.getcwd(), "models", "CHGNet_r2SCAN"),
            "/opt/models/CHGNet_r2SCAN"
        ]
        
        model_path = None
        for p in possible_paths:
            if p and os.path.exists(p):
                model_path = p
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "CHGNet_r2SCAN model not found. Checked locations:\n" + 
                "\n".join(filter(None, possible_paths)) + 
                "\nPlease place the 'CHGNet_r2SCAN' model folder in the current directory or 'models/' subdirectory, "
                "or set the CHGNET_R2SCAN_PATH environment variable."
            )
        
        print(f"Loading CHGNet_r2SCAN model from: {model_path}")
        potential = matgl.load_model(model_path)
        # Move model to device
        potential.to(use_device)
        
        # Wrapper to ensure inputs are moved to the same device as the model
        class GPUPotentialWrapper:
            def __init__(self, potential):
                self.potential = potential
            def __call__(self, g, lat, state_attr, **kwargs):
                # Check device of the underlying model
                device = next(self.potential.model.parameters()).device
                
                # Helper to convert and move
                def to_device(obj):
                    if obj is None: return None
                    if hasattr(obj, 'to'): return obj.to(device)
                    if isinstance(obj, (np.ndarray, list)):
                        return torch.tensor(obj, device=device, dtype=matgl.float_th)
                    return obj

                # Move inputs to device
                g = g.to(device)
                lat = to_device(lat)
                state_attr = to_device(state_attr)
                
                # Handle kwargs that might need moving
                if 'total_charge' in kwargs: kwargs['total_charge'] = to_device(kwargs['total_charge'])
                if 'ext_pot' in kwargs: kwargs['ext_pot'] = to_device(kwargs['ext_pot'])

                return self.potential(g=g, lat=lat, state_attr=state_attr, **kwargs)
            
            def __getattr__(self, name):
                return getattr(self.potential, name)

        return PESCalculator(potential=GPUPotentialWrapper(potential))
    else: raise ValueError(f"Unknown model specified: {model_name}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def optimize_structure(atoms_obj, model_name, fmax=0.01):
    energies, lattice_constants = [], []
    atoms_obj.calc = get_calculator(model_name)
    
    # Revert to ExpCellFilter for memory efficiency and stability
    atoms_filter = ExpCellFilter(atoms_obj)
        
    opt = FIRE(atoms_filter)
    
    def save_step_data(a=atoms_filter):
        energies.append(a.atoms.get_potential_energy())
        lattice_constants.append(np.mean(a.atoms.get_cell().lengths()))
    
    opt.attach(save_step_data)
    
    # Run optimization with a step limit to prevent freezing
    max_steps = 100
    max_attempts = 20 # Total 2000 steps
    converged = False
    
    for i in range(max_attempts):
        converged = opt.run(fmax=fmax, steps=max_steps)
        
        # 🟢 Force memory cleanup between steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if converged:
            print(f"Optimization converged in {(i+1)*max_steps} steps or fewer.")
            break
        print(f"Optimization attempt {i+1}/{max_attempts} finished (not converged). Continuing...")
        
    if not converged:
        print("Warning: Optimization did not fully converge within the step limit. Returning current structure.")

    return atoms_obj, energies, lattice_constants

def _run_single_temp_npt(params):
    # パラメータの受け取り (pfactorを含む)
    (model_name, sim_mode, temp, initial_structure_dict, magmom_specie, user_time_step,
     eq_steps, pressure, ttime, pfactor, use_device) = params
    atoms, calc, dyn = None, None, None
    try:
        # --- 1. 原子の復元 ---
        atoms = Atoms(**initial_structure_dict)
        
        # NPT(Nosé-Hoover)を使うため、maskを作成して直交性を維持します
        if sim_mode == "Legacy (Orthorhombic)":
            cell = atoms.get_cell(); a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
            atoms.set_cell(np.diag([a, b, c]), scale_atoms=True)
            if not NPT._isuppertriangular(atoms.get_cell()): atoms.set_cell(atoms.cell.cellpar(), scale_atoms=True)
            # 角度(alpha, beta, gamma)を固定し、各軸の伸縮のみを許すマスク
            npt_mask = [1, 1, 1, 0, 0, 0] 
        else:
            cell = atoms.get_cell(); q, r = np.linalg.qr(cell)
            for i in range(3):
                if r[i, i] < 0: r[i, :] *= -1
            atoms.set_cell(r, scale_atoms=True)
            npt_mask = None # フル緩和
        
        atoms.calc = get_calculator(model_name, use_device=use_device)
        
        # --- 2. データの準備 ---
        results_data = {
            "energies": [], "instant_temps": [], "volumes": [], "a_lengths": [], "b_lengths": [], "c_lengths": [],
            "alpha": [], "beta": [], "gamma": [], "positions": [], "cells": []
        }

        magmom_indices = []
        magmom_column_keys = []
        if magmom_specie:
            from chgnet.model.dynamics import CHGNetCalculator
            if isinstance(atoms.calc, CHGNetCalculator):
                symbols = atoms.get_chemical_symbols()
                count = 1
                for i, s in enumerate(symbols):
                    if s == magmom_specie:
                        magmom_indices.append(i)
                        key = f"{magmom_specie}_{count}"
                        magmom_column_keys.append(key)
                        results_data[key] = [] 
                        count += 1

        def log_step_data():
            a, b, c, alpha, beta, gamma = atoms.get_cell().cellpar()
            results_data["energies"].append(atoms.get_potential_energy())
            results_data["instant_temps"].append(atoms.get_temperature())
            results_data["volumes"].append(atoms.get_volume())
            results_data["a_lengths"].append(a); results_data["b_lengths"].append(b); results_data["c_lengths"].append(c)
            results_data["alpha"].append(alpha); results_data["beta"].append(beta); results_data["gamma"].append(gamma)
            results_data["positions"].append(atoms.get_positions())
            results_data["cells"].append(atoms.get_cell())
            
            if magmom_indices:
                all_magmoms = atoms.get_magnetic_moments()
                for i, atom_idx in enumerate(magmom_indices):
                    key = magmom_column_keys[i]
                    results_data[key].append(all_magmoms[atom_idx])

        # 🟢 Memory cleaner callback
        def clean_memory_step():
            # Clear CUDA cache periodically to prevent fragmentation/accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # gc.collect() is also useful but can be slow, so maybe less frequent?
            # But empty_cache is the critical one for GPU "stepwise" growth.

        # --- 3. MD初期化 ---
        init_temp = max(temp, 5.0)
        MaxwellBoltzmannDistribution(atoms, temperature_K=init_temp, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)
        
        # ==========================================
        # Phase 0: NVT (Langevin) 緩和
        # ==========================================
        # まず体積固定で温度をなじませる
        dyn_nvt = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=temp, friction=0.02)
        dyn_nvt.run(100) 

        # ==========================================
        # Phase 1 & 2: NPT (Nosé-Hoover)
        # ==========================================
        
        # ✅ 【修正】 0Kでの除算エラー（爆発）を防ぐため、最低温度を1Kに設定する
        target_temp = max(temp, 1.0)

        dyn = NPT(
            atoms, 
            timestep=user_time_step * units.fs, 
            temperature_K=target_temp,   # temp -> target_temp に変更
            externalstress=pressure * units.bar,
            ttime=ttime,     # 呼び出し元から受け取った値 (25fs)
            pfactor=pfactor, # 呼び出し元から受け取った値 (軽い設定)
            mask=npt_mask    
        )
        
        # Attach memory cleaner (run every 50 steps)
        dyn.attach(clean_memory_step, interval=50)
        
        # 初期緩和 (ログなし)
        dyn.run(200)

        # 本番 (ログあり)
        dyn.attach(log_step_data, interval=10)
        dyn.run(eq_steps)
        
        final_structure_dict = {'numbers': atoms.get_atomic_numbers(), 'positions': atoms.get_positions(), 'cell': atoms.get_cell(), 'pbc': atoms.get_pbc()}
        results_data["set_temps"] = [temp] * len(results_data["energies"])
        return temp, final_structure_dict, results_data

    except Exception as e:
        import traceback; print(f"Error at {temp} K:"); traceback.print_exc()
        if "OutOfMemoryError" in str(e):
            print("CRITICAL: GPU Out Of Memory. Try reducing --n-gpu-jobs.")
        return None
    finally:
        del atoms, calc, dyn; clear_memory()

def run_npt_simulation_parallel(initial_atoms, model_name, sim_mode, magmom_specie, temp_range, time_step, eq_steps,
    pressure, n_gpu_jobs, use_device='cuda', progress_callback=None, traj_filepath=None, append_traj=False):
    
    # ✅ パラメータを「成功していた古い設定」に戻す
    
    # 温度制御: 25 fs (強力な制御)
    ttime = 25 * units.fs 
    
    # 圧力制御: 固定値 (非常に軽い設定、体積依存性なし)
    # これが相転移の動きやすさを担保していた
    pfactor = 2e6 * units.GPa * (units.fs**2)
    
    temperatures = np.arange(temp_range[0], temp_range[1] + temp_range[2], temp_range[2])
    all_results = []
    last_structure_dict = {'numbers': initial_atoms.get_atomic_numbers(), 'positions': initial_atoms.get_positions(), 'cell': initial_atoms.get_cell(), 'pbc': initial_atoms.get_pbc()}
    num_batches = int(np.ceil(len(temperatures) / n_gpu_jobs))
    
    for i in range(num_batches):
        if progress_callback: progress_callback(i, num_batches, f"Batch {i+1}/{num_batches} running...", None)
        batch_start_index, batch_end_index = i * n_gpu_jobs, min((i + 1) * n_gpu_jobs, len(temperatures))
        temp_batch = temperatures[batch_start_index:batch_end_index]
        if not len(temp_batch) > 0: continue
        
        # pfactor もタスク引数として渡す
        tasks = [(model_name, sim_mode, t, last_structure_dict, magmom_specie, time_step, eq_steps, pressure, ttime, pfactor, use_device) for t in temp_batch]
        
        if n_gpu_jobs == 1:
            # Single GPU/process: Run sequentially without joblib overhead to avoid CUDA context issues
            print(f"Running batch {i+1}/{num_batches} sequentially (n_gpu_jobs=1)...")
            batch_results = []
            for task in tasks:
                res = _run_single_temp_npt(task)
                batch_results.append(res)
        else:
            # Parallel execution
            batch_results = Parallel(n_jobs=n_gpu_jobs, mmap_mode='r+')(delayed(_run_single_temp_npt)(task) for task in tasks)
        
        valid_results = [res for res in batch_results if res is not None]
        if not valid_results: break
        all_results.extend(valid_results)

        if temp_range[2] > 0:
            next_initial_result = max(valid_results, key=lambda x: x[0])
        else:
            next_initial_result = min(valid_results, key=lambda x: x[0])
        last_structure_dict = next_initial_result[1]
    
        if progress_callback:
            temp_df_list = [pd.DataFrame({k: v for k, v in res[2].items() if k not in ["positions", "cells"]}) for res in all_results]
            partial_df = pd.concat(temp_df_list, ignore_index=True)
            progress_callback(i + 1, num_batches, f"Batch {i+1}/{num_batches} finished.", partial_df)

    if progress_callback: progress_callback(num_batches, num_batches, "NPT simulation finished.", None)
    if not all_results: return pd.DataFrame(), last_structure_dict
    
    if traj_filepath:
        atomic_numbers = initial_atoms.get_atomic_numbers()
        new_frames = [Atoms(numbers=atomic_numbers, positions=p, cell=c, pbc=True) for _, _, res in all_results for p, c in zip(res.get("positions", []), res.get("cells", []))]
        if append_traj and os.path.exists(traj_filepath):
            existing_frames = read(traj_filepath, index=':')
        else:
            existing_frames = []
        all_frames = existing_frames + new_frames
        if all_frames: write(traj_filepath, all_frames, format='extxyz')

    df_list = []
    for temp, final_struct, result_dict in all_results:
        clean_dict = {k: v for k, v in result_dict.items() if k not in ["positions", "cells"]}
        df_list.append(pd.DataFrame(clean_dict))
    
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df, last_structure_dict
