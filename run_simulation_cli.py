# /cli_runner/run_simulation_cli.py
import argparse
import os
import sys
import time
import json
import pandas as pd
from ase.io import read, write
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The required modules (simulation_utils, visualization) are now in the same directory.
import simulation_utils as sim
import visualization as viz

def main():
    parser = argparse.ArgumentParser(description="Run a materials science simulation from the command line.")
    
    # --- Required Arguments ---
    parser.add_argument("--cif-path", type=str, required=True, help="Path to the input CIF file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save all simulation results.")

    # --- Job Type ---
    parser.add_argument("--job-type", type=str, default="full_simulation", choices=["full_simulation", "optimize_only"], help="Type of job to run.")

    # --- Model and Simulation Parameters ---
    parser.add_argument("--model", type=str, default="CHGNet", choices=["CHGNet", "MatterSim", "Orb", "NequipOLM"], help="ML Force Field model to use.")
    parser.add_argument("--sim-mode", type=str, default="Realistic (ISIF=3)", choices=["Realistic (ISIF=3)", "Legacy (Orthorombic)"], help="Simulation mode.")
    parser.add_argument("--skip-optimization", action='store_true', help="Skip the initial structure optimization.")
    
    # --- NPT Parameters (for full_simulation) ---
    parser.add_argument("--magmom-specie", type=str, default="Co", help="Species for magnetic moment tracking (e.g., 'Co').")
    parser.add_argument("--temp-start", type=int, default=1, help="NPT simulation start temperature (K).")
    parser.add_argument("--temp-end", type=int, default=1000, help="NPT simulation end temperature (K).")
    parser.add_argument("--temp-step", type=int, default=5, help="NPT simulation temperature step (K).")
    parser.add_argument("--eq-steps", type=int, default=2000, help="Number of equilibration steps per temperature.")
    parser.add_argument("--n-gpu-jobs", type=int, default=3, help="Number of parallel jobs for NPT simulation.")
    parser.add_argument("--enable-cooling", action='store_true', help="Enable cooling phase after heating.")

    args = parser.parse_args()

    # --- Setup ---
    print(f"--- Starting Job: {os.path.basename(args.cif_path)} ---")
    start_time = time.time()
    
    if not os.path.exists(args.cif_path):
        print(f"Error: Input CIF file not found at {args.cif_path}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    try:
        # --- Run Optimization ---
        print(f"Reading structure from {args.cif_path}...")
        atoms = read(args.cif_path)
        
        if not args.skip_optimization:
            print(f"Optimizing structure with {args.model}...")
            opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=args.model, fmax=0.001)
            
            opt_cif_path = os.path.join(args.output_dir, "optimized_structure.cif")
            write(opt_cif_path, opt_atoms, format="cif")
            print(f"Saved optimized structure to {opt_cif_path}")
        else:
            print("--- Skipping structure optimization as requested. ---")
            opt_atoms = atoms.copy() # Use the original structure

        if args.job_type == "optimize_only":
            if args.skip_optimization:
                print("Warning: --job-type 'optimize_only' has no effect when --skip-optimization is used.")
            print("--- Optimization job finished. ---")
            return

        # --- Run Full NPT Simulation ---
        print("\n--- Starting NPT Simulation ---")
        temp_range = (args.temp_start, args.temp_end, args.temp_step)
        traj_filepath = os.path.join(args.output_dir, "trajectory.xyz")

        # --- Heating Phase ---
        print(f"🔥 Starting Heating Phase ({args.temp_start}K -> {args.temp_end}K)...")
        npt_df_heating, heating_final_struct = sim.run_npt_simulation_parallel(
            initial_atoms=opt_atoms, model_name=args.model, sim_mode=args.sim_mode,
            magmom_specie=args.magmom_specie, temp_range=temp_range,
            time_step=1.0, eq_steps=args.eq_steps, pressure=1.0,
            n_gpu_jobs=args.n_gpu_jobs, progress_callback=None, # No realtime callback for CLI
            traj_filepath=traj_filepath, append_traj=False
        )
        if npt_df_heating.empty:
            raise ValueError("Heating phase failed or produced no data.")
        
        npt_df = npt_df_heating
        print("✅ Heating Phase Completed.")

        # --- Cooling Phase ---
        if args.enable_cooling:
            print(f"❄️ Starting Cooling Phase ({args.temp_end}K -> {args.temp_start}K)...")
            cooling_temp_range = (args.temp_end, args.temp_start, -args.temp_step)
            cooling_initial_atoms = Atoms(**heating_final_struct)
            npt_df_cooling, _ = sim.run_npt_simulation_parallel(
                initial_atoms=cooling_initial_atoms, model_name=args.model, sim_mode=args.sim_mode,
                magmom_specie=args.magmom_specie, temp_range=cooling_temp_range,
                time_step=1.0, eq_steps=args.eq_steps, pressure=1.0,
                n_gpu_jobs=args.n_gpu_jobs, progress_callback=None,
                traj_filepath=traj_filepath, append_traj=True
            )
            if not npt_df_cooling.empty:
                npt_df = pd.concat([npt_df_heating, npt_df_cooling], ignore_index=True)
                print("✅ Cooling Phase Completed.")
            else:
                print("⚠️ Cooling phase failed. Proceeding with heating data only.")

        # --- Save Results ---
        print("\n--- Saving Results ---")
        if not npt_df.empty:
            # Plot
            fig_temp = viz.plot_temperature_dependent_properties(npt_df, 100)
            plot_path = os.path.join(args.output_dir, "npt_vs_temp.png")
            fig_temp.savefig(plot_path)
            plt.close(fig_temp)
            print(f"Saved plot to {plot_path}")

            # Full CSV
            npt_df.to_csv(os.path.join(args.output_dir, "npt_summary_full.csv"), index=False)
            
            # Last Steps CSV
            npt_df.groupby('set_temps').last().reset_index().to_csv(
                os.path.join(args.output_dir, "npt_last_steps.csv"), index=False)

            # Stats CSV
            agg_cols = [col for col in npt_df.columns if col != 'set_temps']
            stats_df = npt_df.groupby('set_temps')[agg_cols].agg(['mean', 'std'])
            stats_df.columns = ['_'.join(map(str, col)).strip() for col in stats_df.columns.values]
            rename_mapping = {f'{col}_mean': col for col in agg_cols}
            stats_df = stats_df.rename(columns=rename_mapping)
            stats_df = stats_df.reset_index()
            stats_df.to_csv(os.path.join(args.output_dir, "npt_summary_stats.csv"), index=False, float_format='%.6f')
            
            # Magmom CSV
            if args.magmom_specie:
                magmom_cols = [col for col in npt_df.columns if col.startswith(f"{args.magmom_specie}_")]
                if magmom_cols:
                    magmom_df = npt_df[magmom_cols].copy()
                    magmom_df.insert(0, 'step', range(len(magmom_df)))
                    magmom_df.to_csv(os.path.join(args.output_dir, "magmoms_per_atom.csv"), index=False)
            
            print("Saved all CSV results.")
        else:
            print("⚠️ NPT simulation did not produce data. Skipping result saving.")

    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred during the simulation: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        # --- Final Memory Cleanup ---
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            print("Clearing CUDA cache...")
            sys.modules['torch'].cuda.empty_cache()
            print("CUDA cache cleared.")

    elapsed_time = time.time() - start_time
    print(f"\n--- Job Finished ---")
    print(f"Total execution time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
