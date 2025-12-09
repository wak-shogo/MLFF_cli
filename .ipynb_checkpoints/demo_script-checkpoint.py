# cli_runner/demo_script.py
# This script contains the code for the demo notebook.
# Please copy and paste the content of each cell block into a new cell in your Jupyter/Colab notebook.

# ==============================================================================
# CELL 1: Environment Setup
# ==============================================================================
print("--- Cell 1: Environment Setup ---")
print("Please execute the following commands in a notebook cell:")
print("""
!echo 'Installing dependencies...'
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas matplotlib ase py3Dmol nglview ipywidgets pymatgen
!pip install chgnet mattersim orb-models nequip-allegro
""")


# ==============================================================================
# CELL 2: Import Modules
# ==============================================================================
print("\n--- Cell 2: Import Modules ---")
import os
import sys
import time
import shutil
import glob
from datetime import datetime
import pandas as pd
from ase.io import read, write
from ase import Atoms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
# In a real notebook, you would import from google.colab
# from google.colab import files
# For now, we will simulate this for local execution if needed.
try:
    from google.colab import files
except ImportError:
    print("Not in Colab environment. File upload will be simulated.")
    files = None

import ipywidgets as widgets
from IPython.display import display, clear_output

# Import local simulation scripts
import simulation_utils as sim
import visualization as viz

print("Modules imported successfully.")


# ==============================================================================
# CELL 3: File Upload
# ==============================================================================
print("\n--- Cell 3: File Upload ---")
UPLOAD_DIR = 'uploaded_cifs'
if os.path.exists(UPLOAD_DIR):
    shutil.rmtree(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR)

print(f"Please run the following code in a notebook cell to upload your CIF files.")
print("""
from google.colab import files
import os
import shutil

UPLOAD_DIR = 'uploaded_cifs'
if os.path.exists(UPLOAD_DIR):
    shutil.rmtree(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR)

print(f"Please upload your CIF files. They will be saved in the '{UPLOAD_DIR}' directory.")
uploaded = files.upload()

if not uploaded:
    print('No files uploaded. Please run the cell again to upload files.')
else:
    for filename, content in uploaded.items():
        with open(os.path.join(UPLOAD_DIR, filename), 'wb') as f:
            f.write(content)
    print(f"\nSuccessfully uploaded {len(uploaded)} file(s): {list(uploaded.keys())}")
""")


# ==============================================================================
# CELL 4: Configure Simulation Parameters
# ==============================================================================
print("\n--- Cell 4: Configure Simulation Parameters ---")
style = {'description_width': 'initial'}
model_widget = widgets.Dropdown(options=['CHGNet', 'MatterSim', 'Orb', 'NequipOLM'], value='CHGNet', description='ML Model:', style=style)
job_type_widget = widgets.Dropdown(options=['full_simulation', 'optimize_only'], value='full_simulation', description='Job Type:', style=style)
temp_start_widget = widgets.IntText(value=1, description='Start Temp (K):', style=style)
temp_end_widget = widgets.IntText(value=1000, description='End Temp (K):', style=style)
temp_step_widget = widgets.IntText(value=5, description='Temp Step (K):', style=style)
eq_steps_widget = widgets.IntText(value=2000, description='Steps per Temp:', style=style)
enable_cooling_widget = widgets.Checkbox(value=False, description='Enable Cooling Phase', style=style)

print("Parameter widgets created. Please display them in a notebook cell by running the following:")
print("""
display(model_widget, job_type_widget, temp_start_widget, temp_end_widget, temp_step_widget, eq_steps_widget, enable_cooling_widget)
""")


# ==============================================================================
# CELL 5: Run Simulation
# ==============================================================================
print("\n--- Cell 5: Run Simulation ---")
run_button = widgets.Button(description='Run All Simulations', button_style='success')
output_log = widgets.Output()

def run_all_jobs(b):
    with output_log:
        clear_output(wait=True)
        print('Starting all jobs...')
        
        uploaded_files_list = glob.glob(os.path.join(UPLOAD_DIR, '*.cif'))
        if not uploaded_files_list:
            print('Error: No CIF files found in the upload directory. Please run the upload cell again.')
            return

        for cif_path in uploaded_files_list:
            base_filename = os.path.splitext(os.path.basename(cif_path))[0]
            project_name = f"{datetime.now().strftime('%Y%m%d')}_{base_filename}_{model_widget.value}"
            output_dir = os.path.join('simulation_results', project_name)
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{'='*50}")
            print(f'Processing: {os.path.basename(cif_path)}')
            print(f'Output Directory: {output_dir}')
            print(f"{'{'}='*50}\n")
            
            try:
                atoms = read(cif_path)
                print(f'Optimizing structure with {model_widget.value}...')
                opt_atoms, _, _ = sim.optimize_structure(atoms, model_name=model_widget.value, fmax=0.001)
                write(os.path.join(output_dir, 'optimized_structure.cif'), opt_atoms, format='cif')
                print('Structure optimization complete.')

                if job_type_widget.value == 'full_simulation':
                    print('\n--- Starting NPT Simulation ---')
                    temp_range = (temp_start_widget.value, temp_end_widget.value, temp_step_widget.value)
                    traj_filepath = os.path.join(output_dir, 'trajectory.xyz')

                    print('🔥 Heating Phase...')
                    npt_df_heating, heating_final_struct = sim.run_npt_simulation_parallel(
                        initial_atoms=opt_atoms, model_name=model_widget.value, sim_mode='Realistic (ISIF=3)',
                        magmom_specie='Co', temp_range=temp_range, time_step=1.0, eq_steps=eq_steps_widget.value, 
                        pressure=1.0, n_gpu_jobs=1, progress_callback=None, # n_gpu_jobs=1 for stability in Colab
                        traj_filepath=traj_filepath, append_traj=False
                    )
                    npt_df = npt_df_heating

                    if enable_cooling_widget.value:
                        print('❄️ Cooling Phase...')
                        cooling_temp_range = (temp_end_widget.value, temp_start_widget.value, -temp_step_widget.value)
                        cooling_initial_atoms = Atoms(**heating_final_struct)
                        npt_df_cooling, _ = sim.run_npt_simulation_parallel(
                            initial_atoms=cooling_initial_atoms, model_name=model_widget.value, sim_mode='Realistic (ISIF=3)',
                            magmom_specie='Co', temp_range=cooling_temp_range, time_step=1.0, eq_steps=eq_steps_widget.value, 
                            pressure=1.0, n_gpu_jobs=1, progress_callback=None,
                            traj_filepath=traj_filepath, append_traj=True
                        )
                        if not npt_df_cooling.empty:
                            npt_df = pd.concat([npt_df_heating, npt_df_cooling], ignore_index=True)

                    if not npt_df.empty:
                        print('\n--- Saving Results ---')
                        fig_temp = viz.plot_temperature_dependent_properties(npt_df, 100)
                        fig_temp.savefig(os.path.join(output_dir, 'npt_vs_temp.png'))
                        plt.close(fig_temp)
                        npt_df.to_csv(os.path.join(output_dir, 'npt_summary_full.csv'), index=False)
                        print('Results saved.')

            except Exception as e:
                import traceback
                print(f'\n❌ An error occurred during processing {os.path.basename(cif_path)}: {e}')
                print(traceback.format_exc())
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f'--- Finished processing {os.path.basename(cif_path)} ---')
        
        print('\nAll jobs completed!')

run_button.on_click(run_all_jobs)
print("Button logic defined. Please display the button and output log in a notebook cell by running the following:")
print("""
display(run_button, output_log)
""")


# ==============================================================================
# CELL 6: View and Download Results
# ==============================================================================
print("\n--- Cell 6: View and Download Results ---")
print("Please run the following code in a notebook cell to see the results:")
print("""
from IPython.display import Image, HTML
import glob
import os
import shutil

result_dirs = glob.glob('simulation_results/*')

if not result_dirs:
    print('No results found. Please run the simulations first.')
else:
    for project_dir in sorted(result_dirs):
        project_name = os.path.basename(project_dir)
        print(f"\n{'='*30}")
        print(f'📁 Results for: {project_name}')
        print(f"{'{'}='*30}")

        # Display plot
        plot_path = os.path.join(project_dir, 'npt_vs_temp.png')
        if os.path.exists(plot_path):
            print('\n--- Temperature Dependent Properties ---')
            display(Image(filename=plot_path))
        
        # Create download links
        print('\n--- Download Files ---')
        result_files = glob.glob(os.path.join(project_dir, '*'))
        if not result_files:
            print('No files found in this project directory.')
        else:
            # Create a temporary zip file for each project
            zip_path = f'{project_dir}.zip'
            shutil.make_archive(project_dir, 'zip', project_dir)
            print('All results for this project have been bundled into a zip file.')
            # Provide a direct download link for the zip file
            display(HTML(f"<a href='./{zip_path}' download>Download {os.path.basename(zip_path)}</a>"))
""")
