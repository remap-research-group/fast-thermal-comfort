# fast-thermal-comfort

## Prepare an Environment (in Terminal)
Once the Jupyter job has started, run the following commands in a Terminal to prepare a virtual environment.

Open a Terminal Session. To open Terminal in Jupyter, click on New (dropdown menu in the upper right corner) -> Terminal.

(For CPU Environment; 1.7 GB):
- module load conda
- conda env create -f environment.yml -n myenv
- conda activate <env_name>
- python -m ipykernel install --user --name <env_name>

(For GPU Environment; 5.2 GB):
- module load conda
- conda env create -f environment.yml -n myenv
- conda activate <env_name>
- conda install tensorflow[and-cuda]  
- python -m ipykernel install --user --name <env_name>

Clone this repo to your home directory and change directory:  
- git clone https://github.com/remap-research-group/fast-thermal-comfort.git

From the main notebook dashboard in Jupyter, launch the notebook:  
Select: Kernel -> Change Kernel -> <env_name>

Limited Space in Home (in Terminal):  
If there not enough space in home folder, you could alternatively create the environment in the scratch folder. Before executing the commands listed above, go to your scratch storage:

cd ~/scratch Then, create the virtual environment as described above.

Source from https://github.com/jeffvaldez/PACE-Apps-of-ML?tab=readme-ov-file#ondemand-website