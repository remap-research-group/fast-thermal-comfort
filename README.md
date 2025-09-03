# fast-thermal-comfort

## OnDemand
Once connected to the VPN, go to the PACE Open OnDemand ICE website and log in using your GT credentials.

## Prepare an Environment (in Terminal)
Once the Jupyter job has started, run the following commands in a Terminal to prepare a virtual environment.

Open a Terminal Session. To open Terminal in Jupyter, click on New (dropdown menu in the upper right corner) -> Terminal.
(For CPU Environment; 1.7 GB):
python3 -m venv PACE-CPU-venv
source PACE-ML-CPU-venv/bin/activate
pip install --upgrade pip
pip install ipykernel matplotlib pandas seaborn scikit-learn tensorflow-cpu gdal geopandas fiona shapely rasterio pyproj
python -m ipykernel install --user --name PACE-CPU-venv

(For GPU Environment; 5.2 GB):
python3 -m venv PACE-GPU-venv
source PACE-ML-GPU-venv/bin/activate
pip install --upgrade pip
pip install matplotlib pandas seaborn scikit-learn ipykernel tensorflow[and-cuda] gdal geopandas fiona shapely rasterio pyproj
python -m ipykernel install --user --name PACE-GPU-venv

Clone this repo to your home directory and change directory:
git clone https://github.com/remap-research-group/fast-thermal-comfort.git

From the main notebook dashboard in Jupyter, launch the notebook:
(For CPU Environment):
Select: Kernel -> Change Kernel -> PACE-CPU-venv

(For GPU Environment):
Select: Kernel -> Change Kernel -> PACE-GPU-venv
Limited Space in Home (in Terminal)
If there not enough space in home folder, you could alternatively create the environment in the scratch folder. Before executing the commands listed above, go to your scratch storage:

cd ~/scratch Then, create the virtual environment as described above.


Source from https://github.com/jeffvaldez/PACE-Apps-of-ML?tab=readme-ov-file#ondemand-website