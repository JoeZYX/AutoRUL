>This is a term paper for the fourth semester. 



# Architecture

<!-- ![figure: 'src/big_data_platform.png'](src\architecture.png)

The figure above shows the architecture of the Pipeline -->


# Implementation

There is a class named RemainingUsefulLife in `rul.py` which takes some parameters (see `rul.py`for more information). You can create an instance of this class by calling `rul.RemainingUsefulLife()`. 
After an instance is created, you should be able to call the methods; for example `compute_piecewise_linear_rul` or `plot_rul` on that instance. 

Every methode of the class is independent. For example: if you want to just get the piecewise RUL values calculated and get the results as dataframes, you can call the method `compute_piecewise_linear_rul` on the object. If you want to do everything at once (including rul calculation, feature extension, normalization, training and evaluation) and just get the final results, you can just call `auto_rul`method of the created object. 




- `rul.py`: contains the automated pipeline 
- `utilities.py`: contains some helper functions for data preparation 
- `kaggel_plant_with_pipeline.ipynb`: In this notebook, some models are trained and evaluated on kaggle production plant dataset using the automated pipeline. 
- `kaggel_original.ipynb`: This notebook uses the automated pipline to train and evaluate models on the original production plant dataset directly retrieved from kaggle web page. 
- `waterpump.ipynb`: In this notebook the automated pipeline is used to train and evaluate some models on the provided waterpump dataset. 
- `CMAPSS_RUL_estimation_with_pipeline.ipynb`: In this notebook, the automated pipeline is used to train and evaluate some models on the CMAPSS dataset. 


# Guide to run the application

run `conda env create -f environment.yml`

or `pip install -r requirements.txt`

