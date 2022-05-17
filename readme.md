# Structuring an ML project with software engineering practices

The following repository gives a hint and an (opinionated) way to structure a machine learning project using software engineering practices. It tries to marry the benefits of using Jupyter Notebooks along with the good practices of developing code using a Python package. To read more details about the reasons for each choice please read the blog post: [Structure your Machine Learning project source code like a pro](https://santiagof.medium.com/structure-your-machine-learning-project-source-code-like-a-pro-44815cac8652).

## About this sample

This sample repository contains a regression model for predicting the prices of cars from UCI. For details about this problem and the dataset you can check [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile). A sample of this dataset is on the folder `data/car-prices/sample`. The regression model was constructed using an XGBoost with hyperparameters optimization using Scikit-Learn grid search.

## Project structure

The project was structured as follows. 

```
├── .aml                                # Azure Machine Learning resources (it's an opinionated choice).
│   ├── data                               # AML Datasets configuration
│   └── jobs                               # AML training job definitions.
├── .cloud                              # Infrastructure as code resources.
├── docs                                # Project documentation.
├── data                                # Data folder to hold metadata related with the datasets.
│   ├── car-prices                          # Specific to the dataset car-prices
│   │   └── sample                              # A sample of this dataset 
│   └── great_expectations                  # Expectations about how the data should look like.
├── notebooks                           # Notebooks for interactive data analysis or modeling
└── src                                 # Main source of the project
    ├── carpricer                           # Package carpricer with model's source code
    │   ├── dataprep                            # Data preparations code
    │   └── train                               # Training code
    └── tests                               # Unit tests for all the packages in the source code
```

> **One model, one repository:** This sample repository uses the convention *one model = one repo* meaning that each repository contains all the resources require to build an specifc model. On those situation where multiple models are involved in a single business problem, then the repository will hold all of them, so don't take the title so literal. This is why the `src` folder contains one package for each model used; it contains the source code for each of the models used. In this example, only one model is built, which is `carpricer`.

## Running the project

This repository was constructed with portability in mind, so you have a couple of options to run it:

### 1) Locally (linux only - changes to make it work on Windows are on you :D)

Follow this steps:
* Clone the repository in your local machine.
* Run `setup.sh` in a bash console. This will setup your `PYTHONPATH` by creating a `pth` file in the user's directory. For more details about why we used this choice read the blog post mentioned in the top. The script will also create a conda environment called `carpricer` and install all the dependencies.
* Open the notebook `notebooks/carpricer_model.ipynb` to train the model in an interactive way.
* Alternatively, you can run a training routine from a console (bash) using:

    `jobtools carpricer.train.trainer train_regressor --data-path data/car-prices/sample --params src/carpricer.params.yml`

### 2) Run it on Azure ML
* Clone the repository in your local machine.
* Install Azure ML CLI v2. If you don't have it, follow the installation instructions at [Install and set up the CLI (v2)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
* Create a compute named `trainer-cpu` or rename the compute specified in [.aml/jobs/carpricer.job.yml](.aml/jobs/carpricer.job.yml).
* Register the dataset with `az ml data create -f .aml/data/carprices.yml` 
* Create the training job with `az ml job create -f .aml/jobs/carpricer.job.yml`

## Contributing

This project welcomes contributions and suggestions. Open an issue and start the discussion!