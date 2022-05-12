CONDA_ENV_NAME="carpricer"
CONDA_ENV_PYTHON="3.8.5"

echo "Configuring PYTHONPATH for the project"
cat >> $(python -m site --user-site)/carpricer.pth <<EOF
$PWD/src
EOF

echo "Creating a conda environment"
conda create -y -n $CONDA_ENV_NAME Python=$CONDA_ENV_PYTHON
conda activate $CONDA_ENV_NAME
pip install jupyter ipykernel
pip install -r .aml/jobs/docker-context/requirements.txt
python -m ipykernel install --user --name $CONDA_ENV_NAME --display-name "Python ($CONDA_ENV_NAME)"