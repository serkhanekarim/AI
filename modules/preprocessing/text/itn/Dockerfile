# Set base image (host OS)
FROM conda/miniconda3

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN conda update -n base -c defaults conda -y
RUN conda install conda-build -y
RUN apt-get update &&     apt-get install -y --reinstall build-essential &&     apt-get upgrade -y &&     apt-get install -y git &&     apt-get install make

# Copy the grammar files and the environment and start the environment
COPY . /app/
RUN conda env create --file environment.yml
#CMD conda init bash && source /root/.bashrc && conda activate itn

# Run live ITN
#CMD ./run_itn.sh fr-FR
