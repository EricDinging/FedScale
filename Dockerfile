# Use an official CUDA image as a parent image
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages

RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN pip3 install --upgrade pip
# Create a virtual environment and activate it
RUN python -m pip install virtualenv
RUN python -m virtualenv venv
RUN /bin/bash -c "source venv/bin/activate"

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project files into the container (assuming your project is in the current directory)
COPY . .

# Install your project using pip
RUN pip install -e .

# Command to run when the container starts
CMD ["bash"]