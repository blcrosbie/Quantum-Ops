# Use a slim Python image for efficiency (Senior Best Practice)
FROM python:3.11-slim

# Set environmental variables to optimize Python for Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /home/jovyan/work

# Install system dependencies (needed for some quantum viz libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Quantum SDKs and project dependencies
# We install these directly to keep the image portable
RUN pip install --no-cache-dir \
    qiskit \
    qiskit-ionq \
    python-dotenv \
    jupyterlab \
    matplotlib \
    networkx \
    pandas

# Expose the Jupyter port
EXPOSE 8888

# Start JupyterLab on launch
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]