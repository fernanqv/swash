FROM mcr.microsoft.com/devcontainers/miniconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Establecer el directorio de trabajo dentro del contenedor
#WORKDIR /app

# Copiar el script de shell al contenedor
COPY deploy_swash.sh /tmp

# Asegurarse de que el script tenga permisos de ejecución
RUN bash -x /tmp/deploy_swash.sh

# Copy the swash PATH to .bashrc
RUN echo 'export PATH=/SWASH-10.05-Linux/bin:SWAN-41.51-Linux/bin:$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.0/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc