FROM mcr.microsoft.com/devcontainers/base:jammy

# Establecer el directorio de trabajo dentro del contenedor
#WORKDIR /app

# Copiar el script de shell al contenedor
COPY deploy_swash.sh /tmp

# Asegurarse de que el script tenga permisos de ejecución
RUN bash -x /tmp/deploy_swash.sh



