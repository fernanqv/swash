wget https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt
sudo apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt -y update
sudo apt install -y intel-oneapi-mkl
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.0/lib/
wget https://swash.sourceforge.io/download/zip/SWASH-10.05-Linux.tar.gz
wget https://swanmodel.sourceforge.io/download/zip/SWAN-41.51-Linux.tar.gz
tar xzvf SWASH-10.05-Linux.tar.gz 
tar xzvf SWAN-41.51-Linux.tar.gz
wget https://swash.sourceforge.io/download/zip/testcases.tar.gz
tar xzvf testcases.tar.gz 
wget https://swanmodel.sourceforge.io/download/zip/refrac.tar.gz
tar xzvf refrac.tar.gz

ROOT=`pwd`
export PATH=$ROOT/SWASH-10.05-Linux/bin:$ROOT/SWAN-41.51-Linux/bin:$PATH
cd testcases/a11stwav
swashrun -input a11stw01
cd /refrac
swanrun -input a11refr

echo 'export PATH=/SWASH-10.05-Linux/bin:SWAN-41.51-Linux/bin:$PATH' >> $HOME/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.0/lib/:$LD_LIBRARY_PATH' >> $HOME/.bashrc

#     2  conda install ipykernel
#     3  conda config --add channels defaults
#     4  conda install ipykernel
#     5  conda config --set solver classic
#     6  conda install ipykernel
#     7  conda install -c conda-forge libarchive
#     8  cd 
#     9  find ./ -name libarchive.so.20
#    10  cd /home/vscode/
#    11  find ./ -name libarchive.so.20
#    12  find ./ -name libarchive.*
#    13  conda update -n base -c defaults conda
#    14  conda install ipykernel
