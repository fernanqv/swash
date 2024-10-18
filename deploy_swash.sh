wget https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt
sudo apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt -y update
sudo apt install -y intel-oneapi-mkl
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2024.2/lib/
wget https://swash.sourceforge.io/download/zip/SWASH-10.05-Linux.tar.gz
tar xzvf SWASH-10.05-Linux.tar.gz 
wget https://swash.sourceforge.io/download/zip/testcases.tar.gz
tar xzvf testcases.tar.gz 
ROOT=`pwd`
export PATH=$ROOT/SWASH-10.05-Linux/bin:$PATH
cd testcases/a11stwav
swashrun -input a11stw01