"module load anaconda-python/3.8 maxwell gcc/9.3
. conda-init
alias mine='salloc --partition=cms-desy --time=24:00:00 --nodes=1 --constraint="P100"'
alias renewal='klist -s ||kinit'" >> .bashrc
conda env create -f ray_lightning.yml