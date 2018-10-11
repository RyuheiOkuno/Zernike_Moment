
#$ -S /usr/bin/zsh
#$ -cwd
#$ -N calc_NUMBER_ZD
#$ -pe mpi* 16
#$ -e w_a_err.logNUMBER
#$ -o w_a_std.logNUMBER

source /home/ryuhei/.bashrc

python /home/ryuhei/zernike_moment/data/program/run/generate_descriptor.py NUMBER 6 True
