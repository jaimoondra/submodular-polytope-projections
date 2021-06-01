universe=vanilla
getenv=true
executable=/opt/python3/latest/bin/python3
+Requirements=((Machine=="isye-wilkins1.isye.gatech.edu")||(Machine=="isye-gpu1001.isye.gatech.edu")||(Machine=="isye-hpc0009.isye.gatech.edu")||(Machine=="isye-leibniz15.isye.gatech.edu")||(Machine=="isye-leibniz14.isye.gatech.edu")||(Machine=="isye-leibniz12.isye.gatech.edu")||(Machine=="isye-leibniz5.isye.gatech.edu")||(Machine=="isye-leibniz4.isye.gatech.edu")||(Machine=="isye-hpc0010.isye.gatech.edu")||(Machine=="isye-hpc0011.isye.gatech.edu")||(Machine=="isye-hpc0012.isye.gatech.edu")||(Machine=="isye-bulldozer.isye.gatech.edu")||(Machine=="isye-goddard2.isye.gatech.edu")||(Machine=="isye-goddard1.isye.gatech.edu")||(Machine=="isye-jacobi8.isye.gatech.edu")||(Machine=="isye-jacobi7.isye.gatech.edu")||(Machine=="isye-jacobi6.isye.gatech.edu")||(Machine=="isye-jacobi5.isye.gatech.edu")||(Machine=="isye-jacobi4.isye.gatech.edu")||(Machine=="isye-jacobi3.isye.gatech.edu")||(Machine=="isye-jacobi2.isye.gatech.edu")||(Machine=="isye-jacobi1.isye.gatech.edu")||(Machine=="isye-krone.isye.gatech.edu")||(Machine=="isye-ruble.isye.gatech.edu")||(Machine=="isye-wilkins2.isye.gatech.edu")||(Machine=="isye-ames.isye.gatech.edu")||(Machine=="isye-fisher1.isye.gatech.edu")||(Machine=="isye-fisher2.isye.gatech.edu")||(Machine=="isye-fisher3.isye.gatech.edu")||(Machine=="isye-fisher4.isye.gatech.edu")||(Machine=="isye-wilkins3.isye.gatech.edu")||(Machine=="isye-wilkins4.isye.gatech.edu"))
arguments=$ENV(HOME)/tight_cuts/tight_cuts1.py
log=tight_cuts.log
error=tight_cuts.error
output=tight_cuts.out
request_memory=10000
notification = error
notification = complete
notify_user = jmoondra3@gatech.edu
queue