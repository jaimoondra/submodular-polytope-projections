universe=vanilla
getenv=true
executable=/opt/python3/latest/bin/python3
+Requirements=((Machine=="isye-wilkins1.isye.gatech.edu")||(Machine=="isye-wilkins3.isye.gatech.edu")||(Machine=="isye-wilkins4.isye.gatech.edu")||Machine==("isye-abel.isye.gatech.edu")||Machine==("isye-ames.isye.gatech.edu")||Machine==("isye-dollar.isye.gatech.edu")||Machine==("isye-fisher2.isye.gatech.edu")||Machine==("isye-goddard1.isye.gatech.edu")||Machine==("isye-gpu0001.isye.gatech.edu")||Machine==("isye-gpu1001.isye.gatech.edu")||Machine==("isye-hpc0200.isye.gatech.edu")||Machine==("isye-hpc0201.isye.gatech.ed")||Machine==("isye-hpc0202.isye.gatech.edu")||Machine==("isye-hps0005.isye.gatech.edu")||Machine==("isye-ruble.isye.gatech.edu"))
arguments=$ENV(HOME)/online_learning_100/card/6_6/online_learning_permutahedron.py
log=online_learning_2.log
error=online_learning_2.error
output=online_learning_2.out
request_memory=5000
notification=error
notification=complete
notify_user=jmoondra3@gatech.edu
queue
