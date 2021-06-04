universe=vanilla
getenv=true
executable=/opt/python3/latest/bin/python3
+Requirements=((Machine=="isye-wilkins1.isye.gatech.edu")||(Machine=="isye-wilkins2.isye.gatech.edu")||(Machine=="isye-wilkins3.isye.gatech.edu")||(Machine=="isye-wilkins4.isye.gatech.edu"))
arguments=$ENV(HOME)/online_learning/online_learning_confidence.py
log=online_learning_confidence.log
error=online_learning_confidence.error
output=online_learning_confidence.out
request_memory=10000
notification = error
notification = complete
notify_user = jmoondra3@gatech.edu
queue