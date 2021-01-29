#External Script

#from IQ_sample_based_sim_LoRa import *
import glob
import subprocess
from subprocess import *
import os
class main_program:
    def __init__(self, path='./Final_EIB.py'):
        self.path=path
    #print("main program")
    def call_python_file(self):
        #print("enter call pyhton^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #os.system('python IQ_sample_based_sim_LoRa.py')
        subprocess.run(["python", "{}".format(self.path)])
        #print("function working")
    

if __name__=="__main__":
    c=main_program()
    #print("Event************** script simulation start")
    for z in range(5): #no. of time want to run event script
        print("Event script simulation start",z)
        c.call_python_file()
