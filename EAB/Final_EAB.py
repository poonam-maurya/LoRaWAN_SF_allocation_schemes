# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:30:27 2021

@author: hp
"""

#the access probabilities will be determined based on the slots
#the channel "attempt rate" will be determined by the sample interval as well

from bitstring import BitArray, BitStream
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import geom
from scipy.fftpack import fft
import os
import xlsxwriter 
NumGW = 1; #no. of gateway
NumDev=300; #no. of mobile devices
PacketLenBytes=40;
#workbook = xlsxwriter.Workbook('hello.xlsx')
#worksheet = workbook.add_worksheet()

row=0
over_all_packet=0
over_all_success_rate=0
over_all_attempt=0

col=0
#total_attempt=[0 for i in range(NumDev)]
target_thinning_prob=.8
mobilityscale=500
a_i=0
a_up=[]
number_of_preamble=7
preamble_up=[]

I_Q_sample_physical_layer_preamble=[]  

log_enabled = {}
log_enabled["NODE"]=0
log_enabled["GW"]=0
log_enabled["MAIN"]=1

def print_log(device,*argv):
    if log_enabled[device]==1:
        print(device,end = "")
        print(":\t",end=" ")
        for arg in argv:
            print(arg,end =" ")
        print()

def save_context(varname,varvalue):
    filename="SavedVars/"+varname
    f=open(filename, "w")
    f.write(varvalue)
    f.close()

def load_context(varname,defaultvalue):
    filename="SavedVars/"+varname
    if os.path.exists(filename):
        f=open(filename, "r")
        return(f.read())
        f.close()
    else:
        return(defaultvalue)

def MAC_PHYSICAL_LAYER_PACKET(mac_payload_size,SF,mac_payload_stream=None):
    if mac_payload_stream==None:
        mac_payload_stream = BitArray(mac_payload_size) ##ba change## #generate bitstream of length mac_payload_size
    #chopping the mac bit-stream into packets of SF length for LoRa modulation 
    step=0
    array_physical_symbol_bit=[]
    array_physical_symbol_decimal=[]
    I_Q_sample_physical_layer=[]
    M=2**SF
    for i in range(int(mac_payload_size/SF)):
        array_physical_symbol_bit.append(mac_payload_stream[step:step+int(SF)])   
        step=int(SF)+step

    #converting the each pysical layer packet bit-stream into its decimal equivalent 
    for j in range(len(array_physical_symbol_bit)):
        i=0
        for bit in array_physical_symbol_bit[j]:
            i=(i<<1) |bit
        array_physical_symbol_decimal.append(i)
        
    # modulating each physical packet symbol with up-chrips
    a_up=array_physical_symbol_decimal
    #preamble aadition in mac payload at physical layer in order to send in air
    for i in range(number_of_preamble):
        for n in range(int(M)):
            preamble_up.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_i/M)-.5)*n)))
      
    for i in range(len(a_up)): #for each symbol
        Lora_signal_up1=[]
        for n in range(int(M)):
            Lora_signal_up1.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_up[i]/M)-.5)*(n))))
            I_Q_sample_physical_layer.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_up[i]/M)-.5)*(n)))) #collecting total I/Q samples of physical layer packet
    
    I_Q_sample_physical_layer_preamble.append(preamble_up+I_Q_sample_physical_layer)

    return I_Q_sample_physical_layer

def LoRa_Receiver_demodulation(I_Q_sample_physical_layer,SF):
    Received_packet_IQ=[]
    Lora_up_conjugate1=[]
    step1=0
    a_i=0
    M=2**SF
    received_symbol=[]
    received_symbol_bits=[]
    received_symbol_bits1=[]
    received_symbol_bits2=[]
    mac_payload_at_receiver=[]

    for i in range(int(len(I_Q_sample_physical_layer)/(M))):
        Received_packet_IQ.append(I_Q_sample_physical_layer[step1:step1+int(M)])
        step1=step1+int(M)
    for i in range(len(Received_packet_IQ)):
        dechriping_lora_up1=[]
        for n in range(int(M)):
            Lora_up_conjugate1.append(np.exp(-1j*2*np.pi*(((n**2)/(2*M))+((a_i/M)-.5)*n)))
            dechriping_lora_up1.append(Received_packet_IQ[i][n]*Lora_up_conjugate1[n])
        d_fft=fft(dechriping_lora_up1)
        maximum_fre=np.argmax(d_fft)
        received_symbol.append(maximum_fre)

    for i in range(len(received_symbol)):
        received_symbol_bits.append(bin(received_symbol[i]))
        received_symbol_bits1.append(received_symbol_bits[i][2:])
        received_symbol_bits2.append(received_symbol_bits1[i].zfill(int(SF)))
    mac_payload_at_receiver.append("".join(received_symbol_bits2))
    
    return received_symbol


def collision_detection(num_recived_sample, collision_status): 
    if num_recived_sample>1:
        if collision_status==0:
            collision_status=1
            print("collision detected at gateway")
    return(collision_status)
    
    
def transmission_parameter(node_id):
    if node_id <50:
        SF=7
        return(SF)
    if ((node_id >49) and (node_id<100)):
        SF=8
        return(SF)
    if ((node_id >99) and (node_id<150)):
        SF=9
        return(SF)
    if ((node_id >149) and (node_id<200)):
        SF=10
        return(SF)
    if ((node_id >199) and (node_id<250)):
        SF=11
        return(SF)
    if ((node_id >249) and (node_id<300)):
        SF=12
        return(SF)
    
    
    # SF = np.random.randint(7,12)
    # return(SF)

def application_payload_format(tx_symbol,num,SF):    #P: grneration of application payload
    #print("transmitting symbol",tx_symbol,num,SF)
    payload= BitArray(int=tx_symbol,length=SF)
    #payload=payload+BitArray(int=num,length=SF)
    #print("payload original",payload)
    #print("transmitting symbol*******",tx_symbol,num,SF)
    return payload

location_node=[]
theta_m=.1
location_node=[]
a=[]
def node_distribution(all_node_num):
    final_location=[]
    def node_distribution_cell(r,st):
        
        location=int(np.ceil(np.random.randint(0,359)/theta_m))
        
        X = r * math.cos(location*.1)  
        Y = r * math.sin(location*.1)  
        location_node.append((X,Y))
        a.append(st)
        if len(location_node)==300:
            xs=[x[0] for x in location_node]
            ys=[x[1] for x in location_node]
            #c=[x[2] for x in location_node]
            plt.scatter(xs,ys,c=a)
            plt.show()
        return(location)
    for r in range(1,7):
        #print("under for loop")
        node_per_zone=int(all_node_num/6)
        if r==1:
          
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(0,.5),'red')
                all_SFs=7
                final_location.append(location)
        if r==2:
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(.5,1),'blue')
                all_SFs=7
                final_location.append(location)
        if r==3:
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(1,1.5),'yellow')
                all_SFs=9
                final_location.append(location)
        if r==4:
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(1.5,2),'brown')
                all_SFs=10
                final_location.append(location)
        if r==5:
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(2,2.5),'orange')
                all_SFs=11
                final_location.append(location)
        if r==6:
            for i in range(node_per_zone):
                location=node_distribution_cell(np.random.uniform(2.5,3),'green')
                all_SFs=12
                final_location.append(location)
    
    
    return(final_location)
def next_node_location(time,loc):
    for i in range(time//mobilityscale):#get time/mobilityscale number of transitions
            if np.random.random()<0.5:
                loc=(loc + 1)%int(360/theta_m)
            else:
                loc=(loc - 1)%int(360/theta_m)
    return(loc)
    
    
class Node(object):
    #initializes the location and other parameters of the node
    def __init__(self,num,node_d):#for symmetric random walk
        strn="node"+str(num)+"loc"
        #initial angle, in theta_m units
        #print_log("NODE","Initial Location ",self.loc);
        #print("all location",self.loc)
        self.mobilityscale=15000000; #mobilityscale is in terms of samples. For each mobilityscale number of samples, the node moves left or right with equal probability
        #this is also the scale at which next transmission probabilities are decided
        strn="node"+str(num)+"p"
        #self.p=float(load_context(strn,np.random.uniform(target_thinning_prob/2.0,target_thinning_prob))); #probability of transmitting in a sample duration
        self.p=float(load_context(strn,np.random.uniform(.5,.5))); #probability of transmitting in a sample duration
        #the above initialization should be less than the target thinning probability as target thinning probability is upper bounded by p
        strn="node"+str(num)+"next_event"
        self.next_event=int(load_context(strn,self.mobilityscale*(np.random.uniform(1,1.02))*geom.rvs(self.p))); #gets the first value for tranmission slot. staggers the exact transnmission slot to avoid inter-node synchronization
        #this is not the global time. this is time-to-next-event
        #print("current time of simulation", self.next_event)
        self.state="IDLE"; 
        self.samplenum=0;  #the ongoing IQ sample number
        self.num=num;
        self.node_d=node_d[self.num]
        #print("node ids",num,self.node_d)
        strn="node"+str(num)+"num_attempts"
        self.loc=int(load_context(strn,self.node_d)); 
        #self.loc=6
        self.num_attempts=int(load_context(strn,1));
        strn="node"+str(num)+"total_attempt"
        self.total_attempt=int(load_context(strn,0))
        strn="node"+str(num)+"total_attempt1"
        self.total_attempt1=int(load_context(strn,0))
        strn="node"+str(num)+"total_num_received"
        self.total_num_received=int(load_context(strn,0))
        self.all_loc=8
        strn="node"+str(num)+"total_collision"
        self.total_collision=int(load_context(strn,0))
        
        
        #print("in node class attempt",self.total_attempt, self.num )
        #2 is added to the length to ensure that the begining and end
        #are zero so that the receiver can perform energy detection.
        #print("node num",self.num)
        strn="node"+str(num)+"SF"
        self.SF=int(load_context(strn,transmission_parameter(self.num))) # Poonam: edited
        #print("SF allocation corresponding node ids",self.SF)
        payload= application_payload_format(self.all_loc,self.num,self.SF)
        
        
        y=MAC_PHYSICAL_LAYER_PACKET(mac_payload_size=len(payload),SF=self.SF,mac_payload_stream=payload)
        
        self.pktlen=len(y)+2; #assume len(y) IQ samples per physical layer transmission.
        self.IQ=(0+0j)*np.ones(self.pktlen); #replace this by IQ samples
        
        self.IQ[1:len(y)+1]=y;
        strn="node"+str(num)+"last_event_time"
        self.last_event_time=int(load_context(strn,0));
        #print_log("NODE","Initial next event schedule",self.last_event_time+self.next_event);
        
    def get_node_num(self):
        return self.num

    def get_next_time(self):
        return self.next_event
    
    def do_event(self):
        self.change_loc=next_node_location(self.next_event,self.loc); #self.next_event is the last time interval
        self.loc=self.change_loc
        self.last_event_time=self.last_event_time+self.next_event;#current time
        #print("last event time of node**********",self.last_event_time)
        if self.state=="IDLE": #next step is transmission
            self.state="Tx";
            self.samplenum=1;
            print_log("NODE", "attempt no. ",self.num,self.num_attempts,self.loc,self.last_event_time)
            self.next_event=1; #next event is IQ sample transmission again
        else:
            if self.state=="Tx":
                if self.samplenum==self.pktlen: #last packet
                    
                    self.state="IDLE"; 
                    self.next_event=int(self.mobilityscale*(np.random.uniform(1,1.05))*geom.rvs(self.p)); #gets the first value for tranmission slot. staggers the exact transnmission slot to avoid inter-node synchronization
                    
                    self.cur_loc=self.get_loc()
                    print_log("NODE", "Going to Idle...",self.num,self.last_event_time,self.cur_loc);
                    self.change_loc=next_node_location(self.next_event,self.loc);
                    self.loc=self.change_loc
                    payload = application_payload_format(self.all_loc,self.num,self.SF)
                    
                    y=MAC_PHYSICAL_LAYER_PACKET(mac_payload_size=len(payload),SF=self.SF,mac_payload_stream=payload)
                    self.IQ[1:len(y)+1]=y;
                    self.samplenum=0;
                    
                    self.num_attempts=self.num_attempts+1;
                    self.total_attempt1=self.total_attempt1+1
                else: #not transiting to IDLE
                    self.state="Tx";
                    self.samplenum=self.samplenum+1;
                    self.next_event=1; #next event is IQ sample transmission again

    def get_state(self):
        return self.state;

    def get_samplenum(self):
        return self.samplenum;

    def get_iq(self,num):
        if num<self.pktlen:
            return self.IQ[num];
        else:
            return 0+0j; #nothing to be sent when going to idle. this should never happen

    def get_pktlen(self):
        return(self.pktlen);

    def get_loc(self):
        return(self.loc);
    def get_SF(self):
        return(self.SF)

    

    def get_last_event_time(self):
        return(self.last_event_time)
    
class GW():
    #initializes the location and other parameters of the node
    def __init__(self):#for symmetric random walk
        strn="gateway_loc"
        self.loc=int(load_context(strn,np.random.randint(0,359))); #Fixed Locations
        self.iq=[];
        self.rx=[];
        self.energy_threshold=0.5;# the energy threshold for detection
        self.frame_ongoing=0; #to differentiate start of frame from end of frame
        self.current_iq_train=[];
        self.is_collision=0;
        self.decoded=0;
        self.was_collision=0;
        self.node_sample_count=1000000;
        self.node_num=[] #just initialise 
        self.num_sample_current_instant=0
        self.received_SF=0
        self.SF_all=[]
        self.node_receive_all=0
        
    def start_receiving_iq(self): #means a new event has happened
        self.num_sample_current_instant=0;#reset for the next sample
        self.iq.append(0+0j);
        self.SF_all=[]
        #print_log("GW", "initialized iq",self.iq);

    def receive_iq(self,loc,source_iq,node,SF): #add iq component to the currently received sample
        #loc is the location of the sender node. this is to get the channel
        self.received_SF=SF
        self.node_num=node
        self.SF_all.append(SF)
        if abs(source_iq)>self.energy_threshold:
            if len(self.SF_all)>1:
                if len(set(self.SF_all))==len(self.SF_all):
                    self.num_sample_current_instant=self.num_sample_current_instant+1
                else:
                    self.num_sample_current_instant=3
                    self.is_collision=collision_detection(self.num_sample_current_instant,self.is_collision)
            
            
            else:
                self.num_sample_current_instant=self.num_sample_current_instant+1;#count the number of transmitters
                self.is_collision=collision_detection(self.num_sample_current_instant,self.is_collision)
                
        self.iq[-1]=self.iq[-1]+self.channel(loc)*source_iq

    def noise(self):
        return(0+0j); #AWGN to be added

    def channel(self, loc):
        return(1); 

    def stop_receiving_iq(self):
        

        if self.num_sample_current_instant>0:
            self.current_iq_train.append(self.iq[-1])
            if self.frame_ongoing==0:
                print_log("GW", "start of a new frame");
                self.frame_ongoing=1;
        else: #means an idle sample
            print_log("GW", "an Idle sample found");
            if self.frame_ongoing==1:
                print_log("GW", "Tx to Idle transition");
                self.frame_ongoing=0; #get ready for detecting the next start of frame
                self.was_collision=self.is_collision;
                if self.is_collision==0:
                    self.node_receive_all=self.node_receive_all+1
                    #self.rx=LoRa_Receiver_demodulation(I_Q_sample_physical_layer=self.current_iq_train,SF=self.received_SF)
                    #if self.node_num==0:
                    #print("received symbol",self.rx, self.received_SF)
                    self.decoded=1;
                else:
                    self.rx=[]; #to ensure that the previous decoded value is not carried over
                self.is_collision=0; #reset so that next frame starts with no collision assumption
                del(self.current_iq_train);
                self.current_iq_train=[];
        del(self.iq)
        self.iq=[];

        if self.was_collision==1: #means an idle sample and also a collision
            self.was_collision=0;
            print("collision is happend between nodes********************************************", self.node_num)
            return("collided");
        if self.decoded==1: #print the message that was received and decoded, when no collision
            self.decoded=0;
            #print_log("GW", "decoded: ",self.rx)
            return(self.node_num);



def find_thinning_prob(sucess,attempt):
    return(sucess/attempt)

node_d=[]
node_d=node_distribution(NumDev)

print("all node location",node_d)
#generate the nodes
#nodes=[Node(num=i) for i in range(NumDev)]
#print("node ID",nodes)
nodes=[]
for i in range(NumDev):
    node_v=Node(i,node_d)
    nodes.append(node_v)
gws=[GW() for i in range(NumGW)]

cur_time=int(load_context("cur_time",0));

loc_est=[[] for i in nodes]
    
num_received=[0 for i in nodes];
#strn="node"+str(nodes)+"total_attempt"
#total_attempt=[int(load_context(strn,0)) for i in nodes]
#print("my totatal attempt",total_attempt)

for j in nodes:
    strn="node"+str(j.num)+"num_received"
    num_received[j.num] = int(load_context(strn,0)); #if something from previous simulation then load otherwise load 0


y=0

max_num_events=300000

if cur_time==0:
    for i in nodes:
        print_log("MAIN",",", cur_time,",",i.p,",","not known",",",i.num_attempts,",", i.num)

for i in range(max_num_events): #number of events
    
    time_to_next_event=10000000;
        
    for j in gws:
        j.start_receiving_iq();#new event has happened, add an IQ element to the array at the receiver

    for j in nodes:
        if j.get_last_event_time()+j.get_next_time()<cur_time+time_to_next_event:
            time_to_next_event=j.get_last_event_time()+j.get_next_time()-cur_time;
    cur_time=cur_time+time_to_next_event; #this gives minimum current time i.e. when the simulation get start
    iq=0;

    for j in nodes:
        if j.get_last_event_time()+j.get_next_time()==cur_time: 
            #node who time matches with current time start the transmission
            
            for g in gws: 
                
                g.receive_iq(source_iq=j.get_iq(j.get_samplenum()), loc=j.get_loc(),node=j.get_node_num(),SF=j.get_SF());#new event has happened, add an IQ element to the array at the receiver
            j.do_event();
    
    for j in gws:
        y=j.stop_receiving_iq();
        if y!=None:#means this was the last IQ sample
            if y!="collided":
                sending_node=y;
                #print_log("MAIN", "One event ended with success",cur_time,sending_node)
                num_received[sending_node]=num_received[sending_node]+1;
                
                if num_received[sending_node]%1 == 0:
                    #print("number of rcecived packets", num_received[sending_node], sending_node)
                    thinning_probability=(nodes[sending_node].p)*find_thinning_prob(num_received[sending_node],nodes[sending_node].num_attempts);
                    
                    if abs(target_thinning_prob-thinning_probability)>0.05:
                        
                        if thinning_probability<target_thinning_prob:
                           # print_log("MAIN","target is more")
                            nodes[sending_node].p=nodes[sending_node].p+0.1*(target_thinning_prob-thinning_probability) #Increase
                        else:
                            #print_log("MAIN","target is less")
                            nodes[sending_node].p=nodes[sending_node].p+0.1*(target_thinning_prob-thinning_probability) #decrease
                        if nodes[sending_node].p<0.00005:
                            nodes[sending_node].p=0.00005
                        if nodes[sending_node].p>0.9:
                            nodes[sending_node].p=0.9
                    print_log("MAIN",",", cur_time,",",nodes[sending_node].p,",",thinning_probability,",",nodes[sending_node].num_attempts,",",y)
                    # worksheet.write(row, col, nodes[sending_node].p )
                    # worksheet.write(row, col+1, (nodes[sending_node].num_attempts-10)/nodes[sending_node].num_attempts )
                    #total_num_attempts[sending_node]=nodes[sending_node].num_attempts+total_num_attempts[sending_node]
                    #print("before addition total attenpt",total_attempt[sending_node])
                    nodes[sending_node].total_attempt=nodes[sending_node].num_attempts+nodes[sending_node].total_attempt
                    nodes[sending_node].total_num_received=nodes[sending_node].total_num_received+num_received[sending_node]
                    nodes[sending_node].total_collision= nodes[sending_node].total_attempt- nodes[sending_node].total_num_received
                    nodes[sending_node].num_attempts=0
                    print("total attmepts of node^^^^^^^^^^^^^^^^^^^^^^^^^^^^",nodes[sending_node].total_attempt, sending_node, nodes[sending_node].total_collision,nodes[sending_node].total_num_received)
                   
                    num_received[sending_node]=0
                row=row+1
                    
            #else:
                #print_log("MAIN", "***********************COLLISION*******************************");

    #exit should be at the end when the event before this IDLE event is processed
    if i>int(0.5*max_num_events):
        idle=1
        for j in nodes:
            if j.state!="IDLE":
                idle=0;
        if idle==1:
            #print_log("MAIN","System found to be idle  ", cur_time);
            break;
#workbook.close()
save_context("cur_time",str(cur_time)); #Store the current status of nodes and gateway
#print("cur_time",cur_time)
for j in nodes:
    strn="node"+str(j.num)+"SF"
    save_context(strn,str(j.SF))
    
    strn="node"+str(j.num)+"loc"
    
    save_context(strn,str(j.get_loc()));
    strn="node"+str(j.num)+"p"
    save_context(strn,str(j.p));
    strn="node"+str(j.num)+"next_event"
    save_context(strn,str(j.next_event));
    strn="node"+str(j.num)+"state"
    save_context(strn,str(j.state));
    strn="node"+str(j.num)+"last_event_time"
    save_context(strn,str(j.last_event_time));
    strn="node"+str(j.num)+"num_attempts"
    save_context(strn,str(j.num_attempts));
    strn="node"+str(j.num)+"total_attempt"
    save_context(strn,str(j.total_attempt));
    strn="node"+str(j.num)+"total_attempt1"
    save_context(strn,str(j.total_attempt1));
    #print("total attemt", j.total_attempt)
    strn="node"+str(j.num)+"num_received"
    save_context(strn,str(num_received[j.num]));
    strn="node"+str(j.num)+"total_num_received"
    save_context(strn,str(j.total_num_received));
    strn="node"+str(j.num)+"total_collision"
    save_context(strn,str(j.total_collision));
for g in gws:
    strn="gateway_loc"
    
    save_context(strn, str(g.loc))
over_all_packet=0
over_all_attempt=0
for j in nodes:
    over_all_packet= j.total_num_received+over_all_packet
    over_all_attempt=j.total_attempt1+over_all_attempt
over_all_success_rate= over_all_packet/over_all_attempt
print("over all attempt and over_all_received",over_all_attempt, over_all_packet)    
print("over all success probability", over_all_success_rate )
##print("success rate , total attempt, total receic=ved", j.num, j.total_attempt, j.total_num_received, j.total_collision)
