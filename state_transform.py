from squander import N_Qubit_Decomposition_adaptive, Gates_Block
import numpy as np
import time
import sys

from scipy.stats import unitary_group
import scipy
import groqQCsim


qbit_num = 8

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)
   
# creating a random unitary to be decomposed
Umtx = unitary_group.rvs(matrix_size)

State_orig = Umtx[:,0]



# define quantum circuit
circuit = Gates_Block(qbit_num)
circuit.add_RX( 5 )


# get the number of free parameters in the circuit
parameter_num = circuit.get_Parameter_Num()
print('Number of parameters in the circuit: ', parameter_num, flush=True)

#create an array of random parameters
parameters = np.random.rand( parameter_num )*2*np.pi


# apply the circuit to a quantum state
State = State_orig.copy()
circuit.Apply_To( parameters, State )




#####################################################
# running Groq 

State_orig_real_float32 = np.real(State_orig).astype( np.float32 )
State_orig_imag_float32 = np.imag(State_orig).astype( np.float32 )

#State_orig_real_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
#State_orig_imag_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)

real_part = groqQCsim.main(State_orig_real_float32, State_orig_imag_float32)
print( 'Error: ', scipy.linalg.norm( real_part - State_orig_real_float32 , 2) )

print( real_part )
print(' ' )
print( State_orig_real_float32 )
