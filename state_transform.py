from squander import N_Qubit_Decomposition_adaptive, Circuit, X
import numpy as np
import time

from scipy.stats import unitary_group
import scipy
import groqQCsim


qbit_num = 8

target_qbit = [(1, 2)]

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)

# creating a random unitary to be decomposed
Umtx = unitary_group.rvs(matrix_size)

State_orig = Umtx[:, 0]


# define quantum circuit
circuit = Circuit(qbit_num)
print(dir(circuit))
print(circuit.add_CNOT(target_qbit[0][0], target_qbit[0][1]))

# get the number of free parameters in the circuit
parameter_num = circuit.get_Parameter_Num()
print('Number of parameters in the circuit: ', parameter_num, flush=True)

# create an array of random parameters
parameters = np.random.rand( parameter_num )*2*np.pi


# apply the circuit to a quantum state
State = State_orig.copy()
State_transformed_oracle = State_orig.copy()
start_time = time.time()
circuit.apply_to( parameters, State_transformed_oracle )
time_Cpp = time.time()-start_time



#####################################################
# get the gate kernel

gate_kernel_single = np.eye( 4, 4, dtype = np.complex64)

gate_kernels = np.zeros((20, 4, 4,), dtype=np.complex64) #np.array([[[0,1],[1,0]]])
for idx in range(20):
	gate_kernels[idx] = gate_kernel_single

gate_kernels[0] = np.array([[0.11+0.51j,0.12-0.52j,0.13-0.53j,0.14-0.54j],
                            [0.15-0.55j,0.16+0.56j,0.17+0.57j,0.18+0.58j],
                            [0.19-0.59j,0.20+0.60j,0.21+0.61j,0.22+0.62j],
                            [0.23+0.63j,0.24+0.64j,0.25+0.65j,0.26+0.66j]], dtype = np.complex64)
print( "gate kernel shape: ", gate_kernels.shape )


#####################################################
# running Groq 

State_orig_real_float32 = np.real(State_orig).astype( np.float32 )
State_orig_imag_float32 = np.imag(State_orig).astype( np.float32 )

#State_orig_real_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
#State_orig_imag_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)

real_part, imag_part = groqQCsim.main(State_orig_real_float32, State_orig_imag_float32, target_qbit, gate_kernels)
print(' ')
print( 'difference between CPU oracle and Groq chip (real part): ', scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2), '(imag part): ', scipy.linalg.norm( imag_part - np.imag( State_transformed_oracle ) , 2) )


print("time elapsed with C++: ", time_Cpp )
'''
print( np.real( State_transformed_oracle ) )
print(' ')
print( real_part )
print(' ' )
print( State_orig_real_float32 )
'''
