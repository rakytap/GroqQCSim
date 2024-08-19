from squander import N_Qubit_Decomposition_adaptive, Circuit
import numpy as np
import time
import sys

from scipy.stats import unitary_group
import scipy
import groqQCsim
import time

qbit_num = groqQCsim.qbit_num

if __name__ == "__main__":
	target_qbit  = 6
	control_qbit = 2 #i.e. no control qubit is involved

	# determine the soze of the unitary to be decomposed
	matrix_size = 1 << qbit_num
	   
	State_orig_real = np.random.rand( matrix_size )
	State_orig_imag = np.random.rand( matrix_size )
	State_orig = State_orig_real + 1j*State_orig_imag
	State_orig = State_orig/np.linalg.norm(State_orig)

	if target_qbit >= 0 and target_qbit < 8 and control_qbit == -1:
		# define quantum circuit
		circuit = Circuit(qbit_num)
		circuit.add_U3( target_qbit )

		circuit_1 = Circuit( 1 )
		circuit_1.add_U3( 0 )
		
	elif target_qbit >= 0 and target_qbit < 8 and control_qbit >= 0 and control_qbit < 8:	
	
		# define quantum circuit
		circuit = Circuit(qbit_num)
		circuit.add_adaptive( target_qbit, control_qbit )
	
		circuit_1 = Circuit( 1 )
		circuit_1.add_RY( 0  )	
		
		
		

	# get the number of free parameters in the circuit
	parameter_num = circuit.get_Parameter_Num()
	print('Number of parameters in the circuit: ', parameter_num, flush=True)

	#create an array of random parameters
	parameters = np.random.rand( parameter_num )*2*np.pi
	#parameters[2] = -parameters[1]




	# apply the circuit to a quantum state
	State = State_orig.copy()
	State_transformed_oracle = State_orig.copy()
	start_time = time.time()
	circuit.apply_to( parameters, State_transformed_oracle )
	time_Cpp = time.time()-start_time


	#####################################################
	# get the gate kernel

	gate_kernel_single = circuit_1.get_Matrix( parameters )
	print( 'The singla qubit gate kernel is:')
	print( gate_kernel_single )

	gate_kernels = np.zeros((80, 2, 2,), dtype=np.complex64)
	for idx in range(80):
		gate_kernels[idx] = gate_kernel_single

	print( "gate kernel shape: ", gate_kernels.shape )


	#####################################################
	# running Groq 

	State_orig_real_float32 = np.real(State_orig).astype( np.float32 )
	State_orig_imag_float32 = np.imag(State_orig).astype( np.float32 )

	#State_orig_real_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
	#State_orig_imag_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)

	real_part, imag_part = groqQCsim.main(State_orig_real_float32, State_orig_imag_float32, target_qbit, control_qbit, gate_kernels, mxm_permutation = False)
	print(' ')
	print( 'difference between CPU oracle and Groq chip (real part): ', scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2), '(imag part): ', scipy.linalg.norm( imag_part - np.imag( State_transformed_oracle ) , 2) )

	print( 'difference between CPU oracle and Groq chip (overlap integral): ', np.abs( (real_part + imag_part*1j) @ State_transformed_oracle.T.conj() ) )


	print("time elapsed with C++: ", time_Cpp )


	print( 'The singla qubit gate kernel is:')
	print( gate_kernel_single )
	print(' ' )

	print( State_orig_real_float32[0:] )
	print(' ' )
	print( np.real( State_transformed_oracle[0:] ) )
	print(' ')
	print( real_part[0:] )


