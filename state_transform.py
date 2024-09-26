from squander import N_Qubit_Decomposition_adaptive, Circuit
import numpy as np
import time
import sys

from scipy.stats import unitary_group
import scipy
import groqQCsim
import time

qbit_num = groqQCsim.qbit_num
gate_count = groqQCsim.gate_count

if __name__ == "__main__":

	target_qbits  = [12, 10]
	control_qbits = [-1, -1] #-1 ... no cotrol qubit is incolved in the operation
	
	
	#target_qbits  = [0,  7, 6,  4, 3, 5]
	#control_qbits = [15, 2, 10, -1, 14, 0] #-1 ... no cotrol qubit is incolved in the operation

	if len(target_qbits) != gate_count:
		raise Exception("Number of target qubits in the list of target qubits should be equal to the gate count")

	if len(control_qbits) != gate_count:
		raise Exception("Number of control qubits in the list of control qubits should be equal to the gate count")

	gate_count = len(target_qbits)

	# determine the soze of the unitary to be decomposed
	matrix_size = 1 << qbit_num
	   
	State_orig_real = np.random.rand( matrix_size )
	State_orig_imag = np.random.rand( matrix_size )
	State_orig = State_orig_real + 1j*State_orig_imag
	State_orig = State_orig/np.linalg.norm(State_orig)


	one_qubit_circuits = []
	circuit = Circuit(qbit_num)
	for gate_idx in range( len(target_qbits) ):

		if target_qbits[gate_idx] == control_qbits[gate_idx]:
			error("The control qubit and target qubit should be different")

		elif target_qbits[gate_idx] >= 0 and target_qbits[gate_idx] != 8 and target_qbits[gate_idx] < 21 and control_qbits[gate_idx] == -1:

			circuit.add_U3( target_qbits[gate_idx] )
		
			circuit_1 = Circuit( 1 )
			circuit_1.add_U3( 0 )
			one_qubit_circuits.append( circuit_1 )
		
		elif target_qbits[gate_idx] >= 0 and target_qbits[gate_idx] < 8 and control_qbits[gate_idx] >= 0 and control_qbits[gate_idx] < 8:	
	

			circuit.add_adaptive( target_qbits[gate_idx], control_qbits[gate_idx] )
	
			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0  )	
			one_qubit_circuits.append( circuit_1 )

		elif target_qbits[gate_idx] >= 0 and target_qbits[gate_idx] < 8 and control_qbits[gate_idx] > 8 and control_qbits[gate_idx] <= 21:	
	
			circuit.add_adaptive( target_qbits[gate_idx], control_qbits[gate_idx] )
	
			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0  )	
			one_qubit_circuits.append( circuit_1 )			
		elif target_qbits[gate_idx] >= 0 and target_qbits[gate_idx] < 8 and control_qbits[gate_idx] == 8:	
	
			circuit.add_adaptive( target_qbits[gate_idx], control_qbits[gate_idx] )
	
			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0  )	
			one_qubit_circuits.append( circuit_1 )			
		else:
			error("Unimplemented case wit target qubit and control qubit.")
		
		
		

	# get the number of free parameters in the circuit
	parameter_num = circuit.get_Parameter_Num()
	print('Number of parameters in the circuit: ', parameter_num, flush=True)

	#create an array of random parameters
	parameters = np.random.rand( parameter_num )*2*np.pi
	
	if parameters.size > 3:
		for gate_idx in range( len(target_qbits)-1 ):
			parameters[gate_idx*3:3*(gate_idx+1)] = parameters[(len(target_qbits)-1)*3:]
	#parameters[2] = -parameters[1]
	
	print( parameters )


	# apply the circuit to a quantum state
	State = State_orig.copy()
	State_transformed_oracle = State_orig.copy()
	start_time = time.time()
	circuit.apply_to( parameters, State_transformed_oracle )
	time_Cpp = time.time()-start_time


	#####################################################
	# get the gate kernel

	print( 'The single qubit gate kernels are:')

	gate_kernels = np.zeros((80, 2, 2,), dtype=np.complex64)
	parameter_idx = parameters.size
	for idx in range(80):
		if idx < gate_count:
			gate_parameter_num = one_qubit_circuits[idx].get_Parameter_Num()
			parameters_1       = parameters[parameter_idx-gate_parameter_num : parameter_idx]
			parameter_idx      = parameter_idx-gate_parameter_num

			print( parameters_1 )
			gate_kernel_single = one_qubit_circuits[idx].get_Matrix( parameters_1 )
			gate_kernels[idx]  = gate_kernel_single
			print( gate_kernel_single )
			print(' ')

	print( "gate kernel shape: ", gate_kernels.shape )


	#####################################################
	# running Groq 

	State_orig_real_float32 = np.real(State_orig).astype( np.float32 )
	State_orig_imag_float32 = np.imag(State_orig).astype( np.float32 )

	#State_orig_real_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
	#State_orig_imag_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)

	real_part, imag_part = groqQCsim.main(State_orig_real_float32, State_orig_imag_float32, target_qbits, control_qbits, gate_kernels, mxm_permutation = False)
	print(' ')
	print( 'difference between CPU oracle and Groq chip (real part): ', scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2), '(imag part): ', scipy.linalg.norm( imag_part - np.imag( State_transformed_oracle ) , 2) )

	print( 'difference between CPU oracle and Groq chip (overlap integral): ', np.abs( (real_part + imag_part*1j) @ State_transformed_oracle.T.conj() ) )


	print("time elapsed with C++: ", time_Cpp )

	print( State_orig_real_float32[0:] )
	print(' ' )
	print( np.real( State_transformed_oracle[0:] ) )
	print(' ')
	print( real_part[0:] )
	isnan = np.isnan( real_part ) 
	for idx in range(real_part.size):
		if isnan[idx] == True:
			print( idx, real_part[idx] )
	#print( imag_part[0:512] )	
	

	scipy.linalg.norm( real_part , 2)
	scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2)

