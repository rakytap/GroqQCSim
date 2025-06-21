from squander import N_Qubit_Decomposition_adaptive, Circuit
import numpy as np
import time
import sys

from scipy.stats import unitary_group
import scipy
from memory_layouts import MemoryLayouts
import groqQCsim
import time
import random

qbit_num = 20
gate_count = 7*256
used_gate_count = 1

ML = MemoryLayouts( qbit_num, gate_count )
ML.used_gate_count = used_gate_count


if __name__ == "__main__":

	'''
	target_qbits = []
	control_qbits = []	
	while len(target_qbits) < used_gate_count:
		target_qbit = random.randint(0, qbit_num-1)
		
		tmp = random.randint(0,1)
		if tmp == 0:
			control_qbit = -1
		else:
			control_qbit = random.randint(0, qbit_num-1)	
		
			

		if target_qbit != control_qbit:		
			target_qbits.append( target_qbit )
			control_qbits.append( control_qbit )		
        
	
	if len(target_qbits) != used_gate_count:
		raise Exception("Number of target qubits in the list of target qubits should be equal to the gate count")

	if len(control_qbits) != used_gate_count:
		raise Exception("Number of control qubits in the list of control qubits should be equal to the gate count")

	'''     

	# determine the soze of the unitary to be decomposed
	matrix_size = 1 << qbit_num
	
	State_orig_real = np.random.rand( matrix_size )
	State_orig_imag = np.random.rand( matrix_size )
	#State_orig_real = np.zeros(matrix_size, dtype=np.float_)
	#State_orig_real[0] = 1.0
	#State_orig_imag = np.zeros(matrix_size, dtype=np.float_)
	State_orig = State_orig_real + 1j*State_orig_imag
	State_orig = State_orig/np.linalg.norm(State_orig)

	'''
	one_qubit_circuits = []
	circuit = Circuit(qbit_num)
	circuit2 = Circuit(qbit_num) ########
	for gate_idx in range( len(target_qbits) ):

		target_qbit  = target_qbits[gate_idx]
		control_qbit = control_qbits[gate_idx]

		if target_qbit == control_qbit:
			error("The control qubit and target qubit should be different")

		elif target_qbit >= 0 and target_qbit < 21 and control_qbit == -1:
			print("adding U3 gate")
			circuit.add_U3( target_qbit )
		
			circuit_1 = Circuit( 1 )
			circuit_1.add_U3( 0 )
			one_qubit_circuits.append( circuit_1 )

			if gate_idx < ML.second_gate_count:
				circuit2.add_U3( target_qbit )
		
		elif target_qbit >= 0 and target_qbit < 21 and control_qbit >= 0 and control_qbit < 21 :	
	
			print("adding controlled gate")
			circuit.add_CRY( target_qbit, control_qbit )
	
			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0  )		
			one_qubit_circuits.append( circuit_1 )

			if gate_idx < ML.second_gate_count:
				circuit2.add_CRY( target_qbit, control_qbit )
		
		else:
			error("Unimplemented case wit target qubit and control qubit.")
		
	'''
     
	target_qbits = []
	control_qbits = []	
	one_qubit_circuits = []
	circuit = Circuit(qbit_num)
	
	circuit.add_CNOT(18,16)
	target_qbits.append(18)
	control_qbits.append(16)

	circuit_1 = Circuit( 1 )
	circuit_1.add_X( 0 )
	one_qubit_circuits.append( circuit_1 )
	'''
	layers = 5000
	gate_idx = 1

	for layer in range(layers):

		
    	# single qubit gates in a ablock are then merged into a single gate kernel.
		circuit.add_RZ(1)
		target_qbits.append(1)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RZ( 0 )
		one_qubit_circuits.append( circuit_1 )

		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
			
		
		circuit.add_RY(1)
		target_qbits.append(1)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RY( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
		
		circuit.add_RZ(1)  
		target_qbits.append(1)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RZ( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
                           
		
		circuit.add_RZ(0)
		target_qbits.append(0)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RZ( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break

		circuit.add_RY(0)
		target_qbits.append(0)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RY( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
			
		circuit.add_RZ(0) 
		target_qbits.append(0)
		control_qbits.append(-1)

		circuit_1 = Circuit( 1 )
		circuit_1.add_RZ( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
			        
          
		     
		circuit.add_CNOT(1,0);
		target_qbits.append(1)
		control_qbits.append(0)

		circuit_1 = Circuit( 1 )
		circuit_1.add_X( 0 )
		one_qubit_circuits.append( circuit_1 )
		gate_idx = gate_idx + 1
		if gate_idx >= used_gate_count:
			break
		
		
		for control_qbit in range(1, qbit_num-1, 2):
			if (control_qbit+2<qbit_num)   :                    

				circuit.add_RZ(control_qbit+1)
				target_qbits.append(control_qbit+1)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RZ( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			
				circuit.add_RY(control_qbit+1)
				target_qbits.append(control_qbit+1)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RY( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			
				circuit.add_RZ(control_qbit+1)
				target_qbits.append(control_qbit+1)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RZ( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			                   
			
				circuit.add_RZ(control_qbit+2)
				target_qbits.append(control_qbit+2)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RZ( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			
				circuit.add_RY(control_qbit+2)
				target_qbits.append(control_qbit+2)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RY( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			
				circuit.add_RZ(control_qbit+2)
				target_qbits.append(control_qbit+2)
				control_qbits.append(-1)

				circuit_1 = Circuit( 1 )
				circuit_1.add_RZ( 0 )
				one_qubit_circuits.append( circuit_1 )
				gate_idx = gate_idx + 1
				if gate_idx >= used_gate_count:
					break
			
                            
				#if (control_qbit >= 15 ):
				#	continue
		  

				circuit.add_CNOT(control_qbit+2,control_qbit+1)
				target_qbits.append(control_qbit+2)
				control_qbits.append(control_qbit+1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_X( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
			
                        
		
			circuit.add_RZ(control_qbit+1)
			target_qbits.append(control_qbit+1)
			control_qbits.append(-1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_RZ( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
		
			circuit.add_RY(control_qbit+1)
			target_qbits.append(control_qbit+1)
			control_qbits.append(-1)
		
			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
		
			circuit.add_RZ(control_qbit+1)
			target_qbits.append(control_qbit+1)
			control_qbits.append(-1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_RZ( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
			             

			circuit.add_RZ(control_qbit)
			target_qbits.append(control_qbit)
			control_qbits.append(-1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_RZ( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
			
			circuit.add_RY(control_qbit)
			target_qbits.append(control_qbit)
			control_qbits.append(-1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_RY( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
			
			circuit.add_RZ(control_qbit)
			target_qbits.append(control_qbit)
			control_qbits.append(-1)

			circuit_1 = Circuit( 1 )
			circuit_1.add_RZ( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
			        

			circuit.add_CNOT(control_qbit+1,control_qbit)
			target_qbits.append(control_qbit+1)
			control_qbits.append(control_qbit)

			circuit_1 = Circuit( 1 )
			circuit_1.add_X( 0 )
			one_qubit_circuits.append( circuit_1 )
			gate_idx = gate_idx + 1
			if gate_idx >= used_gate_count:
				break
		
		if gate_idx >= used_gate_count:
				break
		

       
	'''
	
            
	# get the number of free parameters in the circuit
	parameter_num = circuit.get_Parameter_Num()
	print('Number of parameters in the circuit: ', parameter_num, flush=True)

	#create an array of random parameters
	np.random.seed(0)
	parameters = np.random.rand( parameter_num )*2*np.pi
	#print('parameters: ', parameters )
	

	# apply the circuit to a quantum state
	State = State_orig.copy()
	State_transformed_oracle = State_orig.copy()
	start_time = time.time()
	
	circuit.apply_to( parameters, State_transformed_oracle )
	time_Cpp = time.time()-start_time


	#####################################################
	# get the gate kernels
    
	gates = circuit.get_Gates()
    
	# constructing quantum circuit
	#for gate in gates:


	print( 'The single qubit gate kernels are:')
	start_time = time.time()
	gate_kernels = np.zeros((ML.gate_splits*80, 2, 2,), dtype=np.complex64)
	for idx in range(ML.gate_splits*80):
		if idx < used_gate_count:

			gate = gates[idx]
			parameter_idx = gate.get_Parameter_Start_Index()


			gate_parameter_num = one_qubit_circuits[idx].get_Parameter_Num()

			parameters_1       = parameters[parameter_idx: parameter_idx+gate_parameter_num]

			gate_kernel_single = one_qubit_circuits[idx].get_Matrix( parameters_1 )
			gate_kernels[idx]  = gate_kernel_single

	time_kernel_extract = time.time()-start_time

	print( "gate kernel shape: ", gate_kernels.shape )

	#####################################################
	# running Groq 

	State_orig_real_float32 = np.real(State_orig).astype( np.float32 )
	State_orig_imag_float32 = np.imag(State_orig).astype( np.float32 )

	#State_orig_real_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
	#State_orig_imag_float32 = np.asarray([k for k in range(256)], dtype=np.uint8)
	


	real_part, imag_part = groqQCsim.main(qbit_num, gate_count, State_orig_real_float32, State_orig_imag_float32, target_qbits, control_qbits, gate_kernels, mxm_permutation = False)
	print(' ')
	print( 'difference between CPU oracle and Groq chip (real part): ', scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2), '(imag part): ', scipy.linalg.norm( imag_part - np.imag( State_transformed_oracle ) , 2) )

	print( 'difference between CPU oracle and Groq chip (overlap integral): ', np.abs( (real_part + imag_part*1j) @ State_transformed_oracle.T.conj() ) )
	print("time elapsed with extracting gate kernels: ", time_kernel_extract )	
	print("time elapsed with C++: ", time_Cpp )	



	print( State_orig_real_float32[0:] )
	print(' ' )
	print( np.real( State_transformed_oracle[0:] ) )
	print(' ')
	print( real_part[0:] )
	'''
	for idx in range( State_transformed_oracle.size):
		print( idx, idx & (1 << 8), State_orig_real_float32[idx], np.real( State_transformed_oracle[idx] ), real_part[idx] )
	'''

	isnan = np.isnan( real_part ) 
	for idx in range(real_part.size):
		if isnan[idx] == True:
			print( idx, real_part[idx] )
		'''
		if ( abs( real_part[idx] - np.real( State_transformed_oracle[idx] ) ) > 1e-3 and idx < 5000) :
			print( idx, idx & (1<<8), real_part[idx], np.real( State_transformed_oracle[idx] ) )
		'''
	#print( imag_part[0:512] )	
	

	scipy.linalg.norm( real_part , 2)
	scipy.linalg.norm( real_part - np.real( State_transformed_oracle ) , 2)

