import groq.api as g
import groq.api.nn as nn
import groq.runner.tsp as tsp
from groq.api import instruction as inst
import numpy as np

import os
import shutil
from typing import List
import utils

try:
    import groq.runtime.driver as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')

import time
print("Python packages imported successfully")

# Note: MEM slices 16, 20, 24, and 28 on both the east and west hemispheres, and slice 38 on the west hemisphere are reserved for system use.


qbit_num = 8

# gate count stored in a single packed vector
gate_count = 1

# the number og qubits for which the gate operations need the permutor (need to reorganize the elements in a vector)
small_qbit_num_limit = 8

# the number of permutation maps including single and two-qubit reordering of the state vector  small_qbit_num_limit + (small_qbit_num_limit-1)*small_qbit_num_limit/2
required_permute_map_number = int(small_qbit_num_limit*(small_qbit_num_limit+1)/2)

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)


# putting the state onto the WEST hemisphere and the gate data on the EAST hemisphere

# memory layout to store the quantum state
layout_State_input_real 	="H1(W), S4(0-3)"
layout_State_input_imag 	="H1(W), S4(4-7)"

# memory layout to temporary store the permuted state vector
layout_state_vector_real_permuted = "H1(W), -1, S4(34-38)"
layout_state_vector_imag_permuted = "H1(W), -1, S4(39-43)"



# layout to gather map selecting the permutation map corresponding to the target qubit  -- the MT should be replaced by on chip determined gather map, currently it is uploaded from CPU
layout_state_permute_map_selector = f"H1(W), A{gate_count*3}(320-{320+3*gate_count-1}), S1(17)"
layout_state_1_selector		  = f"H1(E), A{gate_count*2}(0-{2*gate_count-1}), S1(19)"
layout_state_0_selector		  = f"H1(E), A{gate_count*2}(0-{2*gate_count-1}), S1(21)"

layout_gate_kernels_real_packed ="H1(E), S4(40-43)"
layout_gate_kernels_imag_packed	="H1(E), S4(40-43)"

# memory layouts for splitted gate operations (Each 320byte vector contains one gate kernel in the first 4 elements)
# addresses from 0 to gate_count                   on each hemispheres are occupied with the broadcasted 00 (real or imag) element of the gate kernels
# addresses from gate_count to 2*(gate_count)    on EAST hemispheres are occupied with the broadcasted 01 (real or imag) element of the gate kernels
# addresses from 2*gate_count to 3*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from 3*gate_count to 4*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from gate_count*4 to gate_count*5    on EAST hemispheres are occupied with the shifted (and still packed) gate kernels  -- this is temporary used.
layout_gate_kernels_real_EAST 		="H1(E), S4(8-11)"
layout_gate_kernels_imag_EAST		="H1(E), S4(12-15)"
layout_gate_kernels_real_EAST_copy 	="H1(E), S4(0-3)"
layout_gate_kernels_imag_EAST_copy	="H1(E), S4(4-7)"


# memory layout to store the vectors describing states |0> and |1> for different target qubits within a single 320 element vector
# addresses from 0 to small_qbit_num_limit-1       			are occupied by vectors describing states |0> and |1> for different target qubits (states |1> are labeled by bits 1)
# addresses from small_qbit_num_limit to 2*small_qbit_num_limit-1 	are occupied by vectors describing states |0> and |1> for different target qubits (states |0> are labeled by bits 1)
# addresses from 2*small_qbit_num_limit to 2*small_qbit_num_limit+3 	are occupied by distributor maps to used to broadcast elements 01,10,11 of the gate kernel
layout_states_1						= f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(17)"
layout_states_0						= f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(18)"
layout_states_00					= f"A{required_permute_map_number-small_qbit_num_limit}({small_qbit_num_limit+1}-{required_permute_map_number}), H1(E), S1(17)"
layout_states_01					= f"A{required_permute_map_number-small_qbit_num_limit}({small_qbit_num_limit+1}-{required_permute_map_number}), H1(E), S1(18)"
layout_states_10					= f"A{required_permute_map_number-small_qbit_num_limit}({required_permute_map_number+1}-{2*required_permute_map_number-small_qbit_num_limit}), H1(E), S1(17)"
layout_states_11					= f"A{required_permute_map_number-small_qbit_num_limit}({required_permute_map_number+1}-{2*required_permute_map_number-small_qbit_num_limit}), H1(E), S1(18)"

layout_distribute_map_tensor    	= "H1(E), S1(17)"
layout_distribute_map_tensor_state1	= "H1(E), S1(17)"
layout_distribute_map_tensor_state0 	= "H1(E), S1(18)"

layout_state_1_broadcasted			= "H1(E), -1, S4(25-29)"
layout_state_1_broadcasted_copy			= "H1(E), -1, S4(30-33)"
layout_state_0_broadcasted			= "H1(E), -1, S4(36-39)"
layout_state_0_broadcasted_copy			= "H1(E), -1, S4(40-43)"


# permute maps at -- used to reorder the state vector elements according to the given target qubit
layout_permute_maps = f"A{required_permute_map_number}(0-{required_permute_map_number-1}), H1(W), S1(43)"



# label to indicate the 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33 elements of the gate kernel
gate_kernel_element_labels = ["00", "01", "02", "03", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31", "32", "33"]

def compile() -> List[str]:
	"""Compiles a program package with 2 programs.

	Return: (List[str]): A list of IOP files.
	"""
	output_dir = "./build_iop"
	shutil.rmtree(output_dir, ignore_errors=True)
	pgm_pkg = g.ProgramPackage("QCsim_multi_program", output_dir)

	print( dir(pgm_pkg) )
	print( pgm_pkg.add_precompiled_program.__doc__ )
	print( pgm_pkg.create_program_context.__doc__ )

	# Defines a program to upload the inputs
	name = "upload"
	with pgm_pkg.create_program_context(name) as pgm1:

		# number of memory slices to store the wave function
		memory_slices = 1 << (qbit_num-small_qbit_num_limit)  

		State_input_real_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_real", layout=layout_State_input_real + f", A{memory_slices}(0-{memory_slices-1})")
		State_input_imag_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_imag", layout=layout_State_input_imag + f", A{memory_slices}(0-{memory_slices-1})")

		state_permute_map_selector_mt 	= g.input_tensor(shape=(3*gate_count,320,), dtype=g.uint8, name="state_permute_map_selector", layout=layout_state_permute_map_selector)
		state_1_selector_mt		= g.input_tensor(shape=(2*gate_count,320,), dtype=g.uint8, name="state_1_selector", layout=layout_state_1_selector)
		state_0_selector_mt		= g.input_tensor(shape=(2*gate_count,320,), dtype=g.uint8, name="state_0_selector", layout=layout_state_0_selector)
		#print(state_permute_map_selector_mt.shape)
		#print(dir(state_permute_map_selector_mt))

		# gate kernels packed over 320-element vectors
		gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
		gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


		State_input_real_mt.is_static 			= True
		State_input_imag_mt.is_static 			= True
		state_permute_map_selector_mt.is_static 	= True
		gate_kernels_real_packed_mt.is_static 		= True
		gate_kernels_imag_packed_mt.is_static 		= True
		state_1_selector_mt.is_static			= True
		state_0_selector_mt.is_static			= True


		# generate permutation maps to reorder the state vector elements for qubits less than small_qbit_num_limit
		permute_map = np.zeros( (required_permute_map_number,320,), dtype=np.uint32 )

		for target_qbit_loc in range(small_qbit_num_limit):

			permute_map_np        = np.zeros( (256,), dtype=np.uint32 )
			target_qbit_pair_diff = 1 << target_qbit_loc
			#print('target_qubit_pair_diff', target_qbit_pair_diff)
			
			for idx in range(256): # 256 = 2^8
				if (idx < matrix_size):
					permute_map_np[idx] = idx ^ target_qbit_pair_diff
				else:
					permute_map_np[idx] = idx
			
			#print("permute map:", permute_map_np)
			permute_map[target_qbit_loc,:] = inst.encode_permute_map(  permute_map_np.tolist() )
			#print(' ')
			#print(permute_map[target_qbit_loc,:])


		for target_qbit_loc1 in range(small_qbit_num_limit):
			for target_qbit_loc2 in range(target_qbit_loc1+1, small_qbit_num_limit):
				permute_map_np = np.zeros( (256,), dtype=np.uint32 )
				target_qbit_pair_diff = (1 << target_qbit_loc1) + (1 << target_qbit_loc2)
				for idx in range(256):
					if (idx < matrix_size):
						permute_map_np[idx] = idx ^ target_qbit_pair_diff
					else:
						permute_map_np[idx] = idx

				index                = int(small_qbit_num_limit-1+(13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2)
				permute_map[index,:] =  inst.encode_permute_map( permute_map_np.tolist() )
				#print('target_qubit_pair_diff', target_qbit_pair_diff)


		print("permute maps")
		permute_maps_mt = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps )
		permute_maps_mt.is_static = True
		print('permute maps shape: ', permute_maps_mt.shape)
		#print(dir(permute_maps_mt))


		# generate vectors to indicate which elements in a 320 vector correspond to state |1> of target qubit less than small_qbit_num_limit 
		# if target_qubit_loc at the idx-th element is |1> then all the 8 bits of the idx-th element in state_1 is set to 1, thus state_1[idx] = 255
		# if target_qubit_loc at the idx-th element is |0> then all the 8 bits of the idx-th element in state_0 is set to 1, thus state_0[idx] = 255
		states_1_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
		states_0_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )

		# The following 4 arary are analogous to states_1 and states_0, but with 2 qubits
		states_00_np = np.zeros( (required_permute_map_number-small_qbit_num_limit, 320,), dtype=np.uint8 )
		states_01_np = np.zeros( (required_permute_map_number-small_qbit_num_limit, 320,), dtype=np.uint8 )
		states_11_np = np.zeros( (required_permute_map_number-small_qbit_num_limit, 320,), dtype=np.uint8 )
		states_10_np = np.zeros( (required_permute_map_number-small_qbit_num_limit, 320,), dtype=np.uint8 )


		# fill in state_1 and state_0 arrays
		for target_qbit_loc in range(small_qbit_num_limit):
			target_qbit_pair_diff = 1 << target_qbit_loc
			for idx in range(256): # 256 = 2^8
				if (idx & target_qbit_pair_diff > 0):
					states_1_np[target_qbit_loc, idx] = 255
					states_0_np[target_qbit_loc, idx] = 0
				else:
					states_1_np[target_qbit_loc, idx] = 0
					states_0_np[target_qbit_loc, idx] = 255
			#print( "states |1> at target qubit ", target_qbit_loc, " is:" )
			#print( states_1_np[target_qbit_loc, 0:256] )


		# fill in state_00, state_01, state_11 and state_10 arrays
		for target_qbit_loc1 in range(small_qbit_num_limit):
			for target_qbit_loc2 in range(target_qbit_loc1+1, small_qbit_num_limit):
				states_00_np[int((13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2-1)] = states_0_np[target_qbit_loc1] & states_0_np[target_qbit_loc2]
				states_01_np[int((13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2-1)] = states_0_np[target_qbit_loc1] & states_1_np[target_qbit_loc2]
				states_11_np[int((13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2-1)] = states_1_np[target_qbit_loc1] & states_1_np[target_qbit_loc2]
				states_10_np[int((13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2-1)] = states_1_np[target_qbit_loc1] & states_0_np[target_qbit_loc2]
				#print( "state |00> at target qbit ", target_qbit_loc1, target_qbit_loc2, " is:" )
				#print( states_00_np[int((13-target_qbit_loc1)*target_qbit_loc1/2+target_qbit_loc2-1)] )


		states_1_mt = g.from_data( np.asarray(states_1_np, dtype=np.uint8), layout=layout_states_1 )
		states_1_mt.is_static = True
		print("memory tensor states_1 cerated")

		states_0_mt = g.from_data( np.asarray(states_0_np, dtype=np.uint8), layout=layout_states_0 )
		states_0_mt.is_static = True
		print("memory tensor states_0 cerated")

		states_00_mt = g.from_data( np.asarray(states_00_np, dtype=np.uint8), layout=layout_states_00 )
		states_00_mt.is_static = True
		print("memory tensor states_00 cerated")

		states_01_mt = g.from_data( np.asarray(states_01_np, dtype=np.uint8), layout=layout_states_01 )
		states_01_mt.is_static = True
		print("memory tensor states_01 cerated")

		states_10_mt = g.from_data( np.asarray(states_10_np, dtype=np.uint8), layout=layout_states_10 )
		states_10_mt.is_static = True
		print("memory tensor states_10 cerated")

		states_11_mt = g.from_data( np.asarray(states_11_np, dtype=np.uint8), layout=layout_states_11 )
		states_11_mt.is_static = True
		print("memory tensor states_11 cerated")

		pgm_pkg.compile_program_context(pgm1)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)

		print(' ')
		print(' ')
		print(' ')


	#############################################################################################################x
	# program to split the gate kernels and other metada between the EAST-WAST hemispheres 
	name = "split_gates"
	with pgm_pkg.create_program_context(name) as pgm2:

		# create reusable permutor requests
		permutor_requests = []
		permutor_requests.append(  g.tensor.create_permutor_request( [0], 1 ) ) # WEST
		permutor_requests.append(  g.tensor.create_permutor_request( [1], 1 ) ) # EAST
		
		# create reusable distributor requests
		distributor_requests = []
		distributor_requests.append(  g.tensor.create_distributor_request( [0], 1 ) ) # WEST
		distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST		

		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		gate_kernels_real_packed_shared = g.shared_memory_tensor(gate_kernels_real_packed_mt, name="gate_kernels_real_packed") ## 80 gates all packed onto a single vector
		gate_kernels_imag_packed_shared = g.shared_memory_tensor(gate_kernels_imag_packed_mt, name="gate_kernels_imag_packed") ## 80 gates all packed onto a single vector

		# concat real and imag parts to pipeline them at once through a shifter
		gate_kernels_packed_mt = g.concat( [gate_kernels_real_packed_shared, gate_kernels_imag_packed_shared], dim=0 )
		gate_kernels_packed_st = gate_kernels_packed_mt.read( streams=g.SG4[0], time=0 )
		#print( gate_kernels_packed_st.physical_shape )
		#print( gate_kernels_packed_st.shape )

		# split the tensor between hemispheres WEST and EAST (each (2k+1)-th gate kernel goes to the western hemispheres, while 2k-th gates are assigned to the eastern hemisphere
		# the gate kernel elements to be used later are the first 4 elements in each vector
		gate_kernels_imag_list = []
		gate_kernels_real_list = []

		for gate_idx in range(gate_count):

			gate_kernels_packed_list = g.split_inner_splits( gate_kernels_packed_st )
			print(gate_kernels_packed_st.physical_shape)
			gate_kernels_real_packed_st = gate_kernels_packed_list[0]
			gate_kernels_imag_packed_st = gate_kernels_packed_list[1]

			layout_real = layout_gate_kernels_real_EAST
			layout_imag = layout_gate_kernels_imag_EAST

			gate_kernels_imag_mt = gate_kernels_imag_packed_st.write(name=f"gate_kernel_imag_{gate_idx}", layout=layout_imag + f", A1({gate_count*4 + gate_idx})", program_output=True)
			gate_kernels_real_mt = gate_kernels_real_packed_st.write(name=f"gate_kernel_real_{gate_idx}", layout=layout_real + f", A1({gate_count*4 + gate_idx})", program_output=True)

			# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels
			if ( gate_idx > 0 ):
				g.add_mem_constraints(gate_kernels_imag_list, [gate_kernels_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
				g.add_mem_constraints(gate_kernels_real_list, [gate_kernels_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    

			gate_kernels_imag_list.append( gate_kernels_imag_mt )
			gate_kernels_real_list.append( gate_kernels_real_mt )
		

			if gate_idx < gate_count-1 :
				permutor_request = permutor_requests[gate_idx % 2]
				shift = 4 #?????????????????????????????? should be 16
				gate_kernels_packed_st = g.shift(gate_kernels_packed_st, 
								shift, 
								permutor_id=permutor_request, 
								shift_src=[inst.NEW_SRC] * 2, 
								dispatch_set=inst.DispatchSet.SET_1, 
								input_streams=[g.SG4[0]], 
								output_streams=[g.SG4[0]])



		# generate gate kernels to be applied on state |0> and resulting to alternating states |0> and |1> according to the periodicity of the current target qubit

		# create distributor maps for the process to broadcast elements 00,01,10,11 of the gate kernel onto 320 element vectors
		distribute_map_tensor_list = []
		distributor_00_np = np.zeros( (320,), dtype=np.uint8 ) # distribut the 0th element over the 320 element vector
		distribute_map_tensor_mt_00 = g.from_data( np.asarray(distributor_00_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_00)

		distributor_01_np = np.ones( (320,), dtype=np.uint8 )
		distribute_map_tensor_mt_01 = g.from_data( np.asarray(distributor_01_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+1})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_01)

		distributor_02_np = np.ones( (320,), dtype=np.uint8 )*2 # distribut the 2nd element over the 320 element vector
		distribute_map_tensor_mt_02 = g.from_data( np.asarray(distributor_02_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+2})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_02)

		distributor_03_np = np.ones( (320,), dtype=np.uint8 )*3 # distribut the 3rd element over the 320 element vector
		distribute_map_tensor_mt_03 = g.from_data( np.asarray(distributor_03_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+3})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_03)

		distributor_10_np = np.ones( (320,), dtype=np.uint8 )*4 # distribut the 4th element over the 320 element vector
		distribute_map_tensor_mt_10 = g.from_data( np.asarray(distributor_10_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+4})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_10)

		distributor_11_np = np.ones( (320,), dtype=np.uint8 )*5 # distribut the 5th element over the 320 element vector
		distribute_map_tensor_mt_11 = g.from_data( np.asarray(distributor_11_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+5})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_11)

		distributor_12_np = np.ones( (320,), dtype=np.uint8 )*6 # distribut the 6th element over the 320 element vector
		distribute_map_tensor_mt_12 = g.from_data( np.asarray(distributor_12_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+6})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_12)

		distributor_13_np = np.ones( (320,), dtype=np.uint8 )*7 # distribut the 7th element over the 320 element vector
		distribute_map_tensor_mt_13 = g.from_data( np.asarray(distributor_13_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+7})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_13)

		distributor_20_np = np.ones( (320,), dtype=np.uint8 )*8 # distribut the 8th element over the 320 element vector
		distribute_map_tensor_mt_20 = g.from_data( np.asarray(distributor_20_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+8})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_20)

		distributor_21_np = np.ones( (320,), dtype=np.uint8 )*9 # distribut the 9th element over the 320 element vector
		distribute_map_tensor_mt_21 = g.from_data( np.asarray(distributor_21_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+9})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_21)

		distributor_22_np = np.ones( (320,), dtype=np.uint8 )*10 # distribut the 10th element over the 320 element vector
		distribute_map_tensor_mt_22 = g.from_data( np.asarray(distributor_22_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+10})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_22)
		
		distributor_23_np = np.ones( (320,), dtype=np.uint8 )*11 # distribut the 11th element over the 320 element vector
		distribute_map_tensor_mt_23 = g.from_data( np.asarray(distributor_23_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+11})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_23)

		distributor_30_np = np.ones( (320,), dtype=np.uint8 )*12 # distribut the 12th element over the 320 element vector
		distribute_map_tensor_mt_30 = g.from_data( np.asarray(distributor_30_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+12})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_30)

		distributor_31_np = np.ones( (320,), dtype=np.uint8 )*13 # distribut the 13th element over the 320 element vector
		distribute_map_tensor_mt_31 = g.from_data( np.asarray(distributor_31_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+13})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_31)

		distributor_32_np = np.ones( (320,), dtype=np.uint8 )*14 # distribut the 14th element over the 320 element vector
		distribute_map_tensor_mt_32 = g.from_data( np.asarray(distributor_32_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+14})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_32)

		distributor_33_np = np.ones( (320,), dtype=np.uint8 )*15 # distribut the 15th element over the 320 element vector
		distribute_map_tensor_mt_33 = g.from_data( np.asarray(distributor_33_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*required_permute_map_number-small_qbit_num_limit+15})" )
		distribute_map_tensor_list.append(distribute_map_tensor_mt_33)

		print("*********** disribute maps done *************")
		print(' ')
		print(' ')
		print(' ')

		for state_idx in range(len(distribute_map_tensor_list)):
				 g.add_mem_constraints(distribute_map_tensor_list[:state_idx], [distribute_map_tensor_list[state_idx]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     

		gate_kernels_broadcasted_imag_list = []
		gate_kernels_broadcasted_real_list = []
		gate_kernels_broadcasted_imag_copy_list = []
		gate_kernels_broadcasted_real_copy_list = []
		loop_time_long_cycle = 51 # comming from the iterations of the previous cycles (the loops are overlapping in execution)
		loop_time_short_cycle = 31

		# element-wise broadcasted gate elements
		gate_kernels_broadcasted_real_dict = {}
		gate_kernels_broadcasted_imag_dict = {}
		gate_kernels_broadcasted_real_copy_dict = {}
		gate_kernels_broadcasted_imag_copy_dict = {}		

		for gate_idx in range(gate_count):

			gate_kernel_real_mt = gate_kernels_real_list[gate_idx]
			gate_kernel_imag_mt = gate_kernels_imag_list[gate_idx]

			for idx in range(len(gate_kernel_element_labels) ): # iterates over the labels of the gate kernel elements
				label = gate_kernel_element_labels[idx]

				# concat real and imag parts to pipeline them at once through a broadcaster
				gate_kernel_real_st = gate_kernel_real_mt.read( streams=g.SG4[1], time=12 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle + 60) 
				gate_kernel_imag_st = gate_kernel_imag_mt.read( streams=g.SG4[1], time=14 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle + 60 )		
				gate_kernel_st      = g.concat_inner_splits( [gate_kernel_real_st, gate_kernel_imag_st] )


				distributor_request = distributor_requests[(gate_idx+1) % 2]

				gate_kernel_broadcasted_st = g.distribute_8( gate_kernel_st, 
				        distribute_map_tensor_list[idx],
				        distributor_req=distributor_request,
				        map_stream_req=g.SG1[12],
				        bypass8=0b00001111,
				)

				permutor_request = permutor_requests[(gate_idx+1) % 2]
				gate_kernel_broadcasted_st = g.broadcast_lane_0( gate_kernel_broadcasted_st, 
									permutor_request,
									old_bitmap=[inst.NEW_SRC] * 4, 
									mask_bitmap=0b0000,
									input_streams=[g.SG4[3]], 
									output_streams=[g.SG4[2]]
				)

				# split into real ands imaginary parts	
				gate_kernel_broadcasted_list    = g.split_inner_splits( gate_kernel_broadcasted_st )
				gate_kernel_broadcasted_real_st = gate_kernel_broadcasted_list[0]
				gate_kernel_broadcasted_imag_st = gate_kernel_broadcasted_list[1]

				######################## save broadcasted elements into memory and add memory exclusion rules #############################


				layout_real = layout_gate_kernels_real_EAST
				layout_imag = layout_gate_kernels_imag_EAST
				layout_real_copy = layout_gate_kernels_real_EAST_copy
				layout_imag_copy = layout_gate_kernels_imag_EAST_copy

				address_offset = idx*int(gate_count)

				layout_real = layout_real + f", A1({address_offset + gate_idx})"
				layout_imag = layout_imag + f", A1({address_offset + gate_idx})"
				gate_kernels_broadcasted_real_mt = gate_kernel_broadcasted_real_st.write(name=f"gate_kernel_broadcasted_real_{gate_idx}_"+label, layout=layout_real)#, program_output=True)
				gate_kernels_broadcasted_imag_mt = gate_kernel_broadcasted_imag_st.write(name=f"gate_kernel_broadcasted_imag_{gate_idx}_"+label, layout=layout_imag)#, program_output=True)
				
				# copies of the gate elements to support read concurrency in forthcoming processeses
				layout_real = layout_real_copy + f", A1({address_offset + gate_idx})"
				layout_imag = layout_imag_copy + f", A1({address_offset + gate_idx})"
				gate_kernels_broadcasted_real_copy_mt = gate_kernel_broadcasted_real_st.write(name=f"gate_kernel_broadcasted_real_{gate_idx}_"+label, layout=layout_real)#, program_output=True)
				gate_kernels_broadcasted_imag_copy_mt = gate_kernel_broadcasted_imag_st.write(name=f"gate_kernel_broadcasted_imag_{gate_idx}_"+label, layout=layout_imag)#, program_output=True)

				gate_kernels_broadcasted_real_mt.is_static = True
				gate_kernels_broadcasted_imag_mt.is_static = True
				gate_kernels_broadcasted_real_copy_mt.is_static = True
				gate_kernels_broadcasted_imag_copy_mt.is_static = True				


				# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels  
				if ( len(gate_kernels_broadcasted_real_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
				if ( len(gate_kernels_broadcasted_imag_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)   
				if ( len(gate_kernels_broadcasted_real_copy_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_real_copy_list, [gate_kernels_broadcasted_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
				if ( len(gate_kernels_broadcasted_imag_copy_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_imag_copy_list, [gate_kernels_broadcasted_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)   					
    
    
				g.add_mem_constraints(gate_kernels_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    
				g.add_mem_constraints(gate_kernels_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 


				gate_kernels_broadcasted_real_list.append( gate_kernels_broadcasted_real_mt )
				gate_kernels_broadcasted_imag_list.append( gate_kernels_broadcasted_imag_mt )
				gate_kernels_broadcasted_real_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_real_mt
				gate_kernels_broadcasted_imag_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_imag_mt


				gate_kernels_broadcasted_real_copy_list.append( gate_kernels_broadcasted_real_copy_mt )
				gate_kernels_broadcasted_imag_copy_list.append( gate_kernels_broadcasted_imag_copy_mt )
				gate_kernels_broadcasted_real_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_real_copy_mt
				gate_kernels_broadcasted_imag_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_imag_copy_mt


		pgm_pkg.compile_program_context(pgm2)

		g.write_visualizer_data("QCsim_"+name)	

		print( '*********** Scope QCsim_'+ name + ' finished ***************' )	
		print(' ')
		print(' ')
		print(' ')





	#############################################################################################################x		
	# program to perform gate operation in a target qubit less than small_qbit_num_limit
	name = "gate"
	with pgm_pkg.create_program_context(name) as pgm3:

		#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)

		# create reusable distributor requests
		distributor_requests = []
		distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
		distributor_requests.append(  g.tensor.create_distributor_request( [7], 1 ) ) # EAST	
		
		# create reusable permutor requests
		permutor_requests = []
		permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
		permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )

		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		State_input_real_shared = g.shared_memory_tensor(State_input_real_mt, name="State_input_real_shared")
		State_input_imag_shared = g.shared_memory_tensor(State_input_imag_mt, name="State_input_imag_shared")

		state_permute_map_selector_shared 	= g.shared_memory_tensor(state_permute_map_selector_mt, name="state_permute_map_selector_shared")
		state_1_selector_shared 		= g.shared_memory_tensor(state_1_selector_mt, name="state_1_selector_shared")
		state_0_selector_shared 		= g.shared_memory_tensor(state_0_selector_mt, name="state_0_selector_shared")
		
		#state_permute_map_selector_st = state_permute_map_selector_shared.read(streams=g.SG1_W[24])
		#print(state_permute_map_selector_shared.physical_shape)
		#print(dir(state_permute_map_selector_shared.physical_shape))
		#print(state_permute_map_selector_shared.shape)

		#state_permute_map_selector_list = g.split_inner_splits(state_permute_map_selector_st)
		state_permute_map_selector_list = g.split_vectors( input=state_permute_map_selector_shared, splits=gate_count*3*[1] )
		#print(state_permute_map_selector_list)
		#print(state_permute_map_selector_list[1].shape)
		permute_maps_mt_shared = g.shared_memory_tensor(permute_maps_mt, name=f"permute_maps_shared")

		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		states_1_shared = g.shared_memory_tensor(states_1_mt, name="states_1_shared")
		states_0_shared = g.shared_memory_tensor(states_0_mt, name="states_0_shared")
		#target_qbit = 3




		# Creates a shared tensors to reuse the memory allocation made by program pgm2

		# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
		gate_kernels_broadcasted_real_shared_dict = {}
		gate_kernels_broadcasted_imag_shared_dict = {}
		gate_kernels_broadcasted_real_shared_copy_dict = {}
		gate_kernels_broadcasted_imag_shared_copy_dict = {}
		for gate_idx in range( gate_count ):
			for label in gate_kernel_element_labels:

				key = f"{gate_idx}_"+label
				gate_kernel_broadcasted_real_mt = gate_kernels_broadcasted_real_dict[key]
				gate_kernel_broadcasted_imag_mt = gate_kernels_broadcasted_imag_dict[key]

				gate_kernels_broadcasted_real_shared = g.shared_memory_tensor(gate_kernel_broadcasted_real_mt, name=gate_kernel_broadcasted_real_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared = g.shared_memory_tensor(gate_kernel_broadcasted_imag_mt, name=gate_kernel_broadcasted_imag_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_dict[key] = gate_kernels_broadcasted_real_shared
				gate_kernels_broadcasted_imag_shared_dict[key] = gate_kernels_broadcasted_imag_shared
				
				gate_kernel_broadcasted_real_copy_mt = gate_kernels_broadcasted_real_copy_dict[key]
				gate_kernel_broadcasted_imag_copy_mt = gate_kernels_broadcasted_imag_copy_dict[key]

				gate_kernels_broadcasted_real_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_real_copy_mt, name=gate_kernel_broadcasted_real_copy_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_imag_copy_mt, name=gate_kernel_broadcasted_imag_copy_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_copy_dict[key] = gate_kernels_broadcasted_real_shared_copy
				gate_kernels_broadcasted_imag_shared_copy_dict[key] = gate_kernels_broadcasted_imag_shared_copy				



		# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
		Psi_transformed_real_mt = State_input_real_shared
		Psi_transformed_imag_mt = State_input_imag_shared

		# distribute maps to broadcast the gate kernel elements over a vector
		distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
		for idx in range(320):
			distribute_map_np[idx] = idx % 16

		distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state1 + f", A1({2*required_permute_map_number-small_qbit_num_limit+16})" )
		distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state0 + f", A1({2*required_permute_map_number-small_qbit_num_limit+16})" )
		g.add_mem_constraints([states_1_shared], [distribute_map_tensor_state1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
		g.add_mem_constraints([states_0_shared], [distribute_map_tensor_state0_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
		pre = []
		tm = [0, 118]

		gate_diag_real_list = []
		gate_diag_imag_list = []
		gate_offdiag_real_list = []
		gate_offdiag_imag_list = []
		for gate_idx in range( gate_count ):

		
			with g.ResourceScope(name=f"prepare_permuted_states_scope_{gate_idx}", is_buffered=True, time=tm[gate_idx]) as prepare_permuted_states_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
				#target_qbit = 0
				'''
				permute_map_mt = self.permute_maps_mt_list[target_qbit]
				state_real_mt_8 = g.reinterpret(self.state_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				state_real_st_8 = g.permute_inner(state_real_mt_8, permute_map_mt, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )
				'''

				#print( g.mem_gather.__doc__ )
				#print( permute_maps_mt_shared.physical_shape )
				#print( state_permute_map_selector_shared.physical_shape )
				#state_permute_map_selector_st = state_permute_map_selector_shared.read(streams=[g.SG1_W[24]])
				#print(state_permute_map_selector_list[0].shape)


				# the state vectors to be permuted are constructed form 4-byte real, and 4-byte imag elements organized into
				# 8-byte long streams. Therefore, the same permutation map should be used for 8 clock cycles.
				# Then a next permutation map should be used over the next 8 clock cycles, etc.
				# In total 3 different permuattions should be created
				state_permute_map_selectors = g.concat( 8*[ state_permute_map_selector_list[gate_idx*3] ] + \
									8*[ state_permute_map_selector_list[gate_idx*3+1] ] + \
									8*[ state_permute_map_selector_list[gate_idx*3+2] ], dim=0 )



				state_permute_map_selectors_st = state_permute_map_selectors.read(streams=g.SG1_W[24])
				permute_maps_st                = g.mem_gather(permute_maps_mt_shared, state_permute_map_selectors_st, output_streams=[g.SG1_W[24]])
				#print(permute_maps_st.physical_shape)

				state_real_mt_8 = g.reinterpret(Psi_transformed_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				state_imag_mt_8 = g.reinterpret(Psi_transformed_imag_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

				# real and imaginary parts of the state concatenated into a unified tensor
				# the real/imag input states should be repated 3 times to retrieve the 3 permutes
				state_mt_8 = g.concat( 3*[state_real_mt_8, state_imag_mt_8], dim=0 )

				state_st_8 = state_mt_8.read( streams=g.SG1[0] )

			
				#state_mt_8 = g.reinterpret(state_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				
				permuted_states_st_8 = g.permute_inner(state_st_8, permute_maps_st, permutor_req=permutor_requests[gate_idx], input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0)

				print("states:", permuted_states_st_8.physical_shape)
				print("states:", permuted_states_st_8.layout)

				#permuted_state_imag_mt = permuted_states_st_8.write(name=f"permuted_result_TEST")

				
				# the permuted byte vectors are reshaped to form real, imag, real, imag, real, imag 4byte floats
				permuted_states_st_8 = g.reshape( permuted_states_st_8, [6,4,256])

				# split unified tensor into real,imag,real,imag,real,imag parts
				permuted_states_st_8_list = g.split( permuted_states_st_8, num_splits=6, dim=0 )
				#print( permuted_state_st_8_list[0].physical_shape )

				# reinterpret the permuted byte vectors as float32 and store them into teh memory
				permuted_states_real_mt_list = []
				permuted_states_imag_mt_list = []


				for permuted_idx in range(3):
					permuted_state_real_st_8 = g.reshape(permuted_states_st_8_list[2*permuted_idx], [4,256] )
					permuted_state_imag_st_8 = g.reshape(permuted_states_st_8_list[2*permuted_idx+1], [4,256] )
					#state_real_st = g.reinterpret(state_st_8_list[0], g.float32 )
				
					permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
					permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

					permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real_{permuted_idx}", layout=layout_state_vector_real_permuted, program_output=True)
					permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag_{permuted_idx}", layout=layout_state_vector_imag_permuted, program_output=True)


					permuted_states_real_mt_list.append( permuted_state_real_mt )
					permuted_states_imag_mt_list.append( permuted_state_imag_mt )

				
					if permuted_idx > 0:
						g.add_mem_constraints(permuted_states_real_mt_list, [permuted_state_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
						g.add_mem_constraints(permuted_states_imag_mt_list, [permuted_state_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)   



				print( '*********** Scope QCsim_'+ name + ' finished ***************' )	
				print(' ')
				print(' ')
				print(' ')
				
			#################################################################################
			
			
			# component to perform the gate operation
			with g.ResourceScope(name=f"prepare_gate_kernel_scope_{gate_idx}", is_buffered=True, time=tm[gate_idx]) as prepare_gate_kernel_scope :

	
				# combine 00 and 10 elements of the gate kernel to be used in the transformation
				#states_1_mt_list = g.split( states_1_shared, 1, 0)
				#states_0_mt_list = g.split( states_0_shared, 1, 0)

				#state_1_mt = states_1_mt_list[target_qbit] #TODO: do with gather map
				#state_0_mt = states_0_mt_list[target_qbit] #TODO: do with gather map

				# goal of the scope: make int32 from int8 inputs of state_1 and state_0 vectors
				with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=0) as distribute_state_indices_scope :

					#state_1_st = state_1_mt.read( streams=g.SG1[0], time=0 )
					#state_0_st = state_0_mt.read( streams=g.SG1[24], time=0 )

					state_1_selector_st = state_1_selector_shared.read(streams=[g.SG1[0]])
					state_1_st = g.mem_gather(states_1_shared, state_1_selector_st, output_streams=[g.SG1[0]], time=0)

					state_0_selector_st = state_0_selector_shared.read(streams=[g.SG1[24]])
					state_0_st = g.mem_gather(states_0_shared, state_0_selector_st, output_streams=[g.SG1[24]], time=0)
						


					# duplicate 8 bits of state_1_st over 32 bits to have int32
					distributor_request = distributor_requests[0]
					state_1_broadcasted_st = g.distribute_lowest( state_1_st, 
				        	distribute_map_tensor_state1_mt,
				        	distributor_req=distributor_request,
				       		map_stream_req=g.SG1[12],
				        	bypass8=0b00001111,
					)

					# just turn over the stream
					state_1_broadcasted_st = g.transpose_null(
						state_1_broadcasted_st,
						transposer_req=2,
						stream_order=[4,5,6,7],
					)

			
					# duplicate 8 bits of state_0_st over 32 bits to have int32
					distributor_request = distributor_requests[1]
					state_0_broadcasted_st = g.distribute_lowest( state_0_st, 
				        	distribute_map_tensor_state0_mt,
				        	distributor_req=distributor_request,
				        	map_stream_req=g.SG1[31],
				        	bypass8=0b00001111,
					)

					# just turn over the stream
					state_0_broadcasted_st = g.transpose_null(
						state_0_broadcasted_st,
						transposer_req=3,
						stream_order=[12,13,14,15],
					)

			
					state_1_broadcasted_st = g.reinterpret( state_1_broadcasted_st, g.uint32 )
					state_1_broadcasted_mt = state_1_broadcasted_st.write(name=f"state_1_broadcasted_{gate_idx}", layout=layout_state_1_broadcasted, program_output=False)
					state_1_broadcasted_mt_copy = state_1_broadcasted_st.write(name=f"state_1_broadcasted_copy_{gate_idx}", layout=layout_state_1_broadcasted_copy, program_output=False) #TODO layout

					state_0_broadcasted_st = g.reinterpret( state_0_broadcasted_st, g.uint32 )
					state_0_broadcasted_mt = state_0_broadcasted_st.write(name=f"state_0_broadcasted_{gate_idx}", layout=layout_state_0_broadcasted, program_output=False)
					state_0_broadcasted_mt_copy = state_0_broadcasted_st.write(name=f"state_0_broadcasted_copy_{gate_idx}", layout=layout_state_0_broadcasted_copy, program_output=False) #TODO layout
#layout_state_0_broadcasted			= "H1(E), -1, S4(19-23)"
					print( state_0_broadcasted_mt.physical_shape )
					print( state_0_broadcasted_mt.shape )


				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
			
				# the individual rows of the 4x4 gate kernel detemine the output elements where qubits states are 00, 01, 10, 11 
				with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope_{gate_idx}", is_buffered=True, predecessors=[distribute_state_indices_scope], time=None) as prepare_fileterd_gate_elements_scope :

					'''
					state_1_broadcasted_mt_list = g.split( state_1_broadcasted_mt, num_splits=2, dim=0 ) 
					state_1_broadcasted_qubit0_mt = state_1_broadcasted_mt_list[0]
					state_1_broadcasted_qubit1_mt = state_1_broadcasted_mt_list[1]

					state_1_broadcasted_mt_copy_list = g.split( state_1_broadcasted_mt_copy, num_splits=2, dim=0 ) 
					state_1_broadcasted_qubit0_mt_copy = state_1_broadcasted_mt_copy_list[0]
					state_1_broadcasted_qubit1_mt_copy = state_1_broadcasted_mt_copy_list[1]
					'''
					state_0_broadcasted_mt_list = g.split( state_0_broadcasted_mt, num_splits=2, dim=0 ) 
					state_0_broadcasted_qubit0_mt = state_0_broadcasted_mt_list[0]
					state_0_broadcasted_qubit1_mt = state_0_broadcasted_mt_list[1]

					print( state_0_broadcasted_qubit0_mt.physical_shape )
					print( state_0_broadcasted_qubit0_mt.shape )


					state_0_broadcasted_mt_copy_list = g.split( state_0_broadcasted_mt_copy, num_splits=2, dim=0 ) 
					state_0_broadcasted_qubit0_mt_copy = state_0_broadcasted_mt_copy_list[0]
					state_0_broadcasted_qubit1_mt_copy = state_0_broadcasted_mt_copy_list[1]

					#state_1_broadcasted_st = state_1_broadcasted_qubit0_mt.read( streams=g.SG4[1] )
					state_0_broadcasted_qubit0_st = state_0_broadcasted_qubit0_mt.read( streams=g.SG4[7] )
					state_0_broadcasted_qubit1_st = state_0_broadcasted_qubit1_mt_copy.read( streams=g.SG4[6])#, time=20 )

					state_00_broadcasted_st = g.bitwise_and( state_0_broadcasted_qubit0_st, state_0_broadcasted_qubit1_st, alus=[9], output_streams=g.SG4[7] )
					
					#state_00_broadcasted_mt = state_00_broadcasted_st.write( name="jjjjjjjjjjjjjjJ", layout="H1(W), -1, S4" )
					
					
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[6], time=0 )
					gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[6], time=0 )
					gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )
					
					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_00_broadcasted_st, alus=[8], output_streams=g.SG4[4] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )
					
					gate_00_mt = gate_00_st.write( name="gate_00", program_output=True  )
					

	
					'''
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[0], output_streams=g.SG4[3] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

			
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=[5], output_streams=g.SG4[3])
			
			
					layout_real = layout_gate_kernels_real_EAST
					layout_imag = layout_gate_kernels_imag_EAST

					address_offset = 4*gate_count + gate_idx

					layout_real_diag = layout_real + f", A1({address_offset})"
					layout_imag_diag = layout_imag + f", A1({address_offset})"			
					print("layout: ", layout_imag_diag)
				
					gate_diag_st_list	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_st_list[0].write(name=f"gate_diag_real_{gate_idx}", layout=layout_real_diag, program_output=True)
					gate_diag_imag_mt 	= gate_diag_st_list[1].write(name=f"gate_diag_imag_{gate_idx}", layout=layout_imag_diag, program_output=True)
					#if gate_idx > 0:
					#	g.add_mem_constraints(gate_diag_real_list, [gate_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					#	g.add_mem_constraints(gate_diag_imag_list, [gate_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

					#gate_diag_real_list.append(gate_diag_real_mt)
					#gate_diag_imag_list.append(gate_diag_imag_mt)

					#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
					#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)



			#with g.ResourceScope(name="prepare_gate_elements_output_state_1_scope", is_buffered=True, predecessors=[prepare_gate_elements_output_state_0_scope], time=None) as prepare_gate_elements_output_state_1_scope :

					state_1_broadcasted_st = state_1_broadcasted_mt.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_mt.read( streams=g.SG4[7] )
			
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_01_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[10], output_streams=g.SG4[4] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )

					# real and imaginary parts of the state concatenated into a unified tensor
					gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[0], time=7 )
					gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[0], time=7 )
					gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[2], output_streams=g.SG4[3] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
				
					# combine the filtered gate element vectors
					gate_offdiag_st = g.add(gate_01_st, gate_10_st, alus=[7], output_streams=g.SG4[3])
			
			
					layout_real_copy = layout_gate_kernels_real_EAST_copy
					layout_imag_copy = layout_gate_kernels_imag_EAST_copy
	
					address_offset = 4*gate_count+gate_idx

					layout_real_offdiag = layout_real_copy + f", A1({address_offset})"
					layout_imag_offdiag = layout_imag_copy + f", A1({address_offset})"			
				
					gate_offdiag_st_list	= g.split( gate_offdiag_st, num_splits=2, dim=0 )
					gate_offdiag_real_mt 	= gate_offdiag_st_list[0].write(name=f"gate_offdiag_real_{gate_idx}", layout=layout_real_offdiag, program_output=True)
					gate_offdiag_imag_mt 	= gate_offdiag_st_list[1].write(name=f"gate_offdiag_imag_{gate_idx}", layout=layout_imag_offdiag, program_output=True)

					if gate_idx > 0:
						g.add_mem_constraints(gate_offdiag_real_list, [gate_offdiag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
						g.add_mem_constraints(gate_offdiag_imag_list, [gate_offdiag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					gate_offdiag_real_list.append(gate_offdiag_real_mt)
					gate_offdiag_imag_list.append(gate_offdiag_imag_mt)
					#gate_output_state_0_real_copy_mt = gate_output_state_0_st_list[0].write(name=f"gate_output_state_0_real_copy", layout=layout_real_copy, program_output=True)
					#gate_output_state_0_imag_copy_mt = gate_output_state_0_st_list[1].write(name=f"gate_output_state_0_imag_copy", layout=layout_imag_copy, program_output=True)
					#gate_output_state_0_copy_mt = gate_output_state_0_st.write(name=f"gate_output_state_copy_0", layout=layout_real_copy)			
				

					#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
					#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)
					'''
				
			
			'''
			#################################################################################
			with g.ResourceScope(name=f"apply_gate_scope_{gate_idx}", is_buffered=True, predecessors=[prepare_permuted_states_scope], time=None) as apply_gate_scope :

				# calculate Psi_real * gate_diag_real
				state_real_st 			= Psi_transformed_real_mt.read( streams=[g.SG4[0]] )
				gate_diag_real_st 		= gate_diag_real_mt.read( streams=[g.SG4[1]] )
				state_real__gate_diag_real_st	= g.mul( state_real_st, gate_diag_real_st, alus=[12], output_streams=g.SG4_E[0], time=0 )

				# calculate Psi_imag * gate_diag_imag
				state_imag_st 			= Psi_transformed_imag_mt.read( streams=[g.SG4[3]] )
				gate_diag_imag_st 		= gate_diag_imag_mt.read( streams=[g.SG4[2]] )
				state_imag__gate_diag_imag_st	= g.mul( state_imag_st, gate_diag_imag_st, alus=[4], output_streams=g.SG4_E[2] )			

				# calculate Psi_real * gate_diag_real - Psi_imag * gate_diag_imag
				state__gate_diag_real_st	= g.sub( state_real__gate_diag_real_st, state_imag__gate_diag_imag_st, alus=[2], output_streams=g.SG4_E[2] )
				#res_mt = state__gate_diag_real_st.write(name=f"test", layout="H1(E), -1, S4", program_output=True)




				# calculate Psi_imag * gate_diag_real
				state_imag__gate_diag_real_st	= g.mul( state_imag_st, gate_diag_real_st, alus=[1], output_streams=g.SG4_E[1] )
	
				# calculate Psi_real * gate_diag_imag
				state_real__gate_diag_imag_st	= g.mul( state_real_st, gate_diag_imag_st, alus=[0], output_streams=g.SG4_E[3] )

				# calculate Psi_real * gate_diag_imag + Psi_imag * gate_diag_real
				state__gate_diag_imag_st	= g.add( state_imag__gate_diag_real_st, state_real__gate_diag_imag_st, alus=[3], output_streams=g.SG4_W[0]  )
			
				#res3_mt = state__gate_diag_imag_st.write(name=f"test3", layout="H1(E), -1, S4(30-33)", program_output=True)




				# calculate Psi_permuted_real * gate_offdiag_real
				state_permuted_real_st 				= permuted_state_real_mt.read( streams=[g.SG4[7]] )
				gate_offdiag_real_st 				= gate_offdiag_real_mt.read( streams=[g.SG4[6]] )
				state_permuted_real__gate_offdiag_real_st	= g.mul( state_permuted_real_st, gate_offdiag_real_st, alus=[13], output_streams=g.SG4_E[6] )

				# calculate Psi_permuted_imag * gate_offdiag_imag
				state_permuted_imag_st 				= permuted_state_imag_mt.read( streams=[g.SG4[4]] )
				gate_offdiag_imag_st 				= gate_offdiag_imag_mt.read( streams=[g.SG4[5]] )
				state_permuted_imag__gate_offdiag_imag_st	= g.mul( state_permuted_imag_st, gate_offdiag_imag_st, alus=[5], output_streams=g.SG4_E[5] )			

				# calculate Psi_permuted_real * gate_offdiag_real - Psi_permuted_imag * gate_offdiag_imag
				state_permuted__gate_offdiag_real_st	= g.sub( state_permuted_real__gate_offdiag_real_st, state_permuted_imag__gate_offdiag_imag_st, alus=[10], output_streams=g.SG4_E[4]  )
				#res_mt = state_permuted__gate_offdiag_real_st.write(name=f"test2", layout="H1(E), -1, S4(20-28)", program_output=True)


			
				# calculate Psi_permuted_imag * gate_offdiag_real
				state_permuted_imag__gate_offdiag_real_st	= g.mul( state_permuted_imag_st, gate_offdiag_real_st, alus=[9], output_streams=g.SG4_E[4] )
				#res_mt = state_permuted_imag__gate_offdiag_real_st.write(name=f"test4", layout="H1(E), -1, S4(20-28)", program_output=True)
				
				# calculate Psi_permuted_real * gate_diag_imag
				state_permuted_real__gate_offdiag_imag_st	= g.mul( state_permuted_real_st, gate_offdiag_imag_st, alus=[8], output_streams=g.SG4_E[7] )
				#res_mt = state_permuted_real__gate_offdiag_imag_st.write(name=f"test5", layout="H1(E), -1, S4(10-20)", program_output=True)

			
				# calculate Psi_permuted_real * gate_offdiag_imag + Psi_permuted_imag * gate_offdiag_real
				state_permuted__gate_offdiag_imag_st	= g.add( state_permuted_imag__gate_offdiag_real_st, state_permuted_real__gate_offdiag_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
			
				#res4_mt = state_permuted__gate_offdiag_imag_st.write(name=f"test2", layout="H1(E), -1, S4(20-28)", program_output=True)	
			

				# number of memory slices to store the transformed state
				memory_slices = 1 << (qbit_num-small_qbit_num_limit)
				slice_offset  = ((gate_idx+1) % 2) * 4096
	

				layout_transformed_state_real = layout_State_input_real + f", A{memory_slices}({slice_offset}-{slice_offset+memory_slices-1})"
				layout_transformed_state_imag = layout_State_input_imag + f", A{memory_slices}({slice_offset}-{slice_offset+memory_slices-1})"

				# calculate (Psi*gate_diag).real + (Psi_permuted*gate_offdiag).real
				Psi_transformed_real_st	= g.add( state__gate_diag_real_st,  state_permuted__gate_offdiag_real_st, alus=[7], output_streams=g.SG4_W[2] )
				Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_{gate_idx}", layout=layout_transformed_state_real, program_output=True)


			
				# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_offdiag).imag
				Psi_transformed_imag_st	= g.add( state__gate_diag_imag_st,  state_permuted__gate_offdiag_imag_st, alus=[14], output_streams=g.SG4_W[6] )
				Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_{gate_idx}", layout=layout_transformed_state_imag, program_output=True)
			'''

		pgm_pkg.compile_program_context(pgm3)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)		

	iops = pgm_pkg.assemble()

	return iops



def invoke(device, iop, pgm_num, ep_num, tensors):
	"""Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

	pgm = iop[pgm_num]
	ep = pgm.entry_points[ep_num]
	input_buffer = runtime.BufferArray(ep.input, 1)[0]
	output_buffer = runtime.BufferArray(ep.output, 1)[0]


	if ep.input.tensors:
		for input_tensor in ep.input.tensors:
			if input_tensor.name not in tensors:
				raise ValueError(f"Missing input tensor named {input_tensor.name}")
			input_tensor.from_host(tensors[input_tensor.name], input_buffer)

	device.invoke(input_buffer, output_buffer)
	outs = {}

	if ep.output.tensors:
		for output_tensor in ep.output.tensors:
			result_tensor = output_tensor.allocate_numpy_array()
			output_tensor.to_host(output_buffer, result_tensor)
			outs[output_tensor.name] = result_tensor
	return outs


def run(iop_file, input_real, input_imag, target_qbit, gate_kernels_real, gate_kernels_imag):
	"""
    This function interacts with the device driver at a lower level to show
    the control of loading 2 programs and invoking each program through
    program entry points. A higher level abstraction will be made available
    to load and invoke multiple programs in a future release.
	"""

	np.set_printoptions(linewidth=1000, threshold=10000)

	if not os.path.exists(iop_file):
		raise Exception(f"IOP file does not exist: {iop_file}")

	print(f"Running programs from {iop_file}")



	iop = runtime.IOProgram(iop_file)
	device = runtime.devices[0]
	device.open()
	device.load(iop[0], unsafe_keep_entry_points=True)
	device.load(iop[1], unsafe_keep_entry_points=True)
	device.load(iop[2], unsafe_keep_entry_points=True)
	"""
	# encode the target qubits: the 320 lanes are organized into 20 16byte superlanes. The distributor can distribute elements in one superlane in one clock cycle. 
	# so the target qubits is encoded in the first 1 byte of each 16byte segments
	modified_qbits = np.zeros( (320,), dtype=np.uint8 )
	modified_qbits[0:320:16] = target_qbit[0]
	"""
	# map for mem_gather to select the permutation map for the given target qubit. Th epermutor is used only for target qubits smaller than small_qbit_num_limit
	state_permute_map_selector = np.zeros( (gate_count*3,320,), dtype=np.uint8 )
	step = 0
	print('target qubits: ', target_qbit )
	for idx in range(len(target_qbit)):
		for jdx in range(2):
			if ( target_qbit[idx][jdx] < small_qbit_num_limit ) :
				state_permute_map_selector[step,0:320:16] = target_qbit[idx][jdx]
			else:
				state_permute_map_selector[step,0:320:16] = 0
			step += 1
		target_qbit_copy = list(target_qbit[:])
		target_qbit_copy = sorted(target_qbit_copy)
		state_permute_map_selector[step,0:320:16] = int(small_qbit_num_limit-1+(13-target_qbit_copy[idx][0])*target_qbit_copy[idx][0]/2+target_qbit_copy[idx][1])
		step += 1

	#print("state_permute_map_selector: ", state_permute_map_selector)

	# maps for mem_gather to select the correct state_1 indices according to the target qubit
	state_1_selector = state_permute_map_selector[0:2,:]#np.zeros( (2,320,), dtype=np.uint8 )
	state_0_selector = state_permute_map_selector[0:2,:]#np.zeros( (2,320,), dtype=np.uint8 )

	# generate packed gate kernel data
	gate_kernels_real_packed = np.zeros( (320,), dtype=np.float32 )
	gate_kernels_imag_packed = np.zeros( (320,), dtype=np.float32 )

	gate_kernels_real_shape = gate_kernels_real.shape
	gate_kernels_imag_shape = gate_kernels_imag.shape

	if gate_kernels_real_shape != gate_kernels_imag_shape:
		raise ValueError( "the shape of the real part and the imaginary part of gate kernels should match" )


	if gate_kernels_real_shape[1] != 4 or gate_kernels_real_shape[2] != 4:
		raise ValueError( "gate kernels should be of size 4x4" )


	gate_num = gate_kernels_real_shape[0]

	gate_kernels_real_packed = np.ascontiguousarray( gate_kernels_real.reshape( (-1,) ) )
	gate_kernels_imag_packed = np.ascontiguousarray( gate_kernels_imag.reshape( (-1,) ) )

	######## run the first program ########
	pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag, "state_permute_map_selector": state_permute_map_selector, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed, "state_1_selector": state_1_selector, "state_0_selector": state_0_selector})

	######## run the gate split program ########
	index_of_split_program = 1
	pgm_2_output = invoke(device, iop, index_of_split_program, 0, {})
	#print( pgm_2_output.keys() )
	
	'''
	# test to print out the ditributed gate elements
	print( gate_kernels_real )
	print( gate_kernels_imag )
	for idx in range(4):
		for jdx in range(4):
			print(f"{idx}{jdx}")
			print( pgm_2_output[f"gate_kernel_broadcasted_real_0_{idx}{jdx}"] )
			print( pgm_2_output[f"gate_kernel_broadcasted_imag_0_{idx}{jdx}"] )
			print(' ')
	'''

	

	######## run the third program ########
	index_of_gate_program = 2
	pgm_3_output = invoke(device, iop, index_of_gate_program, 0, {})

	#print( pgm_3_output )

	#print(pgm_3_output["tmp_mt"])
	#print(pgm_3_output["tmp_imag_mt"] == pgm_3_output["tmp_imag_mt2"])
	#print(pgm_3_output["tmp_real_mt"] == pgm_3_output["tmp_real_mt2"])
	'''
	print("gate_diag_0")
	print(pgm_3_output["gate_diag_real_0"])
	print(pgm_3_output["gate_diag_imag_0"])
	print("gate_secont_column_0")
	print(pgm_3_output["gate_offdiag_real_0"])
	print(pgm_3_output["gate_offdiag_imag_0"])

	print("gate_diag_1")
	print(pgm_3_output["gate_diag_real_1"])
	print(pgm_3_output["gate_diag_imag_1"])
	print("gate_secont_column_1")
	print(pgm_3_output["gate_offdiag_real_1"])
	print(pgm_3_output["gate_offdiag_imag_1"])
	#print(pgm_3_output["Psi_transformed_real"])
	#print(pgm_3_output["Psi_transformed_imag"])
	'''

	print(' original state ')
	print( input_real[0:8] )
	print(' permuted with target qubit ', target_qbit[0][0])
	print( pgm_3_output["permuted_result_real_0"][0, 0:8] )
	print(' permuted with target qubit ', target_qbit[0][1])
	print( pgm_3_output["permuted_result_real_1"][0, 0:8] )
	print(' permuted with target qubit ', target_qbit[0][0], target_qbit[0][1])
	print( pgm_3_output["permuted_result_real_2"][0, 0:8] )

	print(' ')
	print(' ')
	print( pgm_3_output["gate_00"] )


	return None, None
	#return pgm_3_output[f"Psi_transformed_real_{len(target_qbit)-1}"], pgm_3_output[f"Psi_transformed_imag_{len(target_qbit)-1}"]



def main( State_orig_real_float32, State_orig_imag_float32, target_qbit, gate_kernels ):
	"""Compiles and runs the example programs."""

	iop_files = compile()
	print(f"Program compiled to IOP file: {iop_files}")

	import time
	#return None
	start_time = time.time()
	transformed_state = run(iop_files[0], State_orig_real_float32, State_orig_imag_float32, target_qbit, np.real(gate_kernels), np.imag(gate_kernels) )

	print("time elapsed with groq: ", time.time() - start_time )
	'''
	##################################xx
	import shutil
	shutil.copyfile("build_iop/topo_0/gate/gate.aa", "build_iop/topo_0/gate/gate.0.aa")
	utils.chain_aa("build_iop/topo_0/gate/gate.0.aa", 3)
	##############################xx
	'''
	return transformed_state




if __name__ == "__main__":
    main()
'''
# compile the program
iop_file = g.compile(base_name="QCsim", result_tensor=result)
g.write_visualizer_data("QCsim")


program = tsp.create_tsp_runner(iop_file)
t0 = time.time()
result = program(State_input_real=State_orig_real_float32, State_input_imag=State_orig_real_float32)
groq_result_mm = result['result']
'''
