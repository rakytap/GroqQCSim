import groq.api as g
import groq.api.nn as nn
import groq.runner.tsp as tsp;
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


#Note: MEM slices 16, 20, 24, and 28 on both the east and west hemispheres, and slice 38 on the west hemisphere are reserved for system use.


qbit_num = 8

# the number og qubits for which the gate operations need the permutor (need to reorganize the elements in a vector)
small_qbit_num_limit = 8

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
layout_state_permute_map_selector ="H1(W), A1(320), S2(17-18)"
layout_state_1_selector				="H1(E), A1(0), S2(19-21)" # [19, 21] because 20 is reserved
layout_state_0_selector				="H1(E), A1(0), S2(22-23)"

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
layout_distribute_map_tensor    	= "H1(E), S1(17)"
layout_distribute_map_tensor_state1 = "H1(E), S1(17)"
layout_distribute_map_tensor_state0 = "H1(E), S1(18)"

#layout_state_1_broadcasted			= 


# permute maps at -- used to reorder the state vector elements according to the given target qubit
layout_permute_maps = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(W), S1(43)"

# gate count stored in a single packed vector 
gate_count = 2

# label to indicate the 00, 01, 10, 11 elements of the gate kernel
gate_kernel_element_labels = ["00", "01", "10", "11"]

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

		memory_slices = 1 << (qbit_num-small_qbit_num_limit)

		State_input_real_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_real", layout=layout_State_input_real + f", A{memory_slices}(0-{memory_slices-1})")
		State_input_imag_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_imag", layout=layout_State_input_imag + f", A{memory_slices}(0-{memory_slices-1})")

		state_permute_map_selector_mt 	= g.input_tensor(shape=(2,320,), dtype=g.uint8, name="state_permute_map_selector", layout=layout_state_permute_map_selector)
		state_1_selector_mt				= g.input_tensor(shape=(2,320,), dtype=g.uint8, name="state_1_selector", layout=layout_state_1_selector)
		state_0_selector_mt				= g.input_tensor(shape=(2,320,), dtype=g.uint8, name="state_0_selector", layout=layout_state_0_selector)

		gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
		gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


		State_input_real_mt.is_static 			= True
		State_input_imag_mt.is_static 			= True
		state_permute_map_selector_mt.is_static = True
		gate_kernels_real_packed_mt.is_static 	= True
		gate_kernels_imag_packed_mt.is_static 	= True
		state_1_selector_mt.is_static			= True
		state_0_selector_mt.is_static			= True


		# generate permutation maps to reorder the state vector elements for qubits less than small_qbit_num_limit
		permute_map = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint32 )

		for target_qbit_loc in range(small_qbit_num_limit):

			permute_map_np = np.zeros( (256,), dtype=np.uint32 )
			target_qbit_pair_diff = 1 << target_qbit_loc
			print('target_qubit_pair_diff', target_qbit_pair_diff)
			
			for idx in range(256): # 256 = 2^8
				if (idx < matrix_size):
					permute_map_np[idx] = idx ^ target_qbit_pair_diff
				else:
					permute_map_np[idx] = idx
			

			permute_map[target_qbit_loc,:] = inst.encode_permute_map(  permute_map_np.tolist() )
			print(' ')
			print(permute_map[target_qbit_loc,:])

		permute_maps_mt = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps )
		permute_maps_mt.is_static = True


		# generate vectors to indicate which elements in a 320 vector correspond to state |1> of target qubit less than small_qbit_num_limit 
		# if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		states_1_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
		states_0_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
		for target_qbit_loc in range(small_qbit_num_limit):
			target_qbit_pair_diff = 1 << target_qbit_loc
			for idx in range(256): # 256 = 2^8
				if (idx & target_qbit_pair_diff > 0):
					states_1_np[target_qbit_loc, idx] = 255
					states_0_np[target_qbit_loc, idx] = 0
				else:
					states_1_np[target_qbit_loc, idx] = 0
					states_0_np[target_qbit_loc, idx] = 255
			print( "states |1> at target qubit ", target_qbit_loc, " is:" )
			print( states_1_np[target_qbit_loc, 0:256] )

		states_1_mt = g.from_data( np.asarray(states_1_np, dtype=np.uint8), layout=layout_states_1 )
		states_1_mt.is_static = True

		states_0_mt = g.from_data( np.asarray(states_0_np, dtype=np.uint8), layout=layout_states_0 )
		states_0_mt.is_static = True

		pgm_pkg.compile_program_context(pgm1)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)
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
		print( gate_kernels_packed_st.physical_shape )
		print( gate_kernels_packed_st.shape )

		# split the tensor between hemispheres WEST and EAST (each (2k+1)-th gate kernel goes to the western hemispheres, while 2k-th gates are assigned to the eastern hemisphere
		# the gate kernel elements to be used later are the first 4 elements in each vector
		gate_kernels_imag_list = []
		gate_kernels_real_list = []

		for gate_idx in range(gate_count):

			gate_kernels_packed_list = g.split_inner_splits( gate_kernels_packed_st )
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
				shift = 4
				gate_kernels_packed_st = g.shift(gate_kernels_packed_st, 
								shift, 
								permutor_id=permutor_request, 
								shift_src=[inst.NEW_SRC] * 2, 
								dispatch_set=inst.DispatchSet.SET_1, 
								input_streams=[g.SG4[0]], 
								output_streams=[g.SG4[0]])



		# generate gate kernels to be applied on state |0> and resulting to alternating states |0> and |1> according to the periodicity of the current target qubit

		# create distributo maps for the process to broadcast elements 00,01,10,11 of the gate kernel onto a 320 element vectors
		distributor_00_np = np.zeros( (320,), dtype=np.uint8 )
		distribute_map_tensor_mt_00 = g.from_data( np.asarray(distributor_00_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit})" )
		distributor_01_np = np.ones( (320,), dtype=np.uint8 )
		distribute_map_tensor_mt_01 = g.from_data( np.asarray(distributor_01_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+1})" )
		distributor_10_np = np.ones( (320,), dtype=np.uint8 )*2
		distribute_map_tensor_mt_10 = g.from_data( np.asarray(distributor_10_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+2})" )
		distributor_11_np = np.ones( (320,), dtype=np.uint8 )*3
		distribute_map_tensor_mt_11 = g.from_data( np.asarray(distributor_11_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+3})" )

		g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_00], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
		g.add_mem_constraints([distribute_map_tensor_mt_00, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_01], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
		g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_00, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_10], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 

		distribute_map_tensor_list = [distribute_map_tensor_mt_00, distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11]

  

		gate_kernels_broadcasted_imag_list = []
		gate_kernels_broadcasted_real_list = []
		gate_kernels_broadcasted_imag_copy_list = []
		gate_kernels_broadcasted_real_copy_list = []
		loop_time_long_cycle = 51 # comming from the iterations of the previous cycles (the loops are overlapping in execution)
		loop_time_short_cycle = 31

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
				gate_kernel_real_st = gate_kernel_real_mt.read( streams=g.SG4[1], time=12 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle ) 
				gate_kernel_imag_st = gate_kernel_imag_mt.read( streams=g.SG4[1], time=14 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle )		
				gate_kernel_st = g.concat_inner_splits( [gate_kernel_real_st, gate_kernel_imag_st] )
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
				gate_kernel_broadcasted_list = g.split_inner_splits( gate_kernel_broadcasted_st )
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
				print( f"{gate_idx}_"+label )


				gate_kernels_broadcasted_real_copy_list.append( gate_kernels_broadcasted_real_copy_mt )
				gate_kernels_broadcasted_imag_copy_list.append( gate_kernels_broadcasted_imag_copy_mt )
				gate_kernels_broadcasted_real_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_real_copy_mt
				gate_kernels_broadcasted_imag_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_imag_copy_mt


		pgm_pkg.compile_program_context(pgm2)

		g.write_visualizer_data("QCsim_"+name)	

	#############################################################################################################x		
	# program to perform gate operation in a target qubit less than small_qbit_num_limit
	name = "gate"
	with pgm_pkg.create_program_context(name) as pgm3:

		#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)

		# create reusable distributor requests
		distributor_requests = []
		distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
		distributor_requests.append(  g.tensor.create_distributor_request( [7], 1 ) ) # EAST	


		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		State_input_real_shared = g.shared_memory_tensor(State_input_real_mt, name="State_input_real_shared")
		State_input_imag_shared = g.shared_memory_tensor(State_input_imag_mt, name="State_input_imag_shared")

		state_permute_map_selector_shared 	= g.shared_memory_tensor(state_permute_map_selector_mt, name="state_permute_map_selector_shared")
		state_1_selector_shared 			= g.shared_memory_tensor(state_1_selector_mt, name="state_1_selector_shared")
		state_0_selector_shared 			= g.shared_memory_tensor(state_0_selector_mt, name="state_0_selector_shared")

		permute_maps_mt_shared = g.shared_memory_tensor(permute_maps_mt, name=f"permute_maps_shared")

		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		states_1_shared = g.shared_memory_tensor(states_1_mt, name="states_1_shared")
		states_0_shared = g.shared_memory_tensor(states_0_mt, name="states_0_shared")
		#target_qbit = 3




		# Creates a shared tensor to reuse the memory allocation made by program pgm2

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

		# distribute maps to broadcast teh gate kernel elements over a vector
		distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
		for idx in range(320):
			distribute_map_np[idx] = idx % 16

		distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state1 + f", A1({small_qbit_num_limit+4})" )
		distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state0 + f", A1({small_qbit_num_limit+4})" )
		g.add_mem_constraints([states_1_shared], [distribute_map_tensor_state1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
		g.add_mem_constraints([states_0_shared], [distribute_map_tensor_state0_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

		for gate_idx in range( 1 ):
			with g.ResourceScope(name=f"prepare_state_pair_scope_{gate_idx}", is_buffered=True, time=0) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
				#target_qbit = 0
				'''
				permute_map_mt = self.permute_maps_mt_list[target_qbit]
				state_real_mt_8 = g.reinterpret(self.state_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				state_real_st_8 = g.permute_inner(state_real_mt_8, permute_map_mt, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )
				'''

				print( g.mem_gather.__doc__ )
				print( permute_maps_mt_shared.physical_shape )
				print( state_permute_map_selector_shared.physical_shape )
				state_permute_map_selector_st = state_permute_map_selector_shared.read(streams=[g.SG1_W[24]])
				permute_map_st = g.mem_gather(permute_maps_mt_shared, state_permute_map_selector_st, output_streams=[g.SG1_W[24]])


				state_real_mt_8 = g.reinterpret(Psi_transformed_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				state_imag_mt_8 = g.reinterpret(Psi_transformed_imag_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

				# real and imaginary parts of the state concatenated into a unified tensor
				state_mt_8 = g.concat( [state_real_mt_8, state_imag_mt_8], dim=0 )



				state_st_8 = state_mt_8.read( streams=g.SG1[0] )

			
				#state_mt_8 = g.reinterpret(state_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
				permuted_state_st_8 = g.permute_inner(state_st_8, permute_map_st, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )
			
				permuted_state_st_8 = g.reshape( permuted_state_st_8, [2,4,256] )

				# split unified tensor into real and imaginary perts
				permuted_state_st_8_list = g.split( permuted_state_st_8, num_splits=2, dim=0 )

				print( permuted_state_st_8_list )
				print( permuted_state_st_8_list[0].physical_shape )
				permuted_state_real_st_8 = g.reshape(permuted_state_st_8_list[0], [4,256] )
				permuted_state_imag_st_8 = g.reshape(permuted_state_st_8_list[1], [4,256] )
				#state_real_st = g.reinterpret(state_st_8_list[0], g.float32 )
			
				permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
				permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

				permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real", layout=layout_state_vector_real_permuted)
				permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag", layout=layout_state_vector_imag_permuted)


			#################################################################################
			if gate_idx == 1:
				break
			# component to perform the gate operation
			with g.ResourceScope(name=f"prepare_gate_kernel_scope_{gate_idx}", is_buffered=True, time=0) as prepare_gate_kernel_scope :

	
				# combine 00 and 10 elements of the gate kernel to be used in the transformation
				#states_1_mt_list = g.split( states_1_shared, 1, 0)
				#states_0_mt_list = g.split( states_0_shared, 1, 0)

				#state_1_mt = states_1_mt_list[target_qbit] #TODO: do with gather map
				#state_0_mt = states_0_mt_list[target_qbit] #TODO: do with gather map

				with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=0) as distribute_state_indices_scope :

					#state_1_st = state_1_mt.read( streams=g.SG1[0], time=0 )
					#state_0_st = state_0_mt.read( streams=g.SG1[24], time=0 )

					state_1_selector_st = state_1_selector_shared.read(streams=[g.SG1[0]])
					state_1_st = g.mem_gather(states_1_shared, state_1_selector_st, output_streams=[g.SG1[0]], time=0)

					state_0_selector_st = state_0_selector_shared.read(streams=[g.SG1[24]])
					state_0_st = g.mem_gather(states_0_shared, state_0_selector_st, output_streams=[g.SG1[24]], time=0)
						






					distributor_request = distributor_requests[0]
					state_1_broadcasted_st = g.distribute_lowest( state_1_st, 
				        	distribute_map_tensor_state1_mt,
				        	distributor_req=distributor_request,
				       		map_stream_req=g.SG1[12],
				        	bypass8=0b00001111,
					)

					state_1_broadcasted_st = g.transpose_null(
						state_1_broadcasted_st,
						transposer_req=2,
						stream_order=[4,5,6,7],
					)

			
					distributor_request = distributor_requests[1]
					state_0_broadcasted_st = g.distribute_lowest( state_0_st, 
				        	distribute_map_tensor_state0_mt,
				        	distributor_req=distributor_request,
				        	map_stream_req=g.SG1[31],
				        	bypass8=0b00001111,
					)

					state_0_broadcasted_st = g.transpose_null(
						state_0_broadcasted_st,
						transposer_req=3,
						stream_order=[12,13,14,15],
					)

			
					state_1_broadcasted_st = g.reinterpret( state_1_broadcasted_st, g.uint32 )
					state_1_broadcasted_mt = state_1_broadcasted_st.write(name=f"state_1_broadcasted_{gate_idx}", layout=f"H1(E), -1, S4", program_output=False) #TODO layout
					state_0_broadcasted_st = g.reinterpret( state_0_broadcasted_st, g.uint32 )
					state_0_broadcasted_mt = state_0_broadcasted_st.write(name=f"state_0_broadcasted_{gate_idx}", layout=f"H1(E), -1, S4", program_output=False) #TODO layout

				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

				with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope_{gate_idx}", is_buffered=True, predecessors=[distribute_state_indices_scope], time=None) as prepare_fileterd_gate_elements_scope :

					state_1_broadcasted_st = state_1_broadcasted_mt.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_mt.read( streams=g.SG4[7] )
			
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6], time=0 )
					gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict["0_00"].read( streams=g.SG4[6], time=0 )
					gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[8], output_streams=g.SG4[4] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )




					# real and imaginary parts of the state concatenated into a unified tensor
					gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict["0_11"].read( streams=g.SG4[0])#, time=2 )
					gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["0_11"].read( streams=g.SG4[0])#, time=2 )
					gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[0], output_streams=g.SG4[3] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )

			
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_10_st, alus=[5], output_streams=g.SG4[3])
			
			
					layout_real = layout_gate_kernels_real_EAST
					layout_imag = layout_gate_kernels_imag_EAST

					address_offset = 4*gate_count

					layout_real_diag = layout_real + f", A1({address_offset})"
					layout_imag_diag = layout_imag + f", A1({address_offset})"			
				
					gate_diag_st_list	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_st_list[0].write(name=f"gate_diag_real_{gate_idx}", layout=layout_real_diag, program_output=True)
					gate_diag_imag_mt 	= gate_diag_st_list[1].write(name=f"gate_diag_imag_{gate_idx}", layout=layout_imag_diag, program_output=True)
			

					#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
					#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)



			#with g.ResourceScope(name="prepare_gate_elements_output_state_1_scope", is_buffered=True, predecessors=[prepare_gate_elements_output_state_0_scope], time=None) as prepare_gate_elements_output_state_1_scope :

					state_1_broadcasted_st = state_1_broadcasted_mt.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_mt.read( streams=g.SG4[7] )
			
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_01_real_st = gate_kernels_broadcasted_real_shared_dict["0_01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict["0_01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[10], output_streams=g.SG4[4] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )

					# real and imaginary parts of the state concatenated into a unified tensor
					gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict["0_10"].read( streams=g.SG4[0], time=7 )
					gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["0_10"].read( streams=g.SG4[0], time=7 )
					gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[2], output_streams=g.SG4[3] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )
				
					# combine the filtered gate element vectors
					gate_offdiag_st = g.add(gate_01_st, gate_11_st, alus=[7], output_streams=g.SG4[3])
			
			
					layout_real_copy = layout_gate_kernels_real_EAST_copy
					layout_imag_copy = layout_gate_kernels_imag_EAST_copy
	
					address_offset = 4*gate_count

					layout_real_offdiag = layout_real_copy + f", A1({address_offset})"
					layout_imag_offdiag = layout_imag_copy + f", A1({address_offset})"			
				
					gate_offdiag_st_list	= g.split( gate_offdiag_st, num_splits=2, dim=0 )
					gate_offdiag_real_mt 	= gate_offdiag_st_list[0].write(name=f"gate_offdiag_real_{gate_idx}", layout=layout_real_offdiag, program_output=True)
					gate_offdiag_imag_mt 	= gate_offdiag_st_list[1].write(name=f"gate_offdiag_imag_{gate_idx}", layout=layout_imag_offdiag, program_output=True)
					#gate_output_state_0_real_copy_mt = gate_output_state_0_st_list[0].write(name=f"gate_output_state_0_real_copy", layout=layout_real_copy, program_output=True)
					#gate_output_state_0_imag_copy_mt = gate_output_state_0_st_list[1].write(name=f"gate_output_state_0_imag_copy", layout=layout_imag_copy, program_output=True)
					#gate_output_state_0_copy_mt = gate_output_state_0_st.write(name=f"gate_output_state_copy_0", layout=layout_real_copy)			
				

					#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
					#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)
			
			#################################################################################
			with g.ResourceScope(name=f"apply_gate_scope_{gate_idx}", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :

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
	state_permute_map_selector = np.zeros( (2,320,), dtype=np.uint8 )
	for i in range(len(target_qbit)):
		if ( target_qbit[i] < small_qbit_num_limit ) :
			state_permute_map_selector[i,0:320:16] = target_qbit[i]
		else :
			state_permute_map_selector[i,0:320:16] = 0
		print( f"Permute map for {target_qbit[i]}. qbit" )
		print( state_permute_map_selector[i] )


	# maps for mem_gather to select the correct state_1 indices according to the target qubit
	state_1_selector = state_permute_map_selector#np.zeros( (2,320,), dtype=np.uint8 )
	state_0_selector = state_permute_map_selector#np.zeros( (2,320,), dtype=np.uint8 )

	# generate packed gate kernel data
	gate_kernels_real_packed = np.zeros( (320,), dtype=np.float32 )
	gate_kernels_imag_packed = np.zeros( (320,), dtype=np.float32 )

	gate_kernels_real_shape = gate_kernels_real.shape
	gate_kernels_imag_shape = gate_kernels_imag.shape

	if gate_kernels_real_shape != gate_kernels_imag_shape:
		raise ValueError( "the shape of the real part and the imaginary part of gate kernels should match" )


	if gate_kernels_real_shape[1] != 2 or gate_kernels_real_shape[2] != 2:
		raise ValueError( "gate kernels should be of size 2x2" )


	gate_num = gate_kernels_real_shape[0]

	gate_kernels_real_packed = np.ascontiguousarray( gate_kernels_real.reshape( (-1,) ) )
	gate_kernels_imag_packed = np.ascontiguousarray( gate_kernels_imag.reshape( (-1,) ) )

	# run the first program
	pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag, "state_permute_map_selector": state_permute_map_selector, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed, "state_1_selector": state_1_selector, "state_0_selector": state_0_selector})
	
	# run the gate split program
	index_of_split_program = 1
	pgm_2_output = invoke(device, iop, index_of_split_program, 0, {})
	#print( pgm_2_output.keys() )
	'''
	print( pgm_2_output["gate_kernel_broadcasted_real_0_00"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_00"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_00"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_00"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_01"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_01"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_01"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_01"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_10"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_10"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_10"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_10"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_11"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_11"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_11"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_11"] )
	print(' ')
	print( pgm_2_output["gate_kernel_real_0"] )
	print( pgm_2_output["gate_kernel_imag_0"] )
	'''

	# run the second program
	index_of_gate_program = 2
	pgm_3_output = invoke(device, iop, index_of_gate_program, 0, {})
	#print(pgm_3_output["gate_diag_real"])
	#print(pgm_3_output["gate_diag_imag"])
	#print(pgm_3_output["Psi_transformed_real"])
	#print(pgm_3_output["Psi_transformed_imag"])

	return pgm_3_output["Psi_transformed_real_0"], pgm_3_output["Psi_transformed_imag_0"]



def main( State_orig_real_float32, State_orig_imag_float32, target_qbit, gate_kernels ):
	"""Compiles and runs the example programs."""

	iop_files = compile()
	print(f"Program compiled to IOP file: {iop_files}")

	import time

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
