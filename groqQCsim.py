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


layout_State_input_real 	="H1(W), S4(0-3)"
layout_State_input_imag 	="H1(W), S4(4-7)"

# layout to gather map selecting the permutation map corresponding to the target qubit  -- the MT should be replaced by on chip determined gather map, currently it is uploaded from CPU
layout_permute_map_selector 	="H1(W), A1(320), S1(17)"

layout_gate_kernels_real_packed ="H1(W), S4(40-43)"
layout_gate_kernels_imag_packed	="H1(W), S4(40-43)"

# memory layouts for splitted gate operations (Each 320byte vector contains one gate kernel in the first 4 elements)
# addresses from 0 to gate_count//2                    on each hemispheres are occupied with the broadcasted 00 (real or imag) element of the gate kernels
# addresses from gate_count//2 to 2*(gate_count//2)    on each hemispheres are occupied with the broadcasted 01 (real or imag) element of the gate kernels
# addresses from 2*gate_count//2 to 3*(gate_count//2)  on each hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from 3*gate_count//2 to 4*(gate_count//2)  on each hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from gate_count*2 to gate_count*2+gate_count/2 on each hemispheres are occupied with the shifted (and still packed) gate kernels  -- this is temporary used.
layout_gate_kernels_real_EAST 	="H1(E), S4(8-11)"
layout_gate_kernels_imag_EAST	="H1(E), S4(12-15)"
layout_gate_kernels_real_WEST 	="H1(W), S4(8-11)"
layout_gate_kernels_imag_WEST	="H1(W), S4(12-15)"

# memory layout to store the vectors describing states |0> and |1> for different target qubits within a single 320 element vector
# addresses from 0 to small_qbit_num_limit-1       			are occupied by vectors describing states |0> and |1> for different target qubits (states |1> are labeled by bits 1)
# addresses from small_qbit_num_limit to 2*small_qbit_num_limit-1 	are occupied by vectors describing states |0> and |1> for different target qubits (states |0> are labeled by bits 1)
# addresses from 2*small_qbit_num_limit to 2*small_qbit_num_limit+3 	are occupied by distributor maps to used to broadcast elements 01,10,11 of the gate kernel
layout_state_1			= f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(W), S1(17)"  
layout_state_0			= f"A{small_qbit_num_limit}({small_qbit_num_limit}-{2*small_qbit_num_limit-1}), H1(W), S1(17)"
layout_distribute_map_tensor    = "H1(E), S1(17)"

# permute maps at -- used to reorder the amplitude elements according to the given target qubit
layout_permute_maps = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(W), S1(43)"

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

		State_input_real_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_real", layout=layout_State_input_real)
		State_input_imag_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_imag", layout=layout_State_input_real)

		permute_map_selector_mt = g.input_tensor(shape=(1,320,), dtype=g.uint8, name="permute_map_selector", layout=layout_permute_map_selector)

		gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
		gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


		State_input_real_mt.is_static 		= True
		State_input_imag_mt.is_static 		= True
		permute_map_selector_mt.is_static 	= True
		gate_kernels_real_packed_mt.is_static 	= True
		gate_kernels_imag_packed_mt.is_static 	= True


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
		state_1_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
		state_0_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
		for target_qbit_loc in range(small_qbit_num_limit):
			target_qbit_pair_diff = 1 << target_qbit_loc
			for idx in range(256): # 256 = 2^8
				if (idx & target_qbit_pair_diff > 0):
					state_1_np[target_qbit_loc, idx] = 255
					state_0_np[target_qbit_loc, idx] = 0
				else:
					state_1_np[target_qbit_loc, idx] = 0
					state_0_np[target_qbit_loc, idx] = 255
			print( "states |1> at target qubit ", target_qbit_loc, " is:" )
			print( state_1_np[target_qbit_loc, 0:256] )

		state_1_mt = g.from_data( np.asarray(state_1_np, dtype=np.uint8), layout=layout_state_1 )
		state_1_mt.is_static = True

		state_0_mt = g.from_data( np.asarray(state_0_np, dtype=np.uint8), layout=layout_state_0 )
		state_0_mt.is_static = True

		pgm_pkg.compile_program_context(pgm1)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)
	#############################################################################################################x
	# program to split the gate kernels and other metada between the EAST-WAST hemispheres 
	name = "split_gates"
	with pgm_pkg.create_program_context(name) as pgm2:

		# create reusable permutor requests
		permutor_requests = []
		permutor_requests.append(  g.tensor.create_permutor_request( [1], 1 ) ) # EAST
		permutor_requests.append(  g.tensor.create_permutor_request( [0], 1 ) ) # WEST
		
		# create reusable distributor requests
		distributor_requests = []
		distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
		distributor_requests.append(  g.tensor.create_distributor_request( [0], 1 ) ) # WEST		

		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		gate_kernels_real_packed_shared = g.shared_memory_tensor(gate_kernels_real_packed_mt, name="gate_kernels_real_packed")
		gate_kernels_imag_packed_shared = g.shared_memory_tensor(gate_kernels_imag_packed_mt, name="gate_kernels_imag_packed")

		# concat real and imag parts to pipeline them at once through a shifter
		gate_kernels_packed_mt = g.concat( [gate_kernels_real_packed_shared, gate_kernels_imag_packed_shared], dim=0 )
		gate_kernels_packed_st = gate_kernels_packed_mt.read( streams=g.SG4[0], time=0 )
		print( gate_kernels_packed_st.physical_shape )
		print( gate_kernels_packed_st.shape )

		# split the tensor between hemispheres WEST and EAST (each (2k+1)-th gate kernel goes to the western hemispheres, while 2k-th gates are assigned to the eastern hemisphere
		gate_kernels_imag_list = []
		gate_kernels_real_list = []
		gate_count = 6
		for gate_idx in range(gate_count):

			gate_kernels_packed_list = g.split_inner_splits( gate_kernels_packed_st )
			gate_kernels_real_packed_st = gate_kernels_packed_list[0]
			gate_kernels_imag_packed_st = gate_kernels_packed_list[1]

			if ( gate_idx % 2 == 0 ):
				layout_real = layout_gate_kernels_real_WEST
				layout_imag = layout_gate_kernels_imag_WEST
			else:
				layout_real = layout_gate_kernels_real_EAST
				layout_imag = layout_gate_kernels_imag_EAST

			gate_kernels_imag_mt = gate_kernels_imag_packed_st.write(name=f"gate_kernel_imag_{gate_idx}", layout=layout_imag + f", A1({gate_count*2 + gate_idx//2})", program_output=True)
			gate_kernels_real_mt = gate_kernels_real_packed_st.write(name=f"gate_kernel_real_{gate_idx}", layout=layout_real + f", A1({gate_count*2 + gate_idx//2})", program_output=True)

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

		# create distributo maps for the process to roadcast elements 00,01,10,11 of the gate kernel onto a 320 element vectors
		distributor_00_np = np.zeros( (320,), dtype=np.uint8 )
		distribute_map_tensor_mt_00 = g.from_data( np.asarray(distributor_00_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*small_qbit_num_limit})" )
		distributor_01_np = np.ones( (320,), dtype=np.uint8 )
		distribute_map_tensor_mt_01 = g.from_data( np.asarray(distributor_01_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*small_qbit_num_limit+1})" )
		distributor_10_np = np.ones( (320,), dtype=np.uint8 )*2
		distribute_map_tensor_mt_10 = g.from_data( np.asarray(distributor_10_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*small_qbit_num_limit+2})" )
		distributor_11_np = np.ones( (320,), dtype=np.uint8 )*3
		distribute_map_tensor_mt_11 = g.from_data( np.asarray(distributor_11_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({2*small_qbit_num_limit+3})" )


		g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_00], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
		g.add_mem_constraints([distribute_map_tensor_mt_00, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_01], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
		g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_00, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_10], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 

		distribute_map_tensor_list = [distribute_map_tensor_mt_00, distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11]

  

		gate_kernels_broadcasted_imag_list = []
		gate_kernels_broadcasted_real_list = []
		loop_time = 41 # comming from the iterations of the previous cycles

		gate_kernel_element_labels = ["00", "01", "10", "11"]
		for gate_idx in range(gate_count):

			gate_kernel_real_mt = gate_kernels_real_list[gate_idx]
			gate_kernel_imag_mt = gate_kernels_imag_list[gate_idx]

			for idx in range(len(gate_kernel_element_labels) ): # iterates over the labels of the gate kernel elements
				label = gate_kernel_element_labels[idx]

				# concat real and imag parts to pipeline them at once through a broadcaster
				gate_kernel_real_st = gate_kernel_real_mt.read( streams=g.SG4[1], time=12 + 2*idx + gate_idx*loop_time ) 
				gate_kernel_imag_st = gate_kernel_imag_mt.read( streams=g.SG4[1], time=14 + 2*idx + gate_idx*loop_time )		
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


				if ( gate_idx % 2 == 0 ):
					layout_real = layout_gate_kernels_real_WEST
					layout_imag = layout_gate_kernels_imag_WEST
				else:
					layout_real = layout_gate_kernels_real_EAST
					layout_imag = layout_gate_kernels_imag_EAST

				address_offset = idx*int(gate_count/2)

				layout_real = layout_real + f", A1({address_offset + gate_idx//2})"
				layout_imag = layout_imag + f", A1({address_offset + gate_idx//2})"
				gate_kernels_broadcasted_real_mt = gate_kernel_broadcasted_real_st.write(name=f"gate_kernel_broadcasted_real_{gate_idx}_"+label, layout=layout_real, program_output=True)
				gate_kernels_broadcasted_imag_mt = gate_kernel_broadcasted_imag_st.write(name=f"gate_kernel_broadcasted_imag_{gate_idx}_"+label, layout=layout_imag, program_output=True)


				# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels
				if ( len(gate_kernels_broadcasted_imag_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
				if ( len(gate_kernels_broadcasted_real_list) > 0 ):
					g.add_mem_constraints(gate_kernels_broadcasted_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
    

				g.add_mem_constraints(gate_kernels_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
				g.add_mem_constraints(gate_kernels_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    


				gate_kernels_broadcasted_imag_list.append( gate_kernels_broadcasted_imag_mt )
				gate_kernels_broadcasted_real_list.append( gate_kernels_broadcasted_real_mt )


		

		pgm_pkg.compile_program_context(pgm2)

		g.write_visualizer_data("QCsim_"+name)	

	#############################################################################################################x		
	# program to perform gate operation in a target qubit less than small_qbit_num_limit
	name = "gate"
	with pgm_pkg.create_program_context(name) as pgm3:

		#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)


		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		State_input_real_shared = g.shared_memory_tensor(State_input_real_mt, name="State_input_real_shared")
		State_input_imag_shared = g.shared_memory_tensor(State_input_imag_mt, name="State_input_imag_shared")

		permute_map_selector_shared = g.shared_memory_tensor(permute_map_selector_mt, name="permute_map_selector_shared")

		permute_maps_mt_shared = g.shared_memory_tensor(permute_maps_mt, name=f"permute_maps_shared")
		

		# component to perform the int8 matrix multiplication, splitted into chunks.
		with g.ResourceScope(name="gatescope", is_buffered=True, time=0) as gatescope :

			# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
			print('gatescope')   
			target_qbit = 0
			'''
			permute_map_mt = self.permute_maps_mt_list[target_qbit]
			state_real_mt_8 = g.reinterpret(self.state_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
			state_real_st_8 = g.permute_inner(state_real_mt_8, permute_map_mt, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )
			'''

			print( g.mem_gather.__doc__ )
			print( permute_maps_mt_shared.physical_shape )
			permute_map_selector_st = permute_map_selector_shared.read(streams=[g.SG1_W[24]])
			permute_map_st = g.mem_gather(permute_maps_mt_shared, permute_map_selector_st, output_streams=[g.SG1_W[24]])


			state_real_mt_8 = g.reinterpret(State_input_real_shared, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
			state_imag_mt_8 = g.reinterpret(State_input_imag_shared, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

			# real and imaginary parts of the state concatenated into a unified tensor
			state_mt_8 = g.concat( [state_real_mt_8, state_imag_mt_8], dim=0 )
			print('state_mt_8:')
			print( state_mt_8.layout )
			print( state_mt_8.physical_shape )
			print( state_mt_8.shape )			



			state_st_8 = state_mt_8.read( streams=g.SG1[0] )

			
			#state_mt_8 = g.reinterpret(state_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
			state_st_8 = g.permute_inner(state_st_8, permute_map_st, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )
			
			state_st_8 = g.reshape( state_st_8, [2,4,256] )

			# split unified tensor into real and imaginary perts
			state_st_8_list = g.split( state_st_8, num_splits=2, dim=0 )

			print( state_st_8_list )
			print( state_st_8_list[0].physical_shape )
			state_real_st_8 = g.reshape(state_st_8_list[0], [4,256] )
			state_imag_st_8 = g.reshape(state_st_8_list[1], [4,256] )
			#state_real_st = g.reinterpret(state_st_8_list[0], g.float32 )
			
			state_real_st = g.reinterpret(state_real_st_8, g.float32 )
			state_imag_st = g.reinterpret(state_imag_st_8, g.float32 )

			result_real_mt = state_real_st.write(name=f"result_real", layout=f"H1(E), -1, S4(0-3)", program_output=True)
			result_imag_mt = state_imag_st.write(name=f"result_imag", layout=f"H1(E), -1, S4(4-7)", program_output=True)


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

	# encode the target qubits: the 320 lanes are organized into 20 16byte superlanes. The distributor can distribute elements in one superlane in one clock cycle. 
	# so the target qubits is encoded in the first 1 byte of each 16byte segments
	modified_qbits = np.zeros( (320,), dtype=np.uint8 )
	modified_qbits[0:320:16] = target_qbit

	# map for mem_gather to select the permutation map for the given target qubit. Th epermutor is used only for target qubits smaller than small_qbit_num_limit
	permute_map_selector = np.zeros( (1,320,), dtype=np.uint8 )
	if ( target_qbit < small_qbit_num_limit ) :
		permute_map_selector[0,0:320:16] = target_qbit
	else :
		permute_map_selector[0,0:320:16] = 0

	#print( permute_map_selector )

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
	pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag, "permute_map_selector": permute_map_selector, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed})
	
	# run the gate split program
	index_of_split_program = 1
	pgm_2_output = invoke(device, iop, index_of_split_program, 0, {})
	print( pgm_2_output.keys() )
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
	

	# run the second program
	index_of_gate_program = 2
	pgm_3_output = invoke(device, iop, index_of_gate_program, 0, {})
	#print(pgm_2_output)

	return pgm_3_output["result_real"], pgm_3_output["result_imag"]



def main( State_orig_real_float32, State_orig_imag_float32, target_qbit, gate_kernels ):
	"""Compiles and runs the example programs."""

	iop_files = compile()
	print(f"Program compiled to IOP file: {iop_files}")


	transformed_state = run(iop_files[0], State_orig_real_float32, State_orig_imag_float32, target_qbit, np.real(gate_kernels), np.imag(gate_kernels) )
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
