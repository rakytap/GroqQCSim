from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class IO:
	def __init__(self, pgm_pkg, mxm_permutation=True):

		self.pgm_pkg = pgm_pkg
		self.gate_kernels_real_packed_mt = None
		self.gate_kernels_imag_packed_mt = None
		self.State_input_real_mt = None
		self.State_input_imag_mt = None
		self.State_input_real_copy_mt = None
		self.State_input_imag_copy_mt = None
		self.state_permute_map_selector_mt = None
		self.permute_maps_west = None
		self.permute_maps_east = None
		self.state_1_broadcasted_mt = None
		self.state_0_broadcasted_mt = None
		self.control_state_1_broadcasted_mt = None
		self.control_state_0_broadcasted_mt = None
		self.ones_mt = None
		self.zeros_mt = None

		# create reusable memory requests
		self.state_imag_request = g.tensor.create_storage_request( layout=layout_State_input_imag )
		self.state_real_request = g.tensor.create_storage_request( layout=layout_State_input_real )

		self.state_imag_copy_request = g.tensor.create_storage_request( layout=layout_State_input_imag_copy )
		self.state_real_copy_request = g.tensor.create_storage_request( layout=layout_State_input_real_copy )

		# for target qubits less than 8 MXM slices or the permutors can be used to generate the pairs for the state vector elements
		# set mxm_permutation to True when MXM is used, or set to False when permutor is used
		self.mxm_permutation = mxm_permutation
		
		# the memory addresses for gathering the mxm permutation matrices
		self.state_permute_map_selector_expanded = None
		
		
	def broadcast_element( self, element_to_broadcast_st, distribute_map_mt, distributor_request, transposer_request):
	
		# broadcast the element
		
		if distributor_request.distributors == [4]:
			map_stream_req = g.SG1[12]
		elif distributor_request.distributors == [7]:
			map_stream_req = g.SG1[31]
					
		broadcasted_vector_st = g.distribute_lowest( element_to_broadcast_st, 
				distribute_map_mt,
				distributor_req=distributor_request,
				map_stream_req=map_stream_req,
				bypass8=0b00001111,
		)

		# reverse the direction of the broadcasted vector
		
		if transposer_request.transposers == [2]:
			stream_order = [4,5,6,7]
		elif transposer_request.transposers == [3]:
			stream_order = [12,13,14,15]
		
		broadcasted_vector_st = g.transpose_null(
			broadcasted_vector_st,
			transposer_req=transposer_request,
			stream_order=stream_order,
		)	
		
		
		broadcasted_vector_st = g.reinterpret( broadcasted_vector_st, g.uint32 )
							
				
		return broadcasted_vector_st
		
		
		
	def generate_control_qbit_maps_for_SL(self):
	
		
		control_qbit_min = 9
		control_qbit_max = qbit_num		

	

		
		# controlled operations perform only on half of the elements
		half_memory_slices = memory_slices >> 1
		
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_slices)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_slices+1,...,memory_slices)		
		mem_gather_map = np.zeros( (half_memory_slices << 1, 320,), dtype=np.uint8 )
	
	
		for control_qbit in range(control_qbit_min, control_qbit_max+1):
	
			index_pair = 1 << (control_qbit - small_qbit_num_limit)
			
			# each 2-byte column in the 16-byte blocks of the map tensor corresponds to a specific control qbit 
			if control_qbit <= 16:
				horizontal_offset_in_memory_map = (control_qbit-control_qbit_min)*2 
			else:
				horizontal_offset_in_memory_map = (control_qbit-17)*2 
				
				
			# maps for control_qbits for 9,10,...,16 on the upper half of the memory tensor (addresses 0,...,half_memory_slices)
			# maps for control_qbits for 17,...,21 on the lower half of the memory tensor (addresses half_memory_slices+1,...,memory_slices)		
			if control_qbit <= 16:
				vertical_offset_in_memory_map = 0
			else:
				vertical_offset_in_memory_map = half_memory_slices
							 
			mask_lower_bits = index_pair - 1 
			mask_upper_bits = 1 << ( qbit_num - control_qbit )
			mask_upper_bits = mask_upper_bits - 1
			mask_upper_bits = mask_upper_bits << (control_qbit - small_qbit_num_limit )
			
			# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
			control_qubit_state_1_indices = np.zeros( (half_memory_slices, 2,), dtype=np.uint8 )
			
			half_of_slice = 1 << (21-small_qbit_num_limit-1) # needed to offset the address for the bank alternating storage

			half_memory_slices = memory_slices >> 1
			for memory_idx in range(half_memory_slices):

				lower_bits  = memory_idx & mask_lower_bits
				higher_bits = memory_idx & mask_upper_bits

				# index where the control qubit is in state 0
				state_index_control_qubit_0 = (higher_bits << 1) + lower_bits

				# index where the control qubit is in state 1
				state_index_control_qubit_1 = state_index_control_qubit_0 + index_pair
				
				# odd indices are stored on bank2
				if state_index_control_qubit_1 % 2 == 1:
					state_index_control_qubit_1 = (state_index_control_qubit_1 >> 1 ) + half_of_slice
				else :
					state_index_control_qubit_1 = state_index_control_qubit_1 >> 1


				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses  = state_index_control_qubit_1 & 255
				higher_bits_of_addresses = (state_index_control_qubit_1 >> 8 ) & 255
											
				control_qubit_state_1_indices[memory_idx,0] = lower_bits_of_addresses
				control_qubit_state_1_indices[memory_idx,1] = higher_bits_of_addresses		
				

				
			#print( control_qubit_state_1_indices[:,0].astype(np.uint16) + (control_qubit_state_1_indices[:,1].astype(np.uint16) << 8) )
			
			row_min = vertical_offset_in_memory_map						
			row_max = vertical_offset_in_memory_map+half_memory_slices
			for superlane_idx in range(20):
				col_min = horizontal_offset_in_memory_map + 16*superlane_idx
				col_max = horizontal_offset_in_memory_map+2 + 16*superlane_idx
				#print( row_min, row_max, col_min, col_max )
				mem_gather_map[ row_min:row_max, col_min:col_max ] = control_qubit_state_1_indices
				

		return mem_gather_map
			
	


	def upload_program(self):

		# Defines a program to upload the inputs
		name = "upload"
		with self.pgm_pkg.create_program_context(name) as pgm1:

			self.State_input_real_mt = g.input_tensor(shape=(memory_slices, matrix_size,), dtype=g.float32, name="State_input_real", storage_req=self.state_real_request)
			self.State_input_imag_mt = g.input_tensor(shape=(memory_slices, matrix_size,), dtype=g.float32, name="State_input_imag", storage_req=self.state_imag_request)
			
					
			# dynamic map selectors to vectors where the target_qbit (or control_qbit) takes on state 1 or 0
			# *state_1* is True where the target_qbit (or control_qbit) takes on state 1
			# *state_0* is True where the target_qbit (or control_qbit) takes on state 0			
			state_1_selector_mt         = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_1_selector", layout=layout_state_1_selector)
			state_0_selector_mt         = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_0_selector", layout=layout_state_0_selector)			
			control_state_1_selector_mt = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="control_state_1_selector", layout=layout_control_state_1_selector)
			control_state_0_selector_mt = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="control_state_0_selector", layout=layout_control_state_0_selector)


			# 80 2x2 gate kernels are packed into real and complex 320 float32 vectors
			self.gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
			self.gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


			
			
			# the input vector containing the memory address for the correct target qubits. 
			# TODO: cutrrently supports only a single target qubit, but need to support 80, as the number of the packed gate kernels		
			self.state_permute_map_selector_mt 	    = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="permutation_selector", layout=layout_state_permute_map_selector)				
			self.mxm_state_permute_map_selector_mt 	= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="mxm_permutation_selector", layout=layout_mxm_permute_map_selector)				
			g.add_mem_constraints([self.mxm_state_permute_map_selector_mt], [self.state_permute_map_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			
			# duplicate the state vector to do linear combinations associated with the unitary operation
			State_input_real_st = self.State_input_real_mt.read(streams=[g.SG4[1]], time = 0)
			State_input_imag_st = self.State_input_imag_mt.read(streams=[g.SG4[2]], time = 0)

			self.State_input_real_copy_mt = State_input_real_st.write( name="State_input_real_copy", storage_req=self.state_real_copy_request)
			self.State_input_imag_copy_mt = State_input_imag_st.write( name="State_input_imag_copy", storage_req=self.state_imag_copy_request)
			
			# constant tensor to update the address counters by one
			constant_1_address_offset_np           = np.zeros( (320,), dtype=np.uint8 )
			constant_1_address_offset_np[0:320:16] = 1
			self.constant_1_address_offset_mt = g.from_data( np.asarray(constant_1_address_offset_np, dtype=np.uint8), layout=layout_constant_1_address_offset )
			
			self.State_input_real_mt.is_static 			 = True
			self.State_input_imag_mt.is_static 			 = True
			self.State_input_real_copy_mt.is_static 	 = True
			self.State_input_imag_copy_mt.is_static 	 = True

			self.gate_kernels_real_packed_mt.is_static 	 = True
			self.gate_kernels_imag_packed_mt.is_static 	 = True
			state_1_selector_mt.is_static			     = True
			state_0_selector_mt.is_static			     = True
			control_state_1_selector_mt.is_static        = True
			control_state_0_selector_mt.is_static        = True
			self.state_permute_map_selector_mt.is_static = True
			self.constant_1_address_offset_mt.is_static  = True


			if ( self.mxm_permutation ):
				# if MXM is used to generate the permuted state vector
				
				# expand the permutation maps to query for the whole 320x320 permute matrix, not just for the first row
				
				
				alu_rq_8 = g.tensor.create_alu_request( alus=[8] )

				# split mxm_state_permute_map_selectors according to the gates
				gate_resolved_mxm_state_permute_map_selectors_list = g.split( self.mxm_state_permute_map_selector_mt, num_splits=gate_count, dim=0 )
				
				state_permute_map_selector_new_mt_list = []
				for gate_idx in range(gate_count):

					mxm_state_permute_map_selector_mt = gate_resolved_mxm_state_permute_map_selectors_list[gate_idx]

					for row_idx in range(20) :
				
						layout = layout_expanded_mxm_permute_map_selector + f", A{1}({320 + gate_count + gate_idx*20 + row_idx})"
				
						if row_idx == 0:
							state_permute_map_selector_st = mxm_state_permute_map_selector_mt.read( streams=[g.SG1_E[16]], time=gate_idx*20*31 )
							
							state_permute_map_selector_st = g.vxm_identity( state_permute_map_selector_st, alus=alu_rq_8, output_streams=g.SG4_W[7] )
						
							# save the offsetted memory address
							state_permute_map_selector_new_mt = state_permute_map_selector_st.write(name=f"mxm_permute_selector_offset_{row_idx}", layout=layout)
						else:
					
							constant_1_address_offset_st = self.constant_1_address_offset_mt.read( streams=[g.SG1_E[20]], time=row_idx*31+gate_idx*20*31)
							state_permute_map_selector_st = state_permute_map_selector_new_mt.read( streams=[g.SG1_E[16]] )
				
				
							# offset the memory address by 1
							state_permute_map_selector_new_st = g.add(constant_1_address_offset_st, state_permute_map_selector_st, alus=alu_rq_8, output_streams=g.SG4[7])
				
							# save the offsetted memory address
							state_permute_map_selector_new_mt = state_permute_map_selector_new_st.write(name=f"mxm_permute_selector_offset_{row_idx}", layout=layout)
						
						state_permute_map_selector_new_mt = g.reshape( state_permute_map_selector_new_mt, [1,320] )
					
						g.add_mem_constraints([state_permute_map_selector_new_mt], \
								      [self.mxm_state_permute_map_selector_mt, \
                                                                       self.state_permute_map_selector_mt] + state_permute_map_selector_new_mt_list, \
	   							       g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

						state_permute_map_selector_new_mt_list.append( state_permute_map_selector_new_mt )
					
					
				# generate constant tensors from the offsetted memory addresses
				self.state_permute_map_selector_expanded = g.concat( state_permute_map_selector_new_mt_list )
				self.state_permute_map_selector_expanded.is_static = True
							
			
				# generate the permute matrices

				identity                 = np.identity(320, dtype=np.int8)
				mxm_permutation_matrices = np.zeros( (8,320,320,), dtype=np.int8 )

				for target_qbit_loc in range(small_qbit_num_limit):

					target_qbit_pair_diff = 1 << target_qbit_loc

					matrix_row_indexes = np.zeros( (320,), dtype=np.int32 )
					for idx in range(320):
						if (idx < matrix_size):
							matrix_row_indexes[idx] = idx ^ target_qbit_pair_diff
						else:
							matrix_row_indexes[idx] = idx
					

					mxm_permutation_matrices[ target_qbit_loc,:,: ] = identity[matrix_row_indexes,:]

					
				self.mxm_permute_matrices_mt_west = g.from_data( np.asarray(mxm_permutation_matrices, dtype=np.int8), layout=layout_mxm_permutation_matrices_west )
				self.mxm_permute_matrices_mt_east = g.from_data( np.asarray(mxm_permutation_matrices, dtype=np.int8), layout=layout_mxm_permutation_matrices_east )
				self.mxm_permute_matrices_mt_west.is_static = True
				self.mxm_permute_matrices_mt_east.is_static = True


				self.permute_maps_mt_west = None
				self.permute_maps_mt_east = None
			
			#else:	### !!!!!! gate the SL needs the permutor !!!!!!!!!
			# if the permutor is used to generate the permuted state vector
			
			
			# expand the permutation maps to query for the whole 320x320 permute matrix, not just for the first row
			gate_counter_perm_map_selector_np           = np.zeros( (320,), dtype=np.uint8 )
			self.gate_counter_perm_map_selector_mt      = g.from_data( np.asarray(gate_counter_perm_map_selector_np, 
											dtype=np.uint8),
										 	layout=layout_gate_counter_perm_map_selector )

			self.gate_counter_gate_filtering_selector_mt = g.from_data( np.asarray(gate_counter_perm_map_selector_np, 
											dtype=np.uint8),
										 	layout=layout_gate_counter_gate_filtering_selector )
										 	
			gate_counter_S_control_qbit_selector_np           = np.zeros( (320,), dtype=np.uint8 )
			gate_counter_S_control_qbit_selector_np[0:320:16] = gate_count
			self.gate_counter_S_control_qbit_selector_mt = g.from_data( np.asarray(gate_counter_S_control_qbit_selector_np, 
											dtype=np.uint8),
										 	layout=layout_gate_counter_S_control_qbit_selector )


			gate_element_selector_00_np                   = np.zeros( (320,), dtype=np.uint8 )
			gate_element_selector_00_np[0:320:16]         = 0 + small_qbit_num_limit*20
			self.gate_counter_gate_element_selector_00_mt = g.from_data( gate_element_selector_00_np, layout="H1(E), A1(0), S1(36)" )

			gate_element_selector_11_np                   = np.zeros( (320,), dtype=np.uint8 )
			gate_element_selector_11_np[0:320:16]         = 3*gate_count + small_qbit_num_limit*20
			self.gate_counter_gate_element_selector_11_mt = g.from_data( gate_element_selector_11_np, layout="H1(E), A1(0), S1(35)" )
					
					
			gate_element_selector_01_np                   = np.zeros( (320,), dtype=np.uint8 )
			gate_element_selector_01_np[0:320:16]         = gate_count + small_qbit_num_limit*20
			self.gate_counter_gate_element_selector_01_mt = g.from_data( gate_element_selector_01_np, layout="H1(E), A1(1), S1(36)" )

			gate_element_selector_10_np                   = np.zeros( (320,), dtype=np.uint8 )
			gate_element_selector_10_np[0:320:16]         = 2*gate_count + small_qbit_num_limit*20
			self.gate_counter_gate_element_selector_10_mt = g.from_data( gate_element_selector_10_np, layout="H1(E), A1(1), S1(35)" )	
					
			g.add_mem_constraints([self.gate_counter_gate_element_selector_00_mt], [self.gate_counter_gate_element_selector_01_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([self.gate_counter_gate_element_selector_11_mt], [self.gate_counter_gate_element_selector_10_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)				
			g.add_mem_constraints([self.gate_counter_gate_filtering_selector_mt], [self.gate_counter_S_control_qbit_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)			
										 	
			self.gate_counter_perm_map_selector_mt.is_static = True										 	
			self.gate_counter_gate_filtering_selector_mt.is_static = True	
			self.gate_counter_S_control_qbit_selector_mt.is_static = True				
			self.gate_counter_gate_element_selector_00_mt.is_static = True
			self.gate_counter_gate_element_selector_01_mt.is_static = True
			self.gate_counter_gate_element_selector_10_mt.is_static = True
			self.gate_counter_gate_element_selector_11_mt.is_static = True
			
			# generate permutation maps to reorder the state vector elements for qubits less than small_qbit_num_limit
			permute_map = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint32 )

			for target_qbit_loc in range(small_qbit_num_limit):

				permute_map_np = np.zeros( (256,), dtype=np.uint32 )
				target_qbit_pair_diff = 1 << target_qbit_loc
				
				for idx in range(256): # 256 = 2^8
					if (idx < matrix_size):
						permute_map_np[idx] = idx ^ target_qbit_pair_diff
					else:
						permute_map_np[idx] = idx
				

				permute_map[target_qbit_loc,:] = inst.encode_permute_map(  permute_map_np.tolist() )

			self.permute_maps_mt_west = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps_west )
			self.permute_maps_mt_east = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps_east )
			self.permute_maps_mt_west.is_static = True
			self.permute_maps_mt_east.is_static = True


				#self.mxm_permute_matrices_mt_west = None
				#self.mxm_permute_matrices_mt_east = None
				
				
			
			
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

			# create reusable distributor requests
			distributor_requests = []
			distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
			distributor_requests.append(  g.tensor.create_distributor_request( [7], 1 ) ) # EAST	
			
			# create reusable transposer requests
			transposer_requests = []
			transposer_requests.append(  g.tensor.create_transposer_request( [2], 1 ) ) # EAST
			transposer_requests.append(  g.tensor.create_transposer_request( [3], 1 ) ) # EAST	

			# distribute maps to broadcast the gate kernel elements over a vector
			distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
			for idx in range(320):
				distribute_map_np[idx] = idx % 16

			distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state1 + f", A1({small_qbit_num_limit+4})" )
			distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state0 + f", A1({small_qbit_num_limit+4})" )
			g.add_mem_constraints([states_1_mt], [distribute_map_tensor_state1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([states_0_mt], [distribute_map_tensor_state0_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([state_1_selector_mt], [control_state_1_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([state_0_selector_mt], [control_state_0_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			
			# generate tensors full with float32 ones and zeros  
			# this is needed later when manipulating the broadcates gate kernel elements with control qubits
			self.ones_mt = g.full((1, 1, 320), 1.0, dtype=g.float32, name="ones_mt", layout=layout_ones)
			self.zeros_mt = g.full((1, 1, 320), 0.0, dtype=g.float32, name="zeros_mt", layout=layout_zeros)
			self.ones_mt = g.reinterpret( self.ones_mt, dtype=g.uint32)
			self.zeros_mt = g.reinterpret( self.zeros_mt, dtype=g.uint32)
			self.ones_mt.is_static = True
			self.zeros_mt.is_static = True
			
			
			
			# generate control qubit memory maps for SL gate (to select the correct elements from the state vector if SL gate is performed)
			mem_gather_map = self.generate_control_qbit_maps_for_SL()

			self.control_qubit_address_maps_SL_west_mt = g.from_data(mem_gather_map, layout=control_qubit_address_maps_SL_west_layout)
			self.control_qubit_address_maps_SL_east_mt = g.from_data(mem_gather_map, layout=control_qubit_address_maps_SL_east_layout_a)			
			self.control_qubit_address_maps_SL_west_mt.is_static = True	
			self.control_qubit_address_maps_SL_east_mt.is_static = True
			
			#distributor_map_np = np.zeros( (320,), dtype=np.uint8 )
			#distributor_map_np[0:320:16] = 0
			#distributor_map_np[1:320:16] = 1
			#self.distributor_map_mt   = g.from_data(distributor_map_np, layout=layout_distributor_maps_west)
			

			self.distributor_map_mt = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="control_qubit_distributor_maps", layout=layout_distributor_maps_west)
			
			gate_counter_distributor_maps_west_np            = np.zeros( (320,), dtype=np.uint8 )
			gate_counter_distributor_maps_west_np[0:320:16]  = 1
			self.gate_counter_distributor_maps_west_mt       = g.from_data( gate_counter_distributor_maps_west_np, layout=layout_gate_counter_distributor_maps_west )	
			self.gate_counter_distributor_maps_west_mt.is_static = True
			
									
			
			g.add_mem_constraints([self.gate_counter_perm_map_selector_mt], [self.distributor_map_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=0) as distribute_state_indices_scope :

				state_1_selector_st = state_1_selector_mt.read(streams=[g.SG1[0]])
				state_1_st = g.mem_gather(states_1_mt, state_1_selector_st, output_streams=[g.SG1[0]], time=0)
				
				state_0_selector_st = state_0_selector_mt.read(streams=[g.SG1[24]])
				state_0_st = g.mem_gather(states_0_mt, state_0_selector_st, output_streams=[g.SG1[24]], time=0)
								
				state_1_broadcasted_st = self.broadcast_element(state_1_st,
				                                               distribute_map_tensor_state1_mt, 
				                                               distributor_requests[0],
				                                               transposer_requests[0])

				
				state_0_broadcasted_st = self.broadcast_element(state_0_st,
				                                               distribute_map_tensor_state0_mt, 
				                                               distributor_requests[1],
				                                               transposer_requests[1])

		
				self.state_1_broadcasted_mt = state_1_broadcasted_st.write(name="state_1_broadcasted", layout=layout_state_1_broadcasted, program_output=False)
				self.state_0_broadcasted_mt = state_0_broadcasted_st.write(name="state_0_broadcasted", layout=layout_state_0_broadcasted, program_output=False)

				self.state_1_broadcasted_mt.is_static = True
				self.state_0_broadcasted_mt.is_static = True
				
				# same as before but now for the control qbit
				control_state_1_selector_st = control_state_1_selector_mt.read(streams=[g.SG1[0]])
				control_state_1_st = g.mem_gather(states_1_mt, control_state_1_selector_st, output_streams=[g.SG1[0]], time=gate_count+1)
				
				control_state_0_selector_st = control_state_0_selector_mt.read(streams=[g.SG1[24]])
				control_state_0_st = g.mem_gather(states_0_mt, control_state_0_selector_st, output_streams=[g.SG1[24]], time=gate_count+1)
				
								
				control_state_1_broadcasted_st = self.broadcast_element(control_state_1_st,
				                                               distribute_map_tensor_state1_mt, 
				                                               distributor_requests[0],
				                                               transposer_requests[0])

				
				control_state_0_broadcasted_st = self.broadcast_element(control_state_0_st,
				                                               distribute_map_tensor_state0_mt, 
				                                               distributor_requests[1],
				                                               transposer_requests[1])
				
				
				self.control_state_1_broadcasted_mt = control_state_1_broadcasted_st.write(name="control_state_1_broadcasted", layout=layout_control_state_1_broadcasted, program_output=False)
				self.control_state_0_broadcasted_mt = control_state_0_broadcasted_st.write(name="control_state_0_broadcasted", layout=layout_control_state_0_broadcasted, program_output=False)
				
				self.control_state_1_broadcasted_mt.is_static = True
				self.control_state_0_broadcasted_mt.is_static = True
				g.add_mem_constraints([self.control_state_0_broadcasted_mt], [self.state_0_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.control_state_1_broadcasted_mt], [self.state_1_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.state_0_broadcasted_mt, self.control_state_0_broadcasted_mt], [self.zeros_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

				g.add_mem_constraints([states_1_mt, states_0_mt, distribute_map_tensor_state1_mt, distribute_map_tensor_state0_mt, state_1_selector_mt, state_0_selector_mt], [self.state_0_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			self.pgm_pkg.compile_program_context(pgm1)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)
		#############################################################################################################x
		# program to split the gate kernels and other metada between the EAST-WAST hemispheres 


	
	def download_state_program(self):
		# Defines a program to upload the inputs
		name = "download_state_vector"
		with self.pgm_pkg.create_program_context(name) as pgm2:


			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_transformed_real")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_transformed_imag")


			State_input_real_shared.set_program_output()
			State_input_imag_shared.set_program_output()
	
