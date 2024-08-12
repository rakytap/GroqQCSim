from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class Upload:
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


	def upload_program(self):
		# Defines a program to upload the inputs
		name = "upload"
		with self.pgm_pkg.create_program_context(name) as pgm1:

			self.State_input_real_mt = g.input_tensor(shape=(memory_slices, matrix_size,), dtype=g.float32, name="State_input_real", storage_req=self.state_real_request)
			self.State_input_imag_mt = g.input_tensor(shape=(memory_slices, matrix_size,), dtype=g.float32, name="State_input_imag", storage_req=self.state_imag_request)
			
			State_input_real_st = self.State_input_real_mt.read(streams=[g.SG4[1]], time = 0)
			State_input_imag_st = self.State_input_imag_mt.read(streams=[g.SG4[2]], time = 0)

			self.State_input_real_copy_mt = State_input_real_st.write( name="State_input_real_copy", storage_req=self.state_real_copy_request)
			self.State_input_imag_copy_mt = State_input_imag_st.write( name="State_input_imag_copy", storage_req=self.state_imag_copy_request)
							
			
			
			
			
			state_1_selector_mt				= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_1_selector", layout=layout_state_1_selector)
			state_0_selector_mt				= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_0_selector", layout=layout_state_0_selector)

			self.gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
			self.gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


			self.State_input_real_mt.is_static 			= True
			self.State_input_imag_mt.is_static 			= True
			self.State_input_real_copy_mt.is_static 			= True
			self.State_input_imag_copy_mt.is_static 			= True

			self.gate_kernels_real_packed_mt.is_static 	= True
			self.gate_kernels_imag_packed_mt.is_static 	= True
			state_1_selector_mt.is_static			= True
			state_0_selector_mt.is_static			= True
			
			# the input vector containing the memory address for the correct target qubits. 
			# TODO: cutrrently supports only a single target qubit, but need to support 80, as the number of the packed gate kernels		
			self.state_permute_map_selector_mt 	= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="permutation_selector", layout=layout_state_permute_map_selector)	
			self.state_permute_map_selector_mt.is_static = True	


			if ( self.mxm_permutation ):
				
				# expand the permutation maps to query for the whol 320x320 permute matrix, not just for the first row
				constant_1_address_offset_np           = np.zeros( (320,), dtype=np.uint8 )
				constant_1_address_offset_np[0:320:16] = 1
				constant_1_address_offset_mt = g.from_data( np.asarray(constant_1_address_offset_np, dtype=np.uint8), layout=layout_constant_1_address_offset )
				
				alu_rq_8 = g.tensor.create_alu_request( alus=[8] )
				
				state_permute_map_selector_new_mt_list = []
				for row_idx in range(20) :
					layout = f"H1(W), A{1}({320+gate_count+row_idx}), S1(17)"
				
					
					constant_1_address_offset_st = constant_1_address_offset_mt.read( streams=[g.SG1_E[20]], time=row_idx*31 )
				
					if row_idx == 0:
						state_permute_map_selector_st = self.state_permute_map_selector_mt.read( streams=[g.SG1_E[16]] )
					else:
						state_permute_map_selector_st = state_permute_map_selector_new_mt.read( streams=[g.SG1_E[16]] )
				
				
					# offset the memory address by 1
					state_permute_map_selector_new_st = g.add(constant_1_address_offset_st, state_permute_map_selector_st, alus=alu_rq_8, output_streams=g.SG4[7])
				
					# save the offsetted memory address
					state_permute_map_selector_new_mt = state_permute_map_selector_new_st.write(name=f"mxm_permute_selector_offset_{row_idx}", layout=layout, program_output=False)
					state_permute_map_selector_new_mt = g.reshape( state_permute_map_selector_new_mt, [1,320] )
					
					g.add_mem_constraints([state_permute_map_selector_new_mt], [self.state_permute_map_selector_mt] + state_permute_map_selector_new_mt_list, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

					state_permute_map_selector_new_mt_list.append( state_permute_map_selector_new_mt )
					
					
				# generate constant tensors from the offsetted memory addresses
				self.state_permute_map_selector_expanded = g.concat( state_permute_map_selector_new_mt_list )
				self.state_permute_map_selector_expanded.is_static = True
				print( "iiiiiiiiiiiiiiiiiiiiiiiiiiiiii ")
				print( self.state_permute_map_selector_expanded.shape ) 
			
			
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

			else:	
			
			
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


				self.mxm_permute_matrices_mt_west = None
				self.mxm_permute_matrices_mt_east = None
				
				
			
			
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

			# distribute maps to broadcast teh gate kernel elements over a vector
			distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
			for idx in range(320):
				distribute_map_np[idx] = idx % 16

			distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state1 + f", A1({small_qbit_num_limit+4})" )
			distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state0 + f", A1({small_qbit_num_limit+4})" )
			g.add_mem_constraints([states_1_mt], [distribute_map_tensor_state1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([states_0_mt], [distribute_map_tensor_state0_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

			with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=0) as distribute_state_indices_scope :

				#state_1_st = state_1_mt.read( streams=g.SG1[0], time=0 )
				#state_0_st = state_0_mt.read( streams=g.SG1[24], time=0 )

				state_1_selector_st = state_1_selector_mt.read(streams=[g.SG1[0]])
				state_1_st = g.mem_gather(states_1_mt, state_1_selector_st, output_streams=[g.SG1[0]], time=0)

				state_0_selector_st = state_0_selector_mt.read(streams=[g.SG1[24]])
				state_0_st = g.mem_gather(states_0_mt, state_0_selector_st, output_streams=[g.SG1[24]], time=0)


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
				self.state_1_broadcasted_mt = state_1_broadcasted_st.write(name="state_1_broadcasted", layout=layout_state_1_broadcasted , program_output=False)
				state_0_broadcasted_st = g.reinterpret( state_0_broadcasted_st, g.uint32 )
				self.state_0_broadcasted_mt = state_0_broadcasted_st.write(name="state_0_broadcasted", layout=layout_state_0_broadcasted, program_output=False)

				self.state_1_broadcasted_mt.is_static = True
				self.state_0_broadcasted_mt.is_static = True

				g.add_mem_constraints([states_1_mt, states_0_mt, distribute_map_tensor_state1_mt, distribute_map_tensor_state0_mt, state_1_selector_mt, state_0_selector_mt], [self.state_0_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			self.pgm_pkg.compile_program_context(pgm1)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)
		#############################################################################################################x
		# program to split the gate kernels and other metada between the EAST-WAST hemispheres 
