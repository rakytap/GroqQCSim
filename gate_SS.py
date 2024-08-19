from split_gates import Split_gates
from memory_layouts import *
import groq.api as g
import numpy as np



def gather_permutation_matrix( mxm_permute_map_selector, mxm_permute_matrices):

	### split the permutation matrices over slices ###
							
	# split the stored permutation matrices into 16-line chunks
	# one permute matrix will occupy 20 addresses
	mxm_permute_rows = g.reshape( mxm_permute_matrices, [ small_qbit_num_limit*320, 1, 320 ] )
	mxm_permute_rows_list = g.split( mxm_permute_rows, dim=0, num_splits=small_qbit_num_limit*320 )
	mxm_permute_mt_slices_list = []
					
	for slice_idx in range(16):
		mxm_permute_mt_slice = mxm_permute_rows_list[slice_idx:small_qbit_num_limit*320:16]
		mxm_permute_mt_slices_list.append( g.concat(mxm_permute_mt_slice) )

	# the permutation matrix occupy 20 adresses over 16 slices
	mxm_permute_row_selectors_list = g.split( mxm_permute_map_selector, dim=0, num_splits = 20 )

	mxm_permute_row_st_list = []
	# the permutation matrix occupy 20 adresses over 16 slices
	for permute_selector_idx in range(20):
                                       
		mxm_permute_row_selector_st = mxm_permute_row_selectors_list[permute_selector_idx].read( streams = g.SG1[31])
						                
		for stream_idx in range(16):

			mxm_permute_row_st = g.mem_gather(mxm_permute_mt_slices_list[stream_idx], mxm_permute_row_selector_st, output_streams=[g.SG1[stream_idx]] )
			mxm_permute_row_st_list.append( mxm_permute_row_st )
                                        	
                                        	
                                        	
	permute_matrix_st = g.concat( mxm_permute_row_st_list )
	
	return permute_matrix_st
	
	
	

class Gate(Split_gates):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
	
	def gate_SS_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_SS"
		
		with self.pgm_pkg.create_program_context(name) as pgm5:


			# create reusable permutor requests
			permutor_requests = []
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )

			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")

			
			
			# when self.mxm_permutation is True, MXM is used for permuatations on target qubits less than 8
			if ( self.mxm_permutation ) :
				permute_maps_mt_west_shared = None 
				permute_maps_mt_east_shared = None

				mxm_permute_matrices_west_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_west, name="mxm_permute_matrices_west_shared")
				mxm_permute_matrices_east_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_east, name="mxm_permute_matrices_east_shared")
				
				mxm_permute_map_selector_shared 	= g.shared_memory_tensor(self.state_permute_map_selector_expanded, name="mxm_permute_map_selector_shared")
			else: 
			
				state_permute_map_selector_shared 	= g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")		
				state_permute_map_selector_list = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
				# otherwise the permutor slices are used
				permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name=f"permute_maps_west_shared")
				permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name=f"permute_maps_east_shared")

				mxm_permute_matrices_west_shared = None
				mxm_permute_matrices_east_shared = None


			# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
			state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")
			
			control_state_1_broadcasted_shared = g.shared_memory_tensor(self.control_state_1_broadcasted_mt, name="control_state_1_broadcasted_shared")
			control_state_0_broadcasted_shared = g.shared_memory_tensor(self.control_state_0_broadcasted_mt, name="control_state_0_broadcasted_shared")

			ones_shared = g.shared_memory_tensor(self.ones_mt, name="ones_shared")
			zeros_shared = g.shared_memory_tensor(self.zeros_mt, name="zeros_shared")
			ones_zeros_mt = g.concat([ones_shared, zeros_shared], dim=0, name="ones_zeros_mt")

			# Creates a shared tensor to reuse the memory allocation made by program pgm2

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
			gate_kernels_broadcasted_real_shared_dict = {}
			gate_kernels_broadcasted_imag_shared_dict = {}
			gate_kernels_broadcasted_real_shared_copy_dict = {}
			gate_kernels_broadcasted_imag_shared_copy_dict = {}
			
			for label in gate_kernel_element_labels:

				key = label
				gate_kernel_broadcasted_real_mt = self.gate_kernels_broadcasted_real_dict[key]
				gate_kernel_broadcasted_imag_mt = self.gate_kernels_broadcasted_imag_dict[key]

				gate_kernels_broadcasted_real_shared = g.shared_memory_tensor(gate_kernel_broadcasted_real_mt, name=gate_kernel_broadcasted_real_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared = g.shared_memory_tensor(gate_kernel_broadcasted_imag_mt, name=gate_kernel_broadcasted_imag_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_dict[key] = gate_kernels_broadcasted_real_shared
				gate_kernels_broadcasted_imag_shared_dict[key] = gate_kernels_broadcasted_imag_shared
					
				gate_kernel_broadcasted_real_copy_mt = self.gate_kernels_broadcasted_real_copy_dict[key]
				gate_kernel_broadcasted_imag_copy_mt = self.gate_kernels_broadcasted_imag_copy_dict[key]

				gate_kernels_broadcasted_real_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_real_copy_mt, name=gate_kernel_broadcasted_real_copy_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_imag_copy_mt, name=gate_kernel_broadcasted_imag_copy_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_copy_dict[key] = gate_kernels_broadcasted_real_shared_copy
				gate_kernels_broadcasted_imag_shared_copy_dict[key] = gate_kernels_broadcasted_imag_shared_copy				



			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
			Psi_transformed_real_mt = State_input_real_shared
			Psi_transformed_imag_mt = State_input_imag_shared
			Psi_transformed_real_copy_mt = State_input_real_copy_shared
			Psi_transformed_imag_copy_mt = State_input_imag_copy_shared

			# list of tensors for mutual memory exclusions
			Psi_transformed_real_list = [Psi_transformed_real_mt]
			Psi_transformed_imag_list = [Psi_transformed_imag_mt]
			Psi_transformed_real_copy_list = [Psi_transformed_real_copy_mt]
			Psi_transformed_imag_copy_list = [Psi_transformed_imag_copy_mt]

			pre = []
			tm = 0
			# we can only start the prepare_gate_kernel_scope after the weights have been install because of stream limitations
			tm_pgk = 25

			gate_diag_real_list = []
			gate_diag_imag_list = []
			gate_off_diag_real_list = []
			gate_off_diag_imag_list = []
			
			
			address_offset = 4*gate_count

			layout_real_diag = layout_gate_kernels_real + f", A1({address_offset})"
			layout_imag_diag = layout_gate_kernels_imag + f", A1({address_offset})"		
			layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
			layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
			storage_req_real_diag  = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag  = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )

			alu_rq_0  = g.tensor.create_alu_request( alus=[0] )
			alu_rq_1  = g.tensor.create_alu_request( alus=[1] )
			alu_rq_4  = g.tensor.create_alu_request( alus=[4] )			
			alu_rq_5  = g.tensor.create_alu_request( alus=[5] )				
			alu_rq_8  = g.tensor.create_alu_request( alus=[8] )				
			alu_rq_9  = g.tensor.create_alu_request( alus=[9] )							
			alu_rq_11 = g.tensor.create_alu_request( alus=[11] )			
			alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
			alu_rq_13 = g.tensor.create_alu_request( alus=[13] )			
							
			mxm_requests = []
			mxm_requests.append(g.tensor.create_mxm_request(planes=[0], signed=True))
			mxm_requests.append(g.tensor.create_mxm_request(planes=[1], signed=True))
			mxm_requests.append(g.tensor.create_mxm_request(planes=[2], signed=True))
			mxm_requests.append(g.tensor.create_mxm_request(planes=[3], signed=True))
				
			with g.ResourceScope(name=f"prepare_state_pair_scope_SS", is_buffered=True, time=tm) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
					
					
				if ( self.mxm_permutation ):

					mxm_permute_matrices_west_list = g.split(mxm_permute_matrices_west_shared, dim=0, num_splits=8 )
					mxm_permute_matrices_east_list = g.split(mxm_permute_matrices_east_shared, dim=0, num_splits=8 )
					
					# gathering the permutation matrices over the 16 slices with mem_gather 
					# allows dynamic memory addressing according to the target qubit
					permute_matrix_west_1_st = gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)									
					permute_matrix_west_2_st = gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)
					permute_matrix_east_1_st = gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)	
					permute_matrix_east_2_st = gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)														

					
					# list storing the fetched rows of the permutation matrix
					mxm_permute_row_west_st_list = []					
			
					# mxm only accepts tensors with an inner dim of 320, thus 256 inner splits should be extended to 320
					Psi_transformed_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_real_mt.storage_request, initialized=False)
					Psi_transformed_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_imag_mt.storage_request, initialized=False)

					Psi_transformed_real_mt_320 = g.reinterpret(Psi_transformed_real_mt_320, dtype=g.int8)
					Psi_transformed_imag_mt_320 = g.reinterpret(Psi_transformed_imag_mt_320, dtype=g.int8)

					Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.int8)
					Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.int8)

					# separating the tensors along the slice dimension (separating the bytes)
					Psi_transformed_real_mt_list      = g.split(Psi_transformed_real_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_mt_list      = g.split(Psi_transformed_imag_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_real_copy_mt_list = g.split(Psi_transformed_real_copy_mt, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_copy_mt_list = g.split(Psi_transformed_imag_copy_mt, dim=-2, num_splits=g.float32.size)
						
					

					# 20 cycles to read the permute matrix, 8 for installing the weights and each matmul operation takes "memory_slices" cycles
					time_list = [8, 28, 8+memory_slices, 28+memory_slices]
					weights_list = []
					weights_list.append(g.install_weights(permute_matrix_west_1_st, planes=mxm_requests[0], time=0))
					weights_list.append(g.install_weights(permute_matrix_west_2_st, planes=mxm_requests[1], time=20))					
					weights_list.append(g.install_weights(permute_matrix_east_1_st, planes=mxm_requests[2], time=-26)) #TODO make it time=0
					weights_list.append(g.install_weights(permute_matrix_east_2_st, planes=mxm_requests[3], time=20-26)) #TODO make it time=20
					
					for byte_idx in range(4):
					
						# Real part
						permuted_state_real_st = Psi_transformed_real_mt_list[byte_idx].matmul(weights_list[byte_idx%2+2], planes=mxm_requests[byte_idx%2+2], time=time_list[byte_idx])
						permuted_state_real_st = g.reinterpret(permuted_state_real_st, dtype=g.int8)
						# we only need the last byte from the MXM result
						permuted_state_real_st = g.split(permuted_state_real_st, dim=-2, num_splits=g.int32.size)[0]
						permuted_state_real_st.write(storage_req=Psi_transformed_real_copy_mt_list[byte_idx].storage_request)

						# Imag part
						permuted_state_imag_st = Psi_transformed_imag_mt_list[byte_idx].matmul(weights_list[byte_idx%2], planes=mxm_requests[byte_idx%2], time=time_list[byte_idx])
						permuted_state_imag_st = g.reinterpret(permuted_state_imag_st, dtype=g.int8)
						# we only need the last byte from the MXM result
						permuted_state_imag_st = g.split(permuted_state_imag_st, dim=-2, num_splits=g.int32.size)[0]
						permuted_state_imag_st.write(storage_req=Psi_transformed_imag_copy_mt_list[byte_idx].storage_request)

					Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.float32)
					Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.float32)

					permuted_state_real_mt = g.constant_tensor(Psi_transformed_real_copy_mt.shape, g.float32, name="permuted_result_real", storage_req=self.state_real_copy_request, initialized=False)
					permuted_state_imag_mt = g.constant_tensor(Psi_transformed_imag_copy_mt.shape, g.float32, name="permuted_result_imag", storage_req=self.state_imag_copy_request, initialized=False)

				else: # if permutor is used

					state_permute_map_selector_west_st = state_permute_map_selector_list[0].read(streams=[g.SG1_W[24]])
					state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_E[24]])
					permute_map_west_st                = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_west_st, output_streams=[g.SG1_W[24]])
					permute_map_east_st                = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])

					state_real_mt_8 = g.reinterpret(Psi_transformed_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
					state_imag_mt_8 = g.reinterpret(Psi_transformed_imag_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

					state_real_st_8 = state_real_mt_8.read( streams=g.SG1_E[4] )
					state_imag_st_8 = state_imag_mt_8.read( streams=g.SG1_W[4] )

					
					permuted_state_real_st_8 = g.permute_inner(state_real_st_8, permute_map_west_st, permutor_req=permutor_requests[1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[12], time=0 )
					permuted_state_imag_st_8 = g.permute_inner(state_imag_st_8, permute_map_east_st, permutor_req=permutor_requests[0], input_streams=[g.SG1_W[0], g.SG1_W[24]], output_streams=g.SG1[4], time=0 )

					permuted_state_real_st_8 = g.reshape(permuted_state_real_st_8, [memory_slices, 4, matrix_size] )
					permuted_state_imag_st_8 = g.reshape(permuted_state_imag_st_8, [memory_slices, 4, matrix_size] )
				
					permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
					permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

					permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real", storage_req=self.state_real_copy_request)
					permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag", storage_req=self.state_imag_copy_request)


				g.add_mem_constraints(Psi_transformed_real_copy_list, [permuted_state_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_real_copy_list.append( permuted_state_real_mt )

				g.add_mem_constraints(Psi_transformed_imag_copy_list, [permuted_state_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_imag_copy_list.append( permuted_state_imag_mt )

				
			#################################################################################
			# component to perform the gate operation
				
			with g.ResourceScope(name=f"prepare_gate_kernel_scope_SS", is_buffered=True, time=tm_pgk) as prepare_gate_kernel_scope :

		
				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
					
					
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

				# create_reusable_alu_requests
				alu_requests = []
				alu_requests.append(  g.tensor.create_alu_request( [0], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [8], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [5], 1 ) )
				alu_requests.append(  g.tensor.create_alu_request( [2], 1 ) )
					
				with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope_SS", is_buffered=True, time=0) as prepare_fileterd_gate_elements_scope :

					state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )
					control_state_1_broadcasted_st = control_state_1_broadcasted_shared.read( streams=g.SG4[2] )
					control_state_0_broadcasted_st = control_state_0_broadcasted_shared.read( streams=g.SG4[5] )
					ones_zeros_st = ones_zeros_mt.read( streams=g.SG4[3])

						
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["00"].read( streams=g.SG4[6], time=0 )
					gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict["00"].read( streams=g.SG4[6], time=0 )
					gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )


					# real and imaginary parts of the state concatenated into a unified tensor
					gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict["11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=alu_requests[0], output_streams=g.SG4[3] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

				
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=alu_requests[2], output_streams=g.SG4[3])

					# we want to apply the gate only if the control qbit is |1>
					gate_diag_st = g.reinterpret( gate_diag_st, g.uint32 )
					gate_diag_st = g.bitwise_and( gate_diag_st, control_state_1_broadcasted_st, alus=alu_requests[3], output_streams=g.SG4[3])
					gate_diag_st = g.reinterpret( gate_diag_st, g.float32 )

					# we need to 'normalize' the control_state_0_broadcasted tensor
					control_state_0_broadcasted_st = g.reinterpret( control_state_0_broadcasted_st, g.uint32 )
					control_state_0_broadcasted_st = g.bitwise_and(control_state_0_broadcasted_st, ones_zeros_st, alus=[4], output_streams=g.SG4[5])
					control_state_0_broadcasted_st = g.reinterpret( control_state_0_broadcasted_st, g.float32)

					# when the control qbit is |0> we dont want to change the state
					gate_diag_st = g.add(gate_diag_st, control_state_0_broadcasted_st, alus=[7], output_streams=g.SG4[3])
							
					print("layout: ", layout_imag_diag)
					
					gate_diag_real_st, gate_diag_imag_st	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_real_st.write(name=f"gate_diag_real", storage_req=storage_req_real_diag)
					gate_diag_imag_mt 	= gate_diag_imag_st.write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)


				#with g.ResourceScope(name="prepare_gate_elements_output_state_1_scope", is_buffered=True, predecessors=[prepare_gate_elements_output_state_0_scope], time=None) as prepare_gate_elements_output_state_1_scope :

					state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )
					control_state_1_broadcasted_st = control_state_1_broadcasted_shared.read( streams=g.SG4[2] )
				
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_01_real_st = gate_kernels_broadcasted_real_shared_dict["01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict["01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )

					# real and imaginary parts of the state concatenated into a unified tensor
					gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict["10"].read( streams=g.SG4[0], time=7 )
					gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["10"].read( streams=g.SG4[0], time=7 )
					gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=alu_requests[0], output_streams=g.SG4[3] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
					
					# combine the filtered gate element vectors
					gate_off_diag_st = g.add(gate_01_st, gate_10_st, alus=alu_requests[2], output_streams=g.SG4[3])
				
					gate_off_diag_st = g.reinterpret( gate_off_diag_st, g.uint32 )
					gate_off_diag_st = g.bitwise_and( gate_off_diag_st, control_state_1_broadcasted_st, alus=alu_requests[3], output_streams=g.SG4[3])
					gate_off_diag_st = g.reinterpret( gate_off_diag_st, g.float32 )
					
					gate_off_diag_st_list	= g.split( gate_off_diag_st, num_splits=2, dim=0 )
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag)

						
					gate_off_diag_real_list.append(gate_off_diag_real_mt)
					gate_off_diag_imag_list.append(gate_off_diag_imag_mt)

				
				#################################################################################
			
			with g.ResourceScope(name=f"apply_gate_scope_SS", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :


				# calculate Psi_real * gate_diag_real				
				state_real_st 			      = Psi_transformed_real_mt.read( streams=[g.SG4[0]] )	
				#gate_diag_real_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_diag, name="gate_diag_real_mt_256", initialized=False)	
				#gate_diag_real_st 		      = gate_diag_real_mt_256.read( streams=[g.SG4_W[0]] )
				#state_real__gate_diag_real_st = g.mul( state_real_st, gate_diag_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0], time=150 )
				gate_diag_real_mt.latch(alus=alu_rq_12, input_streams=g.SG4_W[0], time=0)
				state_real__gate_diag_real_st = state_real_st.mul(None, alus=alu_rq_12, output_streams=g.SG4_E[0], time=2)
				

				# calculate Psi_imag * gate_diag_imag
				state_imag_st 			      = Psi_transformed_imag_mt.read( streams=[g.SG4[3]] )
				#gate_diag_imag_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_diag, name="gate_diag_imag_mt_256", initialized=False)	
				#gate_diag_imag_st 		        = gate_diag_imag_mt_256.read( streams=[g.SG4_W[3]] )
				#state_imag__gate_diag_imag_st	= g.mul( state_imag_st, gate_diag_imag_st, alus=[4], output_streams=g.SG4_E[2] )	
				gate_diag_imag_mt.latch(alus=alu_rq_4, input_streams=g.SG4_W[3], time=0)				
				state_imag__gate_diag_imag_st = state_imag_st.mul(None, alus=alu_rq_4, output_streams=g.SG4_E[2])
				

				# calculate Psi_real * gate_diag_real - Psi_imag * gate_diag_imag
				state__gate_diag_real_st	= g.sub( state_real__gate_diag_real_st, state_imag__gate_diag_imag_st, alus=[2], output_streams=g.SG4_E[2] )
						

	
					
					
				# calculate Psi_imag * gate_diag_real
				#state_imag__gate_diag_real_st	= g.mul( state_imag_st, gate_diag_real_st, alus=alu_rq_1, output_streams=g.SG4_E[3] )	
				gate_diag_real_mt.latch(alus=alu_rq_1, input_streams=g.SG4_W[0], time=1)
				state_imag__gate_diag_real_st = state_imag_st.mul(None, alus=alu_rq_1, output_streams=g.SG4_E[3])		
					
				# calculate Psi_real * gate_diag_imag
				#state_real__gate_diag_imag_st	= g.mul( state_real_st, gate_diag_imag_st, alus=[0], output_streams=g.SG4_E[1] )				
				gate_diag_imag_mt.latch(alus=alu_rq_0, input_streams=g.SG4_W[3], time=1)
				state_real__gate_diag_imag_st	= state_real_st.mul( None, alus=alu_rq_0, output_streams=g.SG4_E[1] )
				

				# calculate Psi_real * gate_diag_imag + Psi_imag * gate_diag_real
				state__gate_diag_imag_st	= g.add( state_imag__gate_diag_real_st, state_real__gate_diag_imag_st, alus=[3], output_streams=g.SG4_W[1]  )
									
		# alus 8, 13			
					
					
				# calculate Psi_permuted_real * gate_off_diag_real
				state_permuted_real_st 			            	= permuted_state_real_mt.read( streams=[g.SG4_E[6]] )
				#gate_off_diag_real_mt_256                  = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_off_diag, name="gate_off_diag_real_mt_256", initialized=False)					
				#gate_off_diag_real_st 			         	= gate_off_diag_real_mt_256.read( streams=[g.SG4_W[6]] )
				#state_permuted_real__gate_off_diag_real_st	= g.mul( state_permuted_real_st, gate_off_diag_real_st, alus=[13], output_streams=g.SG4_E[6] )
				gate_off_diag_real_mt.latch(alus=alu_rq_13, input_streams=g.SG4_W[6], time=0)				
				state_permuted_real__gate_off_diag_real_st = state_permuted_real_st.mul(None, alus=alu_rq_13, output_streams=g.SG4_E[6])

				# calculate Psi_permuted_imag * gate_off_diag_imag
				state_permuted_imag_st 				            = permuted_state_imag_mt.read( streams=[g.SG4[4]] )
				#gate_off_diag_imag_mt_256                  = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_off_diag, name="gate_off_diag_imag_mt_256", initialized=False)					
				#gate_off_diag_imag_st                      = gate_off_diag_imag_mt_256.read( streams=[g.SG4_W[5]] )
				#state_permuted_imag__gate_off_diag_imag_st	= g.mul( state_permuted_imag_st, gate_off_diag_imag_st, alus=[5], output_streams=g.SG4_E[5] )
				gate_off_diag_imag_mt.latch(alus=alu_rq_5, input_streams=g.SG4_W[5], time=0)				
				state_permuted_imag__gate_off_diag_imag_st = state_permuted_imag_st.mul(None, alus=alu_rq_5, output_streams=g.SG4_E[5])			

				# calculate Psi_permuted_real * gate_off_diag_real - Psi_permuted_imag * gate_off_diag_imag
				state_permuted__gate_off_diag_real_st	= g.sub( state_permuted_real__gate_off_diag_real_st, state_permuted_imag__gate_off_diag_imag_st, alus=[10], output_streams=g.SG4_E[5]  )	

					
				
				# calculate Psi_permuted_imag * gate_off_diag_real
				#state_permuted_imag__gate_off_diag_real_st	= g.mul( state_permuted_imag_st, gate_off_diag_real_st, alus=[9], output_streams=g.SG4_E[4] )
				gate_off_diag_real_mt.latch(alus=alu_rq_9, input_streams=g.SG4_W[6], time=1)
				state_permuted_imag__gate_off_diag_real_st	= state_permuted_imag_st.mul( None, alus=alu_rq_9, output_streams=g.SG4_E[4] )
					
				# calculate Psi_permuted_real * gate_diag_imag
				#state_permuted_real__gate_off_diag_imag_st	= g.mul( state_permuted_real_st, gate_off_diag_imag_st, alus=[8], output_streams=g.SG4_E[7] )
				gate_off_diag_imag_mt.latch(alus=alu_rq_8, input_streams=g.SG4_W[5], time=1)
				state_permuted_real__gate_off_diag_imag_st	= state_permuted_real_st.mul( None, alus=alu_rq_8, output_streams=g.SG4_E[7] )

				
				# calculate Psi_permuted_real * gate_off_diag_imag + Psi_permuted_imag * gate_off_diag_real
				state_permuted__gate_off_diag_imag_st	= g.add( state_permuted_imag__gate_off_diag_real_st, state_permuted_real__gate_off_diag_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
				
		

				# calculate (Psi*gate_diag).real + (Psi_permuted*gate_off_diag).real
				Psi_transformed_real_st	= g.add( state__gate_diag_real_st,  state_permuted__gate_off_diag_real_st, alus=[7], output_streams=g.SG4_W[2] )
				Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

				Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real", storage_req=self.state_real_request)
				Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy", storage_req=self.state_real_copy_request)

				g.add_mem_constraints(Psi_transformed_real_list, [Psi_transformed_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_real_list.append( Psi_transformed_real_mt )
				
				g.add_mem_constraints(Psi_transformed_real_copy_list, [Psi_transformed_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_real_copy_list.append( Psi_transformed_real_copy_mt )

				# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_off_diag).imag
				Psi_transformed_imag_st	    = g.add( state__gate_diag_imag_st,  state_permuted__gate_off_diag_imag_st, alus=[14], output_streams=g.SG4_E[6] )
				Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


				Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag", storage_req=self.state_imag_request)
				Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy", storage_req=self.state_imag_copy_request)
					
				g.add_mem_constraints(Psi_transformed_imag_list, [Psi_transformed_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_imag_list.append( Psi_transformed_imag_mt )
					
				g.add_mem_constraints(Psi_transformed_imag_copy_list, [Psi_transformed_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_imag_copy_list.append( Psi_transformed_imag_copy_mt )
					
				
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
