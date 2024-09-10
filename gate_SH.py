from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

import numpy as np

class Gate(Split_gates):
	"""
	A class used to implement the gate operation on qubits S with control qubits H.
	H qubits are 18,19,20,21


	Attributes
	----------
    
	mxm_permutation : bool
		If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
		Permutor takes twice as much time as the MXM. Gate SL does not support MXM permutation because of stream overlaps of tranposer and MXM planes.

	"""
    
	def __init__(self, pgm_pkg, mxm_permutation=True):
		"""
		Parameters
		----------
		pgm_pkg : ProgramPackage
			The program package in which the gate operation program is generated
		mxm_permutation : bool
			If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
			Permutor takes twice as much time as the MXM. Gate SL does not support MXM permutation because of stream overlaps of tranposer and MXM planes.
		"""
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation)



	def select_H_control_qubit_map_east(self, time=None):
		"""
		Function used to read out the correct control qubit map on the east hemisphere.
		The control qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The control memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The control memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------
		time : int, optional
			The time when the first read instruction in the function should be scheduled. 
			If None, the timing is deriven from components out of the scope of the function.

		Return
		------
		Returns with the stream tensor of the control qubits memory map.
		"""

		# half of the memory slices on which the state is stored
		half_memory_slices = memory_slices >> 1

		control_qubit_address_maps_SH_east_list = g.split( self.control_qubit_address_maps_SL_east_shared, 1 )

		control_qubit_address_maps_SH_east_mt   = g.concat( control_qubit_address_maps_SH_east_list[half_memory_slices:memory_slices] )
		control_qubit_address_maps_SH_east_st   = control_qubit_address_maps_SH_east_mt.read( streams = g.SG1[27], time=time )


		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )

		control_qubit_address_maps_SH_east_st = g.distribute_8( control_qubit_address_maps_SH_east_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_east,
									map_stream_req=g.SG1[12],
									bypass8=0b11110111,
									)	


		control_qubit_address_maps_SH_east_st = g.transpose_null(control_qubit_address_maps_SH_east_st, 
					                                                         transposer_req=self.transposer_request_east, 
					                                                         stream_order=[11])			

		return control_qubit_address_maps_SH_east_st




	def select_H_control_qubit_map_west(self, time=None):
		"""
		Function used to read out the correct control qubit map on the west hemisphere.
		The control qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The control memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The control memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------
		time : int, optional
			The time when the first read instruction in the function should be scheduled. 
			If None, the timing is deriven from components out of the scope of the function.

		Return
		------
		Returns with the stream tensor of the control qubits memory map.
		"""

		# half of the memory slices on which the state is stored
		half_memory_slices = memory_slices >> 1

		control_qubit_address_maps_SH_west_list = g.split( self.control_qubit_address_maps_SL_west_shared, 1 )

		control_qubit_address_maps_SH_west_mt   = g.concat( control_qubit_address_maps_SH_west_list[half_memory_slices:half_memory_slices << 1] )
		control_qubit_address_maps_SH_west_st   = control_qubit_address_maps_SH_west_mt.read( streams = g.SG1[31], time=time )
		
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )

		control_qubit_address_maps_SH_west_st = g.distribute_8( control_qubit_address_maps_SH_west_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_west,
									map_stream_req=g.SG1[12],
									bypass8=0b01111111,
									)	


		control_qubit_address_maps_SH_west_st = g.transpose_null(control_qubit_address_maps_SH_west_st, 
					                                                         transposer_req=self.transposer_request_west, 
					                                                         stream_order=[15])			

		return control_qubit_address_maps_SH_west_st


	
	def gate_SH_program(self):
		"""
		Function implementing the SH gate operation on the Groq chip
		"""

	
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_SH"
		
		with self.pgm_pkg.create_program_context(name) as pgm5:
		

			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared      = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared      = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")

			
				

			# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
			state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")
	
			

			# Creates a shared tensor to reuse the memory allocation made by program pgm5

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
			
			
			self.control_qubit_address_maps_SL_west_shared = g.shared_memory_tensor(self.control_qubit_address_maps_SL_west_mt, name="control_qubit_address_maps_SL_west_shared")		
			self.control_qubit_address_maps_SL_east_shared = g.shared_memory_tensor(self.control_qubit_address_maps_SL_east_mt, name="control_qubit_address_maps_SL_east_shared")	

			self.transposer_request_west  = g.tensor.create_transposer_request( [1], 1) # request transposer on the west side
			self.transposer_request_east  = g.tensor.create_transposer_request( [3], 1) # request transposer on the east side
			self.distributor_request_west = g.tensor.create_distributor_request( [3], 1 )  # West side	
			self.distributor_request_east = g.tensor.create_distributor_request( [7], 1 )  # East side	
			

			#################################################################################
			with g.ResourceScope(name=f"prepare_second_control_qubit_map_scope", is_buffered=True, time=0) as prepare_second_control_qubit_map_scope :


				control_qubit_address_maps_SL_east_st   = self.select_H_control_qubit_map_east()
						                                                         
				control_qubit_address_maps_SL_east_copy_mt = control_qubit_address_maps_SL_east_st.write( name='control_qubit_address_map_SL_east_copy', layout=control_qubit_address_maps_SL_east_layout_b, time=0 )
			
			
			#################################################################################
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, predecessors=[prepare_second_control_qubit_map_scope], time=None) as prepare_state_pair_scope :


				if ( False ): #self.mxm_permutation ): # if MXM is used to generate qubit pairs


					# we can only start the prepare_gate_kernel_scope after the weights have been install because of stream limitations
					tm_pkg = 25

					mxm_requests = []
					mxm_requests.append(g.tensor.create_mxm_request(planes=[0], signed=True))
					mxm_requests.append(g.tensor.create_mxm_request(planes=[1], signed=True))
					mxm_requests.append(g.tensor.create_mxm_request(planes=[2], signed=True))
					mxm_requests.append(g.tensor.create_mxm_request(planes=[3], signed=True))


					mxm_permute_matrices_west_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_west, name="mxm_permute_matrices_west_shared")
					mxm_permute_matrices_east_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_east, name="mxm_permute_matrices_east_shared")
				
					mxm_permute_map_selector_shared  = g.shared_memory_tensor(self.state_permute_map_selector_expanded, name="mxm_permute_map_selector_shared")

					mxm_permute_matrices_west_list = g.split(mxm_permute_matrices_west_shared, dim=0, num_splits=8 )
					mxm_permute_matrices_east_list = g.split(mxm_permute_matrices_east_shared, dim=0, num_splits=8 )
					
					# gathering the permutation matrices over the 16 slices with mem_gather 
					# allows dynamic memory addressing according to the target qubit
					permute_matrix_west_1_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)
					permute_matrix_west_2_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)
					permute_matrix_east_1_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)	
					permute_matrix_east_2_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)

					
					# list storing the fetched rows of the permutation matrix
					mxm_permute_row_west_st_list = []					
			
					# mxm only accepts tensors with an inner dim of 320, thus 256 inner splits should be extended to 320
					state_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)
					state_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)

					state_real_mt_320 = g.reinterpret(state_real_mt_320, dtype=g.int8)
					state_imag_mt_320 = g.reinterpret(state_imag_mt_320, dtype=g.int8)

					state_permuted_real_mt = g.reinterpret(State_input_real_copy_shared, dtype=g.int8)
					state_permuted_imag_mt = g.reinterpret(State_input_imag_copy_shared, dtype=g.int8)

					# separating the tensors along the slice dimension (separating the bytes)
					Psi_transformed_real_mt_list      = g.split(state_real_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_mt_list      = g.split(state_imag_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_real_copy_mt_list = g.split(state_permuted_real_mt, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_copy_mt_list = g.split(state_permuted_imag_mt, dim=-2, num_splits=g.float32.size)
						
					
					
										
					# 20 cycles to read the permute matrix, 8 for installing the weights and each matmul operation takes "memory_slices" cycles
					time_list = [0, 0, 1*half_memory_slices, 1*half_memory_slices]
					weights_list = []
					weights_list.append(g.install_weights(permute_matrix_west_1_st, planes=mxm_requests[0], time=0))
					weights_list.append(g.install_weights(permute_matrix_west_2_st, planes=mxm_requests[1], time=20))					
					weights_list.append(g.install_weights(permute_matrix_east_1_st, planes=mxm_requests[2], time=-26)) #TODO make it time=0
					weights_list.append(g.install_weights(permute_matrix_east_2_st, planes=mxm_requests[3], time=20-26)) #TODO make it time=20

				
					control_qubit_address_maps_SL_east_list_a = g.split( self.control_qubit_address_maps_SL_east_shared, 1 )
					control_qubit_address_maps_SL_west_tmp   = g.concat( control_qubit_address_maps_SL_west_list[0:half_memory_slices] )
					control_qubit_address_maps_SL_east_tmp_a = g.concat( control_qubit_address_maps_SL_east_list_a[0:half_memory_slices] )
					#control_qubit_address_maps_SL_east_tmp_b = g.concat( control_qubit_address_maps_SL_east_list_b[0:half_memory_slices] )
					
										
					
					
					for byte_idx in range(4):

						if byte_idx % 2 == 0:
							if byte_idx % 2 == 0:
								control_qubit_address_maps_SL_west_st = control_qubit_address_maps_SL_west_tmp.read( streams = g.SG1[26], time=time_list[byte_idx])
								
								control_qubit_address_maps_SL_west_st = g.transpose_null(control_qubit_address_maps_SL_west_st, 
							                                                         transposer_req=transposer_request, 
							                                                         stream_order=[10])						
							else:
								control_qubit_address_maps_SL_west_st = control_qubit_address_maps_SL_west_tmp.read( streams = g.SG1_E[31], time=time_list[byte_idx])

							control_qubit_address_maps_SL_east_a_st = control_qubit_address_maps_SL_east_tmp_a.read( streams = g.SG1_W[31] )					
							control_qubit_address_maps_SL_east_b_st = control_qubit_address_maps_SL_east_tmp_b.read( streams = g.SG1_W[30] )
							
							


						### Real part

						permuted_state_real_st = g.mem_gather(Psi_transformed_real_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[24+byte_idx%2]] )
						permuted_state_real_st = permuted_state_real_st.matmul(weights_list[byte_idx%2+2], planes=mxm_requests[byte_idx%2+2])

						#permuted_state_real_st = Psi_transformed_real_mt_list[byte_idx].matmul(weights_list[byte_idx%2+2], planes=mxm_requests[byte_idx%2+2], time=time_list[byte_idx])
						permuted_state_real_st = g.reinterpret(permuted_state_real_st, dtype=g.int8)

						# we only need the last byte from the MXM result
						permuted_state_real_st = g.split(permuted_state_real_st, dim=-2, num_splits=g.int32.size)[0]

						g.mem_scatter(permuted_state_real_st, Psi_transformed_real_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_a_st)
						#permuted_state_real_st.write(storage_req=Psi_transformed_real_copy_mt_list[byte_idx].storage_request)



						### Imag part

						permuted_state_imag_st = g.mem_gather(Psi_transformed_imag_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_W[24+byte_idx%2]] )
						permuted_state_imag_st = permuted_state_imag_st.matmul(weights_list[byte_idx%2], planes=mxm_requests[byte_idx%2])

						#permuted_state_imag_st = Psi_transformed_imag_mt_list[byte_idx].matmul(weights_list[byte_idx%2], planes=mxm_requests[byte_idx%2], time=time_list[byte_idx])
						permuted_state_imag_st = g.reinterpret(permuted_state_imag_st, dtype=g.int8)

						# we only need the last byte from the MXM result
						permuted_state_imag_st = g.split(permuted_state_imag_st, dim=-2, num_splits=g.int32.size)[0]

						g.mem_scatter(permuted_state_imag_st, Psi_transformed_imag_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_b_st)
						#permuted_state_imag_st.write(storage_req=Psi_transformed_imag_copy_mt_list[byte_idx].storage_request)

					state_permuted_real_mt = g.reinterpret(state_permuted_real_mt, dtype=g.float32)
					state_permuted_imag_mt = g.reinterpret(state_permuted_imag_mt, dtype=g.float32)

					permuted_state_real_mt = g.constant_tensor(state_permuted_real_mt.shape, g.float32, name="permuted_result_real", storage_req=self.state_real_copy_request, initialized=False)
					permuted_state_imag_mt = g.constant_tensor(state_permuted_imag_mt.shape, g.float32, name="permuted_result_imag", storage_req=self.state_imag_copy_request, initialized=False)

				else: # if permutor is used to generate qubit pairs

					tm_pkg = 0
					
					
				
					gate_counter_perm_map_selector_shared = g.shared_memory_tensor(self.gate_counter_perm_map_selector_mt, name="gate_counter_perm_map_selector_shared" )

						

		

					# byte-wise permutation of the state vector
					Psi_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)		
					Psi_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)
					Psi_real_mt_320 = g.reinterpret(Psi_real_mt_320, dtype=g.int8)
					Psi_imag_mt_320 = g.reinterpret(Psi_imag_mt_320, dtype=g.int8)		
	
					# separating the tensors along the slice dimension (separating the bytes)
					Psi_real_mt_list       = g.split(Psi_real_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_imag_mt_list       = g.split(Psi_imag_mt_320, dim=-2, num_splits=g.float32.size)			
			
			
					Psi_real_copy_mt_list = g.split(g.reinterpret(State_input_real_copy_shared, dtype=g.int8), dim=-2, num_splits=g.float32.size)
					Psi_imag_copy_mt_list = g.split(g.reinterpret(State_input_imag_copy_shared, dtype=g.int8), dim=-2, num_splits=g.float32.size)			
			
			
					# create reusable permutor requests
					permutor_requests = []
					permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
					permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )			
			

			
			
					state_permute_map_selector_shared = g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")		
					state_permute_map_selector_list   = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
	
					# otherwise the permutor slices are used
					permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name="permute_maps_west_shared")
					permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name="permute_maps_east_shared")			


					half_memory_slices = memory_slices >> 1
					permutation_time_starts = [0, half_memory_slices, 2*half_memory_slices, 3*half_memory_slices]
									
				

					for byte_idx in range(4):
					
						# Real part
						#control_qubit_address_maps_SL_west_st = control_qubit_address_maps_SL_west_tmp.read( streams = g.SG1_E[31] )
						control_qubit_address_maps_SL_west_st = self.select_H_control_qubit_map_west()	
						
						
						gate_counter_perm_map_selector_st_a   = gate_counter_perm_map_selector_shared.read(streams=[g.SG1_E[0]])						
						gate_counter_perm_map_selector_st_b   = gate_counter_perm_map_selector_shared.read(streams=[g.SG1_E[0]])		
						
						

						control_qubit_address_maps_SL_east_copy_st = control_qubit_address_maps_SL_east_copy_mt.read( streams = g.SG1_W[30] )		
						
						state_real_byte_st                 = g.mem_gather(Psi_real_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[0]] )
						state_permute_map_selector_east_st = g.mem_gather(state_permute_map_selector_shared, gate_counter_perm_map_selector_st_b, output_streams=[g.SG1_E[24]])				
						#state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_E[24]])	
						permute_map_west_st                = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])		
						permuted_state_real_st_8           = g.permute_inner(state_real_byte_st, permute_map_west_st, permutor_req=permutor_requests[1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[4], time=permutation_time_starts[byte_idx] )
				

						g.mem_scatter(permuted_state_real_st_8, Psi_real_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_copy_st)
				
						# Imag part
						#control_qubit_address_maps_SL_east_b_st = control_qubit_address_maps_SL_east_tmp_b.read( streams = g.SG1_W[31] )
						control_qubit_address_maps_SL_east_st = self.select_H_control_qubit_map_east()
						
						state_imag_byte_st                 = g.mem_gather(Psi_imag_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_W[0]] )
						state_permute_map_selector_west_st = g.mem_gather(state_permute_map_selector_shared, gate_counter_perm_map_selector_st_a, output_streams=[g.SG1_W[24]])	
						#state_permute_map_selector_west_st = state_permute_map_selector_list[0].read(streams=[g.SG1_W[24]])	
						permute_map_west_st                = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_west_st, output_streams=[g.SG1_W[24]])
						permuted_state_imag_st_8           = g.permute_inner(state_imag_byte_st, permute_map_west_st, permutor_req=permutor_requests[0], input_streams=[g.SG1_W[0], g.SG1_E[24]], output_streams=g.SG1[4])#, time=permutation_time_starts[byte_idx]+0*memory_addresses )
				

						g.mem_scatter(permuted_state_imag_st_8, Psi_imag_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_st)
				
				
					
			#################################################################################################################x
			
			
			gate_kernels_broadcasted_real_shared_dict = {}
			gate_kernels_broadcasted_imag_shared_dict = {}
			gate_kernels_broadcasted_real_shared_copy_dict = {}
			gate_kernels_broadcasted_imag_shared_copy_dict = {}
			
			for label in gate_kernel_element_labels:

				key = '0_' + label
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


		

			
			address_offset = 4*gate_count

			layout_real_diag     = layout_gate_kernels_real + f", A1({address_offset})"
			layout_imag_diag     = layout_gate_kernels_imag + f", A1({address_offset})"		
			layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
			layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
			storage_req_real_diag     = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag     = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )	

				
			#################################################################################
			# component to perform the gate operation
			
			with g.ResourceScope(name=f"prepare_gate_kernel_scope", is_buffered=True, predecessors=[prepare_second_control_qubit_map_scope], time=None) as prepare_gate_kernel_scope :
					
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

				# create_reusable_alu_requests
				alu_requests = []
				alu_requests.append(  g.tensor.create_alu_request( [0], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [8], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [5], 1 ) )
				alu_requests.append(  g.tensor.create_alu_request( [2], 1 ) )
					
				with g.ResourceScope(name=f"prepare_filtered_gate_elements_scope", is_buffered=True, time=0) as prepare_filtered_gate_elements_scope :
				
				
					gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt,   \
												   										name="gate_counter_gate_filtering_selector_shared") 

					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors()

					#state_1_broadcasted_st = state_1_broadcasted_list[0].read( streams=g.SG4[2] )
					#state_0_broadcasted_st = state_0_broadcasted_list[0].read( streams=g.SG4[0] )

					gate_counter_gate_element_selector_00_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_00_mt, \
									name="gate_counter_gate_element_selector_00_shared") 

					gate_counter_gate_element_selector_01_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_01_mt, \
									name="gate_counter_gate_element_selector_01_shared") 


					gate_counter_gate_element_selector_10_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_10_mt, \
									name="gate_counter_gate_element_selector_10_shared") 

					gate_counter_gate_element_selector_11_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_11_mt, \
									name="gate_counter_gate_element_selector_11_shared") 
					
					
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_00_st = self.gather_gate_elements(gate_counter_gate_element_selector_00_shared, \
															g.SG1_W[30], \
															self.gate_kernels_broadcasted_real_mt, \
															self.gate_kernels_broadcasted_imag_mt, \
															output_stream_offset = 4, \
															time = 0)
															
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_11_st = self.gather_gate_elements(gate_counter_gate_element_selector_11_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 12 )

															
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[12], output_streams=g.SG4[0] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )



					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[4], output_streams=g.SG4[2] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

				
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=[1], output_streams=g.SG4[3])
					
					
					gate_diag_st_list	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_st_list[0].write(name=f"gate_diag_real", storage_req=storage_req_real_diag)#, program_output=True)
					gate_diag_imag_mt 	= gate_diag_st_list[1].write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)#, program_output=True)
					
					g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)



					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors()
					
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_01_st = self.gather_gate_elements(gate_counter_gate_element_selector_01_shared, \
															g.SG1_W[30], \
															self.gate_kernels_broadcasted_real_mt, \
															self.gate_kernels_broadcasted_imag_mt, \
															output_stream_offset = 4 )
															
					# real and imaginary parts of the gate element concatenated into a unified tensor								
					gate_10_st = self.gather_gate_elements(gate_counter_gate_element_selector_10_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 12, \
															time = 7 )
															
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[14], output_streams=g.SG4[0] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )


					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[6], output_streams=g.SG4[2] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
					
					# combine the filtered gate element vectors
					gate_off_diag_st = g.add(gate_01_st, gate_10_st, alus=[3], output_streams=g.SG4[3])
				
							
					
					gate_off_diag_st_list	= g.split( gate_off_diag_st, num_splits=2, dim=0 )
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag)
		
					g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt], [gate_off_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_copy_mt], [gate_off_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					

		




			
			
				
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :

				# define state with 320 vectors instead of 256 inner splits
				Psi_real_mt_320      = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)		
				Psi_imag_mt_320      = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)
				Psi_real_copy_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_real_copy_shared.storage_request, initialized=False)
				Psi_imag_copy_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_imag_copy_shared.storage_request, initialized=False)

				
				# separating the tensors along the slice dimension (separating the bytes)
				Psi_real_mt_320_8       = g.reinterpret(Psi_real_mt_320, dtype=g.int8)
				Psi_imag_mt_320_8       = g.reinterpret(Psi_imag_mt_320, dtype=g.int8)			
				Psi_real_bytes_mt_list  = g.split(Psi_real_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_imag_bytes_mt_list  = g.split(Psi_imag_mt_320_8, dim=-2, num_splits=g.float32.size)


				Psi_real_copy_mt_320_8       = g.reinterpret(Psi_real_copy_mt_320, dtype=g.int8)
				Psi_imag_copy_mt_320_8       = g.reinterpret(Psi_imag_copy_mt_320, dtype=g.int8)			
				Psi_real_bytes_copy_mt_list  = g.split(Psi_real_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_imag_bytes_copy_mt_list  = g.split(Psi_imag_copy_mt_320_8, dim=-2, num_splits=g.float32.size)

								

				control_qubit_address_maps_SL_west_st = self.select_H_control_qubit_map_west(time=0) #control_qubit_address_maps_SL_west_tmp.read( streams = g.SG1_E[31], time=0 )
				control_qubit_address_maps_SL_east_st = self.select_H_control_qubit_map_east()


				# read out the states byte-wise with mem-gather
				state_real_bytes_st_list      = []
				state_imag_bytes_st_list      = []
				state_real_copy_bytes_st_list = []
				state_imag_copy_bytes_st_list = []

				for byte_idx in range(4):
					
					# Real part
					state_real_byte_st  = g.mem_gather(Psi_real_bytes_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[byte_idx]] )
					state_real_byte_st  = g.reshape( state_real_byte_st, [half_memory_slices,1,320] )
					state_real_bytes_st_list.append( state_real_byte_st )

					# Imag part
					state_imag_byte_st  = g.mem_gather(Psi_imag_bytes_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[3*4+byte_idx]] )
					state_imag_byte_st  = g.reshape( state_imag_byte_st, [half_memory_slices,1,320] )
					state_imag_bytes_st_list.append( state_imag_byte_st )


					# Real part of copy
					state_real_copy_byte_st  = g.mem_gather(Psi_real_bytes_copy_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[6*4+byte_idx]] )
					state_real_copy_byte_st  = g.reshape( state_real_copy_byte_st, [half_memory_slices,1,320] )
					state_real_copy_bytes_st_list.append( state_real_copy_byte_st )


					# Imag part of copy
					state_imag_copy_byte_st  = g.mem_gather(Psi_imag_bytes_copy_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[4*4+byte_idx]] )
					state_imag_copy_byte_st  = g.reshape( state_imag_copy_byte_st, [half_memory_slices,1,320] )
					state_imag_copy_bytes_st_list.append( state_imag_copy_byte_st )



				
				state_real_st = g.concat( state_real_bytes_st_list, dim=1 )
				state_real_st = g.reinterpret(state_real_st, dtype=g.float32)


				state_imag_st = g.concat( state_imag_bytes_st_list, dim=1 )
				state_imag_st = g.reinterpret(state_imag_st, dtype=g.float32)


				state_real_copy_st = g.concat( state_real_copy_bytes_st_list, dim=1 )
				state_permuted_real_st = g.reinterpret(state_real_copy_st, dtype=g.float32)

				state_imag_copy_st = g.concat( state_imag_copy_bytes_st_list, dim=1 )
				state_permuted_imag_st = g.reinterpret(state_imag_copy_st, dtype=g.float32)
				

				Psi_transformed_real_st, Psi_transformed_imag_st = self.VXM_S_round_trip_st(state_real_st, 
                                                                                                            state_imag_st, 
                                                                                                            state_permuted_real_st, 
                                                                                                            state_permuted_imag_st, 
                                                                                                            gate_diag_real_mt, 
                                                                                                            gate_diag_imag_mt, 
                                                                                                            gate_off_diag_real_mt, 
                                                                                                            gate_off_diag_imag_mt)


				# write the calculated states into the memory with mem scatter
				Psi_transformed_real_mt_320_8       = g.reinterpret(Psi_transformed_real_st, dtype=g.int8)
				Psi_transformed_imag_mt_320_8       = g.reinterpret(Psi_transformed_imag_st, dtype=g.int8)			
				Psi_transformed_real_bytes_mt_list  = g.split(Psi_transformed_real_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_transformed_imag_bytes_mt_list  = g.split(Psi_transformed_imag_mt_320_8, dim=-2, num_splits=g.float32.size)

				for byte_idx in range(4):
					
					g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_mt_list[byte_idx], control_qubit_address_maps_SL_east_st)
					g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_st)
					g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_mt_list[byte_idx], control_qubit_address_maps_SL_east_st)
					g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_st)


			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters_for_S_gates()	
					
			
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
