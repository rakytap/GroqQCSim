from gate_S_base import Gate_S_base
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate(Gate_S_base):
	"""
	A class used to implement the gate operation on qubits S with control qubits S


	Attributes
	----------
    
	mxm_permutation : bool
		If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
		Permutor takes twice as much time as the MXM.

	"""

	def __init__(self, pgm_pkg, mxm_permutation=True):
		"""
		Parameters
		----------
		pgm_pkg : ProgramPackage
			The program package in which the gate operation program is generated
		mxm_permutation : bool
			If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
			Permutor takes twice as much time as the MXM.
		"""
        
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
		
	def gather_control_qubit_0_selector_SS(self ):
	
		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		control_state_0_broadcasted_shared = g.shared_memory_tensor(self.control_state_0_broadcasted_mt, name="control_state_0_broadcasted_shared")
	
		
		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(self.gate_counter_S_control_qbit_selector_mt,   \
   										name="gate_counter_S_control_qbit_selector_shared") 
		gate_counter_S_control_qbit_selector_st     = gate_counter_S_control_qbit_selector_shared.read( streams=g.SG1[0] )

		# split state_1_broadcasted_shared and state_0_broadcasted_shared into slices to use mem_gather on the tensor
		control_state_0_broadcasted_shared = g.reinterpret(control_state_0_broadcasted_shared, dtype=g.uint8 )
		control_state_0_broadcasted_shared_list = g.split( control_state_0_broadcasted_shared, dim=2, num_splits=4 )
				
		control_state_0_broadcasted_bytes_list = []
		
		#control_state_0_broadcasted_st = control_state_0_broadcasted_shared.read( streams=g.SG4[5] )		

		for slice_idx in range(4):

			control_state_0_broadcasted_byte_st = g.mem_gather(control_state_0_broadcasted_shared_list[slice_idx], 
												gate_counter_S_control_qbit_selector_st, 
												output_streams=[g.SG1[5*4+slice_idx]] )

			control_state_0_broadcasted_byte_st = g.reshape( control_state_0_broadcasted_byte_st, [1,320])
			control_state_0_broadcasted_bytes_list.append( control_state_0_broadcasted_byte_st )
                                        	             	

		control_state_0_broadcasted_st = g.concat( control_state_0_broadcasted_bytes_list )
		control_state_0_broadcasted_st = g.reinterpret(control_state_0_broadcasted_st, dtype=g.uint32 )	
		
		return control_state_0_broadcasted_st


	def gather_control_qubit_selector_SS(self ):
	
		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		control_state_1_broadcasted_shared = g.shared_memory_tensor(self.control_state_1_broadcasted_mt, name="control_state_1_broadcasted_shared")
		control_state_0_broadcasted_shared = g.shared_memory_tensor(self.control_state_0_broadcasted_mt, name="control_state_0_broadcasted_shared")
	
		
		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(self.gate_counter_S_control_qbit_selector_mt,   \
   										name="gate_counter_S_control_qbit_selector_shared") 
		gate_counter_S_control_qbit_selector_st     = gate_counter_S_control_qbit_selector_shared.read( streams=g.SG1[0] )

		# split state_1_broadcasted_shared and state_0_broadcasted_shared into slices to use mem_gather on the tensor
		control_state_1_broadcasted_shared = g.reinterpret(control_state_1_broadcasted_shared, dtype=g.uint8 )
		control_state_1_broadcasted_shared_list = g.split( control_state_1_broadcasted_shared, dim=2, num_splits=4 )

		control_state_0_broadcasted_shared = g.reinterpret(control_state_0_broadcasted_shared, dtype=g.uint8 )
		control_state_0_broadcasted_shared_list = g.split( control_state_0_broadcasted_shared, dim=2, num_splits=4 )
				
		control_state_0_broadcasted_bytes_list = []
		control_state_1_broadcasted_bytes_list = []
		
		#control_state_1_broadcasted_st = control_state_1_broadcasted_shared.read( streams=g.SG4[2] )
		#control_state_0_broadcasted_st = control_state_0_broadcasted_shared.read( streams=g.SG4[5] )		

		for slice_idx in range(4):
			control_state_1_broadcasted_byte_st = g.mem_gather(control_state_1_broadcasted_shared_list[slice_idx], 
												gate_counter_S_control_qbit_selector_st, 
												output_streams=[g.SG1[2*4+slice_idx]] )
			control_state_1_broadcasted_byte_st = g.reshape( control_state_1_broadcasted_byte_st, [1,320])
			control_state_1_broadcasted_bytes_list.append( control_state_1_broadcasted_byte_st )

			control_state_0_broadcasted_byte_st = g.mem_gather(control_state_0_broadcasted_shared_list[slice_idx], 
												gate_counter_S_control_qbit_selector_st, 
												output_streams=[g.SG1[5*4+slice_idx]] )

			control_state_0_broadcasted_byte_st = g.reshape( control_state_0_broadcasted_byte_st, [1,320])
			control_state_0_broadcasted_bytes_list.append( control_state_0_broadcasted_byte_st )
                                        	             	
		control_state_1_broadcasted_st = g.concat( control_state_1_broadcasted_bytes_list )
		control_state_1_broadcasted_st = g.reinterpret(control_state_1_broadcasted_st, dtype=g.uint32 )

		control_state_0_broadcasted_st = g.concat( control_state_0_broadcasted_bytes_list )
		control_state_0_broadcasted_st = g.reinterpret(control_state_0_broadcasted_st, dtype=g.uint32 )	
		
		return control_state_0_broadcasted_st, control_state_1_broadcasted_st


	def gather_control_qubit_1_selector_SS(self ):
	
		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		control_state_1_broadcasted_shared = g.shared_memory_tensor(self.control_state_1_broadcasted_mt, name="control_state_1_broadcasted_shared")
	
		
		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(self.gate_counter_S_control_qbit_selector_mt,   \
   										name="gate_counter_S_control_qbit_selector_shared") 
		gate_counter_S_control_qbit_selector_st     = gate_counter_S_control_qbit_selector_shared.read( streams=g.SG1[0] )

		# split state_1_broadcasted_shared and state_0_broadcasted_shared into slices to use mem_gather on the tensor
		control_state_1_broadcasted_shared = g.reinterpret(control_state_1_broadcasted_shared, dtype=g.uint8 )
		control_state_1_broadcasted_shared_list = g.split( control_state_1_broadcasted_shared, dim=2, num_splits=4 )

		control_state_1_broadcasted_bytes_list = []
		
		#control_state_1_broadcasted_st = control_state_1_broadcasted_shared.read( streams=g.SG4[2] )		

		for slice_idx in range(4):
			control_state_1_broadcasted_byte_st = g.mem_gather(control_state_1_broadcasted_shared_list[slice_idx], 
												gate_counter_S_control_qbit_selector_st, 
												output_streams=[g.SG1[2*4+slice_idx]] )
			control_state_1_broadcasted_byte_st = g.reshape( control_state_1_broadcasted_byte_st, [1,320])
			control_state_1_broadcasted_bytes_list.append( control_state_1_broadcasted_byte_st )

                                        	             	
		control_state_1_broadcasted_st = g.concat( control_state_1_broadcasted_bytes_list )
		control_state_1_broadcasted_st = g.reinterpret(control_state_1_broadcasted_st, dtype=g.uint32 )

		
		return control_state_1_broadcasted_st



	def gather_gate_filter_tensors_SS(self ):
	
		# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
		# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
		state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")	
	
	
		gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt,   \
   										name="gate_counter_gate_filtering_selector_shared") 
		gate_counter_gate_filtering_selector_st     = gate_counter_gate_filtering_selector_shared.read( streams=g.SG1[0] )

		# split state_1_broadcasted_shared and state_0_broadcasted_shared into slices to use mem_gather on the tensor
		state_1_broadcasted_shared = g.reinterpret(state_1_broadcasted_shared, dtype=g.uint8 )
		state_1_broadcasted_shared_list = g.split( state_1_broadcasted_shared, dim=2, num_splits=4 )

		state_0_broadcasted_shared = g.reinterpret(state_0_broadcasted_shared, dtype=g.uint8 )
		state_0_broadcasted_shared_list = g.split( state_0_broadcasted_shared, dim=2, num_splits=4 )
				
		state_1_broadcasted_bytes_list = []
		state_0_broadcasted_bytes_list = []

		for slice_idx in range(4):
			state_1_broadcasted_byte_st = g.mem_gather(state_1_broadcasted_shared_list[slice_idx], 
												gate_counter_gate_filtering_selector_st, 
												output_streams=[g.SG1[1*4+slice_idx]] )
			state_1_broadcasted_byte_st = g.reshape( state_1_broadcasted_byte_st, [1,320])
			state_1_broadcasted_bytes_list.append( state_1_broadcasted_byte_st )

			state_0_broadcasted_byte_st = g.mem_gather(state_0_broadcasted_shared_list[slice_idx], 
												gate_counter_gate_filtering_selector_st, 
												output_streams=[g.SG1[7*4+slice_idx]] )

			state_0_broadcasted_byte_st = g.reshape( state_0_broadcasted_byte_st, [1,320])
			state_0_broadcasted_bytes_list.append( state_0_broadcasted_byte_st )
                                        	             	
		state_1_broadcasted_st = g.concat( state_1_broadcasted_bytes_list )
		state_1_broadcasted_st = g.reinterpret(state_1_broadcasted_st, dtype=g.uint32 )

		state_0_broadcasted_st = g.concat( state_0_broadcasted_bytes_list )
		state_0_broadcasted_st = g.reinterpret(state_0_broadcasted_st, dtype=g.uint32 )	
		
		return state_1_broadcasted_st, state_0_broadcasted_st
	
		


	
	def gate_SS_program(self):
		"""
		Function implementing the SS gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_SS"
		
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

			
			# Creates a shared tensor to reuse the memory allocation made by program pgm2

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
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

			layout_real_diag = layout_gate_kernels_real + f", A1({address_offset})"
			layout_imag_diag = layout_gate_kernels_imag + f", A1({address_offset})"		
			layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
			layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
			storage_req_real_diag  = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag  = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )

			control_state_1_broadcasted_shared = g.shared_memory_tensor(self.control_state_1_broadcasted_mt, name="control_state_1_broadcasted_shared")
			control_state_0_broadcasted_shared = g.shared_memory_tensor(self.control_state_0_broadcasted_mt, name="control_state_0_broadcasted_shared")

			ones_shared = g.shared_memory_tensor(self.ones_mt, name="ones_shared")
			zeros_shared = g.shared_memory_tensor(self.zeros_mt, name="zeros_shared")
			ones_zeros_mt = g.concat([ones_shared, zeros_shared], dim=0, name="ones_zeros_mt")



			with g.ResourceScope(name=f"prepare_state_pair_scope_SS", is_buffered=True, time=0) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
					
					
				if ( self.mxm_permutation ): # if MXM is used to generate qubit pairs

					permuted_state_real_mt, permuted_state_imag_mt = self.MXM_permutation_of_S_qubits(State_input_real_shared, 
                                                                                                                          State_input_imag_shared, 
                                                                                                                          State_input_real_copy_shared, 
                                                                                                                          State_input_imag_copy_shared)

					# we can only start the prepare_gate_kernel_scope after the weights have been install because of stream limitations
					tm_pkg = 25

				else: # if permutor is used to generate qubit pairs

					permuted_state_real_mt, permuted_state_imag_mt = self.permutation_of_S_qubits(State_input_real_shared, 
                                                                                                                      State_input_imag_shared, 
                                                                                                                          State_input_real_copy_shared, 
                                                                                                                          State_input_imag_copy_shared)

					tm_pkg = 0


				
			#################################################################################
			# component to perform the gate operation
				
			with g.ResourceScope(name=f"prepare_gate_kernel_scope_SS", is_buffered=True, time=tm_pkg) as prepare_gate_kernel_scope :

		
				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

				# create_reusable_alu_requests
				alu_requests = []
				alu_requests.append(  g.tensor.create_alu_request( [0], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [8], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [5], 1 ) )
				alu_requests.append(  g.tensor.create_alu_request( [2], 1 ) )	
					
				with g.ResourceScope(name=f"prepare_filtered_gate_elements_scope_SS", is_buffered=True, time=0) as prepare_filtered_gate_elements_scope :

					gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt, name="gate_counter_gate_filtering_selector_shared") 


					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors_SS() ##################################

					control_state_0_broadcasted_st, control_state_1_broadcasted_st = self.gather_control_qubit_selector_SS()
					#control_state_1_broadcasted_st = control_state_1_broadcasted_shared.read( streams=g.SG4[2] )
					#control_state_0_broadcasted_st = control_state_0_broadcasted_shared.read( streams=g.SG4[5] )
					ones_zeros_st = ones_zeros_mt.read( streams=g.SG4[3])


					
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
															output_stream_offset = 4*6, \
															time = 0)
														
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_11_st = self.gather_gate_elements(gate_counter_gate_element_selector_11_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 12 )
					
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )



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
							
					
					gate_diag_real_st, gate_diag_imag_st	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_real_st.write(name=f"gate_diag_real", storage_req=storage_req_real_diag)
					gate_diag_imag_mt 	= gate_diag_imag_st.write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)

					g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)



					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors_SS()

					control_state_1_broadcasted_st = self.gather_control_qubit_1_selector_SS() ##################################


					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_01_st = self.gather_gate_elements(gate_counter_gate_element_selector_01_shared, \
															g.SG1_W[30], \
															self.gate_kernels_broadcasted_real_mt, \
															self.gate_kernels_broadcasted_imag_mt, \
															output_stream_offset = 4*6 )
															
					# real and imaginary parts of the gate element concatenated into a unified tensor								
					gate_10_st = self.gather_gate_elements(gate_counter_gate_element_selector_10_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 0, \
															time = 8 )

				
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )


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


					g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt], [gate_off_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_copy_mt], [gate_off_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

				
				#################################################################################
			
			with g.ResourceScope(name=f"apply_gate_scope_SS", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :


				State_input_imag_shared, State_input_imag_shared = self.VXM_S_round_trip(  State_input_real_shared, 
                                                                                                      State_input_imag_shared, 
                                                                                                      permuted_state_real_mt, 
                                                                                                      permuted_state_imag_mt, 
                                                                                                      gate_diag_real_mt, 
                                                                                                      gate_diag_imag_mt, 
                                                                                                      gate_off_diag_real_mt, 
                                                                                                      gate_off_diag_imag_mt)
                                                                                                      
                                                                                                      
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters_for_S_gates()



			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
