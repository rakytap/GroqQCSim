from gate_S_base import Gate_S_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_S_base):
	"""
	A class used to implement the gate operation on qubits S with no control qubits


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
	
	def gate_S_program(self):
		"""
		Function implementing the S gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_S"
		
		with self.pgm_pkg.create_program_context(name) as pgm4:


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


			state_1_broadcasted_list   = g.split(state_1_broadcasted_shared, dim=0, num_splits=gate_count)
			state_0_broadcasted_list   = g.split(state_0_broadcasted_shared, dim=0, num_splits=gate_count)

			
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
		

					
							
			
				
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, time=0) as prepare_state_pair_scope :

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
				
			with g.ResourceScope(name=f"prepare_gate_kernel_scope", is_buffered=True, time=tm_pkg) as prepare_gate_kernel_scope :

		
				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
					
					
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
					
				with g.ResourceScope(name=f"prepare_filtered_gate_elements_scope", is_buffered=True,  time=0) as prepare_fileterd_gate_elements_scope :

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
												output_streams=[g.SG1[8+slice_idx]] )
						state_1_broadcasted_byte_st = g.reshape( state_1_broadcasted_byte_st, [1,320])
						state_1_broadcasted_bytes_list.append( state_1_broadcasted_byte_st )

						state_0_broadcasted_byte_st = g.mem_gather(state_0_broadcasted_shared_list[slice_idx], 
												gate_counter_gate_filtering_selector_st, 
												output_streams=[g.SG1[slice_idx]] )
						state_0_broadcasted_byte_st = g.reshape( state_0_broadcasted_byte_st, [1,320])
						state_0_broadcasted_bytes_list.append( state_0_broadcasted_byte_st )
                                        	             	
					state_1_broadcasted_st = g.concat( state_1_broadcasted_bytes_list )
					state_1_broadcasted_st = g.reinterpret(state_1_broadcasted_st, dtype=g.uint32 )

					state_0_broadcasted_st = g.concat( state_0_broadcasted_bytes_list )
					state_0_broadcasted_st = g.reinterpret(state_0_broadcasted_st, dtype=g.uint32 )


					#state_1_broadcasted_st = state_1_broadcasted_list[0].read( streams=g.SG4[2] )
					#state_0_broadcasted_st = state_0_broadcasted_list[0].read( streams=g.SG4[0] )

					
				
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[1], time=0 )
					gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict["0_00"].read( streams=g.SG4[1], time=0 )
					gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[12], output_streams=g.SG4[0] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )




					# real and imaginary parts of the state concatenated into a unified tensor
					gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict["0_11"].read( streams=g.SG4[3])#, time=2 )
					gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["0_11"].read( streams=g.SG4[3])#, time=2 )
					gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[4], output_streams=g.SG4[2] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

				
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=[1], output_streams=g.SG4[3])
				
							
					
					gate_diag_st_list	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_st_list[0].write(name=f"gate_diag_real", storage_req=storage_req_real_diag)
					gate_diag_imag_mt 	= gate_diag_st_list[1].write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)
					



					gate_counter_gate_filtering_selector_st     = gate_counter_gate_filtering_selector_shared.read( streams=g.SG1[0] )

					state_1_broadcasted_bytes_list = []
					state_0_broadcasted_bytes_list = []

					for slice_idx in range(4):
						state_1_broadcasted_byte_st = g.mem_gather(state_1_broadcasted_shared_list[slice_idx], 
												gate_counter_gate_filtering_selector_st, 
												output_streams=[g.SG1[8+slice_idx]] )
						state_1_broadcasted_byte_st = g.reshape( state_1_broadcasted_byte_st, [1,320])
						state_1_broadcasted_bytes_list.append( state_1_broadcasted_byte_st )

						state_0_broadcasted_byte_st = g.mem_gather(state_0_broadcasted_shared_list[slice_idx], 
												gate_counter_gate_filtering_selector_st, 
												output_streams=[g.SG1[slice_idx]] )
						state_0_broadcasted_byte_st = g.reshape( state_0_broadcasted_byte_st, [1,320])
						state_0_broadcasted_bytes_list.append( state_0_broadcasted_byte_st )
                                        	             	
					state_1_broadcasted_st = g.concat( state_1_broadcasted_bytes_list )
					state_1_broadcasted_st = g.reinterpret(state_1_broadcasted_st, dtype=g.uint32 )

					state_0_broadcasted_st = g.concat( state_0_broadcasted_bytes_list )
					state_0_broadcasted_st = g.reinterpret(state_0_broadcasted_st, dtype=g.uint32 )

					#state_1_broadcasted_st = state_1_broadcasted_list[0].read( streams=g.SG4[2] )
					#state_0_broadcasted_st = state_0_broadcasted_list[0].read( streams=g.SG4[0] )
				
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_01_real_st = gate_kernels_broadcasted_real_shared_dict["0_01"].read( streams=g.SG4[1])#, time=5 )
					gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict["0_01"].read( streams=g.SG4[1])#, time=5 )
					gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[14], output_streams=g.SG4[0] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )

					# real and imaginary parts of the state concatenated into a unified tensor
					gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict["0_10"].read( streams=g.SG4[3], time=7 )
					gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["0_10"].read( streams=g.SG4[3], time=7 )
					gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[6], output_streams=g.SG4[2] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
					
					# combine the filtered gate element vectors
					gate_off_diag_st = g.add(gate_01_st, gate_10_st, alus=[3], output_streams=g.SG4[3])
				
							
					
					gate_off_diag_st_list	= g.split( gate_off_diag_st, num_splits=2, dim=0 )
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag)
		
					
				
				#################################################################################
			
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :

				State_input_imag_shared, State_input_imag_shared = self.VXM_S_round_trip(  State_input_real_shared, 
                                                                                                      State_input_imag_shared, 
                                                                                                      permuted_state_real_mt, 
                                                                                                      permuted_state_imag_mt, 
                                                                                                      gate_diag_real_mt, 
                                                                                                      gate_diag_imag_mt, 
                                                                                                      gate_off_diag_real_mt, 
                                                                                                      gate_off_diag_imag_mt)
                                                                                                      
			                                                                           
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				constant_1_address_offset_shared = g.shared_memory_tensor(self.constant_1_address_offset_mt, name="constant_1_address_offset_shared" )
				
				#### increase the gate counter for permute map selector ####
				alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
				
				constant_1_address_offset_st     = constant_1_address_offset_shared.read(streams=[g.SG1_E[0]])				

				gate_counter_perm_map_selector_shared = g.shared_memory_tensor(self.gate_counter_perm_map_selector_mt, name="gate_counter_perm_map_selector_shared" )
				gate_counter_perm_map_selector_st  = gate_counter_perm_map_selector_shared.read(streams=[g.SG1_E[4]])
				
				# offset the memory address by 1
				gate_counter_perm_map_selector_st = g.add(constant_1_address_offset_st, gate_counter_perm_map_selector_st, alus=alu_rq_12, output_streams=g.SG4_W[1])
				
				# save the offsetted memory address
				gate_counter_perm_map_selector_mt = gate_counter_perm_map_selector_st.write(name=f"gate_counter_perm_map_selector_mt", \
													 storage_req=gate_counter_perm_map_selector_shared.storage_request, 
													 time=0, program_output=True)
				

				#### increase the gate counter for gate filtering selector selector ####
				alu_rq_15 = g.tensor.create_alu_request( alus=[15] )			
			
				
				constant_1_address_offset_st     = constant_1_address_offset_shared.read(streams=[g.SG1_E[0]])				

				gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt,   \
												   	name="gate_counter_gate_filtering_selector_shared") 
				gate_counter_gate_filtering_selector_st  = gate_counter_gate_filtering_selector_shared.read(streams=[g.SG1_W[4]])
				gate_counter_gate_filtering_selector_st.write(name='test2', layout='H1,-1,S1', program_output=True)
				
				# offset the memory address by 1
				gate_counter_gate_filtering_selector_st = g.add(constant_1_address_offset_st, gate_counter_gate_filtering_selector_st, alus=alu_rq_15, output_streams=g.SG4_E[6])
				
				# save the offsetted memory address
				gate_counter_gate_filtering_selector_mt = gate_counter_gate_filtering_selector_st.write(name=f"gate_counter_gate_filtering_selector_mt", \
																storage_req=gate_counter_gate_filtering_selector_shared.storage_request, 
																time=1, program_output=True)

				

				
			self.pgm_pkg.compile_program_context(pgm4)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
