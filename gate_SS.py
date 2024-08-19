from gate_S_base import Gate_S_base
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate(Gate_S_base):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
	
	def gate_SS_program(self):
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


			tm = 0
			# we can only start the prepare_gate_kernel_scope after the weights have been install because of stream limitations
			tm_pgk = 25
		
			
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


			with g.ResourceScope(name=f"prepare_state_pair_scope_SS", is_buffered=True, time=tm) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
					
					
				if ( self.mxm_permutation ): # if MXM is used to generate qubit pairs

					permuted_state_real_mt, permuted_state_imag_mt = self.MXM_permutation_of_S_qubits(Psi_transformed_real_mt, 
                                                                                                                          Psi_transformed_imag_mt, 
                                                                                                                          Psi_transformed_real_copy_mt, 
                                                                                                                          Psi_transformed_imag_copy_mt)

				else: # if permutor is used to generate qubit pairs

					permuted_state_real_mt, permuted_state_imag_mt = self.permutation_of_S_qubits(Psi_transformed_real_mt, 
                                                                                                                          Psi_transformed_imag_mt, 
                                                                                                                          Psi_transformed_real_copy_mt, 
                                                                                                                          Psi_transformed_imag_copy_mt)

				
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

				
				#################################################################################
			
			with g.ResourceScope(name=f"apply_gate_scope_SS", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :


				Psi_transformed_imag_mt, Psi_transformed_imag_mt = self.VXM_S_round_trip(  Psi_transformed_real_mt, 
                                                                                                      Psi_transformed_imag_mt, 
                                                                                                      permuted_state_real_mt, 
                                                                                                      permuted_state_imag_mt, 
                                                                                                      gate_diag_real_mt, 
                                                                                                      gate_diag_imag_mt, 
                                                                                                      gate_off_diag_real_mt, 
                                                                                                      gate_off_diag_imag_mt)
					
				
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
