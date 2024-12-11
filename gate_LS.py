from gate_L_base import Gate_L_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_L_base):
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
		

		self.alu_req_7  = g.tensor.create_alu_request( [5], 1 )
		self.alu_req_10 = g.tensor.create_alu_request( [10], 1 )		
		self.alu_req_11 = g.tensor.create_alu_request( [8], 1 )
		
		
	def gather_control_qubit_selector_LS(self, save=False ):
	
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
		#return control_state_1_broadcasted_st
		
		
	def filter_gate_element( self, gate_counter_gate_element_selector_mt, filtering_element_mt, address_offset, time, label ):
	
	
		control_state_0_broadcasted_st, control_state_1_broadcasted_st = self.gather_control_qubit_selector_LS()
		
		filtering_element_st = filtering_element_mt.read( streams = g.SG4_W[6] )
			
				
		# we need to 'normalize' the control_state_0_broadcasted tensor
		control_state_0_broadcasted_st = g.reinterpret( control_state_0_broadcasted_st, g.uint32 )
		filtering_element_st = g.bitwise_and(control_state_0_broadcasted_st, filtering_element_st, alus=self.alu_req_11, output_streams=g.SG4[5])
		filtering_element_st = g.reinterpret( filtering_element_st, g.float32)
								
				
		gate_element_st = self.gather_gate_elements(gate_counter_gate_element_selector_mt, \
													g.SG1_W[30], \
													self.gate_kernels_broadcasted_real_mt, \
													self.gate_kernels_broadcasted_imag_mt, \
													output_stream_offset = 4*3, \
													time = time)
															
		# real and imaginary parts of the state concatenated into a unified tensor
		gate_element_st = g.reinterpret( gate_element_st, g.uint32 )
		gate_element_st = g.bitwise_and( gate_element_st, control_state_1_broadcasted_st, alus=self.alu_req_7, output_streams=g.SG4[4] )
		gate_element_st = g.reinterpret( gate_element_st, g.float32 )
				
				
		gate_element_st = g.add( gate_element_st, filtering_element_st, alus=self.alu_req_10, output_streams=g.SG4_E[4] )

							
		# addresses from gate_count*4 to gate_count*5    on EAST hemispheres are occupied with the shifted (and still packed) gate kernels  -- this is temporary used.
		#layout_gate_kernels_real_copy 	="H1(E), S4(0-3)"
		#layout_gate_kernels_imag_copy	="H1(E), S4(4-7)"
		#layout_gate_kernels_real	="H1(E), S4(8-11)"
		#layout_gate_kernels_imag	="H1(E), S4(12-15)"				
				
		gate_element_st_list = gate_element_st.split( dim=0, num_splits=2 )	
		gate_element_real_filtered_mt      = gate_element_st_list[0].write(name='gate_real_'+label+ '_filtered', layout=layout_gate_kernels_real + f', A1({address_offset})' )
		gate_element_imag_filtered_mt      = gate_element_st_list[1].write(name='gate_imag_'+label+ '_filtered', layout=layout_gate_kernels_imag + f', A1({address_offset})' )
		gate_element_real_filtered_copy_mt = gate_element_st_list[0].write(name='gate_real_'+label+ '_copy_filtered', layout=layout_gate_kernels_real_copy + f', A1({address_offset})' )
		gate_element_imag_filtered_copy_mt = gate_element_st_list[1].write(name='gate_imag_'+label+ '_copy_filtered', layout=layout_gate_kernels_imag_copy + f', A1({address_offset})' )
				
				
		
		
		return gate_element_real_filtered_mt, gate_element_imag_filtered_mt, gate_element_real_filtered_copy_mt, gate_element_imag_filtered_copy_mt
		

	def gather_gate_elements_LS(self):
			  
		gate_counter_gate_element_selector_00_st = self.gate_counter_gate_element_selector_00_mt.read( streams=g.SG1[1], time=0 )
		gate_counter_gate_element_selector_11_st = self.gate_counter_gate_element_selector_11_mt.read( streams=g.SG1[1], time=3 )	
		gate_counter_gate_element_selector_01_st = self.gate_counter_gate_element_selector_01_mt.read( streams=g.SG1[1], time=6 )
		gate_counter_gate_element_selector_10_st = self.gate_counter_gate_element_selector_10_mt.read( streams=g.SG1[1], time=9 )				
		
		gate_counter_gate_element_selector_st = g.stack( [gate_counter_gate_element_selector_00_st, 
								  gate_counter_gate_element_selector_11_st,
								  gate_counter_gate_element_selector_01_st,
								  gate_counter_gate_element_selector_10_st], dim=0 )
								  
								  
		gate_counter_gate_element_selector_00_st = self.gate_counter_gate_element_selector_00_mt.read( streams=g.SG1[0], time=1 )
		gate_counter_gate_element_selector_11_st = self.gate_counter_gate_element_selector_11_mt.read( streams=g.SG1[0], time=4 )	
		gate_counter_gate_element_selector_01_st = self.gate_counter_gate_element_selector_01_mt.read( streams=g.SG1[0], time=7 )
		gate_counter_gate_element_selector_10_st = self.gate_counter_gate_element_selector_10_mt.read( streams=g.SG1[0], time=10 )				
		
		gate_counter_gate_element_selector_st_for_imag = g.stack( [gate_counter_gate_element_selector_00_st, 
								  gate_counter_gate_element_selector_11_st,
								  gate_counter_gate_element_selector_01_st,
								  gate_counter_gate_element_selector_10_st], dim=0 )
		
		
		
						
		gate_element_real_mt_8      = g.reinterpret( self.gate_kernels_broadcasted_real_mt, dtype=g.uint8 )
		gate_element_real_byte_list = g.split( gate_element_real_mt_8, dim=1, num_splits=4 )
			
		gate_element_imag_mt_8      = g.reinterpret( self.gate_kernels_broadcasted_imag_mt, dtype=g.uint8 )
		gate_element_imag_byte_list = g.split( gate_element_imag_mt_8, dim=1, num_splits=4 )


		gate_element_real_00_list = []
		gate_element_real_11_list = []
		gate_element_real_list = []		
		gate_element_imag_list = []
		for slice_idx in range(4):
		
			output_stream_offset = 3*4
			gate_element_real_byte_st = g.mem_gather(gate_element_real_byte_list[slice_idx], 
											gate_counter_gate_element_selector_st, 
											output_streams=[g.SG1[output_stream_offset+slice_idx]] )

			gate_element_real_byte_st = g.reshape( gate_element_real_byte_st, [1,4,320])
			gate_element_real_list.append( gate_element_real_byte_st )
			
			
			#output_stream_offset = 6*4
			gate_element_imag_byte_st = g.mem_gather(gate_element_imag_byte_list[slice_idx], 
										gate_counter_gate_element_selector_st_for_imag, 
										output_streams=[g.SG1[output_stream_offset+slice_idx]] )

			gate_element_imag_byte_st = g.reshape( gate_element_imag_byte_st, [1,4,320])
			gate_element_imag_list.append( gate_element_imag_byte_st )	
			
			
		
		gate_element_real_st = g.concat( gate_element_real_list )		
		gate_element_real_st = g.transpose( gate_element_real_st, [1,0,2] )	
		gate_element_real_st = g.reinterpret(gate_element_real_st, dtype=g.float32 )
				
									
		gate_element_imag_st = g.concat( gate_element_imag_list )		
		gate_element_imag_st = g.transpose( gate_element_imag_st, [1,0,2] )	
		gate_element_imag_st = g.reinterpret(gate_element_imag_st, dtype=g.float32 )
		
		gate_elements_st = g.stack( [gate_element_real_st, gate_element_imag_st], dim=0 )
		gate_elements_st = g.transpose( gate_elements_st, [1,0,2,3] )
		
		return gate_elements_st
		
		
		
	def gate_LS_program(self):
		"""
		Function implementing the LS gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_LS"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:
		
			'''
			with g.ResourceScope(name=f"copy_complex_one", is_buffered=True, predecessors=[], time=0) as copy_complex_one :
			
				ones_st = self.ones_mt.read( streams=g.SG4_W[3], time=0 )
				ones_mt = ones_st.write( name="one", layout=layout_gate_kernels_real_copy + f', A1({4*gate_count+4})' )


				zeros_st = self.zeros_mt.read( streams=g.SG4_W[3], time=1 )
				zeros_mt = zeros_st.write( name="zero", layout=layout_gate_kernels_real_copy + f', A1({4*gate_count+5})' )
				
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt], [ones_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt, ones_mt], [zeros_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)	
			
			with g.ResourceScope(name=f"filter_gate_elements_scope2", is_buffered=True, predecessors=[copy_complex_one], time=None) as filter_gate_elements_scope2 :

				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 		
			

				
				complex_one_mt  = g.reshape( g.stack( [ones_mt, zeros_mt], dim = 0 ), [2,320] )
				complex_zero_mt = g.reshape( g.stack( [zeros_mt, zeros_mt], dim = 0 ), [2,320] ) 
				
				filtering_elements_mt = g.stack( [complex_one_mt, complex_one_mt, complex_zero_mt, complex_zero_mt], dim=0 )
				filtering_elements_st = filtering_elements_mt.read( streams=g.SG4_W[6] )#, time=12 )	

				control_state_0_broadcasted_st, control_state_1_broadcasted_st = self.gather_control_qubit_selector_LS()
							
				# we need to filter also the filtering_element_st tensor
				control_state_0_broadcasted_st = g.reinterpret( control_state_0_broadcasted_st, g.uint32 )
				filtering_elements_st = g.bitwise_and(control_state_0_broadcasted_st, filtering_elements_st, alus=[11], output_streams=g.SG4_W[6])
				filtering_elements_st = g.reinterpret( filtering_elements_st, g.float32)
			

				#filtering_elements_st.write(name="tttt", layout='H1(W), -1, S4', time=None )	
				#control_state_1_broadcasted_st.write(name="rrr", layout='H1(W), -1, S4')
				#control_state_0_broadcasted_st.write(name="rrr2", layout='H1(W), -1, S4', time = 5)										
				
				gate_elements_st = self.gather_gate_elements_LS()
				

				
				# real and imaginary parts of the state concatenated into a unified tensor
				gate_elements_st = g.reinterpret( gate_elements_st, g.uint32 )
				gate_elements_st = g.bitwise_and( gate_elements_st, control_state_1_broadcasted_st, alus=[5], output_streams=g.SG4_W[5] )
				gate_elements_st = g.reinterpret( gate_elements_st, g.float32 )
				
			
				
				
				gate_elements_st = g.add( gate_elements_st, filtering_elements_st, alus=[8], output_streams=g.SG4_E[5] )
								
				gate_elements_st_list = g.split( gate_elements_st, dim=1, num_splits=2 )
				gate_elements_real_st = gate_elements_st_list[0]
				gate_elements_imag_st = gate_elements_st_list[1]				
																
				
				gate_elements_filtered_real_mt_copy = gate_elements_real_st.write(name="tttt2", layout=f'H1(E), A4({4*gate_count}-{4*gate_count+3}), S4(0-3)')								
				gate_elements_filtered_real_mt      = gate_elements_real_st.write(name="tttt3", layout=f'H1(E), A4({4*gate_count}-{4*gate_count+3}), S4(8-11)' )				
				gate_elements_filtered_imag_mt_copy = gate_elements_imag_st.write(name="tttt4", layout=f'H1(E), A4({4*gate_count}-{4*gate_count+3}), S4(4-7)')								
				gate_elements_filtered_imag_mt      = gate_elements_imag_st.write(name="tttt5", layout=f'H1(E), A4({4*gate_count}-{4*gate_count+3}), S4(12-15)' )
				
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt, zeros_mt, ones_mt], [gate_elements_filtered_real_mt_copy], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_elements_filtered_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_copy_mt], [gate_elements_filtered_imag_mt_copy], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_elements_filtered_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
				
				gate_elements_filtered_real_mt_copy_list = g.split( gate_elements_filtered_real_mt_copy, dim=0, num_splits=4 )
				gate_elements_filtered_real_mt_list      = g.split( gate_elements_filtered_real_mt, dim=0, num_splits=4 )
				gate_elements_filtered_imag_mt_copy_list = g.split( gate_elements_filtered_imag_mt_copy, dim=0, num_splits=4 )				
				gate_elements_filtered_imag_mt_list      = g.split( gate_elements_filtered_imag_mt, dim=0, num_splits=4 )	
				
				
				gate_00_real_filtered_mt  = gate_elements_filtered_real_mt_list[0]		
				gate_10_real_filtered_mt  = gate_elements_filtered_real_mt_list[3]				
				
				gate_00_imag_filtered_mt  = gate_elements_filtered_imag_mt_list[0]		
				gate_10_imag_filtered_mt  = gate_elements_filtered_imag_mt_list[3]	
				
				gate_11_real_filtered_copy_mt  = gate_elements_filtered_real_mt_copy_list[1]		
				gate_01_real_filtered_copy_mt  = gate_elements_filtered_real_mt_copy_list[2]				
				
				gate_11_imag_filtered_copy_mt  = gate_elements_filtered_imag_mt_copy_list[1]		
				gate_01_imag_filtered_copy_mt  = gate_elements_filtered_imag_mt_copy_list[2]	

				#gate_element_imag_st.write(name="tttt3", layout='H1(W), -1, S4' )						
			'''
			
			with g.ResourceScope(name=f"filter_gate_elements_scope", is_buffered=True, predecessors=[], time=0) as filter_gate_elements_scope :

				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

								
				complex_one_mt  = g.stack( [self.ones_mt, self.zeros_mt], dim = 0 )
				complex_zero_mt = g.stack( [self.zeros_mt, self.zeros_mt], dim = 0 ) 
				
				gate_00_real_filtered_mt, gate_00_imag_filtered_mt, gate_00_real_filtered_copy_mt, gate_00_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_00_mt, complex_one_mt, 200, 0, '00' )
									
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_00_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_00_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)	
				
				
				gate_11_real_filtered_mt, gate_11_imag_filtered_mt, gate_11_real_filtered_copy_mt, gate_11_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_11_mt, complex_one_mt, 200+3, 6, '11' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt], [gate_11_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt], [gate_11_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt], [gate_11_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt], [gate_11_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
													
									
				gate_10_real_filtered_mt, gate_10_imag_filtered_mt, gate_10_real_filtered_copy_mt, gate_10_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_10_mt, complex_zero_mt, 200+1, 12, '10' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt, gate_11_real_filtered_mt], [gate_10_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt, gate_11_imag_filtered_mt], [gate_10_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt, gate_11_real_filtered_copy_mt], [gate_10_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt, gate_11_imag_filtered_copy_mt], [gate_10_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
				
				gate_01_real_filtered_mt, gate_01_imag_filtered_mt, gate_01_real_filtered_copy_mt, gate_01_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_01_mt, complex_zero_mt, 200+2, 18, '10' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt, gate_11_real_filtered_mt, gate_10_real_filtered_mt], [gate_01_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt, gate_11_imag_filtered_mt, gate_10_imag_filtered_mt], [gate_01_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt, gate_11_real_filtered_copy_mt, gate_10_real_filtered_copy_mt], [gate_01_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt, gate_11_imag_filtered_copy_mt, gate_10_imag_filtered_copy_mt], [gate_01_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

			
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[filter_gate_elements_scope], time=None) as apply_gate_scope :


				#index_pairs = [ [x, x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8)) ]

				read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st = self.select_target_qbit_read_map_west()
				
				# readout quantum state to be stransformed
				Psi_transformed_real_state_0_st, Psi_transformed_imag_state_0_st, Psi_transformed_real_state_1_st, Psi_transformed_imag_state_1_st  = \
							 self.read_out_state_2_transform( read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st)

					
				# readout gate elements	
				element_num = read_out_qubit_0_memory_tensor_st.shape[0]
				
				gate_element_first_column_real_list = []
				gate_element_first_column_imag_list = []				
				gate_element_second_column_real_list = []
				gate_element_second_column_imag_list = []				
				for idx in range(element_num//4):
				
					gate_element_first_column_real_list.append( gate_00_real_filtered_mt ) 
					gate_element_first_column_real_list.append( gate_00_real_filtered_mt ) 
					gate_element_first_column_real_list.append( gate_10_real_filtered_mt ) 					
					gate_element_first_column_real_list.append( gate_10_real_filtered_mt )
					
					gate_element_first_column_imag_list.append( gate_00_imag_filtered_mt ) 
					gate_element_first_column_imag_list.append( gate_00_imag_filtered_mt ) 
					gate_element_first_column_imag_list.append( gate_10_imag_filtered_mt ) 					
					gate_element_first_column_imag_list.append( gate_10_imag_filtered_mt ) 
					
					gate_element_second_column_real_list.append( gate_01_real_filtered_copy_mt ) 
					gate_element_second_column_real_list.append( gate_01_real_filtered_copy_mt ) 
					gate_element_second_column_real_list.append( gate_11_real_filtered_copy_mt ) 					
					gate_element_second_column_real_list.append( gate_11_real_filtered_copy_mt )
					
					gate_element_second_column_imag_list.append( gate_01_imag_filtered_copy_mt ) 
					gate_element_second_column_imag_list.append( gate_01_imag_filtered_copy_mt ) 
					gate_element_second_column_imag_list.append( gate_11_imag_filtered_copy_mt ) 					
					gate_element_second_column_imag_list.append( gate_11_imag_filtered_copy_mt ) 										 
				
				
				
				gate_element_first_column_real_mt = g.stack( gate_element_first_column_real_list, dim=0 )
				gate_element_first_column_imag_mt = g.stack( gate_element_first_column_imag_list, dim=0 )
				gate_element_second_column_real_mt = g.stack( gate_element_second_column_real_list, dim=0 )
				gate_element_second_column_imag_mt = g.stack( gate_element_second_column_imag_list, dim=0 )		
												
				
				gate_element_real_st = gate_element_first_column_real_mt.read( streams = g.SG4[7] )
				gate_element_imag_st = gate_element_first_column_imag_mt.read( streams = g.SG4[3] )
				gate_element_real_copy_st = gate_element_second_column_real_mt.read( streams = g.SG4[5] )
				gate_element_imag_copy_st = gate_element_second_column_imag_mt.read( streams = g.SG4[6] )												
						
				#gate_element_real_st, gate_element_imag_st, gate_element_real_copy_st, gate_element_imag_copy_st = \
				#			self.read_out_gate_elements( element_num )	
						
				
				
				

				# calculate the VXM round trip                            
				Psi_transformed_real_st, Psi_transformed_imag_st = self.two_complex_dot_product( 
												Psi_transformed_real_state_0_st,#pairs0[0], 
												Psi_transformed_imag_state_0_st,#pairs0[1], 
												Psi_transformed_real_state_1_st,#pairs1[0], 
												Psi_transformed_imag_state_1_st,#pairs1[1], 
												gate_element_real_st, 
												gate_element_imag_st, 
												gate_element_real_copy_st, 
												gate_element_imag_copy_st)
												
				
				
				# export the calculated state into memory and set program output
				target_qubit_L_write_map_st = self.select_target_qbit_write_map_east()
				self.write_tranformed_states_2_mem( Psi_transformed_real_st, Psi_transformed_imag_st, target_qubit_L_write_map_st )
				
				
				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
