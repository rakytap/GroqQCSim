from gate_8_base import Gate_8_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_8_base):
	"""
	A class used to implement the gate operation on the target qubit 8 with no control qubits


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
		

		self.target_qubit = 8


	def read_out_gate_elements_8S( self, element_num ):
	
		"""
		Function used to read out the the gate kernel elements.
		
		
		Parameters
		----------
		element_num : int
			The number of the required gate elements to perform the unitary transformation on all of the state vector elements
			
			
			
		Return
		------
		Returns with the stream tensors of the extracted gate elements			
			
		g_r_0 : StreamTensor
			The real part of the first column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- second row
			 -- second row -- .... sequence
		g_i_0 : StreamTensor
			The imaginary part of the first column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- 
			second row -- second row -- .... sequence
		g_r_1 : StreamTensor
			The real part of the second column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- s
			econd row -- second row -- .... sequence
		g_i_1 : StreamTensor
			The imaginary part of the second column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- 
			second row -- second row -- .... sequence  		
		
		"""
	
	
		gate_element_counter_first_column_list  = []
		gate_element_counter_second_column_list = []
		for idx in range(element_num//2):
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_00_mt)
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_10_mt )

			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_01_mt )
			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_11_mt )
					
					
		gate_element_counter_first_column_mt  = g.stack( gate_element_counter_first_column_list, dim=0 )
		gate_element_counter_first_column_mt  = gate_element_counter_first_column_mt.reshape( element_num, 320 )
		gate_element_counter_second_column_mt = g.stack( gate_element_counter_second_column_list, dim=0 )
		gate_element_counter_second_column_mt = gate_element_counter_second_column_mt.reshape( element_num, 320 )

			
		gate_element_counter_first_column_st  = gate_element_counter_first_column_mt.read( streams=g.SG1_W[17] )		
		#gate_element_counter_second_column_st = gate_element_counter_second_column_mt.read( streams=g.SG1_W[18] )	
				
				
		gate_kernels_broadcasted_real_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_mt, dtype=g.uint8 )
		gate_element_real_byte_list        = g.split( gate_kernels_broadcasted_real_mt_8, dim=1, num_splits=4 )
				
		gate_kernels_broadcasted_imag_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_mt, dtype=g.uint8 )
		gate_element_imag_byte_list        = g.split( gate_kernels_broadcasted_imag_mt_8, dim=1, num_splits=4 )

		gate_kernels_broadcasted_real_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_copy_mt, dtype=g.uint8 )
		gate_element_real_copy_byte_list        = g.split( gate_kernels_broadcasted_real_copy_mt_8, dim=1, num_splits=4 )

		gate_kernels_broadcasted_imag_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_copy_mt, dtype=g.uint8 )
		gate_element_imag_copy_byte_list        = g.split( gate_kernels_broadcasted_imag_copy_mt_8, dim=1, num_splits=4 )
				
				
		gate_element_real_list = []
		gate_element_imag_list = []
		gate_element_real_copy_list = []
		#gate_element_imag_copy_list = []
		for slice_idx in range(4):

			#### first column real part
			output_stream_offset = 7*4
			gate_element_real_byte_st = g.mem_gather(gate_element_real_byte_list[slice_idx], 
											gate_element_counter_first_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

			gate_element_real_byte_st = g.reshape( gate_element_real_byte_st, [element_num,1,320])
			gate_element_real_list.append( gate_element_real_byte_st )
					




		gate_element_real_st      = g.concat( gate_element_real_list, dim=1 )
		gate_element_imag_st      = g.concat( gate_element_imag_list, dim=1 )
							
		gate_element_real_st      = g.reinterpret(gate_element_real_st, dtype=g.float32 )
		gate_element_imag_st      = g.reinterpret(gate_element_imag_st, dtype=g.float32 )		
		
		
		
		return gate_element_real_st



	
	def gate_8S_program(self):
		"""
		Function implementing the 8S gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_8S"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:


			#index_pairs = [ [x, x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8)) ]


      			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")


			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
			Psi_transformed_real_mt = State_input_real_shared
			Psi_transformed_imag_mt = State_input_imag_shared
			Psi_transformed_real_copy_mt = State_input_real_copy_shared
			Psi_transformed_imag_copy_mt = State_input_imag_copy_shared
			
			
			with g.ResourceScope(name=f"filter_gate_elements_scope", is_buffered=True, predecessors=[], time=0) as filter_gate_elements_scope :

				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

								
				complex_one_mt  = g.stack( [self.ones_mt, self.zeros_mt], dim = 0 )
				complex_zero_mt = g.stack( [self.zeros_mt, self.zeros_mt], dim = 0 ) 
				
				gate_00_real_filtered_mt, gate_00_imag_filtered_mt, gate_00_real_filtered_copy_mt, gate_00_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_00_mt, complex_one_mt, 4*gate_count, 0, '00' )
									
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_00_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_00_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)	
				
				
				gate_11_real_filtered_mt, gate_11_imag_filtered_mt, gate_11_real_filtered_copy_mt, gate_11_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_11_mt, complex_one_mt, 7*gate_count, 6, '11' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt], [gate_11_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt], [gate_11_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt], [gate_11_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt], [gate_11_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
													
									
				gate_10_real_filtered_mt, gate_10_imag_filtered_mt, gate_10_real_filtered_copy_mt, gate_10_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_10_mt, complex_zero_mt, 5*gate_count, 12, '10' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt, gate_11_real_filtered_mt], [gate_10_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt, gate_11_imag_filtered_mt], [gate_10_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt, gate_11_real_filtered_copy_mt], [gate_10_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt, gate_11_imag_filtered_copy_mt], [gate_10_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
				
				gate_01_real_filtered_mt, gate_01_imag_filtered_mt, gate_01_real_filtered_copy_mt, gate_01_imag_filtered_copy_mt = \
									self.filter_gate_element( self.gate_counter_gate_element_selector_01_mt, complex_zero_mt, 6*gate_count, 18, '10' )	
									
	
	
				g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt, gate_00_real_filtered_mt, gate_11_real_filtered_mt, gate_10_real_filtered_mt], [gate_01_real_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt, gate_00_imag_filtered_mt, gate_11_imag_filtered_mt, gate_10_imag_filtered_mt], [gate_01_imag_filtered_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_real_filtered_copy_mt, gate_11_real_filtered_copy_mt, gate_10_real_filtered_copy_mt], [gate_01_real_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([gate_00_imag_filtered_copy_mt, gate_11_imag_filtered_copy_mt, gate_10_imag_filtered_copy_mt], [gate_01_imag_filtered_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
							

			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[filter_gate_elements_scope], time=None) as apply_gate_scope :

				pairs = [ [x,x^1] for x in range(memory_addresses) if x < x^1   ]
 

				inner_dims = [x.reshape(2, matrix_size) for x in g.stack(
						[Psi_transformed_real_mt, Psi_transformed_imag_mt], dim=0).split(dim=1, num_splits=memory_addresses)]

				inner_dims_copy = [x.reshape(2, matrix_size) for x in g.stack(
						[Psi_transformed_real_copy_mt, Psi_transformed_imag_copy_mt], dim=0).split(dim=1, num_splits=memory_addresses)]

				#inner_dims (2, memory_addresses, matrix_size) -> [(2, matrix_size)]*memory_addresses 
				pairs_to_concat0_0 = []
				pairs_to_concat0_1 = []
				pairs_to_concat1_0 = []
				pairs_to_concat1_1 = []

				for i in pairs:
					pairs_to_concat0_0.append( inner_dims[ i[0] ] )
					pairs_to_concat0_1.append( inner_dims[ i[1] ])
					pairs_to_concat1_0.append( inner_dims_copy[ i[0] ] )
					pairs_to_concat1_1.append( inner_dims_copy[ i[1] ] )

				#pairs_to_concat [(2, 1<<small_qbit_num_limit)]*(memory_addresses//2)
				pairs_to_concat0_0 = g.stack(pairs_to_concat0_0, dim=0)
				pairs_to_concat0_1 = g.stack(pairs_to_concat0_1, dim=0)
				pairs_to_concat1_0 = g.stack(pairs_to_concat1_0, dim=0)
				pairs_to_concat1_1 = g.stack(pairs_to_concat1_1, dim=0)

				pairs0 = g.stack([pairs_to_concat0_0, pairs_to_concat1_0], dim=0)
				pairs0 = [x.reshape(2, memory_addresses//2, matrix_size) for x in pairs0.split(num_splits=2, dim=2)]

				pairs1 = g.stack([pairs_to_concat1_1, pairs_to_concat0_1], dim=0)
				pairs1 = [x.reshape(2, memory_addresses//2, matrix_size) for x in pairs1.split(num_splits=2, dim=2)]
			
				new_shape = (2, memory_addresses//2, 1, matrix_size)

				pairs0[0] = pairs0[0].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs0[1] = pairs0[1].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1[0] = pairs1[0].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1[1] = pairs1[1].reshape(*new_shape).transpose(1, 0, 2, 3)

				state_real_0_st = pairs0[0].read(streams=g.SG4[4]) #direction=Direction.EASTWARD
				state_imag_0_st = pairs0[1].read(streams=g.SG4[1]) #direction=Direction.EASTWARD
				state_real_1_st = pairs1[0].read(streams=g.SG4[5]) #direction=Direction.EASTWARD
				state_imag_1_st = pairs1[1].read(streams=g.SG4[7]) #direction=Direction.EASTWARD
				

				# readout gate elements	
				element_num = state_real_0_st.shape[0]*state_real_0_st.shape[1]
				
				
				
				'''	
				gate_first_column_real_st = \
							self.read_out_gate_elements_8S( element_num )	
				'''		
				gate_first_column_real_mt_list = []
				gate_first_column_imag_mt_list = []				
				gate_second_column_real_mt_list = []				
				gate_second_column_imag_mt_list = []								
				
				for idx in range( element_num//2 ):
				
					gate_first_column_real_mt_list.append( gate_00_real_filtered_mt )
					gate_first_column_real_mt_list.append( gate_10_real_filtered_mt )					
					
					gate_first_column_imag_mt_list.append( gate_00_imag_filtered_mt )
					gate_first_column_imag_mt_list.append( gate_10_imag_filtered_mt )						
					
					gate_second_column_real_mt_list.append( gate_01_real_filtered_copy_mt )
					gate_second_column_real_mt_list.append( gate_11_real_filtered_copy_mt )					
					
					gate_second_column_imag_mt_list.append( gate_01_imag_filtered_copy_mt )
					gate_second_column_imag_mt_list.append( gate_11_imag_filtered_copy_mt )
					
					
				gate_first_column_real_mt = g.stack( gate_first_column_real_mt_list, dim=0 )
				gate_first_column_imag_mt = g.stack( gate_first_column_imag_mt_list, dim=0 )				
				gate_second_column_real_mt = g.stack( gate_second_column_real_mt_list, dim=0 )				
				gate_second_column_imag_mt = g.stack( gate_second_column_imag_mt_list, dim=0 )
				
						
				gate_first_column_real_st = gate_first_column_real_mt.read( streams = g.SG4[7] )
				gate_first_column_imag_st = gate_first_column_imag_mt.read( streams = g.SG4[3] )				
				gate_second_column_real_st = gate_second_column_real_mt.read( streams = g.SG4[5] )	
				gate_second_column_imag_st = gate_second_column_imag_mt.read( streams = g.SG4[6] )
				
				
				pairs0[0], pairs0[1], pairs1[0], pairs1[1] = self.two_complex_dot_product_8(	state_real_0_st, 
														state_imag_0_st, 
														state_real_1_st, 
														state_imag_1_st, 
														gate_first_column_real_st, 
														gate_first_column_imag_st, 
														gate_second_column_real_st, 
														gate_second_column_imag_st)


				pairs_to_store0 = g.stack([pairs_to_concat0_0, pairs_to_concat0_1], dim=0)
				pairs_to_store1 = g.stack([pairs_to_concat1_0, pairs_to_concat1_1], dim=0)
				#pairs_to_store (2, memory_addresses//2, 2, matrix_size)

				pairs0_0, pairs0_1 = g.split(pairs_to_store0, num_splits=2, dim=2)
				pairs1_0, pairs1_1 = g.split(pairs_to_store1, num_splits=2, dim=2)
				#pairs (2, memory_addresses//2, 1, matrix_size)


				pairs0_0 = pairs0_0.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs0_1 = pairs0_1.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1_0 = pairs1_0.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1_1 = pairs1_1.reshape(*new_shape).transpose(1, 0, 2, 3)

  
				pairs1, pairs0 = pairs0, pairs1

				pairs0[0].write(storage_req=pairs0_0.storage_request, name="pairs0_0")
				pairs0[1].write(storage_req=pairs0_1.storage_request, name="pairs0_1")
				pairs1[0].write(storage_req=pairs1_0.storage_request, name="pairs1_0")
				pairs1[1].write(storage_req=pairs1_1.storage_request, name="pairs1_1")
				

				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
