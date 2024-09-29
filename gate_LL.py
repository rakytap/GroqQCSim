from gate_L import Gate as Gate_L
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_L):
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
		

		
		self.transposer_request_to_control_qubit_state_readout  = g.tensor.create_transposer_request( [1], 1) # request transposer on the west side
		self.transposer_request_to_control_qubit_state_write    = g.tensor.create_transposer_request( [2], 1) # request transposer on the east side
		self.distributor_request_to_control_qubit_state_readout = g.tensor.create_distributor_request( [2], 1 )  # West side	
		self.distributor_request_to_control_qubit_state_write   = g.tensor.create_distributor_request( [5], 1 )  # East side	
		

	def select_target_qbit_read_map_west_LL(self):
		"""
		Function used to read out the correct target qubit map to gather the target qubits L from the memory
		The target qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The target memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The target memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------

		Return
		------
		Returns with the stream tensor of the target qubits memory map.
		"""

		#################################
		'''
		target_qubit_0_state_read_maps_list = self.target_qubit_read_maps_L_mt.split( dim=0, num_splits=memory_addresses)
		target_qubit_1_state_read_maps_list = self.target_qubit_read_pair_maps_L_mt.split( dim=0, num_splits=memory_addresses)
		read_out_state_0_memory_tensor_list = []
		read_out_state_1_memory_tensor_list = []
		for idx in range(memory_addresses//8):

			# idx, idx+1 needed for bank alternation strategy
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 

			# idx, idx+1 needed for bank alternation strategy
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 


		read_out_qubit_0_memory_tensor_mt = g.stack( read_out_state_0_memory_tensor_list, dim=0 )
		read_out_qubit_0_memory_tensor_st = read_out_qubit_0_memory_tensor_mt.read( streams=g.SG1[0] )
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_0_memory_tensor_st, [memory_addresses//2,320])


		read_out_qubit_1_memory_tensor_mt = g.stack( read_out_state_1_memory_tensor_list, dim=0 )
		#read_out_qubit_1_memory_tensor_mt.set_program_output()
		#read_out_qubit_1_memory_tensor_mt.name = "rrrrwwww1"
		read_out_qubit_1_memory_tensor_st = read_out_qubit_1_memory_tensor_mt.read( streams=g.SG1[1] )
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_1_memory_tensor_st, [memory_addresses//2,320])
		'''
		#############################x

		control_qbit_state_1_selector_st = self.target_qubit_read_maps_LL_mt.read( streams=g.SG1[16] )
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1_W[17]] )		
		distributor_map_st                      = g.mem_gather(self.distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[17]] )
		#distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[17] )
		
		control_qbit_state_1_selector_st = g.distribute_8( control_qbit_state_1_selector_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_control_qubit_state_readout,
									map_stream_req=g.SG1[17],
									bypass8=0b11111110,
									)	
		
		control_qbit_state_1_selector_st = g.transpose_null(control_qbit_state_1_selector_st, 
					                             transposer_req=self.transposer_request_to_control_qubit_state_readout, 
					                             stream_order=[0])
					                             
					                             
		
		
		read_out_qubit_0_memory_tensor_st = g.mem_gather(self.target_qubit_read_maps_L_mt, control_qbit_state_1_selector_st, output_streams=[g.SG1[0]] )
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_0_memory_tensor_st, [memory_addresses//2,320])

		read_out_qubit_1_memory_tensor_st = g.mem_gather(self.target_qubit_read_pair_maps_L_mt, control_qbit_state_1_selector_st, output_streams=[g.SG1[1]] )
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_1_memory_tensor_st, [memory_addresses//2,320])
		###########################xx

		read_out_qubit_memory_tensor_st = g.stack( [read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st], dim=1 )
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_target_qubit_L_distributor_maps_write_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		#distributor_map_st                      = self.target_qubit_distributor_map_mt.read( streams=g.SG1[6] )
		
		read_out_qubit_memory_tensor_st = g.distribute_8( read_out_qubit_memory_tensor_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_readout,
									map_stream_req=g.SG1[6],
									bypass8=0b11111100,
									)	

		
		read_out_qubit_memory_tensor_st = g.transpose_null(read_out_qubit_memory_tensor_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_readout, 
					                             stream_order=[0,1])


		read_out_qubit_memory_tensor_st_list = read_out_qubit_memory_tensor_st.split( dim=1, num_splits=2)	
		
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[0], [memory_addresses//2,320])
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[1], [memory_addresses//2,320])


		#read_out_qubit_0_memory_tensor_st.write( name="rrww0", layout='H1(W), -1, S1', program_output=True )
		#read_out_qubit_1_memory_tensor_st.write( name="rrww1", layout='H1(W), -1, S1(43)', program_output=True )
		#distributor_map_st.write( name="distr_map", layout='H1(W), -1, S1(43)', program_output=True )
		#self.target_qubit_distributor_map_mt.set_program_output()
		

		return read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st		




	def select_target_qbit_write_map_east_LL(self):
		"""
		Function used to read out the correct target qubit map for scattering the target qubits L into the memory
		The target qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The target memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The target memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------

		Return
		------
		Returns with the stream tensor of the target qubits memory map.
		"""

		control_qbit_state_1_selector_st = self.target_qubit_write_maps_LL_mt.read( streams=g.SG1[15] )
		
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1_W[9]] )		
		distributor_map_st                      = g.mem_gather(self.distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[9]] )
		#distributor_map_st                      = self.distributor_map_mt.read( )
		
		control_qbit_state_1_selector_st = g.distribute_8( control_qbit_state_1_selector_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_control_qubit_state_write,
									map_stream_req=g.SG1[9],
									bypass8=0b01111111,
									)	
		
		
		
		control_qbit_state_1_selector_st = g.transpose_null(control_qbit_state_1_selector_st, 
					                             transposer_req=self.transposer_request_to_control_qubit_state_write, 
					                             stream_order=[15])
		
		
		
		
		
		target_qubit_write_maps_L_st = g.mem_gather(self.target_qubit_write_maps_L_mt, control_qbit_state_1_selector_st, output_streams=[g.SG1[16]] )
		#target_qubit_write_maps_L_st = self.target_qubit_write_maps_L_mt.read( streams=g.SG1[16] )

		
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_target_qubit_L_distributor_maps_write_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		#distributor_map_st                      = self.target_qubit_distributor_map_mt.read( streams=g.SG1[8] )
		
		target_qubit_write_maps_L_st = g.distribute_8( target_qubit_write_maps_L_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11111110,
									)	

		
		target_qubit_write_maps_L_st = g.transpose_null(target_qubit_write_maps_L_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[0])	

		#rr = target_qubit_write_maps_L_st.write(name="uuu", layout='H1(W), -1, S1(43)', program_output=True )

		return target_qubit_write_maps_L_st	
	
	def gate_LL_program(self):
		"""
		Function implementing the S gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_LL"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:


			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared      = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared      = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")


			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
			Psi_transformed_real_mt = State_input_real_shared
			Psi_transformed_imag_mt = State_input_imag_shared
			Psi_transformed_real_copy_mt = State_input_real_copy_shared
			Psi_transformed_imag_copy_mt = State_input_imag_copy_shared


			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :


				#index_pairs = [ [x, x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8)) ]


				# print("=============================>", pairs)
				# print(len(inner_dims))
				# print(Psi_transformed_real_mt.physical_shape, Psi_transformed_real_mt.shape)
				#Psi_transformed_mt (memory_addresses, matrix_size)


				Psi_transformed_real_splits = Psi_transformed_real_mt.split(dim=0, num_splits=memory_addresses)
				Psi_transformed_imag_splits = Psi_transformed_imag_mt.split(dim=0, num_splits=memory_addresses)

				Psi_transformed_real_copy_splits = Psi_transformed_real_copy_mt.split(dim=0, num_splits=memory_addresses)
				Psi_transformed_imag_copy_splits = Psi_transformed_imag_copy_mt.split(dim=0, num_splits=memory_addresses)


				# define state with 320 vectors instead of 256 inner splits
				Psi_real_mt_320      = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)		
				Psi_imag_mt_320      = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)
				Psi_real_copy_mt_320 = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=State_input_real_copy_shared.storage_request, initialized=False)
				Psi_imag_copy_mt_320 = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=State_input_imag_copy_shared.storage_request, initialized=False)

				
				# separating the tensors along the slice dimension (separating the bytes)
				Psi_real_mt_320_8       = g.reinterpret(Psi_real_mt_320, dtype=g.int8)
				Psi_imag_mt_320_8       = g.reinterpret(Psi_imag_mt_320, dtype=g.int8)			
				Psi_real_bytes_mt_list  = g.split(Psi_real_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_imag_bytes_mt_list  = g.split(Psi_imag_mt_320_8, dim=-2, num_splits=g.float32.size)


				Psi_real_copy_mt_320_8       = g.reinterpret(Psi_real_copy_mt_320, dtype=g.int8)
				Psi_imag_copy_mt_320_8       = g.reinterpret(Psi_imag_copy_mt_320, dtype=g.int8)			
				Psi_real_bytes_copy_mt_list  = g.split(Psi_real_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_imag_bytes_copy_mt_list  = g.split(Psi_imag_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
				

				'''
				target_qubit_0_state_read_maps_list = self.target_qubit_read_maps_L_mt.split( dim=0, num_splits=memory_addresses)
				target_qubit_1_state_read_maps_list = self.target_qubit_read_pair_maps_L_mt.split( dim=0, num_splits=memory_addresses)
				read_out_state_0_memory_tensor_list = []
				read_out_state_1_memory_tensor_list = []
				for idx in range(memory_addresses//4):

					# idx, idx+1 needed for bank alternation strategy
					read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx ] )	# for upper row of the gate kernel (output state is 0)
					read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
					read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
					read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 


					# idx, idx+1 needed for bank alternation strategy
					read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx ] )	# for upper row of the gate kernel (output state is 0)
					read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
					read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
					read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 


				#read_out_qubit_0_memory_tensor_mt = g.stack( read_out_state_0_memory_tensor_list, dim=0 )
				#read_out_qubit_0_memory_tensor_st = read_out_qubit_0_memory_tensor_mt.read( streams=g.SG1_E[0] )
				


				#read_out_qubit_1_memory_tensor_mt = g.stack( read_out_state_1_memory_tensor_list, dim=0 )
				#read_out_qubit_1_memory_tensor_st = read_out_qubit_1_memory_tensor_mt.read( streams=g.SG1_E[1] )
				'''
				# ???????????????????
				read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st = self.select_target_qbit_read_map_west_LL()
				


				Psi_transformed_real_state_0_bytes_list = []
				Psi_transformed_imag_state_0_bytes_list = []
				Psi_transformed_real_state_0_copy_bytes_list = []
				Psi_transformed_imag_state_0_copy_bytes_list = []
				#gate_element_real_copy_list = []
				#gate_element_imag_copy_list = []
				for byte_idx in range(4):
					
					################
					output_stream_offset = 4*4
					Psi_transformed_real_state_0_byte_st = g.mem_gather( Psi_real_bytes_mt_list[byte_idx], read_out_qubit_0_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
					Psi_transformed_real_state_0_byte_st = g.reshape( Psi_transformed_real_state_0_byte_st, [memory_addresses//2,1,320])
					Psi_transformed_real_state_0_bytes_list.append( Psi_transformed_real_state_0_byte_st )

					################
					output_stream_offset = 1*4
					Psi_transformed_imag_state_0_byte_st = g.mem_gather( Psi_imag_bytes_mt_list[byte_idx], read_out_qubit_0_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
					Psi_transformed_imag_state_0_byte_st = g.reshape( Psi_transformed_imag_state_0_byte_st, [memory_addresses//2,1,320])
					Psi_transformed_imag_state_0_bytes_list.append( Psi_transformed_imag_state_0_byte_st )

					################
					output_stream_offset = 5*4
					Psi_transformed_real_state_0_copy_byte_st = g.mem_gather( Psi_real_bytes_copy_mt_list[byte_idx], read_out_qubit_1_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
					Psi_transformed_real_state_0_copy_byte_st = g.reshape( Psi_transformed_real_state_0_copy_byte_st, [memory_addresses//2,1,320])
					Psi_transformed_real_state_0_copy_bytes_list.append( Psi_transformed_real_state_0_copy_byte_st )


					################
					output_stream_offset = 7*4
					Psi_transformed_imag_state_0_copy_byte_st = g.mem_gather( Psi_imag_bytes_copy_mt_list[byte_idx], read_out_qubit_1_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
					Psi_transformed_imag_state_0_copy_byte_st = g.reshape( Psi_transformed_imag_state_0_copy_byte_st, [memory_addresses//2,1,320])
					Psi_transformed_imag_state_0_copy_bytes_list.append( Psi_transformed_imag_state_0_copy_byte_st )

					
					
				Psi_transformed_real_state_0      = g.concat( Psi_transformed_real_state_0_bytes_list, dim=1 )
				Psi_transformed_imag_state_0      = g.concat( Psi_transformed_imag_state_0_bytes_list, dim=1 )
				Psi_transformed_real_state_1      = g.concat( Psi_transformed_real_state_0_copy_bytes_list, dim=1 )
				Psi_transformed_imag_state_1      = g.concat( Psi_transformed_imag_state_0_copy_bytes_list, dim=1 )
							
				Psi_transformed_real_state_0      = g.reinterpret(Psi_transformed_real_state_0, dtype=g.float32 )
				Psi_transformed_imag_state_0      = g.reinterpret(Psi_transformed_imag_state_0, dtype=g.float32 )
				Psi_transformed_real_state_1      = g.reinterpret(Psi_transformed_real_state_1, dtype=g.float32 )
				Psi_transformed_imag_state_1      = g.reinterpret(Psi_transformed_imag_state_1, dtype=g.float32 )


				###################################################

							
						
				gate_element_counter_first_column_list  = []
				gate_element_counter_second_column_list = []
				for idx in range(memory_addresses//8):
					gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_00_mt)
					gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_00_mt )
					gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_10_mt )
					gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_10_mt )	

					gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_01_mt )
					gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_01_mt )
					gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_11_mt )
					gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_11_mt )	
					
					
				gate_element_counter_first_column_mt  = g.stack( gate_element_counter_first_column_list, dim=0 )
				gate_element_counter_first_column_mt  = gate_element_counter_first_column_mt.reshape( memory_addresses//2, 320 )
				gate_element_counter_second_column_mt = g.stack( gate_element_counter_second_column_list, dim=0 )
				gate_element_counter_second_column_mt = gate_element_counter_second_column_mt.reshape( memory_addresses//2, 320 )

			
				gate_element_counter_first_column_st  = gate_element_counter_first_column_mt.read( streams=g.SG1_W[17] )		
				gate_element_counter_second_column_st = gate_element_counter_second_column_mt.read( streams=g.SG1_W[18] )	
				
				
				gate_kernels_broadcasted_real_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_mt, dtype=g.uint8 )
				gate_element_real_byte_list        = g.split( gate_kernels_broadcasted_real_mt_8, dim=1, num_splits=4 )
				
				gate_kernels_broadcasted_imag_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_mt, dtype=g.uint8 )
				gate_element_imag_byte_list        = g.split( gate_kernels_broadcasted_imag_mt_8, dim=1, num_splits=4 )

				gate_kernels_broadcasted_real_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_copy_mt, dtype=g.uint8 )
				gate_element_real_copy_byte_list        = g.split( gate_kernels_broadcasted_real_copy_mt_8, dim=1, num_splits=4 )

				gate_kernels_broadcasted_imag_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_copy_mt, dtype=g.uint8 )
				gate_element_imag_copy_byte_list        = g.split( gate_kernels_broadcasted_imag_copy_mt_8, dim=1, num_splits=4 )
				'''		
		g_r_0_st_m4 = g_r_0#.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
		g_i_0_st_m4 = g_i_0#.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
		g_r_1_st_m4 = g_r_1#.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
		g_i_1_st_m4 = g_i_1#.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
				'''
				gate_element_real_list = []
				gate_element_imag_list = []
				gate_element_real_copy_list = []
				gate_element_imag_copy_list = []
				for slice_idx in range(4):

					#### first column real part
					output_stream_offset = 7*4
					gate_element_real_byte_st = g.mem_gather(gate_element_real_byte_list[slice_idx], 
											gate_element_counter_first_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

					gate_element_real_byte_st = g.reshape( gate_element_real_byte_st, [memory_addresses//2,1,320])
					gate_element_real_list.append( gate_element_real_byte_st )
					
					#### first column imag part					
					output_stream_offset = 3*4
					gate_element_imag_byte_st = g.mem_gather(gate_element_imag_byte_list[slice_idx], 
										gate_element_counter_first_column_st, 
										output_streams=[g.SG1[output_stream_offset+slice_idx]] )

					gate_element_imag_byte_st = g.reshape( gate_element_imag_byte_st, [memory_addresses//2,1,320])
					gate_element_imag_list.append( gate_element_imag_byte_st )


					#### second column real part
					output_stream_offset = 5*4
					gate_element_real_copy_byte_st = g.mem_gather(gate_element_real_copy_byte_list[slice_idx], 
											gate_element_counter_second_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

					gate_element_real_copy_byte_st = g.reshape( gate_element_real_copy_byte_st, [memory_addresses//2,1,320])
					gate_element_real_copy_list.append( gate_element_real_copy_byte_st )


					#### second column imag part
					output_stream_offset = 6*4
					gate_element_imag_copy_byte_st = g.mem_gather(gate_element_imag_copy_byte_list[slice_idx], 
											gate_element_counter_second_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

					gate_element_imag_copy_byte_st = g.reshape( gate_element_imag_copy_byte_st, [memory_addresses//2,1,320])
					gate_element_imag_copy_list.append( gate_element_imag_copy_byte_st )
					


				gate_element_real_st      = g.concat( gate_element_real_list, dim=1 )
				gate_element_imag_st      = g.concat( gate_element_imag_list, dim=1 )
				gate_element_real_copy_st = g.concat( gate_element_real_copy_list, dim=1 )	
				gate_element_imag_copy_st = g.concat( gate_element_imag_copy_list, dim=1 )	
							
				gate_element_real_st      = g.reinterpret(gate_element_real_st, dtype=g.float32 )
				gate_element_imag_st      = g.reinterpret(gate_element_imag_st, dtype=g.float32 )				
				gate_element_real_copy_st = g.reinterpret(gate_element_real_copy_st, dtype=g.float32 )
				gate_element_imag_copy_st = g.reinterpret(gate_element_imag_copy_st, dtype=g.float32 )
				
				          
				#switch to row pair order (memory_addresses//2, matrix_size) so we can write result and copy without overwriting future needed values (cannot implicit latch gates, but saves double memory if crossing VXM or extra copy operations)

				
				

				# calculate the VXM round trip                            
				Psi_transformed_real_st, Psi_transformed_imag_st = self.two_complex_dot_product( 
												Psi_transformed_real_state_0,#pairs0[0], 
												Psi_transformed_imag_state_0,#pairs0[1], 
												Psi_transformed_real_state_1,#pairs1[0], 
												Psi_transformed_imag_state_1,#pairs1[1], 
												gate_element_real_st, 
												gate_element_imag_st, 
												gate_element_real_copy_st, 
												gate_element_imag_copy_st)
												
				
				
				
				
				
				# write the calculated states into the memory with mem scatter
				Psi_transformed_real_mt_320_8       = g.reinterpret(Psi_transformed_real_st, dtype=g.int8)
				Psi_transformed_imag_mt_320_8       = g.reinterpret(Psi_transformed_imag_st, dtype=g.int8)			
				Psi_transformed_real_bytes_mt_list  = g.split(Psi_transformed_real_mt_320_8, dim=-2, num_splits=g.float32.size)
				Psi_transformed_imag_bytes_mt_list  = g.split(Psi_transformed_imag_mt_320_8, dim=-2, num_splits=g.float32.size)
				
				#target_qubit_L_write_map_st = self.target_qubit_write_maps_L_mt.read( streams=g.SG1_W[16] )

				target_qubit_L_write_map_st = self.select_target_qbit_write_map_east_LL()
				
				for byte_idx in range(4):
					
					g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_mt_list[byte_idx], target_qubit_L_write_map_st)
					g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st)
					g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_mt_list[byte_idx], target_qubit_L_write_map_st)
					g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st)
					
					
					

				Psi_transformed_real_mt_output = g.constant_tensor( Psi_transformed_real_mt.shape, g.float32, storage_req=Psi_transformed_real_mt.storage_request, 
                                                                                    name=f"Psi_transformed_real", initialized=False)
				Psi_transformed_imag_mt_output = g.constant_tensor( Psi_transformed_imag_mt.shape, g.float32, storage_req=Psi_transformed_imag_mt.storage_request, 
                                                                                    name=f"Psi_transformed_imag", initialized=False)

				Psi_transformed_real_mt_output.set_program_output()
				Psi_transformed_imag_mt_output.set_program_output()
				
				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
