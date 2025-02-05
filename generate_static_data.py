from QCsim_IO import IO
from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class Generate_static_data(IO):

	def __init__(self, pgm_pkg, mxm_permutation=True):
	
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )	
		self.gate_kernels_broadcasted_real_dict = {}
		self.gate_kernels_broadcasted_imag_dict = {}
		self.gate_kernels_broadcasted_real_copy_dict = {}
		self.gate_kernels_broadcasted_imag_copy_dict = {}				



	def broadcast_element( self, element_to_broadcast_st, distribute_map_mt, distributor_request, transposer_request):
		"""
		Function to broadcast an element over the whole 320 vector
		This function is used to broadcast either of the first 4 elements
		
		Parameters
		----------
		element_to_broadcast_st : StreamTensor
			A stream tensor containing the element to be broadcasted
		distribute_map_mt : StreamTensor
			A distributor map to bring the selected element onto the 0-th place.
		distributor_request : DistributorRequest
			A resource request for a distributor
		transposer_request : TransposerRequest		
			A resource request for a transposer						
			
			
		Return
		--------
		returns with the broadcatsed stream tensor
		"""
		
		
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
		"""
		Function to generate memory maps for the control qubits to gather state vector elements from the memory for target qubits S corresponding to control qbit state 1
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		
		control_qbit_min = 9
		control_qbit_max = self.ML.qbit_num		

	

		
		# controlled operations perform only on half of the elements
		half_memory_addresses = self.ML.memory_addresses >> 1
		
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
		mem_gather_map = np.zeros( (half_memory_addresses << 1, 320,), dtype=np.uint8 )
	
	
		for control_qbit in range(control_qbit_min, control_qbit_max+1):
	
			index_pair = 1 << (control_qbit - self.ML.small_qbit_num_limit)
			
			# each 2-byte column in the 16-byte blocks of the map tensor corresponds to a specific control qbit 
			if control_qbit <= 16:
				horizontal_offset_in_memory_map = (control_qbit-control_qbit_min)*2 
			else:
				horizontal_offset_in_memory_map = (control_qbit-17)*2 
				
				
			# maps for control_qbits for 9,10,...,16 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
			# maps for control_qbits for 17,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
			if control_qbit <= 16:
				vertical_offset_in_memory_map = 0
			else:
				vertical_offset_in_memory_map = half_memory_addresses
							 
			mask_lower_bits = index_pair - 1 
			mask_upper_bits = 1 << ( self.ML.qbit_num - control_qbit )
			mask_upper_bits = mask_upper_bits - 1
			mask_upper_bits = mask_upper_bits << (control_qbit - self.ML.small_qbit_num_limit )
			
			# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
			control_qubit_state_1_indices = np.zeros( (half_memory_addresses, 2,), dtype=np.uint8 )
			
			half_of_slice = 1 << (21-self.ML.small_qbit_num_limit-1) # needed to offset the address for the bank alternating storage

			half_memory_addresses = self.ML.memory_addresses >> 1
			for memory_idx in range(half_memory_addresses):

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
			row_max = vertical_offset_in_memory_map+half_memory_addresses
			for superlane_idx in range(20):
				col_min = horizontal_offset_in_memory_map + 16*superlane_idx
				col_max = horizontal_offset_in_memory_map+2 + 16*superlane_idx
				#print( row_min, row_max, col_min, col_max )
				mem_gather_map[ row_min:row_max, col_min:col_max ] = control_qubit_state_1_indices
				

		return mem_gather_map
		
		

	def generate_target_qubit_8L_control_qbit_read_maps(self):
		"""
		Function to generate memory maps for the control qubits to gather state vector elements from the memory for target qubit 8 corresponding to control qbit state 1
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
		mem_read_map      = np.zeros( (self.ML.memory_addresses//2, 320,), dtype=np.uint8 )		
		mem_read_map_pair = np.zeros( (self.ML.memory_addresses//2, 320,), dtype=np.uint8 )				
					
		half_slice = 1 << 12

		

		for control_qbit in range(9, 17):

			#print( 'control_qbit: ', control_qbit, memory_addresses )
			
		
			horizontal_offset_in_memory_map = (control_qbit-1-self.ML.small_qbit_num_limit)*2 

			control_qbit_mask = 1<<(control_qbit-self.ML.small_qbit_num_limit)
		
			target_qbit_index_pairs = [ [x, x^1] for x in range(self.ML.memory_addresses) if x < x^1  ]
			target_qbit_index_pairs = np.array( target_qbit_index_pairs )


			# indices in the target qbit maps having control qubit states 1
			controlled_qubit_state_1_map_indices = np.argwhere( (target_qbit_index_pairs[:,0] & control_qbit_mask) > 0 )
			controlled_qubit_state_1_map         = target_qbit_index_pairs[controlled_qubit_state_1_map_indices, 0]
			controlled_qubit_state_1_map_pairs   = target_qbit_index_pairs[controlled_qubit_state_1_map_indices, 1]
			#print( controlled_qubit_state_1_map_indices.size, mem_read_map.shape )
			#print( controlled_qubit_state_1_map )
			
			for idx in range( controlled_qubit_state_1_map.size ):
				
				mem_addr      = controlled_qubit_state_1_map[idx]
				mem_addr_pair = controlled_qubit_state_1_map_pairs[idx]
				
				# adding memory address offset for bank alternation
				mem_addr      = mem_addr//2 if mem_addr % 2 == 0 else mem_addr//2 + half_slice
				mem_addr_pair = mem_addr_pair//2 if mem_addr_pair % 2 == 0 else mem_addr_pair//2 + half_slice
																							
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_mem_addr  = mem_addr & 255
				higher_bits_of_addresses_mem_addr = (mem_addr >> 8 ) & 255

				lower_bits_of_addresses_mem_addr_pair  = mem_addr_pair & 255
				higher_bits_of_addresses_mem_addr_pair = (mem_addr_pair >> 8 ) & 255
					
								
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_read_map[ 2*idx+0, col_idx ]   = lower_bits_of_addresses_mem_addr
					mem_read_map[ 2*idx+0, col_idx+1 ] = higher_bits_of_addresses_mem_addr
					
					mem_read_map[ 2*idx+1, col_idx ]   = lower_bits_of_addresses_mem_addr
					mem_read_map[ 2*idx+1, col_idx+1 ] = higher_bits_of_addresses_mem_addr
					
					mem_read_map_pair[ 2*idx+0, col_idx ]   = lower_bits_of_addresses_mem_addr_pair
					mem_read_map_pair[ 2*idx+0, col_idx+1 ] = higher_bits_of_addresses_mem_addr_pair
					
					mem_read_map_pair[ 2*idx+1, col_idx ]   = lower_bits_of_addresses_mem_addr_pair
					mem_read_map_pair[ 2*idx+1, col_idx+1 ] = higher_bits_of_addresses_mem_addr_pair	
																				

		return mem_read_map, mem_read_map_pair



	def generate_target_qubit_LL_control_qbit_read_maps(self):
		"""
		Function to generate memory maps for the control qubits to gather state vector elements from the memory for target qubits L corresponding to control qbit state 1
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
		mem_read_map = np.zeros( (self.ML.memory_addresses//2, 320,), dtype=np.uint8 )		
					
		half_slice = 1 << 12

		

		for control_qbit in range(9, self.ML.qbit_num):

			if control_qbit == self.ML.qbit_num-1:
				continue  #TODO!!!!!!!!!!!!!!!!!!!!!!!!

			#print( 'control_qbit: ', control_qbit )

			target_qbit = control_qbit + 1 # as we get the same control qubit memory maps for each target qbit
			
		
			horizontal_offset_in_memory_map = (control_qbit-1-self.ML.small_qbit_num_limit)*2 

			control_qbit_mask = 1<<(control_qbit-8)
		
			target_qbit_index_pairs = [ [x, x^(1<<(target_qbit-8))] for x in range(self.ML.memory_addresses) if x < x^(1<<(target_qbit-8))  ]
			target_qbit_index_pairs = np.array( target_qbit_index_pairs )


			# indices in the target qbit maps having control qubit states 1
			controlled_qubit_state_1_map_indices = np.argwhere( (target_qbit_index_pairs[:,0] & control_qbit_mask) > 0 )
			#print( controlled_qubit_state_1_map_indices.size, mem_read_map.shape )
			#print( controlled_qubit_state_1_map_indices )
			
			for idx in range( controlled_qubit_state_1_map_indices.size//2 ):
				
				idx_mem_a = controlled_qubit_state_1_map_indices[2*idx]
				idx_mem_b = controlled_qubit_state_1_map_indices[2*idx+1]
																			
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_idx_mem_a  = idx_mem_a & 255
				higher_bits_of_addresses_idx_mem_a = (idx_mem_a >> 8 ) & 255

				lower_bits_of_addresses_idx_mem_b  = idx_mem_b & 255
				higher_bits_of_addresses_idx_mem_b = (idx_mem_b >> 8 ) & 255
					
								
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_read_map[ 4*idx+0, col_idx ]   = lower_bits_of_addresses_idx_mem_a
					mem_read_map[ 4*idx+0, col_idx+1 ] = higher_bits_of_addresses_idx_mem_a

					mem_read_map[ 4*idx+1, col_idx ]   = lower_bits_of_addresses_idx_mem_b
					mem_read_map[ 4*idx+1, col_idx+1 ] = higher_bits_of_addresses_idx_mem_b

					mem_read_map[ 4*idx+2, col_idx ]   = lower_bits_of_addresses_idx_mem_a
					mem_read_map[ 4*idx+2, col_idx+1 ] = higher_bits_of_addresses_idx_mem_a

					mem_read_map[ 4*idx+3, col_idx ]   = lower_bits_of_addresses_idx_mem_b
					mem_read_map[ 4*idx+3, col_idx+1 ] = higher_bits_of_addresses_idx_mem_b
					
																				
			
		#print( mem_read_map[:,0:2] )
		return mem_read_map
			
	def generate_target_qubit_L_read_maps(self):
		"""
		Function to generate memory maps to gather state vector elements from the memory for target qubits L
		
		Return
		--------
		returns with the generated numpy version of the memmory maps, separate for state 0 and state 1 of the target qbit
		"""
	
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
		mem_read_map = np.zeros( (self.ML.memory_addresses, 320,), dtype=np.uint8 )
		mem_read_pair_map = np.zeros( (self.ML.memory_addresses, 320,), dtype=np.uint8 )
		
		'''
					readout_indices.append( index_pairs[2*idx][0] )      # for upper row of the gate kernel (output state is 0)
					readout_indices.append( index_pairs[2*idx+1][0] )    # for upper row of the gate kernel (output state is 0)
					readout_indices.append( index_pairs[2*idx][0] )      # for lower row of the gate kernel (output state is 1) 
					readout_indices.append( index_pairs[2*idx+1][0] )    # for lower row of the gate kernel (output state is 1)					
										
					readout_indices_pair.append( index_pairs[2*idx][1] )   # for upper row of the gate kernel (output state is 0)
					readout_indices_pair.append( index_pairs[2*idx+1][1] ) # for upper row of the gate kernel (output state is 0)			
					readout_indices_pair.append( index_pairs[2*idx][1] )   # for lower row of the gate kernel (output state is 1)
					readout_indices_pair.append( index_pairs[2*idx+1][1] ) # for lower row of the gate kernel (output state is 1)			
		'''			
					
		half_slice = 1 << 12
		
		for target_qbit in range(9, 17):
		
			horizontal_offset_in_memory_map = (target_qbit-1-self.ML.small_qbit_num_limit)*2 
		
			index_pairs = [ [x, x^(1<<(target_qbit-8))] for x in range(self.ML.memory_addresses) if x < x^(1<<(target_qbit-8)) ]
							
			for idx in range( len(index_pairs)//2 ):
				
				idx1 = index_pairs[2*idx][0]
				idx2 = index_pairs[2*idx+1][0]
				
				idx1_pair = index_pairs[2*idx][1]
				idx2_pair = index_pairs[2*idx+1][1]	
				
				# adding memory address offset for bank alternation
				idx1 = idx1//2 if idx1 % 2 == 0 else idx1//2 + half_slice
				idx2 = idx2//2 if idx2 % 2 == 0 else idx2//2 + half_slice
				
				idx1_pair = idx1_pair//2 if idx1_pair % 2 == 0 else idx1_pair//2 + half_slice
				idx2_pair = idx2_pair//2 if idx2_pair % 2 == 0 else idx2_pair//2 + half_slice
																											
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_idx1  = idx1 & 255
				higher_bits_of_addresses_idx1 = (idx1 >> 8 ) & 255
				
				lower_bits_of_addresses_idx2  = idx2 & 255
				higher_bits_of_addresses_idx2 = (idx2 >> 8 ) & 255
				
				lower_bits_of_addresses_idx1_pair  = idx1_pair & 255
				higher_bits_of_addresses_idx1_pair = (idx1_pair >> 8 ) & 255
				
				lower_bits_of_addresses_idx2_pair  = idx2_pair & 255
				higher_bits_of_addresses_idx2_pair = (idx2_pair >> 8 ) & 255												
					
										
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_read_map[ 2*idx+0, col_idx ]   = lower_bits_of_addresses_idx1
					mem_read_map[ 2*idx+0, col_idx+1 ] = higher_bits_of_addresses_idx1
					
					mem_read_map[ 2*idx+1, col_idx ]   = lower_bits_of_addresses_idx2
					mem_read_map[ 2*idx+1, col_idx+1 ] = higher_bits_of_addresses_idx2
					
					mem_read_pair_map[ 2*idx+0, col_idx ]   = lower_bits_of_addresses_idx1_pair
					mem_read_pair_map[ 2*idx+0, col_idx+1 ] = higher_bits_of_addresses_idx1_pair
					
					mem_read_pair_map[ 2*idx+1, col_idx ]   = lower_bits_of_addresses_idx2_pair
					mem_read_pair_map[ 2*idx+1, col_idx+1 ] = higher_bits_of_addresses_idx2_pair	
																				

		return mem_read_map, mem_read_pair_map


	def generate_target_qubit_8L_control_qbit_write_maps(self):
		"""
		Function to generate memory maps to scatter state vector elements into the memory for target qubits L
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		# memory map containing all of the memory maps		
		mem_write_map      = np.zeros( (self.ML.memory_addresses//2, 320,), dtype=np.uint8 )
		
		half_slice = 1 << 12
		
		for control_qbit in range(9, 17):
		
			horizontal_offset_in_memory_map = (control_qbit-1-self.ML.small_qbit_num_limit)*2 

			control_qbit_mask = 1<<(control_qbit-self.ML.small_qbit_num_limit)
		
			target_qbit_index_pairs = [ [x, x^1] for x in range(self.ML.memory_addresses) if x < x^1  ]
			target_qbit_index_pairs = np.array( target_qbit_index_pairs )


			# indices in the target qbit maps having control qubit states 1
			controlled_qubit_state_1_map_indices = np.argwhere( (target_qbit_index_pairs[:,0] & control_qbit_mask) > 0 )
			controlled_qubit_state_1_map         = target_qbit_index_pairs[controlled_qubit_state_1_map_indices, 0]
			controlled_qubit_state_1_map_pairs   = target_qbit_index_pairs[controlled_qubit_state_1_map_indices, 1]
			#print( controlled_qubit_state_1_map_indices.size, mem_write_map.shape )
			#print( controlled_qubit_state_1_map )
			
			for idx in range( controlled_qubit_state_1_map.size ):
				
				mem_addr      = controlled_qubit_state_1_map[idx]
				mem_addr_pair = controlled_qubit_state_1_map_pairs[idx]
				
				# adding memory address offset for bank alternation
				mem_addr      = mem_addr//2 if mem_addr % 2 == 0 else mem_addr//2 + half_slice
				mem_addr_pair = mem_addr_pair//2 if mem_addr_pair % 2 == 0 else mem_addr_pair//2 + half_slice
																							
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_mem_addr  = mem_addr & 255
				higher_bits_of_addresses_mem_addr = (mem_addr >> 8 ) & 255

				lower_bits_of_addresses_mem_addr_pair  = mem_addr_pair & 255
				higher_bits_of_addresses_mem_addr_pair = (mem_addr_pair >> 8 ) & 255
					
								
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_write_map[ 2*idx+0, col_idx ]   = lower_bits_of_addresses_mem_addr
					mem_write_map[ 2*idx+0, col_idx+1 ] = higher_bits_of_addresses_mem_addr
					
					mem_write_map[ 2*idx+1, col_idx ]   = lower_bits_of_addresses_mem_addr_pair
					mem_write_map[ 2*idx+1, col_idx+1 ] = higher_bits_of_addresses_mem_addr_pair
						
																				

		return mem_write_map




	def generate_target_qubit_LL_control_qbit_write_maps(self):
		"""
		Function to generate memory maps for the control qubits L to scatter the transformed state vector into the memory
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		# memory map containing all of the memory maps
		# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_addresses)
		# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_addresses+1,...,self.ML.memory_addresses)		
		mem_write_map = np.zeros( (self.ML.memory_addresses//2, 320,), dtype=np.uint8 )		
					
		half_slice = 1 << 12

		

		for control_qbit in range(9, self.ML.qbit_num):

			if control_qbit == self.ML.qbit_num-1:
				continue  #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			#print( 'control_qbit: ', control_qbit )

			target_qbit = control_qbit + 1 # as we get the same control qubit memory maps for each target qbit
			
		
			horizontal_offset_in_memory_map = (control_qbit-1-self.ML.small_qbit_num_limit)*2 

			control_qbit_mask = 1<<(control_qbit-8)
		
			target_qbit_index_pairs = [ [x, x^(1<<(target_qbit-8))] for x in range(self.ML.memory_addresses) if x < x^(1<<(target_qbit-8))  ]
			target_qbit_index_pairs = np.array( target_qbit_index_pairs )

			target_qbit_index_pairs_write = np.zeros( (len(target_qbit_index_pairs)*2, ), dtype=np.uint16 )		
			for idx in range( len(target_qbit_index_pairs)//2 ):
				
				idx1 = target_qbit_index_pairs[2*idx][0]
				idx2 = target_qbit_index_pairs[2*idx+1][0]
				idx3 = target_qbit_index_pairs[2*idx][1]
				idx4 = target_qbit_index_pairs[2*idx+1][1]	
											
				target_qbit_index_pairs_write[ 4*idx+0 ] = idx1
				target_qbit_index_pairs_write[ 4*idx+1 ] = idx2
				target_qbit_index_pairs_write[ 4*idx+2 ] = idx3
				target_qbit_index_pairs_write[ 4*idx+3 ] = idx4


			# indices in the target qbit maps having control qubit states 1
			controlled_qubit_state_1_map_indices = np.argwhere( (target_qbit_index_pairs_write[:] & control_qbit_mask) > 0 )
			#print( 'write ', controlled_qubit_state_1_map_indices.size, mem_write_map.shape )
			#print( controlled_qubit_state_1_map_indices )
			#print( target_qbit_index_pairs_write )
											
																				
			
			for idx in range( controlled_qubit_state_1_map_indices.size ):
				
				idx_mem = controlled_qubit_state_1_map_indices[idx]
																			
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_idx_mem  = idx_mem & 255
				higher_bits_of_addresses_idx_mem = (idx_mem >> 8 ) & 255
					
								
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_write_map[ idx, col_idx ]   = lower_bits_of_addresses_idx_mem
					mem_write_map[ idx, col_idx+1 ] = higher_bits_of_addresses_idx_mem
					
																		
			
		return mem_write_map
		
		
	def generate_target_qubit_L_write_maps(self):
		"""
		Function to generate memory maps to scatter state vector elements into the memory for target qubits L
		
		Return
		--------
		returns with the generated numpy version of the memmory maps
		"""
	
		# memory map containing all of the memory maps		
		mem_write_map = np.zeros( (self.ML.memory_addresses, 320,), dtype=np.uint8 )
		
		half_slice = 1 << 12
		
		for target_qbit in range(9, self.ML.qbit_num):
		
			horizontal_offset_in_memory_map = (target_qbit-1-self.ML.small_qbit_num_limit)*2 
		
			index_pairs = [ [x, x^(1<<(target_qbit-8))] for x in range(self.ML.memory_addresses) if x < x^(1<<(target_qbit-8)) ]
							
			for idx in range( len(index_pairs)//2 ):
				
				idx1 = index_pairs[2*idx][0]
				idx2 = index_pairs[2*idx+1][0]
				idx3 = index_pairs[2*idx][1]
				idx4 = index_pairs[2*idx+1][1]	
				
				# adding memory address offset for bank alternation
				idx1 = idx1//2 if idx1 % 2 == 0 else idx1//2 + half_slice
				idx2 = idx2//2 if idx2 % 2 == 0 else idx2//2 + half_slice
				idx3 = idx3//2 if idx3 % 2 == 0 else idx3//2 + half_slice
				idx4 = idx4//2 if idx4 % 2 == 0 else idx4//2 + half_slice
																											
								
				# the 2-byte (16-bit) memory offset needs to be splitted into two bytes
				lower_bits_of_addresses_idx1  = idx1 & 255
				higher_bits_of_addresses_idx1 = (idx1 >> 8 ) & 255
				
				lower_bits_of_addresses_idx2  = idx2 & 255
				higher_bits_of_addresses_idx2 = (idx2 >> 8 ) & 255
				
				lower_bits_of_addresses_idx3  = idx3 & 255
				higher_bits_of_addresses_idx3 = (idx3 >> 8 ) & 255
				
				lower_bits_of_addresses_idx4  = idx4 & 255
				higher_bits_of_addresses_idx4 = (idx4 >> 8 ) & 255
					
										
				for superlane_idx in range(20):
					col_idx = horizontal_offset_in_memory_map + 16*superlane_idx
					
					mem_write_map[ 4*idx+0, col_idx ]   = lower_bits_of_addresses_idx1
					mem_write_map[ 4*idx+0, col_idx+1 ] = higher_bits_of_addresses_idx1
					
					mem_write_map[ 4*idx+1, col_idx ]   = lower_bits_of_addresses_idx2
					mem_write_map[ 4*idx+1, col_idx+1 ] = higher_bits_of_addresses_idx2
					
					mem_write_map[ 4*idx+2, col_idx ]   = lower_bits_of_addresses_idx3
					mem_write_map[ 4*idx+2, col_idx+1 ] = higher_bits_of_addresses_idx3
					
					mem_write_map[ 4*idx+3, col_idx ]   = lower_bits_of_addresses_idx4
					mem_write_map[ 4*idx+3, col_idx+1 ] = higher_bits_of_addresses_idx4	
																				

		return mem_write_map
				

	def reset_state_vector_program(self):
    
		name = "state_vector_reset"
		with self.pgm_pkg.create_program_context(name) as pgm1:

			
			reset_mt_list = [ self.zeros_mt ] * self.ML.memory_addresses
			reset_mt = g.concat( reset_mt_list, dim=0 )
			reset_st = reset_mt.read( streams=[g.SG4_W[0]], time=0 )
		
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_real_mt_320      = g.constant_tensor((self.ML.memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_real_mt.storage_request, initialized=False)		
			state_imag_mt_320      = g.constant_tensor((self.ML.memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_imag_mt.storage_request, initialized=False)
			state_real_copy_mt_320 = g.constant_tensor((self.ML.memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_real_copy_mt.storage_request, initialized=False)
			state_imag_copy_mt_320 = g.constant_tensor((self.ML.memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_imag_copy_mt.storage_request, initialized=False)		
		
			reset_st.write( name="State_reset_real", storage_req=state_real_mt_320.storage_request)
			reset_st.write( name="State_reset_imag", storage_req=state_imag_mt_320.storage_request)				
			reset_st.write( name="State_reset_real_copy", storage_req=state_real_copy_mt_320.storage_request)
			reset_st.write( name="State_reset_imag_copy", storage_req=state_imag_copy_mt_320.storage_request)		
			
			
			state_vector_first_vector      = g.split( state_real_mt_320, num_splits=self.ML.memory_addresses, dim=0 )[0]
			state_vector_copy_first_vector = g.split( state_real_copy_mt_320, num_splits=self.ML.memory_addresses, dim=0 )[0]	
					
			
			single_one_st = self.single_one_mt.read( streams=[g.SG4_W[0]], time=self.ML.memory_addresses )
			single_one_st.write( name="State_reset_real_first_vector", storage_req=state_vector_first_vector.storage_request)
			single_one_st.write( name="State_reset_real_copy_first_vector", storage_req=state_vector_copy_first_vector.storage_request)			
			
			

	def gate_upload_program(self):
		"""
		Function implementing the upload program
		"""

		# Defines a program to upload the inputs
		name = "upload"
		with self.pgm_pkg.create_program_context(name) as pgm1:

			# storage to store reset data for gate counters:
			gate_counter_reset_data_np = np.zeros( (6, 320,), dtype=np.uint8 )
			gate_counter_reset_data_np[0, 0:320:16] = 0 #gate_counter_perm_map_selector, gate_counter_gate_filtering_selector, gate_counter_distributor_maps_west

			gate_counter_reset_data_np[1, 0:320:16] = 0 #gate_counter_S_control_qbit_selector, gate_counter_target_qubit_L_distributor_maps_write
			gate_counter_reset_data_np[1, 1:320:16] = self.ML.gate_memory_offset_factor #gate_counter_S_control_qbit_selector

			gate_counter_reset_data_np[2, 0:320:16] = 0 #gate_element_selector_00
			gate_counter_reset_data_np[2, 1:320:16] = 1 #gate_element_selector_00

			gate_counter_reset_data_np[3, 0:320:16] = 0 #gate_element_selector_11
			gate_counter_reset_data_np[3, 1:320:16] = 1+3*(self.ML.gate_memory_offset_factor) #gate_element_selector_11

			gate_counter_reset_data_np[4, 0:320:16] = 0 #gate_element_selector_01
			gate_counter_reset_data_np[4, 1:320:16] = 1+(self.ML.gate_memory_offset_factor) #gate_element_selector_01

			gate_counter_reset_data_np[5, 0:320:16] = 0 #gate_element_selector_10
			gate_counter_reset_data_np[5, 1:320:16] = 1+2*(self.ML.gate_memory_offset_factor) #gate_element_selector_10

			self.gate_counter_reset_data_mt = g.from_data( np.asarray(gate_counter_reset_data_np, 
										dtype=np.uint8),
										layout=self.ML.layout_gate_counter_reset_data,
										name="gate_counter_reset_data" )

			self.gate_counter_reset_data_mt_list = g.split( self.gate_counter_reset_data_mt, num_splits=6, dim=0 )
			self.gate_counter_reset_data_mt.is_static = True


					
			# dynamic map selectors to vectors where the target_qbit (or control_qbit) takes on state 1 or 0
			# *state_1* is True where the target_qbit (or control_qbit) takes on state 1
			# *state_0* is True where the target_qbit (or control_qbit) takes on state 0			
			state_1_selector_mt         = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                     dtype=g.uint8, 
                                                                     name="state_1_selector", 
                                                                     layout=self.ML.layout_state_1_selector)

			state_0_selector_mt         = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                     dtype=g.uint8, 
                                                                     name="state_0_selector", 
                                                                     layout=self.ML.layout_state_0_selector)	
		
			control_state_1_selector_mt = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                     dtype=g.uint8, 
                                                                     name="control_state_1_selector", 
                                                                     layout=self.ML.layout_control_state_1_selector)

			control_state_0_selector_mt = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                     dtype=g.uint8, 
                                                                     name="control_state_0_selector", 
                                                                     layout=self.ML.layout_control_state_0_selector)


			# 80 2x2 gate kernels are packed into real and complex 320 float32 vectors
			self.gate_kernels_real_packed_mt = g.input_tensor(shape=(self.ML.gate_splits,320,), 
                                                                          dtype=g.float32, 
                                                                          name="gate_kernels_real_packed", 
                                                                          layout=self.ML.layout_gate_kernels_real_packed)

			self.gate_kernels_imag_packed_mt = g.input_tensor(shape=(self.ML.gate_splits,320,), 
                                                                          dtype=g.float32, 
                                                                          name="gate_kernels_imag_packed", 
                                                                          layout=self.ML.layout_gate_kernels_imag_packed)

			exclusion_tensors = [state_1_selector_mt, state_0_selector_mt, control_state_1_selector_mt, control_state_0_selector_mt, self.gate_kernels_real_packed_mt, self.gate_kernels_imag_packed_mt]
			
			
			
			
			# the input vector containing the memory address for the correct target qubits. 
			# TODO: cutrrently supports only a single target qubit, but need to support 80, as the number of the packed gate kernels		
			self.state_permute_map_selector_mt     = g.input_tensor(shape=(self.ML.gate_count,320,),
                                                                                dtype=g.uint8, 
                                                                                name="permutation_selector", 
                                                                                layout=self.ML.layout_state_permute_map_selector)				

			self.mxm_state_permute_map_selector_mt = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                              dtype=g.uint8, 
                                                                              name="mxm_permutation_selector", 
                                                                              layout=self.ML.layout_mxm_permute_map_selector)	
			
			g.add_mem_constraints([self.mxm_state_permute_map_selector_mt], [self.state_permute_map_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			
			
			# constant tensor to update the address counters by one on lower bits
			constant_1_address_offset_lower_bits_np           = np.zeros( (320,), dtype=np.uint8 )
			constant_1_address_offset_lower_bits_np[0:320:16] = 1
			self.constant_1_address_offset_lower_bits_mt = g.from_data( 
                                                                           np.asarray(constant_1_address_offset_lower_bits_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_constant_1_address_offset_lower_bits, 
                                                                           name="constant_1_address_offset_lower_bits" )
			
			constant_1_address_offset_higher_bits_np           = np.zeros( (320,), dtype=np.uint8 )
			constant_1_address_offset_higher_bits_np[1:320:16] = 1
			self.constant_1_address_offset_higher_bits_mt = g.from_data( 
                                                                           np.asarray(constant_1_address_offset_higher_bits_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_constant_1_address_offset_higher_bits )
			

			self.gate_counter_address_offset_higher_bits_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_address_offset_higher_bits, 
										name=f"gate_counter_address_offset_higher_bits", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[0].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_address_offset_higher_bits_mt.storage_request, time=0)


			
			g.add_mem_constraints([self.constant_1_address_offset_lower_bits_mt], [self.constant_1_address_offset_higher_bits_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			
			#self.gate_kernels_real_packed_mt.is_static 	 = True
			#self.gate_kernels_imag_packed_mt.is_static 	 = True
			state_1_selector_mt.is_static			     = True
			state_0_selector_mt.is_static			     = True
			control_state_1_selector_mt.is_static        = True
			control_state_0_selector_mt.is_static        = True
			self.state_permute_map_selector_mt.is_static = True
			self.constant_1_address_offset_lower_bits_mt.is_static   = True
			self.constant_1_address_offset_higher_bits_mt.is_static  = True		
			self.gate_counter_address_offset_higher_bits_mt.is_static = True	

			
			
			if ( self.mxm_permutation ):
				# if MXM is used to generate the permuted state vector
				
				# expand the permutation maps to query for the whole 320x320 permute matrix, not just for the first row
				
				
				alu_rq_8 = g.tensor.create_alu_request( alus=[8] )

				# split mxm_state_permute_map_selectors according to the gates
				gate_resolved_mxm_state_permute_map_selectors_list = g.split( self.mxm_state_permute_map_selector_mt, num_splits=gate_count, dim=0 )
				
				state_permute_map_selector_new_mt_list = []
				for gate_idx in range(self.ML.gate_count):

					mxm_state_permute_map_selector_mt = gate_resolved_mxm_state_permute_map_selectors_list[gate_idx]

					for row_idx in range(20) :
				
						layout = self.ML.layout_expanded_mxm_permute_map_selector + f", A{1}({320 + self.ML.gate_count + gate_idx*20 + row_idx})"
				
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

				for target_qbit_loc in range(self.ML.small_qbit_num_limit):

					target_qbit_pair_diff = 1 << target_qbit_loc

					matrix_row_indexes = np.zeros( (320,), dtype=np.int32 )
					for idx in range(320):
						if (idx < self.ML.matrix_size):
							matrix_row_indexes[idx] = idx ^ target_qbit_pair_diff
						else:
							matrix_row_indexes[idx] = idx
					

					mxm_permutation_matrices[ target_qbit_loc,:,: ] = identity[matrix_row_indexes,:]

					
				self.mxm_permute_matrices_mt_west = g.from_data( np.asarray(mxm_permutation_matrices, dtype=np.int8), layout=self.ML.layout_mxm_permutation_matrices_west, name="mxm_permute_matrices_mt_west" )
				self.mxm_permute_matrices_mt_east = g.from_data( np.asarray(mxm_permutation_matrices, dtype=np.int8), layout=self.ML.layout_mxm_permutation_matrices_east, name="mxm_permute_matrices_mt_east" )
				self.mxm_permute_matrices_mt_west.is_static = True
				self.mxm_permute_matrices_mt_east.is_static = True


				self.permute_maps_mt_west = None
				self.permute_maps_mt_east = None
			
			#else:	### !!!!!! gate the SL needs the permutor !!!!!!!!!
			# if the permutor is used to generate the permuted state vector
			
			
			# expand the permutation maps to query for the whole 320x320 permute matrix, not just for the first row
			
			gate_counter_perm_map_selector_np           = np.zeros( (320,), dtype=np.uint8 )
			gate_counter_perm_map_selector_np[0:320:16] = 0
			

			self.gate_counter_perm_map_selector_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_perm_map_selector, 
										name=f"gate_counter_perm_map_selector", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[0].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_perm_map_selector_mt.storage_request, time=0)



			self.gate_counter_gate_filtering_selector_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_gate_filtering_selector, 
										name=f"gate_counter_gate_filtering_selector", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[0].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_gate_filtering_selector_mt.storage_request, time=0)
			


			self.gate_counter_S_control_qbit_selector_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_S_control_qbit_selector, 
										name=f"gate_counter_S_control_qbit_selector", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[1].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_S_control_qbit_selector_mt.storage_request, time=1)
			
			exclusion_tensors.append( self.gate_counter_gate_filtering_selector_mt)
			exclusion_tensors.append( self.gate_counter_S_control_qbit_selector_mt)


			
			self.gate_counter_gate_element_selector_00_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_element_selector_00, 
										name=f"gate_counter_gate_element_selector_00", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[2].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_gate_element_selector_00_mt.storage_request, time=2)
			
			self.gate_counter_gate_element_selector_11_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_element_selector_11, 
										name=f"gate_counter_gate_element_selector_11", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[3].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_gate_element_selector_11_mt.storage_request, time=3)
			
			self.gate_counter_gate_element_selector_01_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_element_selector_01, 
										name=f"gate_counter_gate_element_selector_01", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[4].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_gate_element_selector_01_mt.storage_request, time=4)
			
			self.gate_counter_gate_element_selector_10_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_element_selector_10, 
										name=f"gate_counter_gate_element_selector_10", 
										initialized=False)
			self.gate_counter_reset_data_mt_list[5].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_gate_element_selector_10_mt.storage_request, time=5)
			

			exclusion_tensors.append( self.gate_counter_gate_element_selector_00_mt )
			exclusion_tensors.append( self.gate_counter_gate_element_selector_11_mt )
			exclusion_tensors.append( self.gate_counter_gate_element_selector_01_mt )
			exclusion_tensors.append( self.gate_counter_gate_element_selector_10_mt )
											
										 	
			self.gate_counter_perm_map_selector_mt.is_static = True										 	
			self.gate_counter_gate_filtering_selector_mt.is_static = True	
			self.gate_counter_S_control_qbit_selector_mt.is_static = True				
			self.gate_counter_gate_element_selector_00_mt.is_static = True
			self.gate_counter_gate_element_selector_01_mt.is_static = True
			self.gate_counter_gate_element_selector_10_mt.is_static = True
			self.gate_counter_gate_element_selector_11_mt.is_static = True

			self.gate_counter_distributor_maps_west_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_distributor_maps_west, 
										name=f"gate_counter_distributor_maps_west", 
										initialized=False)
			self.gate_counter_distributor_maps_west_mt.is_static = True
			self.gate_counter_reset_data_mt_list[0].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_distributor_maps_west_mt.storage_request, time=6)

			# distributor maps to select the correct memory maps to write data on L and H target qubits.
			self.gate_counter_target_qubit_L_distributor_maps_write_mt      = g.constant_tensor( (320,), 
										g.uint8, 
										layout=self.ML.layout_gate_counter_target_qubit_L_distributor_maps, 
										name=f"gate_counter_target_qubit_L_distributor_maps_write", 
										initialized=False)
			self.gate_counter_target_qubit_L_distributor_maps_write_mt.is_static = True
			self.gate_counter_reset_data_mt_list[1].read(streams=[g.SG1[31]]).write( storage_req=self.gate_counter_target_qubit_L_distributor_maps_write_mt.storage_request, time=7)
			

			
			# generate permutation maps to reorder the state vector elements for qubits less than small_qbit_num_limit
			permute_map = np.zeros( (self.ML.small_qbit_num_limit,320,), dtype=np.uint32 )

			for target_qbit_loc in range(self.ML.small_qbit_num_limit):

				permute_map_np = np.zeros( (256,), dtype=np.uint32 )
				target_qbit_pair_diff = 1 << target_qbit_loc
				
				for idx in range(256): # 256 = 2^8
					if (idx < self.ML.matrix_size):
						permute_map_np[idx] = idx ^ target_qbit_pair_diff
					else:
						permute_map_np[idx] = idx
				

				permute_map[target_qbit_loc,:] = inst.encode_permute_map(  permute_map_np.tolist() )

			self.permute_maps_mt_west = g.from_data( np.asarray(permute_map, dtype=np.uint8), 
                                                                 layout=self.ML.layout_permute_maps_west, 
                                                                 name="permute_maps_mt_west" )

			self.permute_maps_mt_east = g.from_data( np.asarray(permute_map, dtype=np.uint8), 
                                                                 layout=self.ML.layout_permute_maps_east, 
                                                                 name="permute_maps_mt_east" )
			self.permute_maps_mt_west.is_static = True
			self.permute_maps_mt_east.is_static = True


				#self.mxm_permute_matrices_mt_west = None
				#self.mxm_permute_matrices_mt_east = None
				
				
			
			


			# create reusable distributor requests
			distributor_requests = []
			distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
			distributor_requests.append(  g.tensor.create_distributor_request( [7], 1 ) ) # EAST	
			
			# create reusable transposer requests
			transposer_requests = []
			transposer_requests.append(  g.tensor.create_transposer_request( [2], 1 ) ) # EAST
			transposer_requests.append(  g.tensor.create_transposer_request( [3], 1 ) ) # EAST	

			
			
			
			# generate tensors full with float32 ones and zeros  
			# this is needed later when manipulating the broadcates gate kernel elements with control qubits
			self.ones_mt = g.full((1, 1, 320), 1.0, dtype=g.float32, name="ones_mt", layout=self.ML.layout_ones)
			self.zeros_mt = g.full((1, 1, 320), 0.0, dtype=g.float32, name="zeros_mt", layout=self.ML.layout_zeros)
			self.ones_mt = g.reinterpret( self.ones_mt, dtype=g.uint32)
			self.zeros_mt = g.reinterpret( self.zeros_mt, dtype=g.uint32)
			self.ones_mt.is_static = True
			self.zeros_mt.is_static = True
			
			# a float vector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...] used to initialize the state vecor
			single_one_np = np.zeros( (320,), dtype=np.float32 )
			single_one_np[0] = 1.0
			self.single_one_mt = g.from_data( np.asarray(single_one_np, 
                                                                     dtype=np.float32), 
                                                                     layout=self.ML.layout_single_one, 
                                                                     name="single_one_mt" )
			self.single_one_mt.is_static = True
			
			
			
			
			with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=10) as distribute_state_indices_scope :
				
				# generate vectors to indicate which elements in a 320 vector correspond to state |1> of target qubit less than small_qbit_num_limit 
				# if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
				# if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
				states_1_np = np.zeros( (self.ML.small_qbit_num_limit,320,), dtype=np.uint8 )
				states_0_np = np.zeros( (self.ML.small_qbit_num_limit,320,), dtype=np.uint8 )
				for target_qbit_loc in range(self.ML.small_qbit_num_limit):
					target_qbit_pair_diff = 1 << target_qbit_loc
					for idx in range(256): # 256 = 2^8
						if (idx & target_qbit_pair_diff > 0):
							states_1_np[target_qbit_loc, idx] = 255
							states_0_np[target_qbit_loc, idx] = 0
						else:
							states_1_np[target_qbit_loc, idx] = 0
							states_0_np[target_qbit_loc, idx] = 255
					#print( "states |1> at target qubit ", target_qbit_loc, " is:" )
					#print( states_1_np[target_qbit_loc, 0:256] )

				states_1_mt = g.from_data( np.asarray(states_1_np, dtype=np.uint8), 
                                                           layout=self.ML.layout_states_1, 
                                                           name="states_1" )
				#states_1_mt.is_static = True

				states_0_mt = g.from_data( np.asarray(states_0_np, dtype=np.uint8), 
                                                           layout=self.ML.layout_states_0, 
                                                           name="states_0" )
				#states_0_mt.is_static = True

			
				# distribute maps to broadcast the gate kernel elements over a vector
				distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
				for idx in range(320):
					distribute_map_np[idx] = idx % 16

				distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), 
                                                                               layout=self.ML.layout_distribute_map_tensor_state1 )

				distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), 
                                                                               layout=self.ML.layout_distribute_map_tensor_state0 )

	
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

					
				self.state_1_broadcasted_mt = state_1_broadcasted_st.write(name="state_1_broadcasted", layout=self.ML.layout_state_1_broadcasted)
				self.state_0_broadcasted_mt = state_0_broadcasted_st.write(name="state_0_broadcasted", layout=self.ML.layout_state_0_broadcasted)


				self.state_1_broadcasted_mt.is_static = True
				self.state_0_broadcasted_mt.is_static = True
				
				
				
				# same as before but now for the control qbit
				control_state_1_selector_st = control_state_1_selector_mt.read(streams=[g.SG1[1]])
				control_state_1_st = g.mem_gather(states_1_mt, control_state_1_selector_st, output_streams=[g.SG1[0]], time=self.ML.gate_count+1)
				#control_state_1_st.write(name="control_state_1_st_oo", layout='H1(E), -1, S1(36)', program_output=True)
				
				control_state_0_selector_st = control_state_0_selector_mt.read(streams=[g.SG1[24]])
				control_state_0_st = g.mem_gather(states_0_mt, control_state_0_selector_st, output_streams=[g.SG1[24]], time=self.ML.gate_count+1)

				#states_1_st = states_1_mt.read( streams=g.SG1[8], time=5000 )
				#control_state_1_selector_st.write(name="states_1_st_ttt", layout='H1(E), -1, S1(43)', program_output=True)
				#control_state_1_st.write(name="control_state_1_st_ttt", layout='H1(E), -1, S1(36)', program_output=True)				
								
				control_state_1_broadcasted_st = self.broadcast_element(control_state_1_st,
				                                               distribute_map_tensor_state1_mt, 
				                                               distributor_requests[0],
				                                               transposer_requests[0])

				
				control_state_0_broadcasted_st = self.broadcast_element(control_state_0_st,
				                                               distribute_map_tensor_state0_mt, 
				                                               distributor_requests[1],
				                                               transposer_requests[1])
				
				
				self.control_state_1_broadcasted_mt = control_state_1_broadcasted_st.write(name="control_state_1_broadcasted", 
                                                                                                           layout=self.ML.layout_control_state_1_broadcasted)

				self.control_state_0_broadcasted_mt = control_state_0_broadcasted_st.write(name="control_state_0_broadcasted", 
                                                                                                           layout=self.ML.layout_control_state_0_broadcasted)

				#control_state_1_broadcasted_st.write(name="control_state_1_broadcasted_oo", layout='H1(E), -1, S4', program_output=True)
				
				self.control_state_1_broadcasted_mt.is_static = True
				self.control_state_0_broadcasted_mt.is_static = True
			
				
				
				exclusion_tensors.append( self.control_state_0_broadcasted_mt )
				exclusion_tensors.append( self.state_0_broadcasted_mt )
				exclusion_tensors.append( self.control_state_1_broadcasted_mt )
				exclusion_tensors.append( self.state_1_broadcasted_mt )
				exclusion_tensors.append( self.zeros_mt )
				exclusion_tensors.append( self.ones_mt )
				exclusion_tensors.append( self.single_one_mt )				
				exclusion_tensors.append( states_1_mt )
				exclusion_tensors.append( states_0_mt )
				exclusion_tensors.append( distribute_map_tensor_state1_mt )
				exclusion_tensors.append( distribute_map_tensor_state0_mt )
				exclusion_tensors.append( state_1_selector_mt )
				exclusion_tensors.append( state_0_selector_mt )

				g.add_mem_constraints(exclusion_tensors, exclusion_tensors, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
			with g.ResourceScope(name="create_memory_maps", is_buffered=False, predecessors=[distribute_state_indices_scope], time=None) as create_memory_maps:
			
				# generate control qubit memory maps for SL gate (to select the correct elements from the state vector if SL gate is performed)
				mem_gather_map = self.generate_control_qbit_maps_for_SL()

				self.control_qubit_address_maps_SL_west_mt = g.from_data(mem_gather_map, layout=self.ML.layout_control_qubit_address_maps_SL_west)
				self.control_qubit_address_maps_SL_east_mt = g.from_data(mem_gather_map, layout=self.ML.layout_control_qubit_address_maps_SL_east_a)			
				self.control_qubit_address_maps_SL_west_mt.is_static = True	
				self.control_qubit_address_maps_SL_east_mt.is_static = True
			
				# generate memeory write maps for target qubits L
				mem_write_map = self.generate_target_qubit_L_write_maps()
			
				self.target_qubit_write_maps_L_mt = g.from_data(mem_write_map, layout=self.ML.layout_target_qubit_write_maps_L)
				self.target_qubit_write_maps_L_mt.is_static = True
			
			
				# generate memeory read maps for target qubits L
				mem_read_map, mem_read_pair_map = self.generate_target_qubit_L_read_maps()
			
				self.target_qubit_read_pair_maps_L_mt = g.from_data(mem_read_pair_map, 
                                                                                    layout=self.ML.layout_target_qubit_read_pair_maps_L)

				self.target_qubit_read_maps_L_mt      = g.from_data(mem_read_map, 
                                                                                    layout=self.ML.layout_target_qubit_read_maps_L)	
		
				self.target_qubit_read_pair_maps_L_mt.is_static = True
				self.target_qubit_read_maps_L_mt.is_static      = True			
			


				####################################

				# tensor to store mem scatter maps for LL gates
				mem_write_map = self.generate_target_qubit_LL_control_qbit_write_maps()
	
				self.target_qubit_write_maps_LL_mt           = g.from_data(mem_write_map, 
                                                                                           layout=self.ML.layout_target_qubit_write_maps_LL)
				self.target_qubit_write_maps_LL_mt.is_static = True

				# tensor to store mem gather maps for LL gates
				mem_read_map = self.generate_target_qubit_LL_control_qbit_read_maps()	

				self.target_qubit_read_maps_LL_mt           = g.from_data(mem_read_map, 
                                                                                          layout=self.ML.layout_target_qubit_read_maps_LL)	
				self.target_qubit_read_maps_LL_mt.is_static = True				


				# tensor to store mem gather maps for 8L gates (two of them is needed because of the peculiar bank alternating strategy)
				mem_read_map, mem_read_map_pair = self.generate_target_qubit_8L_control_qbit_read_maps()			

				self.target_qubit_read_maps_8L_mt      = g.from_data(mem_read_map, 
                                                                                     layout=self.ML.layout_target_qubit_read_maps_8L)	

				self.target_qubit_read_maps_8L_pair_mt = g.from_data(mem_read_map_pair, 
                                                                                     layout=self.ML.layout_target_qubit_read_maps_8L_pair)	

				self.target_qubit_read_maps_8L_mt.is_static      = True				
				self.target_qubit_read_maps_8L_pair_mt.is_static = True		
			
			
				
				# tensor to store mem scatter maps for 8L gates (two of them is needed because of the peculiar bank alternating strategy)
				mem_write_map = self.generate_target_qubit_8L_control_qbit_write_maps()	
	
				self.target_qubit_write_maps_8L_mt      = g.from_data(mem_write_map, layout=self.ML.layout_target_qubit_write_maps_8L_a)	
				self.target_qubit_write_maps_8L_mt_b    = g.from_data(mem_write_map, layout=self.ML.layout_target_qubit_write_maps_8L_b)	
				self.target_qubit_write_maps_8L_mt.is_static   = True	
				self.target_qubit_write_maps_8L_mt_b.is_static = True
				
				####################################	
			
				
				#distributor_map_np = np.zeros( (320,), dtype=np.uint8 )
				#distributor_map_np[0:320:16] = 0
				#distributor_map_np[1:320:16] = 1
				#self.distributor_map_mt   = g.from_data(distributor_map_np, layout=layout_distributor_maps_west)
			

				self.distributor_map_mt = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                         dtype=g.uint8, 
                                                                         name="control_qubit_distributor_maps", 
                                                                         layout=self.ML.layout_distributor_maps_west)

				# distributor maps to select the correct memory maps for L and H target qubits 
				self.target_qubit_distributor_map_mt = g.input_tensor(shape=(self.ML.gate_count,320,), 
                                                                                      dtype=g.uint8, 
                                                                                      name="target_qubit_distributor_maps", 
                                                                                      layout=self.ML.layout_target_qubit_distributor_maps)
			
			
			
									
			
				g.add_mem_constraints([self.gate_counter_perm_map_selector_mt], [self.distributor_map_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.gate_counter_perm_map_selector_mt,self.distributor_map_mt], [self.target_qubit_distributor_map_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)



			
			with g.ResourceScope(name="split_gates_scope", is_buffered=True, predecessors=[distribute_state_indices_scope], time=None) as split_gates_scope :
		
				transposer_requests = []
				transposer_requests.append( g.tensor.create_transposer_request( [0], 1) )# request transposer on the west side
				transposer_requests.append( g.tensor.create_transposer_request( [2], 1) )# request transposer on the east side

			
				# create reusable permutor requests
				permutor_requests = []
				permutor_requests.append(  g.tensor.create_permutor_request( [0], 1 ) ) # WEST
				permutor_requests.append(  g.tensor.create_permutor_request( [1], 1 ) ) # EAST
			
				# create reusable distributor requests
				distributor_requests = []
				distributor_requests.append(  g.tensor.create_distributor_request( [0], 1 ) ) # WEST
				distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST		

				
				# concat real and imag parts to pipeline them at once through a shifter
				
				
				#gate_kernels_packed_mt = g.concat( [gate_kernels_real_packed_list[0].reshape(320,), gate_kernels_imag_packed_list[0].reshape(320,)], dim=0 )				
				gate_kernels_packed_mt = g.stack( [self.gate_kernels_real_packed_mt, self.gate_kernels_imag_packed_mt], dim=0 )	
				
				# get real and imaginary parts next to each other
				gate_kernels_packed_mt = gate_kernels_packed_mt.transpose(1,0,2)				
				gate_kernels_packed_st = gate_kernels_packed_mt.read( streams=g.SG4[0], time=50 )
								
				
				# preallocate tensors for intermediate memory tensors

				gate_kernels_real_separated_mt = g.constant_tensor((self.ML.gate_splits*80, 320), \
										dtype=g.float32, layout=self.ML.layout_gate_kernels_real_buffer, \
										initialized=False)	

				gate_kernels_imag_separated_mt = g.constant_tensor((self.ML.gate_splits*80, 320), \
										dtype=g.float32, layout=self.ML.layout_gate_kernels_imag_buffer, \
										initialized=False)	


				exclusion_tensors.append( gate_kernels_real_separated_mt )
				exclusion_tensors.append( gate_kernels_imag_separated_mt )

				
				gate_kernels_real_separated_mt_list = gate_kernels_real_separated_mt.split( dim=0, num_splits=self.ML.gate_splits*80 )
				gate_kernels_imag_separated_mt_list = gate_kernels_imag_separated_mt.split( dim=0, num_splits=self.ML.gate_splits*80 )	
				
				# split the tensor between hemispheres WEST and EAST (each (2k+1)-th gate kernel goes to the western hemispheres, while 2k-th gates are assigned to the eastern hemisphere
				# the gate kernel elements to be used later are the first 4 elements in each vector
				gate_kernels_packed_list = []
				gate_idx_max = 80
				for gate_idx in range(gate_idx_max):


					if gate_idx < 0:
						program_output = True
					else:
						program_output = False
	

					# concate preallocated memory storates
					buffer_real_mt = g.concat( gate_kernels_real_separated_mt_list[gate_idx::gate_idx_max], dim=0)
					buffer_imag_mt = g.concat( gate_kernels_imag_separated_mt_list[gate_idx::gate_idx_max], dim=0)
					buffer_mt      = g.stack( [buffer_real_mt, buffer_imag_mt], dim=0 )
					buffer_mt      = buffer_mt.transpose(1,0,2)
										
					gate_kernel_packed_mt = gate_kernels_packed_st.write(name=f"gate_kernel_packed_{gate_idx}", 
												storage_req=buffer_mt.storage_request, 
												program_output=program_output)	
										

					gate_kernels_packed_list.append( gate_kernel_packed_mt )
			
					
					if gate_idx < gate_idx_max-1 :
						permutor_request = permutor_requests[(gate_idx+1) % 2]
						shift = 4
						gate_kernels_packed_st = g.shift(gate_kernels_packed_st, 
									shift, 
									permutor_id=permutor_request, 
									shift_src=[inst.NEW_SRC] * 2, 
									dispatch_set=inst.DispatchSet.SET_1, 
									input_streams=[g.SG4[0]], 
									output_streams=[g.SG4[0]])	
					
				# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels									
				g.add_mem_constraints(gate_kernels_packed_list, gate_kernels_packed_list, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				
			with g.ResourceScope(name="split_gates_scope2", is_buffered=True, predecessors=[split_gates_scope], time=None) as split_gates_scope2 :
			
							
				gate_kernels_real_separated_mt = g.concat( gate_kernels_real_separated_mt_list[0:self.ML.gate_count], dim=0)
				gate_kernels_imag_separated_mt = g.concat( gate_kernels_imag_separated_mt_list[0:self.ML.gate_count], dim=0)				
				
				
				
				
				gate_kernels_real_copy_mt_dict = {}
				gate_kernels_imag_copy_mt_dict = {}
				gate_kernels_real_mt_dict      = {}
				gate_kernels_imag_mt_dict      = {}
				exclusion_tensors_real = []
				exclusion_tensors_imag = []
				exclusion_tensors_real_copy = []
				exclusion_tensors_imag_copy = []

				# preallocate tensors for the output    
				for idx in range( len(self.ML.gate_kernel_element_labels) ): # iterates over the labels of the gate kernel elements
				
					label          = self.ML.gate_kernel_element_labels[idx]
					address_offset = 256 + idx*256*(self.ML.gate_memory_offset_factor) # MXM permute matrices are also stored at the begining of these slices
					
					if (address_offset+self.ML.gate_count-1 < 4096):
						# the start and end points of the range are in the same bank				
						
						layout = f"H1(E), B1(0), A{self.ML.gate_count}({address_offset}-{address_offset+self.ML.gate_count-1})"
						
						real_copy = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(0-3)", 
															name = "gate_kernels_real_copy_"+label,
															initialized=False)
															
						imag_copy = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(4-7)", 
															name = "gate_kernels_imag_copy_"+label,
															initialized=False)
															
						real      = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(8-11)", 
															name = "gate_kernels_real_"+label,
															initialized=False)
															
						imag      = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(12-15)", 
															name ="gate_kernels_imag_"+label,
															initialized=False)	

					elif ( address_offset >= 4096 ) :

						
						layout = f"H1(E), B1(1), A{self.ML.gate_count}({address_offset-4096}-{address_offset+self.ML.gate_count-1-4096})"
						
						real_copy = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(0-3)", 
															name = "gate_kernels_real_copy_"+label,
															initialized=False)
															
						imag_copy = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(4-7)", 
															name = "gate_kernels_imag_copy_"+label,
															initialized=False)
															
						real      = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(8-11)", 
															name = "gate_kernels_real_"+label,
															initialized=False)
															
						imag      = g.constant_tensor((self.ML.gate_count, 320), 
															dtype=g.float32, layout=layout + ", S4(12-15)", 
															name ="gate_kernels_imag_"+label,
															initialized=False)	

					else:    
						# the start and end points of the range are in different banks
						
						layout_part1 = f"H1(E), B1(0), A{4096-address_offset}({address_offset}-4095)"
						layout_part2 = f"H1(E), B1(1), A{self.ML.gate_count+address_offset-4096}(0-{self.ML.gate_count+address_offset-4096-1})"	
						real_copy_part1 = g.constant_tensor((4096-address_offset, 320), 
															dtype=g.float32, layout=layout_part1 + ", S4(0-3)", 
															name = "gate_kernels_real_copy_1"+label,
															initialized=False)
															
						real_copy_part2 = g.constant_tensor((self.ML.gate_count+address_offset-4096, 320), 
															dtype=g.float32, layout=layout_part2 + ", S4(0-3)", 
															name = "gate_kernels_real_copy_2"+label,
															initialized=False)					
															
						real_copy = g.concat( [real_copy_part1, real_copy_part2], dim=0)	
						
						
						
						imag_copy_part1 = g.constant_tensor((4096-address_offset, 320), 
															dtype=g.float32, layout=layout_part1 + ", S4(4-7)", 
															name = "gate_kernels_imag_copy_1"+label,
															initialized=False)
															
						imag_copy_part2 = g.constant_tensor((self.ML.gate_count+address_offset-4096, 320), 
															dtype=g.float32, layout=layout_part2 + ", S4(4-7)", 
															name = "gate_kernels_imag_copy_2"+label,
															initialized=False)					
															
						imag_copy = g.concat( [imag_copy_part1, imag_copy_part2], dim=0)	
						
						
						
						real_part1 = g.constant_tensor((4096-address_offset, 320), 
															dtype=g.float32, layout=layout_part1 + ", S4(8-11)", 
															name = "gate_kernels_real_1"+label,
															initialized=False)
															
						real_part2 = g.constant_tensor((self.ML.gate_count+address_offset-4096, 320), 
															dtype=g.float32, layout=layout_part2 + ", S4(8-11)", 
															name = "gate_kernels_real_2"+label,
															initialized=False)					
															
						real = g.concat( [real_part1, real_part2], dim=0)	
						
						
						
						imag_part1 = g.constant_tensor((4096-address_offset, 320), 
															dtype=g.float32, layout=layout_part1 + ", S4(12-15)", 
															name = "gate_kernels_imag_1"+label,
															initialized=False)
															
						imag_part2 = g.constant_tensor((self.ML.gate_count+address_offset-4096, 320), 
															dtype=g.float32, layout=layout_part2 + ", S4(12-15)", 
															name = "gate_kernels_imag_2"+label,
															initialized=False)					
															
						imag = g.concat( [imag_part1, imag_part2], dim=0)

						exclusion_tensors_real_copy.append( real_copy_part1 )
						exclusion_tensors_real_copy.append( real_copy_part2 )
						exclusion_tensors.append(real_copy_part1)
						exclusion_tensors.append(real_copy_part2)
						exclusion_tensors_imag_copy.append( imag_copy_part1 )
						exclusion_tensors_imag_copy.append( imag_copy_part2 )
						exclusion_tensors.append(imag_copy_part1)
						exclusion_tensors.append(imag_copy_part2)
						exclusion_tensors_real.append( real_part1 )
						exclusion_tensors_real.append( real_part2 )
						exclusion_tensors.append(real_part1)	
						exclusion_tensors.append(real_part2)
						exclusion_tensors_imag.append( imag_part1 )
						exclusion_tensors_imag.append( imag_part2 )
						exclusion_tensors.append(imag_part1)	
						exclusion_tensors.append(imag_part2)


						
					real_copy.is_static = True
					imag_copy.is_static = True
					real.is_static = True
					imag.is_static = True															
						
					gate_kernels_real_copy_mt_dict[label] = real_copy
					exclusion_tensors_real_copy.append( real_copy )
					exclusion_tensors.append(real_copy)
					
					
					gate_kernels_imag_copy_mt_dict[label] = imag_copy
					exclusion_tensors_imag_copy.append( imag_copy )					
					exclusion_tensors.append(imag_copy)
					

					gate_kernels_real_mt_dict[label] = real
					exclusion_tensors_real.append( real )					
					exclusion_tensors.append(real)	
						
					
					gate_kernels_imag_mt_dict[label] = imag
					exclusion_tensors_imag.append( imag )					
					exclusion_tensors.append(imag)	
						
	


				# generate gate kernels to be applied on state |0> and resulting to alternating states |0> and |1> according to the periodicity of the current target qubit

				# create distributor maps for the process to broadcast elements 00,01,10,11 of the gate kernel onto a 320 element vectors
				distributor_00_np = np.zeros( (320,), dtype=np.uint8 )
				distribute_map_tensor_mt_00 = g.from_data( np.asarray(distributor_00_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_distribute_map_tensor + f", A1({1320+self.ML.small_qbit_num_limit})" )

				distributor_01_np = np.ones( (320,), dtype=np.uint8 )
				distribute_map_tensor_mt_01 = g.from_data( np.asarray(distributor_01_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_distribute_map_tensor + f", A1({1320+self.ML.small_qbit_num_limit+1})" )

				distributor_10_np = np.ones( (320,), dtype=np.uint8 )*2
				distribute_map_tensor_mt_10 = g.from_data( np.asarray(distributor_10_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_distribute_map_tensor + f", A1({1320+self.ML.small_qbit_num_limit+2})" )

				distributor_11_np = np.ones( (320,), dtype=np.uint8 )*3
				distribute_map_tensor_mt_11 = g.from_data( np.asarray(distributor_11_np, dtype=np.uint8), 
                                                                           layout=self.ML.layout_distribute_map_tensor + f", A1({1320+self.ML.small_qbit_num_limit+3})" )

		
				distribute_map_tensor_list = [distribute_map_tensor_mt_00, distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11]

				exclusion_tensors.append( distribute_map_tensor_mt_00 )
				exclusion_tensors.append( distribute_map_tensor_mt_01 )
				exclusion_tensors.append( distribute_map_tensor_mt_10 )
				exclusion_tensors.append( distribute_map_tensor_mt_11 )

				

				gate_kernels_broadcasted_imag_list = []
				gate_kernels_broadcasted_real_list = []
				gate_kernels_broadcasted_imag_copy_list = []
				gate_kernels_broadcasted_real_copy_list = []
				#loop_time_long_cycle = 51 # comming from the iterations of the previous cycles (the loops are overlapping in execution)
				#loop_time_short_cycle = 31
				
				
				
				
				for idx in range(len(self.ML.gate_kernel_element_labels) ): # iterates over the labels of the gate kernel elements

					label = self.ML.gate_kernel_element_labels[idx]
					
					gate_kernels_separated_mt = g.stack( [gate_kernels_real_separated_mt, gate_kernels_imag_separated_mt], dim=0 )
					gate_kernel_st = gate_kernels_separated_mt.read( streams=g.SG4[1], time=None )
					
					distributor_request = distributor_requests[1]
					gate_kernel_broadcasted_st = g.distribute_8( gate_kernel_st, 
							distribute_map_tensor_list[ idx ],
							distributor_req=distributor_request,
							map_stream_req=g.SG1[12],
							bypass8=0b00001111,
					)
					
					#gate_kernel_broadcasted_st = gate_kernel_st
					
					permutor_request = permutor_requests[1]
					gate_kernel_broadcasted_st = g.broadcast_lane_0( gate_kernel_broadcasted_st, 
										permutor_request,
										old_bitmap=[inst.NEW_SRC] * 4, 
										mask_bitmap=0b0000,
										input_streams=[g.SG4[3]], 
										output_streams=[g.SG4[2]]
					)
					
					gate_kernels_real_mt      = gate_kernels_real_mt_dict[ label ]
					gate_kernels_imag_mt      = gate_kernels_imag_mt_dict[ label ]
					gate_kernels_real_copy_mt = gate_kernels_real_copy_mt_dict[ label ]
					gate_kernels_imag_copy_mt = gate_kernels_imag_copy_mt_dict[ label ]
					
					gate_kernels_mt      = g.stack( [gate_kernels_real_mt, gate_kernels_imag_mt], dim=0 )
					gate_kernels_copy_mt = g.stack( [gate_kernels_real_copy_mt, gate_kernels_imag_copy_mt], dim=0 )
					
					gate_kernel_broadcasted_st.write( storage_req=gate_kernels_mt.storage_request, time= idx*(2*self.ML.gate_count + 1) )
					gate_kernel_broadcasted_st.write( storage_req=gate_kernels_copy_mt.storage_request )

					
				for label in self.ML.gate_kernel_element_labels:

					gate_kernels_real_mt      = gate_kernels_real_mt_dict[ label ]
					gate_kernels_imag_mt      = gate_kernels_imag_mt_dict[ label ]
					gate_kernels_real_copy_mt = gate_kernels_real_copy_mt_dict[ label ]
					gate_kernels_imag_copy_mt = gate_kernels_imag_copy_mt_dict[ label ]

					gate_kernels_real_mt_list      = gate_kernels_real_mt.split( dim=0, num_splits=self.ML.gate_count )
					gate_kernels_imag_mt_list      = gate_kernels_imag_mt.split( dim=0, num_splits=self.ML.gate_count )
					gate_kernels_real_copy_mt_list = gate_kernels_real_copy_mt.split( dim=0, num_splits=self.ML.gate_count )
					gate_kernels_imag_copy_mt_list = gate_kernels_imag_copy_mt.split( dim=0, num_splits=self.ML.gate_count )

					for gate_idx in range(self.ML.gate_count):

						self.gate_kernels_broadcasted_real_dict[f"{gate_idx}_"+label] = gate_kernels_real_mt_list[ gate_idx ]
						self.gate_kernels_broadcasted_imag_dict[f"{gate_idx}_"+label] = gate_kernels_imag_mt_list[ gate_idx ]
						self.gate_kernels_broadcasted_real_copy_dict[f"{gate_idx}_"+label] = gate_kernels_real_copy_mt_list[ gate_idx ]
						self.gate_kernels_broadcasted_imag_copy_dict[f"{gate_idx}_"+label] = gate_kernels_imag_copy_mt_list[ gate_idx ]

						self.gate_kernels_broadcasted_real_dict[f"{gate_idx}_"+label].is_static = True
						self.gate_kernels_broadcasted_imag_dict[f"{gate_idx}_"+label].is_static = True
						self.gate_kernels_broadcasted_real_copy_dict[f"{gate_idx}_"+label].is_static = True
						self.gate_kernels_broadcasted_imag_copy_dict[f"{gate_idx}_"+label].is_static = True

				
				self.gate_kernels_broadcasted_real_mt = g.concat( [gate_kernels_real_mt_dict["00"], 
                                                                                  gate_kernels_real_mt_dict["01"], 
                                                                                  gate_kernels_real_mt_dict["10"], 
                                                                                  gate_kernels_real_mt_dict["11"]], dim=0 )
				
				self.gate_kernels_broadcasted_imag_mt = g.concat( [gate_kernels_imag_mt_dict["00"], 
                                                                                  gate_kernels_imag_mt_dict["01"], 
                                                                                  gate_kernels_imag_mt_dict["10"], 
                                                                                  gate_kernels_imag_mt_dict["11"]], dim=0 )

				self.gate_kernels_broadcasted_real_copy_mt = g.concat( [gate_kernels_real_copy_mt_dict["00"], 
                                                                                  gate_kernels_real_copy_mt_dict["01"], 
                                                                                  gate_kernels_real_copy_mt_dict["10"], 
                                                                                  gate_kernels_real_copy_mt_dict["11"]], dim=0 )

				self.gate_kernels_broadcasted_imag_copy_mt = g.concat( [gate_kernels_imag_copy_mt_dict["00"], 
                                                                                  gate_kernels_imag_copy_mt_dict["01"], 
                                                                                  gate_kernels_imag_copy_mt_dict["10"], 
                                                                                  gate_kernels_imag_copy_mt_dict["11"]], dim=0 )

				
				self.gate_kernels_broadcasted_real_mt.is_static = True
				self.gate_kernels_broadcasted_imag_mt.is_static = True
				self.gate_kernels_broadcasted_real_copy_mt.is_static = True
				self.gate_kernels_broadcasted_imag_copy_mt.is_static = True
				

			g.add_mem_constraints(exclusion_tensors, exclusion_tensors, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			self.pgm_pkg.compile_program_context(pgm1)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)
		#############################################################################################################x
		# program to split the gate kernels and other metada between the EAST-WAST hemispheres 
