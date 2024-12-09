qbit_num = 17

# label to indicate the 00, 01, 10, 11 elements of the gate kernel
gate_kernel_element_labels = ["00", "01", "10", "11"]

# gate count stored in a single packed vector 
gate_count = 240
gate_splits = (gate_count-1+80) // 80  # on each 320-element vector 80 gate kernels are stored in the input
gate_memory_offset_factor = ((gate_count-1+256) // 256) + 2

# the number og qubits for which the gate operations need the permutor (need to reorganize the elements in a vector)
small_qbit_num_limit = 8


# determine the size of the part of the state vector handled in the inner split of the LPU tensors
matrix_size   = int(2**small_qbit_num_limit)
memory_slices = 1 << (qbit_num-small_qbit_num_limit) # TODO:replace all the occurances to memory_addresses
memory_addresses = 1 << (qbit_num-small_qbit_num_limit)

layout_target_qubit_read_maps_8L = f"H1(W), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(43)"
layout_target_qubit_read_maps_LL = f"H1(W), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(42)"

# memory gather maps for L target qubits
layout_target_qubit_read_pair_maps_L = f"H1(W), A{memory_addresses}(0-{memory_addresses-1}), S1(41)"
layout_target_qubit_read_maps_L = f"H1(W), A{memory_addresses}(0-{memory_addresses-1}), S1(40)"

# memory address maps to read out state vector components with control qbit = 1 state 
# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_slices)
# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_slices+1,...,memory_slices)
# each 2-byte column in the 16-byte blocks of the map tensor corresponds to a specific control qbit 	
layout_control_qubit_address_maps_SL_west = f"H1(W), A{memory_slices}(0-{memory_slices-1}), S1(39)"

layout_target_qubit_read_maps_8L_pair = f"H1(W), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(37)"


# gate counter: for reading out the correct permute map selector (each gate act on a different target qubit)
layout_gate_counter_perm_map_selector = f"H1(W), A{1}(3200), S1(36)"

# control qubit dependent distributor map for SL gate
address_offset = 256*(gate_memory_offset_factor)
layout_distributor_maps_west         = f"H1(W), A{gate_count}(0-{gate_count-1}), S1(36)"  #TODO HAVE A disptributor map also on the east
layout_target_qubit_distributor_maps = f"H1(W), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S1(36)"  #TODO HAVE A disptributor map also on the east

# permute maps used to reorder the state vector elements according to the given target qubit (permute map is also stored on the easter hemisphere)
layout_permute_maps_west = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(W), S1(35)"


# gate counter: for reading out the correct distributor map for choosing the correct L control qubit memory maps
layout_gate_counter_distributor_maps_west = "H1(W), A1(0), S1(34)"


#gate counter: for reading control_state_1_broadcasted and control_state_0_boradcasted (each gate act on a different target qubit)
layout_gate_counter_target_qubit_L_distributor_maps = f"H1(W), A{1}(0), S1(33)"



# memory layout for the mxm permutation marices (identical permutation matrices are also stored on the easterns hemisphere)
layout_mxm_permutation_matrices_west = f"H1(W), A{small_qbit_num_limit*20}(4096-{4096+small_qbit_num_limit*20-1}), S16(18-36)" 


# layout to gather map selecting the permutation map corresponding to the target qubit  -- the MT should be replaced by on chip determined gather map, currently it is uploaded from CPU
layout_constant_1_address_offset_lower_bits  = f"H1(W), A{1}(0), S1(18)"
layout_constant_1_address_offset_higher_bits = f"H1(W), A{1}(1), S1(18)"
layout_state_permute_map_selector            = f"H1(W), A{gate_count}(0-{gate_count-1}), S1(17)"
layout_mxm_permute_map_selector              = f"H1(W), A{gate_count}({gate_count}-{2*gate_count-1}), S1(17)" #either state permute map selector or mxm_permute_map selctor is used on the chip, but not booth

# if MXM is used to permute the state vector the single memory address need to be extended to address 20 memory addresses where the MXM permutation matrices are stored
layout_expanded_mxm_permute_map_selector = "H1(W), S1(17)"




# memory layout to store the quantum state
layout_State_input_real 	=f"H1(W), A{memory_slices >> 1}(0-{(memory_slices >> 1)-1}), B2, S4(0-3)"
layout_State_input_imag 	=f"H1(W), A{memory_slices >> 1}(0-{(memory_slices >> 1)-1}), B2, S4(4-7)"
layout_State_input_real_copy 	=f"H1(W), A{memory_slices >> 1}(0-{(memory_slices >> 1)-1}), B2, S4(8-11)"
layout_State_input_imag_copy 	=f"H1(W), A{memory_slices >> 1}(0-{(memory_slices >> 1)-1}), B2, S4(12-15)"

# memory layout for the mxm permutation marices (identical permutation matrices are also stored on the western hemisphere) 
layout_mxm_permutation_matrices_east = f"H1(E), A{small_qbit_num_limit*20}(0-{small_qbit_num_limit*20-1}), S16(0-15)"



# memory layouts for splitted gate operations (Each 320byte vector contains one gate kernel in the first 4 elements)
# addresses from 0 to gate_count                   on each hemispheres are occupied with the broadcasted 00 (real or imag) element of the gate kernels
# addresses from gate_count to 2*(gate_count)    on EAST hemispheres are occupied with the broadcasted 01 (real or imag) element of the gate kernels
# addresses from 2*gate_count to 3*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from 3*gate_count to 4*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from gate_count*4 to gate_count*5    on EAST hemispheres are occupied with the shifted (and still packed) gate kernels  -- this is temporary used.
layout_gate_kernels_real_copy 	="H1(E), S4(0-3)"
layout_gate_kernels_imag_copy	="H1(E), S4(4-7)"
layout_gate_kernels_real	="H1(E), S4(8-11)"
layout_gate_kernels_imag	="H1(E), S4(12-15)"







# memory layout to store the vectors describing states |0> and |1> for different target qubits within a single 320 element vector
# addresses from 0 to small_qbit_num_limit-1       			are occupied by vectors describing states |0> and |1> for different target qubits (states |1> are labeled by bits 1)
# addresses from small_qbit_num_limit to 2*small_qbit_num_limit-1 	are occupied by vectors describing states |0> and |1> for different target qubits (states |0> are labeled by bits 1)
# addresses from 2*small_qbit_num_limit to 2*small_qbit_num_limit+3 	are occupied by distributor maps to used to broadcast elements 01,10,11 of the gate kernel
layout_states_1			    = f"H1(E), A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), S1(32)"
layout_distribute_map_tensor        = "H1(E), S1(17)" #TODO move to another slice
layout_distribute_map_tensor_state1 = f"H1(E), S1(32), A1({small_qbit_num_limit+4})"  
layout_states_0			    = f"H1(E), A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), S1(33)"
layout_distribute_map_tensor_state0 = f"H1(E), S1(33), A1({small_qbit_num_limit+4})"


layout_state_1_selector		= f"H1(E), A{gate_count}(0-{gate_count-1}), S1(19)" 
layout_control_state_1_selector = f"H1(E), A{gate_count}({gate_count}-{2*gate_count-1}), S1(19)"

layout_state_0_selector	        = f"H1(E), A{gate_count}(0-{gate_count-1}), S1(21)"
layout_control_state_0_selector = f"H1(E), A{gate_count}({gate_count}-{2*gate_count-1}), S1(21)"

address_offset = 256*(gate_memory_offset_factor)
layout_state_1_broadcasted		   = f"H1(E), A{gate_count}(0-{gate_count-1}), S4(22-26)" # slice 24 is reserved
layout_control_state_1_broadcasted = f"H1(E), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S4(22-26)" # slice 24 is reserved
layout_state_0_broadcasted	       = f"H1(E), A{gate_count}(0-{gate_count-1}), S4(27-31)" # slice 28 is reserved
layout_control_state_0_broadcasted = f"H1(E), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S4(27-31)" # slice 28 is reserved

# to save controlled operations on target qbit 8, we need two concurrent streams for memory scatter (see layout_target_qubit_write_maps_8L_a)
#layout_target_qubit_write_maps_8L_b = f"H1(E), A{memory_addresses//2}({memory_addresses//2}-{memory_addresses-1}), S1(31)"
#layout_target_qubit_write_maps_8L_b = f"H1(E), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(17)"
layout_target_qubit_write_maps_8L_b = f"H1(E), A{(memory_slices//2) >> 1}(0-{((memory_slices//2) >> 1)-1}), B2, S1(42)"
#layout_target_qubit_write_maps_8L_c = f"H1(E), A{(memory_slices//2) >> 1}(0-{((memory_slices//2) >> 1)-1}), B2, S1(17)"

# memory layout for the tensors full of ones and zeros
layout_zeros = f"H1(E), A1(8001), S4(32-35)"
layout_ones = f"H1(E), A1(8002), S4(32-35)"

# temporary buffer space to store separated 2x2 gate kernels. The first 4 elements in each vector stand for the 2x2 kernel
address_offset = 4096
layout_gate_kernels_real_buffer = f"H1(E), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S4(27-31)"
#layout_gate_kernels_imag_buffer = f"H1(E), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S4(4-7)"
layout_gate_kernels_imag_buffer = f"H1(E), A{gate_count}({address_offset}-{address_offset+gate_count-1}), S4(32-35)"


# gate counter: for reading out broadcasted elements from the gate kernel
layout_gate_element_selector_10 = f"H1(E), A1({2*small_qbit_num_limit}), S1(33)"
layout_gate_element_selector_01 = f"H1(E), A1({2*small_qbit_num_limit}), S1(34)"
layout_gate_element_selector_11 = f"H1(E), A1({2*small_qbit_num_limit}), S1(35)"
layout_gate_element_selector_00 = f"H1(E), A1({2*small_qbit_num_limit}), S1(36)"



# initial gate kernels that are uploaded to the chip
layout_gate_kernels_real_packed ="H1(E), A3(7003-7005), S4(0-3)"
layout_gate_kernels_imag_packed	="H1(E), A3(7006-7008), S4(0-3)"




# permute maps used to reorder the state vector elements according to the given target qubit (permute map is also stored on the western hemisphere)
layout_permute_maps_east = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(37)"

#gate counter: for reading state_1_broadcasted and state_0_boradcasted (each gate act on a different target qubit)
layout_gate_counter_gate_filtering_selector = f"H1(E), A{1}({2*small_qbit_num_limit}), S1(37)"

#gate counter: for reading control_state_1_broadcasted and control_state_0_boradcasted (each gate act on a different target qubit)
layout_gate_counter_S_control_qbit_selector = f"H1(E), A{1}({2*small_qbit_num_limit+1}), S1(37)"





# memory address maps to read out state vector components with control qbit = 1 state 
# maps for control_qbits for 8,9,...,15 on the upper half of the memory tensor (addresses 0,...,half_memory_slices)
# maps for control_qbits for 16,...,21 on the lower half of the memory tensor (addresses half_memory_slices+1,...,memory_slices)
# each 2-byte column in the 16-byte blocks of the map tensor corresponds to a specific control qbit 	
layout_control_qubit_address_maps_SL_east_b = f"H1(E), -1, S1(38)"
layout_control_qubit_address_maps_SL_east_a = f"H1(E), A{memory_slices}(0-{memory_slices-1}), S1(39)"





# memory scatter maps for L target qubits
layout_target_qubit_write_maps_L = f"H1(E), A{memory_addresses}(0-{memory_addresses-1}), S1(40)"
# (41) saved for H target qubit maps

layout_target_qubit_write_maps_LL = f"H1(E), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(18)"

# to save controlled operations on target qbit 8, we need two concurrent streams for memory scatter (see layout_target_qubit_write_maps_8L_b)
layout_target_qubit_write_maps_8L_a = f"H1(E), A{memory_addresses//2}(0-{memory_addresses//2-1}), S1(43)"


