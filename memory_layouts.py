
qbit_num = 10

# label to indicate the 00, 01, 10, 11 elements of the gate kernel
gate_kernel_element_labels = ["00", "01", "10", "11"]

# gate count stored in a single packed vector 
gate_count = 1

# the number og qubits for which the gate operations need the permutor (need to reorganize the elements in a vector)
small_qbit_num_limit = 8

# layout to gather map selecting the permutation map corresponding to the target qubit  -- the MT should be replaced by on chip determined gather map, currently it is uploaded from CPU
layout_state_permute_map_selector   = f"H1(W), A{gate_count}(320-{320+gate_count-1}), S1(17)"
layout_state_1_selector				= f"H1(E), A{gate_count}(0-{gate_count-1}), S1(19)" 
layout_state_0_selector				= f"H1(E), A{gate_count}(0-{gate_count-1}), S1(21)"

layout_gate_kernels_real_packed ="H1(E), A1(320), S4(40-43)"
layout_gate_kernels_imag_packed	="H1(E), A1(321), S4(40-43)"

# memory layouts for splitted gate operations (Each 320byte vector contains one gate kernel in the first 4 elements)
# addresses from 0 to gate_count                   on each hemispheres are occupied with the broadcasted 00 (real or imag) element of the gate kernels
# addresses from gate_count to 2*(gate_count)    on EAST hemispheres are occupied with the broadcasted 01 (real or imag) element of the gate kernels
# addresses from 2*gate_count to 3*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from 3*gate_count to 4*(gate_count)  on EAST hemispheres are occupied with the broadcasted 10 (real or imag) element of the gate kernels
# addresses from gate_count*4 to gate_count*5    on EAST hemispheres are occupied with the shifted (and still packed) gate kernels  -- this is temporary used.
layout_gate_kernels_real		="H1(E), S4(8-11)"
layout_gate_kernels_imag		="H1(E), S4(12-15)"
layout_gate_kernels_real_copy 	="H1(E), S4(0-3)"
layout_gate_kernels_imag_copy	="H1(E), S4(4-7)"



# memory layout to store the vectors describing states |0> and |1> for different target qubits within a single 320 element vector
# addresses from 0 to small_qbit_num_limit-1       			are occupied by vectors describing states |0> and |1> for different target qubits (states |1> are labeled by bits 1)
# addresses from small_qbit_num_limit to 2*small_qbit_num_limit-1 	are occupied by vectors describing states |0> and |1> for different target qubits (states |0> are labeled by bits 1)
# addresses from 2*small_qbit_num_limit to 2*small_qbit_num_limit+3 	are occupied by distributor maps to used to broadcast elements 01,10,11 of the gate kernel
layout_states_1						= f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(17)"  
layout_states_0						= f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(18)"
layout_distribute_map_tensor    	= "H1(E), S1(17)"
layout_distribute_map_tensor_state1 = "H1(E), S1(17)"
layout_distribute_map_tensor_state0 = "H1(E), S1(18)"

layout_state_1_broadcasted			= f"H1(E), A{gate_count}(320-{320+gate_count-1}), S4(22-26)" # slice 24 is reserved
layout_state_0_broadcasted			= f"H1(E), A{gate_count}(320-{320+gate_count-1}), S4(27-31)" # slice 28 is reserved


# permute maps at -- used to reorder the state vector elements according to the given target qubit
layout_permute_maps_west = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(W), S1(43)"
layout_permute_maps_east = f"A{small_qbit_num_limit}(0-{small_qbit_num_limit-1}), H1(E), S1(39)"

# determine the size of the part of the state vector handled in the inner split of the LPU tensors
matrix_size   = int(2**small_qbit_num_limit)
memory_addresses = 1 << (qbit_num-small_qbit_num_limit)


# memory layout to store the quantum state
layout_State_input_real 	=f"H1(W), A{memory_addresses >> 1}(0-{(memory_addresses >> 1)-1}), B2, S4(0-3)"
layout_State_input_imag 	=f"H1(W), A{memory_addresses >> 1}(0-{(memory_addresses >> 1)-1}), B2, S4(4-7)"
layout_State_input_real_copy 	=f"H1(W), A{memory_addresses >> 1}(0-{(memory_addresses >> 1)-1}), B2, S4(8-11)"
layout_State_input_imag_copy 	=f"H1(W), A{memory_addresses >> 1}(0-{(memory_addresses >> 1)-1}), B2, S4(12-15)"

