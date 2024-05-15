from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

class Gate_8_21(Split_gates):
	def __init__(self, pgm_pkg):
		super().__init__(pgm_pkg)
	
	def gate_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_8_21"


		target_qbit = 14

                #small_qbit_num_limit = 

		alu_rqs = []
		for idx in range(16):
			alu_rqs.append( g.tensor.create_alu_request( alus=[idx] ) )
		
		with self.pgm_pkg.create_program_context(name) as pgm3:


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

			# list of tensors for mutual memory exclusions
			Psi_transformed_real_list = [Psi_transformed_real_mt]
			Psi_transformed_imag_list = [Psi_transformed_imag_mt]
			Psi_transformed_real_copy_list = [Psi_transformed_real_copy_mt]
			Psi_transformed_imag_copy_list = [Psi_transformed_imag_copy_mt]




			for gate_idx in range( 1 ):#gate_count ):
							
				
				with g.ResourceScope(name=f"apply_gate_scope_{gate_idx}", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :

					

					Psi_transformed_real_mt_list = g.split( Psi_transformed_real_mt, 1 )
					Psi_transformed_imag_mt_list = g.split( Psi_transformed_imag_mt, 1 )


					Psi_transformed_real_copy_mt_list = g.split( Psi_transformed_real_copy_mt, 1 )
					Psi_transformed_imag_copy_mt_list = g.split( Psi_transformed_imag_copy_mt, 1 )



					index_pair   = 1 << (target_qbit - small_qbit_num_limit )
					pairing_mask = index_pair


					half_memory_slices = memory_slices >> 1
					print("iiiiiiiiiiii ", half_memory_slices)



					stagger_time = 20

					if (stagger_time % 2 ) > 0:
						raise('Stegger time should be a multiple of 2')

					stagger_cycles = half_memory_slices // stagger_time
					if ( memory_slices % stagger_time ) > 0:
						stagger_cycles = stagger_cycles + 1 

					print("iiiiiiiiiiii ", stagger_time, stagger_cycles)


					mask_lower_bits = index_pair - 1 
					mask_upper_bits = 1 << ( qbit_num - target_qbit )
					mask_upper_bits = mask_upper_bits - 1
					mask_upper_bits = mask_upper_bits << (target_qbit - small_qbit_num_limit )



					index = 0
					time = 0
					for outer_idx in range(stagger_cycles):


						index_loc = index 

						# calculating the results for state 1 of the target qubit (lower row of gate kernel)
						# after passing 4 levels of AVX the resulting memory bank will alternating with the read bank of the original state vector
						# read-bank is alternating with respect to the state 0 and 1 of qubit 8. Bank1 contains only state 0 for qubit 8, while bank2 contains only state 1 for qubit 8
						# need to start the reading with state 0 of qubit 8
						for stagger_idx in range(stagger_time):

							lower_bits = index_loc & mask_lower_bits
							higher_bits = index_loc & mask_upper_bits

							state_index_target_qubit_0 = (higher_bits << 1) + lower_bits
							state_index_target_qubit_1 = state_index_target_qubit_0 + index_pair

							Psi_real_target_qubit_0_st = Psi_transformed_real_mt_list[ state_index_target_qubit_0 ].read( streams=[g.SG4_E[0]], time=time )
							Psi_imag_target_qubit_0_st = Psi_transformed_imag_mt_list[ state_index_target_qubit_0 ].read( streams=[g.SG4_E[3]], time=time )

							# this should come from the copy of the state vector
							#Psi_real_target_qubit_1_st = Psi_transformed_real_mt_list[ state_index_target_qubit_1 ].read( streams=[g.SG4_E[0]], time=time )
							#Psi_imag_target_qubit_1_st = Psi_transformed_imag_mt_list[ state_index_target_qubit_1 ].read( streams=[g.SG4_E[3]], time=time )

							Psi_real_target_qubit_0_st = g.vxm_identity( Psi_real_target_qubit_0_st, alus=alu_rqs[0], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_0_st = g.vxm_identity( Psi_real_target_qubit_0_st, alus=alu_rqs[1], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_0_st = g.vxm_identity( Psi_real_target_qubit_0_st, alus=alu_rqs[2], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_0_st = g.vxm_identity( Psi_real_target_qubit_0_st, alus=alu_rqs[3], output_streams=g.SG4_W[0] )

		
							Psi_imag_target_qubit_0_st = g.vxm_identity( Psi_imag_target_qubit_0_st, alus=alu_rqs[4], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_0_st = g.vxm_identity( Psi_imag_target_qubit_0_st, alus=alu_rqs[5], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_0_st = g.vxm_identity( Psi_imag_target_qubit_0_st, alus=alu_rqs[6], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_0_st = g.vxm_identity( Psi_imag_target_qubit_0_st, alus=alu_rqs[7], output_streams=g.SG4_W[3] )


							# retrieve storage request for target qubit = 1 in the original state vector
							storage_rq       = Psi_transformed_real_mt_list[ state_index_target_qubit_1 ].storage_request
							transfromed_name = f"Psi_real_target_qubit_1_transformed_{gate_idx}"
							Psi_real_target_qubit_0_st.write(name=transfromed_name, storage_req=storage_rq)

							storage_rq       = Psi_transformed_real_copy_mt_list[ state_index_target_qubit_1 ].storage_request
							transfromed_name = f"Psi_real_target_qubit_1_transformed_copy_{gate_idx}"
							Psi_real_target_qubit_0_st.write(name=transfromed_name, storage_req=storage_rq)

							# retrieve storage request for target qubit = 1 in the original state vector
							storage_rq       = Psi_transformed_imag_mt_list[ state_index_target_qubit_1 ].storage_request
							transfromed_name = f"Psi_imag_target_qubit_1_transformed_{gate_idx}"
							Psi_imag_target_qubit_0_st.write(name=transfromed_name, storage_req=storage_rq)

							storage_rq       = Psi_transformed_imag_copy_mt_list[ state_index_target_qubit_1 ].storage_request
							transfromed_name = f"Psi_imag_target_qubit_1_transformed_copy_{gate_idx}"
							Psi_imag_target_qubit_0_st.write(name=transfromed_name, storage_req=storage_rq)

	

							print( state_index_target_qubit_0, state_index_target_qubit_1 )

							index_loc = index_loc + 1
							time      = time + 1

							if index_loc >= half_memory_slices:
								break


						index_loc = index 

						# calculating the results for state 0 of the target qubit (upper row of gate kernel)
						# after passing 3 levels of AVX the resulting memory bank will alternating with the read bank of the original state vector
						# read-bank is alternating with respect to the state 0 and 1 of qubit 8. Bank1 contains only state 0 for qubit 8, while bank2 contains only state 1 for qubit 8
						# need to start the reading with state 1 of qubit 8
						for stagger_idx in range(stagger_time):

							lower_bits = index_loc & mask_lower_bits
							higher_bits = index_loc & mask_upper_bits

							state_index_target_qubit_0 = (higher_bits << 1) + lower_bits
							state_index_target_qubit_1 = state_index_target_qubit_0 + index_pair

							Psi_real_target_qubit_1_st = Psi_transformed_real_mt_list[ state_index_target_qubit_1 ].read( streams=[g.SG4_E[0]], time=time )
							Psi_imag_target_qubit_1_st = Psi_transformed_imag_mt_list[ state_index_target_qubit_1 ].read( streams=[g.SG4_E[3]], time=time )

							# this should come from the copy of the state vector
							#Psi_real_target_qubit_0_st = Psi_transformed_real_mt_list[ state_index_target_qubit_0 ].read( streams=[g.SG4_E[0]], time=time )
							#Psi_imag_target_qubit_0_st = Psi_transformed_imag_mt_list[ state_index_target_qubit_0 ].read( streams=[g.SG4_E[3]], time=time )

							Psi_real_target_qubit_1_st = g.vxm_identity( Psi_real_target_qubit_1_st, alus=alu_rqs[0], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_1_st = g.vxm_identity( Psi_real_target_qubit_1_st, alus=alu_rqs[1], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_1_st = g.vxm_identity( Psi_real_target_qubit_1_st, alus=alu_rqs[2], output_streams=g.SG4_E[0] )
							Psi_real_target_qubit_1_st = g.vxm_identity( Psi_real_target_qubit_1_st, alus=alu_rqs[3], output_streams=g.SG4_W[0] )

		
							Psi_imag_target_qubit_1_st = g.vxm_identity( Psi_imag_target_qubit_1_st, alus=alu_rqs[4], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_1_st = g.vxm_identity( Psi_imag_target_qubit_1_st, alus=alu_rqs[5], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_1_st = g.vxm_identity( Psi_imag_target_qubit_1_st, alus=alu_rqs[6], output_streams=g.SG4_E[3] )
							Psi_imag_target_qubit_1_st = g.vxm_identity( Psi_imag_target_qubit_1_st, alus=alu_rqs[7], output_streams=g.SG4_W[3] )


							# retrieve storage request for target qubit = 0 in the original state vector
							storage_rq       = Psi_transformed_real_mt_list[ state_index_target_qubit_0 ].storage_request
							transfromed_name = f"Psi_real_target_qubit_0_transformed_{gate_idx}"
							Psi_real_target_qubit_1_st.write(name=transfromed_name, storage_req=storage_rq)

							storage_rq       = Psi_transformed_real_copy_mt_list[ state_index_target_qubit_0 ].storage_request
							transfromed_name = f"Psi_real_target_qubit_0_transformed_copy_{gate_idx}"
							Psi_real_target_qubit_1_st.write(name=transfromed_name, storage_req=storage_rq)


							# retrieve storage request for target qubit = 0 in the original state vector
							storage_rq       = Psi_transformed_imag_mt_list[ state_index_target_qubit_0 ].storage_request
							transfromed_name = f"Psi_imag_target_qubit_0_transformed_{gate_idx}"
							Psi_imag_target_qubit_1_st.write(name=transfromed_name, storage_req=storage_rq)

							storage_rq       = Psi_transformed_imag_copy_mt_list[ state_index_target_qubit_0 ].storage_request
							transfromed_name = f"Psi_imag_target_qubit_0_transformed_copy_{gate_idx}"
							Psi_imag_target_qubit_1_st.write(name=transfromed_name, storage_req=storage_rq)

							index_loc = index_loc + 1
							time      = time + 1

							if index_loc >= half_memory_slices:
								break


						index = index_loc

						if index >= half_memory_slices:
							break



					Psi_transformed_real_mt_output = g.constant_tensor(Psi_transformed_real_mt.shape, g.float32, storage_req=Psi_transformed_real_mt.storage_request, name=f"Psi_transformed_real_{gate_idx}", initialized=False)
					Psi_transformed_imag_mt_output = g.constant_tensor(Psi_transformed_imag_mt.shape, g.float32, storage_req=Psi_transformed_imag_mt.storage_request, name=f"Psi_transformed_imag_{gate_idx}", initialized=False)	

					Psi_transformed_real_mt_output.set_program_output()
					Psi_transformed_imag_mt_output.set_program_output()
		
					
					
			self.pgm_pkg.compile_program_context(pgm3)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
