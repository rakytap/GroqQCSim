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


		target_qbit = 10

                #small_qbit_num_limit = 
		
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

					

					Psi_transformed_real_copy_mt_list = g.split( Psi_transformed_real_copy_mt, 1 )
					Psi_transformed_imag_copy_mt_list = g.split( Psi_transformed_imag_copy_mt, 1 )

					index_pair   = 1 << (target_qbit - small_qbit_num_limit )
					pairing_mask = index_pair
		

					print( 'oooooooooooooooooooo ', index_pair, memory_slices, len(Psi_transformed_real_copy_mt_list), Psi_transformed_real_copy_mt.shape )

					Psi_permuted_real_mt_list = []
					Psi_permuted_imag_mt_list = []
					for idx in range(memory_slices):


						pairing_idx = idx ^ pairing_mask
						print(idx, pairing_idx)

						Psi_permuted_real_mt_list.append( Psi_transformed_real_copy_mt_list[pairing_idx] )
						Psi_permuted_imag_mt_list.append( Psi_transformed_imag_copy_mt_list[pairing_idx] )

					Psi_permuted_real_mt = g.concat( Psi_permuted_real_mt_list )
					Psi_permuted_imag_mt = g.concat( Psi_permuted_imag_mt_list )



					layout_transformed_state_real = layout_State_input_real
					layout_transformed_state_imag = layout_State_input_imag
					layout_transformed_state_real_copy = layout_State_input_real_copy
					layout_transformed_state_imag_copy = layout_State_input_imag_copy


					Psi_permuted_real_st = Psi_permuted_real_mt.read( streams=[g.SG4_E[0]], time=0 )				
					Psi_permuted_imag_st = Psi_permuted_imag_mt.read( streams=[g.SG4_E[3]], time=0 )

					print( Psi_transformed_real_mt.shape )
					print( Psi_transformed_real_mt.physical_shape )


					Psi_transformed_real_st = g.vxm_identity( Psi_permuted_real_st, alus=[0], output_streams=g.SG4_E[0] )
					Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[1], output_streams=g.SG4_E[0] )
					Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[2], output_streams=g.SG4_E[0] )
					Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[3], output_streams=g.SG4_W[0] )


					Psi_transformed_imag_st = g.vxm_identity( Psi_permuted_imag_st, alus=[4], output_streams=g.SG4_E[3] )
					Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[5], output_streams=g.SG4_E[3] )
					Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[6], output_streams=g.SG4_E[3] )
					Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[7], output_streams=g.SG4_W[3] )

					Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_{gate_idx}", storage_req=self.state_real_request, program_output=True)
					g.add_mem_constraints(Psi_transformed_real_list, [Psi_transformed_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_list.append( Psi_transformed_real_mt )
					

					Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_{gate_idx}", storage_req=self.state_imag_request, program_output=True)
					g.add_mem_constraints(Psi_transformed_imag_list, [Psi_transformed_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_list.append( Psi_transformed_imag_mt )

					
			self.pgm_pkg.compile_program_context(pgm3)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
