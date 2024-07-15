from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

class Gate_8_21(Split_gates):
  def __init__(self, pgm_pkg, target_qubit=9):
        super().__init__(pgm_pkg)
        self.target_qubit = target_qubit

  def two_complex_dot_product(s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1):
        s_r_0_st = s_r_0.read(streams=g.SG4[4]) #direction=Direction.EASTWARD
        s_i_0_st = s_i_0.read(streams=g.SG4[6]) #direction=Direction.EASTWARD
        s_r_1_st = s_r_1.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        s_i_1_st = s_i_1.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        g_r_0_st = g_r_0.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        g_i_0_st = g_i_0.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        g_r_1_st = g_r_1.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        g_i_1_st = g_i_1.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        rr0_st = g.mul(s_r_0_st, g_r_0_st, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[4], g.SG4_W[7]], direction=Direction.WESTWARD
        ii0_st = g.mul(s_i_0_st, g_i_0_st, alus=[10], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[6], g.SG4_W[5]], direction=Direction.WESTWARD
        ri0_st = g.mul(s_r_0_st, g_i_0_st, alus=[7], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[4], g.SG4_W[5]], direction=Direction.WESTWARD
        ir0_st = g.mul(s_i_0_st, g_r_0_st, alus=[15], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[6], g.SG4_W[7]], direction=Direction.WESTWARD
        r0_st = g.sub(rr0_st, ii0_st, alus=[9], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[4], g.SG4_W[7]], direction=Direction.EASTWARD
        i0_st = g.add(ri0_st, ir0_st, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2], g.SG4_W[0]], direction=Direction.EASTWARD
        rr1_st = g.mul(s_r_1_st, g_r_1_st, alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_W[6]], direction=Direction.EASTWARD
        ii1_st = g.mul(s_i_1_st, g_i_1_st, alus=[4], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[5], g.SG4_W[3]], direction=Direction.EASTWARD
        ri1_st = g.mul(s_r_1_st, g_i_1_st, alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_W[3]], direction=Direction.EASTWARD
        ir1_st = g.mul(s_i_1_st, g_r_1_st, alus=[8], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[5], g.SG4_W[6]], direction=Direction.EASTWARD
        r1_st = g.sub(rr1_st, ii1_st, alus=[1], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[0], g.SG4_E[2]], direction=Direction.EASTWARD
        i1_st = g.add(ri1_st, ir1_st, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_E[7]], direction=Direction.EASTWARD
        r_st = g.add(r0_st, r1_st, alus=[6], output_streams=[g.SG4[5]], time=0) #input_streams=[g.SG4_E[5], g.SG4_E[2]], direction=Direction.WESTWARD
        i_st = g.add(i0_st, i1_st, alus=[3], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0], g.SG4_E[1]], direction=Direction.WESTWARD
        r_delay_st = g.vxm_identity(r_st, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[5]], direction=Direction.WESTWARD
        i_delay_st = g.vxm_identity(i_st, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[1]], direction=Direction.WESTWARD
        return r_st, i_st, r_delay_st, i_delay_st #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]

  def gate_program(self):
    ############################################################################################################
    # program to perform gate operation in a target qubit less than small_qbit_num_limit
    name = "gate_8_21"


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

      gate_kernels_broadcasted_real_shared_dict = {}
      gate_kernels_broadcasted_imag_shared_dict = {}
      gate_kernels_broadcasted_real_shared_copy_dict = {}
      gate_kernels_broadcasted_imag_shared_copy_dict = {}

      for gate_idx in range( 1 ):
        for label in gate_kernel_element_labels:
          key = f"{gate_idx}_"+label
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

        with g.ResourceScope(name=f"apply_gate_scope_{gate_idx}", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :
          pairs = [
              [x,x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8))
                ]
          # print("=============================>", pairs)
          # print(len(inner_dims))
          # print(Psi_transformed_real_mt.physical_shape, Psi_transformed_real_mt.shape)
          #Psi_transformed_mt (memory_addresses, matrix_size)
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
            pairs_to_concat0_0.append(inner_dims[i[0]])
            pairs_to_concat0_1.append(inner_dims[i[1]])
            pairs_to_concat1_0.append(inner_dims_copy[i[0]])
            pairs_to_concat1_1.append(inner_dims_copy[i[1]])

          #pairs_to_concat [(2, 1<<small_qbit_num_limit)]*(memory_addresses//2)
          pairs_to_concat0_0 = g.stack(pairs_to_concat0_0, dim=0)
          pairs_to_concat0_1 = g.stack(pairs_to_concat0_1, dim=0)
          pairs_to_concat1_0 = g.stack(pairs_to_concat1_0, dim=0)
          pairs_to_concat1_1 = g.stack(pairs_to_concat1_1, dim=0)
          #pairs_to_concat (memory_addresses//2, 2, matrix_size)

          pairs0 = g.stack([pairs_to_concat0_0, pairs_to_concat0_0], dim=0)
          pairs0 = [x.reshape(2, memory_addresses//2, matrix_size) for x in pairs0.split(num_splits=2, dim=2)]
          pairs1 = g.stack([pairs_to_concat1_1, pairs_to_concat1_1], dim=0)
          pairs1 = [x.reshape(2, memory_addresses//2, matrix_size) for x in pairs1.split(num_splits=2, dim=2)]
          #pairs (2, memory_addresses//2, 2, matrix_size) -> [(2, memory_addresses//2, matrix_size)]*2

          gr0 = g.stack(
                [g.stack(
                      [gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_"+label] for _ in range(memory_addresses//2)], dim=0
                      ) for label in ["00", "10"]
                  ], dim=0 )
          gi0 = g.stack(
                [g.stack(
                      [gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_"+label] for _ in range(memory_addresses//2)], dim=0
                      ) for label in ["00", "10"]
                  ], dim=0 )
          gr1 = g.stack(
                [g.stack(
                      [gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_"+label] for _ in range(memory_addresses//2)], dim=0
                      ) for label in ["01", "11"]
                  ], dim=0 )
          gi1 = g.stack(
                [g.stack(
                      [gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_"+label] for _ in range(memory_addresses//2)], dim=0
                      ) for label in ["01", "11"]
                  ], dim=0 )
          #g (2, memory_addresses//2, matrix_size)
          print(pairs0[0].physical_shape, pairs0[0].shape)
          print(gr0.physical_shape,gr0.shape)
     
          pairs0[0], pairs0[1], pairs1[0], pairs1[1] = Gate_8_21.two_complex_dot_product(
                  pairs1[0], pairs1[1], pairs0[0], pairs0[1], gr0, gi0, gr1, gi1)

          pairs_to_store0 = g.stack([pairs_to_concat0_0, pairs_to_concat0_1], dim=0)
          pairs_to_store1 = g.stack([pairs_to_concat1_0, pairs_to_concat1_1], dim=0)
          #pairs_to_store (2, memory_addresses//2, 2, matrix_size)
          print(pairs_to_store0.shape)

          pairs0_0, pairs0_1 = g.split(pairs_to_store0, num_splits=2, dim=2)
          pairs1_0, pairs1_1 = g.split(pairs_to_store1, num_splits=2, dim=2)
          #pairs (2, memory_addresses//2, 1, matrix_size)

          pairs0[0].write(storage_req=pairs0_0.storage_request, name="pairs0_0")
          pairs0[1].write(storage_req=pairs0_1.storage_request, name="pairs0_1")
          pairs1[0].write(storage_req=pairs1_0.storage_request, name="pairs1_0")
          pairs1[1].write(storage_req=pairs1_1.storage_request, name="pairs1_1")


          Psi_transformed_real_mt_output = g.constant_tensor(
                Psi_transformed_real_mt.shape, g.float32, storage_req=Psi_transformed_real_mt.storage_request, 
                    name=f"Psi_transformed_real_{gate_idx}", initialized=False)
          Psi_transformed_imag_mt_output = g.constant_tensor(
                Psi_transformed_imag_mt.shape, g.float32, storage_req=Psi_transformed_imag_mt.storage_request, name=f"Psi_transformed_imag_{gate_idx}", initialized=False)

          Psi_transformed_real_mt_output.set_program_output()
          Psi_transformed_imag_mt_output.set_program_output()
          gr0.name = "gr0"
          gr1.name = "gr1"
          gi0.name = "gi0"
          gi1.name = "gi1"
          gr0.set_program_output()
          gr1.set_program_output()
          gi0.set_program_output()
          gi1.set_program_output()
      self.pgm_pkg.compile_program_context(pgm3)

      #g.compile(base_name="QCsim", result_tensor=result)
      g.write_visualizer_data("QCsim_"+name)
      
