#include <groq/groqio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#ifdef TEST
#include <time.h>
#endif
#include <dirent.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>

typedef struct {
  float real;
  float imag;
} Complex32;

const char* program_strings[] = { "upload",  //0
                           "uplod_state_vector",  //1
                           "download_state_vector",  //2
                           "gate_S",  //3
                           "gate_SS",  //4
                           "gate_SL",  //5
                           "gate_S8",  //6
                           "gate_SH",  //7
                           "gate_L",  //8
                           "gate_LL",  //9
                           "gate_LS",  //10
                           "gate_8",  //11
                           "gate_8S",  //12
                           "gate_L8",  //13
                           "gate_8L",  //14
                           "gate_8H",  //15
                           "gate_LH",  //16
                           "gate_H",  //17
                           "gate_HS",  //18
                           "gate_HL",  //19
                           "gate_H8",  //20
                           "gate_HH",  //21
                           "state_vector_reset"   //22
                         }; //<target_qubit> or <target_qubit><control_qubit>

#define UPLOAD_IDX 0
#define UPLOAD_SV_IDX 1
#define DOWNLOAD_SV_IDX 2
#define GATE_S_IDX 3
#define GATE_SS_IDX 4
#define GATE_SL_IDX 5
#define GATE_S8_IDX 6
#define GATE_SH_IDX 7
#define GATE_L_IDX 8
#define GATE_LL_IDX 9
#define GATE_LS_IDX 10
#define GATE_8_IDX 11
#define GATE_8S_IDX 12
#define GATE_L8_IDX 13
#define GATE_8L_IDX 14
#define GATE_8H_IDX 15
#define GATE_LH_IDX 16
#define GATE_H_IDX 17
#define GATE_HS_IDX 18
#define GATE_HL_IDX 19
#define GATE_H8_IDX 20
#define GATE_HH_IDX 21
#define SV_RESET_IDX 22
    
// Numbers indicated with ? should be revised
//  qubitProgs[control_qbit][target_qbit] =  { {S, 8, L, H}, {SS, 8S, LS, HS}, {S8, 8, L8, H8}, {SL, 8L, LL, HL}, {SH, 8H, LH, HH} }  
int qubitProgs[][4] = { { 3, 11, 8, 17 }, { 4, 12, 10, 18 }, { 6, 11, 13, 20 }, { 5, 14, 9, 19 }, { 7, 15, 16, 21 } }; //+1 for non-single qubit, 0 for S(< 8), 1 for 8, 2 for H(> 8 < 16), 3 for H(>= 16) 
    
// input/output parameter strings for the individual programs
// "uplod_state_vector"
const char* state_vector_input_strings[] = { "State_input_real", "State_input_imag" };

// "download_state_vector"
const char* state_vector_output_strings[] = { "State_transformed_real", "State_transformed_imag" };

// "upload"
const char* state_vector_gate_upload_strings[] = { "gate_kernels_real_packed", "gate_kernels_imag_packed", //(0, 0), (0, 1), (1, 0), (1, 1)
    "state_0_selector", "state_1_selector", "control_state_0_selector", "control_state_1_selector",
    "target_qubit_distributor_maps", "control_qubit_distributor_maps", "permutation_selector", "mxm_permutation_selector" };

#define NUM_PROGS (sizeof(program_strings) / sizeof(const char*))
#define NUM_GATE_INPUTS (sizeof(state_vector_gate_upload_strings) / sizeof(const char*))
#define NUM_INPUTS (sizeof(state_vector_input_strings) / sizeof(const char*))
#define NUM_OUTPUTS (sizeof(state_vector_output_strings) / sizeof(const char*))
    
#define QUEUE_LIMIT 1
//#define QUEUE_LIMIT 1
#define MAX_DEVICE 8

typedef struct {
    Driver driver;
    unsigned int num_devices; //0 is not loaded
    Device device[MAX_DEVICE];
    unsigned char* iopData[MAX_DEVICE];
    IOP prog[MAX_DEVICE];
    IOBufferArray inputBuffers[MAX_DEVICE][NUM_PROGS];
    IOBufferArray outputBuffers[MAX_DEVICE][NUM_PROGS];
    TensorLayout gateLayouts[MAX_DEVICE][NUM_GATE_INPUTS];
    TensorLayout inpLayouts[MAX_DEVICE][NUM_INPUTS];
    TensorLayout outpLayouts[MAX_DEVICE][NUM_OUTPUTS];
    unsigned int num_qubits[MAX_DEVICE]; //0 until IOP is loaded
    unsigned int max_gates[MAX_DEVICE]; //this should be IOP metadata or filename, or a formula like this
} groq_sv_state;

groq_sv_state state = {}; //only one global variable, easy to move to the init/relieve interfaces

extern "C" void releive_groq_sv()
{
    if (!state.num_devices) return;
    Status status;

    for (size_t i = 0; i < state.num_devices; i++) {
        if (state.num_qubits[i]) {
            status = groq_iop_deinit(state.prog[i]);
            free(state.iopData[i]);
            for (unsigned int j = 0; j < NUM_PROGS; j++) {  
                status = groq_deallocate_iobuffer_array(state.inputBuffers[i][j]);
                if (status) {
                    printf("ERROR: groq deallocate iobuffer error %d\n", status);
                }
                status = groq_deallocate_iobuffer_array(state.outputBuffers[i][j]);
                if (status) {
                    printf("ERROR: groq deallocate iobuffer error %d\n", status);
                }
            }
            state.num_qubits[i] = 0;
        }
    }


    for (size_t i = 0; i < state.num_devices; i++) {
        status = groq_device_close(state.device[i]);
        if (status) {
            printf("ERROR: close device error %d\n", status);
        }
    }

    status = groq_deinit(&state.driver);
    if (status) {
        printf("ERROR: close device error %d\n", status);
    }
    state.num_devices = 0;
}




/**
@brief Call to get the number of available DFEs.
*/
extern "C" size_t get_accelerator_avail_num_sv() {
  Status status = groq_init(&state.driver);
  if (status) {
    printf("ERROR: groq init error %d\n", status);
    return 0;
  }
  unsigned int num_devices = 0;
  status = groq_get_number_of_devices(state.driver, &num_devices);
  if (status) {
    printf("ERROR: get num of devices error %d\n", status);
    return 0;
  }
    status = groq_deinit(&state.driver);
    if (status) {
        printf("ERROR: close device error %d\n", status);
    }  
  return num_devices;
}

/**
@brief Call to get the number of free DFEs.
*/
extern "C" size_t get_accelerator_free_num_sv() {
  Status status = groq_init(&state.driver);
  if (status) {
    printf("ERROR: groq init error %d\n", status);
    return 0;
  }
  unsigned int num_devices = 0, free_devices = 0;
  status = groq_get_number_of_devices(state.driver, &num_devices);
  if (status) {
    printf("ERROR: get num of devices error %d\n", status);
    return 0;
  }
  for (unsigned int i = 0; i < num_devices; ++i) {
      status = groq_get_nth_device(state.driver, i, &state.device[i]);
      if (status) {
        printf("ERROR: get nth device error %d, i = %d\n", status, i);
        continue;
      }

      /*status = groq_device_open(state.device[i]);
      if (status){
        printf("ERROR: device %d open error %d\n", i, status);
        continue;
      }*/

      if(groq_device_is_locked(state.device[i])) {
        printf("ERROR: device is locked\n");
      } else free_devices++;

      /*status = groq_device_close(state.device[i]);
      if (status) {
          printf("ERROR: close device error %d\n", status);
      }*/
  }
  status = groq_deinit(&state.driver);
  if (status) {
      printf("ERROR: close device error %d\n", status);
  }  
  return free_devices;
}

extern "C" int initialize_groq_sv( int accelerator_num_in )
{
  releive_groq_sv();
  Status status = groq_init(&state.driver);
  if (status) {
    printf("ERROR: groq init error %d\n", status);
    return 1;
  }

    //allocate devices
  unsigned int num_devices = 0;
  status = groq_get_number_of_devices(state.driver, &num_devices);
  if (status) {
    printf("ERROR: get num of devices error %d\n", status);
    return 1;
  }
  state.num_devices = 0;
  // Find the size of the file and allocate corresponding buffer
  for (unsigned int i = 0; i < num_devices; ++i) {
      /***** START DEVICE OBJECT TESTING *****/
      //status = groq_get_next_available_device(state.driver, &state.device[state.num_devices]);
      status = groq_get_nth_device(state.driver, i, &state.device[state.num_devices]);
      if (status) {
        printf("ERROR: get nth device error %d, i = %d\n", status, i);
        return 1;
      }

      char const *bus_addr = NULL;
      status = groq_device_bus_address(state.device[state.num_devices], &bus_addr);
      if (status) {
        printf("ERROR: device bus address error %d\n", status);
        return 1;
      }

      status = groq_device_open(state.device[state.num_devices]);
      if (status){
        printf("Device %d open failed with error %d, moving to the next device\n", i, status);
        continue; //return 1;
      }

      if(!groq_device_is_locked(state.device[state.num_devices])) {
        printf("ERROR: device %d is not locked\n", i);
        //return 1;
      }

      unsigned char const *ecid = NULL;
      status = groq_device_ecid(state.device[state.num_devices], &ecid);
      if (status) {
        printf("ERROR: device ecid error %d\n", status);
        return 1;
      }

      unsigned long csr_value = 0;
      status = groq_device_read_csr(state.device[state.num_devices], 0x18, &csr_value);
      if (status) {
        printf("ERROR: device read csr error %d\n", status);
        return 1;
      }

      unsigned int dcr_value = 0;
      status = groq_device_read_dcr(state.device[state.num_devices], 0x18, &dcr_value);
      if (status) {
        printf("ERROR: device read dcr error %d\n", status);
        return 1;
      }

      int curr_proc_id = 0;
      status = groq_device_curr_proc_id(state.device[state.num_devices], &curr_proc_id);
      if (status) {
        printf("ERROR: device curr proc id error %d\n", status);
        return 1;
      }

      int numa_node = 0;
      status = groq_device_numa_node(state.device[state.num_devices], &numa_node);
      if (status) {
        printf("ERROR: device numa node error %d\n", status);
        return 1;
      }

      //printf("device %d %p, %s, %d, %d\n",
      //        i, device[state.num_devices], bus_addr, curr_proc_id, numa_node);
      /***** END DEVICE OBJECT TESTING *****/
      status = groq_device_clear_memory(state.device[state.num_devices]);
      if (status) {
        printf("ERROR: device clear memory error %d\n", status);
        return 1;
      }
      status = groq_device_clear_faults(state.device[state.num_devices]);
      if (status) {
        printf("ERROR: device clear faults error %d\n", status);
        return 1;
      }
      state.num_devices++;
      if (state.num_devices == (unsigned int)accelerator_num_in) break;
    }
    if (state.num_devices != (unsigned int)accelerator_num_in) {
        printf("ERROR: number of requested devices not acquired\n");
        return 1;
    }    
    return 0;
}

int lookup_str_idx(const char* str, const char* strs[], int num_strs)
{
    for (int idx = 0; idx < num_strs; idx++) {
        if (strcmp(str, strs[idx]) == 0) return idx;
    }
    return -1;
}

int load_iop(size_t num_qubits, int device_idx)
{
    if (num_qubits == state.num_qubits[device_idx]) return 0;

    static bool path_fixed = false;
    if (!path_fixed) { 
        Dl_info dl_info;
        dladdr((void*)&initialize_groq_sv, &dl_info);
        char* fixpath = strdup(dl_info.dli_fname);
        fixpath = dirname(fixpath);
        strcat(fixpath, "/.."); //filename removed must be at least 3 characters
        if (chdir(fixpath) != 0) { 
            printf("Could not change working directory to %s\n", fixpath); 
        }
        free(fixpath);
        path_fixed = true;
    }
    char fname[PATH_MAX];
    unsigned int cur_qubits, max_gates, n;
    DIR* d;
    struct dirent* dir;
    d = opendir("build_iop/");
    if (d) {
        while((dir = readdir(d)) != NULL) {
            if (dir->d_type == DT_REG) {

                if (sscanf(dir->d_name, "QCsim_multi_program.%u.%u.0.iop%n", &cur_qubits, &max_gates, &n) == 2 &&
                    n == strlen(dir->d_name) && num_qubits == cur_qubits) {

                    strcpy(fname, "build_iop/");
                    strcat(fname, dir->d_name);
                    break;
                } 
                else {
                    cur_qubits = 0;
                }
            }
        }
        closedir(d);
    }


    if (device_idx > MAX_DEVICE)  {
        printf("Device idx: %d, while max device: %d\n",device_idx, MAX_DEVICE  );
        return 1;    
    }

    if (cur_qubits == 0)  {
        printf("cur_qubits: %d\n", cur_qubits  );
        return 1;    
    }

    struct stat st;
    stat(fname, &st);
    unsigned int prog_size = st.st_size;                        
    FILE* fp = fopen(fname, "rb");

    if (fp == NULL) {
        printf("Cant open file %s\n", fname);
        return 1; 
    }


    state.iopData[device_idx] = (unsigned char*)malloc(prog_size);   // freed
    size_t check = fread(state.iopData[device_idx], 1, prog_size, fp);
    fclose(fp);
    if (check != prog_size) { 
        free(state.iopData[device_idx]); 
        return 1; 
    }

    Status status = groq_iop_init(state.iopData[device_idx], prog_size, &state.prog[device_idx]);
    if (status) {
        printf("ERROR: IOP init %d\n", status);
        return 1;
    }

    unsigned int num_programs = 0;
    status = groq_iop_get_number_of_programs(state.prog[device_idx], &num_programs);
    if (status) {
      printf("ERROR: iop get number of programs error %d\n", status);
      return 1;
    }
      //printf("number of programs: %u\n", num_programs);

      for (unsigned int j = 0; j < num_programs; j++) {
        status = groq_load_program(state.device[device_idx], state.prog[device_idx], j, true);
        if (status) {
            printf("ERROR: load program %d\n", status);
            return 1;
        }

        //printf("loaded program: %u\n", j);
        char* name = NULL;
        status = groq_program_name(state.prog[device_idx], j, &name);
        if (status) {
          printf("ERROR: program name error %d\n", status);
          return 1;
        }
        int prog_idx = lookup_str_idx(name, program_strings, NUM_PROGS);
        if (prog_idx == -1) return 1;
        //printf("program %u name: %s\n", j, name);
        Program p;
        status = groq_get_nth_program(state.prog[device_idx], j, &p);
        if (status) {
            printf("ERROR: get nth program %d\n", status);
            return 1;
        }
        size_t n;
        status = groq_get_number_of_entrypoints(p, &n);
        if (status) {
            printf("ERROR: program get number of entrypoints %d\n", status);
            return 1;
        }
        size_t inpSize, outpSize;
        uint32_t ptindex;
        //for (size_t i = 0; i < n; i++) {
        for (size_t i = 0; i < 1; i++) {//only need the first entrypoint, there are 4 all with identical inputs/outputs
            EntryPoint ep;
            status = groq_get_nth_entrypoint(p, 0, &ep);
            if (status) {
                printf("ERROR: get nth entrypoint %d\n", status);
                return 1;
            }
            status = groq_entrypoint_get_ptindex(ep, &ptindex);
            if (status) {
                printf("ERROR: entrypoint get ptindex %d\n", status);
                return 1;
            }
            char* epname = NULL;
            status = groq_entrypoint_get_name(ep, &epname);
            if (status) {
                printf("ERROR: get entrypoint name %d\n", status);
                return 1;
            }
            //printf("entrypoint %lu ptindex %u\n", i, ptindex);
            IODescriptor iod;
            for (size_t inpoutp = 0; inpoutp < 2; inpoutp++) {
                if (inpoutp == 0) {
                    status = groq_entrypoint_get_input_iodescriptor(ep, &iod);
                    if (status) {
                        printf("ERROR: entrypoint get input iodescriptor %d\n", status);
                        return 1;
                    }        
                    status = groq_iodescriptor_get_size(iod, &inpSize);
                    if (status) {
                        printf("ERROR: iodescriptor get size %d\n", status);
                        return 1;
                    }                                            
                } else {
                    status = groq_entrypoint_get_output_iodescriptor(ep, &iod);
                    if (status) {
                        printf("ERROR: entrypoint get output iodescriptor %d\n", status);
                        return 1;
                    }                                            
                    status = groq_iodescriptor_get_size(iod, &outpSize);
                    if (status) {
                        printf("ERROR: iodescriptor get size %d\n", status);
                        return 1;
                    }                                            
                }
                size_t l;
                status = groq_iodescriptor_get_number_of_tensor_layouts(iod, &l);
                if (status) {
                    printf("ERROR: iodescriptor get number of tensor layouts %d\n", status);
                    return 1;
                }                                            
                for (size_t k = 0; k < l; k++) {
                    TensorLayout tl;
                    status = groq_iodescriptor_get_nth_tensor_layout(iod, k, &tl);                                                      
                    if (status) {
                        printf("ERROR: iodescriptor get nth tensor layout %d\n", status);
                        return 1;
                    }          
                    char* name;
                    status = groq_tensor_layout_get_name(tl, &name);
                    if (status) {
                        printf("ERROR: tensor layout get name %d\n", status);
                        return 1;
                    }
                    int tlidx;

                    if (prog_idx == 0) {
                        // gate upload program
                        tlidx = lookup_str_idx(name, state_vector_gate_upload_strings, NUM_GATE_INPUTS);
                        if (tlidx == -1) {
                            printf("Unknown tensor: %s\n", name);
                            return 1;
                        }
                        memcpy(&state.gateLayouts[device_idx][tlidx], &tl, sizeof(TensorLayout));
                    } 
                    else if (prog_idx == 1) {
                        // state vector upload program
                        tlidx = lookup_str_idx(name, state_vector_input_strings, NUM_INPUTS);
                        if (tlidx == -1) {
                            printf("Unknown tensor: %s\n", name);
                            return 1;
                        }
                        memcpy(&state.inpLayouts[device_idx][tlidx], &tl, sizeof(TensorLayout));
                    } 
                    else if (prog_idx == 2) {
                        // state vector download program 
                        tlidx = lookup_str_idx(name, state_vector_output_strings, NUM_OUTPUTS);
                        if (tlidx == -1) {
                            printf("Unknown tensor: %s\n", name);
                            return 1;
                        }
                        memcpy(&state.outpLayouts[device_idx][tlidx], &tl, sizeof(TensorLayout));

                    } 
                    else {
                        printf("Unknown tensor: %s in program %s\n", name, program_strings[prog_idx]);
                        return 1;
                    }
                    /*uint32_t type;
                    status = groq_tensor_layout_get_type(tl, &type);
                    if (status) {
                        printf("ERROR: tensor layout get type %d\n", status);
                        return 1;
                    }                                                                
                    size_t ndim;
                    status = groq_tensor_layout_get_number_of_dimensions(tl, &ndim);
                    if (status) {
                        printf("ERROR: tensor layout get number of dimensions %d\n", status);
                        return 1;
                    }                                                                
                    int32_t format;
                    status = groq_tensor_layout_get_format(tl, &format);
                    if (status) {
                        printf("ERROR: tensor layout get format %d\n", status);
                        return 1;
                    }                                                                
                    size_t size;
                    status = groq_tensor_layout_get_size(tl, &size);
                    if (status) {
                        printf("ERROR: tensor layout get size %d\n", status);
                        return 1;
                    }                                                                
                    printf("tensor %lu %s type %u %s ndim %lu format %d size %lu (", k, name, type, TENSOR_TYPES[type], ndim, format, size);   
                    for (size_t m = 0; m < ndim; m++) {
                        uint32_t dimSize;
                        status = groq_tensor_layout_get_nth_dimension(tl, m, &dimSize);
                        if (status) {
                            printf("ERROR: tensor layout get nth dimension %d\n", status);
                            return 1;
                        }                                                                
                        printf(m == ndim-1 ? "%u)\n" : "%u, ", dimSize); 
                    }*/
                }
            }
        }
        inpSize = groq_program_get_input_size(state.prog[device_idx], j);
        outpSize = groq_program_get_output_size(state.prog[device_idx], j);
        int numBufs = prog_idx <= 2 ? 1 : QUEUE_LIMIT;
        //printf("input size %lu output size %lu\n", inpSize, outpSize);

        //program table index entry point offsets: 0 is mono/(input, output, compute), 1 is (input only), 2 is (compute only), 3 is (output only)
        status = groq_allocate_iobuffer_array(state.driver, inpSize, numBufs, ptindex, &state.inputBuffers[device_idx][prog_idx]);
        if (status) {
            printf("ERROR: allocate iobuffer array %d\n", status);
            return 1;
        }

        status = groq_allocate_iobuffer_array(state.driver, outpSize, numBufs, ptindex, &state.outputBuffers[device_idx][prog_idx]);
        if (status) {
            printf("ERROR: allocate iobuffer array %d\n", status);
            return 1;
        }
      }

  state.num_qubits[device_idx] = num_qubits;
  state.max_gates[device_idx] = max_gates;
  //printf("Initialized Groq for %lu qbits on device %d\n", num_qubits, device_idx);
  return 0;
}

/**
@brief Call to either upload a state vector onto the chip, or initialize the state vector to (1,0,0,0,0,0, ... ) on the chip
@param data Pointer to the data of the state vector. If NULL pointer, the state (1,0,0,0,0,0, ... ) is initialized on the chip
@param num_qubits The number of qubits for which the IOP progra should be selected (an individual IOP program is compiled for different number of qubits)
@param device_num The number of devices onto which the IOP should be uploaded.
*/
extern "C" int prepare_state_vector(Complex32* data, size_t num_qubits, int device_num)
{
    // load IOP
    // the IOP is preserved on the chip if the same number of qubits (and gate max) are used
    // if the number of qubits is changed, the IOP on the chips are replaced
    if (load_iop(num_qubits, device_num)) { 
        printf("ERROR: prepare_state_vector: failed to upload IOP.\n");
        return 1; 
    }

    Completion completion;
    Status status;
    int completionCode;

    // in case no data are given to be uploaded, the special (1,0,0,0,0,0, ... ) initial state is initialized on the chips
    if ( data == NULL ) {

        status = groq_invoke(state.device[device_num], state.inputBuffers[device_num][SV_RESET_IDX], 0, state.outputBuffers[device_num][SV_RESET_IDX], 0, &completion);
        if (status) {
            printf("ERROR: state vector reset invoke failed %d\n", status); 
            return 1;
        }

        // wait for completion
        while (!(completionCode = groq_poll_completion(completion)));
        if (completionCode != GROQ_COMPLETION_SUCCESS) {
            printf("ERROR: state vector reset completion code %d\n", completionCode);
        }

        return 0;
    }

    // upload the state vector onto the chip

    
    unsigned char* input;
    status = groq_get_data_handle(state.inputBuffers[device_num][UPLOAD_SV_IDX], 0, &input);

    if (status) {
        printf("ERROR: get data handle %d\n", status);
    }

    float* realdata = (float*)malloc(sizeof(float)*(1<<num_qubits));// freed
    float* imagdata = (float*)malloc(sizeof(float)*(1<<num_qubits));// freed
    for (size_t i = 0; i < (size_t)(1<<num_qubits); i++) {
        realdata[i] = data[i].real;
        imagdata[i] = data[i].imag;
    }

    status = groq_tensor_layout_from_host(state.inpLayouts[device_num][0], (unsigned char*)realdata, sizeof(float)*(1<<num_qubits), input, sizeof(Complex32)*320*(1<<(num_qubits-8)));
    if (status) {
        printf("ERROR: tensor layout from host %d\n", status);
    }

    status = groq_tensor_layout_from_host(state.inpLayouts[device_num][1], (unsigned char*)imagdata, sizeof(float)*(1<<num_qubits), input, sizeof(Complex32)*320*(1<<(num_qubits-8)));
    if (status) {
        printf("ERROR: tensor layout from host %d\n", status);
    }

    free(realdata);
    free(imagdata);

    status = groq_invoke(state.device[device_num], state.inputBuffers[device_num][UPLOAD_SV_IDX], 0, state.outputBuffers[device_num][UPLOAD_SV_IDX], 0, &completion);
    if (status) {
        printf("ERROR: state vector upload invoke failed %d\n", status);
        return 1;
    }
  
    // wait for completion
    while (!(completionCode = groq_poll_completion(completion)));
    if (completionCode != GROQ_COMPLETION_SUCCESS) {
        printf("ERROR: state vector upload completion code %d\n", completionCode);
    }

    return 0;
}




// Define the structure for a (circular) doubly linked node
struct Node {
    int data;
    struct Node* next;
    struct Node* prev;
};

// Function to create a new node with given data
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node)); //freed
    newNode->data = data;
    newNode->next = NULL;
    newNode->prev = NULL;
    return newNode;
}

// Function to insert a node to the circular doubly-linked list
void insertNode(struct Node** head_ref, struct Node* newNode) {
    // If the list is empty, make the new node the head
    if (*head_ref == NULL) {
        newNode->prev = newNode->next = newNode;
        *head_ref = newNode;
        return;
    }

    struct Node* last = (*head_ref)->prev;
    // Change the next of last node to new node
    newNode->next = *head_ref;
    newNode->prev = last;
    (*head_ref)->prev = newNode;
    last->next = newNode;
}

void removeNode(struct Node** head_ref, struct Node* node) {
    if (*head_ref == NULL) return;
    if (node->next == node) {
        *head_ref = NULL;
        return;
    }  
    struct Node* prev = node->prev;
    if (node == *head_ref) {
        struct Node* prev = (*head_ref)->prev;
        *head_ref = (*head_ref)->next;
        prev->next = *head_ref;
        (*head_ref)->prev = prev;
    } else if (node->next == *head_ref) {
        struct Node* prev = node->prev;
        prev->next = *head_ref;
        (*head_ref)->prev = prev;
    } else {
        struct Node* temp = node->next;
        prev->next = temp;
        temp->prev = prev;
    }
    
}

// Function to transfer a node from source list to destination list
void transferNode(struct Node** source_ref, struct Node** dest_ref, struct Node* node) {
    if (node == NULL) return;  // If the key node is NULL, do nothing
    removeNode(source_ref, node);
    node->next = node->prev = NULL;
    insertNode(dest_ref, node);
}

// Function to free the linked list memory
void freeList(struct Node* node) {
    if (node == NULL) return;
    struct Node* temp = node;
    do {    
        Node* temp2 = temp->next;
        free(temp);
        temp = temp2;
    } while (node != temp);
}

//result size == num_qubits from load_sv
extern "C" int calcsvKernelGroq(int num_gates, Complex32* gates, int* target_qubits, int* control_qubits, float* realdata, float* imagdata, int device_num)
{
    Completion completion;
    Status status;
    int completionCode;
    int num_qubits = state.num_qubits[device_num];
    unsigned char* input;
    status = groq_get_data_handle(state.inputBuffers[device_num][0], 0, &input);

    if (status) {
        printf("ERROR: get data handle %d\n", status);    
    }

    unsigned int max_gates = state.max_gates[device_num];
    int packedSize = 2*2*max_gates;
    if (packedSize % 320) {
        packedSize += 320 - packedSize % 320;
    }

    float* gatereal = (float*)calloc(sizeof(float), packedSize);
    float* gateimag = (float*)calloc(sizeof(float), packedSize);


    unsigned char* mxm_perm_sel       = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* perm_sel           = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* state_0_sel        = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* state_1_sel        = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* control_0_sel      = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* control_1_sel      = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* target_qubit_dmap  = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);
    unsigned char* control_qubit_dmap = (unsigned char*)calloc(sizeof(unsigned char), max_gates*320);

    Completion* completionQueue = (Completion*)malloc(QUEUE_LIMIT*sizeof(Completion)); // freed
    Node* queueFree = NULL;
    Node* queueOccupied = NULL;


    for (int i = 0; i < QUEUE_LIMIT; i++) {
        insertNode(&queueFree, createNode(i));
    }

    int* bufferIndices = (int*)calloc(sizeof(int), NUM_PROGS-3);

    // breaks the gates into packedSize bunches
    for (unsigned int cur_gate = 0; cur_gate < (unsigned int)num_gates; cur_gate += max_gates) {
        unsigned int tot_gates = num_gates - cur_gate < max_gates ? num_gates - cur_gate : max_gates;
        //printf("cur_gate=%u tot_gates=%u num_gates=%d max_gates=%u\n", cur_gate, tot_gates, num_gates, max_gates);
        for (unsigned int i = 0; i < tot_gates; i++) {

            for (int j = 0; j < 4; j++) {
                gatereal[i*4+j] = gates[(cur_gate+i)*4+j].real;
                gateimag[i*4+j] = gates[(cur_gate+i)*4+j].imag;
            }

            int target_qbit = target_qubits[cur_gate+i];
            int control_qbit = control_qubits[cur_gate+i];

            for (int j = 0; j < 320; j+=16) {

                if (target_qbit < 8) {
                    // S gates
                    state_0_sel[i*320+j] = target_qbit;
                    state_1_sel[i*320+j] = target_qbit;
                    perm_sel[i*320+j] = target_qbit;
                }

                if (control_qbit < 8 && control_qbit >= 0) {
                    // SS, LS, HS gates
                    control_0_sel[i*320+j] = control_qbit;
                    control_1_sel[i*320+j] = control_qbit;
                }


                // control qubit distributor maps for SL, 8L, SH, 8H, LL, HL, LH, HH gates
                if (control_qbit > 8 && control_qbit <= 16 && target_qbit <= 8 ) {
                    // SL and 8L gates
                    control_qubit_dmap[i*320+j] = 2*(control_qbit - 9);
                    control_qubit_dmap[i*320+j+1] = 2*(control_qbit - 9) + 1;
                }
                else if (control_qbit > 16 && control_qbit < 21 && target_qbit <= 8) {
                    // SH and 8H gates
                    control_qubit_dmap[i*320+j] = 2*(control_qbit - 17);
                    control_qubit_dmap[i*320+j+1] = 2*(control_qbit - 17) + 1;
                }
                if (control_qbit > 8 && control_qbit <= 16 && target_qbit > 8 && target_qbit < 21) {
                    // LL and HL gate
                    int offset;
                    if (control_qbit == 16 && control_qbit < target_qbit ) {
                        offset = 9;
                    }
                    else if (control_qbit < target_qbit) {
                        offset = control_qbit;
                    }
                    else {
                        offset = control_qbit - 1;
                    }

                    control_qubit_dmap[i*320+j] = 2*(offset-9);
                    control_qubit_dmap[i*320+j+1] = 2*(offset-9) + 1;
                }
                else if (control_qbit > 16 && control_qbit < 21 && target_qbit > 8 && target_qbit < 21 ) {
                    // LH and HH gate
                    int offset;
                    if (control_qbit < target_qbit) {
                        offset = control_qbit;
                    }
                    else {
                        offset = control_qbit - 1;
                    }

                    control_qubit_dmap[i*320+j] = 2*(offset-9-7);
                    control_qubit_dmap[i*320+j+1] = 2*(offset-9-7) + 1;
                }
                else {
                    // when the control qubits is not L or H, no valid distributor map is required
                }

                // target qubit distributor maps for L and H gates
                if (target_qbit > 8 && target_qbit <= 16 ) {
                    target_qubit_dmap[i*320+j] = 2*(target_qbit - 8 - 1);
                    target_qubit_dmap[i*320+j+1] = 2*(target_qbit - 8 - 1) + 1;
                }
                else if (target_qbit > 16 && target_qbit < 21) {
                    target_qubit_dmap[i*320+j] = 2*(target_qbit - 17);
                    target_qubit_dmap[i*320+j+1] = 2*(target_qbit - 17) + 1;
                }
                else {
                    // when the target qubit is not L or H, no valid distributor map is required
                    // the map selectors are padded with zeros for these cases
                    target_qubit_dmap[i*320+j] = 0;
                    target_qubit_dmap[i*320+j+1] = 0;
                }
			
                /*
                if (target_qubits[cur_gate+i] < 8) {
                    // S gates
                    state_0_sel[i*320+j] = target_qubits[cur_gate+i];
                    state_1_sel[i*320+j] = target_qubits[cur_gate+i];
                    perm_sel[i*320+j] = target_qubits[cur_gate+i];
                }
                if (control_qubits[cur_gate+i] < 8 && control_qubits[cur_gate+i] >= 0) {
                    // ?S gates
                    control_0_sel[i*320+j] = control_qubits[cur_gate+i];
                    control_1_sel[i*320+j] = control_qubits[cur_gate+i];
                }
                    */
                
                /*
                if (target_qubits[cur_gate+i] > 8 && target_qubits[cur_gate+i] <= 16) {
                    // L gates
                    target_qubit_dmap[i*320+j] = 2*(target_qubits[cur_gate+i] - 8 - 1);
                    target_qubit_dmap[i*320+j+1] = 2*(target_qubits[cur_gate+i] - 8 - 1) + 1;

                } else if (target_qubits[cur_gate+i] > 16) {
                    // H gates
                    target_qubit_dmap[i*320+j] = 2*(target_qubits[cur_gate+i] - 17);
                    target_qubit_dmap[i*320+j+1] = 2*(target_qubits[cur_gate+i] - 17) + 1;
                }

                if (control_qubits[cur_gate+i] > 8 && control_qubits[cur_gate+i] <= 16 && target_qubits[cur_gate+i] <= 8) {
                    // SL, 8L gates
                    control_qubit_dmap[i*320+j] = 2*(control_qubits[cur_gate+i] - 8 - 1);
                    control_qubit_dmap[i*320+j+1] = 2*(control_qubits[cur_gate+i] - 8 - 1) + 1;

                } else if (control_qubits[cur_gate+i] > 16 && target_qubits[cur_gate+i] <= 8) {
                    // SH, 8H gates
                    control_qubit_dmap[i*320+j] = 2*(control_qubits[cur_gate+i] - 17);
                    control_qubit_dmap[i*320+j+1] = 2*(control_qubits[cur_gate+i] - 17) + 1;

                } else if (control_qubits[cur_gate+i] > 8 && control_qubits[cur_gate+i] <= 16 && target_qubits[cur_gate+i] > 8) {
                    // LL, HL gates
                    control_qubit_dmap[i*320+j] = 2*(control_qubits[cur_gate+i] - (control_qubits[cur_gate+i] >= target_qubits[cur_gate+i]) - 8 - 1);
                    control_qubit_dmap[i*320+j+1] = 2*(control_qubits[cur_gate+i] - (control_qubits[cur_gate+i] >= target_qubits[cur_gate+i]) - 8 - 1) + 1;
                }           
                */
            }
        }
        //run the gate upload program
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][0], (unsigned char*)gatereal, sizeof(float)*packedSize, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][1], (unsigned char*)gateimag, sizeof(float)*packedSize, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][2], (unsigned char*)state_0_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][3], (unsigned char*)state_1_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][4], (unsigned char*)control_0_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][5], (unsigned char*)control_1_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][6], (unsigned char*)target_qubit_dmap, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][7], (unsigned char*)control_qubit_dmap, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][8], (unsigned char*)perm_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_tensor_layout_from_host(state.gateLayouts[device_num][9], (unsigned char*)mxm_perm_sel, sizeof(unsigned char)*max_gates*320, input, 8*sizeof(unsigned char)*max_gates*320+sizeof(float)*packedSize*2);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
        }
        status = groq_invoke(state.device[device_num], state.inputBuffers[device_num][0], 0, state.outputBuffers[device_num][0], 0, &completion);
        if (status) {
            printf("ERROR: gate upload invoke failed %d\n", status);
            return 1;
        }
        while (!(completionCode = groq_poll_completion(completion)));
        if (completionCode != GROQ_COMPLETION_SUCCESS) {
            printf("ERROR: gate upload completion code %d\n", completionCode);
        }

        //struct timespec starttime;
        //timespec_get(&starttime, TIME_UTC);

        //add debug time messages for precise time information
        //run compute number of gates times - with invoke queuing

        for (unsigned int i = 0; i < tot_gates; i++) {

            int target_qbit = target_qubits[cur_gate+i];
            int control_qbit = control_qubits[cur_gate+i];

            int c = (control_qbit != -1) + (control_qbit >= 8) + (control_qbit > 8) + (control_qbit > 16 - ( (target_qbit > 16) & (target_qbit < 21)) );
            int t = (target_qbit >= 8) + (target_qbit > 8) + (target_qbit > 16);
            int p = qubitProgs[c][t];
            //printf("target_qbit=%d control_qbit=%d program_idx=%d\n", target_qubits[cur_gate+i], control_qubits[cur_gate+i], p);
            int queueIdx = queueFree->data;
            transferNode(&queueFree, &queueOccupied, queueFree);
            int bufIdx = bufferIndices[p-3]++ % QUEUE_LIMIT;

            status = groq_invoke(state.device[device_num], state.inputBuffers[device_num][p], bufIdx, state.outputBuffers[device_num][p], bufIdx, &completionQueue[queueIdx]);
            if (status) {
                printf("ERROR: compute invoke failed %d\n", status);
                return 1;
            }
            Node* ptr = queueOccupied;
            while (queueFree == NULL) {
                completionCode = groq_poll_completion(completionQueue[ptr->data]);
                ptr = ptr->next;
                if (completionCode) {
                    transferNode(&queueOccupied, &queueFree, ptr);
                    if (completionCode != GROQ_COMPLETION_SUCCESS) {
                        printf("ERROR: compute completion code %d\n", completionCode);
                        return 1;
                    }
                }
            }
        }
        while (queueOccupied != NULL) {
            //printf("%d\n", queueOccupied->data); 
            while (!(completionCode = groq_poll_completion(completionQueue[queueOccupied->data])));
            transferNode(&queueOccupied, &queueFree, queueOccupied);
            if (completionCode != GROQ_COMPLETION_SUCCESS) {
                printf("ERROR: compute completion code %d\n", completionCode);
                return 1;
            }
        }

        //struct timespec t;
        //timespec_get(&t, TIME_UTC);
        //printf("Total Gate Set Invocation Compute Processing Time: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);
    }

    free(mxm_perm_sel);
    free(perm_sel);
    free(gatereal);
    free(gateimag);
    free(state_0_sel);
    free(state_1_sel);
    free(control_0_sel);
    free(control_1_sel);
    free(target_qubit_dmap);
    free(control_qubit_dmap);
    freeList(queueFree);
    free(bufferIndices);
    free(completionQueue);

    // download the state vector from the chip
    status = groq_invoke(state.device[device_num], state.inputBuffers[device_num][DOWNLOAD_SV_IDX], 0, state.outputBuffers[device_num][DOWNLOAD_SV_IDX], 0, &completion);
    if (status) {
        printf("ERROR: sv download invoke failed %d\n", status);
        return 1;
    }
    while (!(completionCode = groq_poll_completion(completion)));
    if (completionCode != GROQ_COMPLETION_SUCCESS) {
        printf("ERROR: sv download completion code %d\n", completionCode);
    }

    unsigned char* output;
    status = groq_get_data_handle(state.outputBuffers[device_num][2], 0, &output);
    if (status) printf("ERROR: get data handle %d\n", status);

    status = groq_tensor_layout_to_host(state.outpLayouts[device_num][0], output, sizeof(Complex32)*320*(1<<(num_qubits-8)), (unsigned char*)realdata, sizeof(float)*(1<<num_qubits));
    if (status) {
        printf("ERROR: tensor layout from host %d\n", status);
    }
    status = groq_tensor_layout_to_host(state.outpLayouts[device_num][1], output, sizeof(Complex32)*320*(1<<(num_qubits-8)), (unsigned char*)imagdata, sizeof(float)*(1<<num_qubits));
    if (status) {
        printf("ERROR: tensor layout from host %d\n", status);
    }

    return 0;
}

#ifdef TEST
extern "C" int main() { //U3 and CRY
    initialize_groq_sv(get_accelerator_free_num_sv());
    //initialize_groq_sv(1);
    int num_qubits = 16;
    int device_idx = 0;
    Complex32* data = (Complex32*)calloc(sizeof(Complex32), 1<<num_qubits);
    data[0].real = 1.0;
    prepare_state_vector(data, num_qubits, device_idx);
    int num_gates = 1792*100;
    int* target_qubits = (int*)calloc(sizeof(int), num_gates);
    int* control_qubits = (int*)calloc(sizeof(int), num_gates);
    for (int i = 0; i < num_gates; i++) {
        target_qubits[i] = rand() % num_qubits;
        control_qubits[i] = rand() % num_qubits;
        if (target_qubits[i] == control_qubits[i]) control_qubits[i] = -1;
    }
    Complex32* gates = (Complex32*)calloc(sizeof(Complex32), num_gates);
    //int target_qubits[] = { 6,  2,  12, 1, 15, 8, 12, 4,  6, 16, 10, 11, 15,  8, 12, 13, 8 };
	//int control_qubits[] = { -1, -1, 3,  0, 3, -1, 9, 10,  8,  7, -1,  16,  7, -1, 2, 8, 11 }; //-1 ... no control qubit is incolved in the operation 
    //int target_qubits[] = { 6,  2,  12, 1, 15, 8, 12, 4,  6, 10, 15,  8, 12, 13, 8 };
    //int control_qubits[] = { -1, -1, 3,  0, 3, -1, 9, 10,  8, -1,   7, -1, 2, 8, 11 };
    //int num_gates = sizeof(target_qubits) / sizeof(int);

    /*Complex32 gates[] = {
 { 0.73634757, 0. },  { -0.42020426, -0.5303024 },
 {-0.43154194, -0.52111785 }, { 0.15282814, -0.72031333 },

 { 0.30398456, 0. }, { 0.18162433-0.93520372 },
 { 0.92490917, -0.22833398 }, { 0.01525721, 0.30360143 },

 { 0.99098929, 0. }, { 0.13394116, 0. },
 {-0.13394116, 0. }, { 0.99098929, 0. },

 { 0.68460275, 0. }, { 0.72891637, 0. },
 {-0.72891637, 0. }, { 0.68460275, 0. },

 { 0.17599267, 0. }, { 0.98439148, 0. },
 {-0.98439148, 0. }, { 0.17599267, 0. },

 { 0.85385484, 0. },        { -0.25823217, 0.45193811 },
 { 0.5163168 , 0.06594592 }, { 0.51412171, 0.68172353 },

 { 0.90203619, 0. }, { -0.43166042, 0. },
 { 0.43166042, 0. }, { 0.90203619, 0. },

 { 0.89270217, 0. }, { 0.45064712, 0. },
 {-0.45064712, 0. }, { 0.89270217, 0. },

 { -0.90998992, 0. }, { 0.41463038, 0. },
 {-0.41463038, -0. }, { -0.90998992, -0. },

 { -0.98356463, 0. }, { 0.18055642, 0. },
 {-0.18055642, -0. }, { -0.98356463, -0. },

 { 0.97404947, 0. },        { 0.05866004, -0.21860153 },
 {0.16830275, -0.15133345 }, { 0.44130072, 0.86834673 },

 { 0.77756625, 0. }, { 0.62880103, 0. },
 {-0.62880103, 0. }, { 0.77756625, 0. },

 { -0.92408909, 0. }, { -0.38217712, 0. },
 { 0.38217712, 0. }, { -0.92408909, 0. },

 { -0.9604982 , 0. },        { -0.16932557, -0.22084396 },
 { 0.24687813, -0.1284305 }, { -0.87024037, -0.40649539 },

 { -0.79868996, 0. }, { 0.60174275, 0. },
 {-0.60174275, -0. }, { -0.79868996, -0. },

 { -0.21698192, 0. }, { 0.97617562, 0. },
 {-0.97617562, -0. }, { -0.21698192, -0. },

 { -0.95333378, 0. }, { 0.30191837, 0. },
 {-0.30191837, -0. }, { -0.95333378, -0. }     
    }; */
    float* resultreal = (float*)malloc(sizeof(float)*(1<<num_qubits));
    float* resultimag = (float*)malloc(sizeof(float)*(1<<num_qubits));
    calcsvKernelGroq(num_gates, gates, target_qubits, control_qubits, resultreal, resultimag, device_idx);
    //for (int i = 0; i < 1 << num_qubits; i++)
    //    printf("%f ", result[i].real);        
    free(resultreal);
    free(resultimag);
    free(gates);
    free(target_qubits);
    free(control_qubits);
    free(data);
    releive_groq_sv();       
}
#endif
