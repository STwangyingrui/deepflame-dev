#include "dfTestEqn.H"

// kernel functions

__global__ void warmup(int num_cells)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
}

__global__ void fvc_grad_internal_atom(int num_cells, int num_surfaces,
                                       const int *lower_index, const int *upper_index,
                                       const double *face_vector, const double *weight, const double *pressure,
                                       double *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double sfx = face_vector[index * 3 + 0];
    double sfy = face_vector[index * 3 + 1];
    double sfz = face_vector[index * 3 + 2];

    int owner = upper_index[index];
    int neighbour = lower_index[index];

    double face_p = (w * (pressure[owner] - pressure[neighbour]) + pressure[neighbour]);

    double grad_x = sfx * face_p;
    double grad_y = sfy * face_p;
    double grad_z = sfz * face_p;
    

    // owner
    atomicAdd(&(b[num_cells * 0 + owner]), grad_x);
    atomicAdd(&(b[num_cells * 1 + owner]), grad_y);
    atomicAdd(&(b[num_cells * 2 + owner]), grad_z);

    // neighbour
    atomicAdd(&(b[num_cells * 0 + neighbour]), -grad_x);
    atomicAdd(&(b[num_cells * 1 + neighbour]), -grad_y);
    atomicAdd(&(b[num_cells * 2 + neighbour]), -grad_z);
}

__global__ void fvc_grad_boundary_atom(int num_patch_face, const int *face2Cells, int num_cells,
                                       double *boundary_face_vector, double *boundary_pressure, double *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_patch_face)
        return;
    
    double p = boundary_pressure[index];
    double bouSfx = boundary_face_vector[index * 3 + 0];
    double bouSfy = boundary_face_vector[index * 3 + 1];
    double bouSfz = boundary_face_vector[index * 3 + 2];

    int cellIndex = face2Cells[index];

    double grad_x = bouSfx * p;
    double grad_y = bouSfy * p;
    double grad_z = bouSfz * p;

    atomicAdd(&(b[num_cells * 0 + cellIndex]), grad_x);
    atomicAdd(&(b[num_cells * 1 + cellIndex]), grad_y);
    atomicAdd(&(b[num_cells * 2 + cellIndex]), grad_z);
}

__global__ void fvc_grad_internal_orig(int num_cells,
                                       const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                       const double *face_vector, const double *weight, const double *pressure,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_p = pressure[index];
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    double grad_bx_low = 0;
    double grad_bx_upp = 0;
    double grad_by_low = 0;
    double grad_by_upp = 0;
    double grad_bz_low = 0;
    double grad_bz_upp = 0;
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index;
            double w = weight[neighbor_index];
            double sfx = face_vector[neighbor_index * 3 + 0];
            double sfy = face_vector[neighbor_index * 3 + 1];
            double sfz = face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
            grad_bx_low -= face_p * sfx;
            grad_by_low -= face_p * sfy;
            grad_bz_low -= face_p * sfz;
        }
        // upper
        if (inner_index > diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = weight[neighbor_index];
            double sfx = face_vector[neighbor_index * 3 + 0];
            double sfy = face_vector[neighbor_index * 3 + 1];
            double sfz = face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = w * own_cell_p + (1 - w) * neighbor_cell_p;
            grad_bx_upp += face_p * sfx;
            grad_by_upp += face_p * sfy;
            grad_bz_upp += face_p * sfz;
        }
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] - grad_bx_low - grad_bx_upp;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] - grad_by_low - grad_by_upp;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] - grad_bz_low - grad_bz_upp;
}

__global__ void boundaryPermutation(const int num_boundary_faces, const int *bouPermedIndex,
                                    const double *boundary_pressure_init, double *boundary_pressure)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    int p = bouPermedIndex[index];
    boundary_pressure[index] = boundary_pressure_init[p];
}

__global__ void fvc_grad_boundary_orig(int num_cells, int num_boundary_cells,
                                       const int *boundary_cell_offset, const int *boundary_cell_id,
                                       const double *boundary_face_vector, const double *boundary_pressure,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // compute boundary gradient
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double sfx = boundary_face_vector[i * 3 + 0];
        double sfy = boundary_face_vector[i * 3 + 1];
        double sfz = boundary_face_vector[i * 3 + 2];
        double face_p = boundary_pressure[i];
        grad_bx += face_p * sfx;
        grad_by += face_p * sfy;
        grad_bz += face_p * sfz;
    }

    //// correct the boundary gradient
    // double nx = boundary_face_vector[face_index * 3 + 0] / magSf[face_index];
    // double ny = boundary_face_vector[face_index * 3 + 1] / magSf[face_index];
    // double nz = boundary_face_vector[face_index * 3 + 2] / magSf[face_index];
    // double sn_grad = 0;
    // double grad_correction = sn_grad * volume[cell_index] - (nx * grad_bx + ny * grad_by + nz * grad_bz);
    // grad_bx += nx * grad_correction;
    // grad_by += ny * grad_correction;
    // grad_bz += nz * grad_correction;

    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] - grad_bx;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] - grad_by;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] - grad_bz;
}

// constructor
dfTestEqn::dfTestEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile)
    : dataBase_(dataBase)
{
    stream = dataBase_.stream;

    num_cells = dataBase_.num_cells;
    num_faces = dataBase_.num_faces;
    num_surfaces = dataBase_.num_surfaces;
    num_boundary_cells = dataBase_.num_boundary_cells;
    cell_vec_bytes = dataBase_.cell_vec_bytes;
    csr_value_bytes = dataBase_.csr_value_bytes;
    csr_value_vec_bytes = dataBase_.csr_value_vec_bytes;

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    checkCudaErrors(cudaMalloc((void **)&d_b, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b_ref, cell_vec_bytes));
    checkCudaErrors(cudaMemsetAsync(d_b_ref, 0, cell_vec_bytes, stream));
}

void dfTestEqn::initializeTimeStep(const double *weight, const double *pressure, const double *face_vector)
{
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_pressure, pressure, cell_bytes, cudaMemcpyHostToDevice, stream));

    // initialize RHS value
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));

    checkCudaErrors(cudaMalloc((void **)&d_weight_half, num_surfaces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_faceVector_half, 3 * num_surfaces * sizeof(double)));

    checkCudaErrors(cudaMemcpyAsync(d_weight_half, weight, num_surfaces * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_faceVector_half, face_vector, 3 * num_surfaces * sizeof(double), cudaMemcpyHostToDevice, stream));
}

void dfTestEqn::fvc_grad_old(double *boundary_pressure_init)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    // warmup
    fprintf(stderr, "warmup...\n");
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    warmup<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells);

    // permutate boundary pressure
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_pressure_init, boundary_pressure_init, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    blocks_per_grid = (dataBase_.num_boundary_faces + threads_per_block - 1) / threads_per_block;
    boundaryPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_faces, dataBase_.d_bouPermedIndex, dataBase_.d_boundary_pressure_init, dataBase_.d_boundary_pressure);

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;

    // 使用event计算时间
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    fvc_grad_internal_orig<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                              d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                              dataBase_.d_face_vector, dataBase_.d_weight, dataBase_.d_pressure, d_b_ref, d_b_ref);
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    fprintf(stderr, "try fvc_grad_internal_old 执行时间：%f(ms)\n", time_elapsed);
    

    checkCudaErrors(cudaEventRecord(start, 0));
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_boundary_orig<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
                                                                              dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                              dataBase_.d_boundary_face_vector, dataBase_.d_boundary_pressure, d_b, d_b);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    fprintf(stderr, "try fvc_grad_boundary_old 执行时间：%f(ms)\n", time_elapsed);
}

void dfTestEqn::fvc_grad_new(const double **boundary_pressure_per_patch, const double **boundary_face_vector_per_patch, 
    const int **face_cells_per_patch, int *num_face_per_patch, int patch_size,
    // conbine patch
    double *boundary_pressure, double *boundary_face_vector, int *face_cell_combine, int num_boundary_face)
{
    // prepare data
    // per patch
    double **d_boundary_pressure_per_patch = new double*[patch_size];
    double **d_boundary_face_vector_per_patch = new double*[patch_size];
    int **d_face_cells_per_patch = new int*[patch_size];
    for(int i = 0; i < patch_size; i ++) {
        checkCudaErrors(cudaMalloc((void **)&d_boundary_pressure_per_patch[i], num_face_per_patch[i] * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_boundary_face_vector_per_patch[i], 3 * num_face_per_patch[i] * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_face_cells_per_patch[i], num_face_per_patch[i] * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_boundary_pressure_per_patch[i], boundary_pressure_per_patch[i], num_face_per_patch[i] * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_boundary_face_vector_per_patch[i], boundary_face_vector_per_patch[i], 3 * num_face_per_patch[i] * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_face_cells_per_patch[i], face_cells_per_patch[i], num_face_per_patch[i] * sizeof(int), cudaMemcpyHostToDevice));
    }
    // combine patch
    double *d_boundary_pressure, *d_boundary_face_vector;
    int *d_face_cell_combine;
    checkCudaErrors(cudaMalloc((void **)&d_boundary_pressure, num_boundary_face * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_face_vector, 3 * num_boundary_face * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_face_cell_combine, num_boundary_face * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_boundary_pressure, boundary_pressure, num_boundary_face * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boundary_face_vector, boundary_face_vector, 3 * num_boundary_face * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_face_cell_combine, face_cell_combine, num_boundary_face * sizeof(int), cudaMemcpyHostToDevice));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    // warmup
    fprintf(stderr, "warmup...\n");
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    warmup<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells);

    // 使用event计算时间
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_grad_internal_atom<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces,
                                                                              dataBase_.d_lowerAddr, dataBase_.d_upperAddr,
                                                                              d_faceVector_half, d_weight_half, dataBase_.d_pressure, d_b);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    fprintf(stderr, "try2 fvc_grad_internal_new 执行时间：%f(ms)\n", time_elapsed);


    time_elapsed = 0;
    checkCudaErrors(cudaEventRecord(start, 0));
    
    // per patch solution
    // for (int i = 0; i < patch_size; i++)
    // {
    //     blocks_per_grid = (num_face_per_patch[i] + threads_per_block - 1) / threads_per_block;
    //     fvc_grad_boundary_atom<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_face_per_patch[i], d_face_cells_per_patch[i], 
    //             num_cells, d_boundary_face_vector_per_patch[i], d_boundary_pressure_per_patch[i], d_b);
    // }

    // combined patch solution
    blocks_per_grid = (num_boundary_face + threads_per_block - 1) / threads_per_block;
    fvc_grad_boundary_atom<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_face, d_face_cell_combine,
            num_cells, d_boundary_face_vector, d_boundary_pressure, d_b);


    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    fprintf(stderr, "try2 fvc_grad_boundary_new 执行时间：%f(ms)\n", time_elapsed);
}

void dfTestEqn::checkResult(bool print)
{
    std::vector<double> h_b(3 * num_cells);
    checkCudaErrors(cudaMemcpy(h_b.data(), d_b, 3 * num_cells * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_b_ref(3 * num_cells);
    checkCudaErrors(cudaMemcpy(h_b_ref.data(), d_b_ref, 3 * num_cells * sizeof(double), cudaMemcpyDeviceToHost));
    if (print)
    {
        for (int i = 0; i < 3 * num_cells; i++)
        {
            fprintf(stderr, "h_b[%d]: %.10lf\n", i, h_b[i]);
        }
        for (int i = 0; i < 3 * num_cells; i++)
        {
            fprintf(stderr, "h_b_ref[%d]: %.10lf\n", i, h_b_ref[i]);
        }
    }
    checkVectorEqual(3 * num_cells, h_b_ref.data(), h_b.data(), 1e-5);
}

dfTestEqn::~dfTestEqn()
{
}
