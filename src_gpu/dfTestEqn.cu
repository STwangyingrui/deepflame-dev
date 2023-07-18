#include "dfTestEqn.H"

// kernel functions

__global__ void warmup(int num_cells)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
}

__global__ void ldu_to_csr(int num_cells, int num_surfaces,
        const int *permute_index, const int *csr_row_index, const int *csr_diag_index,
        const double *lower, const double *upper, const double *diag,
        const double *A_csr_input, double *A_csr_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index;

            int perm_index = permute_index[neighbor_index];
            double value = lower[perm_index];
            A_csr_output[i] = A_csr_input[i] + value;
        }
        // diag
        if (inner_index == diag_index)
        {
            A_csr_output[i] = A_csr_input[i] + diag[index];
        }
        // upper
        if (inner_index > diag_index)
        {
            // upper, index - 1, consider of diag
            int neighbor_index = neighbor_offset + inner_index - 1;

            int perm_index = permute_index[neighbor_index];
            double value = upper[perm_index - num_surfaces];
            A_csr_output[i] = A_csr_input[i] + value;
        }
    }
}

__global__ void test_fvm_div_internal(int num_cells, int num_surfaces,
                                 const int *lower_index, const int *upper_index,
                                 const double *weight, const double *phi,
                                 double *lower, double *upper, double *diag)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double f = phi[index];

    lower[index] += (-w) * f;
    upper[index] += (1 - w) * f;

    int l = lower_index[index];
    int u = upper_index[index];
    atomicAdd(&(diag[l]), w * f);
    atomicAdd(&(diag[u]), (w - 1) * f);
}

// constructor
dfTestEqn::dfTestEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile)
    : dataBase_(dataBase)
{
    stream = dataBase_.stream;

    num_cells = dataBase_.num_cells;
    num_faces = dataBase_.num_faces;
    num_surfaces = dataBase_.num_surfaces;
    cell_vec_bytes = dataBase_.cell_vec_bytes;
    csr_value_bytes = dataBase_.csr_value_bytes;
    csr_value_vec_bytes = dataBase_.csr_value_vec_bytes;

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    checkCudaErrors(cudaMalloc((void **)&d_A_csr, csr_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, cell_vec_bytes));
}

void dfTestEqn::initializeTimeStep(const double *phi)
{
    // initialize matrix value
    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_lower, 0, num_surfaces * sizeof(double), stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_upper, 0, num_surfaces * sizeof(double), stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_diag, 0, num_cells * sizeof(double), stream));

    memcpy(dataBase_.h_phi_init, phi, num_surfaces * sizeof(double));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_try_phi, dataBase_.h_phi_init, num_surfaces * sizeof(double), cudaMemcpyHostToDevice, stream));
}

void dfTestEqn::fvm_div()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    // warmup
    fprintf(stderr, "warmup...\n");
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    warmup<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells);

    //使用event计算时间
    float time_elapsed=0;
    cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start,0));
    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    test_fvm_div_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces,
            dataBase_.d_lowerAddr, dataBase_.d_upperAddr,
            dataBase_.d_try_weight, dataBase_.d_try_phi, dataBase_.d_lower, dataBase_.d_upper, dataBase_.d_diag);
/*
    // 
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    ldu_to_csr<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces,
            dataBase_.d_permedIndex, d_A_csr_row_index, d_A_csr_diag_index,
            dataBase_.d_lower, dataBase_.d_upper, dataBase_.d_diag, d_A_csr, d_A_csr);
    checkCudaErrors(cudaMemcpy(&d_A_csr[(num_cell + num_face)], &d_A_csr[0], csr_value_bytes, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(&d_A_csr[(num_cell + num_face) * 2], &d_A_csr[0], csr_value_bytes, cudaMemcpyDeviceToDevice));
*/
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed,start,stop));
    fprintf(stderr, "try fvm_div_internal 执行时间：%f(ms)\n",time_elapsed);
}

void dfTestEqn::checkResult(const double *lower, const double *upper, const double *diag, bool print)
{
    std::vector<double> h_lower(num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), dataBase_.d_lower, num_surfaces * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_upper(num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), dataBase_.d_upper, num_surfaces * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_diag(num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), dataBase_.d_diag, num_cells * sizeof(double), cudaMemcpyDeviceToHost));
    if (print) {
        for (int i = 0; i < num_surfaces; i++) {
            fprintf(stderr, "cpu lower[%d]: %.10lf, gpu lower[%d]: %.10lf\n", i, lower[i], i, h_lower[i]);
        }
        for (int i = 0; i < num_surfaces; i++) {
            fprintf(stderr, "cpu upper[%d]: %.10lf, gpu upper[%d]: %.10lf\n", i, upper[i], i, h_upper[i]);
        }
        for (int i = 0; i < num_cells; i++) {
            fprintf(stderr, "cpu diag[%d]: %.10lf, gpu diag[%d]: %.10lf\n", i, diag[i], i, h_diag[i]);
        }
    }
    checkVectorEqual(num_surfaces, lower, h_lower.data(), 1e-5);
    checkVectorEqual(num_surfaces, upper, h_upper.data(), 1e-5);
    checkVectorEqual(num_cells, diag, h_diag.data(), 1e-5);
}

dfTestEqn::~dfTestEqn()
{
}
