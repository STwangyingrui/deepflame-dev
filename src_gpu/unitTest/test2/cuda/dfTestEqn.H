#pragma once

#include "dfMatrixDataBase.H"

class dfTestEqn
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, num_boundary_cells, cell_vec_bytes, csr_value_bytes, csr_value_vec_bytes;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;
    double *d_weight_half, *d_faceVector_half;

    // Matrix variables
    double *d_A_csr = nullptr;
    double *d_b = nullptr;
    double *d_b_ref = nullptr;

public:
    dfTestEqn();
    dfTestEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile);
    ~dfTestEqn();

    void initializeTimeStep(const double *weight, const double *pressure, const double *face_vector);
    void fvc_grad_old(double *boundary_pressure_init);
    void checkResult(bool print);
    void fvc_grad_new(const double **boundary_pressure_per_patch, const double **boundary_face_vector_per_patch, 
        const int **face_cells_per_patch, int *num_face_per_patch, int patch_size,
        // conbine patch
        double *boundary_pressure, double *boundary_face_vector, int *face_cell_combine, int num_boundary_face);
};
