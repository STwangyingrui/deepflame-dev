#pragma once

#include "dfMatrixDataBase.H"

class dfTestEqn
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, cell_vec_bytes, csr_value_bytes, csr_value_vec_bytes;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;

    // Matrix variables
    double *d_A_csr = nullptr;
    double *d_b = nullptr;

public:
    dfTestEqn();
    dfTestEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile);
    ~dfTestEqn();

    void initializeTimeStep(const double *phi);
    void fvm_div();
    void checkResult(const double *h_lower, const double *h_upper, const double *h_diag, bool print);
    void clear();
    void fvm_div_2(int *upperOffset, int *lowerOffset, int *lowerPermList);
};
