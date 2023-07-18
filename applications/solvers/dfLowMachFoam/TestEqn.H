// validate fvm::div(phi, U)
#ifdef GPUSolver_
    // firstly, run CPUSolver
    tmp<fvVectorMatrix> tmpTestEqn
    (
        fvm::div(phi, U)
    );
    fvVectorMatrix& TestEqn = tmpTestEqn.ref();

    // CPU result
    int num_cells = mesh.nCells();
    int num_surfaces = mesh.neighbour().size();
    std::vector<double> h_lower(num_surfaces);
    memcpy(h_lower.data(), &TestEqn.lower()[0], num_surfaces * sizeof(double));
    std::vector<double> h_upper(num_surfaces);
    memcpy(h_upper.data(), &TestEqn.upper()[0], num_surfaces * sizeof(double));
    std::vector<double> h_diag(num_cells);
    memcpy(h_diag.data(), &TestEqn.diag()[0], num_cells * sizeof(double));
    // for (int i = 0; i < num_surfaces; i++) {
    //     fprintf(stderr, "cpu lower[%d]: %.10lf\n", i, h_lower[i]);
    // }
    // for (int i = 0; i < num_surfaces; i++) {
    //     fprintf(stderr, "cpu upper[%d]: %.10lf\n", i, h_upper[i]);
    // }
    // for (int i = 0; i < num_cells; i++) {
    //     fprintf(stderr, "cpu diag[%d]: %.10lf\n", i, h_diag[i]);
    // }

    // then, run GPUSolver
    TestEqn_GPU.initializeTimeStep(&phi[0]);
    TestEqn_GPU.fvm_div();

    // check result
    TestEqn_GPU.checkResult(h_lower.data(), h_upper.data(), h_diag.data(), false);
#endif

