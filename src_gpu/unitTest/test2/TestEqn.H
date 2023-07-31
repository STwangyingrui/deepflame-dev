// validate fvm::div(phi, U)

    // run GPUSolver
    TestEqn_GPU.initializeTimeStep(&mesh.surfaceInterpolation::weights()[0], &p[0], &mesh.Sf()[0][0]); // need weight, pressure, face_vector

    // prepare boudnary pressure
    double *boundary_pressure_init;
    int offset = 0;
    cudaMallocHost(&boundary_pressure_init, num_boundary_faces*sizeof(double));
    forAll(U.boundaryField(), patchi)
    {
        const scalarField& patchP = p.boundaryField()[patchi];
        int patchSize = patchP.size();
        // boundary pressure
        memcpy(boundary_pressure_init+offset, &patchP[0], patchSize*sizeof(double));
        offset += patchSize;
    }

    TestEqn_GPU.fvc_grad_old(boundary_pressure_init);


    // prepare boundary pressure 2
    const double **boundary_pressure_per_patch = new const double*[U.boundaryField().size()];
    const int **face_cells_per_patch = new const int*[U.boundaryField().size()];
    const double **boundary_face_vector_per_patch = new const double*[U.boundaryField().size()];

    double *boundary_pressure = new double[offset];
    double *boundary_face_vector = new double[offset * 3];
    int *face_cell_combine = new int[offset];

    offset = 0;

    int *num_face_per_patch = new int[U.boundaryField().size()];

    forAll(U.boundaryField(), patchi)
    {
        // per patch 
        const scalarField& patchP = p.boundaryField()[patchi];
        boundary_pressure_per_patch[patchi] = &patchP[0];

        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();
        face_cells_per_patch[patchi] = &pFaceCells[0];

        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        boundary_face_vector_per_patch[patchi] = &pSf[0][0];

        int patchSize = patchP.size();

        // sum
        memcpy(boundary_pressure + offset, &patchP[0], patchSize * sizeof(double));
        memcpy(boundary_face_vector + 3 * offset, &pSf[0][0], 3 * patchSize * sizeof(double));
        memcpy(face_cell_combine + offset, &pFaceCells[0], patchSize * sizeof(int));
        
        num_face_per_patch[patchi] = patchSize;
        offset += patchSize;
    }
    Info << num_face_per_patch[0] << endl;
    
    TestEqn_GPU.fvc_grad_new(boundary_pressure_per_patch, boundary_face_vector_per_patch, 
        face_cells_per_patch, num_face_per_patch, U.boundaryField().size(),
        // conbine patch
        boundary_pressure, boundary_face_vector, face_cell_combine, offset);

    TestEqn_GPU.checkResult(false);

