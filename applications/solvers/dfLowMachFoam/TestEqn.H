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

    // clear result
    TestEqn_GPU.clear();

    // run another implementation of fvm_div

    // --prepare data
    std::vector<int> upperNum(num_cells, 0);
    std::vector<int> lowerNum(num_cells, 0);
    std::vector<int> lowerPermListInit(num_surfaces);

    int *upperOffset = (int*)calloc(num_cells + 1, sizeof(int));
    int *lowerOffset = (int*)calloc(num_cells + 1, sizeof(int));

    forAll(owner, faceI){
        upperNum[owner[faceI]] ++;
        lowerNum[neighbour[faceI]] ++;
    }
    std::partial_sum(upperNum.begin(), upperNum.end(), 
        upperOffset+1);
    std::partial_sum(lowerNum.begin(), lowerNum.end(), 
        lowerOffset+1);

    std::iota(lowerPermListInit.begin(), lowerPermListInit.end(), 0);

    std::multimap<int,int> permutation;
    for (int i = 0; i < num_surfaces; ++i){
        permutation.insert(std::make_pair(neighbour[i], lowerPermListInit[i]));
    }
    std::vector<std::pair<int, int>> permPair(permutation.begin(), permutation.end());
    std::sort(permPair.begin(), permPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    // - form permedIndex list
    std::vector<int> lowerPermList;
    std::transform(permPair.begin(), permPair.end(), std::back_inserter(lowerPermList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    }); 

    // --run fvm_div
    TestEqn_GPU.fvm_div_2(upperOffset, lowerOffset, lowerPermList.data());

    TestEqn_GPU.checkResult(h_lower.data(), h_upper.data(), h_diag.data(), false);
    
#endif

