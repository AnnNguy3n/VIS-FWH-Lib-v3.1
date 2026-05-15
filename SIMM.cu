#include "CppSources/Filter/StreakInvestMultiMethod/SIMM_Filter.cu"


int main(int argc, char* argv[]){
    if (argc == 1){
        raise_error("", "");
    }

    string config_path = argv[1];
    Multi_investMethod vis(config_path);
    vis.run();
}