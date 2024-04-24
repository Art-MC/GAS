extern "C"
__global__ void find_neighbors(int* poslist_round, \
                    int* volume_inds,\
                    int* volume_shape,\
                    int search_rad,\
                    long* center_poslist_inds, \
                    long* neighbors_inds_list,\
                    long* num_neighbors_list, \
                    int Nmax_neighbors,\
                    int tot_num_centers,\
                    int Dim, \
                    bool* pbcs)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < tot_num_centers){
        // index in poslist
        int cind = center_poslist_inds[tid];
        // x,y,z center indices in volume_inds
        int cpos[3] = {(int)poslist_round[cind*Dim], (int)poslist_round[cind*Dim+1], (int)poslist_round[cind*Dim+2]};

        // getting the index for the volume_inds array
        int vol_ind = 0;
        // central ind is cpos[0] * volume_shape[2] * volume_shape[1] + cpos[1] * volume_shape[2] + cpos[2];

        int mod_ind_x = 0;
        int mod_ind_y = 0;
        int mod_ind_z = 0;

        int value = 0;
        int num_neighbors = 0;
        int out_ind_part = tid * Nmax_neighbors;

        // iterate through volume to get neighbors
        // the if pbcs[] and continue statements are redundent, because currently only good
        // centerpoints are given here. doesn't hurt to have em tho.
        for (int i=-1*search_rad; i<=search_rad; i++){
            mod_ind_x = cpos[0] + i;
            if (mod_ind_x < 0){
                if (pbcs[0]) {
                    mod_ind_x += volume_shape[0];
                }
                else {
                    continue;
                }
            }
            else if (mod_ind_x >= volume_shape[0]){
                if (pbcs[0]) {
                    mod_ind_x -= volume_shape[0];
                }
                else {
                    continue;
                }
            }
            mod_ind_x = mod_ind_x * volume_shape[2] * volume_shape[1];

            for (int j=-1*search_rad; j<=search_rad; j++){
                mod_ind_y = cpos[1] + j;
                if (mod_ind_y < 0){
                    if (pbcs[1]) {
                        mod_ind_y += volume_shape[1];
                    }
                    else {
                        continue;
                    }
                }
                else if (mod_ind_y >= volume_shape[1]){
                    if (pbcs[1]) {
                        mod_ind_y -= volume_shape[1];
                    }
                    else {
                        continue;
                    }
                }
                mod_ind_y = mod_ind_y * volume_shape[2];

                for (int k=-1*search_rad; k<=search_rad; k++){
                    mod_ind_z = cpos[2] + k;
                    if (mod_ind_z < 0){
                        if (pbcs[2]) {
                            mod_ind_z += volume_shape[2];
                        }
                        else {
                            continue;
                        }
                    }
                    else if (mod_ind_z >= volume_shape[2]){
                        if (pbcs[2]) {
                            mod_ind_z -= volume_shape[2];
                        }
                        else {
                            continue;
                        }
                    }

                    vol_ind = mod_ind_x + mod_ind_y + mod_ind_z;
                    value = volume_inds[vol_ind];
                    if (value != -1){
                        neighbors_inds_list[out_ind_part + num_neighbors] = value;
                        num_neighbors++;
                        if (num_neighbors > Nmax_neighbors){
                            printf("MAX NEIGHBORS LIMIT REACHED. DECREASE dr_ind OR DECREASE R_max\n");
                        }
                    }
                }
            }
        }
        num_neighbors_list[tid] = num_neighbors;
    }
}