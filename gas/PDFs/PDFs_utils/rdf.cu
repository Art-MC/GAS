extern "C"
// for histo_height = num_threads
__global__ void rdf(double* poslist, \
                    long* center_poslist_inds, \
                    long* neighbor_poslist_inds, \
                    long* neighbor_poslist_shape,\
                    int numrows,\
                    int Dim, \
                    double* rdf_histo, \
                    int histo_height, \
                    int numBins, \
                    double dR, \
                    float* boxSize, \
                    bool* pbcs)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // neighbor_poslist_inds indexed by center, e.g. [i,j] here corresponds to center_inds_skip[i] -> positions[center_ind_skip[i]]
    if (tid < numrows * neighbor_poslist_shape[1] && neighbor_poslist_inds[tid] >= 0){
        double dist = 0;
        double dist_i = 0;
        int ind_0 = tid / neighbor_poslist_shape[1]; // center_index
        int ind_1 = center_poslist_inds[ind_0]*Dim; // position_index of center
        int ind_2 = neighbor_poslist_inds[tid]*Dim; // position index of neighbor
        if (ind_2<0){
            printf("BAD VALUE rdf.cu\n");
        }
        for (int i=0; i<Dim; i++){
            dist_i = fabs(poslist[ind_1+i] - poslist[ind_2+i]);
            if(pbcs[i]){
                dist_i = fmin(dist_i, boxSize[i]-dist_i);
            }
            dist += dist_i * dist_i;
        }
        dist = sqrt(dist);

        double loc = dist / dR;
        int fl = floor(loc);
        double weight = loc - fl;
        int yval = tid % histo_height;
        if (loc < numBins and fl>0) {
            rdf_histo[fl+yval*numBins] += (1-weight);
            if (loc+1 < numBins){
                rdf_histo[fl+yval*numBins+1] += weight;
            }
        }
        else{
            // for bookkeeping
            // rdf_histo[0+yval*numBins] += 1;
        }
    }
    else{
        // printf("outside range (%d)\n", tid);
    }
}