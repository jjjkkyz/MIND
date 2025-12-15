/* ============================================================================
 * Copyright (c) 2009-2016 BlueQuartz Software, LLC
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of BlueQuartz Software, the US Air Force, nor the names of its
 * contributors may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The code contained herein was partially funded by the following contracts:
 *    United States Air Force Prime Contract FA8650-07-D-5800
 *    United States Air Force Prime Contract FA8650-10-D-5210
 *    United States Prime Contract Navy N00173-07-C-2068
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include"MeshStructs.h"



 /**
  * @class M3CSliceBySlice M3CSliceBySlice.h DREAM3DLic/SurfaceMeshingFilters/M3CSliceBySlice.h
  * @brief This filter was contributed by Dr. Sukbin Lee of Carnegi-Mellon University and uses a "MultiMaterial Marching
  * Cubes" algorithm originally proposed by Wu & Sullivan. @n
  * This version of the code only considers 2 slices of the volume at any give instant
  * in time during the algorithm. The 2 slices are meshed and the resulting triangles
  * and nodes are written out to disk. At the conclusion of all slices the entire
  * generated triangle array and node array are read into memory. This version trades
  * off mush lower memory footprint during execution of the filter for some speed.
  * The increase in time to mesh a volume is due to the File I/O of the algorithm. File
  * writes are done in pure binary so to make them as quick as possible. An adaptive
  * memory allocation routine is also employeed to be able to scale the speed of the
  * algorithm from small voxel volumes to very large voxel volumes.
  *
  * Multiple material marching cubes algorithm, Ziji Wu1, John M. Sullivan Jr2, International Journal for Numerical Methods in Engineering
  * Special Issue: Trends in Unstructured Mesh Generation, Volume 58, Issue 2, pages 189
  * @author
  * @date
  * @version 1.0
  */




class M3CEntireVolume
{
  
public:
    using Self = M3CEntireVolume;
    using Pointer = std::shared_ptr<Self>;
    using ConstPointer = std::shared_ptr<const Self>;
    using WeakPointer = std::weak_ptr<Self>;
    using ConstWeakPointer = std::weak_ptr<const Self>;


    ~M3CEntireVolume();
    M3CEntireVolume(int* _d,double* value, size_t* _dim);


    /**
     * @brief This returns a string that is displayed in the GUI. It should be readable
     * and understandable by humans.
     */
    virtual const std::string getHumanLabel()
    {
        return "M3C Surface Meshing (Volume)";
    }


    /**
     * @brief Reimplemented from @see AbstractFilter class
     */
    void execute();


    std::vector<std::vector<float>> m_vetices;
    std::vector<std::vector<int64_t>> m_faces;
    std::vector<std::pair<int, int>> m_face_label;




protected:




    /**
     * @brief Initializes all the private instance variables.
     */
    void initialize();

private:
    std::vector<int8_t> m_SurfaceMeshNodeType;
    int32_t* m_GrainIds = NULL;
    double* m_GrainPos = NULL;
    int* m_mask = NULL;
    size_t* dims;
    int64_t totalPoints;
    std::string nodes_file = "node.ply";
    std::string edges_file = "edges.ply";
    std::string triangles_file = "triangles.ply";

    bool is_udf_value = false;
    //bool m_AddSurfaceLayer = { true };

    int createMesh();

    /**
     * @brief initialize_micro_from_grainIds
     * @param dims
     * @param res
     * @param fileDims
     * @param grainIds
     * @param points
     * @param voxelCoords
     * @return
     */
    int initialize_micro_from_grainIds(size_t dims[3], float res[3], size_t fileDims[3], int32_t* grainIds, std::vector<int32_t>& points, SurfaceMesh::M3C::VoxelCoord* point);


    void get_neighbor_list(SurfaceMesh::M3C::Neighbor* n, int ns, int nsp, int xDim, int yDim, int zDim);

    void initialize_nodes(SurfaceMesh::M3C::VoxelCoord* p, SurfaceMesh::M3C::Node* v, int ns, int sx, int sy, int sz);
    void initialize_squares(SurfaceMesh::M3C::Neighbor* neighbors, SurfaceMesh::M3C::Face* sq, int ns, int nsp);
    int get_number_fEdges(SurfaceMesh::M3C::Face* sq, std::vector<int32_t>& points, SurfaceMesh::M3C::Neighbor* n, int eT2d[20][8], int ns);
    void get_nodes_fEdges(SurfaceMesh::M3C::Face* sq, std::vector<int32_t>& points, SurfaceMesh::M3C::Neighbor* neighbors, SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* e, int eT2d[20][8],
        int nsT2d[20][8], int ns, int nsp, int xDim);
    int get_square_index(int tns[4]);
    int treat_anomaly(int tnst[4], std::vector<int32_t>& points, SurfaceMesh::M3C::Neighbor* n1, int sqid);
    void get_nodes(int cst, int ord, int nidx[2], int* nid, int nsp1, int xDim1);
    void get_spins(std::vector<int32_t>& points, int cst, int ord, int pID[2], int* pSpin, int nsp1, int xDim1);
    int get_number_triangles(std::vector<int32_t>& points, SurfaceMesh::M3C::Face* sq, SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* e, int ns, int nsp, int xDim);
    int get_number_case0_triangles(int* afe, SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* e1, int nfedge);
    int get_number_case2_triangles(int* afe, SurfaceMesh::M3C::Node* v1, SurfaceMesh::M3C::Segment* fedge, int nfedge, int* afc, int nfctr);
    int get_number_caseM_triangles(int* afe, SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* e1, int nfedge, int* afc, int nfctr);
    int get_triangles(SurfaceMesh::M3C::VoxelCoord* p, SurfaceMesh::M3C::Triangle* t, int* mCubeID, SurfaceMesh::M3C::Face* sq, SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* e,
        SurfaceMesh::M3C::Neighbor* neighbors, int ns, int nsp, int xDim);
    void get_case0_triangles(SurfaceMesh::M3C::Triangle* t1, int* mCubeID, int* afe, SurfaceMesh::M3C::Node* v1, SurfaceMesh::M3C::Segment* e1, int nfedge, int tin, int* tout, double tcrd1[3],
        double tcrd2[3], int mcid);
    void get_case2_triangles(SurfaceMesh::M3C::Triangle* triangles1, int* mCubeID, int* afe, SurfaceMesh::M3C::Node* v1, SurfaceMesh::M3C::Segment* fedge, int nfedge, int* afc, int nfctr, int tin,
        int* tout, double tcrd1[3], double tcrd2[3], int mcid);
    void get_caseM_triangles(SurfaceMesh::M3C::Triangle* triangles1, int* mCubeID, int* afe, SurfaceMesh::M3C::Node* v1, SurfaceMesh::M3C::Segment* fedge, int nfedge, int* afc, int nfctr, int tin,
        int* tout, int ccn, double tcrd1[3], double tcrd2[3], int mcid);
    void find_edgePlace(double tvcrd1[3], double tvcrd2[3], double tvcrd3[3], int tw[3], double xh, double xl, double yh, double yl, double zh, double zl);
    int get_number_unique_inner_edges(SurfaceMesh::M3C::Triangle* triangles, int* mCubeID, int nT);
    void get_unique_inner_edges(SurfaceMesh::M3C::Triangle* t, int* mCubeID, SurfaceMesh::M3C::ISegment* ie, int nT, int nfedge);
    void update_triangle_sides_with_fedge(SurfaceMesh::M3C::Triangle* t, int* mCubeID, SurfaceMesh::M3C::Segment* e, SurfaceMesh::M3C::Face* sq, int nT, int xDim, int nsp);
    void arrange_spins(std::vector<int32_t>& points, SurfaceMesh::M3C::VoxelCoord* pCoord, SurfaceMesh::M3C::Triangle* triangles, SurfaceMesh::M3C::Node* v, int numT, int xDim, int nsp);
    void update_node_edge_kind(SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* fe, SurfaceMesh::M3C::ISegment* ie, SurfaceMesh::M3C::Triangle* t, int nT, int nfedge);
    int assign_new_nodeID(SurfaceMesh::M3C::Node* v, std::vector<int32_t>& node_ids, int ns);
    void generate_update_nodes_edges_array(std::vector<int32_t>& new_ids_for_nodes, std::vector<int8_t>& nodeKindPtr, std::vector<SurfaceMesh::M3C::Node>& shortNodes,
        std::vector<SurfaceMesh::M3C::Node>& vertices, std::vector<SurfaceMesh::M3C::Triangle>& triangles,
        std::vector<SurfaceMesh::M3C::Segment>& faceEdges, std::vector<SurfaceMesh::M3C::ISegment>& internalEdges, int maxGrainId);

    void get_output(SurfaceMesh::M3C::Node* v, SurfaceMesh::M3C::Segment* fedge, SurfaceMesh::M3C::ISegment* iedge, SurfaceMesh::M3C::Triangle* triangles, int ns, int nN, int nfe, int nie, int nT, std::vector<int32_t>& new_ids_for_nodes);

public:
    M3CEntireVolume(const M3CEntireVolume&) = delete;            // Copy Constructor Not Implemented
    M3CEntireVolume(M3CEntireVolume&&) = delete;                 // Move Constructor Not Implemented
    M3CEntireVolume& operator=(const M3CEntireVolume&) = delete; // Copy Assignment Not Implemented
    M3CEntireVolume& operator=(M3CEntireVolume&&) = delete;      // Move Assignment Not Implemented
};