Each captured sample is stored as image + semantic mask + geometry metadata

## Core variables (per frame k) include the following: 
Time / indexing, t_k : timestamp (seconds), frame_id : integer frame index, Camera model and K ∈ R^{3×3} : camera intrinsics

K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]

D : distortion parameters (optional; often skipped in sim)

Camera pose

R_wc ∈ R^{3×3} : rotation from camera → world

p_wc ∈ R^{3} : camera position in world coordinates

T_wc ∈ SE(3) : homogeneous transform (camera → world)

Vehicle pose (if camera is mounted)

T_wb : world → body transform (UAV/robot base)

T_bc : body → camera fixed extrinsic (calibration / mounting)

T_wc = T_wb · T_bc

## Scene / labels

S_k(u,v) : semantic mask (pixelwise labels)
e.g., floor=1, obstacle=2

![Isaac]( bev_semantic_final.png)
