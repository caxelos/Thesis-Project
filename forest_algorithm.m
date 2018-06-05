clear all;
clc;


%%%%%%%%%% Defines %%%%%%%%%%%%%%%%%%%%%%%%%
R = 5;



%%%%%%%%%% Open HDF5 training file %%%%%%%%%%
fid = H5F.open('myfile.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');



%%%%%%%%%% Open Specific group %%%%%%%%%%%%%%
grpID = H5G.open(fid, '/g1');



%%%%%%%%%% Open Specific dataset of a group %
dsetID = H5D.open(grpID, '5_nearestIDs');



%%%%%%%%% Read Data %%%%%%%%%%%%%%%%%%%%%%%%%
data = H5D.read(dsetID)


H5D.close(dsetID);
H5G.close(grpID);
H5F.close(fid);




%H5G.get_info(grpID)
