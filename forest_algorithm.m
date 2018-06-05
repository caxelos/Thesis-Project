function forest_algorithm()

clear all;
clc;


%%%%%%%%%% Defines %%%%%%%%%%%%%%%%%%%%%%%%%
R = 5;



%%%%%%%%%% Open HDF5 training file %%%%%%%%%%
fid = H5F.open('myfile.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');



%%%%%%%%%% Open Specific group %%%%%%%%%%%%%%
grpID = H5G.open(fid, '/g1');



%%%%%%%%%% Open Specific dataset of a group %
rnearestID      = H5D.open(grpID, '5_nearestIDs');
centerID        = H5D.open(grpID, 'center');
dataID          = H5D.open(grpID, 'data');
labelID		= H5D.open(grpID, 'label');





%%%%%%%%% Read Data %%%%%%%%%%%%%%%%%%%%%%%%%
rnearest = H5D.read(rnearestID)



%%%%%%%% Now Start building tree(i) %%%%%%%%%
%i need: images, headposes, gazes -> data, label

for i = 1:R 
	localGrpIDs(i) = H5G.open(fid, strcat('/g', num2str( rnearest(i))   )); 
	tempDataID(i) =  H5D.open( localGrpIDs(i),  strcat('/g', num2str( rnearest(i) ), '/label') );

	
	tempData = H5D.read(tempDataID);
end
tempData


%%% doesnt run here
for i = 1:R
	H5D.close( tempDataID(i) );
	H5G.close( localGrpIDs(i) ) ;
end
%%%%%%%%% Close Everything %%%%%%%%%%%%%%%%%%
H5D.close(rnearestID);
H5D.close(centerID);
H5D.close(dataID);
H5D.close(labelID);



H5G.close(grpID);
H5F.close(fid);


end

%H5G.get_info(grpID)
