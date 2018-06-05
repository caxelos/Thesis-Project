function forest_algorithm()

clear all;
clc;


R = 5;


%%%%%%%%%% Open HDF5 training file %%%%%%%%%%
fid = H5F.open('myfile.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');




samplesInTree = 0;
%%%%%%%%%% Start with the central group %%%%%%%%%%
grpID = H5G.open(fid, '/g1');

curr_rnearestID      = H5D.open(grpID, '5_nearestIDs');
curr_centerID        = H5D.open(grpID, 'center');
curr_imgsID          = H5D.open(grpID, 'data');
curr_gazesID 	= H5D.open(grpID, 'gaze');
curr_posesID		= H5D.open(grpID, 'headpose');

curr_rnearest = H5D.read(curr_rnearestID);
curr_center   = H5D.read(curr_centerID);
curr_imgs     = H5D.read(curr_imgsID);
curr_gazes    = H5D.read(curr_gazesID);
curr_poses    = H5D.read(curr_posesID);

samplesInGroup = length( curr_imgs(:,1,1,1) );
contribOfGroup = ceil( sqrt( samplesInGroup ) );

j = 1;
while j <= contribOfGroup

	samplesInTree = samplesInTree + 1;
	random = randi(samplesInGroup,1,1);
	treeImgs(samplesInTree, :, :) =  curr_imgs( random  ,1, :, :);
	treeGazes(samplesInTree, :) = curr_gazes( random, :);%, :);
	treePoses(samplesInTree, :) = curr_poses( random, :);
		
	j = j + 1;		

end



%%%%%%%% Now, continue with the R-nearest %%%%%%%%%




for i = 1:R 
	localGrpID  = H5G.open(fid, strcat('/g', num2str( curr_rnearest(i))   )); 

	tempImgID  = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(i) ), '/data') );
	tempPoseID = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(i) ), '/headpose') );
	tempGazeID = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(i) ), '/gaze') );
	
	tempImgs = H5D.read( tempImgID );
	tempPoses = H5D.read( tempPoseID );
	tempGazes = H5D.read( tempGazeID );



	samplesInGroup = length( tempImgs(:,1,1,1) );
	contribOfGroup = ceil( sqrt( samplesInGroup ) );
	j = 1;
	while j <= contribOfGroup

		samplesInTree = samplesInTree + 1;
		random = randi(samplesInGroup,1,1);
		treeImgs(samplesInTree, :, :) =  tempImgs( random  ,1, :, :);
		treeGazes(samplesInTree, :) = tempGazes( random, :);%, :);
		treePoses(samplesInTree, :) = tempPoses( random, :);
		
		j = j + 1;		

	end

	H5D.close( tempImgID );
	H5D.close( tempPoseID);
	H5D.close( tempGazeID);

	H5G.close( localGrpID ) ;

end


samplesInTree
for i = 1:samplesInTree
	%treeLabel(i, 4)
end




%%%%%%%%% Close Everything %%%%%%%%%%%%%%%%%%

H5D.close(curr_rnearestID);
H5D.close(curr_centerID);
H5D.close(curr_imgsID);
H5D.close(curr_gazesID);
H5D.close(curr_posesID);



H5G.close(grpID);
H5F.close(fid);


end

%H5G.get_info(grpID)
