function forest_algorithm()

clear all;
clc;

addpath /home/trakis/Downloads/MPIIGaze/Data/Normalized%@tree

R = 5;
HEIGHT = 9;
WIDTH = 15;



%%%%%%%%%% Open HDF5 training file %%%%%%%%%%
fid = H5F.open('myfile.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');






for i = 1:140 %for each tree


	samplesInTree(i) = 0;
	%%%%%%%%%% Start with the central group %%%%%%%%%%
	grpID = H5G.open(fid, strcat('/g',num2str(i)) );

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

		samplesInTree(i) = samplesInTree(i) + 1;
		random = randi(samplesInGroup,1,1);
		treeImgs (i, samplesInTree(i), :, :) =  curr_imgs( random  ,1, :, :);
		treeGazes(i, samplesInTree(i), :) = curr_gazes( random, :);%, :);
		treePoses(i, samplesInTree(i), :) = curr_poses( random, :);
		
		j = j + 1;		

	end



	%%%%%%%% Now, continue with the R-nearest %%%%%%%%%

	for k = 1:R 
			
		localGrpID  = H5G.open(fid, strcat('/g', num2str( curr_rnearest(k))   )); 

		tempImgID  = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(k) ), '/data') );
		tempPoseID = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(k) ), '/headpose') );
		tempGazeID = H5D.open( localGrpID,  strcat('/g', num2str( curr_rnearest(k) ), '/gaze') );
	
		tempImgs = H5D.read( tempImgID );
		tempPoses = H5D.read( tempPoseID );
		tempGazes = H5D.read( tempGazeID );

		samplesInGroup = length( tempImgs(:,1,1,1) );
		contribOfGroup = ceil( sqrt( samplesInGroup ) );
		j = 1;
		while j <= contribOfGroup

			samplesInTree(i) = samplesInTree(i) + 1;
			random = randi(samplesInGroup,1,1);
			treeImgs (i, samplesInTree(i), :, :) =  tempImgs( random  ,1, :, :);
			treeGazes(i, samplesInTree(i), :) = tempGazes( random, :);%, :);
			treePoses(i, samplesInTree(i), :) = tempPoses( random, :);
		
			j = j + 1;		

		end

		H5D.close( tempImgID );
		H5D.close( tempPoseID);
		H5D.close( tempGazeID);

		H5G.close( localGrpID ) ;

	end


	clear('curr_rnearest');
	clear('curr_center');
	clear('curr_imgs');
	clear('curr_gazes');
	clear('curr_poses');
end
i = 1;

	%%%%%%%% Now that we created each tree's data, lets implement the algorithm %%%%%%%%%
	% - am really thankful to http://tinevez.github.io/matlab-tree/index.html
	%
	% - Each node:
	%      a) is named '(px1,px2), thres'
	%      b) has variable name: node(k)  
	%	
	% - node(k) can have:
	%      a) parent node(k/2 ) 		
	%      b) left child(2k)
	%      c) right child(2k+1)
	% - Leaves can have:
	%      d) left 2d gaze angle
	%      e) right 2d gaze angle	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








	%%%%%% start by creating the root node of the tree %%%%%%%%%%%%%%%%%%%%

	trees = tree(strcat('RegressionTree_', num2str(i) ));
	[trees node1] = trees.addnode(1, 'NULL');
	[trees node2] = trees.addnode(node1, 'meanLeftGaze');
	[trees node3] = trees.addnode(node1, 'meanRightGaze');

	
	
	if (i == 1)
	   disp(trees.tostring);
	end




	
	%treeGazes(i, samplesInTree(i), :) = tempGazes( random, :);%, :);
	%treePoses(i, samplesInTree(i), :) = tempPoses( random, :);

	%nodeImgs
	%node



	%for each node
	minSquareError = 10000; % a huge value
	minPx1_vert =    10000; % something random here
	minPx1_hor =     10000; % also here
	minPx2_vert=     10000; % and here..
	minPx2_hor =     10000; % and here 
	for px1_vert = 1:HEIGHT
	   for px1_hor = 1:WIDTH
	   	% sorry for the huge equations below
		% these equations are made in order to prevent 2 pixels
		% to be examined twice
		 
		for px2_vert = ( px1_vert + floor(px1_hor/WIDTH)  ):HEIGHT
		  for px2_hor = (1 + mod( px1_hor, WIDTH )):WIDTH
                    if  sqrt( (px1_vert-px2_vert)^2+(px1_hor-px2_hor)^2) < 6.5             
		     for thres = 1:50
			


			l = 0;
			r = 0;			
			meanLeftGaze = [0 0];
			meanRightGaze = [0 0];
			for j = 1:samplesInTree(i)
                            	
			   if  abs(treeImgs(i, j, px1_vert, px1_hor) - treeImgs(i,j,px2_vert, px2_hor))  < thres 
			      %left child

			      l = l + 1;
			      lChild( l ).gazes = treeGazes(i, j, :); 		
			      meanLeftGaze(1) = meanLeftGaze(1) + treeGazes(i,j,1);		
			      meanLeftGaze(2) = meanLeftGaze(2) + treeGazes(i,j,2);	
			   else
			      %right child

			      r = r + 1;
			      rChild( r ).gazes = treeGazes(i, j, :); 
			      meanRightGaze(1) = meanRightGaze(1) + treeGazes(i,j,1);		
			      meanRightGaze(2) = meanRightGaze(2) + treeGazes(i,j,2);			
			   end


			end
		
			


			meanLeftGaze = meanLeftGaze  / l;
			meanRightGaze = meanRightGaze/ r;

			squareError = 0;
			for j = 1:r
			   squareError=squareError + (meanRightGaze(1)-rChild(j).gazes(1))^2 + (meanRightGaze(2)-rChild(j).gazes(2))^2;	
			end
			for j = 1:l
			   squareError=squareError + (meanLeftGaze(1)-lChild(j).gazes(1))^2 + (meanLeftGaze(2)-lChild(j).gazes(2))^2;	
			end

			if squareError < minSquareError
			   minSquareError = squareError;			
			   minPx1_vert =    px1_vert; % something random here
			   minPx1_hor =     px1_hor; % also here
			   minPx2_vert=     px2_vert; % and here..
			   minPx2_hor =     px2_hor; % and here
			end
 			 		 	
		     end%thres
		    end%end if < 6.5
		
		   end%px2_hor
		end%px2_vers
 	
	   end
	end		
%(1,1)->(1,2)
        minPx2_vert
	minPx2_hor 

	%%%%%%%%% Close Central Group %%%%%%%%%%%%%%%%%%
	H5D.close(curr_rnearestID);
	H5D.close(curr_centerID);
	H5D.close(curr_imgsID);
	H5D.close(curr_gazesID);
	H5D.close(curr_posesID);

	H5G.close(grpID);

%end i-loop	


%treeImgs (100, samplesInTree(i), HEIGHT, WIDTH)
%treePoses(100, 1:samplesInTree(100), 1)


H5F.close(fid);


end

%H5G.get_info(grpID) 
