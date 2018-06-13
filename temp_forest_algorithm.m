function tempforest

clear all;
clc;

addpath /home/trakis/Downloads/MPIIGaze/Data/%@tree

R = 5;
HEIGHT = 15;%9;
WIDTH = 9;%15;
NUM_OF_GROUPS = 140;

%%%%%%%%%% Open HDF5 training file %%%%%%%%%%
fid = H5F.open('myfile.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');






samplesInTree = zeros(1,NUM_OF_GROUPS);

for i = 1:NUM_OF_GROUPS %for each tree


	
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


	samplesInGroup = length( curr_imgs(1,1,1,:) );
	contribOfGroup = ceil( sqrt( samplesInGroup ) );
	

	j = 1;
	samplesInTree(i) = 0;
	while j <= contribOfGroup
		samplesInTree(i) = samplesInTree(i) + 1;
		random = randi(samplesInGroup,1,1);

		treeImgs(i,:,:,samplesInTree(i) ) =  curr_imgs( :, :, 1, random);
		treeGazes(i,samplesInTree(i),: ) = curr_gazes(:,random);
		treePoses(i,:,samplesInTree(i) ) = curr_poses(:,random);
	
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
 
		samplesInGroup = length( tempImgs(1,1,1,:) );
		contribOfGroup = ceil( sqrt( samplesInGroup ) );
		j = 1;
		while j <= contribOfGroup


			samplesInTree(i) = samplesInTree(i) + 1;
			random = randi(samplesInGroup,1,1);
			treeImgs (i, :,:,samplesInTree(i)) =  tempImgs(:,:,1,  random);
			treeGazes(i, samplesInTree(i), :) = tempGazes(:, random);%, :);
			treePoses(i, :,samplesInTree(i)) = tempPoses( :,random);
		
			j = j + 1;		
		end

		H5D.close( tempImgID );
		H5D.close( tempPoseID);
		H5D.close( tempGazeID);

		H5G.close( localGrpID ) ;

	end
	
end



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

	
	% xtise mono 6 gia logous oikonomias. Meta vgale tin if

	trees = buildRegressionTree( samplesInTree, treeImgs,  treeGazes, HEIGHT, WIDTH);


 


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%% T E S T   P H A S E %%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%% Open HDF5 test file %%%%%%%%%%
	fid2 = H5F.open('mytest.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

	test_rnearestID      = H5D.open(fid2, '_nearestIDs');
	test_imgsID          = H5D.open(fid2, 'data');
	test_gazesID 	     = H5D.open(fid2, 'gaze');
	test_posesID	     = H5D.open(fid2, 'headpose');

	test_rnearest = H5D.read(test_rnearestID);
	test_imgs     = H5D.read(test_imgsID);
	test_gazes    = H5D.read(test_gazesID);
	test_poses    = H5D.read(test_posesID);

	ntestsamples = length( test_imgs(:,:,:,:) );
	final_error = 0;
	for j = 1:ntestsamples
	   gaze_predict = [0 0]';  
	   for k = 1:(R+1)%each samples, run the R+1 trees

		gaze_predict = gaze_predict + testSampleInTree( trees(test_rnearest(k,j) ), 1, test_imgs(:,:,1,j), test_gazes(:,j) );

	   end
	   gaze_predict = gaze_predict/(R+1);
	   final_error = final_error + abs(test_gazes(1,j) - gaze_predict(1) ) + abs( test_gazes(2,j) -gaze_predict(2) );
	  
	end
	final_error = final_error/(2*ntestsamples);
	rad2deg(final_error)
	
	fileID =  fopen('4nearest.txt','w');
	fprintf(fileID,'%f', final_error);
	fclose(fileID);
	
	H5D.close(test_rnearestID);
	H5D.close(test_imgsID);
	H5D.close(test_gazesID);
	H5D.close(test_posesID);
	 
	H5F.close(fid2);

	%%%%%%%%% Close Central Group %%%%%%%%%%%%%%%%%%
	H5D.close(curr_rnearestID);
	H5D.close(curr_centerID);
	H5D.close(curr_imgsID);
	H5D.close(curr_gazesID);
	H5D.close(curr_posesID);

	H5G.close(grpID);
H5F.close(fid);

end

function val = testSampleInTree(tree, node, test_img, gaze )
   val = [100000 100000];	


   if tree.isleaf(node) 
      val = sscanf(tree.get(node),'(%f,%f)');
	
   else

      data= sscanf(tree.get(node),'Samples:%f,px1(%f,%f)-px2(%f,%f)>=%f');
      childs = tree.getchildren(node);
      if abs(test_img(data(2), data(3), 1, 1) - test_img(data(4), data(5), 1,1)) >= data(6)
         val = testSampleInTree(tree,childs(2) , test_img, gaze );
      else
         val = testSampleInTree(tree, childs(1), test_img, gaze );
      end
      
   end


end



function treesMy = buildRegressionTree( fatherSizeX, treeImgsX,  treeGazesX, HEIGHTX, WIDTHX)
	MAX_DEPTH = 100;
	NUM_OF_WORKERS = 3;
	MAX_FATHER_SIZE = 189;%200;	
	
	treeGazes = Composite();%NUM_OF_WORKERS);
	fatherSizeTrees = Composite();%NUM_OF_WORKERS);
	treeImgs = Composite();%NUM_OF_WORKERS);
	HEIGHT = Composite();%NUM_OF_WORKERS);
	WIDTH = Composite();%NUM_OF_WORKERS);
	fatherSize = Composite();%NUM_OF_WORKERS);

	
	
	for w=1:NUM_OF_WORKERS
	   treeGazes{w} = treeGazesX;
	   fatherSize{w} = fatherSizeX;
	   treeImgs{w} = treeImgsX;
	   HEIGHT{w} = HEIGHTX;
	   WIDTH{w} = WIDTHX;
	   currPtrs{w} = [1:MAX_FATHER_SIZE];
	end

	
	c = parcluster;
	c.NumWorkers = NUM_OF_WORKERS;
	saveProfile(c);

	mypool = gcp('nocreate');
	if isempty(mypool)
	   mypool =  parpool('local',3);  
	end

        spmd;

	%%% recursion staff %%%
	savedSize = zeros(1,MAX_DEPTH);
	savedNode = zeros(1,MAX_DEPTH);
	currPtrs = zeros(1,MAX_FATHER_SIZE); 
	savedPtrs = zeros(MAX_DEPTH, MAX_FATHER_SIZE); %fatherSize(i)) ;

	

	px1_vert  = zeros(1); 
	px1_hor = zeros(1);
	px2_vert = zeros(1);
	px2_hor = zeros(1);
	counter = zeros(1);
	

	minSquareError = zeros(1,3);
	poulo = zeros(1);	
	bestworker = zeros(1);
	container = [];
	container.data = zeros(1,7);
	container.currPtrs = zeros(1, MAX_FATHER_SIZE);
	container.savedPtrs = zeros(1, MAX_FATHER_SIZE);


%	lImgs= zeros(1,MAX_FATHER_SIZE);
%	rImgs = zeros(1, MAX_FATHER_SIZE);
%	final_lImgs = zeros(1, MAX_FATHER_SIZE);
%	final_rImgs = zeros(1,MAX_FATHER_SIZE);
		

	cache_treeImgs = zeros(MAX_FATHER_SIZE, 2);
	l_r_fl_fr_imgs = zeros(4,MAX_FATHER_SIZE);

 for i = 1:140 % for every tree


       %l_r_fl_fr_imgs = zeros(4,fatherSize(i));

       fatherSize(i)

       stackindex = 0;
       state = 1;	
       trees(i) = tree(strcat('RegressionTree_', num2str(i) ));
       node_i = 1;
       currPtrs = [1:fatherSize(i)];
       while state ~= 2 
	
	   %for each node
	   minSquareError = [10000 10000 10000];
	   minPx1_vert =    10000; % something random here
	   minPx1_hor =     10000; % also here
	   minPx2_vert=     10000; % and here..
	   minPx2_hor =     10000; % and here 
	   bestThres  =     10000; % ah, and here
	 
          
	   counter = labindex;
	   while (counter <= HEIGHT*WIDTH-1)
		
	
	        px1_vert = ceil( (counter/WIDTH));
	        px1_hor =  1 +  mod(counter-1, (WIDTH) );

      	       % sorry for the huge equations below
	       % these equations are made in order to prevent 2 pixels
	       % to be examined twice

	       for px2_vert = ( px1_vert + floor(px1_hor/WIDTH)  ):HEIGHT
	          for px2_hor = (1 + mod( px1_hor, WIDTH )):WIDTH

		     %%% create a cache array (px1_vert_px1_hor, curr %%%
	             for j = 1:fatherSize(i)
		        cache_treeImgs(j,1) = treeImgs(i, px1_vert,px1_hor, currPtrs( j)  );
		        cache_treeImgs(j,2) = treeImgs(i, px2_vert,px2_hor, currPtrs( j)  );
		     end
		


                     if  sqrt( (px1_vert -px2_vert)^2+(px1_hor-px2_hor)^2 ) < 6.5             
		        for thres = 20:40
			   l = 0;
			   r = 0;			
			   meanLeftGaze = [0 0];
			   meanRightGaze = [0 0];
			   for j = 1:fatherSize(i)
 
			      if abs(  cache_treeImgs(j,1) - cache_treeImgs(j,2) ) < thres			    

			          %left child

			         l = l + 1;
			         %lImgs(l) = currPtrs(j); 
				 l_r_fl_fr_imgs(1,l) = currPtrs(j);			
			
      	           
				 meanLeftGaze(1) = meanLeftGaze(1) + treeGazes(i,currPtrs(j),1);			       
				 meanLeftGaze(2) = meanLeftGaze(2) + treeGazes(i,currPtrs(j),2);	
			      else
			            %right child
			            r = r + 1;
			            %rImgs(r) = currPtrs(j);
				    l_r_fl_fr_imgs(2,r) = currPtrs(j);
  				      
				    meanRightGaze(1) = meanRightGaze(1) + treeGazes(i,currPtrs(j),1);%,:);
				    meanRightGaze(2) = meanRightGaze(2) + treeGazes(i,currPtrs(j),2);
			       end
			    end
	
			       meanLeftGaze = meanLeftGaze  / l;
			       meanRightGaze = meanRightGaze/ r;

			       squareError = 0;
				for j = 1:l	
				   squareError=squareError + (meanLeftGaze(1)-treeGazes(i, l_r_fl_fr_imgs(1,l),1))^2 + (meanLeftGaze(2)-treeGazes(i,l_r_fl_fr_imgs(1,l),2))^2;	
			       end
			       for j = 1:r
				   squareError=squareError + (meanRightGaze(1)-treeGazes(i,l_r_fl_fr_imgs(2,r),1))^2 + (meanRightGaze(2)-treeGazes(i, l_r_fl_fr_imgs(2,r), 2))^2;	
		               end
			       
		
			       if squareError < minSquareError(labindex)
			           minSquareError(labindex) = squareError;
			           minPx1_vert =    px1_vert; % something random here
			           minPx1_hor =     px1_hor; % also here
			   	   minPx2_vert=     px2_vert; % and here..
			   	   minPx2_hor =     px2_hor; % and here
			   	   bestThres  =     thres;
			   


				   l_r_fl_fr_imgs(3,1:l) = l_r_fl_fr_imgs(1,1:l);
				   l_r_fl_fr_imgs(4,1:r) = l_r_fl_fr_imgs(2,1:r);

			   

			   	   ltreeSize = l;
			   	   rtreeSize = r;
		
                           	   rtree_meanGaze = meanRightGaze;
			   	   ltree_meanGaze = meanLeftGaze;
				end	 	
		             end%thres
		          end%end if < 6.5	
		       end%px2_hor
		    end%px2_vers 	
	         %end %px1_hor
		 counter = counter + numlabs;
           end %endof px1_vert

	   
%	  if numlabs == 3
             rcvWkrIdx = mod(labindex, numlabs) + 1; % one worker to the right
	     srcWkrIdx = mod(labindex - 2, numlabs) + 1; % one worker to the left

	     labBarrier;	 
	     %%% take data from the left and give to the right %%%
	     minSquareError( srcWkrIdx ) = labSendReceive(rcvWkrIdx,srcWkrIdx, minSquareError(labindex) );


	     labBarrier;
	     %%% take data from the right %%%
	     minSquareError(rcvWkrIdx) = labSendReceive(srcWkrIdx,rcvWkrIdx,minSquareError(labindex));

	    labBarrier;
 
	%   elseif numlabs == 2 || numlabs == 4
	      
	%      for o = 1:numlabs
%	         labBarrier;
 %                if o == labindex
%		    labBroadcast(o,minSquareError(o));	
%		 else
%		    minSquareError(o) = labBroadcast(o);
%		 end
%	      end
%	      labBarrier;
%	   end 

	   



	 %%% sychronize before finding the best worker %%%
	 bestworker = 1;
	 minError = minSquareError(1);	
	 for k = 2:numlabs
	    if minSquareError(k) < minError
	       minError = minSquareError(k);
 	       bestworker = k;
	    end
	 end
	

         if bestworker == labindex

	   %%%%%% Recursion starts here %%%%%	
	   if (ltreeSize > 0 && rtreeSize > 0)
	      state = 1;

              trees(i)=trees(i).set(node_i,strcat('Samples:',num2str(fatherSize(i)),',px1(', num2str(minPx1_vert),',',num2str(minPx1_hor),')-','px2(',num2str(minPx2_vert),',',num2str(minPx2_hor),')>=', num2str(bestThres) ));  

	      [trees(i) lnode] = trees(i).addnode(node_i, strcat('(', num2str(ltree_meanGaze(1)), ',', num2str(ltree_meanGaze(2)), ')'));
	      [trees(i) rnode] = trees(i).addnode(node_i, strcat('(', num2str(rtree_meanGaze (1)), ',', num2str(rtree_meanGaze (2)), ')'));

	      % start saving the left brother     
	      stackindex = stackindex + 1;
	      savedSize(stackindex) = ltreeSize;
	      savedNode(stackindex) = lnode;
	      savedPtrs(stackindex, 1:ltreeSize) = l_r_fl_fr_imgs(3,1:ltreeSize);
	 
	      %%%   prepare data for right son %%%
	      node_i = rnode;
	      fatherSize(i) = rtreeSize;
	      currPtrs(1:rtreeSize) =  l_r_fl_fr_imgs(4,1:rtreeSize); 
	      
 
           else  %2
	      if stackindex == 0
		 state = 2;
			         
	      else 
		state = 3;        
	     
	         fatherSize(i) = savedSize(stackindex);
	         node_i = savedNode(stackindex);
	         currPtrs(1:fatherSize(i)) = savedPtrs(stackindex,1:fatherSize(i));
	         stackindex = stackindex - 1;	

	      end
	    
	   end %2	
        end 
	

	%%% Load to container %%%
	if labindex == bestworker
	    if state == 1
		container.data = [state poulo  stackindex  fatherSize(i)  node_i  savedNode(stackindex)  savedSize(stackindex) ];
		container.trees = trees(i);
	        container.currPtrs(1:fatherSize(i) ) = currPtrs(1:fatherSize(i) );
	        container.savedPtrs(1:ltreeSize) =  l_r_fl_fr_imgs(3,1:ltreeSize);
	   			           
	    elseif state == 2
	       container.data(1) = 2;
	   
	    else %state == 3
	       container.data = [state poulo stackindex fatherSize(i) node_i ];
	       container.currPtrs(1:fatherSize(i)) = currPtrs(1:fatherSize(i));
	    end
	
	end
	        

	labBarrier;
	if labindex ~= bestworker
	    container = labBroadcast(bestworker);
	    if container.data(1) == 1 %state = 1

	       stackindex = container.data(3);
	       fatherSize(i) = container.data(4);
	       node_i = container.data(5);
	       savedNode(stackindex) = container.data(6);
	       savedSize(stackindex) = container.data(7);%ltreeSize
	       trees(i) = container.trees;	          	      

	       savedPtrs(stackindex,1:savedSize(stackindex)) = container.savedPtrs(1:savedSize(stackindex));	       
	       currPtrs(1:fatherSize(i)) = container.currPtrs(1:fatherSize(i));
	       
		

	    elseif container.data(1) == 2
	       state = 2;   


	    else  container.data(1) == 3 %[state poulo stackindex fatherSize node_i ];
		state = 3;

	        %%% o stackindex erxetai meiwmenos kata 1 %%%
	       stackindex = container.data(3);
	       fatherSize(i) = container.data(4);
	       node_i = container.data(5);
	       currPtrs(1:fatherSize(i)) = container.currPtrs(1:fatherSize(i));
	       
	    end
	else
	   labBroadcast(bestworker, container); 
	end


	%isws
	labBarrier;
       
   end %while loop


    if labindex == 1
	disp(trees(i).tostring);
    end 
  
 
     %clear('l_r_fl_fr_imgs'); 
  
 end %treeCompleted

 
  
   end%end of spmd

   treesMy = trees{1};


end

