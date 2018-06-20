function forestDB()
%%
clear;
clc;

if exist('myfile.h5', 'file') == 2
  delete('myfile.h5');
end
if exist('mytest.h5', 'file') == 2
  delete('mytest.h5');
end


TOTAL_DATA = 44640;%number of samples
TEST_SIZE= TOTAL_DATA/5;

R = 20;
WIDTH = 15;
HEIGHT = 9;

MAX_NUM_OF_GROUPS = 300;
MAX_SIZE_PER_GROUP = 1500;



%%% structure allocation %%%
groups(1:MAX_NUM_OF_GROUPS) = struct('centerTheta', zeros(1) , 'centerPhi', zeros(1), 'index', 0,  'RnearestCenters', uint16(zeros(R,1)), 'trainData', [], 'numberOfSamples', zeros(1), 'confidence', zeros(1) );




%%% construct cluster centers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Pij lists all p00, p01, p02,...
dirData = dir(pwd);
dirIndex = [dirData.isdir];
Pij = dirData(dirIndex);
%for each Pij...

 numGrps = 0;
  
 for num_Pij=3:length(Pij)
 
  filepath = strcat(Pij(num_Pij).name, '/'); %'p00/';%'MPIIGaze/';
 
  %%% LIST ALL FILES %%%
  dirData = dir(filepath);%path = dir(filepath);
  dirIndex = [dirData.isdir];
  files = {dirData(~dirIndex).name}';

  %%%% STEPS %%%%
  step_size = get_step_size( filepath, TOTAL_DATA);
  curr_step = 1;
  curr_ratio = 0;


  for num_f=1:length(files) 
    readname = [filepath, files{num_f}];
    temp = load(readname);   
    num_data = length(temp.filenames(:,1));   
    for num_i=1:num_data
     if curr_step == step_size  
        curr_step=1;  
       
      if  curr_ratio ~= 4 %0
	   curr_ratio = curr_ratio + 1;	
  
	% for left
        headpose = temp.data.left.pose(num_i, :);
        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
        phi = atan2(Zv(1), Zv(3));
          
	if can_be_center(groups, theta, phi, numGrps)
	   numGrps = numGrps + 1;
	   groups(numGrps).centerTheta = theta;
	   groups(numGrps).centerPhi = phi;
	end

         
        % for right
        headpose = temp.data.right.pose(num_i, :); 
        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
       	phi = atan2(Zv(1), Zv(3));
	if can_be_center(groups, theta, (-1)*phi, numGrps)
	   numGrps = numGrps + 1;
	   groups(numGrps).centerTheta = theta;
	   groups(numGrps).centerPhi = (-1)*phi;
	end
	
       else %curr ratio = 0
         curr_ratio = 0;
       end
      else
        curr_step = curr_step+1;
      end

    end
  end
 end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:numGrps
   groups(i).trainData = struct('gaze', zeros(2,MAX_SIZE_PER_GROUP), 'headpose', zeros(2, MAX_SIZE_PER_GROUP), 'data', zeros(WIDTH,HEIGHT,1,MAX_SIZE_PER_GROUP)  ) ; 
end




%test
testData=[];
testData.data = zeros(WIDTH,HEIGHT ,1, TEST_SIZE);
testData.gaze = zeros(2, TEST_SIZE);%15*375*2);%zeros(2, total_num*2);
testData.headpose = zeros(2, TEST_SIZE);%15*375*2);%zeros(2, total_num*2);S
testData.nTrees = zeros( R+1, TEST_SIZE);
%testData.confidence = zeros(1, 15*375*2);%zeros(1, total_num*2);
testindex = 0;

%temp
tempData=[];
tempData.data = zeros(WIDTH,HEIGHT ,1, 1*2);%zeros(WIDTH,HEIGHT ,1, total_num*2);
tempData.label = zeros(2, 1*2);%zeros(2, total_num*2);
tempData.headpose = zeros(2, 1*2);%zeros(2, total_num*2);
%tempData.confidence = zeros(1, 1*2);%zeros(1, total_num*2);



%Pij lists all p00, p01, p02,...
dirData = dir(pwd);
dirIndex = [dirData.isdir];
Pij = dirData(dirIndex);
%for each Pij...
 for num_Pij=3:length(Pij)
   filepath = strcat(Pij(num_Pij).name, '/'); %'p00/';%'MPIIGaze/';
 
  %%% LIST ALL FILES %%%
  dirData = dir(filepath);%path = dir(filepath);
  dirIndex = [dirData.isdir];
  files = {dirData(~dirIndex).name}';

  %%%% STEPS %%%%
  step_size = get_step_size( filepath, TOTAL_DATA);
  curr_step = 1;
  curr_ratio = 0;

  for num_f=1:length(files) 
   
    readname = [filepath, files{num_f}];
    temp = load(readname);   
    num_data = length(temp.filenames(:,1));   
    for num_i=1:num_data
       if curr_step == step_size  
        curr_step=1;  

	% for left
        img = temp.data.left.image(num_i, 14:22, 23:37);
        img = reshape(img, HEIGHT ,WIDTH);
       	tempData.data(:, :, 1, 1) = img'; % filp the image

        
        Lable_left = temp.data.left.gaze(num_i, :)';
        theta = asin((-1)*Lable_left(2));
        phi = atan2((-1)*Lable_left(1), (-1)*Lable_left(3));
        tempData.label(:,1) = [theta; phi];
 
        headpose = temp.data.left.pose(num_i, :);
        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
        phi = atan2(Zv(1), Zv(3));

        tempData.headpose(:,1) = [theta;phi];         
         

        % for right
        img = temp.data.right.image(num_i, 14:22, 23:37);
        img = reshape(img, HEIGHT ,WIDTH);
        tempData.data(:, :, 1, 2) = double(flip(img, 2))'; % filp the image
	         

        Lable_right = temp.data.right.gaze(num_i,:)';
        theta = asin((-1)*Lable_right(2));
        phi = atan2((-1)*Lable_right(1), (-1)*Lable_right(3));
        tempData.label(:,2) = [theta; (-1)*phi];% flip the direction


        headpose = temp.data.right.pose(num_i, :); 

        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
       	phi = atan2(Zv(1), Zv(3));
        tempData.headpose(:,2) = [theta; (-1)*phi]; % flip the direction

	
	if  curr_ratio == 4 %0
		curr_ratio = 0;
		%%%%%%%%%%%%%%%
		% TEST DATA
		%%%%%%%%%%%%%%%
		%copy left
		testindex = testindex+1;		
		testData.nTrees(1, testindex) = find_nearest_group(tempData.headpose(:,1), groups, numGrps);	
		testData.data(:, :, 1, testindex) = tempData.data(:, :, 1,1);
		testData.gaze(:,testindex) = tempData.label(:,1);
		testData.headpose(:,testindex) = tempData.headpose(:,1);
	      

		%copy right
		testindex = testindex+1;
		testData.nTrees(1, testindex) = find_nearest_group(tempData.headpose(:,2), groups, numGrps);
		testData.data(:, :, 1, testindex) = tempData.data(:, :, 1, 2);
		testData.gaze(:,testindex) = tempData.label(:,2);
		testData.headpose(:,testindex) = tempData.headpose(:,2);
		

	else %0,1,2
		curr_ratio = curr_ratio + 1;
		%%%%%%%%%%%%%%%
                % TRAINING DATA
                %%%%%%%%%%%%%%%

		%copy left
		groupID = find_nearest_group(tempData.headpose(:,1), groups, numGrps);
		groups(groupID).index = groups(groupID).index + 1;
		groups(groupID).trainData.data(:, :,1,groups(groupID).index) = tempData.data(:, :, 1,1);
		groups(groupID).trainData.gaze(:,groups(groupID).index) = tempData.label(:,1);
		groups(groupID).trainData.headpose(:,groups(groupID).index) = tempData.headpose(:,1);
		

                %copy right
		groupID = find_nearest_group(tempData.headpose(:,2), groups, numGrps);
		groups(groupID).index = groups(groupID).index + 1;
		groups(groupID).trainData.data(:, :,1,groups(groupID).index) = tempData.data(:, :, 1,2);
		groups(groupID).trainData.gaze(:,groups(groupID).index) = tempData.label(:,2);
		groups(groupID).trainData.headpose(:,groups(groupID).index) = tempData.headpose(:,2);
		
	

	end % training Or Test????

	
     else % not in the samples
	curr_step = curr_step + 1;
     end

	
    end %data per file


    fprintf('%d / %d !\n', num_f, length(files)); 
  end % for each file

 

end  % for each pij






%hold on;
%axis([-1 1 -1 1]);
%title( strcat('Head pose distribution of 44640 samples. Num of Centers: ', num2str(numGrps)) );
%xlabel('Theta angle(radians)');
%ylabel('Phi angle(radians)');

%for i = 1:numGrps

 %scatter( groups(i).trainData.headpose(1,:), groups(i).trainData.headpose(2,:), '*', 'b' ); 
 %hold on; 
 %scatter( groups(i).centerHor , groups(i).centerVert, '*', 'g' );      
 
 %hold on;
%end
%grid on;
%legend('training samples', 'cluster centers');
%hold off;



%start creating data file for training(HDF5)
fid = H5F.create('myfile.h5');
type_id = H5T.copy('H5T_NATIVE_DOUBLE');
dcpl = 'H5P_DEFAULT';
plist = 'H5P_DEFAULT';


for i = 1:numGrps
	
	groups(i).trainData.data = groups(i).trainData.data;%/255; %normalize
	groups(i).trainData.data = single(groups(i).trainData.data); % must be single data, because caffe want
	groups(i).trainData.gaze = single(groups(i).trainData.gaze);	
	groups(i).trainData.headpose = single(groups(i).trainData.headpose);
	grp = H5G.create(fid, strcat('g', num2str(i)) ,plist,plist,plist);


%%%%%% Dataset 1: numx1xHEIGHTxWIDTH image data %%%%	
	dims = [WIDTH HEIGHT 1  groups(i).index];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(4,h5_dims,h5_maxdims);

	dset = H5D.create(grp,strcat('/g', num2str(i), '/data') ,type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,   groups(i).trainData.data(:,:,1,1:groups(i).index) );

	H5D.close(dset);
	H5S.close(space_id);


%%%%%% Dataset 2: numx4 pose and gaze data %%%%	

	dims =  [2 groups(i).index];%[groups(i).index 4];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(2,h5_dims,h5_maxdims);

	%headpose
	dset = H5D.create(grp,strcat('/g', num2str(i),'/headpose'), type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, groups(i).trainData.headpose(:,1:groups(i).index));
	H5D.close(dset);

	%gaze
	dset = H5D.create(grp,strcat('/g', num2str(i),'/gaze'), type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, groups(i).trainData.gaze(:,1:groups(i).index));
	H5D.close(dset);

	H5S.close(space_id);

%%%%%% Dataset 3: headpose-center of each group  %%%%	
	dims = [2 1];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(2,h5_dims,h5_maxdims);

	dset = H5D.create(grp,strcat('/g', num2str(i),'/center'), type_id,space_id,dcpl);	
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,[groups(i).centerTheta groups(i).centerPhi] );
 
	H5D.close(dset);
	H5S.close(space_id);

%%%%%% Dataset 4: List of R-nearest groups %%%%

	listOfGroupIds = find_R_nearest_groups(groups(i).centerTheta, groups(i).centerPhi, groups, R, [i], numGrps );
	i
	dims = [(R+1) 1];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(2,h5_dims,h5_maxdims);

	dset = H5D.create(grp,strcat('/g', num2str(i),'/',num2str(R),'_nearestIDs'), type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, listOfGroupIds );
	H5D.close(dset);
	H5S.close(space_id);	


%%%%%% Dataset 5: Number of samples per group %%%%

	dims = [1];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(1,h5_dims,h5_maxdims);

	dset = H5D.create(grp,strcat('/g', num2str(i),'/','samples'), type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, groups(i).index );
	H5D.close(dset);
	H5S.close(space_id);	




end
H5F.close(fid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testData.data = testData.data;%/255; %normalize
testData.data = single(testData.data); % must be single data, because caffe want float type
testData.gaze = single(testData.gaze);
testData.headpose = single(testData.headpose);


fid = H5F.create('mytest.h5');

%%%%%% Dataset 1: numx1xHEIGHTxWIDTH image data %%%%	

	dims = [WIDTH HEIGHT 1 testindex];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(4,h5_dims,h5_maxdims);

	dset = H5D.create(fid, '/data' ,type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,testData.data(:,:,1,1:testindex) );

	H5D.close(dset);
	H5S.close(space_id);




%%%%%% Dataset 2: numx4 pose and gaze data %%%%	

	dims = [2 testindex];%[groups(i).index 4];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(2,h5_dims,h5_maxdims);

	%headpose
	dset = H5D.create(fid, '/headpose', type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, testData.headpose(:,1:testindex));
	H5D.close(dset);

	%gaze
	dset = H5D.create(fid, '/gaze', type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist, testData.gaze(:,1:testindex));

	H5D.close(dset);
	H5S.close(space_id);


%%%%%% Dataset 3: List of R-nearest groups %%%%
	for o = 1:testindex
	   testData.nTrees(1:(R+1), o) =find_R_nearest_groups( groups(testData.nTrees(1,o)).centerTheta, groups(testData.nTrees(1,o)).centerPhi, groups, R, [testData.nTrees(1,o)], numGrps);
	
	end


	dims = [(R+1) testindex];
	h5_dims = fliplr(dims);
	h5_maxdims = h5_dims;
	space_id = H5S.create_simple(2,h5_dims,h5_maxdims);

	dset = H5D.create(fid, '_nearestIDs', type_id,space_id,dcpl);
	H5D.write(dset,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,  testData.nTrees( 1:(R+1), 1:testindex  )  );
	H5D.close(dset);
	H5S.close(space_id);	

fprintf('done\n\n');	


tempforest(numGrps);
end





function answer = can_be_center(groups, theta, phi, numGrps)
   MIN_DISTANCE_BETWEEN_CENTERS = 0.06;

   answer = 1;
   for i = 1:numGrps
     	
      if  sqrt( (groups(i).centerTheta-theta)^2 + (groups(i).centerPhi - phi)^2 ) < MIN_DISTANCE_BETWEEN_CENTERS 
         answer = 0; 
         break;	
      end
   end

end





function nearestGroup = find_nearest_group(headpose, groups, NUM_OF_GROUPS)
  minDist = 100;
  nearestGroup = -1;   

  MAX_THETA_DIST = 0.3;
  MAX_PHI_DIST = 0.3;
  
  for i =1:NUM_OF_GROUPS
     if  abs(groups(i).centerPhi - headpose(2)) < MAX_PHI_DIST && abs(groups(i).centerTheta - headpose(1)) < MAX_THETA_DIST     
	dist =  sqrt( (groups(i).centerPhi - headpose(2))^2 ) + (groups(i).centerTheta - headpose(1))^2 ;
        if  dist < minDist
	   minDist = dist;
	   nearestGroup = i;
	end
      end
  end
 

end




function listOfGroupIds = find_R_nearest_groups(centerTheta, centerPhi, groups, R, first, NUM_OF_GROUPS)

  listOfGroupIds = zeros(1, (R+1));
  listOfGroupIds(1) = first;
  minDist = zeros(1, (R+1));

  
  for i=1:(R+1)
     minDist(i) = 7+i;
  end  
 

  for i =1:NUM_OF_GROUPS

     if ismember(i, listOfGroupIds ) == 0 
	dist =  sqrt( (groups(i).centerTheta - centerTheta)^2 + (groups(i).centerPhi - centerPhi)^2 );
	if dist < minDist(R+1) %apodosi
	   for o = 2:R+1 

	      % apo to megalutero sto mikrotero	 
	      if dist < minDist(o)
		 if o == R+1%last
	 	    listOfGroupsIds(o) = i;
		    minDist(o) = dist;
		 else
                    for j = R:-1:o
		       listOfGroupIds(j+1) = listOfGroupIds(j);
		       minDist(j+1) = minDist(j);
		    end
		    listOfGroupIds(o)= i;
		    minDist(o) = dist;
		  
		 end
		 break;
	      end
	   end
        end
      end
  end
 
end






function out = get_step_size( Pij, TOTAL_DATA)



if strcmp(Pij, 'p00/')
	out =  ceil( (TOTAL_DATA*18)/44640 );%18;

elseif strcmp(Pij, 'p01/')
	out = ceil( (TOTAL_DATA*16)/44640 );%16;

elseif strcmp(Pij, 'p02/')
	out = ceil( (TOTAL_DATA*18)/44640 );%18;

elseif strcmp(Pij, 'p03/')
	out = ceil( (TOTAL_DATA*24)/44640 );%24;

elseif strcmp(Pij, 'p04/')
	out = ceil( (TOTAL_DATA*12)/44640 );%11;

elseif strcmp(Pij, 'p05/')
	out = ceil( (TOTAL_DATA*12)/44640 );%12;

elseif strcmp(Pij, 'p06/')
	out = ceil( (TOTAL_DATA*12)/44640 );%12;

elseif strcmp(Pij, 'p07/')
	out = ceil( (TOTAL_DATA*10)/44640 );%10;

elseif strcmp(Pij, 'p08/')
	out = ceil( (TOTAL_DATA*7)/44640 );%7;

elseif strcmp(Pij, 'p09/')
	out = ceil( (TOTAL_DATA*5)/44640 );%5;

elseif strcmp(Pij, 'p10/')
	out = ceil( (TOTAL_DATA*2)/44640 );%3;

elseif strcmp(Pij, 'p11/')
	out = ceil( (TOTAL_DATA*2)/44640 );%2;

elseif strcmp(Pij, 'p12/')
	out = ceil( (TOTAL_DATA*1)/44640 );%1;

elseif strcmp(Pij, 'p13/')
	out = ceil( (TOTAL_DATA*1)/44640 );%1;

else 
	out = ceil( (TOTAL_DATA*1)/44640 );%1;
end


end



