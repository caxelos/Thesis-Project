%%
clear;
clc;



%Pij lists all p00, p01, p02,...
dirData = dir(pwd);
dirIndex = [dirData.isdir];
Pij = dirData(dirIndex);

%for each Pij...
for num_Pij=3:length(Pij)
filepath = strcat(Pij(num_Pij).name, '/'); %'p00/';%'MPIIGaze/';
 
dirData = dir(filepath);%path = dir(filepath);
dirIndex = [dirData.isdir];
files = {dirData(~dirIndex).name}';
%path = path(3:end);
%files = {path.name};

%total_num=0;
%for num_f=1:length(files)
%    readname = [filepath, files{num_f}];
%    temp = load(readname);  
%    total_num = total_num+length(temp.filenames(1,:));%(temp.errors);
%end

fprintf('File Path Ready!\n');

%training
trainData=[];
trainData.data = zeros(60,36,1, 1125*2);%zeros(60,36,1, total_num*2);
trainData.label = zeros(2, 1125*2);%zeros(2, total_num*2);
trainData.headpose = zeros(2, 1125*2);%zeros(2, total_num*2);
trainData.confidence = zeros(1, 1125*2);%zeros(1, total_num*2);
trainindex = 0;

%test
testData=[];
testData.data = zeros(60,36,1, 375*2);%zeros(60,36,1, total_num*2);
testData.label = zeros(2, 375*2);%zeros(2, total_num*2);
testData.headpose = zeros(2, 375*2);%zeros(2, total_num*2);
testData.confidence = zeros(1, 375*2);%zeros(1, total_num*2);
testindex = 0;

%temp
tempData=[];
tempData.data = zeros(60,36,1, 1*2);%zeros(60,36,1, total_num*2);
tempData.label = zeros(2, 1*2);%zeros(2, total_num*2);
tempData.headpose = zeros(2, 1*2);%zeros(2, total_num*2);
tempData.confidence = zeros(1, 1*2);%zeros(1, total_num*2);
tempindex = 0;


 
step_size = get_step_size(1500, filepath);
ratio = 3; % 75% are for training, 25% for test
curr_sample = 0;

for num_f=1:length(files) 


    readname = [filepath, files{num_f}];
    temp = load(readname);   
    num_data = length(temp.filenames(:,1));   
    for num_i=1:num_data
	
        % for left
        img = temp.data.left.image(num_i, :,:);
        img = reshape(img, 36,60);
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
        img = temp.data.right.image(num_i, :,:);
        img = reshape(img, 36,60);
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


	if mod(curr_sample, ratio) == 0
		%%%%%%%%%%%%%%%
		% TEST DATA
		%%%%%%%%%%%%%%%
		%copy left
		testindex = testindex+1;
		testData.data(:, :, 1, testindex)
		testData.label(:,testindex)
		testData.headpose(:,testindex)

		%copy right
		testindex = testindex+1;
		testData.data(:, :, 1, testindex);
		testData.label(:,testindex);
		testData.headpose(:,testindex)
	else
		%%%%%%%%%%%%%%%
                % TRAINING DATA
                %%%%%%%%%%%%%%%

		%copy left
                trainindex = trainindex+1;
                trainData.data(:, :, 1, trainindex)
                trainData.label(:,trainindex)
                trainData.headpose(:,trainindex)

                %copy right
                trainindex = trainindex+1;
                trainData.data(:, :, 1, trainindex);
                trainData.label(:,trainindex);
                trainData.headpose(:,trainindex)

	end	
    end

    fprintf('%d / %d !\n', num_f, length(files)); 
end
 
fprintf('Saving\n');

testData.data = testData.data/255; %normalize
testData.data = single(testData.data); % must be single data, because caffe want float type
testData.label = single(testData.label);
testData.headpose = single(testData.headpose);

trainData.data = trainData.data/255; %normalize
trainData.data = single(trainData.data); % must be single data, because caffe want
trainData.label = single(trainData.label);
trainData.headpose = single(trainData.headpose);



savename = 'MPII_traindata.h5';
%store2hdf5(savename, Data.data, Data.label, 1, 1); % the store2hdf5 function comes from https://github.com/BVLC/caffe/pull/1746
%% You can also use the matlab function for hdf5 saving:
 hdf5write(savename,'/data', trainData.data, '/label',[trainData.label;
trainData.headpose]); 
fprintf('done\n');


savename = 'MPII_testdata.h5';
%store2hdf5(savename, Data.data, Data.label, 1, 1); % the store2hdf5 function co
%% You can also use the matlab function for hdf5 saving:
 hdf5write(savename,'/data', testData.data, '/label',[testData.label;
testData.headpose]); 
fprintf('done\n');





end