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
trainingData=[];
trainingData.data = zeros(60,36,1, 1125*2);%zeros(60,36,1, total_num*2);
trainingData.label = zeros(2, 1125*2);%zeros(2, total_num*2);
trainingData.headpose = zeros(2, 1125*2);%zeros(2, total_num*2);
trainingData.confidence = zeros(1, 1125*2);%zeros(1, total_num*2);
trainingindex = 0;

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
        index = index+1;
        img = temp.data.left.image(num_i, :,:);
        img = reshape(img, 36,60);
        Data.data(:, :, 1, index) = img'; % filp the image
        
        Lable_left = temp.data.left.gaze(num_i, :)';
        theta = asin((-1)*Lable_left(2));
        phi = atan2((-1)*Lable_left(1), (-1)*Lable_left(3));
        Data.label(:,index) = [theta; phi];
 
        headpose = temp.data.left.pose(num_i, :);
        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
        phi = atan2(Zv(1), Zv(3));
        Data.headpose(:,index) = [theta;phi];         
         
        % for right
        index = index+1;
        img = temp.data.right.image(num_i, :,:);
        img = reshape(img, 36,60);
        Data.data(:, :, 1, index) = double(flip(img, 2))'; % filp the image
         
        Lable_right = temp.data.right.gaze(num_i,:)';
        theta = asin((-1)*Lable_right(2));
        phi = atan2((-1)*Lable_right(1), (-1)*Lable_right(3));
        Data.label(:,index) = [theta; (-1)*phi];% flip the direction

        headpose = temp.data.right.pose(num_i, :); 
        M = rodrigues(headpose);
        Zv = M(:,3);
        theta = asin(Zv(2));
       	phi = atan2(Zv(1), Zv(3));
        Data.headpose(:,index) = [theta; (-1)*phi]; % flip the direction


	if mod(curr_sample, ratio) == 0
		%%%%%%%%%%%%%%%
		% TEST DATA
		%%%%%%%%%%%%%%%
		%copy left
		testindex = testindex+1;
		testData.data(:, :, 1, trainingindex)
		testData.label(:,index)
		testData.headpose(:,index)

		%copy right
		testindex = testindex+1;
		testData.data(:, :, 1, index);
		testData.label(:,index);
		testData.headpose(:,index)
	else
		%%%%%%%%%%%%%%%
                % TRAINING DATA
                %%%%%%%%%%%%%%%


		%copy left
                trainingindex = trainingindex+1;
                trainingData.data(:, :, 1, trainingindex)
                trainingData.label(:,index)
                Data.headpose(:,index)

                %copy right
                trainingindex = trainingindex+1;
                Data.data(:, :, 1, index);
                Data.label(:,index);
                Data.headpose(:,index)



	end	
    end

    fprintf('%d / %d !\n', num_f, length(files)); 
end
 
fprintf('Saving\n');

Data.data = Data.data/255; %normalize
Data.data = single(Data.data); % must be single data, because caffe want float type
Data.label = single(Data.label);
Data.headpose = single(Data.headpose);

savename = 'MPII_data.h5';

%store2hdf5(savename, Data.data, Data.label, 1, 1); % the store2hdf5 function comes from https://github.com/BVLC/caffe/pull/1746
%% You can also use the matlab function for hdf5 saving:
 hdf5write(savename,'/data', Data.data, '/label',[Data.label; Data.headpose]); 
fprintf('done\n');


end