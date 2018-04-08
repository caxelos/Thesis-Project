%% root path: p_i...n antigrafa
%% subdirectories: F_day_i...

function my_norm(root)

rightpath = '/struct_data/struct_right/';
leftpath = '/struct_data/struct_left/';

%read from the file the 3d head pose 

% find all the sub-Directories in the current Directory 
allFiles = dir( root );
dirFlags = [allFiles.isdir];
subFolders = allFiles(dirFlags);

if length( subFolders ) > 2 
	for k = 3:length( subFolders) 	
		cd( strcat(pwd, '/', subFolders(k).name, rightpath) ); 

		[status,cmdout] = system('ls | wc -l');
		numOfImages = str2num(cmdout);                

		cd('../../../');
strcat(pwd,subFolders(k).name,rightpath)
		convert3dto2d( strcat(pwd,'/',subFolders(k).name,rightpath), numOfImages );
		convert3dto2d( strcat(pwd,'/',subFolders(k).name,leftpath), numOfImages );	
	end
end

cd('../../../'); % go to /p00/


return;

% convert the gaze direction in the camera cooridnate system to the angle
% in the polar coordinate system
gaze_theta = asin((-1)*gaze(2)); % vertical gaze angle
gaze_phi = atan2((-1)*gaze(1), (-1)*gaze(3)); % horizontal gaze angle

% save as above, conver head pose to the polar coordinate system
M = rodrigues(headpose);
Zv = M(:,3);
headpose_theta = asin(Zv(2)); % vertical head pose angle
headpose_phi = atan2(Zv(1), Zv(3)); % horizontal head pose angle


% write to file the 2d gaze

end




function convert3dto2d(pathToFiles, numOfImages)
	
	% eye gaze
	gaze3d = csvread( strcat(pathToFiles,'gaze.txt'), 0, 0, [0 0 (numOfImages-1) 2] );
	gaze2d(:,1) = asin((-1)*gaze3d(:,2));%theta
	gaze2d(:,2) = atan2((-1)*gaze3d(:, 1),(-1)*gaze3d(:,3));%phi					
	csvwrite( strcat(pathToFiles,'gaze2d.txt'), gaze2d );
	clear('gaze3d');
	clear('gaze2d');

	% eye pose
	pose3d = csvread( strcat(pathToFiles,'pose.txt'), 0, 0, [0 0 (numOfImages-1) 2] );		
	for j = 1:numOfImages
		pose3d(j,:)
		M = rodrigues( pose3d(j,:) );
		Zv = M(:,3);
		pose2d(j,1) = asin(Zv(2));
		pose2d(j,1) = atan2(Zv(1), Zv(3)); 		
	end
	csvwrite( strcat(pathToFiles,'pose2d.txt'), pose2d);
	clear('pose3d');
	clear('pose2d');	

end
