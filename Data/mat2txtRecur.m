%% CONVERT .mat files to .txt files

% Search recursively, until you find a leave in the subFolder system
% When you find a leafe, check if .mat files exist
% if they exist run the function

function mat2txtRecur(root)

%%%%%%%%%%%%%%%%%%%%%% find recursively all subfolders 

%root = strcat(pwd, '/'); You must define from matlab terminal the
%top-directory to start the recursive search for matfiles


% find all the sub-Directories in the current Directory 
allFiles = dir( root );
dirFlags = [allFiles.isdir];
subFolders = allFiles(dirFlags);

% i explain just now why is the ">2".
% In each directory exist 2 sub-Directories. The "." and ".."
% So i must already count them in the total sub-directories of the current one
if length( subFolders ) > 2 
	
	for k = 3:length( subFolders) 	
                % here runs the recursion. Do the previous job but for the
                % following sub-folder. Initial root value is defined in
                % command line
		mat2txtRecur( strcat(root, subFolders(k).name, '/') )%% recursion

	end
else

%%%%%%%%%%%%%%%%%%%%% now make all .mat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% find all ".mat" files in a directory
dirData = dir( strcat(root,'*.mat') );      %# Get the data for the current directory
dirIndex = [dirData.isdir];  %# Find the index for directories
mats = {dirData(~dirIndex).name}';


% if we have for example a file called "mymat.mat".
% I create a folder called "F_mymat.mat" and i "extract" here the data of ".mat" file
% This happens for each ".mat" file

% for each ".mat" file
for matNum = 1:length(mats)

	% load the data of ".mat" file. Advice: Run this command in Matlab
	% terminal to see the contents;)	
	matData = load( strcat(root, mats{matNum})  );
	
	% array with components of ".mat"
	% each ".mat" file may have data about vectors, matrices, structs,
	% images, etc.
	% Each of these different types are extracted differently
	% If the ".mat" contains a struct, i create a folder called
	%    "struct_'structName'", where structName is substituted  
	% Inside this folder i extract the struct fields as ".txt"
        % If the ".mat" contains an image, I create a folder called image.
	% Be careful. It may not run if you have 2 images in the same
	% Directory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	matField = fieldnames(matData);



	% create a folder for each ".mat" file. Look the path carefully	
	folder_mat =  strcat(root, 'F_', mats{matNum} );
	mkdir( folder_mat ) ; 



	%% if it is a struct, take all struct members
	
	% for each "dataField" of a ".mat" file
        for fieldNum = 1:length(matField)	
	
		% if the field is "struct", call "structDecompRecur" to
		% complete the job 
		cond = isstruct( matData.(matField{fieldNum}) );
		if cond == 1
			
			struct_name = matField{fieldNum};% name of struct(string)
 	                struct_ptr = matData.(matField{fieldNum}); % data of struct(pointer)
			structDecompRecur(strcat(folder_mat,'/'), struct_ptr, struct_name ); % 1st param is the new path "struct_structName/"
		else
			% write each field in a different ".txt" file.
			% Columns differ by ',' character
			% Rows differ by '\n' character
			dlmwrite( strcat( folder_mat,'/',matField{fieldNum},'.txt'),matData.(matField{fieldNum}) );    
	
		end
	end
        

end
end 
end


