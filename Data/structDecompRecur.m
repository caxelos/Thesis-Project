%%  A directory is created for each struct
%% See the components of each struct-field
%% If one field is a struct again, do recursion
%% Else create a file for each struct-field

function structDecompRecur(root, struct_ptr, struct_name)
	
	structFields = fieldnames( struct_ptr )
	folder_struct = strcat(root, 'struct_', struct_name)
        mkdir( folder_struct );

	for eachNum = 1:length(structFields)
		cond = isstruct( struct_ptr.(structFields{eachNum}) );%eval(structFields{eachNum} )) 
		if  cond == 1	
			structDecompRecur(strcat(folder_struct,'/'),struct_ptr.(structFields{eachNum}),structFields{eachNum} );
		else

			% if field is image, create a folder and extract it
			% here. Be careful. If an image has dimensions
			% 200x360 there will be a large file with 200 rows
			% and cols!
			if strcmp('image', structFields{eachNum})

				mkdir( strcat( folder_struct,'/images') );
				 
				numOfTxtsPerDir = size(struct_ptr.(structFields{eachNum}), 1)
				for i = 1:numOfTxtsPerDir
					for j = 1:36 
						dlmwrite( strcat(folder_struct,'/images/', num2str(i),'.txt'), struct_ptr.(structFields{eachNum})(i,j,:), '-append' )
					end
				end

			else
				dlmwrite( strcat( folder_struct,'/', structFields{eachNum}, '.txt'),struct_ptr.(structFields{eachNum}))%matData.(matField{fieldNum}) );
		end
	end
end 