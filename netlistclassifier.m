clear all,clc,close all
% netlist classifier

path_directory='jsonfile/isolated/nonisolated'; 
 % Pls note the format of files,change it as required
original_files=dir([path_directory '/*.json']);
for k=1:length(original_files)
    fname=[path_directory '/' original_files(k).name];
    
    fid = fopen(fname);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);

    ncomp = length(val);
    nnode = length(val{1,1});

    for i=1:ncomp
        for j=1:nnode
            row = val{i,1};
            if j==1
                icmatrix{i,j} = row{j,1};
            else
                icmatrix{i,j} = int8(str2num(row{j,1}));
            end
        end
    end

    newmatrix = zeros(16,16);
    for i=1:ncomp
        for j=2:nnode
            if icmatrix{i,j} ~= 0
                switch icmatrix{i,1}
                    case "X"
                        newmatrix(i,j-1)=20+icmatrix{i,j};
                    case "I"
                        newmatrix(i,j-1)=40+icmatrix{i,j};
                    case "V"
                        newmatrix(i,j-1)=60+icmatrix{i,j};
                    case "D"
                        newmatrix(i,j-1)=80+icmatrix{i,j};
                    case "L"
                        newmatrix(i,j-1)=100+icmatrix{i,j};
                    case "C"
                        newmatrix(i,j-1)=120+icmatrix{i,j};
                    case "R"
                        newmatrix(i,j-1)=140+icmatrix{i,j};
                    case "M"
                        newmatrix(i,j-1)=160+icmatrix{i,j};
                end
            end
        end
    end

    newmatrix( ~any(newmatrix,2), : ) = [];  %rows
    newmatrix( :, ~any(newmatrix,1) ) = [];  %columns
    newmatrix(32,32) = 0;

    image = mat2gray(newmatrix,[0 255]);
    imwrite(image, erase([path_directory '/' original_files(k).name '.png'],'.json'));
end