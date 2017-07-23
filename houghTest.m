close all

filename = 'imgs/testdata/m108898482_cdr_jp2';
all_bbox = [];

for j = 1:256
    filename_curr = strcat(filename,'_p',num2str(j),'.jp2');
    A = imread(filename_curr);
    %Variables for imfindcircle function
    rmin = 11:10:111; %Minimum radius of circle
    rmax = 20:10:120; %Maximum radius of circle
    numIter = length(rmin); %Number of different radii

    %Variables for bounding box
    bbox = []; %bounding box coordinates (NW,NE,SW,SE)
    radius_multiplier = 2; %increase radius by this value

for i = 1:numIter
    
    if rmax(i) < 50
        threshold = 0.85; %Sensitivity for small circles
    else
        if rmax(i) < 70
           threshold = 0.90; %Sensitivity for medium circles
        else
            threshold = 0.97; %Sensitivity for large circles
        end
    end
    [centers,radii,metric] = imfindcircles(A,[rmin(i), rmax(i)],'Sensitivity', threshold);

    radii = floor(radii * radius_multiplier); %Calculate enlarged radius.
    if ~isempty(centers)
        new_bbox(:,1) = centers - radii; %NW
        new_bbox(:,2) = [centers(:,1) - radii, centers(:,2) + radii]; %NE
        new_bbox(:,3) = [centers(:,1) + radii, centers(:,2) - radii]; %SW
        new_bbox(:,4) = centers + radii; %SE
        all_bbox = cat(1,all_bbox,bbox);
    end
end

end


