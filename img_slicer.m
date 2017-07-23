function [filenames] = img_slicer(filename, dimSlice)
%Script for slicing up big .jp2 files.
if nargin < 2
    dimSlice = 8;
end

idx = strfind(filename,'.');
name = filename(1:(idx-1));
ext = filename(idx:end);

if(ext == '.IMG')
    fid  = fopen(filename);
    img = fread(fid);
    close(fid);
else
    if(ext == '.tif')
        tiff = Tiff(filename, 'r');
    else
        img = imread(filename);
    end
end

width = size(img,2);
len = size(img,1);

wCutSize = width/dimSlice;
lCutSize = len/dimSlice;
numImg = dimSlice * dimSlice;

imgIndex = 1:dimSlice*dimSlice;
imgIndex = reshape(imgIndex,[dimSlice, dimSlice]);

filenames = cell(numImg, 1);

for i = 1:dimSlice
    for j = 1:dimSlice
        x1 = ((i-1)*wCutSize)+1;
        x2 = i*wCutSize;
        y1 = ((j-1)*lCutSize)+1;
        y2 = j*lCutSize;
        slicedImg = img(y1:y2,x1:x2);
        filenames{imgIndex(i,j)} = strcat(name,'_p',num2str(imgIndex(i,j)),ext);
        imwrite(slicedImg, filenames{imgIndex(i,j)})
    end
end