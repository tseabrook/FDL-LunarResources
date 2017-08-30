function [filenames] = img_slicer(filename, dimSlice)
%Script for slicing up big .jp2 files.
if nargin < 2
    dimSlice = [32, 1];
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
        img = imread(filename);
    else
        img = imread(filename);
    end
end

width = size(img,2);
len = size(img,1);

wCutSize = width/(dimSlice(2));
lCutSize = len/(dimSlice(1));
numImg = dimSlice(1) * dimSlice(2);

imgIndex = 1:dimSlice(1)*dimSlice(2);
imgIndex = reshape(imgIndex,[dimSlice(2), dimSlice(1)]);

filenames = cell(numImg, 1);

for i = 1:dimSlice(2)
    for j = 1:dimSlice(1)
        x1 = ((i-1)*wCutSize)+1;
        x2 = i*wCutSize;
        y1 = ((j-1)*lCutSize)+1;
        y2 = j*lCutSize;
        slicedImg = img(y1:y2,x1:x2);
        filenames{imgIndex(i,j)} = strcat(name,'_p',num2str(imgIndex(i,j)),ext);
        if(ext == '.tif')
            t = Tiff(filenames{imgIndex(i,j)},'w');
            
            setTag(t,'ImageLength',size(slicedImg,1))
            setTag(t,'ImageWidth',size(slicedImg,2))
            setTag(t,'Photometric',Tiff.Photometric.MinIsBlack)
            setTag(t,'BitsPerSample',8)
            setTag(t,'SamplesPerPixel',size(slicedImg,3))
            setTag(t,'TileWidth',128)
            setTag(t,'TileLength',128)
            setTag(t,'Compression',Tiff.Compression.JPEG)
            setTag(t,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky)
            setTag(t,'Software','MATLAB')
            
            write(t,im2uint8(slicedImg));
            close(t)
        else
            imwrite(slicedImg, filenames{imgIndex(i,j)});
        end
    end
end