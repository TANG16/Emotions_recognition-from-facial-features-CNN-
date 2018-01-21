function Iout = readAndPreprocessImage(filename)

hgamma = vision.GammaCorrector(2.0,'Correction','De-gamma');
image = imread(filename);
image=image(:,:,1);
image_bw=mat2gray(image);
image_gamma = step(hgamma, image_bw);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
% if ismatrix(image_gamma)
%     image_gamma = cat(3,image_gamma,image_gamma,image_gamma);
% end

% Resize the image as required for the CNN.
Iout = imresize(image_gamma,[48 48]);% [224 224]); 

% Note that the aspect ratio is not preserved. In Caltech 101, the
% object of interest is centered in the image and occupies a
% majority of the image scene. Therefore, preserving the aspect
% ratio is not critical. However, for other data sets, it may prove
% beneficial to preserve the aspect ratio of the original image
% when resizing.
end