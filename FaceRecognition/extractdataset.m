clear all;
% Root folder
imgDir = fullfile(userpath, 'faces');

% Create Image Set from data
imgSet = imageSet(fullfile(userpath, 'faces'));
s = imgSet.Count;

% Intialise Cascade Object Detector
FaceDetector = vision.CascadeObjectDetector();
FaceDetector.MergeThreshold = 7;

% Loop through all the images in the Image Set
for sc = 1:s
    % Read the image and flip upright
    croppedFace = [];
    I = read(imgSet, sc);
    if isfield(info, 'Orientation')
        orient = info(1).Orientation; 
        switch orient
           case 1

           case 2
               I = I(:,end:-1:1,:);
           case 3
               I = I(end:-1:1,end:-1:1,:);
           case 4
               I = I(end:-1:1,:,:);
           case 5
               I = permute(I, [2 1 3]);
           case 6
               I = rot90(I, 3);
           case 7
               I = rot90(I(end:-1:1,:,:));
           case 8
               I = rot90(I);
           otherwise
               disp('Unknown orientation detected.');
        end
    end
    
    % Get the coordinates of the Faces
    sbbox = step(FaceDetector, I);
    N = size(sbbox, 1);
    
    % Loop through coordinates and extract face
    for i = 1:N
        a = sbbox(i, 1);
        b = sbbox(i, 2);
        c = a+sbbox(i, 3);
        d = b+sbbox(i, 4);

        croppedFace = I(b:d, a:c, :);
    end
    
    % Resize the extracted face
    croppedFace = imresize(croppedFace, [300 300]);
    
    % Create a folder for the extracted images
    if exist('cropface', 'dir')
    else
        mkdir cropface
    end
    dirct = 'cropface/face_';
    fileType = '.jpg';
    
    halfDirct = append(dirct, int2str(sc));
    fullDirct = append(halfDirct, fileType);
    
    % Write the image to the cropface folder
    imwrite(croppedFace, fullDirct);
end


