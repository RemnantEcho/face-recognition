clear all;

% Load image
fileName = 'Class.JPG';
I = imread(fileName);
info = imfinfo(fileName);

% Reset Orientation from EXIF info
if isfield(info, 'Orientation')
   orient = info(1).Orientation; 
   switch orient
       case 1
           
       case 2
           I = I(:,end:-1:1,:); % Right to left
       case 3
           I = I(end:-1:1,end:-1:1,:); % 180 degrees rotation
       case 4
           I = I(end:-1:1,:,:); % bottom to top
       case 5
           I = permute(I, [2 1 3]); % counterclockwise and upside down
       case 6
           I = rot90(I, 3); % undo 90 degrees
       case 7
           I = rot90(I(end:-1:1,:,:)); % undo clockise and flipped left/right
       case 8
           I = rot90(I); % undo 270 degrees rotation
       otherwise
           disp('Unknown orientation detected.');
   end
end

% Run face recognition function
% I - image
% "hog", "surf" - featureType
% "svm", "cnn" - classifierType
output = RecogniseFace(I, "hog", "cnn");
disp(output);

function [P] = RecogniseFace(I, featureType, classifierType)
    % Initialise Cascade Object Detector
    FaceDetector = vision.CascadeObjectDetector;
    %FaceDetector.MinSize = [70 70];
    FaceDetector.MergeThreshold = 7;

    sbbox = step(FaceDetector, I);
    % Amount of detected Faces
    N = size(sbbox, 1);
    
    % If not faces detected return empty Matrix
    if N <= 0
        P = [];
        return;
    end

    % Empty arrays
    imgArray = cell(N);
    O = zeros(N, 3);
    
    % Extract coordinates and faces for processing
    for i = 1:N
        % Coordinates
        a = sbbox(i, 1);
        b = sbbox(i, 2);
        c = a+sbbox(i, 3);
        d = b+sbbox(i, 4);

        % Extract faces
        croppedFace = I(b:d, a:c, :);
        % Resize
        croppedFace = imresize(croppedFace, [300 300]);
        % Add extracted face to Array
        imgArray{i} = croppedFace;

%         disp(i);
%         disp(a);
%         disp(b);
%         disp(c);
%         disp(d);

        % Calculate centre coordinates for P output
        x = ceil((a + c) / 2);
        y = ceil((b + d) / 2);

        % Assign to temporary O matrix
        O(i, 2) = x;
        O(i, 3) = y;
    end
    
    % Amount of faces sucessfully extracted
    l = length(imgArray);

    % Perform CNN
    if classifierType == "cnn"
        % Load pretrained Model for CNN
        load('faceNetTrain.mat');

%         YPred = classify(net, imdsValidation);
%         YValidation = imdsValidation.Labels;
% 
%         accuracy = sum(YPred == YValidation);
        
        % For all the extracted faces
        % Predict the label
        % Add to temporary output Matrix
        for i = 1 : l
            YPred = classify(net, imgArray{i});
            O(i, 1) = YPred;
        end

%         net = facenet;
% 
%         inputSize = net.Layers(1).InputSize;
% 
%         classNames = net.Layers(end).ClassNames;
%         numClasses = numel(classNames);
%         disp(classNames(randperm(numClasses, 10)));
    
    % Perform SVM
    elseif classifierType == "svm"
        % Perform SURF
        if featureType == "surf"
            % Load trained SURF SVM data
            load('surfSVMTrain.mat');
        % Perform HOG
        elseif featureType == "hog"
            % Load trained HOG SVM data
            load('hogSVMTrain.mat');
        end

        % Create Confusion Matrix
        % Train sets
        %confMatrixTrain = evaluate(categoryClassifier, trainingSets);

        % Validation sets
        %confMatrixValid = evaluate(categoryClassifier, validationSets);
        %meanValid = mean(diag(confMatrixValid));
        
        % For all the extracted faces
        % Predict the label
        % Add to temporary output Matrix
        for i = 1 : l
            [labelIdx, ~] = predict(categoryClassifier, imgArray{i});
            O(i, 1) = labelIdx;
            % figure, imshow(imgArray{i});
        end
    else
        % If incorrect featureTypes or classificationTypes passed
        disp('No Feature/Classifier Type by that name');
    end

%     for j = 1 : length(O)
%         disp(O(j, 1));
%         disp(O(j, 2));
%         disp(O(j, 3));
%     end

    % Empty Array for displaying
    posArray = zeros(N, 4);
    imgLabels = cell(N, 1);

    % loop through extracted faces
    % Set array for positions
    % set labels
    for i = 1:N
        a = sbbox(i, 1);
        b = sbbox(i, 2);
        c = sbbox(i, 3);
        d = sbbox(i, 4);

        posArray(i, 1) = a;
        posArray(i, 2) = b;
        posArray(i, 3) = c;
        posArray(i, 4) = d;

        imgLabels{i} = num2str(O(i, 1));
    end

    % Draw boxes and labels to original image
    detectedImg = insertObjectAnnotation(I,'rectangle', posArray, imgLabels);
    figure, imshow(detectedImg);

    % If successful output P using temporary O Matrix
    P = O;
end


