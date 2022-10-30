% Root folder
rootFolder = fullfile('cropface');

% Load DataStore
imds = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource','foldernames');
labelCount = countEachLabel(imds);

% Number of Train files for each Category
numTrainFiles = 4;

% Split Train and Validation Data
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

% Layers
layers = [
    imageInputLayer([300 300 3])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(48)
    softmaxLayer
    classificationLayer
    ];

% Optional
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress' ...
    );

% Train network
net = trainNetwork(imdsTrain, layers, options);

% Save Training
save faceNetTrain;

% Test Validation data against Training
[YPred, scores] = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation);

% [~, idx] = sort(scores, 'descend');
% idx = idx(5:-1:1);
% classNamesTop = net.Layers(end).ClassNames(idx);
% scoresTop = scores(idx);
% 
% figure
% barh(scoresTop)
% xlim([0 1])
% title('Top 5 Predictions')
% xlabel('Probability')
% yticklabels(classNamesTop);


