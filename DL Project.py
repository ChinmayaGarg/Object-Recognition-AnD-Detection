%CODE TO RESIZE IMAGE TO 128X128X3(which will be our input)


srcFiles = dir('C:\Users\chinm\Pictures\2018-05\shoes\*.jpg');
for i = 1 : length(srcFiles)
    thisFileName = fullfile('C:\Users\chinm\Pictures\2018-05\shoes\',srcFiles(i).name);
    img=imread(thisFileName);
    img=imresize(img,[128 128]);
    imwrite(img,strcat(num2str(i+525),'.jpg'))
end


%CODE TO LABEL DATASET

imds = imageDatastore('C:\Users\chinm\Pictures\DATASET',...
'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); 




%CLASSIFIER CODE (NET USED IS MADE BY CHINMAYA GARG)



numImageCategories = 3;
imageSize = [128 128 3];
inputLayer = imageInputLayer(imageSize)
filterSize = [5 5];
numFilters = 32;

middleLayers = [
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride', 2)
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)
convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer() 
maxPooling2dLayer(3, 'Stride',2)

]


finalLayers = [
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(numImageCategories)
softmaxLayer
classificationLayer
]
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]


opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true); 
doTraining = false;

if doTraining   
    classnet = trainNetwork(imds, layers, opts);
   
else
    
    load('rcnnStopSigns.mat','cifar10Net')       
end
X=imread('C:\Users\chinm\Desktop\testmobile.jpg');
X=imresize(X,[128 128]);
[YPred,scores] = classify(classnet,X)

subplot(1,2,1);
imshow(X);
prob = num2str(100*max(scores),3);

predClass = char(YPred);
title([predClass,', ',prob,'%'])





%this code is used for detectin bottle in a scene:


load('C:\Users\chinm\Desktop\bottle try\bottle\bottlelabels.mat','gTruth')
gTruth.LabelDefinitions
gTruth = selectLabels(gTruth,'bottle');
trainingDatabottle = objectDetectorTrainingData(gTruth);
summary(trainingDatabottle)
trainmymodel=false;
if trainmymodel
 options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 10, ...
        'MaxEpochs', 20, ...
        'Verbose', true);
detector = trainRCNNObjectDetector(trainingDatabottle,cifar10Net,options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
end
testImage = imread('C:\Users\chinm\Desktop\img8.jpg');
testImage=imresize(testImage,[128 128])    ;
[bboxes,score,label] = detect(detector,testImage);

[score, idx] = max(score);

% bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score)

outputImage = insertObjectAnnotation(testImage, 'rectangle', bboxes, annotation);
 figure
imshow(outputImage)





