% File: trainSkinDiseaseModel.m
clc
clear
close all

% Load pre-trained AlexNet
net = alexnet;

% Define dataset paths
dataDir = '/MATLAB Drive/SkinDiseasesDetection/data';  % Adjust to your specified path
diseaseFolders = dir(fullfile(dataDir, '*'));  % List all folders in the data directory

% Initialize variables
imagePaths = {};  % Cell array to store paths of all images
labels = {};      % Cell array to store corresponding labels for each image

% Load training data
for i = 1:length(diseaseFolders)
    if diseaseFolders(i).isdir && ~startsWith(diseaseFolders(i).name, '.')
        % Skip non-directories and hidden folders (starting with '.')
        diseaseName = diseaseFolders(i).name;
        diseasePath = fullfile(dataDir, diseaseName);  % Full path to the disease folder
        imageFiles = dir(fullfile(diseasePath, '*.jpg')); % Assuming images are in .jpg format
        
        for j = 1:length(imageFiles)
            imagePaths{end+1} = fullfile(diseasePath, imageFiles(j).name);  % Store image path
            labels{end+1} = diseaseName;  % Store corresponding label (disease name)
        end
    end
end

% Convert labels to categorical array
labels = categorical(labels);

% Check if imagePaths and labels have the same length
if length(imagePaths) ~= length(labels)
    error('Mismatch between number of images and number of labels.');
end

% Extract features for all training images
numImages = length(imagePaths);
allFeatures = zeros(numImages, 4096);  % Pre-allocate a matrix for features (4096 is the number of features from 'fc7' layer of AlexNet)

for i = 1:numImages
    allFeatures(i, :) = extractFeatures(imagePaths{i}, net);  % Extract and store features for each image
end

% Train SVM model using fitcecoc for multi-class classification
svmModel = trainMultiClassSVM(allFeatures, labels);

% Save the trained model and features
save('trainedModel.mat', 'svmModel', 'net');

% Nested function to train a multi-class SVM model using one-vs-all coding
function svmModel = trainMultiClassSVM(features, labels)
    uniqueLabels = unique(labels);  % Find unique labels
    svmModel = fitcecoc(features, labels, 'Coding', 'onevsall', 'ClassNames', uniqueLabels);  % Train SVM using one-vs-all approach
end

% Nested function to extract features from an image using AlexNet
function features = extractFeatures(imagePath, net)
    image = imread(imagePath);  % Read the image from the specified path
    image = imresize(image, [227, 227]);  % Resize image for AlexNet input size
    features = activations(net, image, 'fc7');  % Extract features from the 'fc7' layer of AlexNet
    features = squeeze(features);  % Remove singleton dimensions to get a feature vector
end
