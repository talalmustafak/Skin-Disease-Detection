% File: predictSkinDisease.m

% Load the trained SVM model and pre-trained AlexNet network
load('trainedModel.mat', 'svmModel', 'net');

% Define the path to the test dataset
testDir = '/MATLAB Drive/SkinDiseasesDetection/testingdata/test9';  % Adjust to your specified path
testImageFiles = dir(fullfile(testDir, '*.jpg')); % Get all .jpg files in the test directory

% Predict labels for all test images
for i = 1:length(testImageFiles)
    % Full path of the current test image
    testImagePath = fullfile(testDir, testImageFiles(i).name);
    
    % Classify the image and predict the label
    predictedLabel = classifyImage(testImagePath, net, svmModel);
    
    % Display the predicted label
    fprintf('%s\n', predictedLabel);
end

% Function to extract deep features from an image using AlexNet
function features = extractFeatures(imagePath, net)
    image = imread(imagePath);  % Read the image from the specified path
    image = imresize(image, [227, 227]);  % Resize the image to 227x227 pixels for AlexNet
    features = activations(net, image, 'fc7');  % Extract features from the 'fc7' layer of AlexNet
    features = squeeze(features);  % Remove singleton dimensions to get a feature vector
end

% Function to classify an image using the extracted features and the trained SVM model
function predictedLabel = classifyImage(imagePath, net, svmModel)
    features = extractFeatures(imagePath, net);  % Extract features from the image
    predictedLabel = predict(svmModel, features');  % Predict the label using the SVM model
end
