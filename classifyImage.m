function predictedLabel = classifyImage(imagePath, net, svmModel)
    % classifyImage Classifies an image using a pre-trained neural network and an SVM model.
    %
    % Parameters:
    %   imagePath (string): The path to the image file.
    %   net (DAGNetwork or SeriesNetwork): The pre-trained neural network (e.g., AlexNet).
    %   svmModel (ClassificationSVM): The trained SVM model used for classification.
    %
    % Returns:
    %   predictedLabel (string or numeric): The predicted label for the image.
    
    % Extract deep features from the image using the specified neural network
    features = extractFeatures(imagePath, net);
    
    % Predict the label of the image using the SVM model
    % Transpose the feature vector to match the input format required by the SVM model
    predictedLabel = predict(svmModel, features');
    
    % Return the predicted label
end
