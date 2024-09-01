function svmModel = trainSVM(features, labels)
    % trainSVM Trains an SVM model using provided features and labels.
    %
    % Parameters:
    %   features (matrix): The feature matrix where each row corresponds to an instance and each column to a feature.
    %   labels (array): The labels corresponding to each instance in the feature matrix.
    %
    % Returns:
    %   svmModel (ClassificationSVM): The trained SVM model.
    
    % Convert labels to a categorical array to ensure proper handling by the SVM model
    labels = categorical(labels);
    
    % Identify and display unique labels in the dataset
    uniqueLabels = unique(labels);
    disp('Unique labels:');
    disp(uniqueLabels);
    
    % Train the SVM model using a linear kernel
    % 'Standardize' is set to true to normalize the features before training
    % 'ClassNames' specifies the unique labels in the dataset
    svmModel = fitcsvm(features, labels, ...
                       'KernelFunction', 'linear', ...
                       'Standardize', true, ...
                       'ClassNames', uniqueLabels);
    
    % Return the trained SVM model
end
