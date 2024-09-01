function features = extractFeatures(imagePath, net)
    % extractFeatures Extracts deep features from an image using a specified neural network.
    % 
    % Parameters:
    %   imagePath (string): The path to the image file.
    %   net (DAGNetwork or SeriesNetwork): The pre-trained neural network (e.g., AlexNet).
    %
    % Returns:
    %   features (vector): The extracted features from the specified layer of the network.
    
    % Read the image from the specified file path
    image = imread(imagePath);
    
    % Resize the image to 227x227 pixels to match the input size required by AlexNet
    image = imresize(image, [227, 227]);
    
    % Extract features from the 'fc7' layer of the network
    features = activations(net, image, 'fc7');
    
    % Remove singleton dimensions to convert the feature map into a vector
    features = squeeze(features);
    
    % Return the extracted features
end
