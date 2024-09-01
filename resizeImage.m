function resizedImage = resizeImage(imagePath, targetSize)
    % resizeImage Resizes an image to a specified target size.
    % 
    % Parameters:
    %   imagePath (string): The path to the image file.
    %   targetSize (vector): The desired size for the output image. It can be 
    %                        specified as a two-element vector [rows, columns] 
    %                        or as a scaling factor.
    %
    % Returns:
    %   resizedImage (matrix): The resized image.
    
    % Read the image from the specified file path
    image = imread(imagePath);
    
    % Resize the image to the target size
    resizedImage = imresize(image, targetSize);
    
    % Return the resized image
end
