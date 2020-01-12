function [W, b] = updateWeight(input, label, W, b, alpha)
% function purpose: update weight W (2x1) and bias b (scalar)
% input: n x 2 matrix, column1 = x values, column2 = y values
% label: 1 = red, 0 = blue
% alpha = learning rate
    for i = 1:length(input)
        prediction = predict(input(i,:), W, b);
        classification = label(i) - prediction;
        if classification == 1 % 1-0 incorrectly classified
            W(1) = W(1) + input(i,1)*alpha; % W(1) is for x
            W(2) = W(2) + input(i,2)*alpha; % W(2) is for y
            b = b + alpha;
        elseif classification == -1 % 0-1 incorrectly classified
            W(1) = W(1) - input(i,1)*alpha;
            W(2) = W(2) - input(i,2)*alpha;
            b = b - alpha;
        elseif classification == 0 % 0-0 or 1-1 correctly classified
            %do nothing
        end
    end
end

