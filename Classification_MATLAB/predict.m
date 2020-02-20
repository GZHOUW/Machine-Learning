function prediction = predict(input_i,W,b)
    % input_i: one row of x y data (1x2)
    % weight W (2x1) and bias b (scalar)
    score = input_i*W + b; % score = W1*x + W2*y + b 
    if score >= 0 % above classification line
        prediction = 1; % predict to be red
    else % below classification line
        prediction = 0; % predict to be blue
    end
end

