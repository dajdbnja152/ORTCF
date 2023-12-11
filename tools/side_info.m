function [A] = side_info(A,transform,d,p)

[n1,n2,n3] = size(A);
if nargin < 2
    % fft is the default transform
    transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
end

if isequal(transform.L,@fft)
    % efficient computing for fft transform
    A = A(:,1:d,:);
    At = tran(A);
    At = fft(At,[],3); 
    X = zeros(n1,n1-d,n3);
        for i = 1 : n3
            X(:,:,i) = null(At(:,:,i));
        end
    X = ifft(X,[],3);
    replace_d = ceil(d*p);
    omega = randperm(d,replace_d);
    %A(:,omega,:) = X(:,omega,:);
    A(:,1:replace_d,:) = X(:,1:replace_d,:);

else
    % other transform
    A = A(:,1:d,:);
    At = tran(A,transform);
    At = lineartransform(At,transform);
    X = zeros(n1,n1-d,n3);
        for i = 1 : n3
            X(:,:,i) = null(At(:,:,i));
        end
    X = inverselineartransform(X,transform);
    replace_d = ceil(d*p);
    omega = randperm(d,replace_d);
    %A(:,omega,:) = X(:,omega,:);
    A(:,1:replace_d,:) = X(:,1:replace_d,:);
end