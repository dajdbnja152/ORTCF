function a = mat2ten(A,n1,n3)
k = 1;
    for m =1:n3
        if k <= n3
            a(:,:,m) = A(n1*(k-1)+1:n1*k,:);
        end
        k = k+1;
    end
end
