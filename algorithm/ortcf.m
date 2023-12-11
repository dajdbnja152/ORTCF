function [X_recov,S,O,time,diff] = ortcf(M_0,X,Omega,rank,iter,lambda_1,lambda_2,lambda_3,transform)
dim_ = size(M_0);
d = size(X,2);
A = zeros([[rank,rank],dim_(3:end)]);
B = zeros([[dim_(1),rank],dim_(3:end)]);
D = randn([[dim_(1),rank],dim_(3:end)]);
D_old = randn([[dim_(1),rank],dim_(3:end)]);
L = randn([[d,rank],dim_(3:end)]);
R = zeros([[rank,dim_(2)],dim_(3:end)]);
r = zeros([[rank,1],dim_(3:end)]);
S = zeros(dim_);
O = zeros(dim_);
tol = 1e-7;

for i = 1: iter
    for n = 1:dim_(2)
        tic
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r,o] = R_update(m,omega_,D,r,rank,lambda_1,lambda_2,transform);
        O(:,n,:,:) = o;
        S(:,n,:,:) = s;
        R(:,n,:,:) = r;
        %更新左张量L2
        A = A + tprod(r,tran(r,transform),transform);
        B = B + tprod(m-s-o,tran(r,transform),transform);
        L = L_update(D,X,lambda_1,lambda_3,transform);
        D = D_update(D,A,B,X,L,lambda_3,transform);
        diff(n) = norm(D_old(:)-D(:));
        D_old = D;
        time = toc;
    end
end
X_recov = tprod(tprod(X,L,transform),R,transform);


function [s_,r_,o_] = R_update(m_,Omega_,D_,r_,rank_,lambda_1,lambda_2,transform)
    d_ = size(m_);
    D_1 = tprod(tran(D_,transform),D_,transform)+lambda_1*teye(rank_,d_(3),transform);
    D_1_inv = tinv(D_1,transform);
    D_2 = tran(D_,transform);
    D_tilde = tprod(D_1_inv,D_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    o_ = zeros(d_);
    converged = false;
    iter_ = 0;
    while ~converged
        r_ = tprod(D_tilde,m_-s_-o_,transform);
        s_ = soft_threshold(m_ - tprod(D_,r_,transform)-o_,lambda_2);
        o_ = m_ - s_ -  tprod(D_,r_,transform);
        o_ = o_ .* (1-Omega_);
        error = max([norm(s_(:)-s_new(:),"fro"),norm(r_(:)-r_new(:),"fro")]);
        if error < tol
            converged = true;
        end
        if ~converged && iter_ >= 1000
            converged = true ;    
        end
        s_new = s_;
        r_new = r_;
    end 
end

function [L_] = L_update(D_,X,lambda_1,lambda_3,transform)
    X_1 = tprod(tran(X,transform),X,transform);
    d_ = size(X_1);
    X_2 = tinv((lambda_1/lambda_3)*teye(d_(2),d_(3),transform)+X_1,transform);
    X_3 = tprod(tran(X,transform),D_,transform);
    L_ = tprod(X_2,X_3,transform);         
end

function [D_] = D_update(D_,A_,B_,X,L_,lambda_5,transform)
    d_ = size(B_);
    M_ = tprod(X,L_,transform);
    d_a = size(A_);
    A_ = A_ + lambda_5 * teye(d_a(1),d_a(3),transform);
    A_ = lineartransform(A_,transform);
    B_ = lineartransform(B_,transform);
    M_ = lineartransform(M_,transform);
    D_ = lineartransform(D_,transform);
    B_ = B_ + lambda_5 * M_;
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            D_(:,j,k) = D_(:,j,k) + (B_(:,j,k) - D_(:,:,k) * A_(:,j,k))/A_(j,j,k);
        end
    end
    D_ = inverselineartransform(D_,transform);
end

end

