function [X_recov,S,time] = ortc(M_0,Omega,rank,iter,lambda_1,lambda_2,transform)
dim_ = size(M_0);
Y = zeros([[rank,rank],dim_(3:end)]);
X = zeros([[dim_(1),rank],dim_(3:end)]);
R = zeros([rank,dim_(2:end)]);
r = zeros([[rank,1],dim_(3:end)]);
O_ = zeros(dim_);
S = zeros(dim_);
L = randn([[dim_(1),rank],dim_(3:end)]);
L_old = zeros([[dim_(1),rank],dim_(3:end)]);
tol = 1e-7;

for i = 1: iter
    for n = 1: dim_(2)
        tic
        m = M_0(:,n,:,:);
        omega_ = Omega(:,n,:,:);
        [s,r,O] = R_update(m,r,L,omega_,rank,lambda_1,lambda_2,transform);
        O_(:,n,:,:) = O;
        S(:,n,:,:) = s;
        R(:,n,:,:) = r;
        %更新左张量L
        Y = Y + tprod(r ,tran(r,transform),transform);
        X = X + tprod(m-s-O,tran(r,transform),transform);
        L = L_update(L,X,Y,lambda_1,transform);
        L_old = L;
        time = toc;
    end
    X_recov = tprod(L,R,transform);
end

function [s_,r_,O] = R_update(m_,r_,L_,Omega_,rank_,lambda_1,lambda_2,transform)
    d_ = size(m_);
    L_1 = tprod(tran(L_,transform),L_,transform)+lambda_1*teye(rank_,d_(3),transform);%identity([[rank_,rank_],d_(3:end)]);
    L_1_inv = tinv(L_1,transform);
    L_2 = tran(L_,transform);
    L_tilde = tprod(L_1_inv,L_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    converged = false;
    O = zeros(d_);
    iter_ = 0;
    while ~converged
        iter_ = iter_ + 1 ;
        s_mult = m_ - tprod(L_,r_,transform) - O;
        s_ = soft_threshold(s_mult,lambda_2);
        r_ = tprod(L_tilde,m_-s_-O,transform);
        O = m_ - s_ - tprod(L_,r_,transform);
        O = O .* (1-Omega_);
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

function [L_] = L_update(L_,X_,Y_,lambda_1,transform)
    d_ = size(X_);
    d_y = size(Y_);
    Y_ = Y_ + lambda_1 * teye(d_y(1),d_y(3),transform);
    X_ = lineartransform(X_,transform);
    Y_ = lineartransform(Y_,transform);
    L_ = lineartransform(L_,transform);
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            L_(:,j,k) = L_(:,j,k) + (X_(:,j,k) - L_(:,:,k) * Y_(:,j,k))/Y_(j,j,k);
        end
    end
    L_ = inverselineartransform(L_,transform);
end

end