function [X_recov,S,O] = orhtcf(M_0,X,Omega,rank,iter,lambda_1,lambda_2,lambda_3)
% 完美边信息
% lambda_1: L,R的正则项
% lambda_2: S的正则项
% lambda_3: D的正则项
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
%lambda_2 = sqrt(min(dim_(1),dim_(2))/prod(dim_));
transformH.L = 'fft';transformH.rho = prod(dim_(3:end));
tol = 1e-7;
for i = 1: iter
    %fprintf("iter:%d\n",i);
    for n = 1: dim_(2)
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r,o] = R_update(m,omega_,D,r,rank,lambda_1,lambda_2,transformH);
        O(:,n,:,:) = o;
        S(:,n,:,:) = s;
        R(:,n,:,:) = r;
        %更新左张量L2
        A = A + H_tprod(r ,H_tran(r,transformH),transformH);
        B = B + H_tprod(m-s-o,H_tran(r,transformH),transformH);
        L = L_update(D,X,lambda_1,lambda_3,transformH);
        D = D_update(D,A,B,X,L,lambda_3,transformH);
        diff = norm(D_old(:)-D(:));
        %fprintf("diff_L:%f\n",diff);
        D_old = D;
    end
end
%X_recov = H_tprod(D_M,R_M,transformH);
X_recov = H_tprod(H_tprod(X,L,transformH),R,transformH);

function [s_,r_,o_] = R_update(m_,Omega_,D_,r_,rank_,lambda_1,lambda_2,transform)
    d_ = size(m_);
    D_1 = H_tprod(H_tran(D_,transform),D_,transform)+lambda_1*H_teye([[rank_,rank_],d_(3:end)],transform);
    D_1_inv = H_tinv(D_1,transform);
    D_2 = H_tran(D_,transform);
    D_tilde = H_tprod(D_1_inv,D_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    o_ = zeros(d_);
    converged = false;
    iter_ = 0;
    while ~converged
        r_ = H_tprod(D_tilde,m_-s_-o_,transform);
        s_ = soft_threshold(m_ - H_tprod(D_,r_,transform)-o_,lambda_2);
        o_ = m_ - s_ -  H_tprod(D_,r_,transform);
        o_ = o_ .* (1-Omega_);
        error = max([norm(s_(:)-s_new(:),"fro"),norm(r_(:)-r_new(:),"fro")]);
        %fprintf("error:%d\n",error);
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
    X_1 = H_tprod(H_tran(X,transform),X,transform);
    d_ = size(X_1);
    X_2 = H_tinv((lambda_1/lambda_3)*H_teye(d_,transform)+X_1,transform);
    X_3 = H_tprod(H_tran(X,transform),D_,transform);
    L_ = H_tprod(X_2,X_3,transform);         
end

function [D_] = D_update(D_,A_,B_,X,L_,lambda_5,transform)
    d_ = size(B_);
    M_ = H_tprod(X,L_,transform);
    A_ = A_ + lambda_5 * H_teye(size(A_),transform);
    for n_ = 3:length(d_)
        A_ = fft(A_,[],n_);
        B_ = fft(B_,[],n_);
        M_ = fft(M_,[],n_);
        D_ = fft(D_,[],n_);
    end
    B_ = B_ + lambda_5 * M_;
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            D_(:,j,k) = D_(:,j,k) + (B_(:,j,k) - D_(:,:,k) * A_(:,j,k))/A_(j,j,k);
        end
    end
    for n_ = 3:length(d_)
        D_ = ifft(D_,[],n_);
    end
end



end