function [X_recov,S,O_] = ortcnf(M_0,X,Omega,rank,iter,lambda_2,lambda_3,lambda_4,lambda_5)
% Noisy 边信息
% lambda_3: U_M，V_M的正则项
% lambda_4: U_N，V_N的正则项
% lambda_5: D_M的正则项
dim_ = size(M_0);
d = size(X,2);
A_M = zeros([[rank,rank],dim_(3:end)]);
B_M = zeros([[dim_(1),rank],dim_(3:end)]);
A_N = zeros([[rank,rank],dim_(3:end)]);
B_N = zeros([[dim_(1),rank],dim_(3:end)]);
D_M = randn([[dim_(1),rank],dim_(3:end)]);
D_M_old = randn([[dim_(1),rank],dim_(3:end)]);
L_M = randn([[d,rank],dim_(3:end)]);
L_N = randn([[dim_(1),rank],dim_(3:end)]);
L_N_old = randn([[dim_(1),rank],dim_(3:end)]);
R_M = zeros([[rank,dim_(2)],dim_(3:end)]);
R_N = zeros([[rank,dim_(2)],dim_(3:end)]);
r_M = zeros([[rank,1],dim_(3:end)]);
r_N = zeros([[rank,1],dim_(3:end)]);
S = zeros(dim_);
O_ = zeros(dim_);
%lambda_2 = sqrt(min(dim_(1),dim_(2))/prod(dim_));
transformH.L = 'fft';transformH.rho = prod(dim_(3:end));
flag = 0;
tol = 1e-7;
for i = 1: iter
    %fprintf("iter:%d\n",i);
    for n = 1: dim_(2)
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r_M,r_N,O] = R_update(m,omega_,D_M,L_N,r_M,r_N,rank,lambda_2,lambda_3,lambda_4,transformH);
        O_(:,n,:,:) = O;
        S(:,n,:,:) = s;
        R_M(:,n,:,:) = r_M;
        R_N(:,n,:,:) = r_N;
        ln = H_tprod(L_N,r_N,transformH);
        %更新左张量L2
        A_M = A_M + H_tprod(r_M ,H_tran(r_M,transformH),transformH);
        B_M = B_M + H_tprod(m-s-O-ln,H_tran(r_M,transformH),transformH);
        L_M = LM_update(D_M,X,lambda_3,lambda_5,transformH);
        D_M = D_update(D_M,A_M,B_M,X,L_M,lambda_5,transformH);
        lm = H_tprod(D_M,r_M,transformH);
        A_N = A_N + H_tprod(r_N ,H_tran(r_N,transformH),transformH);
        B_N = B_N + H_tprod(m-s-O-lm,H_tran(r_M,transformH),transformH);
        L_N = LN_update(L_N,A_N,B_N,lambda_4,transformH);
        diff = max([norm(L_N_old(:)-L_N(:)),norm(D_M_old(:)-D_M(:))]);
        %fprintf("diff_L:%f\n",diff);
        L_N_old = L_N;
        D_M_old = D_M;
    end
end
X_recov = H_tprod(D_M,R_M,transformH)+H_tprod(L_N,R_N,transformH);
%X_recov = H_tprod(H_tprod(X,L_M,transformH),R_M,transformH)+H_tprod(L_N,R_N,transformH);

function [s_,r_M,r_N,O] = R_update(m_,Omega_,D_M,L_N,r_M,r_N,rank_,lambda_2,lambda_3,lambda_4,transform)
    d_ = size(m_);
    DM_1 = H_tprod(H_tran(D_M,transform),D_M,transform)+lambda_3*H_teye([[rank_,rank_],d_(3:end)],transform);
    DM_1_inv = H_tinv(DM_1,transform);
    DM_2 = H_tran(D_M,transform);
    DM_tilde = H_tprod(DM_1_inv,DM_2,transform);

    LN_1 = H_tprod(H_tran(L_N,transform),L_N,transform)+lambda_4*H_teye([[rank_,rank_],d_(3:end)],transform);
    LN_1_inv = H_tinv(LN_1,transform);
    LN_2 = H_tran(L_N,transform);
    LN_tilde = H_tprod(LN_1_inv,LN_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_M_new = r_M;
    r_N_new = r_N;
    O = zeros(d_);
    converged = false;
    iter_ = 0;
    while ~converged
        iter_ = iter_ +1;
        s_ = soft_threshold(m_- O - H_tprod(D_M,r_M,transform)- H_tprod(L_N,r_N,transform),lambda_2);
        r_M = H_tprod(DM_tilde,m_-s_-O- H_tprod(L_N,r_N,transform),transform);
        r_N = H_tprod(LN_tilde,m_-s_-O- H_tprod(D_M,r_M,transform),transform);
        O = m_ - s_ - H_tprod(L_N,r_N,transform) - H_tprod(D_M,r_M,transform);
        O = O .* (1-Omega_);
        error = max([norm(s_(:)-s_new(:),"fro"),norm(r_M(:)-r_M_new(:),"fro"),norm(r_N(:)-r_N_new(:),"fro")]);
        %fprintf("error:%d\n",error);
        if error < tol
            converged = true;
        end
        if ~converged && iter_ >= 1000
            converged = true ;    
        end
        s_new = s_;
        r_M_new = r_M;
        r_N_new = r_N;
    end 
end

function [L_M] = LM_update(D_M,X,lambda_3,lambda_5,transform)
    X_1 = H_tprod(H_tran(X,transform),X,transform);
    d_ = size(X_1);
    X_2 = H_tinv((lambda_3/lambda_5)*H_teye([d_],transform)+X_1,transform);
    X_3 = H_tprod(H_tran(X,transform),D_M,transform);
    L_M = H_tprod(X_2,X_3,transform);         
end

function [D_M] = D_update(D_M,A_M,B_M,X,L_M,lambda_5,transform)
    d_ = size(B_M);
    M_ = H_tprod(X,L_M,transform);
    A_M = A_M + lambda_5 * H_teye(size(A_M),transform);
    for n_ = 3:length(d_)
        A_M = fft(A_M,[],n_);
        B_M = fft(B_M,[],n_);
        M_ = fft(M_,[],n_);
        D_M = fft(D_M,[],n_);
    end
    B_M = B_M + lambda_5 * M_;
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            D_M(:,j,k) = D_M(:,j,k) + (B_M(:,j,k) - D_M(:,:,k) * A_M(:,j,k))/A_M(j,j,k);
        end
    end
    for n_ = 3:length(d_)
        D_M = ifft(D_M,[],n_);
    end
end


function [L_N] = LN_update(L_N,A_N,B_N,lambda_4,transform)
    d_ = size(B_N);
    A_N = A_N + lambda_4 * H_teye(size(A_N),transform);
    for n_ = 3:length(d_)
        A_N = fft(A_N,[],n_);
        B_N = fft(B_N,[],n_);
        L_N = fft(L_N,[],n_);
    end
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            L_N(:,j,k) = L_N(:,j,k) + (B_N(:,j,k) - L_N(:,:,k) * A_N(:,j,k))/A_N(j,j,k);
        end
    end
    for n_ = 3:length(d_)
        L_N = ifft(L_N,[],n_);
    end
end


end