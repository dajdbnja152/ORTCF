function [X_recov,S,O_] = ormcnf(M_0,X,Omega,rank,iter,lambda_2,lambda_3,lambda_4,lambda_5)
% Noisy 边信息
% lambda_3: U_M，V_M的正则项
% lambda_4: U_N，V_N的正则项
% lambda_5: D_M的正则项
dim_ = size(M_0);
d = size(X,2);
A_M = zeros([rank,rank]);
B_M = zeros([dim_(1),rank]);
A_N = zeros([rank,rank]);
B_N = zeros([dim_(1),rank]);
D_M = randn([dim_(1),rank]);
D_M_old = randn([dim_(1),rank]);
L_M = randn([d,rank]);
L_N = randn([dim_(1),rank]);
L_N_old = randn([dim_(1),rank]);
R_M = zeros([rank,dim_(2)]);
R_N = zeros([rank,dim_(2)]);
r_M = zeros([rank,1]);
r_N = zeros([rank,1]);
S = zeros(dim_);
O_ = zeros(dim_);
flag = 0;
tol = 1e-7;
for i = 1: iter
    %fprintf("iter:%d\n",i);
    for n = 1: dim_(2)
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r_M,r_N,O] = R_update(m,omega_,D_M,L_N,r_M,r_N,rank,lambda_2,lambda_3,lambda_4);
        O_(:,n) = O;
        S(:,n) = s;
        R_M(:,n) = r_M;
        R_N(:,n) = r_N;
        ln = L_N * r_N;
        %更新左张量L2
        A_M = A_M + r_M * r_M';
        B_M = B_M + (m-s-O-ln) * r_M';
        L_M = LM_update(D_M,X,lambda_3,lambda_5);
        D_M = D_update(D_M,A_M,B_M,X,L_M,lambda_5);
        lm = D_M * r_M;
        A_N = A_N + r_N * r_N';
        B_N = B_N + (m-s-O-lm) * r_M';
        L_N = LN_update(L_N,A_N,B_N,lambda_4);
        diff = max([norm(L_N_old(:)-L_N(:)),norm(D_M_old(:)-D_M(:))]);
        %fprintf("diff_L:%f\n",diff);
        L_N_old = L_N;
        D_M_old = D_M;
    end
end
X_recov = D_M * R_M + L_N * R_N;
%X_recov = X * L_M * R_M + L_N * R_N;

    function [s_,r_M,r_N,O] = R_update(m_,Omega_,D_M,L_N,r_M,r_N,rank_,lambda_2,lambda_3,lambda_4)
    d_ = size(m_);
    DM_1 = D_M' * D_M + lambda_3* eye([rank_,rank_]);
    DM_1_inv = tinv(DM_1);
    DM_2 = tran(D_M);
    DM_tilde = DM_1_inv * DM_2;

    LN_1 = tran(L_N) * L_N + lambda_4 * eye([rank_,rank_]);
    LN_1_inv = tinv(LN_1);
    LN_2 = tran(L_N);
    LN_tilde = LN_1_inv * LN_2;
    s_ = zeros(d_);
    s_new = s_;
    r_M_new = r_M;
    r_N_new = r_N;
    O = zeros(d_);
    converged = false;
    iter_ = 0;
    while ~converged
        iter_ = iter_ +1;
        s_ = soft_threshold(m_- O - D_M *r_M - L_N*r_N,lambda_2);
        r_M = DM_tilde * (m_- s_ - O - L_N * r_N);
        r_N = LN_tilde * (m_- s_ -O - D_M * r_M);
        O = m_ - s_ - L_N * r_N - D_M * r_M;
        O = O .* (1-Omega_);
        error = max([norm(s_-s_new,"fro"),norm(r_M-r_M_new,"fro"),norm(r_N-r_N_new,"fro")]);
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

    function [L_M] = LM_update(D_M,X,lambda_3,lambda_5)
    X_1 = tran(X) * X;
    d_ = size(X_1);
    X_2 = inv((lambda_3/lambda_5)* eye(d_)+X_1);
    X_3 = X' * D_M;
    L_M = X_2 * X_3;         
end

    function [D_M] = D_update(D_M,A_M,B_M,X,L_M,lambda_5)
    d_ = size(B_M);
    M_ = X * L_M;
    A_M = A_M + lambda_5 * eye(size(A_M));
    B_M = B_M + lambda_5 * M_;
    for j = 1:d_(2)
            D_M(:,j) = D_M(:,j) + (B_M(:,j) - D_M(:,:) * A_M(:,j))/A_M(j,j);
    end
end


    function [L_N] = LN_update(L_N,A_N,B_N,lambda_4)
    d_ = size(B_N);
    A_N = A_N + lambda_4 * eye(size(A_N));
    for j = 1:d_(2)
            L_N(:,j) = L_N(:,j) + (B_N(:,j) - L_N(:,:) * A_N(:,j))/A_N(j,j);
    end
end


end